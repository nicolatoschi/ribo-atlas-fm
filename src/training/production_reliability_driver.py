#!/usr/bin/env python3
"""Fixed-work construct microbatch profiling of final-generated ensemble loss.

Same pretrained decoder, frozen geometry/chemistry systems head and50-call
sampler. Logical-case priors/probes and frozen singleton teacher corruptions
are independent of microbatch layout. Native teacher train-mode dropout has
the same distribution but its RNG layout differs across microbatch sizes.
"""
import argparse,collections,datetime,hashlib,json,os,sys,time
from pathlib import Path
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from ensemble_mean_core import FrozenGeometryChemistryHead,MatchedProbeNodeEncoder,seed_member
from ensemble_training_driver import atomic_json,state_sha,sha,gradient_norm
from full_trajectory_core import full_trajectory,native_adjacent_distance_constraint
from differentiable_rotation import install_native_quaternion_derivative
from batched_ensemble_core import (case_mean_loss_and_derivative,logical_priors,repeat_native,
                                   observations,digest_tensor)
from reliability_state import run_reliability, SITE_LIMIT_SECONDS

CHECKPOINT_SHA='bb598a736193734541f63c59dae75d45db8c04f01a8414a758677367c4b1e213'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',default='1l8v_B');p.add_argument('--microbatch',type=int,default=2)
    p.add_argument('--local-constructs',type=int,default=8);p.add_argument('--ensemble-size',type=int,default=32)
    p.add_argument('--grid-points',type=int,default=50);p.add_argument('--seed',type=int,default=20261012)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--cpu-proof',action='store_true')
    p.add_argument('--prepare-fixture',action='store_true')
    p.add_argument('--stage',choices=['continuous','resume'],required=True)
    p.add_argument('--state-root',type=Path,required=True)
    p.add_argument('--campaign-manifest',type=Path,required=True);args=p.parse_args()
    args.out=args.out.resolve();args.state_root=args.state_root.resolve();args.campaign_manifest=args.campaign_manifest.resolve()
    assert not args.cpu_proof and not args.prepare_fixture, 'Separate reliability orchestration only'
    source=args.source.resolve();out=args.out.resolve();fixture_path=args.fixture.resolve()
    assert args.microbatch>0 and args.local_constructs%args.microbatch==0
    assert args.grid_points>=2 and args.ensemble_size>=2
    os.chdir(source);sys.path.insert(0,str(source))
    from src.models.flow_module import FlowModule
    from src.data.pdb_na_dataset_base import PDBNABaseDataset
    from src.data.interpolant import Interpolant
    from src.data import all_atom,base_constants,so3_utils,rigid_utils
    install_native_quaternion_derivative(rigid_utils)
    rank=int(os.environ.get('RANK',os.environ.get('SLURM_PROCID','0')))
    world=int(os.environ.get('WORLD_SIZE',os.environ.get('SLURM_NTASKS','1')))
    local=int(os.environ.get('LOCAL_RANK',os.environ.get('SLURM_LOCALID','0')))
    if args.device=='cuda':
        index=local if torch.cuda.device_count()>local else 0;torch.cuda.set_device(index);device=torch.device('cuda',index)
    else:device=torch.device('cpu')
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',rank=rank,world_size=world,timeout=datetime.timedelta(seconds=SITE_LIMIT_SECONDS))
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')));torch.set_float32_matmul_precision('high')
    torch.backends.mha.set_fastpath_enabled(False);seed_member(args.seed,device)
    cfg=OmegaConf.load(source/'configs/config.yaml');cfg.data_cfg.csv_path=str(source/'metadata/pilot_ensemble.csv')
    cfg.data_cfg.filtering.max_len=800;cfg.experiment.checkpointer.dirpath=str(out/'upstream_checkpoints')
    assert not cfg.interpolant.self_condition
    data=PDBNABaseDataset(cfg.data_cfg,is_training=True)
    def native(case):
        index=next(i for i in range(len(data)) if str(data.csv.iloc[i]['pdb_name'])==case)
        batch={key:torch.as_tensor(value).unsqueeze(0).to(device) for key,value in data[index].items()}
        return {key:value.float() if value.is_floating_point() else value for key,value in batch.items()}
    if args.prepare_fixture:
        assert device.type=='cpu' and world==1
        interpolant=Interpolant(cfg.interpolant);interpolant.set_device(device)
        result={'seed':args.seed,'role':'Frozen singleton native teacher corruptions; no inverse-model activations','cases':{}}
        for case,count in [('1g59_B',2),('1l8v_B',32)]:
            single=native(case);entries=[]
            for logical in range(count):
                seed_member(args.seed+10000019+logical*1009,device)
                noisy=interpolant.corrupt_batch(single)
                frames={key:noisy[key].detach().clone() for key in ['t','trans_t','rotmats_t']}
                entries.append({'logical_id':logical,'frames':{key:value.tolist() for key,value in frames.items()},
                                'raw_tensor_sha256':{key:digest_tensor(value) for key,value in frames.items()}})
            result['cases'][case]={'length':int(single['res_mask'].shape[1]),'entries':entries}
        atomic_json(fixture_path,result);print(json.dumps({'fixture':str(fixture_path),'sha256':sha(fixture_path)}));return
    assert sha(args.checkpoint.resolve())==CHECKPOINT_SHA
    single=native(args.case);length=int(single['res_mask'].shape[1]);fixture=json.loads(fixture_path.read_text())
    assert fixture['seed']==args.seed and fixture['cases'][args.case]['length']==length
    records=fixture['cases'][args.case]['entries']
    assert len(records)>=world*args.local_constructs
    c4_index=base_constants.atom_order["C4'"];assert c4_index==5
    native_c4=all_atom.to_atom23_rna(single['trans_1'],single['rotmats_1'],single['aatype'],torsions=single['torsion_angles_sin_cos'])[:,:,c4_index]
    logical_ids=list(range(rank*args.local_constructs,(rank+1)*args.local_constructs))
    target,weights,probe,chemistry=observations(native_c4,single['res_mask'],logical_ids,args.seed)
    observation_hashes={str(logical):{name:digest_tensor(value[i:i+1]) for name,value in zip(['target_sha256','weights_sha256','probe_sha256','chemistry_sha256'],[target,weights,probe,chemistry])} for i,logical in enumerate(logical_ids)}
    teacher={}
    for key in ['t','trans_t','rotmats_t']:
        chunks=[]
        for logical in logical_ids:
            value=torch.tensor(records[logical]['frames'][key],device=device,dtype=torch.float32)
            assert records[logical]['logical_id']==logical and digest_tensor(value)==records[logical]['raw_tensor_sha256'][key]
            chunks.append(value)
        teacher[key]=torch.cat(chunks)
    class Workload(nn.Module):
        def __init__(self):
            super().__init__();self.flow=FlowModule(cfg)
            saved=torch.load(args.checkpoint.resolve(),map_location='cpu',weights_only=False)
            self.flow.model.load_state_dict({k.removeprefix('model.'):v for k,v in saved['state_dict'].items() if k.startswith('model.')},strict=True)
            self.flow.model.node_embedder=MatchedProbeNodeEncoder(self.flow.model.node_embedder,int(cfg.model.node_embed_size))
            self.flow.model.node_embedder.active=True
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed+701);self.head=FrozenGeometryChemistryHead()
            self.counters={'phase':'setup','entries':collections.Counter()}
        def forward(self,kind,batch,ids,position,member=0,identity=0):
            count=len(ids);self.flow.model.node_embedder.current_probe=probe[position:position+count]
            if kind=='teacher':
                self.flow.model.train();self.counters['entries']['native_teacher_forward']+=1
                parts=self.flow.model_step(batch)
                if not all(bool(torch.isfinite(v).all()) for v in parts.values()):raise RuntimeError('Nonfinite native teacher loss')
                return parts[cfg.experiment.training.loss].mean()+parts['auxiliary_loss'].mean()
            self.flow.model.eval()
            tx,rot=logical_priors(args.seed,0,identity,ids,member,length,device)
            trans,rotation,torsions=full_trajectory(self.flow.model,batch,tx,rot,args.grid_points,so3_utils.geodesic_t,
                                                   torch.is_grad_enabled(),self.counters)
            atoms=all_atom.to_atom23_rna(trans,rotation,batch['aatype'],torsions=torsions)
            profile=self.head(atoms[:,:,c4_index],chemistry[position:position+count],batch['res_mask'])
            constraint=native_adjacent_distance_constraint(atoms[:,:,c4_index],native_c4.expand(count,-1,-1),batch['res_mask'])
            if not bool(torch.isfinite(profile).all()) or not bool(torch.isfinite(constraint)):raise RuntimeError('Nonfinite generated output')
            return profile,constraint
    model=Workload().to(device);torch.autograd.set_detect_anomaly(False);assert not torch.is_anomaly_enabled()
    wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,broadcast_buffers=False,find_unused_parameters=False) if world>1 else model
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};head_sha=state_sha(model.head)
    trainable=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in model.parameters())==17211738
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    optimizer=torch.optim.AdamW(trainable,**OmegaConf.to_container(cfg.experiment.optimizer))
    def update(micro,teacher_weight):
        optimizer.zero_grad(set_to_none=True);accum=args.local_constructs//micro;sync();started=time.perf_counter()
        model.counters={'phase':'mean_forward','entries':collections.Counter()};rows=[];first_signal=None
        for group,position in enumerate(range(0,args.local_constructs,micro)):
            ids=logical_ids[position:position+micro];batch=repeat_native(single,micro)
            first=[];sync();clock=time.perf_counter();model.counters['phase']='mean_forward'
            with wrapped.no_sync() if world>1 else nullcontext():
                with torch.no_grad():
                    for member in range(args.ensemble_size):first.append(wrapped('generated',batch,ids,position,member)[0])
                    bulk,alpha=case_mean_loss_and_derivative(torch.stack(first).mean(0),target[position:position+micro],weights[position:position+micro])
            sync();mean_seconds=time.perf_counter()-clock;forward_seconds=0.;backward_seconds=0.;replay=0.;constraint_mean=0.
            for member in range(args.ensemble_size):
                with wrapped.no_sync() if world>1 else nullcontext():
                    model.counters['phase']='gradient_forward';sync();clock=time.perf_counter()
                    profile,constraint=wrapped('generated',batch,ids,position,member);sync();forward_seconds+=time.perf_counter()-clock
                    replay=max(replay,float((profile.detach()-first[member]).abs().max()))
                    if not torch.allclose(profile.detach(),first[member],atol=1e-5,rtol=1e-4):raise RuntimeError('Batched profile replay differs')
                    model.counters['phase']='backward_recompute';sync();clock=time.perf_counter()
                    ((.1*constraint+(alpha*profile).sum())/(args.ensemble_size*accum)).backward()
                    sync();backward_seconds+=time.perf_counter()-clock;constraint_mean+=float(constraint.detach())/args.ensemble_size
            norm=gradient_norm(model.flow.model.parameters(),True)
            adapter=[p for name,p in model.flow.model.node_embedder.named_parameters() if not name.startswith('base_embedder.')]
            probe_norm=gradient_norm(adapter,True)
            if group==0:first_signal={'generated_inverse_gradient_norm':norm,'generated_probe_gradient_norm':probe_norm}
            noisy=repeat_native(single,micro);noisy.update({key:value[position:position+micro].clone() for key,value in teacher.items()})
            seed_member(args.seed+20000039+position,device);model.counters['phase']='native_teacher_forward';sync();clock=time.perf_counter()
            with wrapped.no_sync() if world>1 and group<accum-1 else nullcontext():
                teacher_loss=wrapped('teacher',noisy,ids,position)
                (teacher_weight*teacher_loss/accum).backward()
            sync();teacher_seconds=time.perf_counter()-clock
            rows.append({'logical_ids':ids,'mean_pass_seconds':mean_seconds,'gradient_forward_seconds':forward_seconds,
                         'backward_recompute_seconds':backward_seconds,'teacher_forward_backward_seconds':teacher_seconds,
                         'finite_K_case_mean_profile_MSE':float(bulk),'adjacent_C4_mean':constraint_mean,
                         'teacher_loss':float(teacher_loss.detach()),'maximum_replay_difference':replay})
        norm=gradient_norm(model.flow.model.parameters(),True)
        if any(p.grad is not None for p in model.head.parameters()):raise RuntimeError('Frozen head received gradients')
        gradients={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None} if args.cpu_proof else None
        torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step();sync();seconds=time.perf_counter()-started
        duration=torch.tensor(seconds,device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(duration,op=dist.ReduceOp.MAX)
        counts=dict(model.counters['entries']);expected=accum*args.ensemble_size*args.grid_points
        assert counts['mean_forward']==expected and counts['gradient_forward']==expected and counts['native_teacher_forward']==accum
        assert 0<counts['backward_recompute']<=expected and state_sha(model.head)==head_sha
        return {'microbatch':micro,'accumulation':accum,'local_constructs':args.local_constructs,
                'global_constructs':args.local_constructs*world,'rank_max_seconds':float(duration),
                'global_case_instances_per_second':args.local_constructs*world/float(duration),
                'first_microbatch_generated_signal':first_signal,'all_gradients_finite':True,'combined_preclip_gradient_norm':norm,
                'frozen_head_gradients_absent':True,'actual_call_entries':counts,'groups':rows},gradients
    original_freeze=json.loads(Path(__file__).with_name('original_frozen_manifest.json').read_text())['freeze_sha256']
    context={'length':length,'source_directory':str(source),'observations_by_logical_id':observation_hashes,
             'teacher_by_logical_id':{str(logical):records[logical]['raw_tensor_sha256'] for logical in logical_ids}}
    run_reliability(args,model,optimizer,update,device,rank,world,logical_ids,fixture_path,head_sha,original_freeze,context)


if __name__=='__main__':main()
