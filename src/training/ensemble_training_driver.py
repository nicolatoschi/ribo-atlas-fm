#!/usr/bin/env python3
"""Systems timing of a one-step denoised finite-K ensemble regularizer.

Full published geometry corruption/decoder/loss is retained. A random FROZEN
geometry+chemistry-only head projects predicted C4' geometry to a synthetic
profile. This is a denoised-ensemble training regularizer, distinct from the
final50-step generated-ensemble objective. It is not scientific efficacy data.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from ensemble_mean_core import (FrozenGeometryChemistryHead,MatchedProbeNodeEncoder,
                                bulk_loss_and_derivative,member_seed,seed_member)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def state_sha(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def gradient_norm(parameters,required):
    gradients=[p.grad for p in parameters if p.grad is not None]
    if not gradients:raise RuntimeError('Missing gradients')
    if not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):raise RuntimeError('Nonfinite gradients')
    norm=torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in gradients]))
    if not bool(torch.isfinite(norm)) or required and not bool(norm>0):raise RuntimeError('Missing finite nonzero gradient signal')
    return float(norm)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--csv',default='metadata/pilot_ensemble.csv')
    p.add_argument('--cases',default='1g59_B,1l8v_B,2r8s_R,7ptk_B')
    p.add_argument('--ensemble-sizes',default='1,8,32');p.add_argument('--modes',default='inactive,active')
    p.add_argument('--repetitions',type=int,default=1);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--seed',type=int,default=20261012)
    p.add_argument('--bulk-weight',type=float,default=1);p.add_argument('--geometry-weight',type=float,default=1)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--bulk-only-check',action='store_true')
    args=p.parse_args();source=args.source.resolve();out=args.out.resolve()
    sizes=[int(x) for x in args.ensemble_sizes.split(',')];modes=args.modes.split(',');names=args.cases.split(',')
    assert all(k in [1,8,32] for k in sizes) and set(modes)<=set(['inactive','active'])
    assert args.steps>0 and args.warmup>=1 and args.repetitions>0 and args.bulk_weight>0 and args.geometry_weight>0
    os.chdir(source);sys.path.insert(0,str(source))
    from src.data.pdb_na_dataset_base import PDBNABaseDataset
    from src.models.flow_module import FlowModule
    from src.data import all_atom,base_constants
    c4_index=base_constants.atom_order["C4'"]
    assert c4_index==5 and all(base_constants.residue_atoms[b][c4_index]=="C4'" for b in ['A','C','G','U'])
    rank=int(os.environ.get('RANK',os.environ.get('SLURM_PROCID','0')))
    world=int(os.environ.get('WORLD_SIZE',os.environ.get('SLURM_NTASKS','1')))
    local=int(os.environ.get('LOCAL_RANK',os.environ.get('SLURM_LOCALID','0')))
    if args.device=='cuda':
        assert torch.cuda.is_available();index=local if torch.cuda.device_count()>local else 0
        torch.cuda.set_device(index);device=torch.device('cuda',index)
    else:device=torch.device('cpu')
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',rank=rank,world_size=world,timeout=datetime.timedelta(minutes=20))
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    seed_member(args.seed,device);torch.set_float32_matmul_precision('high')
    cfg=OmegaConf.load(source/'configs/config.yaml');cfg.data_cfg.csv_path=str(source/args.csv)
    cfg.data_cfg.filtering.max_len=800;cfg.experiment.checkpointer.dirpath=str(out/'upstream_checkpoint_directory')
    assert not cfg.interpolant.self_condition,'Configured self-conditioning needs its own explicit benchmark branch'
    data=PDBNABaseDataset(cfg.data_cfg,is_training=True)
    cases=[]
    for name in names:
        index=next(i for i in range(len(data)) if str(data.csv.iloc[i]['pdb_name'])==name)
        batch={k:torch.as_tensor(v).unsqueeze(0).to(device) for k,v in data[index].items()}
        for key,value in batch.items():
            if value.is_floating_point():batch[key]=value.float()
            if batch[key].is_floating_point() and not bool(torch.isfinite(batch[key]).all()):raise RuntimeError('Nonfinite real native tensor '+key)
        cases.append((name,batch))

    class Workload(nn.Module):
        def __init__(self):
            super().__init__();self.flow=FlowModule(cfg)
            self.flow.model.node_embedder=MatchedProbeNodeEncoder(self.flow.model.node_embedder,int(cfg.model.node_embed_size))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed+701);self.forward_head=FrozenGeometryChemistryHead()
            self.prediction=None
            self.flow.model.register_forward_hook(self.capture_prediction)
        def capture_prediction(self,module,inputs,prediction):self.prediction=prediction
        def forward(self,noisy,chemistry,probe,with_geometry_loss=True):
            self.flow.model.node_embedder.current_probe=probe
            if with_geometry_loss:
                parts=self.flow.model_step(noisy)
                if not all(bool(torch.isfinite(value).all()) for value in parts.values()):raise RuntimeError('Nonfinite original loss component')
                geometry=parts[cfg.experiment.training.loss].mean()+parts['auxiliary_loss'].mean()
            else:
                self.flow.model(noisy);geometry=torch.zeros((),device=device)
            predicted=self.prediction
            atoms=all_atom.to_atom23_rna(predicted['pred_trans'],predicted['pred_rotmats'],noisy['aatype'],torsions=predicted['pred_torsions'])
            profile=self.forward_head(atoms[:,:,c4_index],chemistry,noisy['res_mask'])
            if not bool(torch.isfinite(profile).all()):raise RuntimeError('Nonfinite forward profile')
            return profile,geometry

    model=Workload().to(device);torch.autograd.set_detect_anomaly(False)
    model.flow.interpolant.set_device(device)
    cache_start=time.perf_counter()
    if rank==0:_=model.flow.interpolant.igso3
    if world>1:dist.barrier()
    if rank!=0:_=model.flow.interpolant.igso3
    cache_seconds=time.perf_counter()-cache_start
    wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,
                broadcast_buffers=False,find_unused_parameters=False) if world>1 else model
    initial={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    trainable=[parameter for parameter in model.parameters() if parameter.requires_grad]
    count=sum(p.numel() for p in model.parameters());train_count=sum(p.numel() for p in trainable)
    assert count==17211738 and train_count==17207065
    fixed_head_sha=state_sha(model.forward_head)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def synthetic_observations(batch,case_index):
        chemistry=torch.tensor([rank%2],device=device)
        native=all_atom.to_atom23_rna(batch['trans_1'],batch['rotmats_1'],batch['aatype'],torsions=batch['torsion_angles_sin_cos'])[:,:,c4_index]
        centered=native-native.mean(-2,keepdim=True)
        radius=torch.sqrt(centered.square().sum(-1)+1e-8)
        rng=np.random.default_rng(args.seed+case_index*503+rank*97)
        noise=torch.tensor(rng.normal(0,.02,radius.shape),dtype=radius.dtype,device=device)
        target=(torch.sigmoid(radius/15-1)+chemistry[:,None]*.1+noise).clamp_min(0)
        observed=torch.tensor(rng.random(radius.shape)>.2,device=device);observed[:,0]=True
        observed=observed*batch['res_mask'].bool();error=torch.full_like(target,.05)
        weights=observed.to(target.dtype)/error.square()
        probe=torch.stack([target*observed,error*observed,observed.to(target.dtype),chemistry[:,None].expand_as(target).to(target.dtype)],-1)
        return chemistry,target,weights,probe

    def update(optimizer,batch,case_index,repetition,identity,k,chemistry,target,weights,probe,geometry_weight,require_active):
        optimizer.zero_grad(set_to_none=True);sync();begin=time.perf_counter()
        first=[]
        first_context=wrapped.no_sync() if world>1 else nullcontext()
        with first_context,torch.no_grad():
            for member in range(k):
                seed_member(member_seed(args.seed,case_index,repetition,identity,rank,member),device)
                noisy=model.flow.interpolant.corrupt_batch(batch)
                first.append(wrapped(noisy,chemistry,probe,False)[0].detach())
            mean=torch.stack(first).mean(0);bulk,alpha=bulk_loss_and_derivative(mean,target,weights)
        geometry_total=0.;replay_max=0.
        for member in range(k):
            seed_member(member_seed(args.seed,case_index,repetition,identity,rank,member),device)
            context=wrapped.no_sync() if world>1 and member<k-1 else nullcontext()
            with context:
                noisy=model.flow.interpolant.corrupt_batch(batch)
                profile,geometry=wrapped(noisy,chemistry,probe,True)
                delta=float((profile.detach()-first[member]).abs().max());replay_max=max(replay_max,delta)
                if not torch.allclose(profile.detach(),first[member],atol=1e-5,rtol=1e-4):raise RuntimeError('Common-RNG profile replay differs')
                surrogate=geometry_weight*geometry/k+args.bulk_weight*(alpha*profile).sum()/k
                if not bool(torch.isfinite(surrogate)):raise RuntimeError('Nonfinite streamed surrogate')
                surrogate.backward();geometry_total+=float(geometry.detach())/k
        inverse_norm=gradient_norm(model.flow.model.parameters(),True)
        adapter=model.flow.model.node_embedder
        adapter_norm=gradient_norm([p for name,p in adapter.named_parameters() if not name.startswith('base_embedder.')],require_active)
        if not adapter.active and adapter_norm!=0:raise RuntimeError('Inactive probe gate has unexpected gradient')
        if any(p.grad is not None for p in model.forward_head.parameters()):raise RuntimeError('The forward head was not frozen')
        optimizer.step();sync();seconds=time.perf_counter()-begin
        return {'optimizer_identity':identity,'local_seconds':seconds,'finite_K_mean_profile_MSE':float(bulk),
                'averaged_published_geometry_loss':geometry_total,'logged_objective':geometry_weight*geometry_total+args.bulk_weight*float(bulk),
                'inverse_model_gradient_norm':inverse_norm,'probe_adapter_gradient_norm':adapter_norm,
                'maximum_common_RNG_profile_replay_absolute_difference':replay_max,
                'forward_head_parameter_gradients_absent':True,'first_pass_members':k,'gradient_pass_members':k}

    results=[]
    for case_index,(name,batch) in enumerate(cases):
        chemistry,target,weights,probe=synthetic_observations(batch,case_index)
        for mode in modes:
            model.flow.model.node_embedder.active=mode=='active'
            for k in sizes:
                rows=[]
                for repetition in range(args.repetitions):
                    model.load_state_dict(initial);optimizer=torch.optim.AdamW(trainable,**OmegaConf.to_container(cfg.experiment.optimizer))
                    for i in range(args.warmup):update(optimizer,batch,case_index,repetition,i,k,chemistry,target,weights,probe,args.geometry_weight,False)
                    if world>1:dist.barrier()
                    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                    timed=[update(optimizer,batch,case_index,repetition,args.warmup+i,k,chemistry,target,weights,probe,args.geometry_weight,mode=='active') for i in range(args.steps)]
                    elapsed=torch.tensor(sum(x['local_seconds'] for x in timed),device=device,dtype=torch.float64)
                    if world>1:dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
                    allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
                    reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None
                    total_memory=torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None
                    if device.type=='cuda' and reserved>.85*total_memory:raise RuntimeError('Peak reserved HBM exceeds85%qualification gate')
                    rows.append({'repetition':repetition,'rank_max_measured_seconds':float(elapsed),'global_constructs_per_second':args.steps*world/float(elapsed),
                                 'GPU_peak_allocated_bytes':allocated,'GPU_peak_reserved_bytes':reserved,'GPU_total_memory_bytes':total_memory,'measured_updates':timed})
                assert state_sha(model.forward_head)==fixed_head_sha,'Frozen forward head changed'
                report={'status':'Actually executed one-step denoised finite-K ensemble-mean regularizer training',
                        'rank':rank,'world_size':world,'case':name,'length':int(batch['res_mask'].shape[1]),'mode':mode,'K_train':k,
                        'local_construct_batch':1,'global_construct_batch':world,'repetitions':args.repetitions,'warmup_updates':args.warmup,'measured_updates':args.steps,
                        'parameter_count_total':count,'trainable_inverse_parameter_count':train_count,'frozen_forward_head_parameters':4673,
                        'frozen_forward_head_state_sha256':fixed_head_sha,'forward_head_input_contract':'Predicted native C4coordinates + chemistry identity + native residue validity; no observed profile, errors or probe-observation mask',
                        'observation_role':'Synthetic system-profiling target/conditioning features, fixed across allK members for eachconstruct',
                        'geometry_role':'Real archived native RNAframes/torsions/pairs; one-step denoised projections from independently corrupted frames',
                        'regularizer_scope':'Mean-profile finite-K MSE contains MonteCarlo variance/K; separate from final50-step generated-ensemble objective',
                        'objective':'geometry_weight*mean_K(publishedSE3+auxloss)+bulk_weight*weightedMSE(mean_K(frozen_forward_head(C4denoised_geometry,chemistry)),synthetic_target)',
                        'geometry_weight':args.geometry_weight,'bulk_weight':args.bulk_weight,
                        'gradient_method':'No-grad mean pass; common-RNG graph replay of geometry_loss/K+bulk_weight*dot(detached2*weights*(mean-target)/sumweights,profile)/K',
                        'head_training_role':'Random fixed systems projection, no learned/cross-fitted scientific-head claim',
                        'arithmetic':'FP32parameters/losses,highfloat32matmulprecision,noautocast','seed':args.seed,
                        'source_sha256':{'driver':sha(Path(__file__)),'core':sha(Path(__file__).with_name('ensemble_mean_core.py')),
                                         'flow_model':sha(source/'src/models/flow_model.py'),'flow_module':sha(source/'src/models/flow_module.py'),
                                         'interpolant':sha(source/'src/data/interpolant.py'),'csv':sha(source/args.csv)},
                        'torch_version':torch.__version__,'GPU_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                        'IGSO3_cache_setup_seconds':cache_seconds,'timing_includes':'Bothmean/replaypasses,originalcorruption,fullmodel,frozenhead,originalgeometryloss,backward,numericalchecks,DDPfinalreduction,AdamW,CUDAcompletion',
                        'timing_excludes':'Import/cache/data preload/target creation/model reset/warmup/reporting; actual parent accounting includes these',
                        'repetition_results':rows}
                atomic_json(out/f'rank_{rank:05d}_{name}_{mode}_K{k}.json',report)
                results.append({'case':name,'mode':mode,'K':k,'block_seconds':rows[0]['rank_max_measured_seconds']})
                if rank==0:print(json.dumps(results[-1]),flush=True)
        if args.bulk_only_check:
            model.load_state_dict(initial);model.flow.model.node_embedder.active=True
            optimizer=torch.optim.AdamW(trainable,**OmegaConf.to_container(cfg.experiment.optimizer))
            # One geometric warm-up opens the native denoiser update before the
            # bulk-only gradient check; this is excluded from rate measurements.
            update(optimizer,batch,case_index,0,100,8,chemistry,target,weights,probe,args.geometry_weight,False)
            check=update(optimizer,batch,case_index,0,101,8,chemistry,target,weights,probe,0.,True)
            atomic_json(out/f'rank_{rank:05d}_{name}_bulk_only_check.json',
                        {'status':'Actual finite nonzero bulk-only inverse/probe gradients after geometric warmup','rank':rank,'world_size':world,'case':name,'K':8,'geometry_weight':0,'result':check})
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
