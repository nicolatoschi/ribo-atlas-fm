#!/usr/bin/env python3
"""Train through actual final generated geometry at K8/32 and50 evaluations.

The official pretrained vector field initializes this engineering qualification.
Synthetic observations and a random frozen geometry+chemistry head establish
execution, gradients, memory and cost; they do not establish scientific utility.
Each generated trajectory stays in Torch through the final C4'/mean-profile loss.
"""
from __future__ import annotations
import argparse
import collections
from contextlib import nullcontext
import datetime
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
from scipy.spatial.transform import Rotation
from ensemble_mean_core import (FrozenGeometryChemistryHead,MatchedProbeNodeEncoder,
                                bulk_loss_and_derivative,member_seed,seed_member)
from ensemble_training_driver import atomic_json,state_sha,sha,gradient_norm
from full_trajectory_core import full_trajectory,native_adjacent_distance_constraint
from differentiable_rotation import install_native_quaternion_derivative

CHECKPOINT_SHA='bb598a736193734541f63c59dae75d45db8c04f01a8414a758677367c4b1e213'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--csv',default='metadata/pilot_ensemble.csv');p.add_argument('--case',default='1l8v_B')
    p.add_argument('--ensemble-sizes',default='8,32');p.add_argument('--grid-points',type=int,default=50)
    p.add_argument('--modes',default='inactive,active');p.add_argument('--steps',type=int,default=1)
    p.add_argument('--warmup',type=int,default=0);p.add_argument('--repetitions',type=int,default=1)
    p.add_argument('--seed',type=int,default=20261012);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--bulk-weight',type=float,default=1.);p.add_argument('--constraint-weight',type=float,default=.1)
    p.add_argument('--teacher-weight',type=float,default=1.);p.add_argument('--clip-gradient-norm',type=float,default=1.)
    args=p.parse_args();source=args.source.resolve();out=args.out.resolve();checkpoint_path=args.checkpoint.resolve()
    sizes=[int(k) for k in args.ensemble_sizes.split(',')];modes=args.modes.split(',')
    assert set(sizes)<=set([2,8,32]) and set(modes)<=set(['inactive','active'])
    assert args.grid_points>=2 and args.steps>=1 and args.repetitions>=1 and args.warmup>=0
    assert args.bulk_weight>0 and args.constraint_weight>=0 and args.teacher_weight>=0 and args.clip_gradient_norm>0
    assert sha(checkpoint_path)==CHECKPOINT_SHA,'Official checkpoint content changed'
    os.chdir(source);sys.path.insert(0,str(source))
    from src.models.flow_module import FlowModule
    from src.data.pdb_na_dataset_base import PDBNABaseDataset
    from src.data import all_atom,base_constants,so3_utils,rigid_utils
    install_native_quaternion_derivative(rigid_utils)
    rank=int(os.environ.get('RANK',os.environ.get('SLURM_PROCID','0')))
    world=int(os.environ.get('WORLD_SIZE',os.environ.get('SLURM_NTASKS','1')))
    local=int(os.environ.get('LOCAL_RANK',os.environ.get('SLURM_LOCALID','0')))
    if args.device=='cuda':
        assert torch.cuda.is_available();index=local if torch.cuda.device_count()>local else 0
        torch.cuda.set_device(index);device=torch.device('cuda',index)
    else:device=torch.device('cpu')
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',rank=rank,world_size=world,timeout=datetime.timedelta(minutes=30))
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    torch.set_float32_matmul_precision('high');seed_member(args.seed,device)
    # Eval-mode TransformerEncoder otherwise chooses a fused no-grad path in
    # the mean pass and a different gradient path in replay. Use matching math.
    torch.backends.mha.set_fastpath_enabled(False)
    cfg=OmegaConf.load(source/'configs/config.yaml');cfg.data_cfg.csv_path=str(source/args.csv)
    cfg.data_cfg.filtering.max_len=800;cfg.experiment.checkpointer.dirpath=str(out/'upstream_checkpoint_directory')
    assert not cfg.interpolant.self_condition,'This native schedule is qualified with self_condition=False'
    data=PDBNABaseDataset(cfg.data_cfg,is_training=True)
    index=next(i for i in range(len(data)) if str(data.csv.iloc[i]['pdb_name'])==args.case)
    batch={k:torch.as_tensor(v).unsqueeze(0).to(device) for k,v in data[index].items()}
    for key,value in batch.items():
        if value.is_floating_point():batch[key]=value.float()
        if batch[key].is_floating_point() and not bool(torch.isfinite(batch[key]).all()):raise RuntimeError('Nonfinite native tensor '+key)
    c4_index=base_constants.atom_order["C4'"];assert c4_index==5
    native_c4=all_atom.to_atom23_rna(batch['trans_1'],batch['rotmats_1'],batch['aatype'],torsions=batch['torsion_angles_sin_cos'])[:,:,c4_index]
    chemistry=torch.tensor([rank%2],device=device)
    rng=np.random.default_rng(args.seed+rank*97)
    centered=native_c4-native_c4.mean(-2,keepdim=True);radius=torch.sqrt(centered.square().sum(-1)+1e-8)
    target=(torch.sigmoid(radius/15-1)+chemistry[:,None]*.1+
            torch.tensor(rng.normal(0,.02,radius.shape),device=device,dtype=torch.float32)).clamp_min(0)
    observed=torch.tensor(rng.random(radius.shape)>.2,device=device);observed[:,0]=True
    observed=observed*batch['res_mask'].bool();error=torch.full_like(target,.05)
    weights=observed.to(target.dtype)/error.square()
    probe=torch.stack([target*observed,error*observed,observed.to(target.dtype),chemistry[:,None].expand_as(target).float()],-1)

    class Workload(nn.Module):
        def __init__(self):
            super().__init__();self.flow=FlowModule(cfg)
            saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            state={key.removeprefix('model.'):value for key,value in saved['state_dict'].items() if key.startswith('model.')}
            self.flow.model.load_state_dict(state,strict=True)
            self.flow.model.node_embedder=MatchedProbeNodeEncoder(self.flow.model.node_embedder,int(cfg.model.node_embed_size))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed+701);self.forward_head=FrozenGeometryChemistryHead()
            self.counters={'phase':'setup','entries':collections.Counter()}
        def forward(self,kind,noisy,steps=50,checkpoint_steps=True):
            self.flow.model.node_embedder.current_probe=probe
            if kind=='teacher':
                self.flow.model.train()
                self.counters['entries']['native_teacher_forward']+=1
                parts=self.flow.model_step(noisy)
                if not all(bool(torch.isfinite(v).all()) for v in parts.values()):raise RuntimeError('Nonfinite published teacher loss')
                return parts[cfg.experiment.training.loss].mean()+parts['auxiliary_loss'].mean()
            self.flow.model.eval()  # Native production sampler uses eval mode.
            # Initial prior is parameter-independent; every later state stays
            # differentiable. No native target frame enters the vector field.
            tx=torch.randn(1,int(batch['res_mask'].shape[1]),3,device=device)
            tx=(tx-tx.mean(-2,keepdim=True))*10
            rot=torch.tensor(Rotation.random(int(batch['res_mask'].shape[1])).as_matrix(),device=device,dtype=torch.float32)[None]
            trans,rotation,torsions=full_trajectory(self.flow.model,batch,tx,rot,steps,so3_utils.geodesic_t,checkpoint_steps,self.counters)
            atoms=all_atom.to_atom23_rna(trans,rotation,batch['aatype'],torsions=torsions)
            c4=atoms[:,:,c4_index]
            profile=self.forward_head(c4,chemistry,batch['res_mask'])
            constraint=native_adjacent_distance_constraint(c4,native_c4,batch['res_mask'])
            if not bool(torch.isfinite(profile).all()) or not bool(torch.isfinite(constraint)):raise RuntimeError('Nonfinite final generated output/loss')
            return profile,constraint

    model=Workload().to(device);torch.autograd.set_detect_anomaly(False)
    model.flow.interpolant.set_device(device)
    if rank==0:_=model.flow.interpolant.igso3
    if world>1:dist.barrier()
    if rank!=0:_=model.flow.interpolant.igso3
    wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,
                broadcast_buffers=False,find_unused_parameters=False) if world>1 else model
    initial={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    trainable=[parameter for parameter in model.parameters() if parameter.requires_grad]
    fixed_head_sha=state_sha(model.forward_head)
    total_parameters=sum(p.numel() for p in model.parameters());train_count=sum(p.numel() for p in trainable)
    assert total_parameters==17211738 and train_count==17207065
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def optimizer():return torch.optim.AdamW(trainable,**OmegaConf.to_container(cfg.experiment.optimizer))

    def update(opt,repetition,identity,k,t_steps,constraint_weight,teacher_weight):
        opt.zero_grad(set_to_none=True);sync();started=time.perf_counter()
        model.counters={'phase':'mean_forward','entries':collections.Counter()}
        first=[];mean_started=time.perf_counter()
        context=wrapped.no_sync() if world>1 else nullcontext()
        with context,torch.no_grad():
            for member in range(k):
                seed_member(member_seed(args.seed,0,repetition,identity,rank,member),device)
                first.append(wrapped('generated',batch,t_steps,False)[0])
            bulk,alpha=bulk_loss_and_derivative(torch.stack(first).mean(0),target,weights)
        sync();mean_seconds=time.perf_counter()-mean_started
        replay_seconds=0.;backward_seconds=0.;replay_delta=0.;constraint_mean=0.
        # Every generated member is no_sync. A final published teacher backward
        # touches all parameter hooks (including the pair projection) and then
        # reduces the accumulated generated+teacher gradients once per update.
        for member in range(k):
            seed_member(member_seed(args.seed,0,repetition,identity,rank,member),device)
            context=wrapped.no_sync() if world>1 else nullcontext()
            with context:
                model.counters['phase']='gradient_forward';sync();clock=time.perf_counter()
                profile,constraint=wrapped('generated',batch,t_steps,True)
                sync();replay_seconds+=time.perf_counter()-clock
                replay_delta=max(replay_delta,float((profile.detach()-first[member]).abs().max()))
                if not torch.allclose(profile.detach(),first[member],atol=1e-5,rtol=1e-4):raise RuntimeError('Full-trajectory common-RNG replay mismatch')
                surrogate=(constraint_weight*constraint+args.bulk_weight*(alpha*profile).sum())/k
                if not bool(torch.isfinite(surrogate)):raise RuntimeError('Nonfinite generated surrogate')
                model.counters['phase']='backward_recompute';sync();clock=time.perf_counter()
                surrogate.backward();sync();backward_seconds+=time.perf_counter()-clock
                constraint_mean+=float(constraint.detach())/k
        generated_norm=gradient_norm(model.flow.model.parameters(),True)
        adapter=model.flow.model.node_embedder
        adapter_parameters=[p for name,p in adapter.named_parameters() if not name.startswith('base_embedder.')]
        generated_probe_norm=gradient_norm(adapter_parameters,adapter.active)
        if not adapter.active and generated_probe_norm!=0:raise RuntimeError('Inactive adapter gradient is not zero')
        seed_member(args.seed+10000019+repetition*100003+identity*1009+rank,device)
        model.counters['phase']='native_teacher_forward';sync();teacher_started=time.perf_counter()
        noisy=model.flow.interpolant.corrupt_batch(batch)
        teacher=wrapped('teacher',noisy,t_steps,False)
        (teacher_weight*teacher).backward();sync();teacher_seconds=time.perf_counter()-teacher_started
        final_norm=gradient_norm(model.flow.model.parameters(),True)
        if any(p.grad is not None for p in model.forward_head.parameters()):raise RuntimeError('Forward head is not frozen')
        torch.nn.utils.clip_grad_norm_(trainable,args.clip_gradient_norm,error_if_nonfinite=True)
        opt.step();sync();seconds=time.perf_counter()-started
        counts=dict(model.counters['entries'])
        assert counts.get('mean_forward')==k*t_steps and counts.get('gradient_forward')==k*t_steps
        assert counts.get('backward_recompute',0)>0 and counts.get('native_teacher_forward')==1
        return {'optimizer_identity':identity,'K_train':k,'grid_points':t_steps,'local_seconds':seconds,
                'mean_pass_seconds':mean_seconds,'replay_forward_seconds':replay_seconds,
                'generated_backward_and_recompute_seconds':backward_seconds,'native_teacher_forward_backward_seconds':teacher_seconds,
                'finite_K_final_mean_profile_MSE':float(bulk),'native_adjacent_C4_constraint_mean':constraint_mean,
                'published_native_teacher_loss':float(teacher.detach()),
                'logged_objective':args.bulk_weight*float(bulk)+constraint_weight*constraint_mean+teacher_weight*float(teacher.detach()),
                'generated_only_gradient_norm':generated_norm,'generated_only_probe_gradient_norm':generated_probe_norm,
                'combined_preclip_gradient_norm':final_norm,'gradient_clip_max_norm':args.clip_gradient_norm,
                'maximum_profile_replay_difference':replay_delta,'frozen_head_has_no_parameter_gradients':True,
                'actual_denoiser_call_entries_by_phase':counts,
                'counter_scope':'Entry counts; non-reentrant recomputation may stop early after required saved tensors; measured seconds include actual work'}

    # Full-model bulk-only qualification: no teacher/adjacency contribution to
    # the gradient. The zero-weight teacher supplies DDP hooks/reduction only.
    model.load_state_dict(initial);model.flow.model.node_embedder.active=True
    preflight=update(optimizer(),0,100,2,2,0.,0.)
    atomic_json(out/f'rank_{rank:05d}_full_model_T2_K2_bulk_only_qualification.json',
                {'status':'Actual full-model final-generated bulk-only gradient qualification passed',
                 'rank':rank,'world_size':world,'case':args.case,'result':preflight})
    for mode in modes:
        model.flow.model.node_embedder.active=mode=='active'
        for k in sizes:
            repetitions=[]
            for repetition in range(args.repetitions):
                model.load_state_dict(initial);opt=optimizer()
                for i in range(args.warmup):update(opt,repetition,i,k,args.grid_points,args.constraint_weight,args.teacher_weight)
                if world>1:dist.barrier()
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                rows=[update(opt,repetition,args.warmup+i,k,args.grid_points,args.constraint_weight,args.teacher_weight) for i in range(args.steps)]
                elapsed=torch.tensor(sum(x['local_seconds'] for x in rows),device=device,dtype=torch.float64)
                if world>1:dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
                allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
                reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None
                capacity=torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None
                if device.type=='cuda' and reserved>.85*capacity:raise RuntimeError('Reserved HBM exceeds85%capacity')
                repetitions.append({'repetition':repetition,'rank_max_seconds':float(elapsed),
                                    'global_constructs_per_second':args.steps*world/float(elapsed),
                                    'GPU_peak_allocated_bytes':allocated,'GPU_peak_reserved_bytes':reserved,
                                    'GPU_capacity_bytes':capacity,'updates':rows})
            assert state_sha(model.forward_head)==fixed_head_sha
            report={'status':'Actually executed differentiable final-generated ensemble-mean training qualification',
                    'rank':rank,'world_size':world,'case':args.case,'length':int(batch['res_mask'].shape[1]),
                    'mode':mode,'K_train':k,'time_grid_points':args.grid_points,'Euler_intervals':args.grid_points-1,
                    'model_evaluations_per_logical_generated_trajectory':args.grid_points,
                    'global_construct_batch':world,'local_construct_batch':1,'repetitions':args.repetitions,
                    'warmup_updates':args.warmup,'measured_updates':args.steps,
                    'timing_role':'Initial optimizer-step qualification' if args.warmup==0 else 'Post-warmup optimizer measurements',
                    'parameter_count_total':total_parameters,'trainable_inverse_parameters':train_count,
                    'official_pretrained_checkpoint_sha256':CHECKPOINT_SHA,'strict_pretrained_state_load':True,
                    'frozen_forward_head_state_sha256':fixed_head_sha,'frozen_head_parameters':4673,
                    'head_inputs':'Final generated native C4coordinates, chemistry identity, native residue validity only',
                    'forward_head_role':'Random fixed systems head; training-only cross-fitted scientific head is separate',
                    'observation_role':'Synthetic and fixed across everyK member of a construct; no experimental efficacy claim',
                    'objective':'teacher_weight*published flow-matching loss +constraint_weight*mean_K(nativeadjacentC4distanceconstraint) +bulk_weight*weightedMSE(mean_K(frozenhead(finalgeneratedC4,chemistry)),synthetic_target)',
                    'teacher_weight':args.teacher_weight,'constraint_weight':args.constraint_weight,'bulk_weight':args.bulk_weight,
                    'finite_K_scope':'Mean-profile MSE has MonteCarlo variance/K; no unbiased population-mean claim',
                    'trajectory_scope':f'Native{args.grid_points-1}Eulerintervals+finaldenoiser atT{args.grid_points}; tensors retained through finalprofileloss, activation-checkpointed and Kstreamed; no inference NumPy detach',
                    'synchronization':'All generated backward passes no_sync; final native teacher backward reduces combined gradient including otherwise-unused pairhead hooks',
                    'gradient_clip_max_norm':args.clip_gradient_norm,'seed':args.seed,
                    'arithmetic':'FP32parameters/losses,highfloat32matmulprecision,noautocast',
                    'MHA_fastpath_enabled':torch.backends.mha.get_fastpath_enabled(),
                    'matrix_quaternion_derivative':'Native eigensolver primal; isolated largest-eigenvector analytic derivative, no repeated unused-eigenvalue divisions',
                    'torch_version':torch.__version__,'GPU_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                    'source_sha256':{'driver':sha(Path(__file__)),'trajectory_core':sha(Path(__file__).with_name('full_trajectory_core.py')),
                                     'rotation_derivative_port':sha(Path(__file__).with_name('differentiable_rotation.py')),
                                     'rigid_utils':sha(source/'src/data/rigid_utils.py'),
                                     'ensemble_core':sha(Path(__file__).with_name('ensemble_mean_core.py')),
                                     'flow_model':sha(source/'src/models/flow_model.py'),'flow_module':sha(source/'src/models/flow_module.py'),
                                     'so3_utils':sha(source/'src/data/so3_utils.py'),'csv':sha(source/args.csv)},
                    'timing_includes':'Actual prior RNG/initialization, full trajectories, finalhead/constraint, graphreplay/backward/recompute, originalteacherforward/backward, finitechecks, DDP, clipping, AdamW,CUDAsynchronization',
                    'timing_excludes':'Imports/checkpoint load/data preload/lookup/targetconstruction/modelreset/warmup/report; parent allocation accounting includes those costs',
                    'repetition_results':repetitions}
            atomic_json(out/f'rank_{rank:05d}_{args.case}_{mode}_K{k}_T{args.grid_points}_full_training.json',report)
            if rank==0:print(json.dumps({'mode':mode,'K':k,'T':args.grid_points,'rank_max_seconds':repetitions[0]['rank_max_seconds']}),flush=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
