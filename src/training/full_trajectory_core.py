#!/usr/bin/env python3
"""Differentiable native-schedule Euler/SO3 trajectory with bounded activations.

Unlike the pinned inference sampler, model calls and final coordinates remain
Torch tensors in the graph. T grid points mean T-1 Euler intervals plus a final
t=1 denoiser call, matching the public num_timesteps convention. Checkpointing
uses the non-reentrant API; recomputation uses each step's bound time values.
"""
from __future__ import annotations
import collections
import json
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ensemble_mean_core import FrozenGeometryChemistryHead,bulk_loss_and_derivative


def full_trajectory(vector_field,static,initial_trans,initial_rot,grid_points,geodesic,
                    checkpoint_steps=True,counters=None):
    assert grid_points>=2
    # Native inference uses this float32 CPU grid; time is an external constant.
    grid=torch.linspace(.01,1.,grid_points).tolist()
    device=initial_trans.device;batch=initial_trans.shape[0]
    def invoke(trans,rot,t):
        if counters is not None:
            phase=counters['phase'];counters['entries'][phase]+=1
        inputs={'trans_t':trans,'rotmats_t':rot,'t':torch.full((batch,1),t,device=device,dtype=trans.dtype),
                'res_mask':static['res_mask'],'onehot':static['onehot'],'ss':static['ss']}
        return vector_field(inputs)
    trans,rot=initial_trans,initial_rot
    for left,right in zip(grid[:-1],grid[1:]):
        # Defaults bind the current interval; late-bound loop variables would
        # silently recompute the wrong dynamics during backward.
        def step(tx,rx,t=left,dt=right-left):
            predicted=invoke(tx,rx,t)
            updated_trans=tx+dt*(predicted['pred_trans']-tx)/(1-t)
            updated_rot=geodesic(10*dt,predicted['pred_rotmats'],rx)
            return updated_trans,updated_rot
        if checkpoint_steps and torch.is_grad_enabled():
            trans,rot=checkpoint(step,trans,rot,use_reentrant=False,preserve_rng_state=True)
        else:trans,rot=step(trans,rot)
    def final(tx,rx):
        prediction=invoke(tx,rx,1.)
        return prediction['pred_trans'],prediction['pred_rotmats'],prediction['pred_torsions']
    if checkpoint_steps and torch.is_grad_enabled():
        return checkpoint(final,trans,rot,use_reentrant=False,preserve_rng_state=True)
    return final(trans,rot)


def native_adjacent_distance_constraint(c4,native_c4,valid):
    predicted=torch.sqrt((c4[:,1:]-c4[:,:-1]).square().sum(-1)+1e-8)
    target=torch.sqrt((native_c4[:,1:]-native_c4[:,:-1]).square().sum(-1)+1e-8)
    pairs=valid[:,1:]*valid[:,:-1]
    return (pairs*((predicted-target)/10).square()).sum()/pairs.sum().clamp_min(1)


def cpu_trajectory_test(so3):
    """Actual public SO3 maps + checkpoint/unrolled gradient tests at K2/T2."""
    torch.set_num_threads(2);torch.manual_seed(20261012)
    dtype=torch.float64;length=8;valid=torch.ones(1,length,dtype=dtype)
    native=torch.randn(1,length,3,dtype=dtype);chem=torch.tensor([1])
    observations=torch.randn(1,length,4,dtype=dtype)
    static={'res_mask':valid,'onehot':torch.ones(1,length,4,dtype=dtype)/4,
            'ss':torch.zeros(1,length,length,3,dtype=dtype)}
    target=torch.rand(1,length,dtype=dtype);weights=torch.ones_like(target)
    class TinyVectorField(nn.Module):
        def __init__(self):
            super().__init__();self.translation=nn.Linear(3,3);self.rotation=nn.Linear(3,3)
            self.probe=nn.Linear(4,3);self.active=False
        def forward(self,inputs):
            state=inputs['trans_t']
            offset=float(self.active)*self.probe(observations)
            predicted_trans=state*.8+.1*self.translation(state)+.05*offset
            delta=.05*self.rotation(state)+.02*offset
            predicted_rot=inputs['rotmats_t']@so3.rotvec_to_rotmat(delta)
            torsions=torch.ones(1,length,9,2,dtype=dtype)
            return {'pred_trans':predicted_trans,'pred_rotmats':predicted_rot,'pred_torsions':torsions}
    field=TinyVectorField().to(dtype=dtype);head=FrozenGeometryChemistryHead(width=16).to(dtype=dtype)
    initial={k:v.detach().clone() for k,v in field.state_dict().items()};rows=[]
    def generate(member,use_checkpoint,counters=None):
        torch.manual_seed(3001+member)
        tx=torch.randn(1,length,3,dtype=dtype)
        rot=so3.rotvec_to_rotmat(.1*torch.randn(1,length,3,dtype=dtype))
        trans,rotation,_=full_trajectory(field,static,tx,rot,2,so3.geodesic_t,use_checkpoint,counters)
        # A fixed local rigid attachment tests rotational as well as positional
        # gradients. Actual full-model driver uses native atom23 C4' conversion.
        c4=trans+torch.einsum('...ij,j->...i',rotation,torch.tensor([.5,1.,.2],dtype=dtype))
        return head(c4,chem,valid),native_adjacent_distance_constraint(c4,native,valid)
    for active in (False,True):
        field.active=active;field.load_state_dict(initial);field.zero_grad(set_to_none=True)
        profiles=[];constraints=[]
        for member in range(2):
            profile,constraint=generate(member,False);profiles.append(profile);constraints.append(constraint)
        bulk,_=bulk_loss_and_derivative(torch.stack(profiles).mean(0),target,weights)
        total=bulk+torch.stack(constraints).mean();total.backward()
        gradients={name:p.grad.detach().clone() for name,p in field.named_parameters()}
        field.load_state_dict(initial);field.zero_grad(set_to_none=True)
        counters={'phase':'mean_forward','entries':collections.Counter()}
        with torch.no_grad():
            first=[generate(member,False,counters)[0] for member in range(2)]
            _,alpha=bulk_loss_and_derivative(torch.stack(first).mean(0),target,weights)
        max_profile_delta=0.
        for member in range(2):
            counters['phase']='gradient_forward';profile,constraint=generate(member,True,counters)
            max_profile_delta=max(max_profile_delta,float((profile.detach()-first[member]).abs().max()))
            counters['phase']='backward_recompute'
            (constraint/2+(alpha*profile).sum()/2).backward()
        delta=max(float((p.grad-gradients[name]).abs().max()) for name,p in field.named_parameters())
        assert all(bool(torch.isfinite(p.grad).all()) for p in field.parameters())
        assert delta<1e-8 and max_profile_delta<1e-12
        assert float(field.translation.weight.grad.norm())>0 and float(field.rotation.weight.grad.norm())>0
        if active:assert float(field.probe.weight.grad.norm())>0
        else:assert float(field.probe.weight.grad.norm())==0
        assert all(p.grad is None for p in head.parameters())
        rows.append({'active':active,'K':2,'grid_points':2,'Euler_intervals':1,
                     'maximum_parameter_gradient_difference':delta,'maximum_profile_replay_difference':max_profile_delta,
                     'translation_gradient_norm':float(field.translation.weight.grad.norm()),
                     'rotation_gradient_norm':float(field.rotation.weight.grad.norm()),
                     'probe_gradient_norm':float(field.probe.weight.grad.norm()),
                     'actual_denoiser_call_entries':dict(counters['entries'])})
    return {'status':'Actually executed CPU differentiable native-SO3 K2/T2 checkpoint/two-pass gradient test passed',
            'torch_version':torch.__version__,'dtype':'float64','tests':rows,
            'scope':'Native public SO3/Euler updates plus tiny vector field; full RNAbpFlow T50 GPU graph has a separate qualification',
            'checkpoint_reference':'https://docs.pytorch.org/docs/2.10/checkpoint.html'}
