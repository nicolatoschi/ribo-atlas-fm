#!/usr/bin/env python3
"""Frozen geometry/chemistry head and streamed finite-K mean-loss gradient.

The head takes coordinates, chemistry identity and native residue validity only.
Observed reactivity, observation mask and errors are inputs to the inverse model
or loss, never to this forward head. This systems head is randomly initialized
and frozen; no cross-fitted learned forward model or scientific benefit is claimed.
"""
from __future__ import annotations
import json
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FrozenGeometryChemistryHead(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(6,width),nn.SiLU(),nn.Linear(width,width),
                                   nn.SiLU(),nn.Linear(width,1),nn.Softplus())
        for p in self.parameters():p.requires_grad_(False)

    def forward(self,c4_coordinates,chemistry_id,residue_validity):
        # Geometry invariants preserve translation/rotation invariance. Native
        # validity is structural metadata, separate from the probe observation mask.
        mask=residue_validity.to(c4_coordinates.dtype)
        total=mask.sum(-1,keepdim=True).clamp_min(1)
        center=(c4_coordinates*mask[...,None]).sum(-2)/total
        radial=torch.sqrt(((c4_coordinates-center[:,None])**2).sum(-1)+1e-8)/20
        adjacent=torch.sqrt(((c4_coordinates[:,1:]-c4_coordinates[:,:-1])**2).sum(-1)+1e-8)/10
        adjacent=adjacent*mask[:,1:]*mask[:,:-1]
        left=F.pad(adjacent,(1,0));right=F.pad(adjacent,(0,1))
        distances=torch.cdist(c4_coordinates,c4_coordinates)
        density=(torch.exp(-(distances/10)**2)*mask[:,None]).sum(-1)/total
        geometry=torch.stack([left,right,radial,density],-1)
        context=F.one_hot(chemistry_id.to(torch.long),num_classes=2).to(c4_coordinates.dtype)
        context=context[:,None].expand(-1,c4_coordinates.shape[1],-1)
        return self.network(torch.cat([geometry,context],-1)).squeeze(-1)*mask


class MatchedProbeNodeEncoder(nn.Module):
    """Matched architecture/computation; inactive gate keeps a zero probe signal."""
    def __init__(self,base_embedder,width=256):
        super().__init__();self.base_embedder=base_embedder;self.active=False
        self.probe_features=nn.Sequential(nn.Linear(4,width),nn.GELU(),nn.LayerNorm(width))
        self.cross_attention=nn.MultiheadAttention(width,4,dropout=0,batch_first=True)
        self.output=nn.Linear(width,width);self.current_probe=None

    def forward(self,t,node_mask,onehot):
        query=self.base_embedder(t,node_mask,onehot)
        if self.current_probe is None:raise RuntimeError('Missing declared synthetic observations')
        context=self.probe_features(self.current_probe)
        observed=self.current_probe[...,2]>0
        attended,_=self.cross_attention(query,context,context,key_padding_mask=~observed,need_weights=False)
        return query+float(self.active)*self.output(attended)*node_mask[...,None]


def member_seed(base,case_index,repetition,update,rank,member):
    # K is absent: K8 and K32 begin with the same logical noise-member schedule.
    return (base+case_index*1000003+repetition*100003+update*1009+rank*37+member)%(2**31-1)


def seed_member(seed,device):
    torch.random.default_generator.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed(seed)
    np.random.seed(seed);random.seed(seed)


def bulk_loss_and_derivative(mean_profile,target,weights):
    # A finite-K ensemble-mean MSE, not a population-mean unbiased estimator.
    denominator=weights.sum().clamp_min(1e-12)
    residual=mean_profile-target
    loss=(weights*residual.square()).sum()/denominator
    derivative=2*weights*residual/denominator
    return loss,derivative.detach()


def cpu_gradient_test():
    """Compare full retained-graph and two-pass gradients using frozen heads."""
    torch.set_num_threads(2);torch.manual_seed(20261012)
    dtype=torch.float64;valid=torch.ones(1,8,dtype=dtype);chem=torch.tensor([1])
    native=torch.randn(1,8,3,dtype=dtype);target=torch.rand(1,8,dtype=dtype)
    weights=torch.tensor([[1,1,0,1,1,0,1,1]],dtype=dtype)
    observations=torch.rand(1,8,4,dtype=dtype)
    class TinyGeometry(nn.Module):
        def __init__(self):
            super().__init__();self.geometry=nn.Linear(3,3);self.probe=nn.Linear(4,3)
            self.head=FrozenGeometryChemistryHead(width=16);self.active=False
        def forward(self):
            noise=torch.randn_like(native)
            coordinates=self.geometry(native+noise)+float(self.active)*self.probe(observations)
            coordinates=F.dropout(coordinates,p=.1,training=True)
            profile=self.head(coordinates,chem,valid)
            return profile,(coordinates-native).square().mean()
    model=TinyGeometry().to(dtype=dtype)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    rows=[]
    for active in (False,True):
        model.active=active
        for k in (1,8,32):
            model.load_state_dict(initial);model.zero_grad(set_to_none=True)
            profiles=[];geometric=[]
            for member in range(k):
                torch.manual_seed(991+member);profile,geometry=model();profiles.append(profile);geometric.append(geometry)
            mean=torch.stack(profiles).mean(0)
            bulk,_=bulk_loss_and_derivative(mean,target,weights)
            total=torch.stack(geometric).mean()+bulk;total.backward()
            retained={name:p.grad.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
            model.load_state_dict(initial);model.zero_grad(set_to_none=True)
            with torch.no_grad():
                first=[]
                for member in range(k):
                    torch.manual_seed(991+member);first.append(model()[0])
                streamed_bulk,alpha=bulk_loss_and_derivative(torch.stack(first).mean(0),target,weights)
            max_replay_delta=0.
            for member in range(k):
                torch.manual_seed(991+member);profile,geometry=model()
                max_replay_delta=max(max_replay_delta,float((profile.detach()-first[member]).abs().max()))
                (geometry/k+(alpha*profile).sum()/k).backward()
            delta=max(float((p.grad-retained[name]).abs().max()) for name,p in model.named_parameters() if p.requires_grad)
            assert delta<1e-9 and max_replay_delta==0
            assert all(p.grad is None for p in model.head.parameters())
            assert float(model.geometry.weight.grad.norm())>0
            if active:assert float(model.probe.weight.grad.norm())>0
            else:assert float(model.probe.weight.grad.norm())==0
            # Bulk-only replay proves the frozen geometry head's derivative
            # reaches inverse-model coordinates and active probe parameters.
            model.load_state_dict(initial);model.zero_grad(set_to_none=True)
            for member in range(k):
                torch.manual_seed(991+member);profile,_=model()
                ((alpha*profile).sum()/k).backward()
            assert float(model.geometry.weight.grad.norm())>0
            if active:assert float(model.probe.weight.grad.norm())>0
            rows.append({'active':active,'K':k,'maximum_gradient_absolute_difference':delta,
                         'maximum_replayed_profile_absolute_difference':max_replay_delta,
                         'finite_K_bulk_loss':float(streamed_bulk),
                         'frozen_head_has_no_parameter_gradients':True,
                         'bulk_only_geometry_gradient_norm':float(model.geometry.weight.grad.norm()),
                         'bulk_only_probe_gradient_norm':float(model.probe.weight.grad.norm())})
    return {'status':'Actually executed CPU tiny retained-graph vs two-pass autograd test passed',
            'torch_version':torch.__version__,'dtype':'float64','tests':rows,
            'scope':'Exact loss-gradient algebra with matched RNG in a tiny geometry toy; full RNAbpFlow GPU execution is a separate qualification',
            'head_contract':'Coordinates+chemistry+native residue validity only; no observed values/errors/probe mask',
            'finite_K_scope':'K1/8/32 mean-profile MSE contains Monte Carlo variance/K; no unbiased population-mean claim'}


if __name__=='__main__':print(json.dumps(cpu_gradient_test(),indent=2,allow_nan=False))
