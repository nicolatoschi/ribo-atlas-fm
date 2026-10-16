#!/usr/bin/env python3
"""Case-mean reduction and stable logical-case priors for batching experiments."""
import hashlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from ensemble_mean_core import member_seed,seed_member


def case_mean_loss_and_derivative(mean,target,weights):
    denominator=weights.sum(-1,keepdim=True).clamp_min(1)
    difference=mean-target
    losses=(weights*difference.square()).sum(-1)/denominator.squeeze(-1)
    return losses.mean(),2*weights*difference/denominator/mean.shape[0]


def logical_priors(seed,repetition,identity,ids,member,length,device):
    translations=[];rotations=[]
    for logical in ids:
        seed_member(member_seed(seed,0,repetition,identity,int(logical),member),device)
        tx=torch.randn(1,length,3,device=device)
        translations.append((tx-tx.mean(-2,keepdim=True))*10)
        rotations.append(torch.tensor(Rotation.random(length).as_matrix(),device=device,dtype=torch.float32)[None])
    return torch.cat(translations),torch.cat(rotations)


def repeat_native(singleton,count):
    return {key:value.expand(count,*value.shape[1:]).clone() for key,value in singleton.items()}


def digest_tensor(value):
    data=value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(data.shape).encode()+str(data.dtype).encode()+data.tobytes()).hexdigest()


def observations(native_c4,mask,ids,seed):
    radius=torch.sqrt((native_c4-native_c4.mean(-2,keepdim=True)).square().sum(-1)+1e-8)
    target=[];observed=[];chemistry=[]
    for logical in ids:
        random=np.random.default_rng(seed+int(logical)*97)
        chemistry.append(int(logical)%2)
        noise=torch.tensor(random.normal(0,.02,radius.shape),device=radius.device,dtype=torch.float32)
        target.append((torch.sigmoid(radius/15-1)+(int(logical)%2)*.1+noise).clamp_min(0))
        seen=torch.tensor(random.random(radius.shape)>.2,device=radius.device)
        seen[:,0]=True;observed.append(seen*mask.bool())
    target=torch.cat(target);seen=torch.cat(observed);error=torch.full_like(target,.05)
    chemistry=torch.tensor(chemistry,device=target.device)
    probe=torch.stack([target*seen,error*seen,seen.float(),chemistry[:,None].expand_as(target).float()],-1)
    return target,seen.float()/error.square(),probe,chemistry
