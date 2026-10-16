#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native-primal quaternion conversion with an isolated-eigenvector derivative.

The native quaternion K matrix has eigenvalues (1,-1/3,-1/3,-1/3) for SO3.
Only its isolated largest eigenvector is used. Differentiating every eigenvector
introduces unnecessary zero spectral-gap divisions among the other three.
This port returns exactly the native eigensolver's largest vector and uses its
analytic derivative against only the three strictly separated lower values.
It neither perturbs rotations nor clips/replaces nonfinite gradients.
"""
import torch


class _IsolatedLargestEigenvector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix):
        values,vectors=torch.linalg.eigh(matrix)
        gaps=values[..., -1,None]-values[..., :-1]
        threshold=64*torch.finfo(matrix.dtype).eps
        if not bool(torch.isfinite(gaps).all()) or not bool((gaps>threshold).all()):
            raise RuntimeError('Quaternion largest eigenvector is not isolated')
        ctx.save_for_backward(values,vectors)
        return vectors[..., -1]

    @staticmethod
    def backward(ctx, gradient):
        values,vectors=ctx.saved_tensors
        selected=vectors[..., -1]
        others=vectors[..., :-1]
        gaps=values[..., -1,None]-values[..., :-1]
        coefficients=torch.einsum('...ji,...j->...i',others,gradient)/gaps
        tangent=torch.einsum('...ji,...i->...j',others,coefficients)
        # Symmetric matrix perturbations: the two outer products each receive
        # half. No denominator compares the repeated unselected eigenvalues.
        result=(tangent[..., :,None]*selected[..., None,:]+
                selected[..., :,None]*tangent[..., None,:])/2
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError('Nonfinite isolated-quaternion derivative')
        return result


def differentiable_rot_to_quat(rot):
    if rot.shape[-2:]!=(3,3):raise ValueError('Input rotation is incorrectly shaped')
    # Exact native RNAbpFlow K construction/order and eigensolver convention.
    [[xx,xy,xz],[yx,yy,yz],[zx,zy,zz]]=[[rot[...,i,j] for j in range(3)] for i in range(3)]
    rows=[[xx+yy+zz,zy-yz,xz-zx,yx-xy],
          [zy-yz,xx-yy-zz,xy+yx,xz+zx],
          [xz-zx,xy+yx,yy-xx-zz,yz+zy],
          [yx-xy,xz+zx,yz+zy,zz-xx-yy]]
    matrix=(1./3.)*torch.stack([torch.stack(row,dim=-1) for row in rows],dim=-2)
    return _IsolatedLargestEigenvector.apply(matrix)


def install_native_quaternion_derivative(rigid_utils):
    """Replace this one converter; native matrices/frames/solver stay unchanged."""
    rigid_utils.rot_to_quat=differentiable_rot_to_quat


def cpu_rotation_test(rigid_utils,so3):
    torch.set_num_threads(2);torch.manual_seed(20261012)
    reference=rigid_utils.rot_to_quat
    rows=[]
    for dtype in (torch.float64,torch.float32):
        angles=[0.,1e-8,.4,3.141592653589793-1e-6,3.141592653589793]
        axis=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,2.,3.]],dtype=dtype)
        axis=axis/axis.norm(dim=-1,keepdim=True)
        rv=torch.cat([axis*a for a in angles]+[torch.randn(1000,3,dtype=dtype)],dim=0)
        rotation=so3.rotvec_to_rotmat(rv).detach().requires_grad_(True)
        q=differentiable_rot_to_quat(rotation);old=reference(rotation.detach())
        # Native eigenvector/sign outputs are identical, not merely equal
        # orientations after a quaternion-sign convention change.
        primal_delta=float((q.detach()-old).abs().max())
        assert primal_delta==0.
        reconstructed=rigid_utils.quat_to_rot(q)
        original=rigid_utils.quat_to_rot(old)
        rotation_delta=float((reconstructed.detach()-original).abs().max())
        assert rotation_delta==0.
        loss=(reconstructed*torch.randn_like(reconstructed)).sum()
        loss.backward()
        assert bool(torch.isfinite(rotation.grad).all())
        rows.append({'dtype':str(dtype),'rotations':int(rotation.shape[0]),
                     'identity_near_identity_pi_cases':20,'maximum_native_quaternion_difference':primal_delta,
                     'maximum_native_reconstructed_rotation_difference':rotation_delta,
                     'matrix_gradient_norm':float(rotation.grad.norm()),'all_gradients_finite':True})
    # Gradcheck the phase-invariant rotation output, including exact identity
    # and pi. This validates derivatives rather than arbitrary eigenvector sign.
    rv=torch.tensor([[0.,0.,0.],[3.141592653589793,0.,0.],[.2,.3,.4]],dtype=torch.float64)
    matrices=so3.rotvec_to_rotmat(rv).detach().requires_grad_(True)
    gradcheck=torch.autograd.gradcheck(lambda x:rigid_utils.quat_to_rot(differentiable_rot_to_quat(x)),
                                      (matrices,),eps=1e-6,atol=1e-5,rtol=1e-4)
    assert gradcheck
    return {'status':'Actually executed native-primal isolated-quaternion CPU derivative tests passed',
            'torch_version':torch.__version__,'tests':rows,'phase_invariant_rotation_gradcheck':gradcheck,
            'scope':'Exact native eigensolver primal; analytic largest-eigenvector derivative only, no eigenvalue perturbation/gradient replacement',
            'reference':'https://docs.pytorch.org/docs/2.10/generated/torch.linalg.eigh.html'}
