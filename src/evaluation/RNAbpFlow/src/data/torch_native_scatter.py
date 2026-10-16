"""Narrow torch_scatter replacement for RNAbpFlow's two grouped reductions.

Only the dim=0, one-dimensional group-index sum/mean operations used by the
published Kabsch utility are supported. Uses the existing CUDA Torch kernels.
It is a portability change, not a change to model architecture or training loss.
Unsupported calls fail rather than silently changing semantics.
"""

import torch


def scatter_add(src, index, dim=0, out=None, dim_size=None):
    if dim != 0 or index.ndim != 1 or src.shape[0] != index.numel() or out is not None:
        raise NotImplementedError('Port supports only dim=0, vector index, out=None')
    if index.numel() and bool((index < 0).any()):
        raise ValueError('Group indices must be nonnegative')
    size = int(index.max().item()) + 1 if index.numel() else 0
    size = size if dim_size is None else int(dim_size)
    result = src.new_zeros((size,) + tuple(src.shape[1:]))
    return result.index_add(0, index.to(dtype=torch.long), src)


def scatter(src, index, dim=0, out=None, dim_size=None, reduce='sum'):
    if reduce in ('sum', 'add'):
        return scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)
    if reduce != 'mean':
        raise NotImplementedError('Port supports only sum/add/mean')
    summed = scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)
    counts = scatter_add(src.new_ones((src.shape[0],)), index, dim_size=summed.shape[0])
    return summed / counts.clamp_min(1).reshape((-1,) + (1,) * (src.ndim - 1))


if __name__ == '__main__':
    import json
    torch.manual_seed(20261012)
    tests=[]
    for shape in [(11,3),(11,3,3)]:
        for dtype in (torch.float32,torch.float64):
            values=torch.randn(shape,dtype=dtype,requires_grad=True)
            groups=torch.tensor([0,0,0,2,2,2,2,3,3,3,3])
            for reduction in ('sum','mean'):
                expected=torch.stack([values[groups==i].sum(0) if reduction=='sum'
                                      else values[groups==i].mean(0) if bool((groups==i).any())
                                      else torch.zeros(shape[1:],dtype=dtype)
                                      for i in range(4)])
                actual=scatter(values,groups,reduce=reduction)
                delta=float((actual-expected).abs().max())
                assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-6)
                actual.sum().backward(retain_graph=True)
                assert values.grad.isfinite().all()
                tests.append({'shape':shape,'dtype':str(dtype),'reduction':reduction,'max_abs_difference':delta})
    print(json.dumps({'portability_adapter_CPU_validation':'passed','tests':tests},indent=2))
