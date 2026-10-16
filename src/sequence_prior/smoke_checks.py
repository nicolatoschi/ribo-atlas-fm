#!/usr/bin/env python3
"""CPU checks of waveE_mlm.py building blocks with the real RNA-FM weights (no GPU, no DDP).

1. Masks and step composition do not depend on the number of ranks (partition coverage).
2. Every sequence of a step is used exactly once; micro-batches respect the token budgets.
3. Official attention vs SDPA (need_weights=False): same logits on a padded batch.
4. One fwd/bwd micro-batch in fp32 and in bf16 autocast: loss agreement and CPU timing.
5. Held-out style evaluation of a few sequences (all-<mask>), fp32.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_trainer(path):
    spec = importlib.util.spec_from_file_location('waveE_mlm', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trainer', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    W = load_trainer(args.trainer)
    cfg = json.loads(Path(args.config).read_text())
    res = {}
    t = time.time()
    data = W.Data(Path(args.dataset))
    res['data_load_s'] = round(time.time() - t, 3)
    res['split_sizes'] = {'train': int(len(data.train)), **{k: int(len(v)) for k, v in data.eval_sets.items()}}
    G, seed = cfg['global_batch_sequences'], cfg['seed']
    base = W.stream_base(seed, 2)

    # 1-2: partition invariance and coverage for one production-size step
    step = 7
    pos = np.arange((step - 1) * G, step * G, dtype=np.int64)
    rows = data.stream(pos, seed)
    L = data.lengths[rows]
    ref = None
    budgets = cfg['micro_batching']
    coverage = {}
    t = time.time()
    for world in [32, 64, 128, 256]:
        order, owner = W.snake_partition(L, pos, world)
        masks = {}
        n_masked = 0
        load = np.zeros(world)
        micro_max = 0
        for r in range(world):
            mine = order[owner == r]
            load[r] = (L[mine] + 2).sum()
            chunks = W.micro_batches(L[mine], budgets['max_tokens_per_micro'], budgets['max_quadratic_per_micro'])
            micro_max = max(micro_max, len(chunks))
            seen = []
            for c in chunks:
                it = mine[c]
                T = int(L[it].max()) + 2
                assert len(it) * T <= budgets['max_tokens_per_micro'] or len(it) == 1
                assert len(it) * T * T <= budgets['max_quadratic_per_micro'] or len(it) == 1
                ms, mp, mr = W.make_masks(L[it], pos[it].astype(np.uint64), base, train=True)
                for s_local, q, rep in zip(ms, mp, mr):
                    masks.setdefault(int(pos[it][s_local]), []).append((int(q), int(rep)))
                n_masked += len(ms)
                seen.extend(it.tolist())
        assert sorted(masks) == sorted(pos.tolist())
        assert n_masked == int(W.mask_count(L).sum())
        canon = {k: sorted(v) for k, v in masks.items()}
        if ref is None:
            ref = canon
        assert canon == ref, f'masks differ at world {world}'
        coverage[world] = {'sequences_per_rank_mean': G / world, 'token_load_max_over_mean': float(load.max() / load.mean()),
                           'micro_batches_max_rank': micro_max}
    flat = [rep for v in ref.values() for _, rep in v]
    res['partition_invariance'] = {'step': step, 'global_batch': G, 'nucleotides': int(L.sum()), 'tokens_incl_cls_eos': int(L.sum() + 2 * G),
                                   'masked_positions': int(W.mask_count(L).sum()), 'identical_masks_at_world_sizes': [32, 64, 128, 256],
                                   'replacement_fractions': {'mask': float(np.mean(np.array(flat) == W.MASK)), 'random': float(np.mean((np.array(flat) >= 4) & (np.array(flat) <= 7))),
                                                             'keep': float(np.mean(np.array(flat) == -1))},
                                   'per_world': coverage, 'seconds': round(time.time() - t, 3)}

    # 3-5: model checks
    model, info = W.load_rnafm(args.weights, 'official')
    res['model'] = {k: info[k] for k in ['parameters_total', 'parameters_trainable', 'checkpoint_embed_equals_lm_head', 'loaded_embed_equals_checkpoint_lm_head', 'mask_row_norm', 'initial_state_digest_sha256', 'layer_norm_class', 'load_seconds']}
    model.eval()
    rng = np.random.default_rng(0)
    pick = rng.choice(data.train[data.lengths[data.train] <= 90], 4, replace=False)
    ms, mp, mr = W.make_masks(data.lengths[pick], pick.astype(np.uint64), base, train=True)
    tok, grid, targets, _ = W.build_tensors(data, pick, ms, mp, mr)
    tok_t, grid_t, tgt = torch.from_numpy(tok), torch.from_numpy(grid), torch.from_numpy(targets)
    with torch.no_grad():
        out_official = model(tok_t)['logits']
    import fm.modules as M
    original_forward = M.TransformerLayer.forward
    W.patch_sdpa()
    with torch.no_grad():
        out_sdpa = model(tok_t)['logits']
    M.TransformerLayer.forward = original_forward
    nonpad = tok_t.ne(W.PAD)
    diff = (out_official - out_sdpa).abs()[nonpad]
    res['sdpa_vs_official'] = {'batch_shape': list(tok.shape), 'max_abs_logit_diff_nonpad': float(diff.max()),
                               'argmax_agreement': float((out_official.argmax(-1) == out_sdpa.argmax(-1))[nonpad].float().mean())}
    model.train()
    timing = {}
    for name, ctx in [('fp32', None), ('bf16_autocast', torch.bfloat16)]:
        model.zero_grad(set_to_none=True)
        t = time.time()
        if ctx is None:
            logits = model(tok_t, masked_tokens=grid_t)['logits']
        else:
            with torch.autocast('cpu', dtype=ctx):
                logits = model(tok_t, masked_tokens=grid_t)['logits']
        loss = F.cross_entropy(logits.float(), tgt)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1e9)
        timing[name] = {'loss': float(loss), 'grad_norm': float(gn), 'fwd_bwd_seconds': round(time.time() - t, 3),
                        'tokens': int(nonpad.sum()), 'masked': int(len(targets))}
    res['micro_batch_fwd_bwd'] = timing
    res['micro_batch_fwd_bwd']['bf16_minus_fp32_loss'] = timing['bf16_autocast']['loss'] - timing['fp32']['loss']
    model.eval()
    held = data.eval_sets['heldout_family'][:16]
    ms, mp, mr = W.make_masks(data.lengths[held], held.astype(np.uint64) * np.uint64(16), W.stream_base(seed, 3), train=False)
    tok, grid, targets, _ = W.build_tensors(data, held, ms, mp, mr)
    t = time.time()
    with torch.no_grad():
        logits = model(torch.from_numpy(tok), masked_tokens=torch.from_numpy(grid))['logits']
    ce = F.cross_entropy(logits, torch.from_numpy(targets))
    res['heldout_eval_16_sequences'] = {'ce_per_token': float(ce), 'perplexity': float(torch.exp(ce)), 'masked': int(len(targets)),
                                        'all_inputs_are_mask_at_targets': bool((tok[grid] == W.MASK).all()),
                                        'seconds': round(time.time() - t, 3)}
    res['torch'] = torch.__version__
    res['threads'] = torch.get_num_threads()
    Path(args.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    sys.exit(main())
