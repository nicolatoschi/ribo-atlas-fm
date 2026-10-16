#!/usr/bin/env python3
"""Wave E trainer: masked-LM continued pretraining of the official RNA-FM on Rfam 15.1 seeds.

Model: rna-fm 0.2.2 (`fm` package, unmodified) built by fm.pretrained.load_model_and_alphabet_core
from RNA-FM_pretrained.pth, loaded with torch.load(weights_only=True, mmap=True); the only
allow-listed global is argparse.Namespace (the fairseq args object).

Modes
  scaling  untimed warm-up steps + timed steps, no evaluation (phase-1 strong scaling)
  train    held-out evaluation at step 0, timing block, continued training to total_steps with
           evaluation at epoch boundaries and at the end, final model saved (phase 2)

World-size invariance: the sequences of step s, their masks and the loss normaliser
(total masked positions of the global batch) depend only on (seed, s), never on the number
of ranks. Each rank backpropagates sum-CE * W / N_global, and DDP averages over W, so every
world size computes the same gradient up to floating-point order.

Launch: one process per GPU via step_launch.py (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR,
MASTER_PORT), or a single process without any of these (CPU smoke tests).
"""
import time

T_START = time.time()
import argparse
import contextlib
import datetime
import gc
import hashlib
import json
import math
import os
import platform
import resource
import socket
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import fm

T_IMPORTED = time.time()
NUC = {'A': 4, 'C': 5, 'G': 6, 'U': 7}
CLS, PAD, EOS, MASK = 0, 1, 2, 24
MAX_NT = 1022
U64 = np.uint64


def log(msg, rank=0):
    if rank == 0:
        print(f'[{datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- counter-based hashing
def mix64(x):
    with np.errstate(over='ignore'):
        x = x + U64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> U64(27))) * U64(0x94D049BB133111EB)
        return x ^ (x >> U64(31))


def stream_base(seed, tag):
    return mix64(np.array([seed * 1000003 + tag], dtype=np.uint64))[0]


def mask_count(lengths):
    return np.maximum(1, np.floor(0.15 * lengths + 0.5).astype(np.int64))


def make_masks(lengths, keys, base, train):
    """Return per-sequence lists of (positions, input replacement tokens) for nucleotide positions.

    lengths: int64 array of sequence lengths; keys: uint64 array of per-sequence keys (stream
    position or eval item id). Positions are 0-based nucleotide indices.
    """
    n = len(lengths)
    total = int(lengths.sum())
    seq_idx = np.repeat(np.arange(n), lengths)
    starts = np.zeros(n, dtype=np.int64)
    starts[1:] = np.cumsum(lengths)[:-1]
    pos = np.arange(total, dtype=np.int64) - np.repeat(starts, lengths)
    with np.errstate(over='ignore'):
        h = mix64(mix64(U64(base) ^ mix64(keys.astype(np.uint64)))[seq_idx] + pos.astype(np.uint64))
    order = np.lexsort((h, seq_idx))
    rank_in_seq = np.empty(total, dtype=np.int64)
    rank_in_seq[order] = np.arange(total) - np.repeat(starts, lengths)
    chosen = rank_in_seq < mask_count(lengths)[seq_idx]
    sel = np.nonzero(chosen)[0]
    h2 = mix64(h[sel] ^ U64(0xD1B54A32D192ED03))
    if train:
        u = (h2 >> U64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
        rnd = (4 + (mix64(h2) % U64(4))).astype(np.int64)
        repl = np.where(u < 0.8, MASK, np.where(u < 0.9, rnd, -1))
    else:
        repl = np.full(len(sel), MASK, dtype=np.int64)
    return seq_idx[sel], pos[sel], repl


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, path, limit_train=None, limit_eval=None):
        rows = {'train': [], 'val_in_family': [], 'heldout_family': []}
        self.meta = []
        self.tokens = []
        lut = np.full(256, -1, dtype=np.int8)
        for k, v in NUC.items():
            lut[ord(k)] = v
        with open(path) as f:
            header = f.readline().rstrip('\n').split('\t')
            assert header == ['seq_sha256', 'split', 'length', 'group_id', 'primary_family', 'families', 'clans', 'sequence'], header
            for i, line in enumerate(f):
                sha, split, length, group, primary, fams, clans, seq = line.rstrip('\n').split('\t')
                arr = lut[np.frombuffer(seq.encode(), dtype=np.uint8)]
                assert len(arr) == int(length) and (arr >= 0).all()
                if len(arr) > MAX_NT:
                    raise ValueError(f'{sha}: {len(arr)} nt exceeds RNA-FM limit {MAX_NT}; no crop policy is enabled')
                self.tokens.append(arr)
                self.meta.append((sha, split, group, primary))
                rows[split].append(i)
        self.lengths = np.array([len(t) for t in self.tokens], dtype=np.int64)
        self.train = np.array(rows['train'][:limit_train] if limit_train else rows['train'], dtype=np.int64)
        self.eval_sets = {k: np.array(rows[k][:limit_eval] if limit_eval else rows[k], dtype=np.int64)
                          for k in ['heldout_family', 'val_in_family']}
        self._perm = {}

    def stream(self, positions, seed):
        n = len(self.train)
        out = np.empty(len(positions), dtype=np.int64)
        epochs = positions // n
        for e in np.unique(epochs):
            if e not in self._perm:
                self._perm[e] = np.random.default_rng([seed, 1, int(e)]).permutation(n)
            sel = epochs == e
            out[sel] = self.train[self._perm[e][positions[sel] % n]]
        return out


def snake_partition(lengths, keys, world):
    order = np.lexsort((keys, -lengths))
    i = np.arange(len(order))
    row, col = i // world, i % world
    owner = np.where(row % 2 == 0, col, world - 1 - col)
    return order, owner


def micro_batches(lengths_sorted_desc, max_tokens, max_quad):
    chunks, cur, tmax = [], [], 0
    for j, L in enumerate(lengths_sorted_desc):
        T = int(L) + 2
        t = max(tmax, T)
        if cur and ((len(cur) + 1) * t > max_tokens or (len(cur) + 1) * t * t > max_quad):
            chunks.append(cur)
            cur, t = [], T
        cur.append(j)
        tmax = t
    if cur:
        chunks.append(cur)
    return chunks


def build_tensors(data, idx, mask_seq, mask_pos, mask_repl):
    """idx: dataset rows of one micro-batch (any order). mask_* index into idx positions."""
    lengths = data.lengths[idx]
    T = int(lengths.max()) + 2
    tok = np.full((len(idx), T), PAD, dtype=np.int64)
    for r, d in enumerate(idx):
        tok[r, 0] = CLS
        tok[r, 1:1 + lengths[r]] = data.tokens[d]
        tok[r, 1 + lengths[r]] = EOS
    col = mask_pos + 1
    targets_grid = np.zeros_like(tok, dtype=bool)
    targets_grid[mask_seq, col] = True
    # targets in row-major order of the boolean mask (matches features[masked_tokens])
    flat = np.ravel_multi_index((mask_seq, col), tok.shape)
    order = np.argsort(flat, kind='stable')
    targets = tok[mask_seq[order], col[order]].copy()
    rep = mask_repl
    keep = rep >= 0
    tok[mask_seq[keep], col[keep]] = rep[keep]
    return tok, targets_grid, targets, mask_seq[order]


# ---------------------------------------------------------------- model
def load_rnafm(weights, attention):
    import argparse as ap
    info = {}
    if hasattr(torch.serialization, 'get_default_mmap_options'):
        import mmap
        assert torch.serialization.get_default_mmap_options() == mmap.MAP_PRIVATE, 'mmap must be copy-on-write'
    t = time.time()
    with torch.serialization.safe_globals([ap.Namespace]):
        data = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    ck_embed = data['model']['encoder.encoder.embed_tokens.weight'].clone()
    ck_lmhead = data['model']['encoder.encoder.lm_head.weight'].clone()
    info['checkpoint_optimizer_history_num_updates'] = sorted({h.get('num_updates') for h in data['optimizer_history']})
    info['checkpoint_embed_equals_lm_head'] = bool(torch.equal(ck_embed, ck_lmhead))
    model, alphabet = fm.pretrained.load_model_and_alphabet_core('RNA-FM_pretrained', data, None, theme='rna')
    del data
    gc.collect()
    assert alphabet.mask_idx == MASK and alphabet.padding_idx == PAD and alphabet.cls_idx == CLS and alphabet.eos_idx == EOS
    assert [alphabet.get_idx(c) for c in 'ACGU'] == [4, 5, 6, 7]
    assert model.lm_head.weight is model.embed_tokens.weight
    info['loaded_embed_equals_checkpoint_lm_head'] = bool(torch.equal(model.embed_tokens.weight.detach(), ck_lmhead))
    info['mask_row_norm'] = float(model.embed_tokens.weight.detach()[MASK].norm())
    info['mask_row_note'] = ('fm loader zeroes embed_tokens[<mask>] in the state dict, but lm_head.weight (tied, '
                             'loaded afterwards) restores the trained row; the model equals the checkpoint')
    for name, p in model.named_parameters():
        if name.startswith('contact_head.'):
            p.requires_grad_(False)
    info['parameters_total'] = sum(p.numel() for p in model.parameters())
    info['parameters_trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info['layer_norm_class'] = f'{type(model.layers[0].self_attn_layer_norm).__module__}.{type(model.layers[0].self_attn_layer_norm).__name__}'
    info['args'] = {k: v for k, v in vars(model.args).items() if k in ('arch', 'layers', 'embed_dim', 'ffn_embed_dim', 'attention_heads', 'max_positions', 'emb_layer_norm_before')}
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().contiguous().float().numpy().tobytes())
    info['initial_state_digest_sha256'] = h.hexdigest()
    info['load_seconds'] = round(time.time() - t, 3)
    if attention == 'sdpa':
        patch_sdpa()
    return model, info


def patch_sdpa():
    import fm.modules as M
    gelu = M.gelu

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask,
                                 need_weights=need_head_weights, need_head_weights=need_head_weights,
                                 attn_mask=self_attn_mask)
        x = residual + x
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x, attn

    M.TransformerLayer.forward = forward


# ---------------------------------------------------------------- runtime helpers
def host_rss_gib():
    try:
        with open('/proc/self/status') as f:
            vals = {line.split(':')[0]: int(line.split()[1]) for line in f if line.startswith(('VmRSS', 'VmHWM'))}
        return vals['VmRSS'] / 2**20, vals['VmHWM'] / 2**20
    except OSError:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = peak / 2**30 if sys.platform == 'darwin' else peak / 2**20
        return peak, peak


def lr_at(step, cfg):
    o = cfg['optimizer']
    peak, warm, total = o['lr_peak'], o['lr_warmup_steps'], cfg['total_steps']
    if step <= warm:
        return peak * step / warm
    progress = (step - warm) / max(1, total - warm)
    final = o['lr_final_fraction'] * peak
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def summarise(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return None
    return {'n': int(v.size), 'median': float(np.median(v)), 'mean': float(v.mean()), 'min': float(v.min()),
            'max': float(v.max()), 'p10': float(np.percentile(v, 10)), 'p90': float(np.percentile(v, 90))}


class Runner:
    def __init__(self, args, cfg):
        self.args, self.cfg = args, cfg
        self.timings = {'import_s': round(T_IMPORTED - T_START, 3)}
        env = os.environ
        self.distributed = 'RANK' in env and 'WORLD_SIZE' in env
        self.rank = int(env.get('RANK', 0))
        self.world = int(env.get('WORLD_SIZE', 1))
        self.local_rank = int(env.get('LOCAL_RANK', 0))
        if args.device == 'cuda':
            n_vis = torch.cuda.device_count()
            if n_vis == 1:
                dev = 0
            elif self.local_rank < n_vis:
                dev = self.local_rank
            else:
                raise RuntimeError(f'LOCAL_RANK {self.local_rank} but {n_vis} visible GPUs')
            torch.cuda.set_device(dev)
            self.device = torch.device('cuda', dev)
        else:
            self.device = torch.device('cpu')
        t = time.time()
        if self.distributed:
            backend = 'nccl' if args.device == 'cuda' else 'gloo'
            kw = {'device_id': self.device} if args.device == 'cuda' else {}
            dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=10), **kw)
            assert dist.get_world_size() == self.world and dist.get_rank() == self.rank
        self.timings['process_group_init_s'] = round(time.time() - t, 3)

    # collective helpers (no-ops when single process)
    def allreduce(self, values, op):
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.distributed:
            dist.all_reduce(t, op=op)
        return t.cpu().numpy()

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def placement(self):
        rec = {'host': socket.gethostname(), 'rank': self.rank, 'local_rank': self.local_rank, 'pid': os.getpid()}
        if hasattr(os, 'sched_getaffinity'):
            cpus = sorted(os.sched_getaffinity(0))
            rec['cpus'] = f'{cpus[0]}-{cpus[-1]} ({len(cpus)})' if cpus else ''
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            rec['gpu'] = props.name
            rec['gpu_uuid'] = str(getattr(props, 'uuid', f'idx{self.device.index}'))
            rec['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        out = [None] * self.world
        if self.distributed:
            dist.all_gather_object(out, rec)
        else:
            out = [rec]
        if self.device.type == 'cuda':
            seen = {}
            for r in out:
                key = (r['host'], r['gpu_uuid'])
                if key in seen:
                    raise RuntimeError(f'ranks {seen[key]} and {r["rank"]} share GPU {key}: binding error')
                seen[key] = r['rank']
        return out

    def run(self):
        args, cfg = self.args, self.cfg
        out = Path(args.out)
        if self.rank == 0:
            out.mkdir(parents=True, exist_ok=True)
        placement = self.placement()
        log(f'world={self.world} device={self.device.type} hosts={len({p["host"] for p in placement})}', self.rank)

        t = time.time()
        data_path = Path(args.dataset)
        if args.verify_dataset:
            got = sha_file(data_path)
            assert got == cfg['dataset_sha256'], f'dataset sha256 {got}'
        data = Data(data_path, args.limit_train, args.limit_eval)
        self.timings['data_load_s'] = round(time.time() - t, 3)

        st = os.stat(args.weights)
        assert args.allow_weights_size_mismatch or st.st_size == cfg['weights_bytes'], f'weights size {st.st_size}'
        model, model_info = load_rnafm(args.weights, args.attention)
        self.timings['model_load_s'] = model_info['load_seconds']
        t = time.time()
        model.to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        o = cfg['optimizer']
        opt_kw = {'fused': True} if (self.device.type == 'cuda' and o.get('fused_on_cuda')) else {}
        optim = torch.optim.AdamW(params, lr=lr_at(1, cfg), betas=tuple(o['betas']), eps=o['eps'],
                                  weight_decay=o['weight_decay'], **opt_kw)
        net = model
        if self.distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            net = DDP(model, device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                      gradient_as_bucket_view=True, find_unused_parameters=False)
        self.timings['to_device_and_ddp_s'] = round(time.time() - t, 3)

        G = args.global_batch or cfg['global_batch_sequences']
        assert G >= self.world, 'global batch must give every rank at least one sequence'
        total = args.total_steps or cfg['total_steps']
        warm = cfg['warmup_steps_untimed'] if args.warmup_untimed is None else args.warmup_untimed
        timed = cfg['timed_steps'] if args.timed_steps is None else args.timed_steps
        last = min(total, args.stop_after) if args.stop_after else total
        if args.mode == 'scaling':
            assert last == warm + timed, 'scaling runs stop exactly after the timing block'
        mb = cfg['micro_batching']
        max_tok = args.max_tokens_per_micro or mb['max_tokens_per_micro']
        max_quad = mb['max_quadratic_per_micro']
        seed = cfg['seed']
        base_train, base_eval = stream_base(seed, 2), stream_base(seed, 3)
        use_bf16 = cfg['precision'].startswith('bf16') and not args.fp32
        n_train = len(data.train)
        eval_steps = []
        if args.mode == 'train':
            k = 1
            while math.ceil(k * n_train / G) < total:
                eval_steps.append(math.ceil(k * n_train / G))
                k += 1
            eval_steps.append(total)
            eval_steps = sorted(set(s for s in eval_steps if s <= last))

        def autocast():
            if not use_bf16:
                return contextlib.nullcontext()
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

        def evaluate(step):
            t0 = time.time()
            model.eval()
            res = {}
            R = cfg['evaluation']['replicates']
            for split, rows in data.eval_sets.items():
                rows_rep = np.repeat(rows, R)
                reps = np.tile(np.arange(R), len(rows))
                items = np.arange(len(rows_rep))
                keys = rows_rep.astype(np.uint64) * U64(16) + reps.astype(np.uint64)
                L = data.lengths[rows_rep]
                order, owner = snake_partition(L, items, self.world)
                mine = order[owner == self.rank]
                stats = torch.zeros((len(items), 3), dtype=torch.float64, device=self.device)
                with torch.no_grad():
                    for chunk in micro_batches(L[mine], max_tok, max_quad):
                        it = mine[chunk]
                        ms, mp, mr = make_masks(L[it], keys[it], base_eval, train=False)
                        tok, grid, targets, row_of_target = build_tensors(data, rows_rep[it], ms, mp, mr)
                        tok_t = torch.from_numpy(tok).to(self.device)
                        grid_t = torch.from_numpy(grid).to(self.device)
                        logits = model(tok_t, masked_tokens=grid_t)['logits'].float()
                        tgt = torch.from_numpy(targets).to(self.device)
                        ce = F.cross_entropy(logits, tgt, reduction='none').double()
                        correct = (logits.argmax(-1) == tgt).double()
                        rows_t = torch.from_numpy(it[row_of_target]).to(self.device)
                        stats[:, 0].index_add_(0, rows_t, ce)
                        stats[:, 1].index_add_(0, rows_t, torch.ones_like(ce))
                        stats[:, 2].index_add_(0, rows_t, correct)
                if self.distributed:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                acc = stats.cpu().numpy()
                per_seq = acc.reshape(len(rows), R, 3).sum(axis=1)
                assert (per_seq[:, 1] == R * mask_count(data.lengths[rows])).all(), 'evaluation lost masked positions'
                ce_sum, n, corr = per_seq[:, 0].sum(), per_seq[:, 1].sum(), per_seq[:, 2].sum()
                res[split] = {'sequences': int(len(rows)), 'masked_positions': int(n), 'ce_per_token': float(ce_sum / n),
                              'perplexity': float(math.exp(ce_sum / n)), 'accuracy': float(corr / n)}
                if self.rank == 0:
                    with open(out / f'eval_step{step:04d}_{split}.tsv', 'w') as f:
                        f.write('seq_sha256\tgroup_id\tprimary_family\tlength\tmasked_positions\tce_sum\tcorrect\n')
                        for r, (cs, nn, cc) in zip(rows, per_seq):
                            sha, _, group, primary = data.meta[r]
                            f.write(f'{sha}\t{group}\t{primary}\t{data.lengths[r]}\t{int(nn)}\t{cs:.10g}\t{int(cc)}\n')
            model.train()
            res['seconds'] = round(time.time() - t0, 3)
            log(f'eval step {step}: ' + ', '.join(f'{k} ce {v["ce_per_token"]:.5f} ppl {v["perplexity"]:.4f}'
                                                  for k, v in res.items() if isinstance(v, dict)), self.rank)
            return res

        def train_step(step):
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            self.sync()
            t0 = time.perf_counter()
            pos = np.arange((step - 1) * G, step * G, dtype=np.int64)
            rows = data.stream(pos, seed)
            L = data.lengths[rows]
            n_global = int(mask_count(L).sum())
            order, owner = snake_partition(L, pos, self.world)
            mine = order[owner == self.rank]
            chunks = micro_batches(L[mine], max_tok, max_quad)
            prepared = []
            real_tokens = padded_tokens = 0
            for chunk in chunks:
                it = mine[chunk]
                ms, mp, mr = make_masks(L[it], pos[it].astype(np.uint64), base_train, train=True)
                tok, grid, targets, _ = build_tensors(data, rows[it], ms, mp, mr)
                real_tokens += int(L[it].sum()) + 2 * len(it)
                padded_tokens += tok.size
                prepared.append((tok, grid, targets))
            t_data = time.perf_counter() - t0
            lr = lr_at(step, cfg)
            for g in optim.param_groups:
                g['lr'] = lr
            loss_acc = torch.zeros((), dtype=torch.float64, device=self.device)
            corr_acc = torch.zeros((), dtype=torch.float64, device=self.device)
            n_local = 0
            scale = self.world / n_global
            for j, (tok, grid, targets) in enumerate(prepared):
                sync_ctx = net.no_sync() if (self.distributed and j < len(prepared) - 1) else contextlib.nullcontext()
                with sync_ctx:
                    tok_t = torch.from_numpy(tok).to(self.device, non_blocking=True)
                    grid_t = torch.from_numpy(grid).to(self.device, non_blocking=True)
                    tgt = torch.from_numpy(targets).to(self.device, non_blocking=True)
                    with autocast():
                        logits = net(tok_t, masked_tokens=grid_t)['logits']
                    loss_sum = F.cross_entropy(logits.float(), tgt, reduction='sum')
                    (loss_sum * scale).backward()
                    loss_acc += loss_sum.detach().double()
                    corr_acc += (logits.detach().argmax(-1) == tgt).sum().double()
                    n_local += len(targets)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, o['grad_clip_norm'])
            optim.step()
            optim.zero_grad(set_to_none=True)
            self.sync()
            t_step = time.perf_counter() - t0
            if self.device.type == 'cuda':
                alloc = torch.cuda.max_memory_allocated(self.device) / 2**30
                reserved = torch.cuda.max_memory_reserved(self.device) / 2**30
            else:
                alloc = reserved = 0.0
            rss, hwm = host_rss_gib()
            s = self.allreduce([float(loss_acc), float(corr_acc), n_local, real_tokens, padded_tokens, len(prepared), len(mine)], dist.ReduceOp.SUM)
            m = self.allreduce([t_step, -t_step, t_data, len(prepared), alloc, reserved, rss, hwm], dist.ReduceOp.MAX)
            assert int(s[2]) == n_global, f'masked positions {s[2]} != {n_global}'
            assert int(s[6]) == G
            return {'step': step, 'lr': lr, 'loss': s[0] / s[2], 'accuracy': s[1] / s[2], 'masked_positions': int(s[2]),
                    'real_tokens': int(s[3]), 'nucleotides': int(L.sum()), 'padded_tokens': int(s[4]), 'micro_batches_total': int(s[5]),
                    'micro_batches_max_rank': int(m[3]), 't_step_max': m[0], 't_step_min': -m[1], 't_step_rank0': t_step,
                    't_data_max': m[2], 'peak_alloc_gib_max': m[4], 'peak_reserved_gib_max': m[5], 'host_rss_gib_max': m[6],
                    'host_rss_peak_gib_max': m[7], 'grad_norm': float(grad_norm)}

        status = {'status': 'running'}
        evals = {}
        steps = []
        if args.mode == 'train':
            evals[0] = evaluate(0)
        step_file = open(out / 'steps.jsonl', 'w') if self.rank == 0 else None
        t_loop = time.time()
        post_eval = False
        for step in range(1, last + 1):
            rec = train_step(step)
            rec['phase'] = 'warmup' if step <= warm else ('timed' if step <= warm + timed else 'continue')
            rec['after_eval'] = post_eval
            post_eval = False
            steps.append(rec)
            if step_file:
                step_file.write(json.dumps(rec) + '\n')
                step_file.flush()
            if step == 1:
                self.timings['first_step_s'] = round(rec['t_step_max'], 3)
            if step % args.log_every == 0 or step in (1, last):
                log(f'step {step} {rec["phase"]} loss {rec["loss"]:.5f} t_max {rec["t_step_max"]:.4f}s '
                    f'tok/s {rec["real_tokens"] / rec["t_step_max"]:.0f} hbm {rec["peak_alloc_gib_max"]:.2f} GiB', self.rank)
            if step in eval_steps:
                evals[step] = evaluate(step)
                post_eval = True
        self.timings['step_loop_s'] = round(time.time() - t_loop, 3)
        if step_file:
            step_file.close()

        saved = None
        if args.mode == 'train' and cfg.get('save_final_model') and not args.no_save and self.rank == 0:
            path = out / 'final_model_state.pt'
            torch.save({'model': model.state_dict(), 'rnafm_args': vars(model.args), 'step': last,
                        'config_sha256': args.config_sha256}, path)
            saved = {'file': path.name, 'sha256': sha_file(path), 'bytes': path.stat().st_size}
        timed_steps = [r for r in steps if r['phase'] == 'timed']
        clean = [r for r in timed_steps if not r['after_eval']]
        summary = None
        if timed_steps:
            tt = [r['t_step_max'] for r in timed_steps]
            summary = {
                'timed_steps': len(timed_steps), 't_step_max': summarise(tt),
                't_step_max_excluding_after_eval': summarise([r['t_step_max'] for r in clean]),
                't_step_min': summarise([r['t_step_min'] for r in timed_steps]),
                't_data_max': summarise([r['t_data_max'] for r in timed_steps]),
                'tokens_per_s_from_totals': sum(r['real_tokens'] for r in timed_steps) / sum(tt),
                'nucleotides_per_s_from_totals': sum(r['nucleotides'] for r in timed_steps) / sum(tt),
                'tokens_per_s_at_median_step': float(np.median([r['real_tokens'] for r in timed_steps])) / float(np.median(tt)),
                'padding_fraction': 1 - sum(r['real_tokens'] for r in timed_steps) / sum(r['padded_tokens'] for r in timed_steps),
                'peak_alloc_gib_max': max(r['peak_alloc_gib_max'] for r in steps),
                'peak_reserved_gib_max': max(r['peak_reserved_gib_max'] for r in steps),
                'host_rss_peak_gib_max': max(r['host_rss_peak_gib_max'] for r in steps),
            }
        status = {'status': 'completed'}
        result = {
            **status, 'label': args.label, 'mode': args.mode, 'world_size': self.world,
            'nodes': len({p['host'] for p in placement}), 'attention': args.attention, 'bf16_autocast': use_bf16,
            'global_batch_sequences': G, 'steps_run': last, 'schedule_total_steps': total, 'warmup_untimed': warm,
            'timed_steps_configured': timed, 'eval_steps': eval_steps, 'n_train': n_train,
            'n_eval': {k: int(len(v)) for k, v in data.eval_sets.items()},
            'config_sha256': args.config_sha256, 'overrides': args.overrides, 'model': model_info,
            'timings': self.timings, 'timing_summary': summary, 'evaluations': {str(k): v for k, v in evals.items()},
            'final_model': saved, 'wall_seconds_since_start': round(time.time() - T_START, 3),
            'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
                            'nccl': str(torch.cuda.nccl.version()) if (self.device.type == 'cuda') else None,
                            'fm': fm.__version__, 'fm_file': fm.__file__, 'numpy': np.__version__, 'platform': platform.platform(),
                            'executable': sys.executable},
            'placement': placement,
        }
        if self.rank == 0:
            (out / 'run_summary.json').write_text(json.dumps(result, indent=2) + '\n')
            log(f'done: {json.dumps(summary["t_step_max"] if summary else None)}', self.rank)
        if self.distributed:
            dist.barrier()
            dist.destroy_process_group()
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--mode', choices=['scaling', 'train'], required=True)
    p.add_argument('--stop-after', type=int, default=None, help='last step to run (scaling: warm-up + timed)')
    p.add_argument('--attention', choices=['official', 'sdpa'], default=None)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--verify-dataset', action='store_true', help='hash the dataset file against the config')
    p.add_argument('--log-every', type=int, default=10)
    # overrides for local smoke tests only (recorded in the output)
    p.add_argument('--global-batch', type=int, default=None)
    p.add_argument('--total-steps', type=int, default=None)
    p.add_argument('--warmup-untimed', type=int, default=None)
    p.add_argument('--timed-steps', type=int, default=None)
    p.add_argument('--limit-train', type=int, default=None)
    p.add_argument('--limit-eval', type=int, default=None)
    p.add_argument('--max-tokens-per-micro', type=int, default=None)
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--no-save', action='store_true')
    p.add_argument('--allow-weights-size-mismatch', action='store_true')
    args = p.parse_args(argv)
    raw = Path(args.config).read_bytes()
    cfg = json.loads(raw)
    args.config_sha256 = hashlib.sha256(raw).hexdigest()
    if args.attention is None:
        args.attention = cfg['attention']
    keys = ['global_batch', 'total_steps', 'warmup_untimed', 'timed_steps', 'limit_train', 'limit_eval', 'max_tokens_per_micro', 'fp32', 'no_save', 'allow_weights_size_mismatch']
    args.overrides = {k: getattr(args, k) for k in keys if getattr(args, k) not in (None, False)}
    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA requested but unavailable'
    runner = Runner(args, cfg)
    try:
        return runner.run()
    except Exception:
        tb = traceback.format_exc()
        sys.stderr.write(f'[rank {runner.rank}] {tb}\n')
        if runner.rank == 0:
            try:
                Path(args.out).mkdir(parents=True, exist_ok=True)
                (Path(args.out) / 'run_summary.json').write_text(json.dumps({'status': 'failed', 'label': args.label, 'traceback': tb, 'timings': runner.timings}, indent=2) + '\n')
            except Exception:
                pass
        raise


if __name__ == '__main__':
    main()
