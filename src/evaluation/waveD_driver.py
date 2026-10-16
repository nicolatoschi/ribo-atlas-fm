#!/usr/bin/env python3
"""RIBO-ATLAS-FM Wave D: frozen evaluation-pipeline cost and robustness across RNA length.

One process per GPU (launched by step_launch.py inside one srun per node). Each
rank reads its static case list from partition_plan.json and processes it
sequentially:
  1. published pretrained RNAbpFlow generation, K=32 samples, T=50 steps,
     seeds 420*(sample_id+1), original FlowModule.predict_step / Interpolant.sample
     / all-atom mapping / PDB writer (as in inference packet 6cbee47ef9de);
  2. the project's OL3/GBn2 OH-fragment refinement (refinement packet
     e665e8ed11ae physics via waveD_refine.py) of the pre-declared sample IDs
     0..R(L)-1, in a persistent spawned worker process on the same GPU;
  3. a descriptive physical-validity panel and C4' lDDT (rna_backbone_metrics)
     of every generated and refined sample versus the native coordinates.
Every case writes one JSON atomically; a failure in a sample or case is recorded
and the rank continues. These cases come from RNAbpFlow's own training
metadata: lDDT here is PIPELINE QA, NOT held-out accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform as py_platform
import queue as queue_module
import resource
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

EXPECTED_CHECKPOINT_SHA = 'bb598a736193734541f63c59dae75d45db8c04f01a8414a758677367c4b1e213'
SCOPE = ('RNAbpFlow training cases (plus the P4-P6 calibration anchor, whose family is in training): '
         'lDDT/validity are pipeline QA, NOT held-out accuracy; timings are engineering cost measurements.')
PACKET_CODE = ('waveD_driver.py', 'waveD_refine.py', 'openmm_rna_refinement.py', 'rna_backbone_metrics.py',
               'step_launch.py', 'partition_plan.json', 'case_table.json')


class Interrupted(BaseException):
    pass


def _on_signal(signum, frame):
    raise Interrupted(f'signal {signum}')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    os.replace(temporary, path)


def clean(value):
    """JSON-safe copy: non-finite floats become None (recorded explicitly elsewhere)."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ----------------------------------------------------------------------------- packet

def verify_packet(packet, case_ids, check_all=False):
    manifest = json.loads((packet / 'FROZEN_MANIFEST.json').read_text())
    files = manifest['files']
    recomputed = hashlib.sha256(json.dumps({'configuration': manifest['configuration'],
                                            'files': {k: v['sha256'] for k, v in files.items()}},
                                           sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if recomputed != manifest['freeze_sha256']:
        raise RuntimeError('FROZEN_MANIFEST freeze sha does not match its configuration+files')
    wanted = [k for k in files if check_all or k in PACKET_CODE or k.startswith('RNAbpFlow/src/')
              or any(k.startswith(f'cases/{c}/') for c in case_ids)]
    bad = [k for k in wanted if digest(packet / k) != files[k]['sha256']]
    if bad:
        raise RuntimeError(f'Packet files differ from FROZEN_MANIFEST: {bad[:10]}')
    return {'freeze_sha256': manifest['freeze_sha256'], 'verified_file_count': len(wanted),
            'packet_directory_name_matches': packet.name == 'waveD_' + manifest['freeze_sha256'][:12]}


# ----------------------------------------------------------------------------- model/input

def load_model(checkpoint, timesteps):
    import torch
    from omegaconf import OmegaConf
    from src.models.flow_module_inf import FlowModule
    model = FlowModule.load_from_checkpoint(str(checkpoint), map_location='cpu', weights_only=False, strict=True)
    model.eval()
    # RNAbpFlow training flow_module.py enables anomaly mode at import; the
    # inference module does not, but the state is forced off and asserted here.
    torch.autograd.set_detect_anomaly(False)
    assert not torch.is_anomaly_enabled()
    model._infer_cfg = OmegaConf.create({'output_format': 'pdb'})
    model._interpolant_cfg = OmegaConf.create({'sampling': {'num_timesteps': timesteps}})
    return model


def prepare_case(cases_dir, case_id):
    import torch
    from src.data.data_transform import make_atom_mask
    from src.data import base_constants
    directory = cases_dir / case_id
    fasta = directory / f'{case_id}.fasta'
    sequence = ''.join(x.strip() for x in fasta.read_text().splitlines() if not x.startswith('>'))
    if not sequence or set(sequence) - set('AUGC'):
        raise ValueError('sequence must be non-empty AUGC')
    data = np.load(directory / 'case_input.npz')
    ss = data['ss']; L = len(sequence)
    if ss.shape != (L, L, 3) or not np.all((ss == 0) | (ss == 1)):
        raise ValueError('ss must be binary LxLx3')
    if not all(np.array_equal(ss[..., c], ss[..., c].T) for c in range(3)):
        raise ValueError('ss channels must be symmetric')
    mapfeat = make_atom_mask(case_id, str(cases_dir))
    tensors = {k: torch.as_tensor(v).unsqueeze(0) for k, v in mapfeat.items()}
    atoms = list(base_constants.atom_types)
    mask = np.asarray([[name in base_constants.residue_atoms[b] for name in atoms] for b in sequence], dtype=bool)
    return {'sequence': sequence, 'length': L, 'mapfeat': tensors, 'ss': torch.as_tensor(ss).unsqueeze(0).float(),
            'onehot': tensors['onehot'].float(), 'atom_names': atoms, 'atom_mask': mask,
            'native': np.asarray(data['native_atom28'], dtype=float), 'native_mask': np.asarray(data['native_mask28'], dtype=bool),
            'pair_counts': [int(ss[..., c].sum() // 2) for c in range(3)],
            'input_hashes': {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()}}


# ----------------------------------------------------------------------------- panel / metrics

def stats(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None
    return {'count': int(v.size), 'minimum': float(v.min()), 'median': float(np.median(v)), 'maximum': float(v.max()),
            'mean': float(v.mean())}


def validity_panel(coords, mask, atom_names):
    """Descriptive geometry panel on one structure (atom28 layout). Not an energy test."""
    from scipy.spatial import cKDTree
    coords = np.asarray(coords, dtype=float)
    present = coords[mask]
    out = {'atoms_evaluated': int(mask.sum()), 'all_coordinates_finite': bool(np.isfinite(present).all())}
    if not out['all_coordinates_finite']:
        return out
    P, C4, O3 = (atom_names.index(n) for n in ('P', "C4'", "O3'"))
    L = coords.shape[0]

    def adjacent(a, b, offset):
        idx = [i for i in range(L - offset) if mask[i, a] and mask[i + offset, b]]
        return np.asarray([np.linalg.norm(coords[i, a] - coords[i + offset, b]) for i in idx])
    o3p = adjacent(O3, P, 1)
    c4 = coords[:, C4]; c4ok = mask[:, C4]
    d = np.linalg.norm(c4[:, None] - c4[None, :], axis=-1)
    far = np.triu(np.ones((L, L), dtype=bool), k=2) & c4ok[:, None] & c4ok[None, :]
    residue_of = np.repeat(np.arange(L), mask.sum(1))
    pairs = cKDTree(present).query_pairs(r=2.0, output_type='ndarray')
    sep = np.abs(residue_of[pairs[:, 0]] - residue_of[pairs[:, 1]]) if len(pairs) else np.zeros(0)
    out.update({
        'O3prime_next_P_angstrom': stats(o3p),
        'O3prime_next_P_above_2p1_count': int((o3p > 2.1).sum()), 'O3prime_next_P_below_1p3_count': int((o3p < 1.3).sum()),
        'O3prime_next_P_connections': int(o3p.size),
        'adjacent_P_angstrom': stats(adjacent(P, P, 1)), 'adjacent_C4prime_angstrom': stats(adjacent(C4, C4, 1)),
        'nonadjacent_C4prime_pairs_below_3_angstrom_count': int(((d < 3) & far).sum()),
        'heavy_atom_pairs_below_2_angstrom_residue_separation_ge2_count': int((sep >= 2).sum()),
        'definition': ('Screens: O3prime(i)-P(i+1) outside 1.3-2.1 A; nonadjacent (|i-j|>=2) C4prime pairs < 3 A; '
                       'heavy-atom pairs < 2.0 A between residues |i-j|>=2. Descriptive, not an energy/stereochemistry certificate.')})
    return out


def kabsch_rmsd(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    sign = np.sign(np.linalg.det(u @ vt))
    s[-1] *= sign
    return float(np.sqrt(max(0.0, (np.sum(a * a) + np.sum(b * b) - 2 * s.sum()) / len(a))))


def accuracy(c4_pred, prepared):
    from rna_backbone_metrics import c4prime_lddt
    names = prepared['atom_names']; C4 = names.index("C4'")
    ok = prepared['native_mask'][:, C4]
    native = prepared['native'][ok, C4]; pred = np.asarray(c4_pred, dtype=float)[ok]
    if not np.isfinite(pred).all():
        return {'C4prime_lDDT': None, 'C4prime_Kabsch_RMSD_angstrom': None, 'residues': int(ok.sum())}
    return {'C4prime_lDDT': c4prime_lddt(pred, native)['mean_score'], 'C4prime_Kabsch_RMSD_angstrom': kabsch_rmsd(pred, native),
            'residues': int(ok.sum())}


# ----------------------------------------------------------------------------- refinement worker

def refine_worker(code_dir, tasks, results):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.path.insert(0, code_dir)
    import waveD_refine
    while True:
        task = tasks.get()
        if task is None:
            return
        def progress(tag, task_id=task['task_id']):
            results.put({'event': 'progress', 'task_id': task_id, 'tag': tag, 'time': time.time()})
        record = waveD_refine.refine_sample(task['pdb'], task['sequence'], task['sample_id'], task['out'],
                                            platform_name=task['platform'], device_index=task['device_index'],
                                            stage_cap_seconds=task['stage_cap_seconds'], progress=progress)
        results.put({'event': 'result', 'task_id': task['task_id'], 'record': record})


class RefinementRunner:
    def __init__(self, code_dir, memory_probe):
        self.context = mp.get_context('spawn'); self.code_dir = code_dir; self.memory_probe = memory_probe
        self.spawned = 0; self.task_counter = 0; self.process = None; self.start()

    def start(self):
        self.tasks = self.context.Queue(); self.results = self.context.Queue()
        self.process = self.context.Process(target=refine_worker, args=(self.code_dir, self.tasks, self.results), daemon=True)
        self.process.start(); self.spawned += 1

    def restart(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate(); self.process.join(10)
            if self.process.is_alive():
                self.process.kill(); self.process.join(5)
        self.start()

    def run(self, task, timeout):
        self.task_counter += 1; task = dict(task, task_id=self.task_counter)
        start = time.perf_counter(); submitted = time.time(); memory = []; events = []
        self.tasks.put(task)
        while True:
            try:
                message = self.results.get(timeout=1.0)
            except queue_module.Empty:
                sample = self.memory_probe()
                if sample is not None:
                    memory.append(sample)
                if not self.process.is_alive():
                    code = self.process.exitcode; self.restart()
                    return self._failed(task, 'worker_died', f'refinement worker exit code {code}', start, memory, events)
                if time.perf_counter() - start > timeout:
                    self.restart()
                    return self._failed(task, 'worker_timeout', f'no result within {timeout:.0f} s', start, memory, events)
                continue
            if message.get('task_id') != task['task_id']:
                continue
            if message['event'] == 'progress':
                events.append({'tag': message['tag'], 'seconds_after_submit': message['time'] - submitted})
                sample = self.memory_probe()
                if sample is not None:
                    memory.append(sample)
                continue
            record = message['record']
            record['parent_observed_wall_seconds'] = time.perf_counter() - start
            record['device_used_bytes_max_sampled'] = max(memory) if memory else None
            record['worker_spawn_count'] = self.spawned
            record['progress_events'] = events
            return record

    def _failed(self, task, category, message, start, memory, events):
        return {'sample_id': task['sample_id'], 'status': 'failed', 'failure_category': category, 'failure_message': message,
                'parent_observed_wall_seconds': time.perf_counter() - start,
                'device_used_bytes_max_sampled': max(memory) if memory else None, 'worker_spawn_count': self.spawned,
                'progress_events': events, 'timing_seconds': {}}

    def stop(self):
        try:
            self.tasks.put(None); self.process.join(15)
        finally:
            if self.process.is_alive():
                self.process.terminate(); self.process.join(5)


# ----------------------------------------------------------------------------- per case

def run_case(ctx, case, out_root):
    import torch
    args = ctx['args']; device = ctx['device']; model = ctx['model']; capture = ctx['capture']
    case_id = case['case_id']; L = case['length']
    case_dir = out_root / 'cases' / case_id; samples_dir = case_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    started = time.time(); t_case = time.perf_counter()
    record = {'case_id': case_id, 'length': L, 'role': case['role'], 'origin': case.get('origin'), 'length_bin': case['length_bin'], 'scope': SCOPE,
              'status': 'running', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started)),
              'rank': ctx['rank_info'], 'config': {'K': args.K, 'timesteps': args.timesteps,
              'refined_sample_ids_planned': case['refined_sample_ids'], 'seed_rule': '420*(sample_id+1)'},
              'samples': [], 'refinement': [], 'timing_seconds': {}, 'memory': {}}
    case_json = case_dir / 'case_result.json'

    def flush(status=None):
        if status:
            record['status'] = status
        record['timing_seconds']['case_wall'] = time.perf_counter() - t_case
        json_write(case_json, clean(record))
    try:
        t0 = time.perf_counter(); prepared = prepare_case(ctx['cases_dir'], case_id)
        record['timing_seconds']['input_preparation'] = time.perf_counter() - t0
        record['input'] = {'sequence': prepared['sequence'], 'ss_channel_pair_counts': prepared['pair_counts'],
                           'input_hashes': prepared['input_hashes']}
        if prepared['length'] != L:
            raise ValueError('prepared length differs from plan')
        t0 = time.perf_counter()
        record['native_panel'] = validity_panel(prepared['native'], prepared['native_mask'], prepared['atom_names'])
        record['timing_seconds']['native_panel'] = time.perf_counter() - t0
        gpu_inputs = {k: ctx['move'](prepared[k], device) for k in ('mapfeat', 'ss', 'onehot')}
        if ctx['cuda']:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        coords = {}; gen_wall = time.perf_counter(); analysis_seconds = 0.0; observed = []
        for k in range(args.K):
            estimate = max(observed) * 1.5 if observed else case['g_est'] * 3.0
            if time.time() + estimate > ctx['deadline']:
                record['samples'].append({'sample_id': k, 'status': 'skipped_deadline'}); continue
            sample = {'sample_id': k, 'seed': 420 * (k + 1)}
            try:
                model._output_dir = str(samples_dir)
                batch = {'sample_id': torch.tensor([k]), 'seed': int(420 * (k + 1)), **gpu_inputs}
                capture.clear(); ctx['sync'](); t0 = time.perf_counter()
                with torch.no_grad():
                    model.predict_step(batch, device)
                ctx['sync'](); sample['predict_step_including_PDB_write_seconds'] = time.perf_counter() - t0
                observed.append(sample['predict_step_including_PDB_write_seconds'])
                sample['published_sampling_seconds'] = capture['sampling_seconds']
                array = capture['coordinates']
                if array.shape != (1, L, len(prepared['atom_names']), 3):
                    raise ValueError(f'unexpected generated shape {array.shape}')
                pdb = samples_dir / f'Sample_{k}.pdb'
                sample['PDB_sha256'] = digest(pdb)
                sample['coordinate_sha256'] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                coords[k] = array[0]
                t0 = time.perf_counter()
                sample['panel'] = validity_panel(array[0], prepared['atom_mask'], prepared['atom_names'])
                sample.update(accuracy(array[0][:, prepared['atom_names'].index("C4'")], prepared))
                analysis_seconds += time.perf_counter() - t0
                sample['status'] = 'ok' if sample['panel']['all_coordinates_finite'] else 'nonfinite'
            except Interrupted:
                sample['status'] = 'interrupted'; record['samples'].append(sample); raise
            except Exception as error:
                sample.update({'status': 'failed', 'failure_message': f'{type(error).__name__}: {error}',
                               'traceback': traceback.format_exc()[-3000:]})
                if ctx['cuda']:
                    torch.cuda.empty_cache()
            record['samples'].append(sample)
        record['timing_seconds']['generation_loop_wall_including_panel'] = time.perf_counter() - gen_wall
        ok = [s for s in record['samples'] if s.get('status') == 'ok']
        record['timing_seconds']['generation_sampler_gpu_seconds_sum'] = sum(s['published_sampling_seconds'] for s in ok)
        record['timing_seconds']['generation_predict_step_seconds_sum'] = sum(s['predict_step_including_PDB_write_seconds'] for s in ok)
        record['timing_seconds']['panel_and_lDDT_seconds'] = analysis_seconds
        if ctx['cuda']:
            record['memory']['generation_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated(device))
            record['memory']['generation_peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved(device))
        record['memory']['host_max_rss_ru_maxrss_process'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record['memory']['host_max_rss_unit'] = 'kilobytes (Linux)' if sys.platform.startswith('linux') else 'bytes (macOS)'
        if coords:
            ids = sorted(coords)
            np.savez_compressed(case_dir / 'generated_coordinates.npz', sample_id=np.asarray(ids),
                                coordinates_angstrom=np.stack([coords[i] for i in ids]))
        flush('generation_done')
        # ---- refinement of the pre-declared subset --------------------------------
        if ctx['cuda']:
            torch.cuda.empty_cache()
            record['memory']['device_used_bytes_before_refinement'] = ctx['memory_probe']()
        stage_cap = args.stage_cap_base * max(1.0, (L / 158.0) ** 2)
        ref_observed = []
        for k in case['refined_sample_ids']:
            gen = next((s for s in record['samples'] if s['sample_id'] == k), None)
            if gen is None or gen.get('status') != 'ok':
                record['refinement'].append({'sample_id': k, 'status': 'not_attempted_generation_unavailable'}); continue
            estimate = max(ref_observed) * 1.5 if ref_observed else case['r_est'] * 2.0
            if time.time() + estimate > ctx['deadline']:
                record['refinement'].append({'sample_id': k, 'status': 'skipped_deadline'}); continue
            timeout = min(4 * stage_cap + 600.0, max(30.0, ctx['hard_deadline'] - time.time()))
            task = {'pdb': str(samples_dir / f'Sample_{k}.pdb'), 'sequence': prepared['sequence'], 'sample_id': k,
                    'out': str(case_dir / 'refined' / f'sample_{k:02d}'), 'platform': args.openmm_platform,
                    'device_index': ctx['device_index'], 'stage_cap_seconds': stage_cap}
            try:
                result = ctx['refiner'].run(task, timeout)
            except Interrupted:
                record['refinement'].append({'sample_id': k, 'status': 'interrupted'}); raise
            result['first_refinement_on_rank'] = not ctx['refinement_started']; ctx['refinement_started'] = True
            if result.get('status') == 'ok':
                result.update({('after_' + key): value for key, value in accuracy(result.pop('refined_C4prime_angstrom'), prepared).items()})
                result['before_C4prime_lDDT'] = gen.get('C4prime_lDDT')
                ref_observed.append(result['parent_observed_wall_seconds'])
            else:
                result.pop('refined_C4prime_angstrom', None)
            record['refinement'].append(result)
            flush()
        done = [r for r in record['refinement'] if r.get('status') == 'ok']
        record['timing_seconds']['refinement_worker_seconds_sum'] = sum(r['timing_seconds']['complete_case_wall'] for r in done)
        record['timing_seconds']['refinement_parent_observed_seconds_sum'] = sum(
            r.get('parent_observed_wall_seconds', 0.0) for r in record['refinement'] if 'parent_observed_wall_seconds' in r)
        statuses = [s.get('status') for s in record['samples']] + [r.get('status') for r in record['refinement']]
        record['counts'] = {'generated_ok': sum(s == 'ok' for s in statuses[:len(record['samples'])]),
                            'generated_failed_or_nonfinite': sum(s in ('failed', 'nonfinite') for s in statuses[:len(record['samples'])]),
                            'generation_skipped_deadline': sum(s == 'skipped_deadline' for s in statuses[:len(record['samples'])]),
                            'refined_ok': len(done), 'refinement_failed': sum(r.get('status') == 'failed' for r in record['refinement']),
                            'refinement_skipped_deadline': sum(r.get('status') == 'skipped_deadline' for r in record['refinement'])}
        complete = (record['counts']['generated_ok'] == args.K and record['counts']['refined_ok'] == len(case['refined_sample_ids']))
        flush('complete' if complete else 'partial')
    except Interrupted:
        record['interrupted'] = True; flush('interrupted'); raise
    except Exception as error:
        record.update({'failure_message': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()[-4000:]})
        flush('failed')
    return record


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--node-slot', type=int, required=True)
    parser.add_argument('--gpu-slot', type=int, default=None, help='default: LOCAL_RANK/SLURM_LOCALID')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True, help='no new sample starts after this time')
    parser.add_argument('--hard-deadline-epoch', type=float, required=True, help='running refinement is abandoned after this time')
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--timesteps', type=int, default=50)
    parser.add_argument('--stage-cap-base', type=float, default=240.0, help='per-stage minimizer wall cap at 158 nt; scaled by (L/158)^2')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--openmm-platform', choices=('CUDA', 'CPU'), default='CUDA')
    parser.add_argument('--smoke', action='store_true', help='local CPU smoke test: allows K/T/case/refine overrides')
    parser.add_argument('--cases', default=None, help='smoke only: comma-separated case ids')
    parser.add_argument('--refine-count', type=int, default=None, help='smoke only')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _on_signal); signal.signal(signal.SIGUSR1, _on_signal)
    packet = args.packet.resolve(); out_root = args.out.resolve()
    local = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0')))
    gpu_slot = local if args.gpu_slot is None else args.gpu_slot
    plan = json.loads((packet / 'partition_plan.json').read_text())
    table = {c['case_id']: c for c in json.loads((packet / 'case_table.json').read_text())['cases']}
    if args.smoke:
        assert args.cases, '--smoke requires --cases'
        case_ids = args.cases.split(',')
    else:
        assert args.K == plan['K'] == 32 and args.timesteps == plan['timesteps'] == 50, 'production run is K32/T50'
        assert args.device == 'cuda' and args.openmm_platform == 'CUDA'
        assert args.cases is None and args.refine_count is None
        case_ids = plan['assignments'][str(args.node_slot)][str(gpu_slot)]
    slot_name = f'{args.node_slot:02d}_{gpu_slot}'
    rank_json = out_root / 'ranks' / f'rank_{slot_name}.json'
    rank = {'node_slot': args.node_slot, 'gpu_slot': gpu_slot, 'hostname': socket.gethostname(), 'pid': os.getpid(),
            'case_ids': case_ids, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'status': 'starting',
            'deadline_epoch': args.deadline_epoch, 'hard_deadline_epoch': args.hard_deadline_epoch, 'smoke': args.smoke,
            'env': {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_PROCID', 'SLURM_LOCALID', 'RANK', 'LOCAL_RANK',
                                                   'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES', 'CUDA_CACHE_PATH', 'OMP_NUM_THREADS')},
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'machine': py_platform.machine(), 'python': sys.version.split()[0], 'scope': SCOPE, 'cases': {}}
    json_write(rank_json, rank)
    t_setup = time.perf_counter()
    try:
        rank['packet_verification'] = verify_packet(packet, case_ids)
        configuration = json.loads((packet / 'FROZEN_MANIFEST.json').read_text())['configuration']
        import openmm
        from openmm import app as openmm_app
        ff_dir = Path(openmm_app.__file__).resolve().parent / 'data'
        rank['openmm'] = {'version': openmm.__version__,
                          'forcefield_hashes': {n: digest(ff_dir / n) for n in configuration['forcefield_hashes']}}
        if (rank['openmm']['version'] != configuration['openmm_version']
                or rank['openmm']['forcefield_hashes'] != configuration['forcefield_hashes']):
            raise RuntimeError('OpenMM version or OL3/GBn2 XML hashes differ from the frozen refinement protocol')
        rank['checkpoint_sha256_verified'] = digest(args.checkpoint) == EXPECTED_CHECKPOINT_SHA
        if not rank['checkpoint_sha256_verified']:
            raise RuntimeError('checkpoint differs from the verified official RNA3DB.ckpt')
        source = packet / 'RNAbpFlow'
        os.chdir(source); sys.path.insert(0, str(source)); sys.path.insert(1, str(packet))
        import torch
        torch.set_num_threads(args.cpu_threads)
        torch.set_float32_matmul_precision('high')
        cuda = args.device == 'cuda'
        if cuda:
            visible = torch.cuda.device_count()
            device_index = 0 if visible == 1 else gpu_slot if visible > gpu_slot else None
            if device_index is None:
                raise RuntimeError(f'{visible} visible GPUs; cannot map gpu slot {gpu_slot}')
            torch.cuda.set_device(device_index); device = torch.device('cuda', device_index)
            props = torch.cuda.get_device_properties(device_index)
            rank['gpu'] = {'device_index': device_index, 'visible_device_count': visible, 'name': props.name,
                           'total_memory_bytes': int(props.total_memory), 'uuid': str(getattr(props, 'uuid', '')),
                           'pci_bus_id': getattr(props, 'pci_bus_id', None)}
            sync = lambda: torch.cuda.synchronize(device)
            def memory_probe():
                free, total = torch.cuda.mem_get_info(device_index); return int(total - free)
        else:
            device_index = 0; device = torch.device('cpu'); sync = lambda: None; memory_probe = lambda: None
        t0 = time.perf_counter(); model = load_model(args.checkpoint, args.timesteps).to(device); sync()
        rank['model_load_and_transfer_seconds'] = time.perf_counter() - t0
        rank['autograd_anomaly_enabled'] = torch.is_anomaly_enabled()
        rank['parameter_count'] = sum(p.numel() for p in model.parameters())
        rank['torch_version'] = torch.__version__
        capture = {}
        original_sample = model.interpolant.sample
        def recorded_sample(*a, **kw):
            sync(); start = time.perf_counter(); array = original_sample(*a, **kw); sync()
            capture['sampling_seconds'] = time.perf_counter() - start; capture['coordinates'] = array
            return array
        model.interpolant.sample = recorded_sample
        def move(value, dev):
            if torch.is_tensor(value):
                return value.to(dev)
            if isinstance(value, dict):
                return {k: move(v, dev) for k, v in value.items()}
            return value
        cases = []
        for cid in case_ids:
            c = dict(table[cid]); L = c['length']
            c['g_est'] = 0.8 + 0.95 * (L / 158.0) ** 2; c['r_est'] = 4.0 + 26.0 * (L / 158.0) ** 2
            if args.refine_count is not None:
                c['refined_sample_ids'] = list(range(min(args.refine_count, args.K)))
            cases.append(c)
        # warm-up: one excluded sample (separate ID/seed/path) before any measured case
        warm = time.perf_counter()
        for c in cases:  # first case whose input prepares; a warm-up failure never stops the rank
            try:
                prepared = prepare_case(packet / 'cases', c['case_id'])
                model._output_dir = str(out_root / 'warmup' / slot_name); Path(model._output_dir).mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    model.predict_step({'sample_id': torch.tensor([-1]), 'seed': 42000042 + args.node_slot * 4 + gpu_slot,
                                        **{k: move(prepared[k], device) for k in ('mapfeat', 'ss', 'onehot')}}, device)
                sync(); rank['warmup_case'] = c['case_id']; break
            except Exception as error:
                rank.setdefault('warmup_failures', []).append({'case_id': c['case_id'], 'error': f'{type(error).__name__}: {error}'})
                if cuda:
                    torch.cuda.empty_cache()
        rank['warmup_generation_seconds'] = time.perf_counter() - warm
        refiner = RefinementRunner(str(packet), memory_probe)
        rank['setup_seconds'] = time.perf_counter() - t_setup; rank['status'] = 'running'; json_write(rank_json, clean(rank))
        ctx = {'args': args, 'device': device, 'device_index': device_index, 'model': model, 'capture': capture,
               'cases_dir': packet / 'cases', 'cuda': cuda, 'sync': sync, 'move': move, 'memory_probe': memory_probe,
               'refiner': refiner, 'refinement_started': False, 'deadline': args.deadline_epoch,
               'hard_deadline': args.hard_deadline_epoch,
               'rank_info': {'node_slot': args.node_slot, 'gpu_slot': gpu_slot, 'hostname': rank['hostname'],
                             'device_index': device_index, 'gpu_name': rank.get('gpu', {}).get('name')}}
        try:
            for c in cases:
                result = run_case(ctx, c, out_root)
                rank['cases'][c['case_id']] = {'status': result['status'], 'case_wall_seconds': result['timing_seconds'].get('case_wall')}
                json_write(rank_json, clean(rank))
        finally:
            refiner.stop()
        rank['status'] = 'finished'
    except Interrupted as error:
        rank['status'] = 'interrupted'; rank['interrupt'] = str(error)
    except Exception as error:
        rank['status'] = 'failed'; rank['failure_message'] = f'{type(error).__name__}: {error}'
        rank['traceback'] = traceback.format_exc()[-4000:]
    rank['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    rank['rank_wall_seconds'] = time.perf_counter() - t_setup
    json_write(rank_json, clean(rank))
    print(json.dumps({'rank': slot_name, 'status': rank['status'], 'cases': rank['cases']}), flush=True)
    return 0 if rank['status'] == 'finished' else 1


if __name__ == '__main__':
    sys.exit(main())
