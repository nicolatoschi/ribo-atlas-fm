"""Checkpoint/state contracts for the native full-objective reliability pilot.

This module never implements a forward, loss, derivative or optimizer update.
"""
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

SITE_LIMIT_SECONDS = 43200
PARAMETER_ATOL = 1e-6
PARAMETER_RTOL = 1e-5
LOSS_ATOL = 1e-5
LOSS_RTOL = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def tree_sha(value):
    h = hashlib.sha256()
    def add(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            h.update(b'tensor' + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
            h.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(b'array' + str(item.dtype).encode() + str(item.shape).encode() + item.tobytes())
        elif isinstance(item, dict):
            h.update(b'dict')
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                add(key); add(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode())
            for entry in item:
                add(entry)
        else:
            h.update(type(item).__name__.encode() + repr(item).encode())
    add(value)
    return h.hexdigest()


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_device': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['torch_device'], device)
    assert tree_sha(rng_state(device)) == tree_sha(state), 'RNG restoration mismatch'


def assert_finite_tree(value):
    if torch.is_tensor(value):
        assert bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor'
    elif isinstance(value, dict):
        for item in value.values(): assert_finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value: assert_finite_tree(item)
    elif isinstance(value, float):
        assert math.isfinite(value), 'Nonfinite checkpoint scalar'


def compare_tree(actual, expected, atol=PARAMETER_ATOL, rtol=PARAMETER_RTOL):
    """Require identical structure/metadata and predeclared tensor tolerance."""
    maximum = 0.0
    def compare(a, b):
        nonlocal maximum
        assert type(a) is type(b), (type(a), type(b))
        if torch.is_tensor(a):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            if a.is_floating_point():
                if a.numel(): maximum = max(maximum, float((a - b).abs().max()))
                assert torch.allclose(a, b, atol=atol, rtol=rtol), 'Restart tensor exceeds prospective tolerance'
            else: assert torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and np.array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a: compare(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b): compare(x, y)
        else: assert a == b, (a, b)
    compare(actual, expected)
    return {'maximum_absolute_tensor_difference': maximum,
            'bitwise_identical': tree_sha(actual) == tree_sha(expected),
            'atol': atol, 'rtol': rtol, 'passed': True}


def atomic_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    with temporary.open('rb') as handle: os.fsync(handle.fileno())
    temporary.replace(path)
    return sha(path)


def checkpoint_payload(model, optimizer, device, identity, completed_updates, logical_ids):
    payload = {'format_version': 1, 'identity': identity, 'completed_updates': completed_updates,
               'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
               'scheduler': {'kind': 'constant_learning_rate_no_scheduler',
                             'lr_by_parameter_group': [group['lr'] for group in optimizer.param_groups]},
               'rng': cpu_tree(rng_state(device)),
               'data_cursor': {'next_update': completed_updates + 1, 'logical_ids': logical_ids,
                               'fixture_policy': 'Fixed same logical instances on each update; no shuffled-data claim'}}
    assert_finite_tree(payload)
    return payload


def restore_checkpoint(payload, model, optimizer, device, identity, logical_ids):
    assert payload['format_version'] == 1 and payload['identity'] == identity
    assert payload['completed_updates'] == 1
    assert payload['data_cursor']['next_update'] == 2 and payload['data_cursor']['logical_ids'] == logical_ids
    assert payload['scheduler']['kind'] == 'constant_learning_rate_no_scheduler'
    assert_finite_tree(payload)
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    assert tree_sha(cpu_tree(model.state_dict())) == tree_sha(payload['model'])
    assert tree_sha(cpu_tree(optimizer.state_dict())) == tree_sha(payload['optimizer'])
    assert [group['lr'] for group in optimizer.param_groups] == payload['scheduler']['lr_by_parameter_group']
    restore_rng(payload['rng'], device)


def verify_manifest(path):
    path = Path(path).resolve(); manifest = json.loads(path.read_text()); base = path.parent
    for name, expected in manifest['files'].items():
        assert sha(base / name) == expected, 'Campaign source identity: ' + name
    return manifest


def run_reliability(args, model, optimizer, update, device, rank, world, logical_ids,
                    fixture_path, head_sha, original_freeze, report_context):
    """Two uninterrupted updates, then one update in a separately launched process."""
    manifest_path = args.campaign_manifest.resolve(); manifest = verify_manifest(manifest_path)
    assert args.ensemble_size == 32 and args.grid_points == 50 and args.microbatch == 1
    assert world == 128 and args.local_constructs == 1 and args.case == '5zwm_B'
    assert manifest['configuration']['original_freeze_sha256'] == original_freeze
    assert report_context['length'] == manifest['configuration']['length'] == 175
    assert args.seed == manifest['configuration']['seed']
    assert head_sha == manifest['configuration']['frozen_head_sha256']
    assert sha(fixture_path) == manifest['configuration']['teacher_fixture_sha256']
    assert Path(report_context['source_directory']) == manifest_path.parent / 'RNAbpFlow_training'
    out = args.out.resolve(); root = args.state_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    identity = {'campaign_manifest_sha256': sha(manifest_path), 'original_freeze_sha256': original_freeze,
                'slurm_job_id': int(os.environ['SLURM_JOB_ID']),
                'slurm_restart_count': int(os.environ.get('SLURM_RESTART_COUNT', '0')),
                'rank': rank, 'world_size': world, 'case': args.case, 'length': report_context['length'],
                'seed': args.seed, 'microbatch': args.microbatch, 'local_constructs': args.local_constructs,
                'global_case_instances': 128, 'K': 32, 'T': 50, 'fixture_sha256': sha(fixture_path),
                'official_checkpoint_sha256': manifest['configuration']['official_checkpoint_sha256'],
                'frozen_head_sha256': head_sha,
                'observations_by_logical_id': report_context['observations_by_logical_id'],
                'teacher_by_logical_id': report_context['teacher_by_logical_id']}
    stem = f'rank_{rank:05d}'; initial_path = root / (stem + '_after_update1.pt')
    reference_path = root / (stem + '_after_update2_reference.pt')
    stage_start = time.perf_counter(); rows = []; checkpoint_rows = []; restore_proof = None
    if args.stage == 'continuous':
        assert not initial_path.exists() and not reference_path.exists(), 'Do not overwrite prior campaign state'
        updates = [1, 2]
    else:
        assert initial_path.exists() and reference_path.exists(), 'Both continuous-reference states required'
        restore_start = time.perf_counter()
        baseline_report = json.loads((out / 'continuous' / (stem + '.json')).read_text())
        saved_files = {item['completed_updates']: item for item in baseline_report['checkpoint_files']}
        assert sha(initial_path) == saved_files[1]['sha256'], 'Update1 checkpoint integrity mismatch'
        assert sha(reference_path) == saved_files[2]['sha256'], 'Update2 reference integrity mismatch'
        saved = torch.load(initial_path, map_location='cpu', weights_only=False)
        restore_checkpoint(saved, model, optimizer, device, identity, logical_ids)
        restore_proof = {'checkpoint_file_sha256': sha(initial_path),
                         'restored_model_sha256': tree_sha(saved['model']),
                         'restored_optimizer_sha256': tree_sha(saved['optimizer']),
                         'restored_rng_sha256': tree_sha(saved['rng']),
                         'restored_scheduler_sha256': tree_sha(saved['scheduler']),
                         'restored_data_cursor_sha256': tree_sha(saved['data_cursor']),
                         'seconds': time.perf_counter() - restore_start, 'exact_restoration_passed': True}
        del saved
        updates = [2]
    if world > 1: dist.barrier()
    comparison = None
    for completed in updates:
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        report, _ = update(args.microbatch, 1.)
        assert report['all_gradients_finite'] and report['frozen_head_gradients_absent']
        assert report['first_microbatch_generated_signal']['generated_inverse_gradient_norm'] > 0
        assert report['first_microbatch_generated_signal']['generated_probe_gradient_norm'] > 0
        if device.type == 'cuda':
            reserved = torch.cuda.max_memory_reserved(device); capacity = torch.cuda.get_device_properties(device).total_memory
            assert reserved <= .85 * capacity
            report.update(GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          GPU_peak_reserved_bytes=reserved, GPU_capacity_bytes=capacity)
        snapshot = checkpoint_payload(model, optimizer, device, identity, completed, logical_ids)
        steps = [float(state['step']) for state in snapshot['optimizer']['state'].values()]
        assert steps and all(step == completed for step in steps), 'AdamW moments or step counter reset'
        report.update(completed_updates=completed, model_state_sha256=tree_sha(snapshot['model']),
                      optimizer_state_sha256=tree_sha(snapshot['optimizer']), rng_state_sha256=tree_sha(snapshot['rng']),
                      scheduler_state_sha256=tree_sha(snapshot['scheduler']), data_cursor_sha256=tree_sha(snapshot['data_cursor']))
        report['AdamW_step_counter_range'] = [min(steps), max(steps)]
        if rows:
            assert report['model_state_sha256'] != rows[-1]['model_state_sha256'], 'No second accepted parameter update'
            assert report['optimizer_state_sha256'] != rows[-1]['optimizer_state_sha256'], 'Optimizer moments did not advance'
        if args.stage == 'continuous':
            state_path = initial_path if completed == 1 else reference_path
            started = time.perf_counter(); digest = atomic_checkpoint(state_path, snapshot)
            checkpoint_rows.append({'completed_updates': completed, 'file': state_path.name, 'sha256': digest,
                                    'bytes': state_path.stat().st_size, 'seconds': time.perf_counter() - started})
        else:
            reference = torch.load(reference_path, map_location='cpu', weights_only=False)
            assert reference['identity'] == identity and reference['completed_updates'] == 2
            continuous_report = json.loads((out / 'continuous' / (stem + '.json')).read_text())['updates'][1]
            comparison = {'model': compare_tree(snapshot['model'], reference['model']),
                          'optimizer': compare_tree(snapshot['optimizer'], reference['optimizer']),
                          'RNG': compare_tree(snapshot['rng'], reference['rng'], atol=0, rtol=0),
                          'scheduler': compare_tree(snapshot['scheduler'], reference['scheduler'], atol=0, rtol=0),
                          'data_cursor': compare_tree(snapshot['data_cursor'], reference['data_cursor'], atol=0, rtol=0),
                          'groups': []}
            assert len(report['groups']) == len(continuous_report['groups'])
            for actual, expected in zip(report['groups'], continuous_report['groups']):
                assert actual['logical_ids'] == expected['logical_ids']
                delta = {}
                for key in ['finite_K_case_mean_profile_MSE', 'adjacent_C4_mean', 'teacher_loss', 'maximum_replay_difference']:
                    a, b = actual[key], expected[key]
                    assert math.isclose(a, b, abs_tol=LOSS_ATOL, rel_tol=LOSS_RTOL), ('Restart loss differs', key, a, b)
                    delta[key] = abs(a - b)
                comparison['groups'].append(delta)
            comparison['passed'] = True
            del reference
        del snapshot
        rows.append(report)
        if world > 1: dist.barrier()
    result = {'status': 'Full-objective reliability stage completed', 'stage': args.stage,
              'slurm_job_id': int(os.environ['SLURM_JOB_ID']), 'slurm_step_id': str(os.environ['SLURM_STEP_ID']),
              'rank': rank, 'world_size': world, 'identity': identity, 'updates': rows,
              'checkpoint_files': checkpoint_rows, 'restore_proof': restore_proof,
              'restart_comparison': comparison, 'stage_wall_seconds': time.perf_counter() - stage_start,
              'collective_timeout_seconds': SITE_LIMIT_SECONDS,
              'torch_version': torch.__version__, 'GPU_name': torch.cuda.get_device_name(device),
              'host_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
              'host_memory_scope': 'Cumulative process maximum RSS including setup, full updates, checkpoint/state hashing and comparison; not simultaneous node RSS',
              'timing_scope': 'Consecutive complete optimizer updates; timed native body unchanged; state hashing/checkpoint I/O reported separately',
              'scope': 'One actual 175nt RNA; fixed 128 synthetic stochastic case instances and random frozen head; engineering reliability, not biological efficacy',
              'scheduler_scope': 'Original constant AdamW learning rate; no learning-rate scheduler exists in this native pilot',
              'restart_scope': 'Fresh srun processes within the same allocated job; different-node preemption recovery and shuffled-data cursor are not tested'}
    atomic_json(out / args.stage / (stem + '.json'), result)
    if world > 1: dist.barrier()
    if rank == 0: print(json.dumps({'stage': args.stage, 'status': result['status'], 'updates': len(rows)}), flush=True)
    if world > 1: dist.destroy_process_group()
    return result
