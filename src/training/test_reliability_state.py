#!/usr/bin/env python3
"""Execute checkpoint/RNG/AdamW continuity tests in a genuinely fresh process."""
import argparse
import copy
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from reliability_state import (atomic_checkpoint, checkpoint_payload, compare_tree,
                               restore_checkpoint, sha, tree_sha)

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device('cpu')
IDENTITY = {'case': 'CPU orchestration test only', 'rank': 0, 'fixture': 'fixed'}
IDS = [0, 1]


def setup(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    return model, optimizer


def step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    # All three host RNG streams affect the update; dropout also uses Torch RNG.
    values = torch.randn(2, 4) + random.random() + float(np.random.normal())
    loss = model(values).square().mean(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def child(directory):
    directory = Path(directory); model, optimizer = setup(999)
    initial = torch.load(directory / 'initial.pt', map_location='cpu', weights_only=False)
    restore_checkpoint(initial, model, optimizer, DEVICE, IDENTITY, IDS)
    loss = step(model, optimizer)
    actual = checkpoint_payload(model, optimizer, DEVICE, IDENTITY, 2, IDS)
    reference = torch.load(directory / 'reference.pt', map_location='cpu', weights_only=False)
    comparison = {key: compare_tree(actual[key], reference[key], atol=0, rtol=0)
                  for key in ['model', 'optimizer', 'rng', 'scheduler', 'data_cursor']}
    assert loss == json.loads((directory / 'loss.json').read_text())['second_loss']
    (directory / 'fresh_process_result.json').write_text(json.dumps({'passed': True, 'comparison': comparison,
                                                                  'loss': loss, 'process_id': os.getpid()}))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--child'); args = parser.parse_args()
    if args.child: child(args.child); return
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='ribo_reliability_state_') as directory:
        directory = Path(directory); model, optimizer = setup(20261014)
        first_loss = step(model, optimizer)
        initial = checkpoint_payload(model, optimizer, DEVICE, IDENTITY, 1, IDS)
        atomic_checkpoint(directory / 'initial.pt', initial)
        second_loss = step(model, optimizer)
        reference = checkpoint_payload(model, optimizer, DEVICE, IDENTITY, 2, IDS)
        atomic_checkpoint(directory / 'reference.pt', reference)
        (directory / 'loss.json').write_text(json.dumps({'second_loss': second_loss}))
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', str(directory)],
                                 capture_output=True, text=True, timeout=60)
        assert process.returncode == 0, process.stderr + process.stdout
        fresh = json.loads((directory / 'fresh_process_result.json').read_text()); assert fresh['passed']
        assert fresh['process_id'] != os.getpid()
        rejected = []
        for label, mutate in [
            ('identity', lambda item: item['identity'].update(fixture='changed')),
            ('cursor', lambda item: item['data_cursor'].update(next_update=3)),
            ('scheduler', lambda item: item['scheduler'].update(kind='changed')),
            ('nonfinite_optimizer', lambda item: next(iter(item['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))),
        ]:
            altered = copy.deepcopy(initial); mutate(altered)
            trial, trial_optimizer = setup(998)
            try: restore_checkpoint(altered, trial, trial_optimizer, DEVICE, IDENTITY, IDS)
            except (AssertionError, RuntimeError): rejected.append(label)
            else: raise AssertionError('Checkpoint corruption accepted: ' + label)
        drift = copy.deepcopy(reference['model']); next(iter(drift.values())).add_(.01)
        try: compare_tree(drift, reference['model'])
        except AssertionError: rejected.append('parameter_drift')
        else: raise AssertionError('State drift accepted')
        result = {'passed': True, 'GPU_allocation': False,
                  'scope': 'Actually executed CPU AdamW/dropout checkpoint orchestration in a fresh Python process; full native K32/T50 GPU path remains unexecuted',
                  'first_loss': first_loss, 'second_loss': second_loss,
                  'fresh_process': fresh, 'rejected_corruptions': rejected,
                  'reliability_state_sha256': sha(ROOT / 'reliability_state.py'),
                  'test_source_sha256': sha(Path(__file__)), 'torch_version': torch.__version__,
                  'platform': platform.platform(), 'python_version': platform.python_version(),
                  'parent_process_id': os.getpid()}
        (ROOT / 'reliability_local_tests.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'passed': True, 'rejected_corruptions': rejected,
                          'fresh_process_all_states_bitwise_equal': all(item['bitwise_identical'] for item in fresh['comparison'].values())}))


if __name__ == '__main__': main()
