#!/usr/bin/env python3
"""Freeze the wide-job campaign packet: fixed anomaly-off code + length-ladder data.

Code files are byte-identical to native_anomaly_off_b63550bd42ee. Added data only:
RNAbpFlow features/metadata rows for five extra lengths (from the published
RNAbpFlow-Train archive) and an extended teacher fixture generated on JUPITER with
the original recipe and environment. Existing fixture entries must reproduce
bit-for-bit. Never uploads or submits.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / 'native_anomaly_off_b63550bd42ee'
NEW_CASES = ['1e8o_E', '6cc1_B', '1grz_B', '6t3n_A', '6chr_A']
FIXTURE_COUNTS = {'5zwm_B': 512, '1g59_B': 64, '1e8o_E': 64, '6cc1_B': 64, '1l8v_B': 64,
                  '1grz_B': 64, '6t3n_A': 64, '6chr_A': 64, '7ptk_B': 64}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--staged-packet', type=Path, required=True, help='Base packet copy with added features and CSV rows')
    p.add_argument('--fixtures', type=Path, required=True, help='Per-case fixtures generated on JUPITER')
    args = p.parse_args()
    base_manifest = json.loads((BASE / 'frozen_manifest.json').read_text())
    payload = {}
    for name, expected in base_manifest['files'].items():
        data = (BASE / name).read_bytes(); assert sha(data) == expected, name
        payload[name] = data
    staged = args.staged_packet
    for name in base_manifest['files']:
        if name != 'RNAbpFlow_training/metadata/pilot_ensemble.csv':
            assert (staged / name).read_bytes() == payload[name], 'Staged copy differs: ' + name
    old_csv = payload['RNAbpFlow_training/metadata/pilot_ensemble.csv']
    new_bytes = (staged / 'RNAbpFlow_training/metadata/pilot_ensemble.csv').read_bytes()
    assert new_bytes.startswith(old_csv) and old_csv.endswith(b'\r\n'), 'Existing CRLF metadata rows must stay byte-identical'
    payload['RNAbpFlow_training/metadata/pilot_ensemble.csv'] = new_bytes
    new_csv = new_bytes.decode()
    for case in NEW_CASES:
        assert case + ',' in new_csv
        for rel in [f'RNAbpFlow_training/Feature/train_pkl/{case}/{case}.pkl', f'RNAbpFlow_training/pkls/{case}_base/{case}_feat.pkl']:
            payload[rel] = (staged / rel).read_bytes()
    merged = {'seed': None, 'role': 'Frozen singleton native teacher corruptions; no inverse-model activations',
              'cases': {}, 'generation_by_case': {}}
    for case, count in FIXTURE_COUNTS.items():
        part = json.loads((args.fixtures / f'{case}.json').read_text())
        assert merged['seed'] in (None, part['seed']); merged['seed'] = part['seed']
        entries = part['cases'][case]['entries']; assert len(entries) == count
        assert [e['logical_id'] for e in entries] == list(range(count))
        merged['cases'][case] = part['cases'][case]; merged['generation_by_case'][case] = part['generation']
    original = json.loads(payload['native_fixed_teacher_fixture.json'])['cases']['5zwm_B']['entries']
    batched = json.loads(payload['batched_teacher_fixture.json'])['cases']
    checks = {'5zwm_B_first128_vs_native_fixture': [e['raw_tensor_sha256'] for e in merged['cases']['5zwm_B']['entries'][:128]] == [e['raw_tensor_sha256'] for e in original],
              '1l8v_B_first32_vs_batched_fixture': [e['raw_tensor_sha256'] for e in merged['cases']['1l8v_B']['entries'][:32]] == [e['raw_tensor_sha256'] for e in batched['1l8v_B']['entries']],
              '1g59_B_first2_vs_batched_fixture': [e['raw_tensor_sha256'] for e in merged['cases']['1g59_B']['entries'][:2]] == [e['raw_tensor_sha256'] for e in batched['1g59_B']['entries']]}
    assert all(checks.values()), checks
    payload['campaign_teacher_fixture.json'] = (json.dumps(merged) + '\n').encode()
    payload['step_launch.py'] = (HERE / 'step_launch.py').read_bytes()
    payload['freeze_campaign_packet.py'] = Path(__file__).read_bytes()
    lengths = {case: merged['cases'][case]['length'] for case in FIXTURE_COUNTS}
    cfg = dict(base_manifest['configuration'])
    cfg.update({'derived_from_freeze_sha256': base_manifest['freeze_sha256'],
                'change': 'Data-only extension of the anomaly-off packet: five length-ladder cases and an extended teacher fixture; code byte-identical',
                'campaign_fixture_filename': 'campaign_teacher_fixture.json',
                'campaign_fixture_counts': FIXTURE_COUNTS, 'campaign_case_lengths': lengths,
                'fixture_reproduction_checks': checks, 'step_launcher': 'step_launch.py (environment only)'})
    cfg.pop('runs', None)
    files = {name: sha(data) for name, data in payload.items()}
    freeze = sha(json.dumps({'configuration': cfg, 'files': files}, sort_keys=True).encode())
    folder = HERE / ('native_campaign_' + freeze[:12])
    for name, data in payload.items():
        destination = folder / name; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): assert destination.read_bytes() == data
        else: destination.write_bytes(data)
        destination.chmod(0o444)
    manifest = {key: base_manifest[key] for key in ['source_data_sha256', 'external_dependencies', 'source_licence', 'runtime_mutable_exclusions']}
    manifest.update({'status': 'Prepared locally; no submission or GPU execution', 'configuration': cfg, 'files': files, 'freeze_sha256': freeze})
    path = folder / 'frozen_manifest.json'; data = (json.dumps(manifest, indent=2) + '\n').encode()
    if path.exists(): assert path.read_bytes() == data
    else: path.write_bytes(data)
    path.chmod(0o444)
    print(json.dumps({'packet': folder.name, 'freeze_sha256': freeze, 'files': len(files), 'lengths': lengths, 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
