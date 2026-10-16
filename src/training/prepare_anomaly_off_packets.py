#!/usr/bin/env python3
"""Freeze anomaly-off re-measurement packets derived from native 670450638bf1; never submits.

RNAbpFlow src/models/flow_module.py enables torch.autograd.set_detect_anomaly(True)
at import. The 157-nt full-trajectory, ensemble and geometry-scaling drivers disable
it after model construction; the batched driver used for every 175-nt measurement
did not. The only numerical-path change here is that same statement (plus an
assertion). Workload and update are asserted AST-identical to the original.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENSEMBLE = HERE.parent
BASE = ENSEMBLE / 'native_fixed_frozen_670450638bf1'
SITE_LAUNCHER = ENSEMBLE / 'site_limit_override' / 'native_fixed_site_limit_launcher.py'
SITE_LAUNCHER_SHA = '9c5deddc5e033c9c46c44167e27eac68da6a689aac0df3b8df78ae3ea44da453'
RELIABILITY = ENSEMBLE / 'production_reliability_pilot'
FIX = 'torch.autograd.set_detect_anomaly(False);assert not torch.is_anomaly_enabled()'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def nested(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


def fixed_batched_driver(original):
    text = replace_once(original, '    model=Workload().to(device)\n', '    model=Workload().to(device);' + FIX + '\n')
    text = replace_once(text, "    report.update(status='Actual fixed-work batched final-generated ensemble training update completed',rank=rank,world_size=world,",
                        "    report.update(status='Actual fixed-work batched final-generated ensemble training update completed',rank=rank,world_size=world,\n                  autograd_anomaly_enabled=torch.is_anomaly_enabled(),")
    before, after = ast.parse(original), ast.parse(text)
    for name in ['Workload', 'update']:
        assert ast.dump(nested(before, name)) == ast.dump(nested(after, name)), name
    return text


def fixed_launcher(original):
    text = replace_once(original, "'numerical_backend':'Exact native isolated-largest-eigenvector derivative; unchanged8d2 driver',",
                        "'numerical_backend':'Exact native isolated-largest-eigenvector derivative; 8d2 Workload/update; autograd anomaly-detection debug mode disabled after model construction',")
    text = replace_once(text, "'source_driver_unchanged':True,",
                        "'source_driver_unchanged':False,'source_driver_change':'Only torch.autograd.set_detect_anomaly(False) plus assertion after model construction; Workload and update AST-identical to 670450638bf1',")
    return text


README = """ANOMALY-OFF RE-MEASUREMENT PACKETS (derived from native 670450638bf1)

Diagnosis: RNAbpFlow src/models/flow_module.py line 25 calls
torch.autograd.set_detect_anomaly(True) at import. The batched driver never
disabled it, so every 175-nt timing (2305296/97/98, 2319595) recorded Python
stack traces per autograd node and NaN-checked every backward output. The
157-nt full_trajectory_driver (2298974) disables it after model construction.

Change: one statement after model construction,
  torch.autograd.set_detect_anomaly(False);assert not torch.is_anomaly_enabled()
plus a report field autograd_anomaly_enabled and corrected launcher metadata
strings. Workload and update are AST-identical to 670450638bf1; inputs,
fixture, checkpoint, head, losses, guards and optimizer are unchanged.
Anomaly mode changes speed only: local CPU A/B gave identical gradient norms.

Runs: 0 = 1-node A/B gate (this packet vs untouched 670450638bf1, K8/T50,
4 ranks x 1 case); 1 = 32-node reliability (2 updates + fresh-process
resume); 2 = 8-node K8 and K32 at local 4, two repeats each.
Scope unchanged: one 175-nt RNA, synthetic observations, random frozen head.
Cost and scaling only; no biological or convergence claim.
"""


def main():
    original_manifest = json.loads((BASE / 'frozen_manifest.json').read_text())
    payload = {}
    for name, expected in original_manifest['files'].items():
        data = (BASE / name).read_bytes(); assert sha(data) == expected, name
        payload[name] = data
    payload['batched_full_trajectory_driver.py'] = fixed_batched_driver(payload['batched_full_trajectory_driver.py'].decode()).encode()
    payload['native_fixed_launcher_driver.py'] = fixed_launcher(payload['native_fixed_launcher_driver.py'].decode()).encode()
    site = SITE_LAUNCHER.read_bytes(); assert sha(site) == SITE_LAUNCHER_SHA
    payload['native_fixed_site_limit_launcher.py'] = site
    payload['README_anomaly_off.txt'] = README.encode()
    payload['prepare_anomaly_off_packets.py'] = Path(__file__).read_bytes()
    cfg = dict(original_manifest['configuration'])
    cfg.update({'derived_from_freeze_sha256': original_manifest['freeze_sha256'],
                'change': 'autograd anomaly-detection debug mode disabled after model construction; Workload/update AST-identical',
                'source_driver_unchanged_sha256': None,
                'fixed_driver_sha256': sha(payload['batched_full_trajectory_driver.py']),
                'site_wrapper_sha256': SITE_LAUNCHER_SHA,
                'runs': {'run0_gate': {'nodes': 1, 'ranks': 4, 'K': 8, 'T': 50, 'local_constructs': 1, 'arms': ['anomaly_off_this_packet', 'anomaly_on_original_670450638bf1']},
                         'run2_N8': {'nodes': 8, 'ranks': 32, 'local_constructs': 4, 'accumulation': 4, 'K': [8, 32], 'repeats': 2}}})
    files = {name: sha(data) for name, data in payload.items()}
    freeze = sha(json.dumps({'configuration': cfg, 'files': files}, sort_keys=True).encode())
    native_dir = HERE / ('native_anomaly_off_' + freeze[:12])
    write_packet(native_dir, payload)
    manifest = {key: original_manifest[key] for key in ['source_data_sha256', 'external_dependencies', 'source_licence', 'runtime_mutable_exclusions']}
    manifest.update({'status': 'Prepared locally; no submission or GPU execution', 'configuration': cfg, 'files': files, 'freeze_sha256': freeze})
    native_manifest = (json.dumps(manifest, indent=2) + '\n').encode()
    write_file(native_dir / 'frozen_manifest.json', native_manifest)

    spec = importlib.util.spec_from_file_location('reliability_prep', RELIABILITY / 'prepare_reliability_campaign.py')
    prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
    reliability = dict(payload)
    reliability['production_reliability_driver.py'] = prep.generate_driver(payload['batched_full_trajectory_driver.py'].decode()).encode()
    for name in ['reliability_state.py', 'prepare_reliability_campaign.py', 'test_reliability_state.py',
                 'reliability_local_tests.json', 'verify_reliability_results.py', 'README_reliability.txt']:
        reliability[name] = (RELIABILITY / name).read_bytes()
    reliability['original_frozen_manifest.json'] = native_manifest
    test = json.loads(reliability['reliability_local_tests.json'])
    assert test['passed'] and test['reliability_state_sha256'] == sha(reliability['reliability_state.py'])
    assert FIX in reliability['production_reliability_driver.py'].decode()
    previous = json.loads((RELIABILITY / 'reliability_frozen_aa8af773fd10' / 'campaign_manifest.json').read_text())['configuration']
    rcfg = dict(previous)
    rcfg.update({'original_freeze_sha256': freeze, 'derived_from_reliability_freeze_sha256': 'aa8af773fd10cd410a012108185e66f18442f7562fcd8b5b07ffc6ac201a38b7',
                 'change': 'autograd anomaly-detection debug mode disabled after model construction; Workload/update AST-identical to the qualified reliability driver',
                 'estimated_GPU_hours_from_three_initial_update_timings': 3 * 570 * 32 * 4 / 3600,
                 'estimate_basis': 'Upper end: 157-nt anomaly-off K32 update 459 s x quadratic length factor 1.242',
                 'maximum_reserved_GPU_hours': 192, 'maximum_reserved_node_hours': 48, 'batch_time_limit_seconds': 5400})
    rfiles = {name: sha(data) for name, data in reliability.items()}
    rfreeze = sha(json.dumps({'configuration': rcfg, 'files': rfiles}, sort_keys=True).encode())
    reliability_dir = HERE / ('reliability_anomaly_off_' + rfreeze[:12])
    write_packet(reliability_dir, reliability)
    rmanifest = {'status': 'Prepared locally; no submission or GPU execution', 'configuration': rcfg,
                 'files': rfiles, 'freeze_sha256': rfreeze, 'external_dependencies': original_manifest['external_dependencies']}
    rdata = (json.dumps(rmanifest, indent=2) + '\n').encode()
    write_file(reliability_dir / 'campaign_manifest.json', rdata)
    result = {'status': 'Prepared locally; not uploaded or submitted',
              'native_packet': native_dir.name, 'native_freeze_sha256': freeze, 'native_manifest_sha256': sha(native_manifest),
              'reliability_packet': reliability_dir.name, 'reliability_freeze_sha256': rfreeze, 'reliability_manifest_sha256': sha(rdata),
              'changed_files_vs_670450638bf1': sorted(n for n in files if original_manifest['files'].get(n) != files[n]),
              'AST_identical': ['Workload', 'update'], 'fix': FIX}
    (HERE / 'prepared_packets.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def write_file(destination, data):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): assert destination.read_bytes() == data, destination
    else: destination.write_bytes(data)
    destination.chmod(0o444)


def write_packet(folder, payload):
    for name, data in payload.items(): write_file(folder / name, data)


if __name__ == '__main__':
    main()
