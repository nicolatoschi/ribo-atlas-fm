#!/usr/bin/env python3
"""Freeze the Wave E packet: code, frozen config, split (rebuilt and compared byte-for-byte),
step_launch.py (verbatim copy), sources record. Writes waveE_<first 12 hex of freeze sha>/ with
read-only files and waveE_manifest.json (sha256 of every file, configuration, freeze sha).
freeze sha = sha256(json.dumps({'configuration': ..., 'files': ...}, sort_keys=True)).
Never uploads or submits anything.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../sequence_prior_waveE/src
TOP = HERE.parent                                # .../sequence_prior_waveE
WORK = TOP.parent                                # .../work
STEP_LAUNCH_SOURCE = WORK / 'geometry_pilot/ensemble_training/anomaly_off_remeasure_20261015/step_launch.py'
STEP_LAUNCH_SHA = 'eb0d567702d9c32d2fc236605591dcf8479eaa81fe5f9325b6566fb5a45e3e63'  # = native_campaign_e8ff330ee775 (Wave B) copy
CODE = ['waveE_mlm.py', 'verify_packet.py', 'build_waveE_split.py', 'analyze_waveE.py', 'smoke_checks.py',
        'setup_waveE_env.sh', 'requirements-waveE.txt', 'waveE_config.json', 'freeze_waveE_packet.py']
ROOT = '/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    payload = {name: (HERE / name).read_bytes() for name in CODE}
    payload['step_launch.py'] = STEP_LAUNCH_SOURCE.read_bytes()
    assert sha(payload['step_launch.py']) == STEP_LAUNCH_SHA, 'step_launch.py differs from the Wave B copy'
    payload['rnafm_sources.json'] = (TOP / 'sources/rnafm_sources.json').read_bytes()
    for name in ['waveE_dataset.tsv', 'waveE_split_summary.json']:
        payload[name] = (TOP / 'staging' / name).read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, '-I', str(HERE / 'build_waveE_split.py'), '--inputs', str(WORK / 'resource_costing'), '--out', tmp],
                       check=True, stdout=subprocess.DEVNULL)
        for name in ['waveE_dataset.tsv', 'waveE_split_summary.json']:
            assert (Path(tmp) / name).read_bytes() == payload[name], f'split rebuild differs: {name}'
    cfg_train = json.loads(payload['waveE_config.json'])
    assert cfg_train['dataset_sha256'] == sha(payload['waveE_dataset.tsv'])
    split = json.loads(payload['waveE_split_summary.json'])
    sources = json.loads(payload['rnafm_sources.json'])
    assert sources['weights']['sha256'] == cfg_train['weights_sha256']
    files = {name: sha(data) for name, data in sorted(payload.items())}
    configuration = {
        'experiment': cfg_train['experiment'],
        'status': 'Prepared locally by Claude 15 Oct 2026; not uploaded, not submitted',
        'approval': 'PI 15 Oct 2026: Wave E funded; RNA-FM code and weights may be downloaded to JUPITER',
        'account': 'e-aif-2026fl01-993', 'partition': 'booster', 'nodes': 64, 'tasks_per_node': 4, 'cpus_per_task': 8, 'gpus_per_node': 4,
        'remote_root': ROOT, 'packet_destination': ROOT + '/waveE_<freeze12>', 'results_root': ROOT + '/results',
        'venv': ROOT + '/waveE_env', 'base_env': '/e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/env',
        'weights_path': ROOT + '/waveE_assets/RNA-FM_pretrained.pth',
        'weights_sha256': cfg_train['weights_sha256'], 'weights_url': sources['weights']['download_url_pinned'],
        'rnafm_wheel_sha256': sources['code']['wheel_sha256'], 'rnafm_github_commit': sources['code']['github_head_commit'],
        'step_launch_source': str(STEP_LAUNCH_SOURCE.relative_to(WORK)), 'step_launch_sha256': files['step_launch.py'],
        'dataset_sha256': files['waveE_dataset.tsv'],
        'split': {k: {x: v[x] for x in ['sequences', 'families', 'groups', 'clans', 'sha256_of_sorted_sequence_ids']} for k, v in split['splits'].items()},
        'plan': {
            'phase1_concurrent': {'P1_N8': 'nodes 0-7, official attention', 'P1_N16': 'nodes 8-23, official', 'P1_N32': 'nodes 24-55, official',
                                  'P1_N8_sdpa': 'nodes 56-63, supplementary: same run with SDPA attention'},
            'phase1_protocol': '10 untimed + 50 timed steps at 8,192 sequences/step, mode scaling, stop after step 60',
            'phase2': 'P2_N64 on all 64 nodes, mode train: eval step 0, steps 1-60 (timing), continue to step 72 (~1 more epoch), eval at epoch boundaries and step 72, save final model',
            'gate': 'phase 2 runs only if at least one primary phase-1 step exits 0',
        },
        'primary_endpoint': cfg_train['primary_endpoint'],
        'secondary_endpoints': cfg_train['secondary_endpoints'],
        'leakage_note': cfg_train['leakage_note'],
        'analysis': 'analyze_waveE.py: efficiency vs P1_N8, tokens/s, HBM/RSS, per-step loss consistency; PRIMARY held-out-family CE change at the first evaluation >= 1.0 epoch (step 12) vs step 0 with family bootstrap (10,000) and group sensitivity; SECONDARY step-72 multi-epoch change, in-family contrast and epoch curve',
    }
    freeze = sha(json.dumps({'configuration': configuration, 'files': files}, sort_keys=True).encode())
    folder = TOP / ('waveE_' + freeze[:12])
    for name, data in payload.items():
        dest = folder / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.read_bytes() == data, f'existing frozen file differs: {name}'
        else:
            dest.write_bytes(data)
        dest.chmod(0o444)
    manifest = {'freeze_sha256': freeze, 'configuration': configuration, 'files': files}
    mpath = folder / 'waveE_manifest.json'
    text = (json.dumps(manifest, indent=2) + '\n').encode()
    if mpath.exists():
        assert mpath.read_bytes() == text
    else:
        mpath.write_bytes(text)
    mpath.chmod(0o444)
    print(json.dumps({'packet': folder.name, 'freeze_sha256': freeze, 'files': len(files), 'step_launch_sha256': files['step_launch.py']}, indent=2))


if __name__ == '__main__':
    main()
