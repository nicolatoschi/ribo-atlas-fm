#!/usr/bin/env python3
"""Wave D preparation step 3: assemble, hash, freeze and make read-only the packet,
then render the sbatch (outside the packet) and the submission manifest.

freeze_sha256 = sha256 of compact sorted JSON {"configuration": CONFIG, "files": {relpath: sha256}}
over every packet file except FROZEN_MANIFEST.json. Packet directory = waveD_<first 12 hex>.
Nothing is submitted; no remote access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REMOTE_ROOT = '/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015'
CONFIG = {
    'wave': 'D', 'campaign': 'RIBO-ATLAS-FM evaluation-pipeline cost and robustness across RNA length (PI-approved)',
    'account': 'e-aif-2026fl01-993', 'partition': 'booster', 'nodes': 32, 'tasks_per_node': 4, 'cpus_per_task': 8,
    'gres': 'gpu:4', 'time_limit': '01:30:00', 'srun_step_minutes': 84, 'soft_deadline_margin_minutes': 14,
    'hard_deadline_margin_minutes': 4, 'exclude': 'jpbo-055-37', 'launch_pattern': 'one srun per node (node agent), static per-GPU case lists',
    'remote_python': '/e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/env/bin/python',
    'remote_checkpoint': '/e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/pretrained_checkpoint/RNA3DB.ckpt',
    'checkpoint_sha256': 'bb598a736193734541f63c59dae75d45db8c04f01a8414a758677367c4b1e213',
    'remote_packet_root': REMOTE_ROOT, 'remote_results_root': REMOTE_ROOT + '/results',
    'K': 32, 'timesteps': 50, 'seed_rule': '420*(sample_id+1)', 'refined_subset_rule': 'sample IDs 0..R(L)-1, see partition_plan.json',
    'stage_cap_base_seconds_at_158nt': 240.0, 'stage_cap_scaling': 'max(1,(L/158)^2)', 'openmm_platform': 'CUDA',
    'openmm_version': '8.4', 'forcefield_hashes': {'amber14/RNA.OL3.xml': '61b793a49ded3c9c3bd9fbab328bc1b20b2b4ce7bc2358d537892ca64b085127',
                                                   'implicit/gbn2.xml': 'd12c14820a413146eb008758c405b8690898811b4e08827b00a75a163395bde3'},
    'upstream_commit': 'e8b1c077f7951d4d6ebe00c851d22c37777dfc9f', 'inference_packet': 'inference_frozen_6cbee47ef9de',
    'refinement_packet': 'refinement_frozen_e665e8ed11ae',
    'scope': 'Engineering cost/robustness measurement. Cases are RNAbpFlow training cases: lDDT is pipeline QA, not held-out accuracy.'}
COPY_FROM_INFERENCE = ['environment_freeze.txt', 'pretrained_checkpoint_manifest.json', 'PORTABILITY_AND_LICENSE.txt',
                       'portability.diff', 'source_manifest.json', 'FROZEN_MANIFEST.json', 'README_pretrained_inference.txt']
COPY_FROM_REFINEMENT = ['refinement_manifest.json', 'README_refinement_engineering.txt', 'FROZEN_MANIFEST.json']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deliverable', type=Path, required=True)
    parser.add_argument('--geometry-pilot', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--native-arrays', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, required=True, help='hpc_pilot/rna_backbone_metrics.py')
    parser.add_argument('--pilot-1gid-reference', type=Path, required=True)
    args = parser.parse_args()
    D = args.deliverable.resolve(); G = args.geometry_pilot.resolve()
    inference = G / 'inference_frozen_6cbee47ef9de'; refinement = G / 'refinement_frozen_e665e8ed11ae'
    stage = D / '_staging_packet'
    if stage.exists():
        raise SystemExit(f'{stage} exists; remove it first')
    if list(D.glob('waveD_*/FROZEN_MANIFEST.json')):
        raise SystemExit('a frozen packet already exists in the deliverable; refusing to overwrite')
    subprocess.run([sys.executable, '-I', str(D / 'prepare/select_cases_and_plan.py'), '--audit', str(args.audit),
                    '--native-arrays', str(args.native_arrays), '--archive', str(G / 'RNAbpFlow-Train.tar.gz'),
                    '--inference-packet', str(inference), '--native-1gid', str(refinement / 'inputs/1GID.pdb'),
                    '--pilot-1gid-reference', str(args.pilot_1gid_reference), '--packet', str(stage)], check=True)
    shutil.copytree(inference / 'RNAbpFlow', stage / 'RNAbpFlow')
    for name in ('waveD_driver.py', 'waveD_refine.py', 'verify_packet.py', 'aggregate_waveD.py', 'waveD_eval_cost.sbatch.template'):
        shutil.copyfile(D / 'src' / name, stage / name)
    shutil.copyfile(refinement / 'openmm_rna_refinement.py', stage / 'openmm_rna_refinement.py')
    shutil.copyfile(args.metrics, stage / 'rna_backbone_metrics.py')
    shutil.copyfile(G / 'ensemble_training/anomaly_off_remeasure_20261015/step_launch.py', stage / 'step_launch.py')
    (stage / 'provenance/inference_packet').mkdir(parents=True); (stage / 'provenance/refinement_packet').mkdir(parents=True)
    for name in COPY_FROM_INFERENCE:
        shutil.copyfile(inference / name, stage / 'provenance/inference_packet' / name)
    for name in COPY_FROM_REFINEMENT:
        shutil.copyfile(refinement / name, stage / 'provenance/refinement_packet' / name)
    (stage / 'provenance/prepare').mkdir()
    for name in ('audit_training_cases.py', 'select_cases_and_plan.py', 'build_packet.py', 'check_refinement_equivalence.py'):
        shutil.copyfile(D / 'prepare' / name, stage / 'provenance/prepare' / name)
    shutil.copyfile(args.audit, stage / 'provenance/training_case_audit.json')
    shutil.copyfile(D / 'src/README_packet.txt', stage / 'README_packet.txt')

    # byte-identity checks against the frozen sources --------------------------------
    inf_manifest = json.loads((inference / 'FROZEN_MANIFEST.json').read_text())['core_payload_files']
    import zipfile
    upstream_zip = G / 'RNAbpFlow-e8b1c07.zip'
    assert digest(upstream_zip) == '193a01c7f3e54be7710af83ab4e2dbf9fe4f95ff82a2fea4f566da9bb650ff08'
    prefix = 'RNAbpFlow-e8b1c077f7951d4d6ebe00c851d22c37777dfc9f/'
    with zipfile.ZipFile(upstream_zip) as z:
        pristine = {n[len(prefix):]: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')}
    identity = {}
    for path in sorted(p for p in (stage / 'RNAbpFlow').rglob('*') if p.is_file()):
        rel = str(path.relative_to(stage)); sha = digest(path); inner = str(path.relative_to(stage / 'RNAbpFlow'))
        identity[rel] = {'matches_inference_packet_manifest': inf_manifest.get(rel) == sha,
                         'matches_pristine_upstream_zip_e8b1c07': (pristine[inner] == sha) if inner in pristine else None}
    differing = sorted(k for k, v in identity.items() if v['matches_pristine_upstream_zip_e8b1c07'] is False)
    assert differing == ['RNAbpFlow/src/data/utils.py'], differing  # the documented one-line torch_scatter port
    singles = {'openmm_rna_refinement.py': digest(refinement / 'openmm_rna_refinement.py'),
               'rna_backbone_metrics.py': digest(args.metrics),
               'step_launch.py': digest(G / 'ensemble_training/anomaly_off_remeasure_20261015/step_launch.py')}
    for rel, sha in singles.items():
        identity[rel] = {'matches_source': digest(stage / rel) == sha, 'source_sha256': sha}
    assert all(v.get('matches_inference_packet_manifest', True) is not False and v.get('matches_source', True) for v in identity.values())
    assert digest(stage / 'openmm_rna_refinement.py') == '219f02670dda0dea74c1457eb9c13352c0960fd3478c221fea84da15cc37ee69'
    (stage / 'provenance/byte_identity_checks.json').write_text(json.dumps(identity, indent=1) + '\n')
    plan = json.loads((stage / 'partition_plan.json').read_text())

    files = {str(p.relative_to(stage)): {'sha256': digest(p), 'bytes': p.stat().st_size}
             for p in sorted(stage.rglob('*')) if p.is_file()}
    freeze = hashlib.sha256(json.dumps({'configuration': CONFIG, 'files': {k: v['sha256'] for k, v in files.items()}},
                                       sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    name = 'waveD_' + freeze[:12]
    manifest = {'packet': name, 'freeze_sha256': freeze,
                'freeze_definition': 'sha256 of compact sorted JSON {"configuration": configuration, "files": {relpath: sha256}}; excludes this manifest',
                'configuration': CONFIG, 'files': files, 'file_count': len(files),
                'total_bytes': sum(v['bytes'] for v in files.values()),
                'cases': len(json.loads((stage / 'case_table.json').read_text())['cases']),
                'estimated_total_gpu_seconds': plan['estimated_total_gpu_seconds'],
                'estimated_max_gpu_slot_seconds': plan['estimated_max_slot_seconds'],
                'job_submitted': False, 'remote_destination': f'{REMOTE_ROOT}/{name}'}
    (stage / 'FROZEN_MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    final = D / name
    os.rename(stage, final)
    for p in final.rglob('*'):
        if p.is_file():
            p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    # sbatch outside the packet ------------------------------------------------------
    template = (final / 'waveD_eval_cost.sbatch.template').read_text()
    sbatch_text = template.replace('@PACKET_NAME@', name).replace('@FREEZE_SHA@', freeze)
    assert '@PACKET_NAME@' not in sbatch_text and '@FREEZE_SHA@' not in sbatch_text
    (D / 'sbatch').mkdir(exist_ok=True)
    sbatch = D / 'sbatch/waveD_eval_cost.sbatch'
    sbatch.write_text(sbatch_text); sbatch.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    submission = {'packet': name, 'freeze_sha256': freeze, 'packet_manifest_sha256': digest(final / 'FROZEN_MANIFEST.json'),
                  'sbatch': str(sbatch.relative_to(D)), 'sbatch_sha256': digest(sbatch),
                  'remote_packet_destination': f'{REMOTE_ROOT}/{name}/', 'remote_results': f'{REMOTE_ROOT}/results/job_<JOBID>/',
                  'account': CONFIG['account'], 'nodes': CONFIG['nodes'], 'time_limit': CONFIG['time_limit'],
                  'maximum_node_hours': 32 * 1.5, 'maximum_GPU_hours': 32 * 4 * 1.5,
                  'estimated_total_gpu_seconds_planning_model': plan['estimated_total_gpu_seconds'],
                  'estimated_max_gpu_slot_seconds_planning_model': plan['estimated_max_slot_seconds'],
                  'submitted': False, 'submitted_by': None}
    (D / 'waveD_submission_manifest.json').write_text(json.dumps(submission, indent=1) + '\n')
    print(json.dumps(submission, indent=1))


if __name__ == '__main__':
    main()
