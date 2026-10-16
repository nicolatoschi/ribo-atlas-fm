#!/usr/bin/env python3
"""Prepare, validate and freeze a reliability campaign; never submit it."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'native_fixed_frozen_670450638bf1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def generate_driver(original):
    text = original
    text = replace_once(text, 'from batched_ensemble_core import (case_mean_loss_and_derivative,logical_priors,repeat_native,\n                                   observations,digest_tensor)',
        'from batched_ensemble_core import (case_mean_loss_and_derivative,logical_priors,repeat_native,\n                                   observations,digest_tensor)\nfrom reliability_state import run_reliability, SITE_LIMIT_SECONDS')
    text = replace_once(text, "    p.add_argument('--prepare-fixture',action='store_true');args=p.parse_args()",
        "    p.add_argument('--prepare-fixture',action='store_true')\n    p.add_argument('--stage',choices=['continuous','resume'],required=True)\n    p.add_argument('--state-root',type=Path,required=True)\n    p.add_argument('--campaign-manifest',type=Path,required=True);args=p.parse_args()\n    args.out=args.out.resolve();args.state_root=args.state_root.resolve();args.campaign_manifest=args.campaign_manifest.resolve()\n    assert not args.cpu_proof and not args.prepare_fixture, 'Separate reliability orchestration only'")
    text = replace_once(text, 'timeout=datetime.timedelta(minutes=60)', 'timeout=datetime.timedelta(seconds=SITE_LIMIT_SECONDS)')
    text = replace_once(text, '    target,weights,probe,chemistry=observations(native_c4,single[\'res_mask\'],logical_ids,args.seed)',
        "    target,weights,probe,chemistry=observations(native_c4,single['res_mask'],logical_ids,args.seed)\n    observation_hashes={str(logical):{name:digest_tensor(value[i:i+1]) for name,value in zip(['target_sha256','weights_sha256','probe_sha256','chemistry_sha256'],[target,weights,probe,chemistry])} for i,logical in enumerate(logical_ids)}")
    text = replace_once(text, '    def update(micro,teacher_weight):\n        model.load_state_dict(initial);optimizer=torch.optim.AdamW(trainable,**OmegaConf.to_container(cfg.experiment.optimizer))',
        '    optimizer=torch.optim.AdamW(trainable,**OmegaConf.to_container(cfg.experiment.optimizer))\n    def update(micro,teacher_weight):')
    marker = '    if args.cpu_proof:\n        assert device.type==\'cpu\''
    assert text.count(marker) == 1
    text = text[:text.index(marker)] + """    original_freeze=json.loads(Path(__file__).with_name('original_frozen_manifest.json').read_text())['freeze_sha256']
    context={'length':length,'source_directory':str(source),'observations_by_logical_id':observation_hashes,
             'teacher_by_logical_id':{str(logical):records[logical]['raw_tensor_sha256'] for logical in logical_ids}}
    run_reliability(args,model,optimizer,update,device,rank,world,logical_ids,fixture_path,head_sha,original_freeze,context)


if __name__=='__main__':main()
"""
    original_tree = ast.parse(original); modified_tree = ast.parse(text)
    def nested(tree, name):
        return next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    assert ast.dump(nested(original_tree, 'Workload')) == ast.dump(nested(modified_tree, 'Workload'))
    original_update = nested(original_tree, 'update'); modified_update = nested(modified_tree, 'update')
    # The only update-body alteration removes the per-call initial-state and
    # new-optimizer reset. The AdamW instance now persists between updates.
    original_update.body = original_update.body[2:]
    assert ast.dump(original_update) == ast.dump(modified_update), 'Numerical update body changed'
    return text


def main():
    original_manifest = json.loads((BASE / 'frozen_manifest.json').read_text())
    payload = {}
    for name, expected in original_manifest['files'].items():
        data = (BASE / name).read_bytes(); assert sha(data) == expected, name
        payload[name] = data
    original = payload['batched_full_trajectory_driver.py'].decode()
    generated = generate_driver(original)
    payload['production_reliability_driver.py'] = generated.encode()
    for name in ['reliability_state.py', 'prepare_reliability_campaign.py', 'test_reliability_state.py',
                 'reliability_local_tests.json', 'verify_reliability_results.py', 'README_reliability.txt']:
        payload[name] = (ROOT / name).read_bytes()
    payload['original_frozen_manifest.json'] = (BASE / 'frozen_manifest.json').read_bytes()
    test = json.loads(payload['reliability_local_tests.json'])
    assert test['passed'] and test['reliability_state_sha256'] == sha(payload['reliability_state.py'])
    cfg = {'purpose': 'Two consecutive full K32/T50 updates and fresh-process update-2 checkpoint/restart comparison',
           'original_freeze_sha256': original_manifest['freeze_sha256'], 'case': '5zwm_B', 'length': 175,
           'seed': 20261012, 'nodes': 32, 'ranks': 128, 'GPUs_per_node': 4, 'global_case_instances': 128,
           'microbatch': 1, 'local_constructs': 1, 'K': 32, 'T': 50,
           'continuous_updates': 2, 'resumed_updates': 1, 'recomputed_update_number': 2,
           'original_math_preserved': True, 'full_update_AST_equivalence_except_reset_removal': True,
           'original_workload_class_AST_identical': True,
           'site_max_wall_seconds': 43200, 'collective_timeout_seconds': 43200,
           'official_checkpoint_sha256': original_manifest['configuration']['official_checkpoint_sha256'],
           'teacher_fixture_sha256': original_manifest['configuration']['teacher_fixture_sha256'],
           'frozen_head_sha256': original_manifest['configuration']['expected_frozen_head_sha256'],
           'restart_parameter_tolerance': {'atol': 1e-6, 'rtol': 1e-5},
           'restart_loss_tolerance': {'atol': 1e-5, 'rtol': 1e-4},
           'exact_guards': ['checkpoint model load', 'checkpoint optimizer load', 'checkpoint RNG load',
                            'scheduler/constant learning rate', 'logical data cursor', 'all source and input identities'],
           'estimated_GPU_hours_from_three_initial_update_timings': 3 * 5362.857046860008 * 32 * 4 / 3600,
           'maximum_reserved_GPU_hours': 1536, 'maximum_reserved_node_hours': 384,
           'temporary_checkpoint_storage_estimate_GB': 56,
           'scope': 'One physical RNA, fixed synthetic observations/random frozen head; same allocated nodes in fresh processes; no convergence, biological efficacy or different-node preemption claim'}
    files = {name: sha(data) for name, data in payload.items()}
    freeze = sha(json.dumps({'configuration': cfg, 'files': files}, sort_keys=True).encode())
    folder = ROOT / ('reliability_frozen_' + freeze[:12])
    for name, data in payload.items():
        destination = folder / name; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): assert destination.read_bytes() == data
        else: destination.write_bytes(data)
        destination.chmod(0o444)
    manifest = {'status': 'Prepared locally; no submission or GPU execution', 'configuration': cfg,
                'files': files, 'freeze_sha256': freeze, 'external_dependencies': original_manifest['external_dependencies']}
    data = (json.dumps(manifest, indent=2) + '\n').encode(); path = folder / 'campaign_manifest.json'
    if path.exists(): assert path.read_bytes() == data
    else: path.write_bytes(data)
    path.chmod(0o444)
    remote = '/e/project1/e-aif-2026fl01-991/ribo_production_reliability_20261014/' + folder.name
    python = '/e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/env/bin/python'
    checkpoint = original_manifest['external_dependencies'][0]['path']
    script = ROOT / 'ribo-full-K32-reliability-N32.sbatch'
    text = f'''#!/bin/bash
#SBATCH --job-name=ribo-full-K32-reliability
#SBATCH --account=e-aif-2026fl01-993
#SBATCH --partition=booster
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output={remote}/results/reliability-%j.out
#SBATCH --error={remote}/results/reliability-%j.err
set -euo pipefail
cd {remote}
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
RESULT_DIRECTORY={remote}/results/job_${{SLURM_JOB_ID}}
STATE_DIRECTORY=$RESULT_DIRECTORY/checkpoint_states
mkdir -p "$RESULT_DIRECTORY" "$STATE_DIRECTORY"
srun --kill-on-bad-exit=1 {python} rank_launch.py production_reliability_driver.py --source RNAbpFlow_training --checkpoint {checkpoint} --fixture native_fixed_teacher_fixture.json --out "$RESULT_DIRECTORY" --state-root "$STATE_DIRECTORY" --campaign-manifest {remote}/campaign_manifest.json --case 5zwm_B --microbatch 1 --local-constructs 1 --ensemble-size 32 --grid-points 50 --stage continuous
srun --kill-on-bad-exit=1 {python} rank_launch.py production_reliability_driver.py --source RNAbpFlow_training --checkpoint {checkpoint} --fixture native_fixed_teacher_fixture.json --out "$RESULT_DIRECTORY" --state-root "$STATE_DIRECTORY" --campaign-manifest {remote}/campaign_manifest.json --case 5zwm_B --microbatch 1 --local-constructs 1 --ensemble-size 32 --grid-points 50 --stage resume
'''
    script.write_text(text)
    result = {'status': 'Prepared locally; root must review, upload, verify and perform duplicate audit before submission',
              'frozen_local_directory': str(folder), 'frozen_remote_directory': remote,
              'manifest_sha256': sha(data), 'freeze_sha256': freeze,
              'script': str(script), 'script_sha256': sha(script.read_bytes()),
              'configuration': cfg, 'files': files,
              'verification': {'Workload_AST_identical': True, 'update_AST_identical_except_initial_reset_removal': True,
                               'local_checkpoint_tests_passed': True},
              'qualification_requires': 'Parent plus exact continuous and resume srun application steps COMPLETED0:0; all128 continuous and128 resume reports; 384 accepted update reports, complete source/input/optimizer/RNG/scheduler/data-cursor checks; strict final state/loss tolerance and frozen-head/finite/HBM/replay guards'}
    (ROOT / 'production_reliability_prepared_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['status', 'frozen_local_directory', 'frozen_remote_directory', 'freeze_sha256', 'script']}, indent=2))


if __name__ == '__main__': main()
