#!/usr/bin/env python3
"""Strict offline qualification of both reliability stages; no SSH or submission."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--accounting', type=Path, required=True)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--expected-restart-count', type=int, default=0)
    parser.add_argument('--account', choices=['e-aif-2026fl01-991', 'e-aif-2026fl01-993'], required=True)
    parser.add_argument('--qualified-input-reports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); manifest = json.loads(args.manifest.read_text()); cfg = manifest['configuration']
    # accounting must be sacct -nP --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes,Account
    accounting = {}
    for line in args.accounting.read_text().splitlines():
        columns = line.strip().split('|')
        if len(columns) >= 6 and columns[0]:
            row = columns[:6]
            assert columns[0] not in accounting, (
                'Duplicate accounting rows require a separate requeue/charge audit; no earlier attempt may be silently overwritten', columns[0])
            accounting[columns[0]] = row
    for key in [str(args.job_id), str(args.job_id) + '.0', str(args.job_id) + '.1']:
        row = accounting[key]
        assert row[1:3] == ['COMPLETED', '0:0'], (key, row)
        assert int(row[4]) == 32
        if key == str(args.job_id): assert row[5] == args.account
        elif row[5]: assert row[5] == args.account
    input_identity = {}
    source_reports = sorted(args.qualified_input_reports.glob('rank_*_micro1_K32_T50.json'))
    assert len(source_reports) == 128
    for path in source_reports:
        record = json.loads(path.read_text()); contract = record['native_fixed_work_contract']
        assert record['world_size'] == 128 and record['case'] == '5zwm_B' and record['length'] == 175
        assert record['official_checkpoint_sha256'] == cfg['official_checkpoint_sha256']
        assert record['frozen_head_sha256'] == cfg['frozen_head_sha256']
        assert record['teacher_fixture_sha256'] == cfg['teacher_fixture_sha256']
        for logical in record['logical_ids']:
            assert logical not in input_identity
            input_identity[logical] = {'observations': contract['observations_by_logical_id'][str(logical)],
                                       'teacher': contract['teacher_by_logical_id'][str(logical)]}
    assert sorted(input_identity) == list(range(128))
    rows = []; report_hashes = {}; total_states = 0
    for rank in range(128):
        paths = {stage: args.results / stage / f'rank_{rank:05d}.json' for stage in ['continuous', 'resume']}
        reports = {stage: json.loads(path.read_text()) for stage, path in paths.items()}
        baseline, resumed = reports['continuous'], reports['resume']
        assert baseline['identity'] == resumed['identity']
        identity = baseline['identity']
        assert identity['campaign_manifest_sha256'] == sha(args.manifest)
        assert identity['slurm_job_id'] == args.job_id
        assert identity['slurm_restart_count'] == args.expected_restart_count
        assert identity['original_freeze_sha256'] == cfg['original_freeze_sha256']
        assert identity['rank'] == rank and identity['world_size'] == 128
        assert identity['case'] == '5zwm_B' and identity['length'] == 175 and identity['seed'] == cfg['seed']
        assert identity['microbatch'] == identity['local_constructs'] == 1 and identity['global_case_instances'] == 128
        assert identity['K'] == 32 and identity['T'] == 50
        assert identity['official_checkpoint_sha256'] == cfg['official_checkpoint_sha256']
        assert identity['fixture_sha256'] == cfg['teacher_fixture_sha256'] and identity['frozen_head_sha256'] == cfg['frozen_head_sha256']
        assert identity['observations_by_logical_id'] == {str(rank): input_identity[rank]['observations']}
        assert identity['teacher_by_logical_id'] == {str(rank): input_identity[rank]['teacher']}
        assert [item['completed_updates'] for item in baseline['updates']] == [1, 2]
        assert [item['completed_updates'] for item in resumed['updates']] == [2]
        assert len(baseline['checkpoint_files']) == 2
        checkpoints = {item['completed_updates']: item for item in baseline['checkpoint_files']}
        restore = resumed['restore_proof']; assert restore['exact_restoration_passed']
        assert restore['checkpoint_file_sha256'] == checkpoints[1]['sha256']
        for kind in ['model', 'optimizer', 'rng', 'scheduler', 'data_cursor']:
            assert restore['restored_' + kind + '_sha256'] == baseline['updates'][0][kind + '_state_sha256' if kind in ['model', 'optimizer', 'rng', 'scheduler'] else 'data_cursor_sha256']
        for stage, record in reports.items():
            assert record['stage'] == stage and record['rank'] == rank and record['world_size'] == 128
            assert record['slurm_job_id'] == args.job_id and record['slurm_step_id'] == ('0' if stage == 'continuous' else '1')
            assert record['status'] == 'Full-objective reliability stage completed'
            assert record['collective_timeout_seconds'] == cfg['site_max_wall_seconds'] == 43200
            report_hashes[str(paths[stage].relative_to(args.results))] = sha(paths[stage])
            for update in record['updates']:
                total_states += 1
                assert update['global_constructs'] == 128 and update['microbatch'] == update['accumulation'] == update['local_constructs'] == 1
                assert update['all_gradients_finite'] and update['frozen_head_gradients_absent']
                assert update['first_microbatch_generated_signal']['generated_inverse_gradient_norm'] > 0
                assert update['first_microbatch_generated_signal']['generated_probe_gradient_norm'] > 0
                assert math.isfinite(update['combined_preclip_gradient_norm']) and update['combined_preclip_gradient_norm'] > 0
                assert math.isfinite(update['rank_max_seconds']) and update['rank_max_seconds'] > 0
                assert update['GPU_peak_reserved_bytes'] < .85 * update['GPU_capacity_bytes']
                assert update['AdamW_step_counter_range'] == [update['completed_updates']] * 2
                counts = update['actual_call_entries']
                assert counts['mean_forward'] == counts['gradient_forward'] == 1600 and counts['native_teacher_forward'] == 1
                assert 0 < counts['backward_recompute'] <= 1600
                assert len(update['groups']) == 1 and update['groups'][0]['logical_ids'] == [rank]
                assert update['groups'][0]['maximum_replay_difference'] <= 1e-5
        comparison = resumed['restart_comparison']; assert comparison['passed']
        for kind in ['model', 'optimizer', 'RNG', 'scheduler', 'data_cursor']:
            item = comparison[kind]; assert item['passed']
            if kind in ['model', 'optimizer']:
                assert item['atol'] == 1e-6 and item['rtol'] == 1e-5
            else: assert item['atol'] == item['rtol'] == 0 and item['bitwise_identical']
        before = baseline['updates'][1]; after = resumed['updates'][0]
        for kind in ['model', 'optimizer']:
            assert comparison[kind]['bitwise_identical'] == (before[kind + '_state_sha256'] == after[kind + '_state_sha256'])
        for expected_group, actual_group in zip(before['groups'], after['groups']):
            assert expected_group['logical_ids'] == actual_group['logical_ids']
            for key in ['finite_K_case_mean_profile_MSE', 'adjacent_C4_mean', 'teacher_loss', 'maximum_replay_difference']:
                assert math.isfinite(actual_group[key]) and math.isfinite(expected_group[key])
                assert math.isclose(actual_group[key], expected_group[key], abs_tol=1e-5, rel_tol=1e-4), ('Independent restart loss mismatch', rank, key)
        rows.append({'rank': rank, 'first_update_seconds': baseline['updates'][0]['rank_max_seconds'],
                     'second_update_seconds': baseline['updates'][1]['rank_max_seconds'],
                     'resumed_second_update_seconds': resumed['updates'][0]['rank_max_seconds'],
                     'model_restart_bitwise_identical': comparison['model']['bitwise_identical'],
                     'optimizer_restart_bitwise_identical': comparison['optimizer']['bitwise_identical'],
                     'model_restart_maximum_absolute_difference': comparison['model']['maximum_absolute_tensor_difference'],
                     'optimizer_restart_maximum_absolute_difference': comparison['optimizer']['maximum_absolute_tensor_difference']})
    assert total_states == 384
    for key in ['first_update_seconds', 'second_update_seconds', 'resumed_second_update_seconds']:
        assert len({row[key] for row in rows}) == 1, 'All-reduced rank maximum differs across rank reports: ' + key
    for stage, index in [('continuous', 0), ('continuous', 1), ('resume', 0)]:
        norms = {json.loads((args.results / stage / f'rank_{rank:05d}.json').read_text())['updates'][index]['combined_preclip_gradient_norm'] for rank in range(128)}
        assert len(norms) == 1, 'DDP-reduced combined gradient norm differs across ranks'
    elapsed = int(accounting[str(args.job_id)][3]); node_hours = elapsed * 32 / 3600
    summary = {'qualified': True, 'status': 'All128 ranks in both application stages passed strict full-objective reliability qualification',
               'job_id': args.job_id, 'account': args.account, 'continuous_rank_reports': 128, 'resumed_rank_reports': 128,
               'accepted_rank_update_reports': total_states, 'manifest_sha256': sha(args.manifest),
               'accounting_sha256': sha(args.accounting), 'report_sha256': report_hashes,
               'rank_max_seconds_by_update': {key: max(row[key] for row in rows) for key in
                                              ['first_update_seconds', 'second_update_seconds', 'resumed_second_update_seconds']},
               'all_final_model_states_bitwise_identical': all(row['model_restart_bitwise_identical'] for row in rows),
               'all_final_optimizer_states_bitwise_identical': all(row['optimizer_restart_bitwise_identical'] for row in rows),
               'parent_allocated_node_hours': node_hours, 'parent_allocated_GPU_hours': node_hours * 4,
               'accounting_scope': 'This qualified completed parent only; any earlier requeued allocated attempts require retained duplicate-aware scheduler history and a separate charge audit',
               'scope': cfg['scope'], 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in ['qualified', 'job_id', 'accepted_rank_update_reports', 'rank_max_seconds_by_update', 'all_final_model_states_bitwise_identical']}))


if __name__ == '__main__': main()
