#!/usr/bin/env python3
"""Offline Run-0 gate: anomaly-off vs anomaly-on on the same node; no SSH or submission.

Pass requires: 4+4 completed reports, anomaly disabled in the fixed arm, identical
inputs, identical DDP-reduced gradient norms and losses, and a >=5x speed ratio.
Also projects Run 1/2 wall times from the measured anomaly-off K8 update.
"""
import argparse
import json
import math
from pathlib import Path

RUN1_LIMIT, RUN2_LIMIT, SETUP = 5400, 10800, 300


def close(a, b, atol, rtol):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    arms = {}
    for arm in ['anomaly_off', 'anomaly_on']:
        files = sorted((args.results / arm).glob('rank_*_micro1_K8_T50.json'))
        assert len(files) == 4, (arm, len(files))
        arms[arm] = [json.loads(f.read_text()) for f in files]
    checks, rows = [], []
    for off, on in zip(arms['anomaly_off'], arms['anomaly_on']):
        assert off['rank'] == on['rank'] and off['logical_ids'] == on['logical_ids']
        assert off['autograd_anomaly_enabled'] is False and 'autograd_anomaly_enabled' not in on
        assert off['native_fixed_work_contract']['source_driver_unchanged'] is False
        assert on['native_fixed_work_contract']['source_driver_unchanged'] is True
        assert off['native_fixed_work_contract']['observations_by_logical_id'] == on['native_fixed_work_contract']['observations_by_logical_id']
        assert off['native_fixed_work_contract']['teacher_by_logical_id'] == on['native_fixed_work_contract']['teacher_by_logical_id']
        for key in ['official_checkpoint_sha256', 'frozen_head_sha256', 'teacher_fixture_sha256']:
            assert off[key] == on[key], key
        assert off['actual_call_entries'] == on['actual_call_entries']
        checks.append(close(off['combined_preclip_gradient_norm'], on['combined_preclip_gradient_norm'], 0, 1e-6))
        for key in ['generated_inverse_gradient_norm', 'generated_probe_gradient_norm']:
            checks.append(close(off['first_microbatch_generated_signal'][key], on['first_microbatch_generated_signal'][key], 0, 1e-6))
        for a, b in zip(off['groups'], on['groups']):
            for key in ['finite_K_case_mean_profile_MSE', 'adjacent_C4_mean', 'teacher_loss']:
                checks.append(close(a[key], b[key], 1e-5, 1e-4))
        rows.append({'rank': off['rank'], 'off_seconds': off['rank_max_seconds'], 'on_seconds': on['rank_max_seconds'],
                     'off_norm': off['combined_preclip_gradient_norm'], 'on_norm': on['combined_preclip_gradient_norm'],
                     'norm_bitwise_equal': off['combined_preclip_gradient_norm'] == on['combined_preclip_gradient_norm'],
                     'off_phases': {k: off['groups'][0][k] for k in ['mean_pass_seconds', 'gradient_forward_seconds', 'backward_recompute_seconds', 'teacher_forward_backward_seconds']},
                     'on_phases': {k: on['groups'][0][k] for k in ['mean_pass_seconds', 'gradient_forward_seconds', 'backward_recompute_seconds', 'teacher_forward_backward_seconds']}})
    t_off = max(r['off_seconds'] for r in rows); t_on = max(r['on_seconds'] for r in rows); ratio = t_on / t_off
    # K32 at local 1 ~ 4x the K8 member work; Run 1 = 3 timed K32 updates; Run 2 = 2x(4 cases K8) + 2x(4 cases K32).
    run1 = SETUP * 2 + 3 * 4 * t_off
    run2 = SETUP * 4 + 2 * 4 * t_off + 2 * 16 * t_off
    numerics = all(checks)
    passed = numerics and ratio >= 5
    summary = {'gate_passed': passed, 'numerics_identical_within_tolerance': numerics, 'speed_ratio_on_over_off': ratio,
               'anomaly_off_K8_update_seconds': t_off, 'anomaly_on_K8_update_seconds': t_on,
               'projected_run1_seconds': run1, 'run1_limit_seconds': RUN1_LIMIT, 'run1_fits_75pct': run1 <= .75 * RUN1_LIMIT,
               'projected_run2_seconds': run2, 'run2_limit_seconds': RUN2_LIMIT, 'run2_fits_75pct': run2 <= .75 * RUN2_LIMIT,
               'criteria': 'identical inputs; gradient norms rel<=1e-6; losses atol1e-5/rtol1e-4; ratio>=5', 'rows': rows}
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in summary if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
