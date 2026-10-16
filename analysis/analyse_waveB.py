#!/usr/bin/env python3
"""Offline Wave B analysis: per-step costs, length curve, scaling, microbatch, re-costing.

Input: a retrieved snapshot of results/job_<id> (reports, steps.tsv). Output: JSON +
human-readable table. Node-seconds per case = nodes x all-reduced rank-max update
seconds / global case instances. Budget re-costing uses the unchanged ledger formula
[12*Swarm*c8 + 21*Sfine*c32]/3600 with Swarm=8064, Sfine=8192 (final_resource_plan).
Missing steps are reported as missing, never imputed. No SSH or submission.
"""
import argparse
import glob
import json
import math
import statistics as st
from pathlib import Path

PHASES = ['mean_pass_seconds', 'gradient_forward_seconds', 'backward_recompute_seconds', 'teacher_forward_backward_seconds']
SWARM, SFINE = 8064, 8192


def native_dir(path):
    files = sorted(glob.glob(str(path / 'rank_*_micro*_K*_T50.json')))
    if not files: return None
    reports = [json.loads(Path(f).read_text()) for f in files]
    r0 = reports[0]
    world = r0['world_size']; assert len(reports) == world, (path, len(reports), world)
    seconds = {r['rank_max_seconds'] for r in reports}; assert len(seconds) == 1, path
    t = r0['rank_max_seconds']; nodes = world // 4; cases = r0['global_constructs']
    phases = {k: st.median(sum(g[k] for g in r['groups']) for r in reports) for k in PHASES}
    return {'case': r0['case'], 'length': r0['length'], 'K': r0['K_train'], 'microbatch': r0['microbatch'],
            'local_constructs': r0['local_constructs'], 'nodes': nodes, 'world': world, 'global_cases': cases,
            'update_seconds': t, 'node_seconds_per_case': nodes * t / cases,
            'GPU_seconds_per_case': 4 * nodes * t / cases,
            'peak_reserved_GiB': max(r['GPU_peak_reserved_bytes'] for r in reports) / 2**30,
            'anomaly_enabled': r0.get('autograd_anomaly_enabled', 'not recorded (original code)'),
            'gradient_norm': r0['combined_preclip_gradient_norm'], 'median_phase_seconds': phases}


def reliability(path):
    out = {}
    for stage in ['continuous', 'resume']:
        files = sorted(glob.glob(str(path / stage / 'rank_*.json')))
        if not files: continue
        reports = [json.loads(Path(f).read_text()) for f in files]
        out[stage] = [{'completed_updates': u['completed_updates'], 'update_seconds': u['rank_max_seconds'],
                       'gradient_norm': u['combined_preclip_gradient_norm'],
                       'node_seconds_per_case': 32 * u['rank_max_seconds'] / 128} for u in reports[0]['updates']]
        out[stage + '_ranks'] = len(reports)
    return out or None


def fit_power(points):
    if len(points) < 2: return None
    xs = [math.log(L) for L, _ in points]; ys = [math.log(t) for _, t in points]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = math.exp(my - b * mx)
    return {'a': a, 'exponent': b, 'points': points}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); R = args.results
    steps = {}
    ledger = R / 'steps.tsv'
    if ledger.exists():
        for line in ledger.read_text().splitlines()[1:]:
            cols = line.split('\t'); steps[cols[0]] = {'nodes_cap_min': cols[2], 'start': cols[3], 'end': cols[4], 'exit': cols[5]}
    native = {}
    for d in sorted({Path(f).parent for f in glob.glob(str(R / '**' / 'rank_*_micro*_K*_T50.json'), recursive=True)}):
        if d.name in ('continuous', 'resume'): continue
        row = native_dir(d)
        if row: native[str(d.relative_to(R))] = row
    rel = reliability(R / 'B1')
    summary = {'steps': steps, 'native': native, 'reliability_B1': rel}
    # Gate
    off, on = native.get('B0/anomaly_off'), native.get('B0/anomaly_on')
    if off and on:
        summary['gate'] = {'speed_ratio_on_over_off': on['update_seconds'] / off['update_seconds'],
                           'gradient_norm_equal': off['gradient_norm'] == on['gradient_norm'],
                           'gradient_norm_rel_diff': abs(off['gradient_norm'] - on['gradient_norm']) / on['gradient_norm']}
    ld = native.get('B0/anomaly_off_ldoms')
    if off and ld: summary['binding_ldoms_over_default'] = ld['update_seconds'] / off['update_seconds']
    # Cost basis at production layouts
    c32 = None
    if rel and rel.get('continuous'):
        times = [u['update_seconds'] for u in rel['continuous']] + [u['update_seconds'] for u in rel.get('resume', [])]
        c32 = {'source': 'B1 N32/K32/local1, all timed updates', 'update_seconds': times,
               'node_seconds_per_case_mean': 32 * st.mean(times) / 128, 'node_seconds_per_case_max': 32 * max(times) / 128}
    k8 = [native[k] for k in native if k.startswith('B2/') and k.endswith('_K8')]
    k32n8 = [native[k] for k in native if k.startswith('B2/') and k.endswith('_K32')]
    c8 = {'source': 'B2 N8/K8/local4 replicas', 'values': [r['node_seconds_per_case'] for r in k8]} if k8 else None
    if c8: c8['mean'] = st.mean(c8['values']); c8['max'] = max(c8['values'])
    summary['cost_basis'] = {'c32_N32': c32, 'c8_N8': c8,
                             'c32_N8': {'values': [r['node_seconds_per_case'] for r in k32n8]} if k32n8 else None,
                             'previous_anomaly_on_basis': {'c32_N32': 1340.714262, 'c8_N8': 323.511384, 'c32_N8': 1304.791795}}
    # Strong scaling K32: N8 local4 vs N32 local1, same B128 work
    if c32 and k32n8:
        t8 = st.mean(r['update_seconds'] for r in k32n8); t32 = st.mean(c32['update_seconds'])
        summary['strong_scaling_K32_B128'] = {'T8': t8, 'T32': t32, 'speedup': t8 / t32, 'efficiency': t8 / t32 / 4}
    # Weak scaling K32 local1: N32 (B1) vs N64 (P2)
    p2 = native.get('P2/weak_N64_K32')
    if c32 and p2:
        summary['weak_scaling_K32_local1'] = {'T32': st.mean(c32['update_seconds']), 'T64': p2['update_seconds'],
                                             'efficiency': st.mean(c32['update_seconds']) / p2['update_seconds']}
    # Length curve (per GPU-case seconds at local 1)
    for K in (8, 32):
        pts = sorted({(r['length'], r['GPU_seconds_per_case']) for k, r in native.items()
                      if k.startswith('B3/') and r['K'] == K and r['local_constructs'] == 1})
        if K == 32 and c32: pts = sorted(set(pts) | {(175, 4 * c32['node_seconds_per_case_mean'])})
        summary[f'length_curve_K{K}'] = fit_power(pts)
    # Microbatch
    summary['microbatch'] = {k: {'case': r['case'], 'microbatch': r['microbatch'], 'GPU_seconds_per_case': r['GPU_seconds_per_case'],
                                 'peak_reserved_GiB': r['peak_reserved_GiB']} for k, r in native.items() if k.startswith('B4/')}
    # Re-costing with the unchanged curriculum formula
    if c32 and c8:
        curve = summary.get('length_curve_K32')
        f177 = (177 / 175) ** curve['exponent'] if curve else (177 / 175) ** 2
        for label, a8, a32 in [('mean', c8['mean'], c32['node_seconds_per_case_mean']), ('max', c8['max'], c32['node_seconds_per_case_max'])]:
            warm = 12 * SWARM * a8 * f177 / 3600; fine = 21 * SFINE * a32 * f177 / 3600
            summary.setdefault('recost_current_curriculum', {})[label] = {'length_factor_177': f177, 'warm_NH': warm, 'fine_NH': fine,
                                                                          'trajectory_NH': warm + fine, 'previous_trajectory_NH': 74436.94}
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(f"{'step':34s} {'L':>4s} {'K':>3s} {'m':>3s} {'loc':>4s} {'N':>4s} {'upd_s':>9s} {'nodes_s/case':>12s} {'GiB':>6s}")
    for k, r in native.items():
        print(f"{k:34s} {r['length']:4d} {r['K']:3d} {r['microbatch']:3d} {r['local_constructs']:4d} {r['nodes']:4d} {r['update_seconds']:9.1f} {r['node_seconds_per_case']:12.2f} {r['peak_reserved_GiB']:6.2f}")
    for key in ['gate', 'binding_ldoms_over_default', 'strong_scaling_K32_B128', 'weak_scaling_K32_local1', 'recost_current_curriculum']:
        if key in summary: print(key, json.dumps(summary[key]))
    for K in (8, 32):
        c = summary.get(f'length_curve_K{K}')
        if c: print(f'length curve K{K}: GPU-s/case = {c["a"]:.3g} * L^{c["exponent"]:.2f} over {len(c["points"])} points')


if __name__ == '__main__':
    main()
