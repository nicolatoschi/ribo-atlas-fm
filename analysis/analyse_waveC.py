#!/usr/bin/env python3
"""Offline Wave C analysis: concurrent microbatch-8 groups (C3), 128-node point (C4),
four cross-node reliability trajectories (C1/C2 summary; strict qualification is done
separately with verify_reliability_multistep.py). Emits unit_costs for the v2 resource plan.
No SSH or submission.
"""
import argparse
import json
import statistics as st
from pathlib import Path
from analyse_waveB import native_dir, reliability


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); R = a.results
    groups = {}
    for d in sorted((R / 'C3').iterdir()) if (R / 'C3').exists() else []:
        row = native_dir(d)
        if row: groups[d.name] = row
    out = {'C3_groups': groups, 'unit_costs': {}}
    for K in (8, 32):
        rows = [r for n, r in groups.items() if n.startswith(f'K{K}_m8')]
        if rows:
            v = [r['node_seconds_per_case'] for r in rows]
            out['unit_costs'][f'K{K}_m8'] = {
                'node_seconds_per_case': st.median(v), 'min': min(v), 'max': max(v), 'groups': len(v),
                'source': f'Wave C C3: median of {len(v)} concurrent 8-node trajectories (32 GPUs, micro 8, local 8, B256), one update each',
                'update_seconds': [r['update_seconds'] for r in rows], 'peak_reserved_GiB': max(r['peak_reserved_GiB'] for r in rows)}
    c4 = native_dir(R / 'C4' / 'wide_N128_K32') if (R / 'C4' / 'wide_N128_K32').exists() else None
    out['C4_wide_N128_K32'] = c4
    trajs = {}
    for t in range(4):
        rel = reliability(R / 'C' / f'traj{t}')
        if rel: trajs[f'traj{t}'] = rel
    out['C1_C2_trajectories'] = trajs
    times = [u['update_seconds'] for t in trajs.values() for stage in ('continuous', 'resume') for u in t.get(stage, [])]
    if times:
        out['reliability_update_seconds'] = {'all': times, 'median': st.median(times), 'max': max(times)}
    norms = {t: [u['gradient_norm'] for u in v.get('continuous', [])] for t, v in trajs.items()}
    out['identical_inputs_gradient_norms_across_trajectories'] = norms
    if c4 and times:
        out['weak_scaling_N32_to_N128'] = {'T32_median': st.median(times), 'T128': c4['update_seconds'],
                                           'efficiency': st.median(times) / c4['update_seconds']}
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    for n, r in groups.items():
        print(f"{n:16s} K{r['K']:<3d} m{r['microbatch']} upd {r['update_seconds']:7.1f}s  node-s/case {r['node_seconds_per_case']:6.2f}  HBM {r['peak_reserved_GiB']:.1f} GiB")
    for k, v in out['unit_costs'].items(): print('UNIT', k, json.dumps({x: v[x] for x in ['node_seconds_per_case', 'min', 'max', 'groups']}))
    if c4: print('C4', json.dumps({k: c4[k] for k in ['nodes', 'update_seconds', 'node_seconds_per_case', 'peak_reserved_GiB']}))
    if times: print('C1/C2 update seconds', [round(x, 1) for x in times])
    if 'weak_scaling_N32_to_N128' in out: print('weak scaling 32->128 nodes', json.dumps(out['weak_scaling_N32_to_N128']))
    print('gradient norms by trajectory', json.dumps(norms))


if __name__ == '__main__':
    main()
