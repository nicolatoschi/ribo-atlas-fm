#!/usr/bin/env python3
"""RIBO-ATLAS-FM resource plan v2: request derived from MEASURED unit costs x data x design.

Every unit cost is read from retrieved JUPITER analyses (anomaly-off, Waves B-D, 15-16 Oct
2026); design and exposure choices live in a separate config (design_v2.json) so the PI can
change them and re-run. Nothing here is back-solved to a target: the script prints the total
and its breakdown; the call bounds (JUPITER Booster 20,000-90,000 node-hours; inference <=10%)
are checked, not used as inputs. Node-seconds per case = nodes x update seconds / global cases.
"""
import argparse
import json
import math
from pathlib import Path

GPUS_PER_NODE = 4
CALL_MIN_NH, CALL_MAX_NH, INFERENCE_CAP = 20000, 90000, 0.10


def load(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--design', type=Path, required=True)
    p.add_argument('--waveB', type=Path, required=True, help='waveB_analysis.json')
    p.add_argument('--waveC', type=Path, help='waveC analysis JSON with measured K8/K32 microbatch rates (optional)')
    p.add_argument('--waveD', type=Path, required=True, help='Wave D aggregate summary JSON')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    design = load(a.design); B = load(a.waveB); D = load(a.waveD)
    native = B['native']
    unit = {}
    # Warm K8 at microbatch m (fixed-work single node, B4) and fine K32 at microbatch 1 (B1, N32).
    unit['K8_m1'] = native['B0/anomaly_off']['node_seconds_per_case']
    unit['K8_m8'] = native['B4/175_m8']['node_seconds_per_case']
    unit['K32_m1'] = B['cost_basis']['c32_N32']['node_seconds_per_case_mean']
    sources = {'K8_m1': 'Wave B B0 (1 node, local 1)', 'K8_m8': 'Wave B B4 (1 node, local 8, micro 8)',
               'K32_m1': 'Wave B B1 (32 nodes, B128, 3 timed updates)'}
    if a.waveC and a.waveC.exists():
        C = load(a.waveC)
        for key in ['K8_m8', 'K32_m8']:
            if key in C.get('unit_costs', {}):
                unit[key] = C['unit_costs'][key]['node_seconds_per_case']; sources[key] = C['unit_costs'][key]['source']
    if 'K32_m8' not in unit:
        unit['K32_m8'] = unit['K8_m8'] * unit['K32_m1'] / unit['K8_m1']
        sources['K32_m8'] = 'ESTIMATE: K8_m8 x (K32_m1/K8_m1); replace with Wave C C3 measurement'
    curve = B.get('length_curve_K32') or {'exponent': 2.0}
    length_factor = lambda L: (L / 175.0) ** curve['exponent']
    m = design['microbatch']
    c8 = unit[f'K8_m{m}'] * length_factor(design['training_length_nt'])
    c32 = unit[f'K32_m{m}'] * length_factor(design['training_length_nt'])
    rows = []

    def block(name, trajectories, warm_cases, warm_passes, fine_cases, fine_passes, rationale, conditional=False):
        warm = trajectories['warm'] * warm_cases * warm_passes * c8 / 3600
        fine = trajectories['fine'] * fine_cases * fine_passes * c32 / 3600
        rows.append({'block': name, 'node_hours': warm + fine, 'warm_NH': warm, 'fine_NH': fine,
                     'trajectories': trajectories, 'warm_case_updates_per_trajectory': warm_cases * warm_passes,
                     'fine_case_updates_per_trajectory': fine_cases * fine_passes,
                     'optimizer_updates_per_trajectory': {'warm': math.ceil(warm_cases * warm_passes / design['global_batch']),
                                                          'fine': math.ceil(fine_cases * fine_passes / design['global_batch'])},
                     'conditional': conditional, 'rationale': rationale})

    X = design['exposure']; T = design['trajectories']
    block('Matched core (3 arms x seeds)', T['core'], X['warm_cases'], X['warm_passes'], X['fine_cases'], X['fine_passes'], design['rationale']['core'])
    block('Mechanism ablations (fine refinement)', T['ablations'], 0, 0, X['fine_cases'], X['fine_passes'], design['rationale']['ablations'])
    block('Locked reproduction (one seed per arm)', T['reproduction'], X['warm_cases'], X['warm_passes'], X['fine_cases'], X['fine_passes'], design['rationale']['reproduction'])
    for s in design.get('data_scaling', []):
        block(f"Data-scaling arm ({s['fraction']:.0%} of data)", s['trajectories'], X['warm_cases'] * s['fraction'], X['warm_passes'],
              X['fine_cases'] * s['fraction'], X['fine_passes'], design['rationale']['data_scaling'])
    if design.get('expansion'):
        e = design['expansion']
        block('Coverage expansion (admitted additional RMDB deposits)', e['trajectories'], e['added_warm_cases'], e['warm_passes'], 0, 0,
              design['rationale']['expansion'], conditional=True)
    training = sum(r['node_hours'] for r in rows)
    # Inference: measured Wave D K32 generation+refinement node-seconds per case at the training length.
    # Wave D's 127 node-s/case is perfectly packed per-GPU occupancy (fitted); the static-partition
    # measurement job itself packed at 0.22. Production uses a dynamic queue; price at a declared
    # packing efficiency and count development checks per block (pass boundaries actually run).
    nsK32_packed = D['pricing']['node_seconds_per_K32_case_at_177nt']
    I = design['inference']; pack = I['packing_efficiency']; nsK32 = nsK32_packed / pack
    X_ = design['exposure']
    checks = {'core': X_['warm_passes'] + X_['fine_passes'], 'ablations': X_['fine_passes'],
              'reproduction': X_['warm_passes'] + X_['fine_passes'], 'data_scaling': X_['warm_passes'] + X_['fine_passes']}
    n_traj = {'core': T['core']['fine'], 'ablations': T['ablations']['fine'], 'reproduction': T['reproduction']['fine'],
              'data_scaling': sum(s['trajectories']['fine'] for s in design.get('data_scaling', []))}
    dev_checks = I['dev_cases'] * sum(checks[k] * n_traj[k] for k in checks)
    inference_cases = I['final_eval_cohorts'] + dev_checks
    inference = inference_cases * nsK32 / 3600 + I['external_reference_and_preprocessing_NH']
    other = design['fixed']['shared_fitting_NH'] + design['fixed']['operations_NH']
    total = training + inference + other
    plan = {'unit_costs_node_seconds_per_case': unit, 'unit_cost_sources': sources, 'microbatch': m,
            'length_factor_exponent': curve['exponent'], 'c8_used': c8, 'c32_used': c32,
            'blocks': rows, 'training_NH': training,
            'inference': {'cases': inference_cases, 'dev_check_cases': dev_checks, 'checks_per_block': checks, 'packing_efficiency': pack, 'node_seconds_per_K32_case_packed': nsK32_packed, 'node_seconds_per_K32_case': nsK32, 'NH': inference,
                          'share_of_total': inference / total, 'cap': INFERENCE_CAP},
            'shared_fitting_and_operations_NH': other, 'total_NH': total, 'total_GPU_hours': total * GPUS_PER_NODE,
            'call_bounds_NH': [CALL_MIN_NH, CALL_MAX_NH], 'within_call_bounds': CALL_MIN_NH <= total <= CALL_MAX_NH,
            'inference_within_cap': inference / total <= INFERENCE_CAP, 'design': design}
    a.out.write_text(json.dumps(plan, indent=2) + '\n')
    print(f"unit costs (node-s/case): " + ', '.join(f'{k}={v:.2f}' for k, v in unit.items()))
    print(f"using microbatch {m}: c8={c8:.2f}, c32={c32:.2f} (length factor exponent {curve['exponent']:.2f})")
    for r in rows:
        flag = ' [conditional]' if r['conditional'] else ''
        print(f"{r['block']:58s} {r['node_hours']:9,.0f} NH  (updates/traj warm {r['optimizer_updates_per_trajectory']['warm']:,}, fine {r['optimizer_updates_per_trajectory']['fine']:,}){flag}")
    print(f"{'Inference (dev checks + final evaluation + references)':58s} {inference:9,.0f} NH  ({inference/total:.1%} of total; cap 10%)")
    print(f"{'Shared fitting + operations':58s} {other:9,.0f} NH")
    print(f"{'TOTAL':58s} {total:9,.0f} NH = {total*GPUS_PER_NODE:,.0f} GPU-h  (call bounds {CALL_MIN_NH:,}-{CALL_MAX_NH:,} NH: {'OK' if plan['within_call_bounds'] else 'OUT OF BOUNDS'})")
    for k, v in sources.items():
        if v.startswith('ESTIMATE'): print('NOTE', k, v)


if __name__ == '__main__':
    main()
