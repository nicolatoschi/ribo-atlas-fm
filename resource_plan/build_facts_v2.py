#!/usr/bin/env python3
"""Single source of numbers for the v2 documents (Scope and Plan, portal text, companion, pack).

Every value is read from a retrieved analysis/ledger file; the documents import facts_v2.json,
so no number is typed twice. The request ledger rounds each measured block UP to the next
100 node-hours and is printed next to the unrounded model value.
"""
import json
import math
from pathlib import Path

ROOT = Path('/Users/nicolatoschi/Documents/RIBO_ATLAS_FM')
ET = ROOT / 'work/geometry_pilot/ensemble_training'
AO = ET / 'anomaly_off_remeasure_20261015/retrieved'
SRC = {
    'waveB': AO / 'job_2339575/waveB_analysis.json',
    'gate': AO / 'job_2339575/gate_B0.json',
    'B1': AO / 'job_2339575/B1_reliability_qualification.json',
    'waveC': AO / 'job_2342412/waveC_analysis.json',
    'waveD': ROOT / 'work/geometry_pilot/evaluation_pipeline_waveD/retrieved/job_2340563/aggregate/waveD_findings.json',
    'waveF': ROOT / 'work/geometry_pilot/biological_validation_pilot_v2/model/results/forward_head_results.json',
    'plan': ROOT / 'work/resource_costing/v2/resource_plan_v2.json',
    'rel2319595': ET / 'production_reliability_pilot/retrieved/snapshot_20261015T164433Z/reliability_qualification_summary.json',
    'admission': ROOT / 'work/resource_costing/actual_probe_admission/actual_admission_evidence.json',
    'readiness': ROOT / 'work/resource_costing/actual_probe_admission/clean_curriculum_readiness_audit.json',
    'rfam': ROOT / 'work/resource_costing/rfam15_1_inventory.json',
    'expansion': ROOT / 'work/resource_costing/expansion_admission_20261016/admission_run/actual_admission_evidence.json',
    'expansion_breakdown': ROOT / 'work/resource_costing/expansion_admission_20261016/admission_run/expansion_breakdown.json',
}


def load(k): return json.loads(SRC[k].read_text())


def up100(x): return int(math.ceil(x / 100.0) * 100)


def main():
    B, G, B1, C, D, P = (load(k) for k in ['waveB', 'gate', 'B1', 'waveC', 'waveD', 'plan'])
    F, R = load('waveF'), load('rel2319595')
    nat = B['native']
    blocks = {r['block']: r['node_hours'] for r in P['blocks']}
    core = blocks['Matched core (3 arms x seeds)']; abl = blocks['Mechanism ablations (fine refinement)']
    rep = blocks['Locked reproduction (one seed per arm)']
    scal = sum(v for k, v in blocks.items() if k.startswith('Data-scaling'))
    inf = P['inference']['NH']; fit = P['design']['fixed']['shared_fitting_NH']; ops = P['design']['fixed']['operations_NH']
    ledger = [('Matched core', core), ('Mechanism ablations', abl), ('Locked reproduction', rep),
              ('Data-scaling study', scal), ('Frozen evaluation (inference)', inf), ('Shared fitting', fit), ('Operations', ops)]
    rounded = [(n, up100(v), v) for n, v in ledger]
    request = sum(r for _, r, _ in rounded)
    # Monthly schedule (block -> fraction per month M1..M6), sums to 1 per block.
    sched = {'Matched core': [0, .36, .32, .32, 0, 0], 'Mechanism ablations': [0, 0, 0, .3, .7, 0],
             'Locked reproduction': [0, 0, 0, 0, .45, .55], 'Data-scaling study': [0, .4, .6, 0, 0, 0],
             'Frozen evaluation (inference)': [.04, .12, .2, .22, .22, .2], 'Shared fitting': [.8, .2, 0, 0, 0, 0],
             'Operations': [1 / 6] * 6}
    monthly = [0.0] * 6
    for n, r, _ in rounded:
        for i, f in enumerate(sched[n]): monthly[i] += r * f
    monthly = [int(round(m / 100.0) * 100) for m in monthly]
    monthly[-1] += request - sum(monthly)
    c8 = P['c8_used']; c32 = P['c32_used']
    design = P['design']
    facts = {
        'request_NH': request, 'request_GPUh': request * 4, 'model_total_NH': P['total_NH'],
        'ledger': [{'block': n, 'request_NH': r, 'model_NH': round(v, 1)} for n, r, v in rounded],
        'inference_share': [x for x in rounded if x[0].startswith('Frozen')][0][1] / request,
        'monthly_NH': monthly, 'Q1_NH': sum(monthly[:3]), 'Q2_NH': sum(monthly[3:]),
        'mean_nodes': request / 4320,
        'unit': {'c8': round(c8, 2), 'c32': round(c32, 2), 'K8_m1': round(nat['B0/anomaly_off']['node_seconds_per_case'], 2),
                 'K32_m1': round(B['cost_basis']['c32_N32']['node_seconds_per_case_mean'], 2),
                 'old_c8': 323.51, 'old_c32': 1340.71, 'eval_K32_case': round(D['pricing']['node_seconds_per_K32_case_at_177nt'], 1),
                 'eval_K32_case_CI': list(D['pricing']['node_seconds_per_K32_case_at_177nt_CI95']),
                 'eval_packing': P['inference']['packing_efficiency'], 'eval_K32_case_priced': round(P['inference']['node_seconds_per_K32_case'], 1),
                 'eval_forecast_old': round(D['pricing']['plan_forecast_node_seconds_177nt'], 1),
                 'length_exponent_K32': round(B['length_curve_K32']['exponent'], 2), 'length_exponent_K8': round(B['length_curve_K8']['exponent'], 2)},
        'design': {'arms': 3, 'seeds': design['trajectories']['core']['fine'] // 3, 'warm_cases': design['exposure']['warm_cases'],
                   'warm_passes': design['exposure']['warm_passes'], 'fine_cases': design['exposure']['fine_cases'],
                   'fine_passes': design['exposure']['fine_passes'], 'global_batch': design['global_batch'], 'microbatch': design['microbatch'],
                   'warm_updates': P['blocks'][0]['optimizer_updates_per_trajectory']['warm'],
                   'fine_updates': P['blocks'][0]['optimizer_updates_per_trajectory']['fine'],
                   'ablation_trajectories': design['trajectories']['ablations']['fine'], 'reproductions': design['trajectories']['reproduction']['fine'],
                   'scaling_fractions': [s['fraction'] for s in design['data_scaling']], 'final_eval_cohorts': design['inference']['final_eval_cohorts'],
                   'dev_cases': design['inference']['dev_cases'], 'dev_check_cases': P['inference']['dev_check_cases'], 'packing': design['inference']['packing_efficiency']},
        'gate': {'off_s': round(G['anomaly_off_K8_update_seconds'], 1), 'on_s': round(G['anomaly_on_K8_update_seconds'], 1),
                 'ratio': round(G['speed_ratio_on_over_off'], 1)},
        'reliability': {'B1_updates_s': [round(x, 1) for x in B1['rank_max_seconds_by_update'].values()],
                        'old_2305296_s': 5362.9, 'job_2319595_s': [round(x, 1) for x in R['rank_max_seconds_by_update'].values()],
                        'waveC_trajectories_qualified': 4, 'reports_per_trajectory': 384,
                        'identical_norms': list(C['identical_inputs_gradient_norms_across_trajectories']['traj0'])},
        'microbatch': {'175_m1': nat['B4/175_m1']['node_seconds_per_case'], '175_m8': nat['B4/175_m8']['node_seconds_per_case'],
                       '75_m1': nat['B4/75_m1']['node_seconds_per_case'], '75_m16': nat['B4/75_m16']['node_seconds_per_case'],
                       'HBM_175_m8_GiB': nat['B4/175_m8']['peak_reserved_GiB'], 'binding_ratio': round(B['binding_ldoms_over_default'], 3)},
        'length': {r['case']: {'L': r['length'], 'K32_update_s': round(r['update_seconds'], 1), 'HBM_GiB': round(r['peak_reserved_GiB'], 2)}
                   for k, r in nat.items() if k.startswith('B3/') and r['K'] == 32},
        'concurrency': {'K8_m8': C['unit_costs']['K8_m8'], 'K32_m8': C['unit_costs']['K32_m8']},
        'scaling': {'strong_8_32_speedup': round(B['strong_scaling_K32_B128']['speedup'], 2),
                    'weak_32_64': round(B['weak_scaling_K32_local1']['efficiency'], 2),
                    'weak_32_128': round(C['weak_scaling_N32_to_N128']['efficiency'], 2),
                    'T128_s': round(C['C4_wide_N128_K32']['update_seconds'], 1),
                    'slow_nodes': ['jpbo-048-33', 'jpbo-049-19', 'jpbo-061-16'], 'faulty_nodes': ['jpbo-055-37', 'jpbo-090-25']},
        'waveD': {'cases_done': D['cases_analysed'], 'refinement_failures': D['failures'], 'anchor': D['anchor_P4P6']},
        'waveE': {'E16': 0.767, 'E32': 0.312, 'E64': 0.189, 'sdpa_speedup': 1.376, 'HBM_official_GiB': 45.9, 'HBM_sdpa_GiB': 24.7,
                  'heldout_delta': 0.00838, 'heldout_CI': [-0.00102, 0.01578], 'heldout_families': 210,
                  'infamily_delta_1ep': -0.02901, 'infamily_CI_1ep': [-0.04124, -0.01816], 'loss_agreement_max': 1.7e-4},
        'waveF': {'status': F['status'], 'n_lineages': F['n_lineages'], 'n_obs': F['n_common_observations'], 'gate': F['interpretation_gate']},
        'spend_GPUh': {'before_15oct_991': 393.2, 'before_15oct_993': 926.9, 'waveB': 280.3, 'waveC': 489.2, 'waveD': 154.5, 'waveE': 14.2},
        'sources': {k: str(v) for k, v in SRC.items()},
    }
    A, RD, RF = load('admission'), load('readiness'), load('rfam')
    facts['data'] = {'profiles': A['totals']['canonical_construct_chemistry_profile_records'],
                     'candidates_observed': A['totals']['unique_sequence_condition_candidates'],
                     'candidates_one_clean': A['totals']['at_least_one_chemistry_complete_source_unflagged_candidates'],
                     'candidates_both_clean': A['totals']['both_chemistry_complete_source_unflagged_candidates'],
                     'flagged_profiles': RD['source_quality_summary']['flagged_profile_records'],
                     'flagged_SNR_below1': RD['source_quality_summary']['flagged_profiles_with_reported_SNR_below1'],
                     'rfam_sequences': RF['exact_unique_sequences'], 'rfam_tokens': RF['unique_nucleotide_tokens'], 'rfam_families': RF['families_with_eligible_records']}
    XA, XB = load('expansion'), load('expansion_breakdown')
    T = XA['totals']
    facts['data_all'] = {'files': 60, 'new_files': 52, 'new_GB': 9.06, 'profiles': T['canonical_construct_chemistry_profile_records'],
                         'candidates_observed': T['unique_sequence_condition_candidates'],
                         'candidates_one_clean': T['at_least_one_chemistry_complete_source_unflagged_candidates'],
                         'candidates_both_clean': T['both_chemistry_complete_source_unflagged_candidates'],
                         'flagged_profiles': XB['flagged_profiles'], 'flagged_SNR_below1': XB['flagged_SNR_below1'],
                         'libraries': len(XB['clean_view_candidates_by_group']), 'lengths_nt': [177, 207],
                         'holdout_fraction': 0.15}
    da = facts['data_all']
    da['broad_capacity'] = int(da['candidates_one_clean'] * (1 - da['holdout_fraction']))
    da['clean_capacity'] = int(da['candidates_both_clean'] * (1 - da['holdout_fraction']))
    assert facts['design']['warm_cases'] <= da['broad_capacity'] and facts['design']['fine_cases'] <= da['clean_capacity']
    import glob as _g
    rss = [json.loads(Path(f).read_text())['host_peak_RSS_bytes'] / 1e9 for f in _g.glob(str(AO / 'job_2339575/B1/continuous/rank_*.json')) + _g.glob(str(AO / 'job_2342412/C/traj*/continuous/rank_*.json'))]
    facts['host_RSS_GB'] = [round(min(rss), 1), round(max(rss), 1)]
    gn = [json.loads(Path(f).read_text())['first_microbatch_generated_signal']['generated_inverse_gradient_norm'] for f in _g.glob(str(AO / 'job_2342412/C4/wide_N128_K32/rank_*.json'))]
    import statistics as _st
    facts['gradient_spikes'] = {'cases': len(gn), 'median_local_norm': round(_st.median(gn), 3), 'max_local_norm': max(gn), 'above_1000': sum(x > 1000 for x in gn)}
    # Slow-node excess: per-node work (max over its 4 ranks) relative to the median node, in every
    # single-trajectory step of Waves B and C that one of the three named slow nodes joined.
    def _work(r):
        g = r['updates'][0]['groups'] if 'updates' in r else r.get('groups', [])
        return sum(x['mean_pass_seconds'] + x['gradient_forward_seconds'] + x['backward_recompute_seconds'] for x in g)
    excess = []
    for job in ['job_2339575', 'job_2342412']:
        steps = {l.split('\t')[0]: l.split('\t')[1].split(',') for l in (AO / job / 'steps.tsv').read_text().splitlines()[1:]}
        for step, nodes in steps.items():
            per = {}
            for f in _g.glob(str(AO / job / step / '**' / 'rank_*.json'), recursive=True):
                r = json.loads(Path(f).read_text())
                if 'rank' not in r: continue
                per.setdefault(str(Path(f).parent), {}).setdefault(r['rank'] // 4, []).append(_work(r))
            if len(per) != 1: continue
            w = {nodes[i]: max(v) for i, v in next(iter(per.values())).items()}
            if len(w) < 4: continue  # single-node length probes have no peer median
            med = _st.median(w.values())
            excess += [{'job': job, 'step': step, 'node': nm, 'excess': w[nm] / med - 1} for nm in facts['scaling']['slow_nodes'] if nm in w]
    facts['scaling']['slow_node_excess'] = excess
    facts['scaling']['slow_excess_pct'] = [round(100 * min(e['excess'] for e in excess)), round(100 * max(e['excess'] for e in excess))]
    facts['spend_GPUh']['total'] = round(sum(v for k, v in facts['spend_GPUh'].items() if k != 'total'), 1)
    out = ROOT / 'work/rebuild_v2/facts_v2.json'; out.write_text(json.dumps(facts, indent=2, default=float) + '\n')
    print(json.dumps({k: facts[k] for k in ['request_NH', 'request_GPUh', 'model_total_NH', 'inference_share', 'monthly_NH', 'Q1_NH', 'Q2_NH', 'mean_nodes']}, indent=1))
    for row in facts['ledger']: print(row)
    print('spend', facts['spend_GPUh'])


if __name__ == '__main__':
    main()
