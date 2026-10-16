#!/usr/bin/env python3
"""Wave D aggregation: per-case JSONs -> cost vs length, fitted cost model, pricing,
validity/lDDT summaries, failures and the P4-P6 anchor cross-check.

All structural metrics are on RNAbpFlow TRAINING cases: they are PIPELINE QA,
not held-out accuracy. Costs are engineering measurements on JUPITER GH200.

Cost definitions (per case, one GPU, sequential):
  gen_per_sample      = mean published predict_step seconds (sampler + all-atom + PDB write)
  ref_per_sample      = mean refinement worker seconds (frozen protocol complete_case_wall),
                        first refinement on each rank excluded (OpenMM kernel compile)
  occupancy_K32       = input prep + 32/K_ok * generation-loop wall (incl. panel/lDDT)
                        + 32 * mean parent-observed refinement wall per refined sample
  node_seconds_K32    = occupancy_K32 / 4   (GPU-seconds on a 4-GPU node, perfectly packed)
The refined subset R(L) <= 32 is extrapolated linearly to 32 samples (independent,
sequential minimisations); the extrapolation is labelled in every output.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SCOPE = 'Pipeline QA on RNAbpFlow training cases (anchor P4-P6 family also in training); NOT held-out accuracy.'
PILOT = {'generation_4gpu_cohort_wall_s_158nt': 14.114765, 'refinement_4worker_cohort_wall_s_158nt': 279.275999,
         'generation_1gpu_sampler_sum_s_158nt': 56.04009600601057, 'refinement_case_wall_sum_s_158nt': 969.6,
         'plan_length_factor_177': (177 / 158) ** 2, 'plan_full_cohorts': 25500, 'plan_dev_cohorts': 75600,
         'plan_dev_seconds_per_cohort': 22.0, 'plan_inference_forecast_NH': 3070.05, 'plan_frozen_evaluation_cap_NH': 5000.0}
EVAL_LENGTHS = [30, 50, 75, 100, 130, 158, 177, 200, 250, 300, 400, 500, 600, 720]


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def q(v, p):
    v = [x for x in v if x is not None]
    return float(np.percentile(v, p)) if v else None


def summary(v):
    v = [x for x in v if x is not None]
    if not v:
        return {'n': 0}
    return {'n': len(v), 'mean': float(np.mean(v)), 'median': float(np.median(v)), 'p25': q(v, 25), 'p75': q(v, 75),
            'min': float(np.min(v)), 'max': float(np.max(v))}


# ----------------------------------------------------------------------------- fitting

def nnls2(x1, x2, y):
    """min ||a*x1 + b*x2 - y|| with a,b >= 0 (2-parameter closed form)."""
    A = np.stack([x1, x2], 1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    if (sol >= 0).all():
        return sol
    best = None
    for idx in (0, 1):
        col = A[:, idx]; c = max(0.0, float(col @ y / (col @ col))) if col @ col > 0 else 0.0
        s = np.zeros(2); s[idx] = c; r = float(np.sum((A @ s - y) ** 2))
        if best is None or r < best[0]:
            best = (r, s)
    return best[1]


def fit_power(L, y, grid=np.round(np.arange(0.5, 3.0001, 0.02), 2)):
    """y = a + b * (L/158)^p, a,b >= 0, p on a grid, minimising RELATIVE squared error
    (rows weighted by 1/y) so that the many short cases, which dominate pool-weighted
    pricing, are fitted as well as the long ones."""
    L = np.asarray(L, float); y = np.asarray(y, float)
    if len(L) < 3 or np.ptp(L) == 0 or not (y > 0).all():
        return None
    w = 1.0 / y
    best = None
    for p in grid:
        x = (L / 158.0) ** p; a, b = nnls2(w, x * w, np.ones_like(y))
        sse = float(np.sum(((a + b * x - y) * w) ** 2))
        if best is None or sse < best['sse']:
            best = {'a': float(a), 'b': float(b), 'p': float(p), 'sse': sse}
    resid = y - (best['a'] + best['b'] * (L / 158.0) ** best['p'])
    best['rmse'] = float(np.sqrt(np.mean(resid ** 2)))
    best['r2'] = float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)) if np.ptp(y) > 0 else None
    best['median_abs_relative_error'] = float(np.median(np.abs(resid) / np.maximum(np.abs(y), 1e-9)))
    best['n'] = int(len(L)); best['form'] = 'a + b*(L/158)^p, a,b>=0, relative-error least squares'
    return best


def fit_quadratic(L, y):
    """y = a + b*(L/158)^2 (fixed exponent, the planning form)."""
    L = np.asarray(L, float); y = np.asarray(y, float)
    if len(L) < 2 or not (y > 0).all():
        return None
    w = 1.0 / y
    x = (L / 158.0) ** 2; a, b = nnls2(w, x * w, np.ones_like(y))
    resid = y - (a + b * x)
    return {'a': float(a), 'b': float(b), 'p': 2.0, 'rmse': float(np.sqrt(np.mean(resid ** 2))),
            'r2': float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)) if np.ptp(y) > 0 else None,
            'n': int(len(L)), 'form': 'a + b*(L/158)^2, a,b>=0, relative-error least squares'}


def predict(model, L):
    return model['a'] + model['b'] * (np.asarray(L, float) / 158.0) ** model['p']


def bootstrap(L, y, fitter, at, n=1000, seed=20261015):
    L = np.asarray(L, float); y = np.asarray(y, float); rng = np.random.default_rng(seed); preds = []
    if len(L) < 4:
        return None
    for _ in range(n):
        idx = rng.integers(0, len(L), len(L))
        m = fitter(L[idx], y[idx])
        if m is not None:
            preds.append(predict(m, at))
    preds = np.asarray(preds)
    return {'lengths': list(map(int, at)), 'p2.5': np.percentile(preds, 2.5, 0).tolist(), 'p97.5': np.percentile(preds, 97.5, 0).tolist(),
            'resamples': int(len(preds)), 'unit': 'case-level bootstrap'}


# ----------------------------------------------------------------------------- loading

def load_cases(results):
    cases = {}
    for path in sorted((results / 'cases').glob('*/case_result.json')):
        record = json.loads(path.read_text()); cases[record['case_id']] = record
    return cases


def case_row(c):
    t = c.get('timing_seconds', {}); samples = c.get('samples', []); refs = c.get('refinement', [])
    ok = [s for s in samples if s.get('status') == 'ok']
    rok = [r for r in refs if r.get('status') == 'ok']
    rfit = [r for r in rok if not r.get('first_refinement_on_rank')]
    k_ok = len(ok)
    gen_per = mean([s['predict_step_including_PDB_write_seconds'] for s in ok])
    sampler_per = mean([s['published_sampling_seconds'] for s in ok])
    ref_per = mean([r['timing_seconds']['complete_case_wall'] for r in rfit]) if rfit else None
    ref_per_all = mean([r['timing_seconds']['complete_case_wall'] for r in rok]) if rok else None
    ref_obs = mean([r['parent_observed_wall_seconds'] for r in rfit]) if rfit else None
    minim = mean([r['timing_seconds'].get('minimizer_stage_total') for r in rfit]) if rfit else None
    gen_loop = t.get('generation_loop_wall_including_panel')
    occupancy = None
    if k_ok and gen_loop is not None and ref_obs is not None:
        occupancy = t.get('input_preparation', 0.0) + t.get('native_panel', 0.0) + gen_loop * 32.0 / k_ok + 32.0 * ref_obs
    counts = c.get('counts', {})
    return {'case_id': c['case_id'], 'length': c['length'], 'length_bin': c.get('length_bin'), 'role': c.get('role'), 'origin': c.get('origin'),
            'status': c.get('status'), 'node_slot': c.get('rank', {}).get('node_slot'), 'gpu_slot': c.get('rank', {}).get('gpu_slot'),
            'hostname': c.get('rank', {}).get('hostname'), 'K_ok': k_ok, 'R_planned': len(c.get('config', {}).get('refined_sample_ids_planned', [])),
            'R_ok': len(rok), 'R_used_for_cost': len(rfit),
            'generation_predict_step_seconds_per_sample': gen_per, 'generation_sampler_seconds_per_sample': sampler_per,
            'generation_GPU_seconds_sum': t.get('generation_predict_step_seconds_sum'),
            'generation_K32_GPU_seconds': gen_per * 32 if gen_per is not None else None,
            'refinement_worker_seconds_per_sample': ref_per, 'refinement_worker_seconds_per_sample_incl_first': ref_per_all,
            'refinement_minimizer_seconds_per_sample': minim,
            'refinement_worker_seconds_sum_measured': t.get('refinement_worker_seconds_sum'),
            'refinement_K32_worker_seconds_extrapolated': ref_per * 32 if ref_per is not None else None,
            'occupancy_K32_GPU_seconds': occupancy, 'node_seconds_K32': occupancy / 4 if occupancy is not None else None,
            'case_wall_seconds': t.get('case_wall'),
            'generation_peak_allocated_GiB': (c.get('memory', {}).get('generation_peak_allocated_bytes') or 0) / 2 ** 30 or None,
            'refinement_device_used_max_GiB': (max([r.get('device_used_bytes_max_sampled') or 0 for r in refs] or [0]) / 2 ** 30) or None,
            'generation_failed_or_nonfinite': counts.get('generated_failed_or_nonfinite'),
            'generation_skipped_deadline': counts.get('generation_skipped_deadline'),
            'refinement_failed': counts.get('refinement_failed'), 'refinement_skipped_deadline': counts.get('refinement_skipped_deadline'),
            'mean_generated_C4prime_lDDT': mean([s.get('C4prime_lDDT') for s in ok]),
            'mean_refined_C4prime_lDDT': mean([r.get('after_C4prime_lDDT') for r in rok]),
            'pipeline_QA_not_held_out': True}


def write_csv(path, rows):
    if not rows:
        path.write_text(''); return
    keys = list(rows[0].keys())
    with path.open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=keys); w.writeheader(); w.writerows(rows)


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True, help='job results dir (cases/, ranks/)')
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sacct', type=Path, default=None, help='optional: sacct -P -o JobID,ElapsedRaw,NNodes,AllocTRES output')
    parser.add_argument('--pilot-refinement-results', type=Path, default=None, help='retrieved job_2298016 directory')
    parser.add_argument('--pilot-refinement-manifest', type=Path, default=None, help='refinement_manifest.json (generated SHA list)')
    parser.add_argument('--price-lengths', type=Path, default=None, help='optional JSON list of case lengths to price')
    parser.add_argument('--full-cohorts', type=int, default=PILOT['plan_full_cohorts'])
    parser.add_argument('--dev-cohorts', type=int, default=PILOT['plan_dev_cohorts'])
    parser.add_argument('--dev-seconds-per-cohort', type=float, default=PILOT['plan_dev_seconds_per_cohort'])
    parser.add_argument('--allocation-node-hours', type=float, default=85500.0)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    table = json.loads((args.packet / 'case_table.json').read_text())
    plan = json.loads((args.packet / 'partition_plan.json').read_text())
    planned = {c['case_id']: c for c in table['cases']}
    cases = load_cases(args.results)
    rows = [case_row(c) for c in cases.values()]
    rows.sort(key=lambda r: (r['length'], r['case_id']))
    write_csv(args.out / 'waveD_case_costs.csv', rows)

    # per-sample metric table
    sample_rows = []
    for c in cases.values():
        refs = {r['sample_id']: r for r in c.get('refinement', []) if 'sample_id' in r}
        for s in c.get('samples', []):
            p = s.get('panel') or {}; r = refs.get(s['sample_id'], {}); aq = r.get('after_geometry_QC') or {}; bq = r.get('before_geometry_QC') or {}
            sample_rows.append({'case_id': c['case_id'], 'length': c['length'], 'role': c.get('role'), 'origin': c.get('origin'), 'sample_id': s['sample_id'],
                                'generation_status': s.get('status'), 'predict_step_seconds': s.get('predict_step_including_PDB_write_seconds'),
                                'sampler_seconds': s.get('published_sampling_seconds'), 'C4prime_lDDT': s.get('C4prime_lDDT'),
                                'C4prime_RMSD_A': s.get('C4prime_Kabsch_RMSD_angstrom'), 'finite': p.get('all_coordinates_finite'),
                                'O3P_above_2p1': p.get('O3prime_next_P_above_2p1_count'), 'O3P_below_1p3': p.get('O3prime_next_P_below_1p3_count'),
                                'O3P_connections': p.get('O3prime_next_P_connections'),
                                'nonadjacent_C4_below_3A': p.get('nonadjacent_C4prime_pairs_below_3_angstrom_count'),
                                'heavy_pairs_below_2A': p.get('heavy_atom_pairs_below_2_angstrom_residue_separation_ge2_count'),
                                'refinement_status': r.get('status'), 'refinement_failure': r.get('failure_category'),
                                'refinement_worker_seconds': (r.get('timing_seconds') or {}).get('complete_case_wall'),
                                'first_refinement_on_rank': r.get('first_refinement_on_rank'),
                                'refined_C4prime_lDDT': r.get('after_C4prime_lDDT'),
                                'refined_connection_screen_passed': r.get('predeclared_connection_screen_passed'),
                                'before_OL3_bond_RMS_A': (bq.get('heavy_bond_absolute_error_to_OL3_target_angstrom') or {}).get('RMS'),
                                'after_OL3_bond_RMS_A': (aq.get('heavy_bond_absolute_error_to_OL3_target_angstrom') or {}).get('RMS'),
                                'before_OL3_angle_RMS_deg': (bq.get('heavy_angle_absolute_error_to_OL3_target_degree') or {}).get('RMS'),
                                'after_OL3_angle_RMS_deg': (aq.get('heavy_angle_absolute_error_to_OL3_target_degree') or {}).get('RMS'),
                                'before_LJ_flags': bq.get('heavy_nonbonded_LJ_proximity_screen_count'),
                                'after_LJ_flags': aq.get('heavy_nonbonded_LJ_proximity_screen_count'),
                                'after_O3P_above_2p1': aq.get('O3prime_next_P_above_2p1_angstrom_count')})
    write_csv(args.out / 'waveD_sample_metrics.csv', sample_rows)

    # ---------------------------------------------------------------- cost model
    train = [r for r in rows if r['role'] == 'training_case_pipeline_QA']
    fitrows = {
        'generation_predict_step_seconds_per_sample': [r for r in train if r['generation_predict_step_seconds_per_sample'] is not None],
        'refinement_worker_seconds_per_sample': [r for r in train if r['refinement_worker_seconds_per_sample'] is not None],
        'occupancy_K32_GPU_seconds': [r for r in train if r['occupancy_K32_GPU_seconds'] is not None]}
    at = np.asarray(EVAL_LENGTHS, float); models = {}
    for key, subset in fitrows.items():
        L = [r['length'] for r in subset]; y = [r[key] for r in subset]
        power = fit_power(L, y); quad = fit_quadratic(L, y)
        models[key] = {'n_cases': len(subset), 'power_law': power, 'quadratic_planning_form': quad,
                       'prediction_power_law': dict(zip(EVAL_LENGTHS, predict(power, at).tolist())) if power else None,
                       'bootstrap_power_law': bootstrap(L, y, fit_power, at, args.bootstrap) if power else None}
    occ = models['occupancy_K32_GPU_seconds']['power_law']
    # pricing ------------------------------------------------------------
    if args.price_lengths:
        lengths = json.loads(args.price_lengths.read_text()); length_source = str(args.price_lengths)
    else:
        lengths = table['pool_lengths']; length_source = 'deduplicated eligible RNAbpFlow pool (case_table.json pool_lengths)'
    pricing = {'length_source': length_source, 'n_lengths': len(lengths),
               'basis': 'node-seconds per K32 case = fitted occupancy_K32(L)/4 (4 GH200 GPUs per node, perfectly packed)'}
    if occ:
        per_case_ns = predict(occ, np.asarray(lengths, float)) / 4.0
        boot = models['occupancy_K32_GPU_seconds']['bootstrap_power_law']
        at177 = float(predict(occ, [177])[0]) / 4.0; at158 = float(predict(occ, [158])[0]) / 4.0
        full_nh_177 = args.full_cohorts * at177 / 3600; full_nh_dist = args.full_cohorts * float(per_case_ns.mean()) / 3600
        dev_nh = args.dev_cohorts * args.dev_seconds_per_cohort / 3600
        pricing.update({
            'node_seconds_per_K32_case_at_158nt': at158, 'node_seconds_per_K32_case_at_177nt': at177,
            'node_seconds_per_K32_case_mean_over_lengths': float(per_case_ns.mean()),
            'pilot_one_node_wall_node_seconds_158nt': PILOT['generation_4gpu_cohort_wall_s_158nt'] + PILOT['refinement_4worker_cohort_wall_s_158nt'],
            'plan_forecast_node_seconds_177nt': (PILOT['generation_4gpu_cohort_wall_s_158nt'] + PILOT['refinement_4worker_cohort_wall_s_158nt']) * PILOT['plan_length_factor_177'],
            'full_cohorts': args.full_cohorts, 'full_cohort_node_hours_at_177nt': full_nh_177,
            'full_cohort_node_hours_over_length_distribution': full_nh_dist,
            'dev_cohort_node_hours_unchanged_from_plan': dev_nh,
            'total_node_hours_177nt': full_nh_177 + dev_nh, 'total_node_hours_length_distribution': full_nh_dist + dev_nh,
            'percent_of_allocation_177nt': 100 * (full_nh_177 + dev_nh) / args.allocation_node_hours,
            'percent_of_allocation_length_distribution': 100 * (full_nh_dist + dev_nh) / args.allocation_node_hours,
            'ten_percent_cap_node_hours': 0.1 * args.allocation_node_hours,
            'plan_frozen_evaluation_cap_node_hours': PILOT['plan_frozen_evaluation_cap_NH'],
            'plan_generation_refinement_forecast_node_hours': PILOT['plan_inference_forecast_NH'],
            'fits_10_percent_cap_length_distribution': (full_nh_dist + dev_nh) <= 0.1 * args.allocation_node_hours})
        if boot:
            i177 = EVAL_LENGTHS.index(177)
            pricing['node_seconds_per_K32_case_at_177nt_CI95'] = [boot['p2.5'][i177] / 4, boot['p97.5'][i177] / 4]
    # anchor --------------------------------------------------------------
    anchor = {}
    a = cases.get('P4P6_ref158')
    if a:
        ok = [s for s in a['samples'] if s.get('status') == 'ok']
        anchor = {'generated_ok': len(ok), 'refined_ok': sum(r.get('status') == 'ok' for r in a.get('refinement', [])),
                  'mean_generated_C4prime_lDDT': mean([s.get('C4prime_lDDT') for s in ok]),
                  'pilot_job2297413_mean_C4prime_lDDT_316atom_alignment_run': 0.557643,
                  'generation_predict_step_sum_s': sum(s['predict_step_including_PDB_write_seconds'] for s in ok),
                  'generation_sampler_sum_s': sum(s['published_sampling_seconds'] for s in ok),
                  'pilot_1gpu_sampler_sum_s': PILOT['generation_1gpu_sampler_sum_s_158nt'],
                  'refinement_worker_seconds_sum': a['timing_seconds'].get('refinement_worker_seconds_sum'),
                  'pilot_refinement_case_wall_sum_s': PILOT['refinement_case_wall_sum_s_158nt']}
        if args.pilot_refinement_manifest:
            m = json.loads(args.pilot_refinement_manifest.read_text())
            pilot_sha = {g['sample_id']: g['sha256'] for g in m['generated']}
            anchor['generated_PDB_sha_matches_job2297413'] = sum(pilot_sha.get(s['sample_id']) == s.get('PDB_sha256') for s in ok)
        if args.pilot_refinement_results:
            same_after = 0; same_stage = 0; compared = 0
            for r in a.get('refinement', []):
                f = args.pilot_refinement_results / 'cohort' / f"sample_{r['sample_id']:02d}" / 'case_refinement.json'
                if r.get('status') != 'ok' or not f.exists():
                    continue
                p = json.loads(f.read_text()); compared += 1
                same_after += p['after_geometry_QC'] == r['after_geometry_QC']
                same_stage += all(abs(x['potential_energy_after_kJ_per_mol'] - y['potential_energy_after_kJ_per_mol']) <= 1e-6 * max(1, abs(x['potential_energy_after_kJ_per_mol']))
                                  for x, y in zip(p['minimization_stages'], r['minimization_stages']))
            anchor.update({'refined_samples_compared_to_job2298016': compared, 'after_QC_identical_count': same_after,
                           'stage_energies_match_count': same_stage})
    # validity / lDDT summaries by bin ------------------------------------------
    bins = {}
    for c in cases.values():
        b = bins.setdefault((c.get('origin') or '?', c.get('length_bin', '?')), {'cases': 0, 'gen': [], 'ref': []})
        b['cases'] += 1; b['gen'] += [s for s in c.get('samples', []) if s.get('status') in ('ok', 'nonfinite')]
        b['ref'] += [r for r in c.get('refinement', []) if r.get('status') == 'ok']
    validity = {}
    for key_, b in sorted(bins.items(), key=lambda kv: (kv[0][0], int(kv[0][1].strip('[').split(',')[0]) if kv[0][1] != '?' else 0)):
        name = f'{key_[0]} {key_[1]}'
        gen = b['gen']; ref = b['ref']
        conn = sum((s['panel'].get('O3prime_next_P_connections') or 0) for s in gen if s['panel'].get('all_coordinates_finite'))
        validity[name] = {
            'cases': b['cases'], 'generated_samples': len(gen),
            'finite_fraction': mean([1.0 if s['panel'].get('all_coordinates_finite') else 0.0 for s in gen]),
            'O3P_above_2p1_fraction_of_connections': (sum(s['panel'].get('O3prime_next_P_above_2p1_count', 0) for s in gen if s['panel'].get('all_coordinates_finite')) / conn) if conn else None,
            'nonadjacent_C4_below_3A_per_sample': summary([s['panel'].get('nonadjacent_C4prime_pairs_below_3_angstrom_count') for s in gen]),
            'heavy_pairs_below_2A_per_sample': summary([s['panel'].get('heavy_atom_pairs_below_2_angstrom_residue_separation_ge2_count') for s in gen]),
            'generated_C4prime_lDDT_pipeline_QA': summary([s.get('C4prime_lDDT') for s in gen]),
            'refined_samples': len(ref),
            'refined_connection_screen_pass_fraction': mean([1.0 if r.get('predeclared_connection_screen_passed') else 0.0 for r in ref]),
            'refined_after_bond_RMS_A': summary([(r['after_geometry_QC']['heavy_bond_absolute_error_to_OL3_target_angstrom'] or {}).get('RMS') for r in ref]),
            'refined_after_angle_RMS_deg': summary([(r['after_geometry_QC']['heavy_angle_absolute_error_to_OL3_target_degree'] or {}).get('RMS') for r in ref]),
            'refined_after_LJ_flags': summary([r['after_geometry_QC'].get('heavy_nonbonded_LJ_proximity_screen_count') for r in ref]),
            'refined_lDDT_change_paired': summary([(r.get('after_C4prime_lDDT') - r.get('before_C4prime_lDDT'))
                                                   for r in ref if r.get('after_C4prime_lDDT') is not None and r.get('before_C4prime_lDDT') is not None])}
    # failures -----------------------------------------------------------------
    failures = {'planned_cases': len(planned), 'case_json_present': len(cases),
                'missing_case_json': sorted(set(planned) - set(cases)),
                'case_status': {}, 'generation_status': {}, 'refinement_status': {}, 'refinement_failure_category': {}}
    for c in cases.values():
        failures['case_status'][c.get('status')] = failures['case_status'].get(c.get('status'), 0) + 1
        for s in c.get('samples', []):
            failures['generation_status'][s.get('status')] = failures['generation_status'].get(s.get('status'), 0) + 1
        for r in c.get('refinement', []):
            failures['refinement_status'][r.get('status')] = failures['refinement_status'].get(r.get('status'), 0) + 1
            if r.get('failure_category'):
                failures['refinement_failure_category'][r['failure_category']] = failures['refinement_failure_category'].get(r['failure_category'], 0) + 1
    ranks = [json.loads(p.read_text()) for p in sorted((args.results / 'ranks').glob('rank_*.json'))] if (args.results / 'ranks').exists() else []
    failures['rank_status'] = {}
    for r in ranks:
        failures['rank_status'][r.get('status')] = failures['rank_status'].get(r.get('status'), 0) + 1
    memory = {'generation_peak_allocated_GiB_vs_length': [(r['length'], r['generation_peak_allocated_GiB']) for r in rows if r['generation_peak_allocated_GiB']],
              'refinement_device_used_max_GiB_vs_length': [(r['length'], r['refinement_device_used_max_GiB']) for r in rows if r['refinement_device_used_max_GiB']]}
    allocation = {}
    if args.sacct and args.sacct.exists():
        lines = [ln.split('|') for ln in args.sacct.read_text().splitlines() if ln.strip()]
        head = lines[0]; body = [dict(zip(head, ln)) for ln in lines[1:]]
        main_rows = [b for b in body if '.' not in b.get('JobID', '.')]
        if main_rows:
            el = float(main_rows[0]['ElapsedRaw']); nn = int(main_rows[0]['NNodes'])
            allocation = {'elapsed_seconds': el, 'nodes': nn, 'node_hours': el * nn / 3600, 'GPU_hours': el * nn * 4 / 3600, 'source': 'sacct'}
    elif ranks:
        import datetime as dt
        def parse(s):
            return dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').timestamp()
        starts = [parse(r['started_utc']) for r in ranks if r.get('started_utc')]; ends = [parse(r['finished_utc']) for r in ranks if r.get('finished_utc')]
        if starts and ends:
            el = max(ends) - min(starts)
            allocation = {'rank_span_seconds': el, 'node_hours_lower_bound': el * plan['nodes'] / 3600, 'source': 'rank timestamps (lower bound; use sacct)'}
    gpu_busy = sum(r['case_wall_seconds'] or 0 for r in rows)
    allocation['sum_case_wall_GPU_seconds'] = gpu_busy
    if allocation.get('GPU_hours'):
        allocation['GPU_packing_efficiency'] = gpu_busy / 3600 / allocation['GPU_hours']
    origin_note = ('PDB_chain references are experimental training structures; bpRNA_RFAM_distillation references are '
                   'cross-distillation targets (not experimental). Report them separately; neither is held-out.')
    result = {'scope': SCOPE, 'origin_note': origin_note, 'cases_analysed': len(rows), 'cost_model': models, 'pricing': pricing, 'anchor_P4P6': anchor,
              'validity_by_length_bin_pipeline_QA': validity, 'failures': failures, 'memory': memory, 'allocation': allocation,
              'definitions': __doc__, 'pilot_reference': PILOT}
    (args.out / 'waveD_findings.json').write_text(json.dumps(result, indent=1, default=float) + '\n')
    # text summary ----------------------------------------------------------------
    lines = ['RIBO-ATLAS-FM Wave D evaluation-pipeline cost/robustness summary', SCOPE, '']
    lines.append(f"Cases analysed {len(rows)} of {len(planned)} planned; case status {failures['case_status']}; rank status {failures['rank_status']}.")
    for key, m in models.items():
        pl = m['power_law']
        if pl:
            lines.append(f"{key}: a={pl['a']:.3f} + b={pl['b']:.3f}*(L/158)^{pl['p']:.2f}  (n={pl['n']}, R2={pl['r2'] if pl['r2'] is None else round(pl['r2'], 3)}, median |rel err| {pl['median_abs_relative_error']:.3f})")
    if pricing.get('node_seconds_per_K32_case_at_177nt') is not None:
        lines.append(f"Node-seconds per K32 case: 158 nt {pricing['node_seconds_per_K32_case_at_158nt']:.1f} (pilot one-node wall {pricing['pilot_one_node_wall_node_seconds_158nt']:.1f}); "
                     f"177 nt {pricing['node_seconds_per_K32_case_at_177nt']:.1f} (plan forecast {pricing['plan_forecast_node_seconds_177nt']:.1f}); "
                     f"mean over {pricing['n_lengths']} pool lengths {pricing['node_seconds_per_K32_case_mean_over_lengths']:.1f}.")
        lines.append(f"{pricing['full_cohorts']} K32 cohorts + unchanged dev term: {pricing['total_node_hours_177nt']:.0f} NH at 177 nt, "
                     f"{pricing['total_node_hours_length_distribution']:.0f} NH over the pool length distribution "
                     f"({pricing['percent_of_allocation_length_distribution']:.2f}% of {args.allocation_node_hours:.0f}; 10% cap {pricing['ten_percent_cap_node_hours']:.0f} NH).")
    if anchor:
        lines.append(f"P4-P6 anchor: {json.dumps(anchor)}")
    lines.append(f"Failures: generation {failures['generation_status']}; refinement {failures['refinement_status']} {failures['refinement_failure_category']}; missing case JSON {len(failures['missing_case_json'])}.")
    lines.append(f"Allocation: {json.dumps(allocation)}")
    lines.append('Refinement cost for cases with R<32 is extrapolated linearly from the measured refined subset (sample IDs 0..R-1).')
    (args.out / 'waveD_findings.txt').write_text('\n'.join(lines) + '\n')
    try:
        import matplotlib
        import matplotlib.ticker
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        grid = np.linspace(30, 720, 200)
        for ax, key, label in ((axes[0], 'generation_predict_step_seconds_per_sample', 'generation s / sample'),
                               (axes[1], 'refinement_worker_seconds_per_sample', 'refinement worker s / sample'),
                               (axes[2], 'occupancy_K32_GPU_seconds', 'K32 case GPU-seconds')):
            sub = fitrows[key]
            ax.scatter([r['length'] for r in sub], [r[key] for r in sub], s=10, color='#3b6ea5', label='training cases')
            anc = [r for r in rows if r['role'] != 'training_case_pipeline_QA' and r[key] is not None]
            ax.scatter([r['length'] for r in anc], [r[key] for r in anc], s=40, marker='D', color='#c0504d', label='P4-P6 anchor')
            m = models[key]['power_law']
            if m:
                ax.plot(grid, predict(m, grid), color='black', lw=1, label=f"fit p={m['p']:.2f}")
            ax.set_xlabel('RNA length (nt)'); ax.set_ylabel(label); ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(fontsize=7)
            ax.set_xticks([30, 50, 100, 200, 400, 700]); ax.set_xticklabels(['30', '50', '100', '200', '400', '700'])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        fig.suptitle('Wave D measured cost vs length (engineering measurement; training cases = pipeline QA)', fontsize=9)
        fig.tight_layout(); fig.savefig(args.out / 'waveD_cost_vs_length.png', dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([r['length'] for r in sample_rows if r['C4prime_lDDT'] is not None], [r['C4prime_lDDT'] for r in sample_rows if r['C4prime_lDDT'] is not None], s=3, alpha=.3)
        ax.set_xscale('log'); ax.set_xlabel('RNA length (nt)'); ax.set_ylabel("C4' lDDT (pipeline QA, training cases)"); ax.set_ylim(0, 1)
        ax.set_xticks([30, 50, 100, 200, 400, 700]); ax.set_xticklabels(['30', '50', '100', '200', '400', '700'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        fig.tight_layout(); fig.savefig(args.out / 'waveD_lddt_vs_length_pipeline_QA.png', dpi=150); plt.close(fig)
    except Exception as error:  # figures are optional
        (args.out / 'figure_error.txt').write_text(repr(error) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
