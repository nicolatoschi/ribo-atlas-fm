#!/usr/bin/env python3
"""Trace the numbers that are not read from a file to the shipped evidence.

Most numbers in the proposal documents come from facts_v2.json. build_facts_v2.py reads most of
those from retrieved analyses, but this script also recomputes the rest from the execution-pack
evidence tree:
  A. constants that build_facts_v2.py types itself (earlier cost basis, Wave E summary, GPU-hour
     spend, node lists, data-expansion counts);
  B. measurement and data numbers that the document text types by hand rather than taking from facts;
  C. design parameters, which are checked against design_v2.json where they are recorded there.
It prints PASS/FAIL per item and exits non-zero on any FAIL. Plan parameters with no measurement
behind them, and references to external documents, are listed in docs/NUMBER_SOURCES.md.

Usage: python3 -I tools/check_number_sources.py --evidence-root <pack>/evidence [--facts resource_plan/facts_v2.json]
"""
import argparse
import csv
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ET = 'work/geometry_pilot/ensemble_training'
AO = f'{ET}/anomaly_off_remeasure_20261015'
RET = f'{AO}/retrieved'
SLOW = {'jpbo-048-33', 'jpbo-049-19', 'jpbo-061-16'}


def load(p):
    return json.loads(Path(p).read_text())


def node_excess(E):
    """Per-node gradient-phase work relative to the median node, for every multi-node step with rank reports."""
    out = []
    for job in ['job_2339575', 'job_2342412']:
        lines = (E / RET / job / 'steps.tsv').read_text().splitlines()[1:]
        for line in lines:
            f = line.split('\t'); label, nodes = f[0], f[1].split(',')
            if len(nodes) < 2:
                continue
            d = E / RET / job / label
            m = re.match(r'C1/continuous_(traj\d)', label) or re.match(r'C2/resume_(traj\d)_on', label)
            if m:
                d = E / RET / job / 'C' / m.group(1) / ('continuous' if label.startswith('C1') else 'resume')
            m = re.match(r'B1/(continuous|resume)_step', label)
            if m:
                d = E / RET / job / 'B1' / m.group(1)
            files = sorted(d.glob('rank_*.json'))
            for ui in range(3):
                per = {}
                for fn in files:
                    r = load(fn)
                    g = r['groups'] if ('groups' in r and ui == 0) else (r['updates'][ui]['groups'] if 'updates' in r and ui < len(r['updates']) else None)
                    if g is None:
                        continue
                    w = sum(x['mean_pass_seconds'] + x['gradient_forward_seconds'] + x['backward_recompute_seconds'] for x in g)
                    n = nodes[r['rank'] // 4]; per[n] = max(per.get(n, 0), w)
                if len(per) >= 2:
                    med = st.median(per.values())
                    out.append({'step': f'{job}/{label}', 'update': ui, 'excess': {n: w / med - 1 for n, w in per.items()}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-root', type=Path, required=True)
    ap.add_argument('--facts', type=Path, default=REPO / 'resource_plan/facts_v2.json')
    a = ap.parse_args()
    E = a.evidence_root.resolve(); F = load(a.facts)
    checks = []

    def check(section, name, stated, recomputed, source):
        checks.append({'section': section, 'item': name, 'stated': stated, 'recomputed': recomputed, 'source': source,
                       'result': 'PASS' if stated == recomputed else 'FAIL'})

    # ---------------- A. constants typed inside build_facts_v2.py -------------------------------
    nfp = f'{ET}/native_fixed_site_limit_retrieved/snapshot_20261014T072248715026Z/native_fixed_summary.json'
    g = load(E / nfp)['GPUmeasurements']
    check('A', 'unit.old_c32 (anomaly-on basis, node-s/case)', F['unit']['old_c32'], round(g[0]['node_seconds_per_case'], 2), nfp)
    check('A', 'unit.old_c8', F['unit']['old_c8'], round(g[1]['node_seconds_per_case'], 2), nfp)
    check('A', 'reliability.old_2305296_s', F['reliability']['old_2305296_s'], round(g[0]['rank_max_seconds'], 1), nfp)
    wp = 'work/sequence_prior_waveE/retrieved/job_2340325/analysis/waveE_analysis.json'
    W = load(E / wp); s, r, lh = W['scaling'], W['runs'], W['learning']
    ho = lh['heldout_family']['first_epoch']['family_bootstrap']; inf = lh['val_in_family']['first_epoch']['family_bootstrap']
    for key, val in [('E16', round(s['P1_N16']['efficiency_vs_reference'], 3)), ('E32', round(s['P1_N32']['efficiency_vs_reference'], 3)),
                     ('E64', round(s['P2_N64']['efficiency_vs_reference'], 3)), ('sdpa_speedup', round(s['P1_N8_sdpa']['speedup_vs_reference'], 3)),
                     ('HBM_official_GiB', round(r['P1_N8']['peak_alloc_gib_max'], 1)), ('HBM_sdpa_GiB', round(r['P1_N8_sdpa']['peak_alloc_gib_max'], 1)),
                     ('heldout_delta', round(ho['delta_ce_token_weighted'], 5)), ('heldout_CI', [round(x, 5) for x in ho['delta_ce_token_weighted_ci95']]),
                     ('heldout_families', ho['units']), ('infamily_delta_1ep', round(inf['delta_ce_token_weighted'], 5)),
                     ('infamily_CI_1ep', [round(x, 5) for x in inf['delta_ce_token_weighted_ci95']]),
                     ('loss_agreement_max', round(max(v['max_abs_loss_diff_vs_reference'] for v in W['loss_consistency'].values()), 5))]:
        check('A', f'waveE.{key}', F['waveE'][key], val, wp)
    led = subprocess.run([sys.executable, '-I', str(REPO / 'tools/build_job_ledger.py'), '--evidence-root', str(E), '--facts', str(a.facts)],
                         capture_output=True, text=True)
    L = json.loads(led.stdout)
    for k, c in L['facts_comparison'].items():
        check('A', f'spend_GPUh.{k} (scheduler-record ledger)', c['facts'], c.get('ledger', c.get('ledger_sum_of_rounded')),
              'tools/build_job_ledger.py over every shipped sacct record')
    ex = node_excess(E)
    slow_found = sorted({n for e in ex for n, v in e['excess'].items() if v > 0.15})
    check('A', 'scaling.slow_nodes (nodes >15% slower than the median node in some step)', sorted(F['scaling']['slow_nodes']), slow_found,
          f'{RET}/job_*/steps.tsv + rank reports')
    wd = load(E / 'work/geometry_pilot/evaluation_pipeline_waveD/retrieved/job_2340563/ranks/rank_24_1.json')
    check('A', 'scaling.faulty_nodes includes jpbo-090-25 (CUDA out of memory at start-up)', True,
          wd['hostname'].startswith('jpbo-090-25') and 'out of memory' in wd.get('failure_message', ''),
          'work/geometry_pilot/evaluation_pipeline_waveD/retrieved/job_2340563/ranks/rank_24_1.json')
    sub = load(E / f'{AO}/submission_records.json')
    check('A', 'scaling.faulty_nodes includes jpbo-055-37 (recorded faulty GPU)', True,
          any('jpbo-055-37' in json.dumps(x.get('node_health_exclusions', {})) for x in sub['runs']), f'{AO}/submission_records.json')
    XD = 'work/resource_costing/expansion_admission_20261016'
    dm = load(E / f'{XD}/admission_run/download_manifest.json')['assets']
    new = [l.split('\t') for l in (E / f'{XD}/download_list.tsv').read_text().splitlines() if l.strip()]
    newb = {x[0]: int(x[3]) for x in new}
    check('A', 'data_all.files (hash-verified RDATs)', F['data_all']['files'], sum(1 for x in dm if x['sha256_verified'] and x['size_verified']),
          f'{XD}/admission_run/download_manifest.json')
    check('A', 'data_all.new_files', F['data_all']['new_files'], len(new), f'{XD}/download_list.tsv')
    check('A', 'data_all.new_GB', F['data_all']['new_GB'], round(sum(newb.values()) / 1e9, 2), f'{XD}/download_list.tsv')
    xm = load(E / f'{XD}/expansion_manifest.json')['files']
    check('A', 'data_all.lengths_nt (template lengths)', F['data_all']['lengths_nt'], sorted({f['template_length'] for f in xm} | {177}),
          f'{XD}/expansion_manifest.json (the 8 original files are 177 nt)')
    design = load(E / 'work/resource_costing/v2/design_v2.json')
    check('A', 'data_all.holdout_fraction 0.15 (design statement)', True, '15% holdout' in json.dumps(design) or '15% family' in json.dumps(design),
          'work/resource_costing/v2/design_v2.json')
    # ---------------- B. numbers typed by hand in the document text ------------------------------
    c4 = sorted((E / RET / 'job_2342412/C4/wide_N128_K32').glob('rank_*.json'))
    r0 = load(c4[0])
    check('B', 'largest measured job: 512 GPUs (job 2342412, C4)', 512, len(c4), f'{RET}/job_2342412/C4/wide_N128_K32')
    check('B', 'decoder parameters 17.2M', 17.2, round(r0['parameter_count'] / 1e6, 1), 'C4 rank report parameter_count')
    rs = load(E / 'work/sequence_prior_waveE/waveE_05ec1ab6b438/rnafm_sources.json')
    check('B', 'RNA-FM parameters 99.5M', 99.5, round(rs['weights']['parameters_unique'] / 1e6, 1), 'waveE packet rnafm_sources.json')
    cfg = (E / f'{AO}/native_campaign_e8ff330ee775/RNAbpFlow_training/configs/config.yaml').read_text()
    grab = lambda k: int(re.search(rf'^\s*{k}:\s*(\d+)', cfg, re.M).group(1))
    check('B', 'decoder: 256-channel nodes, 128-channel pairs, 6 IPA blocks, 50 steps', [256, 128, 6, 50],
          [grab('node_embed_size'), grab('edge_embed_size'), grab('num_blocks'), grab('num_timesteps')], 'RNAbpFlow_training/configs/config.yaml')
    check('B', 'HBM 12.8 GiB at eight 177-nt RNAs/GPU', 12.8, round(F['microbatch']['HBM_175_m8_GiB'], 1), 'facts.microbatch')
    check('B', 'HBM 26 GiB at 720 nt', 26, round(F['length']['7ptk_B']['HBM_GiB']), 'facts.length')
    check('B', 'GPU memory available 95 GiB', 95.0, round(r0['GPU_capacity_bytes'] / 2 ** 30, 1), 'C4 rank report GPU_capacity_bytes')
    st2 = [l.split('\t') for l in (E / RET / 'job_2342412/steps.tsv').read_text().splitlines()[1:]]
    c3 = [x for x in st2 if x[0].startswith('C3/')]
    check('B', '16 concurrent 8-node trajectories in one 128-node job', [16, {8}], [len(c3), {len(x[1].split(',')) for x in c3}], f'{RET}/job_2342412/steps.tsv')
    f1 = load(E / 'work/rebuild_v2/figure1_measured_cost.json')['length_points']
    check('B', 'cost per case about 1.4-1.5x at 720 nt (K32, K8)', [1.4, 1.5],
          [round(f1['32']['720'] / f1['32']['175'], 1), round(f1['8']['720'] / f1['8']['175'], 1)], 'work/rebuild_v2/figure1_measured_cost.json')
    check('B', 'cost flat up to about 400 nt (394 nt within 10% of 175 nt, K32 and K8)', True,
          abs(f1['32']['394'] / f1['32']['175'] - 1) < 0.1 and abs(f1['8']['394'] / f1['8']['175'] - 1) < 0.1, 'work/rebuild_v2/figure1_measured_cost.json')
    nm = load(E / f'{AO}/native_anomaly_off_b63550bd42ee/frozen_manifest.json')['configuration']
    check('B', 'cost runs use one 175-nt RNA (5zwm_B)', ['5zwm_B', 175], [nm['case'], nm['length']], 'native_anomaly_off_b63550bd42ee/frozen_manifest.json')
    other = [v for e in ex for n, v in e['excess'].items() if n not in SLOW]
    check('B', 'all other nodes within about 5% of the median node', True, max(other) < 0.06, 'rank reports + steps.tsv')
    # independent re-implementation of the builder's slow-node rule (single-trajectory multi-node steps)
    mine = []
    for job in ['job_2339575', 'job_2342412']:
        for line in (E / RET / job / 'steps.tsv').read_text().splitlines()[1:]:
            step, nodes = line.split('\t')[0], line.split('\t')[1].split(',')
            groups = {}
            for fn in (E / RET / job / step).glob('**/rank_*.json') if (E / RET / job / step).is_dir() else []:
                r = load(fn)
                if 'rank' not in r:
                    continue
                gg = r['updates'][0]['groups'] if 'updates' in r else r.get('groups', [])
                w = sum(x['mean_pass_seconds'] + x['gradient_forward_seconds'] + x['backward_recompute_seconds'] for x in gg)
                groups.setdefault(str(fn.parent), {}).setdefault(r['rank'] // 4, []).append(w)
            if len(groups) != 1:
                continue
            per = {nodes[i]: max(v) for i, v in next(iter(groups.values())).items()}
            if len(per) < 4:
                continue
            med = st.median(per.values())
            mine += [(job, step, n, round(per[n] / med - 1, 9)) for n in F['scaling']['slow_nodes'] if n in per]
    theirs = [(e['job'], e['step'], e['node'], round(e['excess'], 9)) for e in F['scaling']['slow_node_excess']]
    check('A', 'scaling.slow_node_excess / slow_excess_pct (independent recomputation)', [sorted(theirs), F['scaling']['slow_excess_pct']],
          [sorted(mine), [round(100 * min(x[3] for x in mine)), round(100 * max(x[3] for x in mine))]], f'{RET}/job_*/steps.tsv + rank reports')
    wk = load(E / 'work/rebuild_v2/figure1_measured_cost.json')['weak']['median_node_work']
    check('B', 'median per-node work within about 1% across 32/64/128 nodes', True, (max(wk) - min(wk)) / min(wk) < 0.015,
          'work/rebuild_v2/figure1_measured_cost.json')
    wdf = load(E / 'work/geometry_pilot/evaluation_pipeline_waveD/retrieved/job_2340563/aggregate/waveD_findings.json')
    check('B', 'Wave D static partition packed at 22%', 22, round(wdf['allocation']['GPU_packing_efficiency'] * 100), 'waveD_findings.json allocation')
    rf = F['waveD']['refinement_failures']['refinement_status']
    check('B', 'refinement failed on 2.9% of samples', 2.9, round(100 * rf['failed'] / sum(rf.values()), 1), 'facts.waveD')
    k8 = st.median(F['concurrency']['K8_m8']['update_seconds']); k32 = st.median(F['concurrency']['K32_m8']['update_seconds'])
    check('B', 'trajectory durations about 11 and 20 days = 22 and 40 twelve-hour segments (builder arithmetic on facts)', [11, 20, 22, 40],
          [round(F['design']['warm_updates'] * k8 / 86400), round(F['design']['fine_updates'] * k32 / 86400),
           round(F['design']['warm_updates'] * k8 / 43200), round(F['design']['fine_updates'] * k32 / 43200)], 'facts design x concurrency')
    ck = load(sorted((E / RET / 'job_2342412/C/traj0/continuous').glob('rank_*.json'))[0])['checkpoint_files']
    check('B', 'trajectory checkpoint about 0.2 GB', 0.2, round(ck[0]['bytes'] / 1e9, 1), 'C traj0 rank report checkpoint_files')
    check('B', 'RMDB conditions 24 C and 10 mM MgCl2 (52 acquired files)', [['24C'], ['MgCl2:10mM']],
          [sorted({t for f in xm for t in f['temperature']}), sorted({c for f in xm for c in f['conditions'] if 'Mg' in c})], f'{XD}/expansion_manifest.json')
    qc = list(csv.DictReader(open(E / 'work/hpc_pilot/data/data_curation_qc.csv')))
    p4 = load(E / 'work/hpc_pilot/data/P4P61_MGTI_0001_protocol.json')
    check('B', 'P4-P6 158-nt core mapped to 1GID from 202-nt and 172-nt assays', [158, [172, 202]],
          [p4['canonical_core_length'], sorted({int(x['construct_length']) for x in qc if x['reference'] == '1GID'} | {p4['full_construct_length']})],
          'work/hpc_pilot/data/data_curation_qc.csv, P4P61_MGTI_0001_protocol.json')
    obs = sorted((E / 'work/hpc_pilot/data').glob('*_observations.csv'))
    check('B', 'five RMDB deposits, twelve profiles (P4-P6 and tRNA)', [5, 12], [len({o.name.rsplit('_', 2)[0] for o in obs}), len(obs)],
          'work/hpc_pilot/data/*_observations.csv')
    rfi = load(E / 'work/resource_costing/rfam15_1_inventory.json')
    check('B', 'Rfam window 30-1,024 nt', [30, True], [rfi['length_min'], 'excluded_length_above1024' in rfi and rfi['length_max'] <= 1024],
          'work/resource_costing/rfam15_1_inventory.json')
    check('B', 'sixty RDATs (8 + 52) total 10.9 GB, all hash-verified', [60, 52, 10.9],
          [sum(1 for x in dm if x['sha256_verified']), len(new), round(sum(x['actual_bytes'] for x in dm) / 1e9, 1)], f'{XD}/admission_run/download_manifest.json')
    rv = load(E / 'work/geometry_pilot/biological_validation_pilot_v2/cohort/raw_extraction_verification.json')
    check('B', 'forward-head cohort verified 103/103 from raw files', [103, 0], [rv['n_checks'], rv['n_failed']], 'Wave F raw_extraction_verification.json')
    quals = [E / RET / 'job_2339575/B1_reliability_qualification.json'] + [E / RET / f'job_2342412/C_traj{i}_reliability_qualification.json' for i in range(4)] + \
            [E / f'{ET}/production_reliability_pilot/retrieved/snapshot_20261015T164433Z/reliability_qualification_summary.json']
    qs = [load(q) for q in quals]
    check('B', 'six restart tests, all qualified, 2,304 rank-update reports', [6, 2304], [sum(1 for q in qs if q['qualified']), sum(q['accepted_rank_update_reports'] for q in qs)],
          'six qualification JSONs')
    check('B', '192 reports pass (2305296-98)', 192, sum(x['ranks'] for x in g if x['qualified']), nfp)
    gs = load(E / 'work/geometry_pilot/geometry_retrieved/geometry_summary.json')['strong_scaling_points']
    e48 = sorted(round(p['strong_efficiency_from_8_nodes'], 2) for p in gs if p['nodes'] == 48)
    check('B', 'geometry teacher batch 384, 0.89-0.91 at 48 nodes', [384, [0.89, 0.91]], [gs[0]['global_batch'], e48],
          'work/geometry_pilot/geometry_retrieved/geometry_summary.json')
    exs = load(E / 'work/hpc_pilot/expanded_retrieved/expanded_campaign_summary.json')
    check('B', 'systems prototype scaled to 192 GPUs (2297211-16)', 192, 4 * max(int(j['allocated_nodes']) for j in exs['jobs']),
          'work/hpc_pilot/expanded_retrieved/expanded_campaign_summary.json')
    check('B', 'first-update gradient norm 88.6 (averaged over 128 ranks)', [88.6, 128],
          [round(F['reliability']['identical_norms'][0], 1), load(quals[0])['continuous_rank_reports']], 'facts.reliability, B1 qualification')
    check('B', '28% slower sustained updates in 2319595 than 2305296 (anomaly on)', 28,
          round(100 * (F['reliability']['job_2319595_s'][0] / F['reliability']['old_2305296_s'] - 1)), 'facts.reliability')
    check('B', 'GPU-hours by account: 393 on FL01-991, 1,865 on FL01-993', [393, 1865],
          [round(L['GPU_hours_by_account']['e-aif-2026fl01-991']), round(L['GPU_hours_by_account']['e-aif-2026fl01-993'])], 'scheduler-record ledger')
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, '-I', str(REPO / 'tools/build_job_ledger.py'), '--evidence-root', str(E), '--out', f'{td}/l.json'],
                       capture_output=True, text=True)
        jobs = {j['job_id'] for j in load(f'{td}/l.json')['jobs']}
    cited = [2339575, 2342412, 2340563, 2340325, 2319595, 2305296, 2305297, 2305298] + list(range(2297211, 2297217)) + list(range(2297399, 2297404))
    check('B', 'every cited job has a scheduler record', [], [j for j in cited if j not in jobs], 'scheduler-record ledger')
    check('B', 'PyTorch 2.10 and CUDA 12.9 on JUPITER', ['2.10', '12.9'],
          [r0['torch_version'][:4], load(E / 'work/sequence_prior_waveE/retrieved/job_2340325/P2_N64/run_summary.json').get('cuda',
           re.search(r'"cuda": "([0-9.]+)"', (E / 'work/sequence_prior_waveE/retrieved/job_2340325/P2_N64/run_summary.json').read_text()).group(1))],
          'C4 rank report, Wave E run_summary.json')
    # ---------------- C. design parameters recorded in design_v2.json -----------------------------
    fx = design['fixed']
    check('C', 'fixed allowances 3,000 NH (1,500 references + 1,000 fitting + 500 operations)', 3000,
          design['inference']['external_reference_and_preprocessing_NH'] + fx['shared_fitting_NH'] + fx['operations_NH'], 'design_v2.json')
    check('C', 'evaluation priced at 80% packing; 300-case development panel', [0.8, 300],
          [design['inference']['packing_efficiency'], design['inference']['dev_cases']], 'design_v2.json')
    check('C', 'data-scaling fractions 10% and 30%', [0.1, 0.3], [x['fraction'] for x in design['data_scaling']], 'design_v2.json')
    fails = [c for c in checks if c['result'] != 'PASS']
    print(json.dumps({'checks': checks, 'passed': len(checks) - len(fails), 'failed': len(fails)}, indent=1, default=str))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
