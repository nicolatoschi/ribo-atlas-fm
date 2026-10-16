#!/usr/bin/env python3
"""Re-derive the shipped analyses, ledger and facts from an unzipped execution pack.

Every script is run from the pack's own frozen copy (evidence/...), with inputs from the pack and
outputs written only under --work. Each output is compared with the shipped file:
  - resource plan, facts (values), job ledger, Wave B/C analyses, B0 gate, reliability
    qualifications of B1, C traj0-3 and 2319595, and Wave D aggregation: byte-identical (facts:
    identical values; 'sources' paths differ only by root; ledger: identical JSON);
  - Wave E analysis: identical except the two path-valued fields results_dir and config.
Needs Python 3 with NumPy (Waves D and E) and Matplotlib (Wave D plots).
Reproduced on the build machine with Python 3.13.13, NumPy 2.5.0, Matplotlib 3.11.1.

Usage: python3 -I tools/reproduce_from_pack.py --pack <RIBO_ATLAS_FM_Execution_Pack_v2> --work <empty dir>
"""
import argparse
import filecmp
import json
import shutil
import subprocess
import sys
from pathlib import Path

ET = 'work/geometry_pilot/ensemble_training'
AO = f'{ET}/anomaly_off_remeasure_20261015'
REL = f'{AO}/reliability_anomaly_off_32e2cc0f8974'
WDD = 'work/geometry_pilot/evaluation_pipeline_waveD'
WED = 'work/sequence_prior_waveE'
REF2305296 = f'{ET}/native_fixed_site_limit_retrieved/snapshot_20261014T072248715026Z/results/job_2305296'


def run(cmd, cwd):
    r = subprocess.run([sys.executable, '-B'] + [str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-600:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    a = ap.parse_args()
    E = (a.pack / 'evidence').resolve()
    REPO = (a.pack / 'repository/ribo-atlas-fm').resolve()
    W = a.work.resolve(); W.mkdir(parents=True, exist_ok=True)
    results = []

    def record(name, ok, detail=''):
        results.append({'check': name, 'result': 'PASS' if ok else 'FAIL', 'detail': detail})

    def same(name, produced, shipped, rc_out):
        rc, out = rc_out
        ok = rc == 0 and Path(produced).is_file() and filecmp.cmp(produced, shipped, shallow=False)
        record(name, ok, '' if ok else out)

    # 1. resource plan
    rp = W / 'resource_plan_v2.json'
    same('resource_plan_v2.json (design + measured inputs -> ledger)', rp, E / 'work/resource_costing/v2/resource_plan_v2.json',
         run([E / 'work/resource_costing/v2/build_resource_plan_v2.py', '--design', E / 'work/resource_costing/v2/design_v2.json',
              '--waveB', E / f'{AO}/retrieved/job_2339575/waveB_analysis.json',
              '--waveC', E / f'{AO}/retrieved/job_2342412/waveC_analysis.json',
              '--waveD', E / f'{WDD}/retrieved/job_2340563/aggregate/waveD_findings.json', '--out', rp], W))
    plan = json.loads(rp.read_text()) if rp.is_file() else {}
    shipped_plan = json.loads((E / 'work/resource_costing/v2/resource_plan_v2.json').read_text())
    record(f"model total node-hours = {shipped_plan['total_NH']:,.1f} (as shipped)", plan.get('total_NH') == shipped_plan['total_NH'], str(plan.get('total_NH')))
    # 2. facts (unchanged builder via wrapper; rebuilt plan as input)
    fx = W / 'facts_v2.json'
    rc, out = run([REPO / 'resource_plan/rebuild_facts.py', '--evidence-root', E, '--plan', rp, '--out', fx,
                   '--compare', E / 'work/rebuild_v2/facts_v2.json'], W)
    record('facts_v2.json values identical (sources differ only by root)', rc == 0, out if rc else '')
    facts = json.loads(fx.read_text()) if fx.is_file() else {}
    shipped_facts = json.loads((E / 'work/rebuild_v2/facts_v2.json').read_text())
    record(f"request = {shipped_facts['request_NH']:,} node-hours = {shipped_facts['request_GPUh']:,} GPU-hours; model total "
           f"{shipped_facts['model_total_NH']:,.1f} (as shipped)",
           all(facts.get(k) == shipped_facts[k] for k in ('request_NH', 'request_GPUh', 'model_total_NH', 'ledger', 'monthly_NH', 'inference_share')),
           str(facts.get('request_NH')))
    # 2b. job ledger from every shipped scheduler record
    rc, out = run([REPO / 'tools/build_job_ledger.py', '--evidence-root', E, '--facts', E / 'work/rebuild_v2/facts_v2.json',
                   '--out', W / 'RIBO_JOB_LEDGER.json'], W)
    same_ledger = rc == 0 and json.loads((W / 'RIBO_JOB_LEDGER.json').read_text()) == json.loads((REPO / 'results/job_ledger/RIBO_JOB_LEDGER.json').read_text())
    record('job ledger reproduces facts spend_GPUh (all attempts, both accounts) and the shipped ledger', same_ledger, out if not same_ledger else '')
    # 3. Wave B / C analyses and gate
    same('Wave B analysis', W / 'waveB_analysis.json', E / f'{AO}/retrieved/job_2339575/waveB_analysis.json',
         run([E / f'{AO}/analyse_waveB.py', '--results', E / f'{AO}/retrieved/job_2339575', '--output', W / 'waveB_analysis.json'], W))
    same('Wave C analysis', W / 'waveC_analysis.json', E / f'{AO}/retrieved/job_2342412/waveC_analysis.json',
         run([E / f'{AO}/analyse_waveC.py', '--results', E / f'{AO}/retrieved/job_2342412', '--output', W / 'waveC_analysis.json'], W))
    same('B0 anomaly A/B gate', W / 'gate_B0.json', E / f'{AO}/retrieved/job_2339575/gate_B0.json',
         run([E / f'{AO}/evaluate_gate.py', '--results', E / f'{AO}/retrieved/job_2339575/B0', '--output', W / 'gate_B0.json'], W))
    # 4. strict reliability qualifications
    quals = [('B1 (job 2339575)', f'{AO}/retrieved/job_2339575/B1', f'{AO}/retrieved/job_2339575/B1_reliability_qualification.json',
              f'{AO}/retrieved/job_2339575/scheduler_accounting.txt', 2339575)]
    for i in range(4):
        quals.append((f'C traj{i} (job 2342412)', f'{AO}/retrieved/job_2342412/C/traj{i}',
                      f'{AO}/retrieved/job_2342412/C_traj{i}_reliability_qualification.json',
                      f'{AO}/retrieved/job_2342412/scheduler_accounting.txt', 2342412))
    for name, res, shipped, acct, job in quals:
        q = json.loads((E / shipped).read_text()); out = W / Path(shipped).name
        same(f'reliability qualification {name}', out, E / shipped,
             run([E / f'{AO}/verify_reliability_multistep.py', '--results', E / res, '--manifest', E / f'{REL}/campaign_manifest.json',
                  '--accounting', E / acct, '--job-id', job, '--account', 'e-aif-2026fl01-993',
                  '--qualified-input-reports', E / REF2305296, '--output', out,
                  '--continuous-step', q['continuous_step'], '--resume-step', q['resume_step']], W))
    P = f'{ET}/production_reliability_pilot'
    SN = f'{P}/retrieved/snapshot_20261015T164433Z'
    same('reliability qualification job 2319595', W / 'rq_2319595.json', E / f'{SN}/reliability_qualification_summary.json',
         run([E / f'{P}/reliability_frozen_aa8af773fd10/verify_reliability_results.py', '--results', E / f'{SN}/results/job_2319595',
              '--manifest', E / f'{P}/reliability_frozen_aa8af773fd10/campaign_manifest.json', '--accounting', E / f'{SN}/scheduler_accounting.txt',
              '--job-id', 2319595, '--account', 'e-aif-2026fl01-993', '--qualified-input-reports', E / REF2305296,
              '--output', W / 'rq_2319595.json'], W))
    # 5. Wave D aggregation
    rc_out = run([E / f'{WDD}/waveD_efec09f1317a/aggregate_waveD.py', '--results', E / f'{WDD}/retrieved/job_2340563',
                  '--packet', E / f'{WDD}/waveD_efec09f1317a', '--out', W / 'waveD', '--sacct', E / f'{WDD}/retrieved/job_2340563/sacct.txt',
                  '--pilot-refinement-manifest', E / f'{WDD}/waveD_efec09f1317a/provenance/refinement_packet/refinement_manifest.json',
                  '--pilot-refinement-results', E / 'work/geometry_pilot/refinement_retrieved/snapshot_20261012T195345687174Z/results/job_2298016',
                  '--full-cohorts', 25500], W)
    for f in ['waveD_findings.json', 'waveD_findings.txt', 'waveD_case_costs.csv', 'waveD_sample_metrics.csv']:
        same(f'Wave D aggregation {f}', W / 'waveD' / f, E / f'{WDD}/retrieved/job_2340563/aggregate/{f}', rc_out)
    # 6. Wave E analysis (path-valued fields excluded)
    rc, out = run([E / f'{WED}/waveE_05ec1ab6b438/analyze_waveE.py', '--results', E / f'{WED}/retrieved/job_2340325',
                   '--out', W / 'waveE', '--config', E / f'{WED}/waveE_05ec1ab6b438/waveE_config.json'], W)
    ok = rc == 0 and (W / 'waveE/waveE_analysis.json').is_file()
    if ok:
        new = json.loads((W / 'waveE/waveE_analysis.json').read_text())
        old = json.loads((E / f'{WED}/retrieved/job_2340325/analysis/waveE_analysis.json').read_text())
        diff = sorted(k for k in set(new) | set(old) if new.get(k) != old.get(k))
        ok = set(diff) <= {'results_dir', 'config'}
        out = 'differing keys: ' + ', '.join(diff)
    record('Wave E analysis (all values; path fields results_dir/config excluded)', ok, out)
    fails = [r for r in results if r['result'] != 'PASS']
    print(json.dumps({'checks': results, 'passed': len(results) - len(fails), 'failed': len(fails)}, indent=1))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
