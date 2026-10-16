#!/usr/bin/env python3
"""Build the RIBO-ATLAS-FM JUPITER job ledger from the shipped scheduler records.

Reads every scheduler record under <evidence-root>/work (scheduler_accounting.txt, sacct*.txt and
native175_requeue_accounting_sacct.txt, all pipe-separated sacct exports). It keeps one record per
parent job and attempt: a terminal state is preferred over RUNNING/PENDING, then the latest
retrieval snapshot. A REQUEUED attempt counts as its own attempt. GPU-hours are
parent elapsed seconds x allocated nodes x 4 GPUs / 3,600, which is allocation, not utilisation.
When a record carries no account field, the account comes from the campaign's submission
records (*submission_records*.json).

With --facts, the per-account and per-wave totals are compared with facts_v2.json spend_GPUh.
Usage: python3 -I build_job_ledger.py --evidence-root <pack>/evidence [--facts facts_v2.json] [--out ledger.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

RECORD_NAMES = re.compile(r'^(scheduler_accounting\.txt|sacct.*\.txt|native175_requeue_accounting_sacct\.txt)$')
SKIP_PARTS = {'execution_pack', 'extended_execution_pack', '__pycache__'}
TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'REQUEUED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED'}
WAVES = {2339575: 'waveB', 2342412: 'waveC', 2340563: 'waveD', 2340325: 'waveE'}


def skipped(rel):
    parts = rel.split('/')
    return any(p in SKIP_PARTS or p.startswith(('qa_', 'delivery_')) or p == 'prior_revision' for p in parts)


def submission_accounts(work):
    acc = {}
    for f in sorted(work.rglob('*submission_records*.json')):
        rel = f.relative_to(work).as_posix()
        if skipped(rel):
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue

        def walk(o, ctx):
            if isinstance(o, dict):
                a = o.get('account', ctx)
                a = a if isinstance(a, str) and a.startswith('e-aif-') else ctx
                j = o.get('job_id', o.get('id'))
                if isinstance(j, int) and j > 2_000_000 and a:
                    acc.setdefault(j, set()).add(a)
                for v in o.values():
                    walk(v, a)
            elif isinstance(o, list):
                for v in o:
                    walk(v, ctx)
        walk(data, None)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-root', type=Path, required=True)
    ap.add_argument('--facts', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    root = a.evidence_root.resolve(); work = root / 'work'
    files = sorted(p for p in work.rglob('*') if p.is_file() and RECORD_NAMES.match(p.name) and not skipped(p.relative_to(work).as_posix()))
    best = {}
    for p in files:
        rel = p.relative_to(root).as_posix()
        m = re.search(r'snapshot_(\d{8}T\d+Z)', rel)
        snap = m.group(1) if m else ''
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        header = lines[0].split('|') if lines and lines[0].startswith('JobID') else None
        for line in lines:
            x = line.split('|')
            if not x[0].isdigit():
                continue
            job = int(x[0])
            if header:
                d = dict(zip(header, x)); elapsed, nodes, state = int(d['ElapsedRaw']), int(d['NNodes']), d['State']
            else:
                state = next((y for y in x[1:] if y.split(' ')[0] in TERMINAL | {'RUNNING', 'PENDING'}), '?')
                nums = [y for y in x[1:] if y.isdigit()]
                elapsed, nodes = int(nums[0]), int(nums[1])
            acct = next((y for y in x if y.startswith('e-aif-')), None)
            st = state.split(' ')[0]
            key = (job, 'REQUEUED') if st == 'REQUEUED' else (job, 'final')
            rank = (st in TERMINAL, snap, rel)
            rec = {'job_id': job, 'attempt': 'requeued' if st == 'REQUEUED' else 'final', 'state': state, 'elapsed_s': elapsed,
                   'nodes': nodes, 'account': acct, 'account_source': 'sacct' if acct else None, 'record': rel}
            if key not in best or rank >= best[key][0]:
                best[key] = (rank, rec)
    subs = submission_accounts(work)
    ledger, problems = [], []
    for key in sorted(best):
        rec = best[key][1]
        if not rec['account']:
            cands = subs.get(rec['job_id'], set())
            if len(cands) == 1:
                rec['account'] = next(iter(cands)); rec['account_source'] = 'submission records'
            else:
                problems.append(f"job {rec['job_id']}: account not determinable ({sorted(cands)})")
        rec['GPU_hours'] = rec['elapsed_s'] * rec['nodes'] * 4 / 3600
        ledger.append(rec)
    by_acct = {}
    for r in ledger:
        by_acct[r['account']] = by_acct.get(r['account'], 0.0) + r['GPU_hours']
    spend = {'before_15oct_991': sum(r['GPU_hours'] for r in ledger if r['account'] == 'e-aif-2026fl01-991' and r['job_id'] < 2339000),
             'before_15oct_993': sum(r['GPU_hours'] for r in ledger if r['account'] == 'e-aif-2026fl01-993' and r['job_id'] < 2339000)}
    for j, w in WAVES.items():
        spend[w] = sum(r['GPU_hours'] for r in ledger if r['job_id'] == j)
    out = {'description': __doc__.split('\n\n')[0], 'records_read': [p.relative_to(root).as_posix() for p in files],
           'attempts': len(ledger), 'attempts_by_account': {k: sum(1 for r in ledger if r['account'] == k) for k in by_acct},
           'GPU_hours_by_account': {k: round(v, 2) for k, v in by_acct.items()},
           'spend_GPUh_recomputed': {k: round(v, 2) for k, v in spend.items()},
           'total_GPU_hours': round(sum(by_acct.values()), 2), 'problems': problems, 'jobs': ledger}
    if a.facts:
        F = json.loads(a.facts.read_text())['spend_GPUh']
        cmp = {k: {'facts': F[k], 'ledger': round(v, 1), 'match': round(v, 1) == F[k]} for k, v in spend.items()}
        total = round(sum(round(v, 1) for v in spend.values()), 1)
        cmp['total'] = {'facts': F['total'], 'ledger_sum_of_rounded': total, 'match': total == F['total']}
        out['facts_comparison'] = cmp
        if not all(c['match'] for c in cmp.values()):
            problems.append('ledger does not reproduce facts spend_GPUh')
    text = json.dumps(out, indent=1) + '\n'
    if a.out:
        a.out.write_text(text)
    print(json.dumps({k: out[k] for k in out if k not in ('jobs', 'records_read')}, indent=1))
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
