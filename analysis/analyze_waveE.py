#!/usr/bin/env python3
"""Analyse a Wave E results directory (CPU only; numpy only).

Scaling: for each completed run, the median over timed steps of the rank-max step time
T_W, with a bootstrap CI (timed steps resampled). Strong-scaling efficiency relative to the
reference run (default P1_N8, 32 GPUs): E = (T_ref / T_W) / (W / W_ref), with a CI from
independent bootstraps of both runs. Tokens/s = model input tokens (nucleotides + <cls> +
<eos>, no padding) summed over timed steps / summed rank-max step time.

Consistency: every run executes the same steps (same sequences, masks, normaliser), so the
per-step losses of all runs should agree to bf16 precision; the maximum deviation from the
reference run is reported.

Learning: paired evaluation files (identical masks before/after) from the train run. The endpoint
definitions are read from waveE_config.json next to this script (or --config).
PRIMARY (pre-specified): held-out-family CE at the first evaluation step with
epochs_seen = step * global_batch / n_train >= 1.0 (step 12 in the production design) versus
step 0, with the family bootstrap interval.
SECONDARY: the last evaluation step (step 72, about 6.4 epochs of a small corpus), explicitly
labelled multi-epoch; the in-family split; and the full epoch-boundary curve.
Caveat carried into every report: RNA-FM was pretrained on RNAcentral, which very likely already
contains these Rfam sequences; the split controls leakage within Wave E only.
Delta CE = (sum CE_after - sum CE_before) / masked positions (token-weighted, primary) and
the family-macro mean of per-family deltas. 95% percentile bootstrap intervals resample
held-out families (primary family per sequence) with replacement; a group-level
(clan/shared-sequence component) bootstrap is reported as a sensitivity analysis.
Perplexity ratio = exp(Delta CE).
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_run(path):
    summary = path / 'run_summary.json'
    if not summary.exists():
        return None
    s = json.loads(summary.read_text())
    steps = []
    if (path / 'steps.jsonl').exists():
        steps = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines() if line.strip()]
    return {'dir': path, 'summary': s, 'steps': steps}


def boot_median(x, rng, reps):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    return np.median(x[idx], axis=1)


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def read_eval(path):
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    for r in rows:
        r['masked_positions'] = int(r['masked_positions'])
        r['ce_sum'] = float(r['ce_sum'])
        r['correct'] = int(r['correct'])
    return rows


def paired(before, after, unit, rng, reps):
    b = {r['seq_sha256']: r for r in before}
    a = {r['seq_sha256']: r for r in after}
    assert set(a) == set(b), 'before/after evaluation sets differ'
    agg = defaultdict(lambda: np.zeros(4))
    for k, rb in b.items():
        ra = a[k]
        assert ra['masked_positions'] == rb['masked_positions']
        agg[rb[unit]] += [rb['ce_sum'], ra['ce_sum'], rb['masked_positions'], ra['correct'] - rb['correct']]
    units = sorted(agg)
    M = np.array([agg[u] for u in units])
    cb, ca, n = M[:, 0], M[:, 1], M[:, 2]
    delta = (ca.sum() - cb.sum()) / n.sum()
    per_unit = (ca - cb) / n
    idx = rng.integers(0, len(units), size=(reps, len(units)))
    boot_micro = (ca[idx].sum(1) - cb[idx].sum(1)) / n[idx].sum(1)
    boot_macro = per_unit[idx].mean(1)
    return {
        'units': len(units), 'masked_positions': int(n.sum()),
        'ce_before': float(cb.sum() / n.sum()), 'ce_after': float(ca.sum() / n.sum()),
        'perplexity_before': float(math.exp(cb.sum() / n.sum())), 'perplexity_after': float(math.exp(ca.sum() / n.sum())),
        'delta_ce_token_weighted': float(delta), 'delta_ce_token_weighted_ci95': ci(boot_micro),
        'perplexity_ratio': float(math.exp(delta)), 'perplexity_ratio_ci95': [math.exp(x) for x in ci(boot_micro)],
        'delta_ce_unit_macro': float(per_unit.mean()), 'delta_ce_unit_macro_ci95': ci(boot_macro),
        'fraction_units_improved': float((per_unit < 0).mean()),
        'accuracy_change': float(M[:, 3].sum() / n.sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', default='P1_N8')
    p.add_argument('--train-run', default='P2_N64')
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--loss-tolerance', type=float, default=0.02, help='flag per-step loss deviations above this (nats)')
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parent / 'waveE_config.json')
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    runs = {}
    for d in sorted(x for x in args.results.iterdir() if x.is_dir()):
        r = load_run(d)
        if r:
            runs[d.name] = r
    cfg = json.loads(args.config.read_text())
    primary_spec = cfg['primary_endpoint']
    report = {'results_dir': str(args.results), 'config': str(args.config), 'primary_endpoint_spec': primary_spec,
              'secondary_endpoints_spec': cfg['secondary_endpoints'], 'leakage_note': cfg['leakage_note'],
              'runs': {}, 'scaling': {}, 'loss_consistency': {}, 'learning': {}}
    ledger = args.results / 'steps.tsv'
    if ledger.exists():
        report['ledger'] = list(csv.DictReader(open(ledger), delimiter='\t'))

    timed = {}
    for name, r in runs.items():
        s = r['summary']
        entry = {'status': s.get('status'), 'mode': s.get('mode'), 'world_size': s.get('world_size'), 'nodes': s.get('nodes'),
                 'attention': s.get('attention'), 'timings': s.get('timings'), 'wall_seconds': s.get('wall_seconds_since_start')}
        ts = [x for x in r['steps'] if x.get('phase') == 'timed']
        if s.get('status') == 'completed' and ts:
            t = np.array([x['t_step_max'] for x in ts])
            tokens = np.array([x['real_tokens'] for x in ts])
            timed[name] = (t, s['world_size'])
            bm = boot_median(t, rng, 2000)
            entry.update({
                'timed_steps': len(ts), 'median_step_s': float(np.median(t)), 'median_step_s_ci95': ci(bm),
                'p10_p90_step_s': [float(np.percentile(t, 10)), float(np.percentile(t, 90))],
                'tokens_per_s': float(tokens.sum() / t.sum()), 'tokens_per_s_per_gpu': float(tokens.sum() / t.sum() / s['world_size']),
                'nucleotides_per_s': float(sum(x['nucleotides'] for x in ts) / t.sum()),
                'rank_imbalance_median': float(np.median([x['t_step_max'] / x['t_step_min'] for x in ts])),
                'data_prep_share_median': float(np.median([x['t_data_max'] / x['t_step_max'] for x in ts])),
                'peak_alloc_gib_max': max(x['peak_alloc_gib_max'] for x in r['steps']),
                'peak_reserved_gib_max': max(x['peak_reserved_gib_max'] for x in r['steps']),
                'host_rss_peak_gib_max': max(x['host_rss_peak_gib_max'] for x in r['steps']),
                'padding_fraction': float(1 - tokens.sum() / sum(x['padded_tokens'] for x in ts)),
            })
        report['runs'][name] = entry

    if args.reference in timed:
        t_ref, w_ref = timed[args.reference]
        b_ref = boot_median(t_ref, rng, 2000)
        for name, (t, w) in timed.items():
            b = boot_median(t, rng, 2000)
            eff = (np.median(t_ref) / np.median(t)) / (w / w_ref)
            report['scaling'][name] = {'world_size': w, 'gpus_ratio_vs_reference': w / w_ref,
                                       'speedup_vs_reference': float(np.median(t_ref) / np.median(t)),
                                       'efficiency_vs_reference': float(eff),
                                       'efficiency_ci95': ci((b_ref / b) / (w / w_ref)),
                                       'attention': report['runs'][name]['attention']}
    ref_steps = {x['step']: x['loss'] for x in runs[args.reference]['steps']} if args.reference in runs else {}
    for name, r in runs.items():
        common = [x for x in r['steps'] if x['step'] in ref_steps]
        if common and name != args.reference:
            dev = [abs(x['loss'] - ref_steps[x['step']]) for x in common]
            report['loss_consistency'][name] = {'steps_compared': len(common), 'max_abs_loss_diff_vs_reference': float(max(dev)),
                                                'within_tolerance': bool(max(dev) <= args.loss_tolerance)}

    train = runs.get(args.train_run)
    if train and train['summary'].get('status') == 'completed':
        steps = sorted(int(k) for k in train['summary']['evaluations'])
        G, n_train = train['summary']['global_batch_sequences'], train['summary']['n_train']
        epochs = {s: s * G / n_train for s in steps}
        first, last = steps[0], steps[-1]
        one_epoch = [s for s in steps if s > first and epochs[s] >= 1.0]
        primary_step = one_epoch[0] if one_epoch else None
        report['primary_step'] = primary_step
        report['primary_step_epochs'] = epochs.get(primary_step)
        report['secondary_multi_epoch_step'] = last
        report['secondary_multi_epoch_epochs'] = epochs[last]
        for split in ['heldout_family', 'val_in_family']:
            fb = train['dir'] / f'eval_step{first:04d}_{split}.tsv'
            if not fb.exists():
                continue
            before = read_eval(fb)
            entry = {'before_step': first}

            def compare(step, label):
                f = train['dir'] / f'eval_step{step:04d}_{split}.tsv'
                if step is None or not f.exists():
                    return None
                after = read_eval(f)
                return {'label': label, 'after_step': step, 'epochs_seen': epochs[step],
                        'family_bootstrap': paired(before, after, 'primary_family', rng, args.bootstrap),
                        'group_bootstrap': paired(before, after, 'group_id', rng, args.bootstrap)}
            is_primary_split = split == primary_spec['split']
            entry['first_epoch'] = compare(primary_step, 'PRIMARY (pre-specified)' if is_primary_split else 'secondary: in-family contrast at the primary step')
            entry['multi_epoch_final'] = compare(last, 'SECONDARY: multi-epoch result on a small corpus')
            curve = []
            for s in steps:
                f = train['dir'] / f'eval_step{s:04d}_{split}.tsv'
                if f.exists():
                    rows = read_eval(f)
                    n = sum(x['masked_positions'] for x in rows)
                    ce = sum(x['ce_sum'] for x in rows) / n
                    curve.append({'step': s, 'ce_per_token': ce, 'perplexity': math.exp(ce), 'accuracy': sum(x['correct'] for x in rows) / n})
            entry['curve'] = curve
            by_step = {}
            for s in steps[1:]:
                f = train['dir'] / f'eval_step{s:04d}_{split}.tsv'
                if f.exists():
                    fb_s = paired(before, read_eval(f), 'primary_family', rng, args.bootstrap)
                    by_step[s] = {k: fb_s[k] for k in ['delta_ce_token_weighted', 'delta_ce_token_weighted_ci95', 'perplexity_ratio', 'fraction_units_improved']}
                    by_step[s]['epochs_seen'] = s * train['summary']['global_batch_sequences'] / train['summary']['n_train']
            entry['paired_change_vs_step0_by_eval_step'] = by_step
            report['learning'][split] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'waveE_analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['Wave E analysis', f'results: {args.results}', '', 'Timing (timed steps; rank-max step time):']
    for name, e in report['runs'].items():
        if 'median_step_s' in e:
            lines.append(f"  {name:14s} W={e['world_size']:4d} {e['attention']:8s} median {e['median_step_s']:.4f}s "
                         f"[{e['median_step_s_ci95'][0]:.4f},{e['median_step_s_ci95'][1]:.4f}] tok/s {e['tokens_per_s']:.4g} "
                         f"({e['tokens_per_s_per_gpu']:.4g}/GPU) HBM {e['peak_alloc_gib_max']:.2f}/{e['peak_reserved_gib_max']:.2f} GiB RSS {e['host_rss_peak_gib_max']:.2f} GiB")
        else:
            lines.append(f"  {name:14s} status {e['status']}")
    if report['scaling']:
        lines.append(f'Strong scaling vs {args.reference}: E = (T_ref/T_W)/(W/W_ref)')
        for name, e in report['scaling'].items():
            lines.append(f"  {name:14s} x{e['gpus_ratio_vs_reference']:.0f} GPUs speedup {e['speedup_vs_reference']:.3f} "
                         f"E {e['efficiency_vs_reference']:.3f} [{e['efficiency_ci95'][0]:.3f},{e['efficiency_ci95'][1]:.3f}]")
    for name, e in report['loss_consistency'].items():
        lines.append(f"  loss vs reference {name}: max |diff| {e['max_abs_loss_diff_vs_reference']:.2e} over {e['steps_compared']} steps")
    if report['learning']:
        lines.append(f"Learning endpoints (paired masks; 95% bootstrap over held-out families). Note: {report['leakage_note']}")
    for split, e in report['learning'].items():
        for key in ['first_epoch', 'multi_epoch_final']:
            c = e.get(key)
            if not c:
                lines.append(f"{split} {key}: not available")
                continue
            fb, gb = c['family_bootstrap'], c['group_bootstrap']
            lines.append(f"{c['label']} | {split}: step {e['before_step']} -> {c['after_step']} ({c['epochs_seen']:.3f} epochs): "
                         f"CE {fb['ce_before']:.5f} -> {fb['ce_after']:.5f} nats (ppl {fb['perplexity_before']:.4f} -> {fb['perplexity_after']:.4f}); "
                         f"delta {fb['delta_ce_token_weighted']:+.5f} 95% CI [{fb['delta_ce_token_weighted_ci95'][0]:+.5f},{fb['delta_ce_token_weighted_ci95'][1]:+.5f}] "
                         f"over {fb['units']} families; group CI [{gb['delta_ce_token_weighted_ci95'][0]:+.5f},{gb['delta_ce_token_weighted_ci95'][1]:+.5f}] "
                         f"over {gb['units']} groups; {100 * fb['fraction_units_improved']:.1f}% families improved")
        lines.append(f"  {split} curve (paired change vs step 0):")
        for st, v in e['paired_change_vs_step0_by_eval_step'].items():
            lines.append(f"    step {st:3d} ({v['epochs_seen']:.2f} epochs): delta {v['delta_ce_token_weighted']:+.5f} "
                         f"[{v['delta_ce_token_weighted_ci95'][0]:+.5f},{v['delta_ce_token_weighted_ci95'][1]:+.5f}]")
    (args.out / 'waveE_analysis.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
