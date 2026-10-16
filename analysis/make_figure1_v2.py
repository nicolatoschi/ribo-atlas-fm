#!/usr/bin/env python3
"""Figure 1 (v2): measured cost per case, length dependence and weak scaling on JUPITER.

Reads only retrieved rank reports / analyses (Waves B and C, 15-16 Oct 2026). Palette:
reference categorical slots 1-2 (blue, orange) in fixed order; marker shapes and direct
labels give a second encoding for greyscale print.
"""
import glob
import json
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path('/Users/nicolatoschi/Documents/RIBO_ATLAS_FM')
AO = ROOT / 'work/geometry_pilot/ensemble_training/anomaly_off_remeasure_20261015/retrieved'
OUT = ROOT / 'work/rebuild_v2/figure1_measured_cost.png'
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#1f1f1f', '#5c5c5c', '#e3e3e3'
for f in ['/Applications/Microsoft Word.app/Contents/Resources/DFonts/Aptos.ttf',
          '/Applications/Microsoft Word.app/Contents/Resources/DFonts/Aptos-Bold.ttf']:
    if Path(f).exists(): font_manager.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'Aptos', 'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': .6, 'xtick.major.width': .6,
                     'ytick.major.width': .6, 'axes.spines.top': False, 'axes.spines.right': False})

B = json.loads((AO / 'job_2339575/waveB_analysis.json').read_text())
C = json.loads((AO / 'job_2342412/waveC_analysis.json').read_text())
nat = B['native']


def per_node_median(files, nodes, groups_of):
    per = {}
    for f in files:
        r = json.loads(Path(f).read_text()); g = groups_of(r)
        w = sum(x['mean_pass_seconds'] + x['gradient_forward_seconds'] + x['backward_recompute_seconds'] + x['teacher_forward_backward_seconds'] * 0 for x in g)
        per[nodes[r['rank'] // 4]] = max(per.get(nodes[r['rank'] // 4], 0), w)
    return st.median(per.values())


def ledger(job):
    return {l.split('\t')[0]: l.split('\t')[1].split(',') for l in (AO / job / 'steps.tsv').read_text().splitlines()[1:]}


fig, axes = plt.subplots(1, 3, figsize=(6.72, 2.25), gridspec_kw={'width_ratios': [1.15, 1, 1]})
# (a) cost per case
ax = axes[0]
configs = ['Before fix\n(14 Oct)', 'Fixed,\n1 RNA/GPU', 'Fixed,\n8 RNAs/GPU']
k8 = [323.511384, nat['B0/anomaly_off']['node_seconds_per_case'], C['unit_costs']['K8_m8']['node_seconds_per_case']]
k32 = [1340.714262, B['cost_basis']['c32_N32']['node_seconds_per_case_mean'], C['unit_costs']['K32_m8']['node_seconds_per_case']]
x = range(len(configs)); w = .36
for off, vals, col, lab in [(-w / 2 - .01, k8, BLUE, 'K = 8 (broad)'), (w / 2 + .01, k32, ORANGE, 'K = 32 (clean)')]:
    bars = ax.bar([i + off for i in x], vals, width=w, color=col, label=lab, zorder=2)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v * 1.12, f'{v:,.0f}' if v >= 10 else f'{v:.1f}', ha='center', va='bottom', fontsize=6.3, color=INK)
ax.set_yscale('log'); ax.set_ylim(1, 5000); ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(['1', '10', '100', '1,000']); ax.set_xticks(list(x)); ax.set_xticklabels(configs, fontsize=6.6)
ax.set_ylabel('Node-seconds per training case'); ax.grid(axis='y', color=GRID, lw=.5, zorder=0)
ax.legend(frameon=False, fontsize=6.4, loc='upper right', handlelength=1, borderaxespad=0)
ax.set_title('a  Cost per case', loc='left', fontsize=8, fontweight='bold', color=INK)
# (b) length dependence: per-GPU update time with one case per GPU
ax = axes[1]
pts = {8: {}, 32: {}}
for k, r in nat.items():
    if k.startswith('B3/'): pts[r['K']].setdefault(r['length'], []).append(r['update_seconds'])
pts[8].setdefault(175, []).append(nat['B0/anomaly_off']['update_seconds'])
pts[32].setdefault(175, []).extend(B['cost_basis']['c32_N32']['update_seconds'])
SLOW = {(32, 247)}  # B3/1grz_B_K32 ran on jpbo-049-19, a node 33% slower than peers in every Wave B step it joined
for K, col, mk, lab in [(8, BLUE, 'o', 'K = 8'), (32, ORANGE, 's', 'K = 32')]:
    Ls = sorted(l for l in pts[K] if (K, l) not in SLOW); ys = [st.mean(pts[K][l]) for l in Ls]
    ax.plot(Ls, ys, color=col, lw=1.6, marker=mk, ms=4.2, mec='white', mew=.6, zorder=3)
    ax.text(Ls[-1] + 12, ys[-1], lab, color=INK, fontsize=6.4, va='center')
    for (k2, l2) in SLOW:
        if k2 == K:
            v = st.mean(pts[K][l2]); ax.plot([l2], [v], marker=mk, ms=4.6, mfc='white', mec=col, mew=1, ls='none', zorder=4)
            ax.annotate('slow node', (l2, v), xytext=(l2 + 40, v + 70), fontsize=6.2, color=INK,
                        arrowprops=dict(arrowstyle='-', color=MUTED, lw=.5))
ax.set_xlim(0, 820); ax.set_ylim(0, 800); ax.set_xlabel('RNA length (nt)'); ax.set_ylabel('Seconds per update (1 RNA/GPU)')
ax.grid(color=GRID, lw=.5, zorder=0)
ax.set_title('b  Length dependence', loc='left', fontsize=8, fontweight='bold', color=INK)
# (c) weak scaling at one case per GPU, K = 32
ax = axes[2]
lb, lc = ledger('job_2339575'), ledger('job_2342412')
t32 = st.mean(B['cost_basis']['c32_N32']['update_seconds']); t64 = nat['P2/weak_N64_K32']['update_seconds']; t128 = C['C4_wide_N128_K32']['update_seconds']
m32 = per_node_median(sorted(glob.glob(str(AO / 'job_2339575/B1/continuous/rank_*.json'))), lb['B1/continuous_step'], lambda r: r['updates'][0]['groups'])
m64 = per_node_median(sorted(glob.glob(str(AO / 'job_2339575/P2/weak_N64_K32/rank_*.json'))), lb['P2/weak_N64_K32'], lambda r: r['groups'])
m128 = per_node_median(sorted(glob.glob(str(AO / 'job_2342412/C4/wide_N128_K32/rank_*.json'))), lc['C4/wide_N128_K32'], lambda r: r['groups'])
nodes = [32, 64, 128]
ax.plot(nodes, [t32, t64, t128], color=ORANGE, lw=1.6, marker='s', ms=4.5, mec='white', mew=.6, zorder=3)
ax.plot(nodes, [m32, m64, m128], color=ORANGE, lw=1.0, ls='--', marker='s', ms=4.5, mfc='white', mec=ORANGE, mew=1, zorder=3)
ax.text(132, t128, 'update\n(slowest node)', fontsize=6.2, color=INK, va='center')
ax.text(132, m128 - 18, 'median\nnode work', fontsize=6.2, color=INK, va='center')
ax.set_xscale('log', base=2); ax.set_xticks(nodes); ax.set_xticklabels(['32\n128 GPUs', '64\n256', '128\n512'], fontsize=6.6)
ax.set_xlim(26, 210); ax.set_ylim(0, 800); ax.set_xlabel('Nodes'); ax.set_ylabel('Seconds per K = 32 update')
ax.grid(axis='y', color=GRID, lw=.5, zorder=0)
ax.set_title('c  Weak scaling', loc='left', fontsize=8, fontweight='bold', color=INK)
fig.tight_layout(pad=.4, w_pad=1.1)
fig.savefig(OUT, dpi=300, facecolor='white')
summary = {'k8': k8, 'k32': k32, 'weak': {'T': [t32, t64, t128], 'median_node_work': [m32, m64, m128]},
           'length_points': {K: {L: st.mean(v) for L, v in d.items()} for K, d in pts.items()}}
(OUT.with_suffix('.json')).write_text(json.dumps(summary, indent=2, default=float) + '\n')
print(json.dumps(summary['weak']), OUT)
