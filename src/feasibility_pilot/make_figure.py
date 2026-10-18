#!/usr/bin/env python3
"""Two-panel figure (half-page width): A = Analysis A ligand-response localisation AUCs;
B = Analysis B held-out MSE change of the base-resolved head versus each reference arm."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = json.loads((ROOT / "results/analysisA_results.json").read_text())
B = json.loads((ROOT / "results/analysisB_results.json").read_text())
PH = json.loads((ROOT / "results/analysisB_posthoc_correlation_calibration.json").read_text())

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
TIER = {"primary": ("#2a78d6", "o", "primary"), "replication": ("#eb6834", "s", "replication (same riboswitch)"),
        "extension": ("#1baf7a", "^", "extension (other riboswitches)")}
SYS = {"cdigmp": "c-di-GMP", "adocbl": "AdoCbl", "adenine": "adenine", "glycine": "glycine", "fmn": "FMN", "tpp": "TPP", "ydao": "ydaO"}
SRC = {"DCP": "DCP", "K2011": "K2011", "PZ6": "PZ6", "M2": "M2"}
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK, "axes.linewidth": 0.6})

fig = plt.figure(figsize=(3.5, 7.0), dpi=300)
gs = fig.add_gridspec(2, 1, height_ratios=[2.35, 1.0], hspace=0.42)

# ---------------- Panel A ----------------
ax = fig.add_subplot(gs[0])
pairs = [p for p in A["pairs"] if p["tier"] in TIER]
order = ["primary", "replication", "extension"]
pairs.sort(key=lambda p: (order.index(p["tier"]), list(SYS).index(p["system"])))
labels, y = [], 0
ypos = []
for p in pairs:
    r = p["results"]["primary_2_8_normalised"]["LT_vs_R"]
    col, mk, _ = TIER[p["tier"]]
    lo, hi = r["auc_moving_block_bootstrap_95ci"]
    sig = r["p_circular_shift_one_sided"] < 0.05
    ax.plot([lo, hi], [y, y], color=col, lw=1.2, solid_capstyle="round", zorder=2)
    ax.scatter([r["auc"]], [y], s=26, marker=mk, facecolor=col if sig else "white", edgecolor=col, linewidth=1.1, zorder=3)
    parts = p["id"].split("_")
    chem = parts[2] if len(parts) > 2 else ""
    labels.append(f"{SYS[p['system']]} {SRC.get(parts[1], parts[1])} {chem}")
    ypos.append(y)
    y += 1
pool = A["pooled_endpoint"]
y += 0.4
ax.scatter([pool["mean_auc"]], [y], s=40, marker="D", facecolor=INK, edgecolor=INK, zorder=3)
ax.text(pool["mean_auc"] + 0.03, y, f"p = {pool['p_independent_circular_shifts']:.3f}", va="center", fontsize=6.5, color=INK)
labels.append("pooled: 5 riboswitches, DCP 1M7")
ypos.append(y)
ax.axvline(0.5, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=6.3)
ax.invert_yaxis()
ax.set_xlim(0.2, 1.0)
ax.set_xlabel("AUC: |Δreactivity| at ligand + tertiary contacts vs rest (95% block bootstrap)", fontsize=6.3)
ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("A  Apo→holo change vs a-priori crystal contact classes", loc="left", fontsize=7.4, fontweight="bold", color=INK)
handles = [plt.Line2D([], [], marker=m, color=c, ls="", markersize=4.5, label=l) for c, m, l in TIER.values()]
handles.append(plt.Line2D([], [], marker="o", color=INK2, markerfacecolor="white", ls="", markersize=4.5, label="open: circular-shift p ≥ 0.05"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.32, -0.095), ncol=2, fontsize=5.8, frameon=False,
          handletextpad=0.3, columnspacing=0.8)

# ---------------- Panel B ----------------
bx = fig.add_subplot(gs[1])
refs = [("secondary_context", "secondary"), ("shuffled_base_context", "shuffled base"),
        ("geometry_context", "C4′ geometry"), ("sequence_context", "sequence")]
schemes = [("error_weighted", "error_weighted_mse", "#2a78d6", "o", "error-weighted (primary)"),
           ("unweighted", "unweighted_mse", "#eb6834", "s", "unweighted refit")]
yt, yl = [], []
for i, (ref, lab) in enumerate(refs):
    for j, (sch, met, col, mk, _) in enumerate(schemes):
        c = next(x for x in B["comparisons"] if x["scheme"] == sch and x["metric"] == met and x["reference"] == ref and x["candidate"] == "base_context")
        refmean = c["mean_difference"] / c["relative_mse_reduction"]
        yy = i + (j - 0.5) * 0.36
        lo, hi = (100 * v / refmean for v in c["descriptive_bootstrap_95_interval"])
        bx.plot([lo, hi], [yy, yy], color=col, lw=1.2, solid_capstyle="round", zorder=2)
        bx.scatter([100 * c["relative_mse_reduction"]], [yy], s=24, marker=mk, color=col, zorder=3)
        d = np.array(list(c["lineage_differences_reference_minus_candidate"].values())) * 100 / refmean
        bx.scatter(d, [yy] * len(d), s=6, color=INK2, alpha=0.7, zorder=2.5, linewidths=0)
    yt.append(i)
    yl.append(f"vs {lab}")
bx.axvline(0, color=INK2, lw=0.7, ls=(0, (3, 2)))
bx.set_yticks(yt)
bx.set_yticklabels(yl, fontsize=6.5)
bx.invert_yaxis()
bx.set_xlabel("held-out MSE reduction by base-resolved head (%)\n(mean, 95% lineage bootstrap; grey = 4 held-out lineages)", fontsize=6.5)
bx.grid(axis="x", color=GRID, lw=0.5)
bx.set_axisbelow(True)
for s in ("top", "right"):
    bx.spines[s].set_visible(False)
bx.set_title("B  Base-resolved observation head: frozen MSE gate not met", loc="left", fontsize=7.4, fontweight="bold", color=INK)
hb = [plt.Line2D([], [], marker=m, color=c, ls="", markersize=4.5, label=l) for _, _, c, m, l in schemes]
bx.legend(handles=hb, loc="upper center", bbox_to_anchor=(0.32, -0.30), ncol=2, fontsize=5.8, frameon=False,
          handletextpad=0.3, columnspacing=0.8)
ph = PH["results"]["error_weighted:base_context"]
bx.text(-0.62, -0.47, f"Held-out Pearson r (post hoc): base {ph['observed_mean_r']:.2f} vs 500 shuffles {ph['null_mean_r']:.2f} (p = {ph['p_one_sided_r_ge_observed']:.3f})",
        transform=bx.transAxes, fontsize=5.9, color=INK2)
fig.savefig(ROOT / "figures/feasibility_pilot_figure.png", bbox_inches="tight", facecolor="white")
print("saved")
