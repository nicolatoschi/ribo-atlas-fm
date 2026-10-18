#!/usr/bin/env python3
"""POST-HOC, DESCRIPTIVE ONLY (not in the frozen plan): global ligand-response magnitude per pair,
computed after the frozen Analysis A run to help interpret null pairs. No p-values."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
tab = json.loads((ROOT / "results/analysisA_per_position.json").read_text())
out = {"status": "post_hoc_descriptive_not_preregistered", "pairs": {}}
for pid, rows in tab.items():
    a = np.array([r["apo_norm"] for r in rows]); h = np.array([r["holo_norm"] for r in rows]); d = h - a
    sd = [r["delta_norm_sd"] for r in rows]
    z = np.abs(d) / np.array(sd) if all(s is not None and s > 0 for s in sd) else None
    out["pairs"][pid] = {"n": len(rows), "median_abs_delta_norm": float(np.median(np.abs(d))),
                         "p90_abs_delta_norm": float(np.quantile(np.abs(d), 0.9)),
                         "spearman_apo_vs_holo": float(spearmanr(a, h).statistic),
                         "fraction_abs_delta_gt_2sd": float(np.mean(z > 2)) if z is not None else None}
    print(f"{pid:30s} median|d|={out['pairs'][pid]['median_abs_delta_norm']:.3f} p90|d|={out['pairs'][pid]['p90_abs_delta_norm']:.3f} rho={out['pairs'][pid]['spearman_apo_vs_holo']:.3f} frac|d|>2sd={out['pairs'][pid]['fraction_abs_delta_gt_2sd']}")
(ROOT / "results/analysisA_posthoc_response_magnitude.json").write_text(json.dumps(out, indent=1) + "\n")
