#!/usr/bin/env python3
"""POST-HOC (not in the frozen plan B): calibrate the held-out within-profile Pearson r (a
pre-specified Wave F v2 secondary endpoint) against the SAME 500 shuffles used by the frozen MSE
calibration (identical seeds; the MSE null is recomputed and must match the frozen run).
Endpoint: equal-lineage mean of lineage-mean Pearson r. One-sided p = (1 + #{shuffle r >= obs}) / 501."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_b as B  # noqa: E402  (frozen module; imported, not modified)
ROOT = HERE.parent
res = json.loads((ROOT / "results/analysisB_results.json").read_text())
desc = json.loads(B.DESC.read_text())
examples = B.fhp.normalize_cohort(json.loads(B.COHORT.read_text()))
N = 500
out = {"status": "post_hoc_not_preregistered", "endpoint": "equal-lineage mean of held-out within-profile Pearson r", "n_permutations": N, "results": {}}
for weighting in B.fhp.WEIGHTING_SCHEMES:
    rows, _ = B.prepared_rows_b(examples, weighting, desc)
    metric = "error_weighted_mse" if weighting == "error_weighted" else "unweighted_mse"
    for block, arm in (("base", "base_context"), ("c4prime", "geometry_context")):
        lm_obs = res["schemes"][weighting]["arms"][arm]["lineage_metrics"]
        obs_r = float(np.mean([r["pearson_correlation"] for r in lm_obs]))
        null_r, null_mse = [], []
        for b in range(N):
            lm = B.fit_one_arm(rows, B.calibration_rows(examples, weighting, desc, block, b))
            null_r.append(float(np.mean([r["pearson_correlation"] for r in lm])))
            null_mse.append(float(np.mean([r[metric] for r in lm])))
        null_r = np.asarray(null_r)
        frozen = res["shuffle_calibration"][f"{weighting}:{arm}"]
        out["results"][f"{weighting}:{arm}"] = {
            "observed_mean_r": obs_r, "null_mean_r": float(null_r.mean()),
            "null_r_2.5_97.5": [float(np.quantile(null_r, 0.025)), float(np.quantile(null_r, 0.975))],
            "p_one_sided_r_ge_observed": float((1 + np.sum(null_r >= obs_r)) / (1 + N)),
            "mse_null_mean_matches_frozen_run": abs(float(np.mean(null_mse)) - frozen["null_mean_mse"]) < 1e-12,
            "observed_lineage_r": {r["lineage_id"]: r["pearson_correlation"] for r in lm_obs},
            "secondary_lineage_r": {r["lineage_id"]: r["pearson_correlation"] for r in res["schemes"][weighting]["arms"]["secondary_context"]["lineage_metrics"]}}
        v = out["results"][f"{weighting}:{arm}"]
        print(weighting, arm, "obs r", round(obs_r, 3), "null mean", round(v["null_mean_r"], 3), v["null_r_2.5_97.5"], "p", round(v["p_one_sided_r_ge_observed"], 4), "mse-null-match", v["mse_null_mean_matches_frozen_run"])
(ROOT / "results/analysisB_posthoc_correlation_calibration.json").write_text(json.dumps(out, indent=1) + "\n")
