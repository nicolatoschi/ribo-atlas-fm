#!/usr/bin/env python3
"""Final verification + index: (1) every file frozen by each plan is unchanged; (2) every local copy
and every RIBO-tree original still has its manifest SHA-256 (no original modified); (3) every
download still matches; (4) write results/RESULTS_INDEX.json and OUTPUT_MANIFEST.json."""
import datetime as dt, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks = {}
for name in ("analysisA_plan.json", "analysisB_plan.json"):
    plan = json.loads((ROOT / "plans" / name).read_text())
    recorded = (ROOT / "plans" / name.replace(".json", ".sha256")).read_text().split()[0]
    changed = [f for f, h in plan["frozen_files_sha256"].items() if sha(ROOT / f) != h]
    checks[name] = {"plan_sha256_matches_record": sha(ROOT / "plans" / name) == recorded, "frozen_files_changed_after_freeze": changed}
man = json.loads((ROOT / "inputs/INPUT_MANIFEST.json").read_text())
checks["local_copies_and_originals_unchanged"] = all(sha(ROOT / r["local_copy"]) == r["sha256"] == sha(r["original_path"]) for r in man["local_inputs"])
checks["downloads_unchanged"] = all(sha(ROOT / r["path"]) == r["sha256"] for r in man["public_downloads"])
A = json.loads((ROOT / "results/analysisA_results.json").read_text())
B = json.loads((ROOT / "results/analysisB_results.json").read_text())
PH = json.loads((ROOT / "results/analysisB_posthoc_correlation_calibration.json").read_text())
byid = {p["id"]: p for p in A["pairs"]}
def a(pid):
    r = byid[pid]["results"]["primary_2_8_normalised"]
    return {"n": byid[pid]["n_scored"], "class_counts": byid[pid]["class_counts"], "auc_LT": r["LT_vs_R"]["auc"],
            "auc_95ci_block_bootstrap": r["LT_vs_R"]["auc_moving_block_bootstrap_95ci"],
            "p_circular": r["LT_vs_R"]["p_circular_shift_one_sided"], "p_block": r["LT_vs_R"]["p_block_permutation_one_sided"],
            "auc_L": r["L_vs_R"]["auc"], "auc_T": r["T_vs_R"]["auc"]}
comp = lambda sch, met, ref: next({k: c[k] for k in ("mean_difference", "relative_mse_reduction", "descriptive_bootstrap_95_interval", "n_lineages_favouring_candidate")}
                                  for c in B["comparisons"] if c["scheme"] == sch and c["metric"] == met and c["reference"] == ref and c["candidate"] == "base_context")
index = {
 "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "verification": checks,
 "analysisA": {"plan_sha256": A["plan_sha256"], "primary": {k: a(k) for k in A["multiplicity"]["primary_ids"]},
               "primary_holm_adjusted_p": dict(zip(A["multiplicity"]["primary_ids"], A["multiplicity"]["holm_adjusted_p"])),
               "secondary": {k: {**a(k), "bh_q": q} for k, q in zip(A["multiplicity"]["secondary_ids"], A["multiplicity"]["secondary_bh_adjusted_p"])},
               "pooled_DCP_1M7": A["pooled_endpoint"], "not_executed": A["not_executed"]},
 "analysisB": {"plan_sha256": B["plan_sha256"], "wavef_reproduction": B["wavef_reproduction_check"],
               "equal_lineage_mse": {s: {arm: B["schemes"][s]["arms"][arm]["equal_lineage_" + ("error_weighted" if s == "error_weighted" else "unweighted") + "_mse"] for arm in B["arms"]} for s in ("error_weighted", "unweighted")},
               "base_vs": {f"{s}:{ref}": comp(s, "error_weighted_mse" if s == "error_weighted" else "unweighted_mse", ref)
                           for s in ("error_weighted", "unweighted") for ref in ("secondary_context", "shuffled_base_context", "sequence_context", "geometry_context")},
               "frozen_gate": B["interpretation_gate_base"], "shuffle_calibration_preregistered": B["shuffle_calibration"],
               "posthoc_correlation_calibration": PH["results"]},
}
(ROOT / "results/RESULTS_INDEX.json").write_text(json.dumps(index, indent=1) + "\n")
files = sorted(p for d in ("scripts", "plans", "results", "figures", "logs") for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in str(p))
files += [ROOT / "RESULTS_SUMMARY.md", ROOT / "inputs/INPUT_MANIFEST.json", ROOT / "inputs/download_log.jsonl"]
(ROOT / "OUTPUT_MANIFEST.json").write_text(json.dumps({"created_utc": index["created_utc"],
    "files": {str(p.relative_to(ROOT)): sha(p) for p in files}}, indent=1) + "\n")
print(json.dumps(checks, indent=1))
