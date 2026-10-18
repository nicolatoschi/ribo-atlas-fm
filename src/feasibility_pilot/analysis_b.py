#!/usr/bin/env python3
"""Analysis B: base-resolved observation head on the frozen Wave F v2 cohort.

Reuses, unchanged, the Wave F v2 functions (forward_head_pilot.py imported read-only from a
byte-identical, hash-checked copy): sequence/secondary/C4' feature builders, chemistry blocks,
precision weights, ridge fit, nested leave-one-lineage-out alpha selection, metrics, paired
lineage bootstrap and study holdout. Adds two arms: base_context (sequence + secondary +
base-resolved block) and shuffled_base_context (same within-profile row permutation as the
Wave F shuffled-geometry control, applied to the base-resolved block).

The run first reproduces the four published Wave F arms exactly; if that fails it stops.

usage: analysis_b.py --plan plans/analysisB_plan.json --expected-sha <sha256> [--synthetic]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORIG = ROOT / "inputs/local_copies/forward_head_pilot.py"
ORIG_SHA = "1adceaa9d89b26d11a07ffeceac7394bb4e181c44ea2a26b73337402077a8ee4"
PUBLISHED = ROOT / "inputs/local_copies/v2_forward_head_results.json"
COHORT = ROOT / "inputs/local_copies/paired_core_cohort.json"
DESC = ROOT / "results/analysisB_descriptors.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if sha(ORIG) != ORIG_SHA:
    raise SystemExit("Wave F v2 code copy differs from the frozen hash")
_spec = importlib.util.spec_from_file_location("fhp_v2", ORIG)
fhp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fhp)

ARMS_B = fhp.ARMS + ("base_context", "shuffled_base_context")
COMPARISONS_B = fhp.COMPARISONS + (("secondary_context", "base_context"), ("shuffled_base_context", "base_context"),
                                   ("sequence_context", "base_context"), ("geometry_context", "base_context"))


def base_block(desc, e):
    rows = desc["examples"][e["example_id"]]["rows"]
    if len(rows) != len(e["sequence"]):
        raise ValueError("descriptor/core length mismatch")
    X = np.asarray([[v if v is not None else 0.0 for v in r["scaled"]] for r in rows], float)
    valid = np.asarray([r["valid"] for r in rows], bool)
    return X, valid


def feature_matrices_b(e, desc):
    seq = fhp.sequence_features(e)
    sec, secondary_valid = fhp.secondary_features(e)
    geo, geometry_valid = fhp.geometry_features(e)
    structural_valid = geometry_valid & secondary_valid  # identical scored-row definition to Wave F v2
    base, base_valid = base_block(desc, e)
    if not base_valid[structural_valid].all():
        raise ValueError(f"{e['example_id']}: base descriptors missing on a structurally valid row")
    stable = int(hashlib.sha256(e["example_id"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(fhp.SEED + stable)
    indices = np.flatnonzero(structural_valid & fhp.chemical_eligibility(e))
    perm = rng.permutation(indices)  # identical draw to Wave F v2 feature_matrices
    pgeo, pbase = geo.copy(), base.copy()
    pgeo[indices] = geo[perm]
    pbase[indices] = base[perm]
    raw = {"sequence_context": seq,
           "secondary_context": np.concatenate([seq, sec], axis=1),
           "geometry_context": np.concatenate([seq, sec, geo], axis=1),
           "shuffled_geometry_context": np.concatenate([seq, sec, pgeo], axis=1),
           "base_context": np.concatenate([seq, sec, base], axis=1),
           "shuffled_base_context": np.concatenate([seq, sec, pbase], axis=1)}
    chem = fhp.chemistry_group(e["chemistry"])
    mats = {a: np.concatenate([x, x * float(chem == "DMS"), x * float(chem == "SHAPE")], axis=1) for a, x in raw.items()}
    return mats, structural_valid, base.shape[1], seq.shape[1] + sec.shape[1]


def prepared_rows_b(examples, weighting, desc):
    matrices = {a: [] for a in ARMS_B}
    y, precision, profile, lineage, study, chemistry, position = [], [], [], [], [], [], []
    audit = []
    for e in examples:
        feats, structural_valid, nb, start = feature_matrices_b(e, desc)
        eligible = fhp.chemical_eligibility(e)
        observed = np.asarray(e["observed_mask"], bool)
        if np.any(observed & ~eligible):
            raise ValueError("observation at ineligible position")
        selected = observed & structural_valid
        for a in ARMS_B:
            matrices[a].append(feats[a][selected])
        b = feats["base_context"][selected][:, start:start + nb]
        s = feats["shuffled_base_context"][selected][:, start:start + nb]
        audit.append({"example_id": e["example_id"], "n_scored_rows": int(selected.sum()),
                      "scored_row_base_marginal_preserved": bool(np.array_equal(np.sort(b, 0), np.sort(s, 0))),
                      "fraction_rows_with_changed_base_block": float(np.mean(np.any(b != s, axis=1)))})
        for i in np.flatnonzero(selected):
            y.append(float(e["reactivity"][i])); precision.append(fhp.precision_of(e["stderr"][i], weighting))
            profile.append(e["example_id"]); lineage.append(e["lineage_id"]); study.append(e["study_id"])
            chemistry.append(fhp.chemistry_group(e["chemistry"])); position.append(int(i))
    return {"X": {a: np.concatenate(v) for a, v in matrices.items()}, "y": np.asarray(y), "precision": np.asarray(precision),
            "profile": np.asarray(profile), "lineage": np.asarray(lineage), "study": np.asarray(study),
            "chemistry": np.asarray(chemistry), "position": np.asarray(position), "weighting": weighting}, audit


def fit_scheme_b(rows):
    groups = sorted(set(rows["lineage"]))
    results, preds_out = {}, []
    for arm in ARMS_B:
        pred = np.full(len(rows["y"]), np.nan)
        folds = []
        for held in groups:
            train = rows["lineage"] != held
            test = ~train
            alpha, scores = fhp.inner_alpha(rows, arm, train)
            w = fhp.balanced_weights(rows, train)
            model = fhp.ridge_fit(rows["X"][arm][train], rows["y"][train], w[train], alpha)
            pred[test] = fhp.predict(model, rows["X"][arm][test])
            folds.append({"held_out_lineage": str(held), "alpha": alpha, "effective_degrees_of_freedom": model["effective_df"],
                          "inner_selection": scores})
        profiles = fhp.profile_metrics(rows, np.ones(len(pred), bool), pred)
        lineages = fhp.lineage_metrics(profiles)
        chem = {}
        for c in sorted(set(rows["chemistry"])):
            cl = fhp.lineage_metrics([p for p in profiles if p["chemistry"] == c])
            chem[c] = {"n_lineages": len(cl), "lineage_metrics": cl,
                       "equal_lineage_error_weighted_mse": float(np.mean([r["error_weighted_mse"] for r in cl])),
                       "equal_lineage_unweighted_mse": float(np.mean([r["unweighted_mse"] for r in cl]))}
        results[arm] = {"folds": folds, "profile_metrics": profiles, "lineage_metrics": lineages, "chemistry_stratified_metrics": chem,
                        "equal_lineage_error_weighted_mse": float(np.mean([r["error_weighted_mse"] for r in lineages])),
                        "equal_lineage_unweighted_mse": float(np.mean([r["unweighted_mse"] for r in lineages]))}
        for i, p in enumerate(pred):
            preds_out.append({"weighting": rows["weighting"], "arm": arm, "example_id": str(rows["profile"][i]),
                              "position_zero_based": int(rows["position"][i]), "target": float(rows["y"][i]), "prediction": float(p)})
    return results, preds_out


def study_holdout_b(rows):
    out = []
    for st in sorted(set(rows["study"])):
        test = rows["study"] == st
        tl = sorted(set(rows["lineage"][test]))
        train = ~np.isin(rows["lineage"], tl)
        rem = sorted(set(rows["lineage"][train]))
        sr = {"held_out_study": str(st), "held_out_lineages": tl, "train_lineages": rem}
        if len(rem) < 2:
            sr["status"] = "not_estimable_insufficient_training_lineages"
        else:
            sr["status"] = "executed"
            if not set(rows["chemistry"][test]) <= set(rows["chemistry"][train]):
                sr["chemistry_extrapolation"] = "Held-out chemistry absent from training (extrapolation)."
            sr["arms"] = {}
            for arm in ARMS_B:
                alpha, _ = fhp.inner_alpha(rows, arm, train)
                w = fhp.balanced_weights(rows, train)
                m = fhp.ridge_fit(rows["X"][arm][train], rows["y"][train], w[train], alpha)
                pred = np.full(len(rows["y"]), np.nan)
                pred[test] = fhp.predict(m, rows["X"][arm][test])
                sr["arms"][arm] = {"alpha": alpha, "lineage_metrics": fhp.lineage_metrics(fhp.profile_metrics(rows, test, pred))}
        out.append(sr)
    return out


def calibration_rows(examples, weighting, desc, block, b):
    """Rows for one extra shuffled arm: seq + sec + (block rows permuted with seed (SEED, stable, b))."""
    X, y_rows = [], None
    for e in examples:
        seq = fhp.sequence_features(e)
        sec, secondary_valid = fhp.secondary_features(e)
        geo, geometry_valid = fhp.geometry_features(e)
        structural_valid = geometry_valid & secondary_valid
        blk = base_block(desc, e)[0] if block == "base" else geo
        stable = int(hashlib.sha256(e["example_id"].encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng([fhp.SEED, stable, b + 1])
        idx = np.flatnonzero(structural_valid & fhp.chemical_eligibility(e))
        pb = blk.copy()
        pb[idx] = blk[rng.permutation(idx)]
        x = np.concatenate([seq, sec, pb], axis=1)
        chem = fhp.chemistry_group(e["chemistry"])
        x = np.concatenate([x, x * float(chem == "DMS"), x * float(chem == "SHAPE")], axis=1)
        sel = np.asarray(e["observed_mask"], bool) & structural_valid
        X.append(x[sel])
    return np.concatenate(X)


def fit_one_arm(rows, X):
    r = dict(rows)
    r["X"] = {"arm": X}
    groups = sorted(set(r["lineage"]))
    pred = np.full(len(r["y"]), np.nan)
    for held in groups:
        train = r["lineage"] != held
        alpha, _ = fhp.inner_alpha(r, "arm", train)
        w = fhp.balanced_weights(r, train)
        m = fhp.ridge_fit(X[train], r["y"][train], w[train], alpha)
        pred[~train] = fhp.predict(m, X[~train])
    return fhp.lineage_metrics(fhp.profile_metrics(r, np.ones(len(pred), bool), pred))


def shuffle_calibration(examples, desc, schemes, n_perm):
    out = {}
    for weighting in fhp.WEIGHTING_SCHEMES:
        rows, _ = prepared_rows_b(examples, weighting, desc)
        metric = "error_weighted_mse" if weighting == "error_weighted" else "unweighted_mse"
        for block, arm in (("base", "base_context"), ("c4prime", "geometry_context")):
            obs = {r["lineage_id"]: r[metric] for r in schemes[weighting]["arms"][arm]["lineage_metrics"]}
            obs_mean = float(np.mean(list(obs.values())))
            null_mean, null_lineage = [], []
            for b in range(n_perm):
                lm = fit_one_arm(rows, calibration_rows(examples, weighting, desc, block, b))
                null_mean.append(float(np.mean([r[metric] for r in lm])))
                null_lineage.append({r["lineage_id"]: r[metric] for r in lm})
            null_mean = np.asarray(null_mean)
            out[f"{weighting}:{arm}"] = {
                "metric": metric, "n_permutations": n_perm, "observed_equal_lineage_mse": obs_mean,
                "null_mean_mse": float(null_mean.mean()), "null_2.5_97.5": [float(np.quantile(null_mean, 0.025)), float(np.quantile(null_mean, 0.975))],
                "observed_minus_null_mean": obs_mean - float(null_mean.mean()),
                "p_one_sided_observed_mse_le_shuffled": float((1 + np.sum(null_mean <= obs_mean)) / (1 + n_perm)),
                "per_lineage_fraction_of_shuffles_beaten": {g: float(np.mean([nl[g] > obs[g] for nl in null_lineage])) for g in obs}}
    return out


def gate_b(comparisons, candidate, references):
    status = {}
    for scheme in fhp.WEIGHTING_SCHEMES:
        metric = "error_weighted_mse" if scheme == "error_weighted" else "unweighted_mse"
        for ref in references:
            c = next(x for x in comparisons if x["scheme"] == scheme and x["reference"] == ref and x["candidate"] == candidate and x["metric"] == metric)
            status[f"{scheme}:{ref}"] = bool(c["mean_difference"] > 0 and c["n_lineages_favouring_candidate"] >= 3)
    return {"criteria_met": status, "adds_information_by_frozen_gate": all(status.values())}


def reproduction_check(schemes, published):
    worst, checked = 0.0, 0
    for scheme in fhp.WEIGHTING_SCHEMES:
        for arm in fhp.ARMS:
            mine = {r["lineage_id"]: r for r in schemes[scheme]["arms"][arm]["lineage_metrics"]}
            pub = {r["lineage_id"]: r for r in published["schemes"][scheme]["arms"][arm]["lineage_metrics"]}
            for lin, r in pub.items():
                for k in ("error_weighted_mse", "unweighted_mse", "pearson_correlation"):
                    worst = max(worst, abs(mine[lin][k] - r[k])); checked += 1
            ma = [f["alpha"] for f in schemes[scheme]["arms"][arm]["folds"]]
            pa = [f["alpha"] for f in published["schemes"][scheme]["arms"][arm]["folds"]]
            if ma != pa:
                return {"passed": False, "reason": f"alpha mismatch {scheme} {arm}: {ma} vs {pa}"}
    return {"passed": worst < 1e-10, "max_abs_difference": worst, "n_values_checked": checked,
            "published_results_sha256": sha(PUBLISHED)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    plan_path = ROOT / args.plan
    if sha(plan_path) != args.expected_sha:
        raise SystemExit("Plan hash mismatch: refusing to run")
    plan = json.loads(plan_path.read_text())
    for rel, h in plan["frozen_files_sha256"].items():
        if rel == "scripts/analysis_b.py" and args.synthetic:
            continue
        if sha(ROOT / rel) != h:
            raise SystemExit(f"Frozen file changed after the plan: {rel}")
    desc = json.loads(DESC.read_text())
    examples = fhp.normalize_cohort(json.loads(COHORT.read_text()))
    if args.synthetic:
        rng = np.random.default_rng(99)
        for e in examples:
            e["reactivity"] = [float(rng.lognormal(-1.5, 1.0)) if o else None for o in e["observed_mask"]]
    schemes, preds, comparisons, audit = {}, [], [], None
    for weighting in fhp.WEIGHTING_SCHEMES:
        rows, audit_w = prepared_rows_b(examples, weighting, desc)
        audit = audit or audit_w
        if not all(a["scored_row_base_marginal_preserved"] for a in audit_w):
            raise SystemExit("shuffled base control does not preserve scored-row marginals")
        res, p = fit_scheme_b(rows)
        schemes[weighting] = {"arms": res, "study_holdout_sensitivity": study_holdout_b(rows)}
        preds += p
        metrics = ["error_weighted_mse", "unweighted_mse"] if weighting == "error_weighted" else ["unweighted_mse"]
        for metric in metrics:
            for ref, cand in COMPARISONS_B:
                comparisons.append({"scheme": weighting, **fhp.descriptive_comparison(res, ref, cand, metric)})
    calib = shuffle_calibration(examples, desc, schemes, plan["shuffle_calibration"]["n_permutations"])
    repro = reproduction_check(schemes, json.loads(PUBLISHED.read_text())) if not args.synthetic else {"passed": None, "reason": "synthetic"}
    if not args.synthetic and not repro["passed"]:
        raise SystemExit(f"Wave F reproduction failed: {repro}")
    report = {"schema": "feasibility-analysisB-v1", "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "synthetic_plumbing_test": bool(args.synthetic), "plan_sha256": args.expected_sha, "code_sha256": sha(__file__),
              "wavef_code_sha256": ORIG_SHA, "cohort_sha256": sha(COHORT), "descriptors_sha256": sha(DESC), "numpy": np.__version__,
              "n_common_observations": int(len(rows["y"])), "arms": list(ARMS_B),
              "feature_counts": {a: int(rows["X"][a].shape[1]) for a in ARMS_B},
              "shuffled_base_marginal_audit": audit, "wavef_reproduction_check": repro,
              "shuffle_calibration": calib,
              "schemes": schemes, "comparisons": comparisons,
              "interpretation_gate_base": {"frozen_rule": plan["interpretation_gate"],
                                           **gate_b(comparisons, "base_context", ("secondary_context", "shuffled_base_context"))},
              "interpretation_gate_c4prime_reproduced": gate_b(comparisons, "geometry_context", ("secondary_context", "shuffled_geometry_context"))}
    name = "synthetic_test_analysisB.json" if args.synthetic else "analysisB_results.json"
    (ROOT / "results" / name).write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    if not args.synthetic:
        (ROOT / "results" / "analysisB_held_out_predictions.json").write_text(json.dumps(preds) + "\n")
    for scheme in fhp.WEIGHTING_SCHEMES:
        m = "error_weighted_mse" if scheme == "error_weighted" else "unweighted_mse"
        print(scheme, {a: round(schemes[scheme]["arms"][a][f"equal_lineage_{m}"], 4) for a in ARMS_B})
    for c in comparisons:
        if c["candidate"] == "base_context":
            print(f"{c['scheme']:15s} {c['metric']:19s} {c['reference']:24s} -> base  mean {c['mean_difference']:+.4f} rel {c['relative_mse_reduction']:+.3f} CI [{c['descriptive_bootstrap_95_interval'][0]:+.4f},{c['descriptive_bootstrap_95_interval'][1]:+.4f}] favour {c['n_lineages_favouring_candidate']}/4")
    print("gate", report["interpretation_gate_base"]["criteria_met"], report["interpretation_gate_base"]["adds_information_by_frozen_gate"])
    for k, v in calib.items():
        print("calibration", k, "obs", round(v["observed_equal_lineage_mse"], 4), "null mean", round(v["null_mean_mse"], 4), "p", round(v["p_one_sided_observed_mse_le_shuffled"], 4))
    print("reproduction", repro)


if __name__ == "__main__":
    main()
