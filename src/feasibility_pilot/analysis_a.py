#!/usr/bin/env python3
"""Analysis A: ligand-response localisation (runs only against an unchanged frozen plan).

usage: analysis_a.py --plan plans/analysisA_plan.json --expected-sha <sha256> [--synthetic]

--synthetic replaces every reactivity by seeded random numbers (plumbing test before the
freeze; writes results/synthetic_test_analysisA.json and is never a scientific result).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from rdat import read_rdat  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def eligible(chem, base):
    if chem == "DMS":
        return base in "AC"
    if chem == "CMCT":
        return base in "GU"
    return True  # SHAPE-family (1M7, NMIA): every nucleotide


def norm_scale(v, frac_drop, frac_avg):
    s = np.sort(np.asarray(v))[::-1]
    n = len(s)
    k2 = int(round(frac_drop * n))
    k8 = max(1, int(round(frac_avg * n)))
    return float(np.mean(s[k2:k2 + k8]))


def auc(r, pos, neg):
    x = np.concatenate([r[pos], r[neg]])
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    rk = rankdata(x)
    n1, n0 = int(pos.sum()), int(neg.sum())
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stat_for(r, lab, contrast):
    if contrast == "LT_vs_R":
        return auc(r, np.isin(lab, ["L", "T"]), lab == "R")
    if contrast == "L_vs_R":
        return auc(r, lab == "L", lab == "R")
    if contrast == "T_vs_R":
        return auc(r, lab == "T", lab == "R")
    raise ValueError(contrast)


def top_decile_enrichment(r, lab):
    n = len(r)
    k = int(math.ceil(0.1 * n))
    order = np.argsort(-r, kind="stable")[:k]
    s = np.isin(lab, ["L", "T"])
    base = s.mean()
    return float(s[order].mean() / base) if base > 0 else float("nan")


def circular_null(r, lab, fn):
    n = len(r)
    return np.asarray([fn(r, np.roll(lab, k)) for k in range(n)])  # k=0 is the observed labelling


def block_perm_null(r, lab, fn, block, n_perm, rng):
    n = len(r)
    blocks = [lab[i:i + block] for i in range(0, n, block)]
    out = np.empty(n_perm)
    for b in range(n_perm):
        order = rng.permutation(len(blocks))
        out[b] = fn(r, np.concatenate([blocks[i] for i in order]))
    return out


def block_bootstrap_ci(r, lab, fn, block, n_boot, rng):
    n = len(r)
    nb = int(math.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        v = fn(r[idx], lab[idx])
        if np.isfinite(v):
            vals.append(v)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))], len(vals)


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def bh(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, ps[i] * m / (rank + 1))
        adj[i] = prev
    return adj.tolist()


def load_pair(pair, classes, plan, synthetic_rng=None):
    system = classes["systems"][pair["system"]]
    struct = system if pair.get("structure", "primary") == "primary" else system["sensitivity"][pair["structure"]]
    cmap = struct["constructs"][pair["construct"]]["mapping"]
    res = {a["chain_index"]: a for a in struct["residues"]}
    apo = read_rdat(ROOT / pair["apo"]["file"])
    holo = read_rdat(ROOT / pair["holo"]["file"])
    if apo["sequence"].upper() != holo["sequence"].upper():
        raise ValueError(f"{pair['id']}: apo/holo construct sequences differ")
    ra, rh = apo["rows"][pair["apo"]["row"]], holo["rows"][pair["holo"]["row"]]
    ia = dict(zip(apo["construct_index"], range(len(apo["construct_index"]))))
    ih = dict(zip(holo["construct_index"], range(len(holo["construct_index"]))))
    rows = []
    for m in cmap:
        ci = m["construct_index"]
        if ci is None or ci not in ia or ci not in ih:
            continue
        base = apo["sequence"][ci - 1].upper()
        if not eligible(pair["chemistry"], base):
            continue
        ya, yh = ra["reactivity"][ia[ci]], rh["reactivity"][ih[ci]]
        if ya is None or yh is None:
            continue
        ea = ra.get("error", [None] * len(ra["reactivity"]))[ia[ci]]
        eh = rh.get("error", [None] * len(rh["reactivity"]))[ih[ci]]
        a = res[m["chain_index"]]
        rows.append({"construct_index": ci, "resi": a["resi"], "base": base, "class": a["class"],
                     "apo": ya, "holo": yh, "apo_err": ea, "holo_err": eh})
    rows.sort(key=lambda x: x["construct_index"])
    if synthetic_rng is not None:
        for x in rows:
            x["apo"], x["holo"] = float(synthetic_rng.lognormal(-1.5, 1.0)), float(synthetic_rng.lognormal(-1.5, 1.0))
    return rows


def analyse_pair(pair, rows, plan, seed):
    P = plan["statistics"]
    rng = np.random.default_rng(seed)
    lab = np.asarray([x["class"] for x in rows])
    apo = np.asarray([x["apo"] for x in rows], float)
    holo = np.asarray([x["holo"] for x in rows], float)
    has_err = all(x["apo_err"] is not None and x["holo_err"] is not None for x in rows)
    ea = np.asarray([x["apo_err"] if x["apo_err"] is not None else np.nan for x in rows], float)
    eh = np.asarray([x["holo_err"] if x["holo_err"] is not None else np.nan for x in rows], float)
    sa = norm_scale(apo, P["normalisation"]["drop_top_fraction"], P["normalisation"]["average_next_fraction"])
    sh = norm_scale(holo, P["normalisation"]["drop_top_fraction"], P["normalisation"]["average_next_fraction"])
    out = {"id": pair["id"], "system": pair["system"], "tier": pair["tier"], "chemistry": pair["chemistry"],
           "n_scored": len(rows), "class_counts": {c: int((lab == c).sum()) for c in "LTR"},
           "normalisation_scale_apo": sa, "normalisation_scale_holo": sh, "reported_errors": has_err}
    if sa <= 0 or sh <= 0 or len(rows) < P["min_scored"] or min((lab == "R").sum(), np.isin(lab, ["L", "T"]).sum()) < P["min_class"]:
        out["status"] = "stopped_insufficient_data_or_nonpositive_scale"
        return out, None
    d_norm = holo / sh - apo / sa
    variants = {"primary_2_8_normalised": np.abs(d_norm), "as_deposited": np.abs(holo - apo)}
    if has_err:
        sd = np.sqrt((ea / sa) ** 2 + (eh / sh) ** 2 + P["error_scaled_floor"] ** 2)
        variants["error_scaled"] = np.abs(d_norm) / sd
    res = {}
    for vname, r in variants.items():
        vres = {}
        for contrast in P["contrasts"]:
            fn = (lambda rr, ll, c=contrast: stat_for(rr, ll, c))
            null = circular_null(r, lab, fn)
            obs = float(null[0])
            entry = {"auc": obs, "p_circular_shift_one_sided": float(np.mean(null >= obs - 1e-12)),
                     "n_shifts": int(len(null)), "null_median": float(np.median(null)),
                     "null_95pct": float(np.quantile(null, 0.95))}
            if vname == "primary_2_8_normalised":
                bn = block_perm_null(r, lab, fn, P["block_length"], P["n_block_permutations"], rng)
                entry["p_block_permutation_one_sided"] = float((1 + np.sum(bn >= obs - 1e-12)) / (1 + len(bn)))
                ci, nok = block_bootstrap_ci(r, lab, fn, P["block_length"], P["n_bootstrap"], rng)
                entry["auc_moving_block_bootstrap_95ci"] = ci
                entry["bootstrap_valid_resamples"] = nok
            if contrast == "LT_vs_R" and vname == "primary_2_8_normalised":
                entry["circular_null_all_shifts"] = null.tolist()
            vres[contrast] = entry
        if vname == "primary_2_8_normalised":
            fn = top_decile_enrichment
            null = circular_null(r, lab, fn)
            vres["top_decile_enrichment_LT"] = {"ratio": float(null[0]), "p_circular_shift_one_sided": float(np.mean(null >= null[0] - 1e-12)),
                                                "k_top": int(math.ceil(0.1 * len(r))), "fraction_LT_overall": float(np.isin(lab, ["L", "T"]).mean())}
        res[vname] = vres
    out["results"] = res
    out["descriptive"] = {c: {"n": int((lab == c).sum()),
                              "median_abs_delta_norm": float(np.median(np.abs(d_norm[lab == c]))) if (lab == c).any() else None,
                              "mean_delta_norm": float(np.mean(d_norm[lab == c])) if (lab == c).any() else None,
                              "fraction_protected": float(np.mean(d_norm[lab == c] < 0)) if (lab == c).any() else None}
                          for c in "LTR"}
    out["status"] = "executed"
    table = [{"construct_index": x["construct_index"], "resi": x["resi"], "base": x["base"], "class": x["class"],
              "apo_norm": float(x["apo"] / sa), "holo_norm": float(x["holo"] / sh), "delta_norm": float(dn),
              "delta_norm_sd": float(np.sqrt((x["apo_err"] / sa) ** 2 + (x["holo_err"] / sh) ** 2)) if has_err else None}
             for x, dn in zip(rows, d_norm)]
    return out, table


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
        if rel == "scripts/analysis_a.py" and args.synthetic:
            continue
        if sha(ROOT / rel) != h:
            raise SystemExit(f"Frozen file changed after the plan: {rel}")
    classes = json.loads((ROOT / plan["classes_file"]).read_text())
    srng = np.random.default_rng(12345) if args.synthetic else None
    results, tables = [], {}
    for k, pair in enumerate(plan["profile_pairs"]):
        rows = load_pair(pair, classes, plan, srng)
        out, table = analyse_pair(pair, rows, plan, plan["statistics"]["seed"] + k)
        results.append(out)
        if table:
            tables[pair["id"]] = table
        r = out.get("results", {}).get("primary_2_8_normalised", {}).get("LT_vs_R", {})
        print(f"{pair['id']:32s} n={out['n_scored']:4d} {out['class_counts']} AUC_LT={r.get('auc', float('nan')):.3f} p_circ={r.get('p_circular_shift_one_sided', float('nan')):.4f} {out['status']}")
    by_id = {r["id"]: r for r in results}
    # primary family (Holm)
    prim = [p["id"] for p in plan["profile_pairs"] if p["tier"] == "primary"]
    prim_p = [by_id[i]["results"]["primary_2_8_normalised"]["LT_vs_R"]["p_circular_shift_one_sided"] for i in prim]
    family = {"primary_ids": prim, "raw_p": prim_p, "holm_adjusted_p": holm(prim_p)}
    # secondary family (BH) over executed non-primary pairs, LT_vs_R primary statistic
    sec = [r["id"] for r in results if r["tier"] in ("replication", "extension") and r["status"] == "executed"]
    sec_p = [by_id[i]["results"]["primary_2_8_normalised"]["LT_vs_R"]["p_circular_shift_one_sided"] for i in sec]
    family["secondary_ids"], family["secondary_raw_p"], family["secondary_bh_adjusted_p"] = sec, sec_p, bh(sec_p) if sec_p else []
    # pooled endpoint
    pooled_ids = plan["pooled_endpoint"]["pair_ids"]
    nulls = [np.asarray(by_id[i]["results"]["primary_2_8_normalised"]["LT_vs_R"]["circular_null_all_shifts"]) for i in pooled_ids]
    obs = float(np.mean([n[0] - 0.5 for n in nulls]))
    rng = np.random.default_rng(plan["statistics"]["seed"])
    B = plan["pooled_endpoint"]["n_draws"]
    draws = np.mean([n[rng.integers(0, len(n), size=B)] - 0.5 for n in nulls], axis=0)
    p_pool = float((1 + np.sum(draws >= obs - 1e-12)) / (1 + B))
    fisher_x = float(-2 * np.sum([np.log(by_id[i]["results"]["primary_2_8_normalised"]["LT_vs_R"]["p_circular_shift_one_sided"]) for i in pooled_ids]))
    pooled = {"pair_ids": pooled_ids, "mean_auc_minus_half": obs, "mean_auc": obs + 0.5,
              "p_independent_circular_shifts": p_pool, "n_draws": B,
              "fisher_chi2": fisher_x, "fisher_df": 2 * len(pooled_ids), "fisher_p_approx": float(chi2.sf(fisher_x, 2 * len(pooled_ids)))}
    report = {"schema": "feasibility-analysisA-v1", "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "synthetic_plumbing_test": bool(args.synthetic), "plan_sha256": args.expected_sha,
              "code_sha256": sha(__file__), "classes_sha256": sha(ROOT / plan["classes_file"]),
              "numpy": np.__version__, "pairs": results, "multiplicity": family, "pooled_endpoint": pooled,
              "not_executed": plan["not_executed"]}
    name = "synthetic_test_analysisA.json" if args.synthetic else "analysisA_results.json"
    (ROOT / "results" / name).write_text(json.dumps(report, indent=1) + "\n")
    if not args.synthetic:
        (ROOT / "results" / "analysisA_per_position.json").write_text(json.dumps(tables, indent=1) + "\n")
    print("pooled", pooled)
    print("primary family", family["primary_ids"], family["raw_p"], family["holm_adjusted_p"])


if __name__ == "__main__":
    main()
