#!/usr/bin/env python3
"""CPU corruption and leakage checks; generated toy values are code tests only.

v2 adds: masked-subset (DMS A/C) shuffle marginals, ineligible-observation
rejection, near-identity lineage guard, unweighted-refit precision, extraction-
audit binding at freeze, and an end-to-end toy freeze/run of both weighting schemes.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile

import numpy as np

import forward_head_pilot as m


def example(k):
    n = 14
    rng = np.random.default_rng(100 + k)
    coords = np.cumsum(rng.normal(size=(n, 3)), axis=0) * 4.0
    return {
        "example_id": f"TOY_CODE_TEST_{k}", "lineage_id": f"toy_lineage_{k}",
        "study_id": f"toy_study_{k % 2}", "chemistry": "DMS" if k % 2 else "SHAPE",
        "sequence": "ACGUACGUACGUAC", "reactivity": list(rng.normal(size=n)),
        "stderr": [None if i % 3 else 0.2 for i in range(n)], "observed_mask": [True] * n,
        "coordinates": coords.tolist(), "secondary_dot_bracket": "(((........)))",
        "admission": {"eligible": True, "geometry_compatibility": "TOY_NOT_BIOLOGICAL"},
        "provenance": {"scope": "Generated unit-test fixture, not evidence"},
        "context": {"ph": 7.5, "magnesium_mm": 10.0},
    }


def dms_mask(e):
    e = copy.deepcopy(e)
    if m.chemistry_group(e["chemistry"]) == "DMS":
        e["observed_mask"] = [nt in "AC" for nt in e["sequence"]]
    return e


def toy_sequence(k, n=30):
    rng = np.random.default_rng(500 + k)
    return "".join(rng.choice(list("ACGU"), size=n))


def main():
    checks = []
    e = example(0)
    original, valid = m.feature_matrices(e)
    corrupt = copy.deepcopy(e)
    corrupt["reactivity"] = [9999.0] * len(e["sequence"])
    corrupt["stderr"] = [0.00001] * len(e["sequence"])
    corrupt["observed_mask"] = [False] * len(e["sequence"])
    changed, _ = m.feature_matrices(corrupt)
    assert all(np.array_equal(original[a], changed[a]) for a in m.ARMS)
    d = example(1)  # DMS: the A/C-restricted shuffle must not read observations either
    corrupt_d = copy.deepcopy(d)
    corrupt_d["reactivity"] = [-5.0] * len(d["sequence"])
    corrupt_d["observed_mask"] = [False] * len(d["sequence"])
    assert all(np.array_equal(m.feature_matrices(d)[0][a], m.feature_matrices(corrupt_d)[0][a]) for a in m.ARMS)
    checks.append("Observed values/errors/masks never enter any feature matrix (SHAPE and DMS)")

    # Translations and rotations must leave all physical features unchanged.
    q, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(3, 3)))
    transformed = copy.deepcopy(e)
    transformed["coordinates"] = (np.asarray(e["coordinates"]) @ q + [90.0, -30.0, 17.0]).tolist()
    moved, _ = m.feature_matrices(transformed)
    assert all(np.allclose(original[a], moved[a], atol=1e-12) for a in m.ARMS)
    checks.append("Rigid translation/rotation leaves local geometric invariants and predictions unchanged")

    assert original["geometry_context"].shape == original["shuffled_geometry_context"].shape
    start = original["secondary_context"].shape[1]
    assert np.allclose(np.sort(original["geometry_context"][:, start:], axis=0),
                       np.sort(original["shuffled_geometry_context"][:, start:], axis=0))
    checks.append("Pairing negative control preserves all-row geometry marginals and exact coefficient count")

    # v2: marginals over the scored DMS A/C subset are preserved exactly.
    dm = dms_mask(example(1))
    feats, structural = m.feature_matrices(dm)
    sel = np.asarray(dm["observed_mask"]) & structural
    s0, s1 = feats["secondary_context"].shape[1] // 3, feats["geometry_context"].shape[1] // 3
    g, s = feats["geometry_context"][sel][:, s0:s1], feats["shuffled_geometry_context"][sel][:, s0:s1]
    assert np.array_equal(np.sort(g, axis=0), np.sort(s, axis=0))
    assert np.any(g != s)  # the pairing is actually broken
    # Sensitivity: the v1 scheme (permute every structurally valid row) fails this check.
    geo, _ = m.geometry_features(dm)
    v1_idx = np.flatnonzero(structural)
    v1 = geo.copy()
    v1[v1_idx] = geo[np.random.default_rng(m.SEED + 1).permutation(v1_idx)]
    assert not np.array_equal(np.sort(geo[sel], axis=0), np.sort(v1[sel], axis=0))
    rows_dm = m.prepared_rows([dm, dms_mask(example(0))])
    assert all(x["scored_row_geometry_marginal_preserved"] for x in m.shuffle_marginal_audit(rows_dm))
    checks.append("Shuffle preserves geometry marginals on the scored DMS A/C subset (v1 all-row scheme detectably does not)")

    bad_dms = copy.deepcopy(example(1))  # observed G/U under DMS is outside the frozen scoring rule
    try:
        m.prepared_rows([bad_dms])
    except ValueError:
        pass
    else:
        raise AssertionError("DMS G/U observation accepted")
    checks.append("DMS observations at chemically ineligible G/U positions are rejected")

    examples = [dms_mask(example(k)) for k in range(4)]  # DMS toys scored on A/C only (v2 rule)
    rows = m.prepared_rows(examples)
    outer_train = rows["lineage"] != "toy_lineage_0"
    alpha, _ = m.inner_alpha(rows, "geometry_context", outer_train)
    weights = m.balanced_weights(rows, outer_train)
    fit = m.ridge_fit(rows["X"]["geometry_context"][outer_train], rows["y"][outer_train], weights[outer_train], alpha)
    changed_rows = copy.deepcopy(rows)
    changed_rows["y"][~outer_train] = 1e8
    changed_rows["precision"][~outer_train] = 1e-10
    alpha2, _ = m.inner_alpha(changed_rows, "geometry_context", outer_train)
    weights2 = m.balanced_weights(changed_rows, outer_train)
    fit2 = m.ridge_fit(changed_rows["X"]["geometry_context"][outer_train], changed_rows["y"][outer_train], weights2[outer_train], alpha2)
    assert alpha == alpha2
    assert all(np.array_equal(fit[k], fit2[k]) for k in ("mean", "scale", "beta"))
    assert fit["target_mean"] == fit2["target_mean"]
    checks.append("Changing held-out targets and errors cannot change that fold's normalization, alpha or coefficients")

    # Held-out features can be arbitrarily different without contaminating fit scales.
    moved_x = rows["X"]["geometry_context"].copy()
    moved_x[~outer_train] = 1e12
    fit3 = m.ridge_fit(moved_x[outer_train], rows["y"][outer_train], weights[outer_train], alpha)
    assert all(np.array_equal(fit[k], fit3[k]) for k in ("mean", "scale", "beta"))
    checks.append("Held-out feature distribution cannot contaminate training standardization")

    # Precision changes relative nucleotide contribution, never profile/lineage total.
    duplicate = copy.deepcopy(examples[0]); duplicate["example_id"] += "_replicate"
    twice = m.prepared_rows(examples + [duplicate])
    w = m.balanced_weights(twice, np.ones(len(twice["y"]), dtype=bool))
    for lineage in sorted(set(twice["lineage"])):
        assert np.isclose(w[twice["lineage"] == lineage].sum(), 0.25)
    assert np.isclose(w[twice["profile"] == duplicate["example_id"]].sum(), 0.125)
    checks.append("Replicate profile cannot increase its lineage's total training weight")

    unweighted = m.prepared_rows(examples, "unweighted")
    assert np.all(unweighted["precision"] == 1.0) and np.any(rows["precision"] != 1.0)
    wu = m.balanced_weights(unweighted, np.ones(len(unweighted["y"]), dtype=bool))
    for p in sorted(set(unweighted["profile"])):
        assert np.allclose(wu[unweighted["profile"] == p], wu[unweighted["profile"] == p][0])
    assert all(np.array_equal(rows["X"][a], unweighted["X"][a]) for a in m.ARMS)
    checks.append("Unweighted refit uses unit precision (uniform within profile) with identical features")

    missing = copy.deepcopy(e)
    missing["reactivity"][2] = None; missing["observed_mask"][2] = False
    missing["coordinates"][4] = None
    r = m.prepared_rows([missing])
    assert 2 not in r["position"] and 4 not in r["position"]
    assert len({len(x) for x in r["X"].values()}) == 1
    checks.append("Missing observations and unmapped coordinates use a common mask in all arms")

    bad = copy.deepcopy(e); bad["reactivity"][2] = None
    try:
        m.normalize_cohort({"examples": [bad]})
    except ValueError:
        pass
    else:
        raise AssertionError("Observed-null corruption passed")
    try:
        m.normalize_cohort({"synthetic": True, "examples": examples})
    except ValueError:
        pass
    else:
        raise AssertionError("Synthetic source admitted as biological cohort")
    checks.append("Observed-null corruption and explicitly synthetic biological inputs are rejected")

    paired, secondary_known = m.dot_pairs("(.xX-?.)")
    assert paired[0] == 7 and paired[7] == 0
    assert secondary_known.tolist() == [True, True, False, False, False, False, True, True]
    checks.append("Unknown secondary annotations x/X/?/- stay masked rather than becoming pair labels")

    # v2: near-identity guard (exact, single substitution, sub-construct containment).
    base = toy_sequence(0, 60)
    variant = base[:20] + ("A" if base[20] != "A" else "C") + base[21:]
    def pair(seq_b):
        a = {"example_id": "a", "lineage_id": "L_a", "sequence": base}
        b = {"example_id": "b", "lineage_id": "L_b", "sequence": seq_b}
        return m.cross_lineage_similarity([a, b])[0]
    assert pair(base)["near_identical"] and pair(variant)["near_identical"]
    assert pair(base[10:45])["near_identical"]  # contained sub-construct
    assert not pair(toy_sequence(1, 60))["near_identical"]
    try:
        m.check_lineage_independence([{"example_id": "a", "lineage_id": "L_a", "sequence": base},
                                      {"example_id": "b", "lineage_id": "L_b", "sequence": variant}])
    except ValueError:
        pass
    else:
        raise AssertionError("Near-identical cross-lineage sequence accepted")
    checks.append("Exact, single-substitution and contained near-identical sequences cannot cross lineages; unrelated ones pass")

    with tempfile.TemporaryDirectory() as temporary:
        p = Path(temporary)
        source_example = copy.deepcopy(e)
        source_example["provenance"] = {}
        for name in ("rdat", "metadata", "pdb"):
            source = p / f"{name}.txt"
            source.write_text(f"Generated toy source {name}; not biological evidence")
            source_example["provenance"][f"{name}_path"] = source.name
            source_example["provenance"][f"{name}_sha256"] = m.sha(source)
        assert len(m.verify_sources([source_example], p / "cohort.json")) == 3
        (p / "pdb.txt").write_text("Intentionally corrupted source bytes")
        try:
            m.verify_sources([source_example], p / "cohort.json")
        except ValueError:
            pass
        else:
            raise AssertionError("Changed coordinate source accepted")
    checks.append("Raw coordinate-source byte corruption is rejected against the provenance pin")

    # A source or code mutation after freezing must be rejected before fitting.
    with tempfile.TemporaryDirectory() as temporary:
        p = Path(temporary)
        cohort = p / "cohort.json"
        cohort.write_text(json.dumps({"examples": examples}))
        plan = p / "plan.json"
        m.dump(plan, {"schema_version": m.VERSION, "cohort_sha256": m.sha(cohort), "code_sha256": "intentionally_corrupt"})
        try:
            m.run_plan(cohort, plan, p / "out")
        except ValueError:
            pass
        else:
            raise AssertionError("Changed code accepted against frozen plan")
        assert not (p / "out").exists()
    checks.append("Changed-code/frozen-plan mismatch aborts before any model output")

    # v2: freeze is bound to a passed raw-extraction audit of the exact cohort; end-to-end toy run.
    with tempfile.TemporaryDirectory() as temporary:
        p = Path(temporary)
        (p / "sources").mkdir()
        toys = []
        for k in range(4):
            t = dms_mask(example(k))
            t["sequence"] = toy_sequence(10 + k, 14)
            t = dms_mask(t)
            t["reactivity"] = [float(x) for x in np.random.default_rng(900 + k).normal(size=14)]
            t["provenance"] = {}
            for name in ("rdat", "metadata", "pdb"):
                src = p / "sources" / f"toy{k}_{name}.txt"
                src.write_text(f"Generated toy source {k} {name}; not biological evidence")
                t["provenance"][f"{name}_path"] = f"sources/{src.name}"
                t["provenance"][f"{name}_sha256"] = m.sha(src)
            toys.append(t)
        cohort = p / "cohort.json"
        cohort.write_text(json.dumps({"synthetic": False, "examples": toys}))
        audit = p / "audit.json"
        for bad_audit in ({"status": "passed", "n_failed": 0, "cohort_sha256": "0" * 64},
                          {"status": "FAILED", "n_failed": 1, "cohort_sha256": m.sha(cohort)}):
            audit.write_text(json.dumps(bad_audit))
            try:
                m.make_plan(cohort, p / "plan.json", audit)
            except ValueError:
                pass
            else:
                raise AssertionError("Freeze accepted a mismatched or failed extraction audit")
            assert not (p / "plan.json").exists()
        audit.write_text(json.dumps({"status": "passed", "n_failed": 0, "n_checks": 1, "cohort_sha256": m.sha(cohort)}))
        plan = m.make_plan(cohort, p / "plan.json", audit)
        assert all(x["scored_row_geometry_marginal_preserved"] for x in plan["shuffled_geometry_control"]["scored_row_marginal_verification"])
        report = m.run_plan(cohort, p / "plan.json", p / "out")
        assert set(report["schemes"]) == set(m.WEIGHTING_SCHEMES)
        assert {(c["scheme"], c["reference"], c["metric"]) for c in report["comparisons"]} >= {
            ("error_weighted", "sequence_context", "error_weighted_mse"), ("unweighted", "shuffled_geometry_context", "unweighted_mse")}
        audit.write_text(json.dumps({"status": "passed", "n_failed": 0, "n_checks": 2, "cohort_sha256": m.sha(cohort)}))
        try:
            m.run_plan(cohort, p / "plan.json", p / "out2")
        except ValueError:
            pass
        else:
            raise AssertionError("Run accepted an extraction audit changed after freezing")
        assert not (p / "out2").exists()
    checks.append("Freeze requires a passed extraction audit of the exact cohort; audit change after freeze aborts the run")
    checks.append("End-to-end toy freeze/run executes both weighting schemes and all frozen comparisons")

    report = {"scope": "Generated toy inputs exercise code contracts only; no biological efficacy evidence",
              "status": "passed", "checks": checks, "n_checks": len(checks),
              "code_sha256": m.sha(m.__file__), "test_sha256": m.sha(__file__), "numpy_version": np.__version__}
    m.dump(Path(__file__).with_name("code_contract_checks.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
