#!/usr/bin/env python3
"""Leakage-resistant, bounded structure-to-probe forward-model pilot (Wave F v2).

This is a learned measurement operator, not an inverse structure predictor. It
does not use probe observations, errors, profile IDs or study IDs as features.
No synthetic data are admitted to a biological run. See --help for freeze/run.

v2 changes versus v1 (see ../CHANGES.md for the audit-point mapping):
  * shuffled-geometry control permutes within the chemically eligible, structurally
    annotated rows (DMS: A/C; SHAPE: all), so the evaluated-row geometry marginal is
    preserved; eligibility uses sequence+chemistry only, never observations;
  * freeze verifies and records that marginal preservation per profile;
  * a full unweighted refit (unit precision) is run as a pre-specified sensitivity
    alongside the error-weighted primary; REACTIVITY_ERROR semantics are recorded
    as unknown and the 0.1 floor as an arbitrary constant;
  * freeze requires a passed raw-extraction audit bound to the exact cohort hash;
  * the cross-lineage guard now implements a near-identity check (global and
    local alignment) in addition to exact identity;
  * a sequence-versus-geometry comparison is added to the frozen comparison set.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

VERSION = "forward-head-lineage-ridge-v2"
ARMS = ("sequence_context", "secondary_context", "geometry_context", "shuffled_geometry_context")
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
SEED = 20261014
ERROR_FLOOR = 0.1
BOOTSTRAPS = 10000
CHEMISTRIES = ["DMS", "SHAPE", "CMCT", "OTHER"]
NUMERIC_CONTEXT = ["temperature_c", "ph", "magnesium_mm", "monovalent_mm", "ligand_um"]
WEIGHTING_SCHEMES = ("error_weighted", "unweighted")
# Near-identity guard (pre-specified; scores match +1, mismatch -1, linear gap -2).
NEAR_IDENTITY_GLOBAL = 0.80          # global (Needleman-Wunsch) identity over alignment columns
NEAR_IDENTITY_LOCAL = 0.90           # local (Smith-Waterman) identity over aligned columns ...
NEAR_IDENTITY_LOCAL_COVERAGE = 0.50  # ... spanning at least this fraction of the shorter sequence
COMPARISONS = (("sequence_context", "geometry_context"), ("secondary_context", "geometry_context"),
               ("shuffled_geometry_context", "geometry_context"), ("sequence_context", "secondary_context"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def chemistry_group(text):
    text = str(text).upper()
    if "DMS" in text:
        return "DMS"
    if any(k in text for k in ("SHAPE", "1M7", "NMIA", "2A3", "NAI")):
        return "SHAPE"
    return "CMCT" if "CMCT" in text else "OTHER"


def chemical_eligibility(e):
    """Positions the chemistry can score, from sequence and chemistry only (never y).

    DMS primary scoring is restricted to A/C (N1-A/N3-C methylation); other
    chemistries score every nucleotide. Observation masks must be a subset."""
    if chemistry_group(e["chemistry"]) == "DMS":
        return np.asarray([nt in "AC" for nt in e["sequence"]], dtype=bool)
    return np.ones(len(e["sequence"]), dtype=bool)


def dot_pairs(dot):
    """Support standard pseudoknot brackets; unknown entries remain unknown."""
    # Reserve x/X for explicitly unknown secondary annotation, as below.
    opened = "([{<ABCDEFGHIJKLMNOPQRSTUVWYZ"
    closed = ")]}>abcdefghijklmnopqrstuvwyz"
    stacks = {c: [] for c in opened}
    pairs = np.full(len(dot), -1, dtype=int)
    valid = np.ones(len(dot), dtype=bool)
    for i, c in enumerate(dot):
        if c in opened:
            stacks[c].append(i)
        elif c in closed:
            opener = opened[closed.index(c)]
            if not stacks[opener]:
                raise ValueError("Unbalanced secondary structure")
            j = stacks[opener].pop()
            pairs[i], pairs[j] = j, i
        elif c in ("?", "-", "x", "X"):
            valid[i] = False
        elif c != ".":
            raise ValueError(f"Unsupported dot-bracket token {c!r}")
    if any(stacks.values()):
        raise ValueError("Unbalanced secondary structure")
    return pairs, valid


def normalize_cohort(raw):
    """Validate explicit vectors. Never derive targets or secondary pairs from y."""
    if raw.get("synthetic", False):
        raise ValueError("Synthetic cohorts cannot be used for a biological run")
    examples = raw.get("examples", [])
    if not examples:
        raise ValueError("Cohort must supply a nonempty examples array")
    ids = set()
    out = []
    for e in examples:
        x = dict(e)
        for field in ("example_id", "lineage_id", "study_id", "chemistry", "sequence"):
            if not isinstance(x.get(field), str) or not x[field]:
                raise ValueError(f"Missing nonempty {field}")
        if x["example_id"] in ids:
            raise ValueError("Duplicate example_id")
        ids.add(x["example_id"])
        x["sequence"] = x["sequence"].upper().replace("T", "U")
        n = len(x["sequence"])
        for field in ("reactivity", "stderr", "observed_mask", "coordinates"):
            if len(x.get(field, [])) != n:
                raise ValueError(f"{x['example_id']}: {field} length differs from sequence")
        if not isinstance(x.get("secondary_dot_bracket"), str) or len(x["secondary_dot_bracket"]) != n:
            raise ValueError("An independently sourced/mapped secondary structure is required")
        if not isinstance(x.get("admission"), dict):
            raise ValueError("Explicit scientific admission metadata required")
        if x["admission"].get("eligible") is not True:
            continue
        if not x.get("provenance"):
            raise ValueError("Source/mapping provenance is required")
        for v in x["coordinates"]:
            if v is not None and not (len(v) == 3 and all(finite_number(c) for c in v)):
                raise ValueError("Coordinates must be finite 3-vectors or null")
        for obs, y, error in zip(x["observed_mask"], x["reactivity"], x["stderr"]):
            if not isinstance(obs, bool):
                raise ValueError("Observation masks must be booleans")
            if obs and not finite_number(y):
                raise ValueError("An observed reactivity must be finite; missing is null, not zero")
            if error is not None and (not finite_number(error) or error < 0):
                raise ValueError("Errors must be nonnegative finite values or null")
        dot_pairs(x["secondary_dot_bracket"])
        out.append(x)
    if not out:
        raise ValueError("No scientifically admitted examples")
    return out


def verify_sources(examples, cohort_path):
    """Recheck pinned raw observations, metadata and coordinates before any fit."""
    base = Path(cohort_path).resolve().parent
    checked = {}
    for e in examples:
        provenance = e["provenance"]
        pairs = [(provenance.get(f"{key}_path"), provenance.get(f"{key}_sha256"))
                 for key in ("rdat", "metadata", "pdb")]
        secondary_id = provenance.get("secondary_annotation_rmdb_source")
        if secondary_id:
            pairs.append((f"sources/{secondary_id}.rdat", provenance.get("secondary_annotation_rdat_sha256")))
        for name, expected in pairs:
            if not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"{e['example_id']}: source path/SHA-256 is incomplete")
            path = (base / name).resolve()
            if not path.is_relative_to(base):
                raise ValueError("Source path must remain within the packaged cohort")
            observed = sha(path)
            if observed != expected:
                raise ValueError(f"Pinned source changed: {name}")
            if name in checked and checked[name] != observed:
                raise ValueError(f"Inconsistent source pin: {name}")
            checked[name] = observed
    return checked


def _align(a, b, local):
    """Deterministic NW (global) / SW (local) alignment; returns identity statistics."""
    match, mismatch, gap = 1, -1, -2
    n, m = len(a), len(b)
    H = np.zeros((n + 1, m + 1))
    T = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 stop, 1 diagonal, 2 up, 3 left
    if not local:
        H[1:, 0] = gap * np.arange(1, n + 1)
        H[0, 1:] = gap * np.arange(1, m + 1)
        T[1:, 0], T[0, 1:] = 2, 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (H[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch),
                       H[i - 1, j] + gap, H[i, j - 1] + gap)
            k = int(np.argmax(options))  # ties prefer diagonal, then up
            best = options[k]
            if local and best <= 0:
                H[i, j], T[i, j] = 0.0, 0
            else:
                H[i, j], T[i, j] = best, k + 1
    if local:
        i, j = np.unravel_index(int(np.argmax(H)), H.shape)
    else:
        i, j = n, m
    end_i, end_j = int(i), int(j)
    matches = columns = 0
    while (i > 0 or j > 0) and T[i, j] != 0:
        step = T[i, j]
        columns += 1
        if step == 1:
            matches += int(a[i - 1] == b[j - 1])
            i, j = i - 1, j - 1
        elif step == 2:
            i -= 1
        else:
            j -= 1
    span_a, span_b = end_i - int(i), end_j - int(j)
    return {"matches": matches, "columns": columns, "identity": matches / columns if columns else 0.0,
            "span_a": span_a, "span_b": span_b}


def cross_lineage_similarity(examples):
    """Pairwise sequence similarity for every pair of profiles in different lineages."""
    rows = []
    seen = {}
    for e in examples:
        seen.setdefault((e["lineage_id"], e["sequence"]), e["example_id"])
    items = sorted(seen.items())
    for x in range(len(items)):
        for y in range(x + 1, len(items)):
            (la, sa), ea = items[x]
            (lb, sb), eb = items[y]
            if la == lb:
                continue
            g = _align(sa, sb, local=False)
            s = _align(sa, sb, local=True)
            shorter = min(len(sa), len(sb))
            coverage = (s["span_a"] if len(sa) <= len(sb) else s["span_b"]) / shorter
            near = (sa == sb or g["identity"] >= NEAR_IDENTITY_GLOBAL
                    or (s["identity"] >= NEAR_IDENTITY_LOCAL and coverage >= NEAR_IDENTITY_LOCAL_COVERAGE))
            rows.append({"lineage_a": la, "example_a": ea, "lineage_b": lb, "example_b": eb,
                         "length_a": len(sa), "length_b": len(sb), "exact_identical": sa == sb,
                         "global_identity": round(g["identity"], 4),
                         "local_identity": round(s["identity"], 4),
                         "local_coverage_of_shorter": round(coverage, 4), "near_identical": bool(near)})
    return rows


def check_lineage_independence(examples):
    rows = cross_lineage_similarity(examples)
    flagged = [r for r in rows if r["near_identical"]]
    if flagged:
        raise ValueError(f"Exact or near-identical sequences assigned to different lineages: {flagged}")
    return rows


def sequence_features(e):
    n = len(e["sequence"])
    rows = []
    for i in range(n):
        r = []
        # A five-position sequence window; boundary/unknown is its own channel.
        for offset in (-2, -1, 0, 1, 2):
            j = i + offset
            nt = e["sequence"][j] if 0 <= j < n else "N"
            r += [float(nt == base) for base in "ACGU"] + [float(nt not in "ACGU")]
        chem = chemistry_group(e["chemistry"])
        r += [float(chem == c) for c in CHEMISTRIES]
        context = dict(e.get("context") or {})
        # Accept only explicit numeric source-protocol values. Do not infer
        # unreported pH/salt/ligand values from names or target profiles.
        protocol = e.get("protocol") or {}
        for target, source in (("temperature_c", "temperature_C"), ("magnesium_mm", "MgCl2_mM")):
            if target not in context and finite_number(protocol.get(source)):
                context[target] = protocol[source]
        scales = [100.0, 14.0, 100.0, 1000.0, 1000.0]
        for key, scale in zip(NUMERIC_CONTEXT, scales):
            v = context.get(key)
            r += [float(v) / scale if finite_number(v) else 0.0, float(finite_number(v))]
        r += [float(i) / max(n - 1, 1), float(n) / 1000.0]
        rows.append(r)
    return np.asarray(rows, dtype=np.float64)


def secondary_features(e):
    n = len(e["sequence"])
    pairs, valid = dot_pairs(e["secondary_dot_bracket"])
    paired = pairs >= 0
    rows = []
    for i, j in enumerate(pairs):
        nt = e["sequence"][j] if j >= 0 else "N"
        r = [float(paired[i]), abs(j - i) / n if j >= 0 else 0.0]
        r += [float(j >= 0 and nt == base) for base in "ACGU"]
        for offset in (-2, -1, 1, 2):
            k = i + offset
            r += [float(0 <= k < n and paired[k])]
        rows.append(r)
    return np.asarray(rows), valid


def geometry_features(e):
    """C4' invariants in fixed physical units; no observation-dependent features."""
    n = len(e["sequence"])
    valid = np.asarray([v is not None for v in e["coordinates"]])
    coords = np.asarray([v if v is not None else [0.0] * 3 for v in e["coordinates"]], dtype=float)
    distance = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
    index = np.arange(n)
    rows = []
    for i in range(n):
        nonlocal_mask = (np.abs(index - i) > 3) & valid
        ds = np.sort(distance[i, nonlocal_mask])
        r = []
        for offset in (-2, -1, 1, 2):
            j = i + offset
            present = 0 <= j < n and valid[i] and valid[j]
            r += [distance[i, j] / 20.0 if present else 0.0, float(present)]
        # Bounded contacts represent local packing, without coordinates/IDs as features.
        r += [float(np.sum(ds < radius)) / 20.0 for radius in (8.0, 12.0, 16.0)]
        r += [float(ds[0]) / 30.0 if len(ds) else 0.0,
              float(np.mean(ds[:3])) / 30.0 if len(ds) else 0.0,
              float(bool(len(ds)))]
        r += [float(np.exp(-((ds / radius) ** 2)).sum()) / 20.0 for radius in (8.0, 16.0)]
        angle_present = i > 0 and i + 1 < n and valid[i - 1:i + 2].all()
        cosine = 0.0
        if angle_present:
            a, b = coords[i - 1] - coords[i], coords[i + 1] - coords[i]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            cosine = float(np.dot(a, b) / denominator) if denominator > 0 else 0.0
        r += [cosine, float(angle_present)]
        rows.append(r)
    return np.asarray(rows, dtype=float), valid


def feature_matrices(e):
    seq = sequence_features(e)
    sec, secondary_valid = secondary_features(e)
    geo, geometry_valid = geometry_features(e)
    structural_valid = geometry_valid & secondary_valid
    # v2 negative control: a deterministic within-profile permutation of geometry
    # rows restricted to the chemically eligible, structurally annotated positions
    # (DMS: A/C; SHAPE: all). Every scored row is in this set, so the geometry
    # marginal over the scored rows is preserved whenever every eligible row is
    # observed (verified and recorded at freeze). Eligibility uses sequence and
    # chemistry only; the fixed seed never depends on observations. This breaks the
    # residue-to-geometry pairing; it is not a biological randomisation test.
    stable = int(hashlib.sha256(e["example_id"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(SEED + stable)
    permuted = geo.copy()
    indices = np.flatnonzero(structural_valid & chemical_eligibility(e))
    permuted[indices] = geo[rng.permutation(indices)]
    raw = {
        "sequence_context": seq,
        "secondary_context": np.concatenate([seq, sec], axis=1),
        "geometry_context": np.concatenate([seq, sec, geo], axis=1),
        "shuffled_geometry_context": np.concatenate([seq, sec, permuted], axis=1),
    }
    # Chemistry-specific slopes are necessary: DMS and SHAPE measure different
    # chemistry. A shared block plus fixed DMS/SHAPE interaction blocks permits
    # different residue/structure effects, without an accession-specific model.
    chemistry = chemistry_group(e["chemistry"])
    matrices = {arm: np.concatenate([x, x * float(chemistry == "DMS"), x * float(chemistry == "SHAPE")], axis=1)
                for arm, x in raw.items()}
    return matrices, structural_valid


def precision_of(error, weighting):
    if weighting == "unweighted":
        return 1.0
    if weighting != "error_weighted":
        raise ValueError(f"Unknown weighting scheme {weighting}")
    # No source error means unit precision, explicitly reported in the plan.
    return 1.0 / max(float(error), ERROR_FLOOR) ** 2 if error is not None else 1.0


def prepared_rows(examples, weighting="error_weighted"):
    matrices = {a: [] for a in ARMS}
    y, precision, profile, lineage, study, chemistry, position = [], [], [], [], [], [], []
    counts = []
    for e in examples:
        features, structural_valid = feature_matrices(e)
        eligible = chemical_eligibility(e)
        observed = np.asarray(e["observed_mask"], dtype=bool)
        if np.any(observed & ~eligible):
            raise ValueError(f"{e['example_id']}: observation mask includes chemically ineligible positions")
        selected = observed & structural_valid
        if selected.sum() < 3:
            raise ValueError(f"{e['example_id']}: fewer than 3 common measured/mapped nucleotides")
        for a in ARMS:
            matrices[a].append(features[a][selected])
        indices = np.flatnonzero(selected)
        permutation_set = structural_valid & eligible
        for i in indices:
            error = e["stderr"][i]
            y.append(float(e["reactivity"][i]))
            precision.append(precision_of(error, weighting))
            profile.append(e["example_id"])
            lineage.append(e["lineage_id"])
            study.append(e["study_id"])
            chemistry.append(chemistry_group(e["chemistry"]))
            position.append(int(i))
        reported = [e["stderr"][i] for i in indices if e["stderr"][i] is not None]
        counts.append({"example_id": e["example_id"], "lineage_id": e["lineage_id"], "study_id": e["study_id"],
                       "chemistry": chemistry_group(e["chemistry"]), "n_common": int(selected.sum()),
                       "n_error_missing": sum(e["stderr"][i] is None for i in indices),
                       "n_error_zero": sum(x == 0 for x in reported),
                       "n_error_at_or_below_floor": sum(x <= ERROR_FLOOR for x in reported),
                       "n_shuffle_permutation_rows": int(permutation_set.sum()),
                       "n_shuffle_eligible_rows_unobserved": int((permutation_set & ~observed).sum()),
                       "geometry_compatibility": e["admission"].get("geometry_compatibility", "unspecified")})
    return {"X": {a: np.concatenate(v) for a, v in matrices.items()}, "y": np.asarray(y),
            "precision": np.asarray(precision), "profile": np.asarray(profile), "lineage": np.asarray(lineage),
            "study": np.asarray(study), "chemistry": np.asarray(chemistry),
            "position": np.asarray(position), "counts": counts, "weighting": weighting}


def shuffle_marginal_audit(rows):
    """Feature-only check: geometry vs shuffled-geometry columns on the scored rows."""
    start = rows["X"]["secondary_context"].shape[1] // 3  # unexpanded secondary width
    width = rows["X"]["geometry_context"].shape[1] // 3
    out = []
    for p in sorted(set(rows["profile"])):
        m = rows["profile"] == p
        g = rows["X"]["geometry_context"][m][:, start:width]
        s = rows["X"]["shuffled_geometry_context"][m][:, start:width]
        out.append({"example_id": str(p), "n_scored_rows": int(m.sum()),
                    "scored_row_geometry_marginal_preserved": bool(np.array_equal(np.sort(g, axis=0), np.sort(s, axis=0))),
                    "fraction_rows_with_changed_geometry": float(np.mean(np.any(g != s, axis=1)))})
    return out


def balanced_weights(rows, selection):
    """Equal lineage weight, equal profile within lineage, precision within profile."""
    w = np.zeros(len(rows["y"]))
    groups = sorted(set(rows["lineage"][selection]))
    for group in groups:
        g = selection & (rows["lineage"] == group)
        profiles = sorted(set(rows["profile"][g]))
        for p in profiles:
            m = g & (rows["profile"] == p)
            w[m] = rows["precision"][m] / rows["precision"][m].sum() / len(profiles) / len(groups)
    return w


def ridge_fit(X, y, weights, alpha):
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite features or targets")
    w = weights / weights.sum()
    mean = (X * w[:, None]).sum(0)
    scale = np.sqrt(((X - mean) ** 2 * w[:, None]).sum(0))
    scale[scale < 1e-12] = 1.0
    z = (X - mean) / scale
    target_mean = float(np.dot(w, y))
    gram = z.T @ (w[:, None] * z) + alpha * np.eye(X.shape[1])
    beta = np.linalg.solve(gram, z.T @ (w * (y - target_mean)))
    return {"mean": mean, "scale": scale, "target_mean": target_mean, "beta": beta,
            "alpha": float(alpha), "effective_df": float(np.trace(np.linalg.solve(gram, gram - alpha * np.eye(X.shape[1]))))}


def predict(model, X):
    return model["target_mean"] + ((X - model["mean"]) / model["scale"]) @ model["beta"]


def profile_metrics(rows, indices, predictions):
    values = []
    for p in sorted(set(rows["profile"][indices])):
        m = indices & (rows["profile"] == p)
        y, pred, precision = rows["y"][m], predictions[m], rows["precision"][m]
        correlation = None
        if len(y) >= 3 and np.std(y) > 1e-12 and np.std(pred) > 1e-12:
            correlation = float(np.corrcoef(y, pred)[0, 1])
        values.append({"example_id": str(p), "lineage_id": str(rows["lineage"][m][0]),
                       "study_id": str(rows["study"][m][0]), "chemistry": str(rows["chemistry"][m][0]), "n": int(m.sum()),
                       "error_weighted_mse": float(np.dot(precision, (pred - y) ** 2) / precision.sum()),
                       "unweighted_mse": float(np.mean((pred - y) ** 2)), "pearson_correlation": correlation})
    return values


def lineage_metrics(profiles):
    grouped = defaultdict(list)
    for p in profiles:
        grouped[p["lineage_id"]].append(p)
    values = []
    for group, ps in sorted(grouped.items()):
        correlations = [p["pearson_correlation"] for p in ps if p["pearson_correlation"] is not None]
        values.append({"lineage_id": group, "profiles": len(ps), "n": sum(p["n"] for p in ps),
                       "error_weighted_mse": float(np.mean([p["error_weighted_mse"] for p in ps])),
                       "unweighted_mse": float(np.mean([p["unweighted_mse"] for p in ps])),
                       "pearson_correlation": float(np.mean(correlations)) if correlations else None})
    return values


def inner_alpha(rows, arm, outer_train):
    groups = sorted(set(rows["lineage"][outer_train]))
    if len(groups) < 2:
        raise ValueError("Inner selection requires at least two training lineages")
    choices = []
    for alpha in ALPHAS:
        scores = []
        for held in groups:
            train = outer_train & (rows["lineage"] != held)
            validation = outer_train & (rows["lineage"] == held)
            w = balanced_weights(rows, train)
            model = ridge_fit(rows["X"][arm][train], rows["y"][train], w[train], alpha)
            pred = np.full(len(rows["y"]), np.nan)
            pred[validation] = predict(model, rows["X"][arm][validation])
            scores.append(lineage_metrics(profile_metrics(rows, validation, pred))[0]["error_weighted_mse"])
        choices.append({"alpha": alpha, "inner_equal_lineage_mse": float(np.mean(scores))})
    # Grid order favours the smaller penalty only for an exact score tie; no test tuning.
    selected = min(choices, key=lambda r: (r["inner_equal_lineage_mse"], r["alpha"]))["alpha"]
    return selected, choices


def check_extraction_audit(audit_path, cohort):
    audit = json.loads(Path(audit_path).read_text())
    if audit.get("status") != "passed" or audit.get("n_failed") != 0:
        raise ValueError("Raw-source extraction audit did not pass; do not freeze")
    if audit.get("cohort_sha256") != sha(cohort):
        raise ValueError("Raw-source extraction audit was made for a different cohort file")
    return {"path": str(Path(audit_path).resolve()), "sha256": sha(audit_path), "n_checks": audit.get("n_checks"),
            "status": audit["status"], "verifier_sha256": audit.get("verifier_sha256")}


def make_plan(cohort, path, extraction_audit):
    examples = normalize_cohort(json.loads(Path(cohort).read_text()))
    source_hashes = verify_sources(examples, cohort)
    audit = check_extraction_audit(extraction_audit, cohort)
    rows = prepared_rows(examples)
    groups = sorted(set(rows["lineage"]))
    if len(groups) < 4:
        raise ValueError("At least four admitted lineages required for this cross-family pilot")
    studies = sorted(set(rows["study"]))
    if len(studies) < 2:
        raise ValueError("At least two original studies required; single-study source is insufficient")
    # Exact and near-identical constructs must never cross a lineage split.
    similarity = check_lineage_independence(examples)
    marginals = shuffle_marginal_audit(rows)
    shuffle_exact = all(m["scored_row_geometry_marginal_preserved"] for m in marginals)
    if not shuffle_exact:
        raise ValueError("Shuffled-geometry control does not preserve scored-row marginals; respecify before freezing")
    chem_by_lineage = {g: sorted(set(rows["chemistry"][rows["lineage"] == g])) for g in groups}
    study_by_lineage = {g: sorted(set(rows["study"][rows["lineage"] == g])) for g in groups}
    lineages_per_chemistry = {c: sorted(set(rows["lineage"][rows["chemistry"] == c])) for c in sorted(set(rows["chemistry"]))}
    plan = {
        "schema_version": VERSION, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cohort_path": str(Path(cohort).resolve()), "cohort_sha256": sha(cohort), "code_sha256": sha(__file__),
        "verified_source_sha256": source_hashes,
        "raw_extraction_audit": audit,
        "scope": "Exploratory cross-family learned structure-to-probe measurement operator on four NON-VIRAL RNA lineages; no inverse-model, blind structure or causal condition-response efficacy claim",
        "no_pretrained_model": True, "required_lineages": 4, "n_lineages": len(groups), "n_studies": len(studies),
        "lineages": groups, "studies": studies, "arms": list(ARMS), "alpha_grid": ALPHAS,
        "feature_counts": {a: int(rows["X"][a].shape[1]) for a in ARMS}, "common_masks": rows["counts"],
        "chemistry_by_lineage": chem_by_lineage, "study_by_lineage": study_by_lineage,
        "lineages_per_chemistry": lineages_per_chemistry,
        "split": "Outer leave-one-entire-lineage-out; inner leave-one-training-lineage-out selection.",
        "lineage_independence_guard": {
            "rule": f"Reject if any cross-lineage pair is exactly identical, has global Needleman-Wunsch identity >= {NEAR_IDENTITY_GLOBAL}, or a Smith-Waterman local alignment with identity >= {NEAR_IDENTITY_LOCAL} spanning >= {NEAR_IDENTITY_LOCAL_COVERAGE} of the shorter sequence (match +1, mismatch -1, linear gap -2).",
            "pairwise": similarity,
            "limit": "Sequence similarity is a guard, not proof of lineage independence; related variants, shared ancestry or repeated crystal copies remain a scientific-curation responsibility."},
        "cross_study_limit": "Lineage folds may share a study/protocol. A leave-one-study-out sensitivity is separate and never labelled cross-study when fewer than two training lineages remain.",
        "outcome": "As-deposited reactivity; no per-test-profile normalization, clipping, rescaling, centering or target transformation",
        "features": "Observed sequence-window identity, chemistry group, five numeric assay covariates with explicit missing flags; supplied source-deposited secondary-structure indicators (not learned from this pilot's target values, but their original derivation may have used chemical-mapping data: annotation-informed baseline); C4-prime distance/packing/angle invariants. Every arm has shared plus fixed DMS-specific and SHAPE-specific slope blocks. No reactivity, uncertainty, observation mask, accession, lineage or study ID enters a predictor.",
        "shuffled_geometry_control": {
            "specification": "Within-profile deterministic permutation (seed = SEED + first 32 bits of SHA-256(example_id)) of geometry rows among positions that are chemically eligible (DMS: A/C; SHAPE: all; from sequence and chemistry only) and have known geometry and secondary annotation.",
            "claim": "Exactly matches the geometry arm's coefficient count, alpha grid and selection; the geometry-feature marginal over the scored rows of every profile is preserved (verified below). Effective degrees of freedom may still differ. A single seeded permutation is a diagnostic negative control, not a randomisation p-value.",
            "scored_row_marginal_verification": marginals},
        "weights": {"error_floor_reactivity_units": ERROR_FLOOR, "missing_error_precision": 1.0,
                    "error_semantics": "REACTIVITY_ERROR as deposited in the RDAT, in deposited reactivity units. Whether it is a standard deviation across replicates or a standard error of the replicate mean is not stated in the local source files: semantics UNKNOWN. The 0.1 floor is an arbitrary pre-specified constant, not measurement calibration. Zero errors would receive floor precision (none occur in this cohort). Profiles without the field (TPP, ydaO) get uniform weight within profile.",
                    "objective": "Inverse-variance precision within profile; equal profiles per lineage; equal lineages. Training-only weighted feature standardization and intercept.",
                    "schemes": list(WEIGHTING_SCHEMES),
                    "sensitivity": "The complete nested analysis (inner alpha selection, training weights and evaluation) is repeated with unit precision for every nucleotide (unweighted refit)."},
        "primary_endpoint": "Equal-lineage average of profile error-weighted MSE (error_weighted scheme); paired held-out lineage differences, reference minus geometry (positive favours geometry)",
        "comparisons": [{"reference": r, "candidate": c} for r, c in COMPARISONS],
        "secondary_endpoints": ["Equal-lineage unweighted MSE", "Full unweighted refit (pre-specified weighting sensitivity)", "Mean within-profile Pearson correlation, constant profiles/predictions reported null", "Chemistry-stratified held-out MSE", "Geometry minus shuffled-pairing negative control", "Study holdout sensitivity"],
        "uncertainty": {"unit": "RNA lineage", "resamples": BOOTSTRAPS, "seed": SEED,
                        "interval": "Descriptive paired percentile 95% bootstrap over 4 lineages (at most 35 distinct resample means); no nucleotide-level significance claim and no powered confirmatory claim"},
        "interpretation_gate": "Geometry is described as adding held-out information only if, in BOTH the error-weighted primary and the unweighted refit, the mean lineage difference favours geometry over the secondary arm AND over the shuffled-geometry control, with at least 3 of 4 held-out lineages favouring geometry in each of those four comparisons. Otherwise the result is reported as null or mixed. Geometry-versus-sequence is reported descriptively. Small lineage count, crystal-as-solution-proxy, annotation-informed secondary baseline and study sensitivity accompany every estimate.",
        "synthetic_data": False,
    }
    # Assert printed coefficient counts instead of silently letting prose drift.
    plan["capacity"] = f"Nested linear ridge with {plan['feature_counts']} feature coefficients plus intercept, identical alpha grid and nested selection. Geometry versus shuffled geometry exactly matches capacity; secondary versus geometry is a nested-feature comparison."
    dump(path, plan)
    return plan


def descriptive_comparison(arms, reference, candidate, metric="error_weighted_mse"):
    r = {x["lineage_id"]: x for x in arms[reference]["lineage_metrics"]}
    c = {x["lineage_id"]: x for x in arms[candidate]["lineage_metrics"]}
    groups = sorted(r.keys() & c.keys())
    d = np.asarray([r[g][metric] - c[g][metric] for g in groups])
    rng = np.random.default_rng(SEED)
    bs = d[rng.integers(0, len(d), size=(BOOTSTRAPS, len(d)))].mean(axis=1)
    reference_mean = float(np.mean([r[g][metric] for g in groups]))
    return {"reference": reference, "candidate": candidate, "metric": metric, "unit": "lineage", "n": len(groups),
            "lineage_differences_reference_minus_candidate": dict(zip(groups, map(float, d))),
            "mean_difference": float(d.mean()),
            "relative_mse_reduction": float(d.mean() / reference_mean) if reference_mean > 0 else None,
            "descriptive_bootstrap_95_interval": list(map(float, np.quantile(bs, [0.025, 0.975]))),
            "n_lineages_favouring_candidate": int((d > 0).sum()), "confirmatory_p_value": None}


def fit_scheme(rows):
    groups = sorted(set(rows["lineage"]))
    results, models, prediction_rows = {}, {}, []
    for arm in ARMS:
        predictions = np.full(len(rows["y"]), np.nan)
        folds = []
        for held in groups:
            train = rows["lineage"] != held
            test = ~train
            alpha, scores = inner_alpha(rows, arm, train)
            w = balanced_weights(rows, train)
            model = ridge_fit(rows["X"][arm][train], rows["y"][train], w[train], alpha)
            predictions[test] = predict(model, rows["X"][arm][test])
            folds.append({"held_out_lineage": str(held), "train_lineages": sorted(set(rows["lineage"][train])),
                          "test_profiles": sorted(set(rows["profile"][test])), "alpha": alpha,
                          "effective_degrees_of_freedom": model["effective_df"], "inner_selection": scores})
            models[f"{rows['weighting']}::{arm}::{held}"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()}
        if not np.isfinite(predictions).all():
            raise RuntimeError("Outer predictions missing")
        profiles = profile_metrics(rows, np.ones(len(predictions), dtype=bool), predictions)
        lineages = lineage_metrics(profiles)
        chemistry_metrics = {}
        for chemistry in sorted(set(rows["chemistry"])):
            subset = [p for p in profiles if p["chemistry"] == chemistry]
            chemistry_lineages = lineage_metrics(subset)
            chemistry_metrics[chemistry] = {"n_lineages": len(chemistry_lineages), "lineage_metrics": chemistry_lineages,
                                           "equal_lineage_error_weighted_mse": float(np.mean([r["error_weighted_mse"] for r in chemistry_lineages]))}
        results[arm] = {"folds": folds, "profile_metrics": profiles, "lineage_metrics": lineages,
                        "chemistry_stratified_metrics": chemistry_metrics,
                        "equal_lineage_error_weighted_mse": float(np.mean([r["error_weighted_mse"] for r in lineages])),
                        "equal_lineage_unweighted_mse": float(np.mean([r["unweighted_mse"] for r in lineages]))}
        for i, pred in enumerate(predictions):
            prediction_rows.append({"weighting": rows["weighting"], "arm": arm, "example_id": str(rows["profile"][i]),
                                    "lineage_id": str(rows["lineage"][i]), "position_zero_based": int(rows["position"][i]),
                                    "target": float(rows["y"][i]), "prediction": float(pred)})
    return results, models, prediction_rows


def study_holdout(rows):
    study_sensitivity = []
    for study in sorted(set(rows["study"])):
        test = rows["study"] == study
        # Exclude all lineages represented in the test study, even if they also
        # occur in another study. This is stronger than accession-only holdout.
        test_lineages = sorted(set(rows["lineage"][test]))
        train = ~np.isin(rows["lineage"], test_lineages)
        remaining = sorted(set(rows["lineage"][train]))
        sr = {"held_out_study": str(study), "held_out_lineages": test_lineages, "train_lineages": remaining,
              "train_chemistries": sorted(set(rows["chemistry"][train])), "test_chemistries": sorted(set(rows["chemistry"][test]))}
        if len(remaining) < 2:
            sr["status"] = "not_estimable_insufficient_training_lineages"
        else:
            sr["status"] = "executed"
            if not set(sr["test_chemistries"]) <= set(sr["train_chemistries"]):
                sr["chemistry_extrapolation"] = "Held-out chemistry absent from training: chemistry-specific slopes are untrained (extrapolation)."
            sr["arms"] = {}
            for arm in ARMS:
                alpha, _ = inner_alpha(rows, arm, train)
                w = balanced_weights(rows, train)
                model = ridge_fit(rows["X"][arm][train], rows["y"][train], w[train], alpha)
                pred = np.full(len(rows["y"]), np.nan)
                pred[test] = predict(model, rows["X"][arm][test])
                sr["arms"][arm] = {"alpha": alpha, "lineage_metrics": lineage_metrics(profile_metrics(rows, test, pred))}
        study_sensitivity.append(sr)
    return study_sensitivity


def gate(comparisons):
    """Apply the frozen interpretation gate to the four gate comparisons."""
    needed = [(s, ref) for s in WEIGHTING_SCHEMES for ref in ("secondary_context", "shuffled_geometry_context")]
    status = {}
    for scheme, ref in needed:
        c = next(x for x in comparisons if x["scheme"] == scheme and x["reference"] == ref
                 and x["candidate"] == "geometry_context" and x["metric"] == ("error_weighted_mse" if scheme == "error_weighted" else "unweighted_mse"))
        status[f"{scheme}:{ref}"] = bool(c["mean_difference"] > 0 and c["n_lineages_favouring_candidate"] >= 3)
    return {"criteria_met": status, "geometry_adds_information_by_frozen_gate": all(status.values())}


def run_plan(cohort, plan_path, output):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema_version") != VERSION or plan.get("cohort_sha256") != sha(cohort) or plan.get("code_sha256") != sha(__file__):
        raise ValueError("Cohort or code changed after freezing; create and review a new plan, do not reuse this plan")
    examples = normalize_cohort(json.loads(Path(cohort).read_text()))
    if verify_sources(examples, cohort) != plan["verified_source_sha256"]:
        raise ValueError("Pinned source collection changed after analysis freeze")
    if sha(plan["raw_extraction_audit"]["path"]) != plan["raw_extraction_audit"]["sha256"]:
        raise ValueError("Raw-source extraction audit changed after analysis freeze")
    schemes, models, prediction_rows, comparisons = {}, {}, [], []
    for weighting in WEIGHTING_SCHEMES:
        rows = prepared_rows(examples, weighting)
        results, m, p = fit_scheme(rows)
        schemes[weighting] = {"arms": results, "study_holdout_sensitivity": study_holdout(rows)}
        models.update(m)
        prediction_rows += p
        metrics = ["error_weighted_mse", "unweighted_mse"] if weighting == "error_weighted" else ["unweighted_mse"]
        for metric in metrics:
            for reference, candidate in COMPARISONS:
                comparisons.append({"scheme": weighting, **descriptive_comparison(results, reference, candidate, metric)})
    report = {"schema_version": VERSION, "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "cohort_sha256": sha(cohort), "plan_sha256": sha(plan_path), "code_sha256": sha(__file__),
              "n_examples": len(examples), "n_lineages": plan["n_lineages"], "n_common_observations": int(len(prepared_rows(examples)["y"])),
              "status": "executed_exploratory_forward_operator_only", "scope": plan["scope"],
              "primary_scheme": "error_weighted", "schemes": schemes, "comparisons": comparisons,
              "interpretation_gate": {"frozen_rule": plan["interpretation_gate"], **gate(comparisons)},
              "limitations": ["Four lineages: exploratory readiness test only; bootstrap intervals over 4 units are descriptive and coarse.",
                              "Residues, replicate profiles and crystal copies are not independent lineages.",
                              "Nested feature models have different dimensions; equal-capacity shuffled-geometry control is reported separately.",
                              "As-deposited scales and protocol/study heterogeneity limit transfer; no held-out target normalization was used.",
                              "Deposited crystal conformations are stand-ins for solution ensembles; buffers, ligands, packing and ions differ. ydaO is a partial 85-nt core with truncated packing context and high B factors.",
                              "The secondary baseline uses supplied deposited annotations whose derivation may have used chemical-mapping data; it is an annotation-informed baseline, possibly advantaged relative to a blind sequence-only structure.",
                              "REACTIVITY_ERROR semantics are unknown and the 0.1 floor is arbitrary; two of four lineages have no reported errors; the unweighted refit is reported alongside.",
                              "No inverse-model adaptation or condition-response inference was run in this pilot."]}
    output = Path(output)
    dump(output / "forward_head_results.json", report)
    dump(output / "held_out_predictions.json", {"scope": "True out-of-lineage predictions", "predictions": prediction_rows})
    dump(output / "fitted_outer_models.json", models)
    dump(output / "execution_provenance.json", {"plan_sha256": sha(plan_path), "cohort_sha256": sha(cohort),
         "code_sha256": sha(__file__), "numpy_version": np.__version__, "python_cpu_only": True,
         "result_sha256": sha(output / "forward_head_results.json"), "prediction_sha256": sha(output / "held_out_predictions.json"),
         "models_sha256": sha(output / "fitted_outer_models.json")})
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="Admit dataset and pin exact cohort/code/splits before fitting")
    freeze.add_argument("--cohort", required=True)
    freeze.add_argument("--plan", required=True)
    freeze.add_argument("--extraction-audit", required=True, help="Passed raw_extraction_verification.json for this exact cohort")
    run = sub.add_parser("run", help="Execute only against unchanged frozen plan")
    run.add_argument("--cohort", required=True)
    run.add_argument("--plan", required=True)
    run.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.command == "freeze":
        p = make_plan(args.cohort, args.plan, args.extraction_audit)
        print(json.dumps({"status": "frozen_not_fitted", "n_lineages": p["n_lineages"], "feature_counts": p["feature_counts"], "plan_sha256": sha(args.plan)}, indent=2))
    else:
        r = run_plan(args.cohort, args.plan, args.output)
        print(json.dumps({"status": r["status"], "n_lineages": r["n_lineages"], "interpretation_gate": r["interpretation_gate"]}, indent=2))


if __name__ == "__main__":
    main()
