#!/usr/bin/env python3
"""Analysis B, step 1 (structure only; reads no reactivity values).

Base-resolved descriptors for every core nucleotide of the frozen Wave F v2 cohort, computed
from the all-atom deposited crystal chain (whole chain, not only the probed core) plus the
assay ligand where the probed profile is ligand-present. Per nucleotide (fixed physical scales):

  dms_atom_sasa  SASA of the DMS-reactive atom (A N1, C N3; 0 for G/U)          / 10 A^2
  o2p_sasa       SASA of the 2'-O (O2')                                         / 10 A^2
  base_sasa      SASA of all base heavy atoms (base-face exposure)              / 100 A^2
  canonical_pair cis Watson-Crick or G.U pair to any residue of the chain (0/1)
  base_hbond     number of residues with a base-base N/O...N/O <= 3.4 A, capped at 2, / 2
  stack_5p       base stacked on the 5' chain neighbour (0/1)
  stack_3p       base stacked on the 3' chain neighbour (0/1)
  stack_nonlocal base stacked on any non-neighbour residue (0/1)

Geometric criteria are those of structure_features.py (frozen; identical to Analysis A).
Writes results/analysisB_descriptors.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from pdbtools import sha256  # noqa: E402
import structure_features as sf  # noqa: E402

COHORT = ROOT / "inputs/local_copies/paired_core_cohort.json"
LC = ROOT / "inputs/local_copies"
FEATURES = ["dms_atom_sasa", "o2p_sasa", "base_sasa", "canonical_pair", "base_hbond", "stack_5p", "stack_3p", "stack_nonlocal"]
SCALES = {"dms_atom_sasa": 10.0, "o2p_sasa": 10.0, "base_sasa": 100.0}
# Assay ligand present in the scored profile -> include the crystal ligand heavy atoms in the SASA calculation.
LIGANDS = {"adenine_add_SHAPE_holo": [["ADE", "X", 90]], "P4P6_SHAPE_aqueous": [], "P4P6_DMS_aqueous": [],
           "TPP_DMS_holo_WT": [["TPP", "X", 100]], "ydaO_DMS_holo_WT": [["2BA", "A", 203], ["2BA", "A", 204]]}


def describe(pdb, chain, ligand):
    res, lig = sf.load_structure(pdb, chain, ligand)
    pairs = sf.canonical_pairs(res)
    paired = {i for p in pairs for i in p}
    stacked, hb = sf.stacking_and_hbonds(res)
    sasa = sf.sasa_descriptors(res, lig)
    n = len(res)
    out = {}
    for k, r in enumerate(res):
        nb5 = k - 1 if k > 0 and res[k - 1]["resi"] == r["resi"] - 1 else None
        nb3 = k + 1 if k + 1 < n and res[k + 1]["resi"] == r["resi"] + 1 else None
        nonlocal_stack = any(stacked[k, j] for j in range(n) if j not in (k, nb5, nb3) and abs(j - k) > 1)
        s = sasa[k]
        raw = {"dms_atom_sasa": s["dms_atom_sasa"] if s["dms_atom"] else 0.0, "o2p_sasa": s["o2p_sasa"],
               "base_sasa": s["base_sasa"], "canonical_pair": float(k in paired),
               "base_hbond": min(int(hb[k].sum()), 2) / 2.0,
               "stack_5p": float(nb5 is not None and stacked[k, nb5]), "stack_3p": float(nb3 is not None and stacked[k, nb3]),
               "stack_nonlocal": float(nonlocal_stack)}
        out[(r["resi"], r["icode"])] = {"base": r["base"], "raw": raw, "missing_atoms": s["missing_atoms"],
                                        "atoms": {nm: a["xyz"].tolist() for nm, a in r["atoms"].items()}}
    return out


def main():
    cohort = json.loads(COHORT.read_text())
    out = {"script_sha256": sha256(__file__), "structure_features_sha256": sha256(HERE / "structure_features.py"),
           "pdbtools_sha256": sha256(HERE / "pdbtools.py"), "cohort_sha256": sha256(COHORT),
           "reactivity_values_read": False, "features": FEATURES, "scales": SCALES, "ligands_in_sasa": LIGANDS,
           "definition": __doc__, "examples": {}}
    cache = {}
    for e in cohort["examples"]:
        p = e["provenance"]
        pdb = LC / Path(p["pdb_path"]).name
        if sha256(pdb) != p["pdb_sha256"]:
            raise SystemExit(f"PDB hash differs from cohort pin: {pdb}")
        key = (pdb.name, p["pdb_chain"], json.dumps(LIGANDS[e["example_id"]]))
        if key not in cache:
            cache[key] = describe(pdb, p["pdb_chain"], LIGANDS[e["example_id"]])
        desc = cache[key]
        rows, max_dev, missing = [], 0.0, []
        for k, m in enumerate(e["mapping"]):
            d = desc[(m["pdb_resnum"], m["pdb_icode"])]
            if d["base"] != e["sequence"][k]:
                raise SystemExit(f"{e['example_id']}: base mismatch at core {k + 1}")
            # consistency: cohort all-atom coordinates must equal the PDB atoms used here
            for nm, xyz in (e["all_atom_coordinates"][k] or {}).items():
                if nm in d["atoms"]:
                    max_dev = max(max_dev, float(np.max(np.abs(np.asarray(xyz) - np.asarray(d["atoms"][nm])))))
            if d["missing_atoms"]:
                missing.append([k, d["missing_atoms"]])
            raw = dict(d["raw"])
            vals = [raw[f] / SCALES.get(f, 1.0) if raw[f] is not None else None for f in FEATURES]
            rows.append({"core_index_1based": k + 1, "pdb_resnum": m["pdb_resnum"], "base": d["base"], "raw": raw, "scaled": vals,
                         "valid": all(v is not None for v in vals)})
        out["examples"][e["example_id"]] = {"pdb": pdb.name, "chain": p["pdb_chain"], "ligand_in_sasa": LIGANDS[e["example_id"]],
                                            "n_core": len(rows), "n_valid": sum(r["valid"] for r in rows),
                                            "max_abs_coordinate_deviation_vs_cohort": max_dev, "residues_with_missing_atoms": missing,
                                            "rows": rows}
        means = np.nanmean(np.array([[v if v is not None else np.nan for v in r["scaled"]] for r in rows], float), axis=0)
        print(e["example_id"], "n", len(rows), "valid", sum(r["valid"] for r in rows), "max_dev", round(max_dev, 4),
              "missing", missing[:3], "feature means", np.round(means, 3).tolist())
    path = ROOT / "results" / "analysisB_descriptors.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path, sha256(path))


if __name__ == "__main__":
    main()
