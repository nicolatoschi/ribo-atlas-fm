#!/usr/bin/env python3
"""Analysis A, step 1 (structure and sequence only; reads NO reactivity values).

For every riboswitch system: load the designated holo crystal chain + ligand, compute the
a-priori nucleotide classes (L ligand contact, T tertiary contact, R rest; definitions in
structure_features.py), and map crystal residues onto each probed construct.

Mapping rule (frozen): a single numbering offset per (structure, construct) is taken as the
most common (construct index - PDB residue number) over Smith-Waterman identity columns; a
residue maps to construct index resi+offset only if it has no insertion code, is not in an
engineered (non-native) segment, its base equals the construct base, and no residue within
+-2 residue numbers mismatches the construct (guards against coincidental matches in
substituted loops). Engineered segments (crystallisation modules) are listed per system and are
never contact partners. Writes results/analysisA_classes.json.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pdbtools import local_align, sha256  # noqa: E402
from rdat import read_rdat  # noqa: E402
import structure_features as sf  # noqa: E402

ROOT = HERE.parent
LC = ROOT / "inputs" / "local_copies"
DL = ROOT / "inputs" / "downloaded"

SYSTEMS = {
    "cdigmp": {"label": "c-di-GMP-I riboswitch (V. cholerae Vc2)", "pdb": LC / "3MXH.pdb", "chain": "R",
               "ligand": [["C2E", "R", 1]], "engineered_resi": list(range(660, 670)),
               "engineered_note": "U1A-binding crystallisation loop (residues 660-669) replaces the native P3 loop",
               "sensitivity_structures": {"3IRW": {"pdb": DL / "pdb" / "3IRW.pdb", "chain": "R", "ligand": [["C2E", "R", 1]],
                                                   "engineered_resi": list(range(660, 670))}}},
    "adocbl": {"label": "AdoCbl riboswitch (S. thermophilum)", "pdb": DL / "pdb" / "4GXY.pdb", "chain": "A",
               "ligand": [["B1Z", "A", 301]], "engineered_resi": [],
               "engineered_note": "none; 18-atom B1Z A 302 fragment (12.5 A from the complete AdoCbl B1Z A 301) excluded from the primary ligand set",
               "sensitivity_structures": {"4GXY_with_B1Z302": {"pdb": DL / "pdb" / "4GXY.pdb", "chain": "A",
                                                               "ligand": [["B1Z", "A", 301], ["B1Z", "A", 302]], "engineered_resi": []}}},
    "adenine": {"label": "adenine (add) riboswitch (V. vulnificus)", "pdb": LC / "1Y26.pdb", "chain": "X",
                "ligand": [["ADE", "X", 90]], "engineered_resi": [], "engineered_note": "none"},
    "glycine": {"label": "glycine riboswitch tandem (F. nucleatum)", "pdb": LC / "3P49.pdb", "chain": "A",
                "ligand": [["GLY", "A", 745], ["GLY", "A", 746]], "engineered_resi": list(range(700, 800)),
                "engineered_note": "U1A-binding crystallisation loop (residues 721-734) replaces the native P1 loop"},
    "fmn": {"label": "FMN riboswitch (F. nucleatum)", "pdb": DL / "pdb" / "3F2Q.pdb", "chain": "X",
            "ligand": [["FMN", "X", 200]], "engineered_resi": [], "engineered_note": "none"},
    "tpp": {"label": "TPP riboswitch (E. coli thiM)", "pdb": LC / "2GDI.pdb", "chain": "X",
            "ligand": [["TPP", "X", 100]], "engineered_resi": [], "engineered_note": "none (chain Y copy ignored)"},
    "ydao": {"label": "ydaO c-di-AMP riboswitch", "pdb": LC / "4QLM.pdb", "chain": "A",
             "ligand": [["2BA", "A", 203], ["2BA", "A", 204]], "engineered_resi": [], "engineered_note": "none"},
}

CONSTRUCTS = {  # one entry per distinct probed construct (sequence-identical files share a mapping)
    "cdigmp": {"DCP": DL / "rmdb_rdat" / "CDGAPO_DCP_0000.rdat", "K2011": LC / "CIDGMP_SHP_0001.rdat"},
    "adocbl": {"DCP": LC / "CBLAPO_DCP_0000.rdat", "PZ6": DL / "rmdb_rdat" / "RNAPZ6_STD_0001.rdat",
               "M2": DL / "rmdb_rdat" / "ADOCBL_DMS_0000.rdat"},
    "adenine": {"DCP": DL / "rmdb_rdat" / "ADDAPO_DCP_0000.rdat", "K2011": LC / "ADDRSW_SHP_0001.rdat"},
    "glycine": {"DCP": DL / "rmdb_rdat" / "GLYAPO_DCP_0000.rdat", "K2011": LC / "GLYCFN_SHP_0001.rdat"},
    "fmn": {"DCP": DL / "rmdb_rdat" / "FMNAPO_DCP_0000.rdat"},
    "tpp": {"M2": LC / "TPPRSW_DMS_0000.rdat"},
    "ydao": {"M2": LC / "YDAORS_DMS_0000.rdat"},
}


def classes_for(pdb, chain, ligand, engineered_resi):
    res, lig = sf.load_structure(pdb, chain, ligand)
    eng = {k for k, r in enumerate(res) if r["resi"] in set(engineered_resi) or r["icode"]}
    ann, summ = sf.annotate(res, lig, ignore=frozenset(eng))
    for k, a in enumerate(ann):
        a["engineered"] = k in eng
    summ["n_engineered_residues"] = len(eng)
    return res, ann, summ


def map_construct(res, ann, rdat_path):
    r = read_rdat(rdat_path, load_values=False)
    seq = r["sequence"].upper().replace("T", "U")
    cseq = "".join(x["base"] for x in res)
    pairs, _ = local_align(cseq, seq)
    offs = collections.Counter(j + 1 - res[i]["resi"] for i, j in pairs if cseq[i] == seq[j] and not res[i]["icode"])
    offset = offs.most_common(1)[0][0]
    status = {}
    for k, x in enumerate(res):
        ci = x["resi"] + offset
        if x["icode"] or ann[k]["engineered"]:
            status[k] = "engineered_or_icode"
        elif not 1 <= ci <= len(seq):
            status[k] = "outside_construct"
        elif seq[ci - 1] != x["base"]:
            status[k] = "mismatch"
        else:
            status[k] = "identity"
    bad = {res[k]["resi"] for k, s in status.items() if s in ("mismatch", "engineered_or_icode")}
    mapping = []
    for k, x in enumerate(res):
        st = status[k]
        if st == "identity" and any(abs(x["resi"] - b) <= 2 for b in bad):
            st = "excluded_near_mismatch"
        mapping.append({"chain_index": k, "resi": x["resi"], "base": x["base"], "construct_index": x["resi"] + offset if st == "identity" else None,
                        "status": st})
    n_map = sum(m["construct_index"] is not None for m in mapping)
    return {"rdat": str(rdat_path.relative_to(ROOT)), "rdat_sha256": sha256(rdat_path), "construct_length": len(seq),
            "offset_construct_minus_resi": offset, "offset_votes": dict(offs.most_common(3)), "n_mapped": n_map,
            "status_counts": dict(collections.Counter(m["status"] for m in mapping)), "mapping": mapping}


def main():
    out = {"script_sha256": sha256(__file__), "structure_features_sha256": sha256(HERE / "structure_features.py"),
           "pdbtools_sha256": sha256(HERE / "pdbtools.py"), "rdat_sha256": sha256(HERE / "rdat.py"),
           "reactivity_values_read": False, "definitions": sf.__doc__,
           "thresholds": {"HB": sf.HB, "C1C1": sf.C1C1, "PAIR_ANGLE": sf.PAIR_ANGLE, "CONTACT": sf.CONTACT,
                          "LIGAND_CONTACT": sf.LIGAND_CONTACT}, "systems": {}}
    for name, s in SYSTEMS.items():
        res, ann, summ = classes_for(s["pdb"], s["chain"], s["ligand"], s["engineered_resi"])
        entry = {"label": s["label"], "pdb": str(s["pdb"].relative_to(ROOT)), "pdb_sha256": sha256(s["pdb"]),
                 "chain": s["chain"], "ligand": s["ligand"], "engineered_note": s["engineered_note"],
                 "summary": summ, "residues": ann, "constructs": {}, "sensitivity": {}}
        for cname, path in CONSTRUCTS[name].items():
            entry["constructs"][cname] = map_construct(res, ann, path)
        for sname, ss in s.get("sensitivity_structures", {}).items():
            r2, a2, s2 = classes_for(ss["pdb"], ss["chain"], ss["ligand"], ss["engineered_resi"])
            sens = {"pdb": str(ss["pdb"].relative_to(ROOT)), "pdb_sha256": sha256(ss["pdb"]), "ligand": ss["ligand"],
                    "summary": s2, "residues": a2, "constructs": {}}
            for cname, path in CONSTRUCTS[name].items():
                sens["constructs"][cname] = map_construct(r2, a2, path)
            entry["sensitivity"][sname] = sens
        out["systems"][name] = entry
        print(name, summ["class_counts"], {c: v["n_mapped"] for c, v in entry["constructs"].items()})
    path = ROOT / "results" / "analysisA_classes.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path, sha256(path))


if __name__ == "__main__":
    main()
