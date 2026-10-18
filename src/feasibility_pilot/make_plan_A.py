#!/usr/bin/env python3
"""Write the Analysis A pre-registration (plans/analysisA_plan.json). Reads RDAT headers only
(row annotations, sequences) to verify that each apo/holo pair is matched; never reads values."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from rdat import read_rdat  # noqa: E402

DL, LC = "inputs/downloaded/rmdb_rdat/", "inputs/local_copies/"
LIG = {"cdigmp": "cyclic-diguanosine-monophosphate", "adocbl": "adenosyl", "adenine": "adenine", "glycine": "glycine",
       "fmn": "FMN", "tpp": "thiamine", "ydao": "c-di-AMP"}


def P(pid, system, tier, construct, chem, apo, arow, holo, hrow, structure="primary", note=""):
    return {"id": pid, "system": system, "tier": tier, "construct": construct, "chemistry": chem, "structure": structure,
            "apo": {"file": apo, "row": arow}, "holo": {"file": holo, "row": hrow}, "note": note}


PAIRS = [
    P("cdigmp_DCP_1M7", "cdigmp", "primary", "DCP", "SHAPE", DL + "CDGAPO_DCP_0000.rdat", 1, DL + "CDGHLO_DCP_0000.rdat", 1, note="1M7; 0 vs 10 uM c-di-GMP"),
    P("adocbl_DCP_1M7", "adocbl", "primary", "DCP", "SHAPE", LC + "CBLAPO_DCP_0000.rdat", 1, LC + "CBLHLO_DCP_0000.rdat", 1, note="1M7; 0 vs 70 uM AdoCbl"),
    P("cdigmp_DCP_DMS", "cdigmp", "replication", "DCP", "DMS", DL + "CDGAPO_DCP_0000.rdat", 2, DL + "CDGHLO_DCP_0000.rdat", 2),
    P("cdigmp_K2011_SHAPE", "cdigmp", "replication", "K2011", "SHAPE", LC + "CIDGMP_SHP_0001.rdat", 2, LC + "CIDGMP_SHP_0001.rdat", 1, note="same RDAT; 14/15 replicates"),
    P("adocbl_DCP_DMS", "adocbl", "replication", "DCP", "DMS", LC + "CBLAPO_DCP_0000.rdat", 2, LC + "CBLHLO_DCP_0000.rdat", 2),
    P("adocbl_PZ6_1M7", "adocbl", "replication", "PZ6", "SHAPE", DL + "RNAPZ6_STD_0001.rdat", 5, DL + "RNAPZ6_STD_0001.rdat", 6, note="same RDAT; preliminary, 2 replicates"),
    P("adocbl_PZ6_DMS", "adocbl", "replication", "PZ6", "DMS", DL + "RNAPZ6_STD_0001.rdat", 1, DL + "RNAPZ6_STD_0001.rdat", 2),
    P("adocbl_PZ6_CMCT", "adocbl", "replication", "PZ6", "CMCT", DL + "RNAPZ6_STD_0001.rdat", 3, DL + "RNAPZ6_STD_0001.rdat", 4),
    P("adocbl_M2_DMS", "adocbl", "replication", "M2", "DMS", DL + "ADOCBL_DMS_0000.rdat", 1, DL + "ADOCBL_DMS_0001.rdat", 1, note="mutate-and-map WT rows; 300 mM cacodylate pH 7"),
    P("adenine_DCP_1M7", "adenine", "extension", "DCP", "SHAPE", DL + "ADDAPO_DCP_0000.rdat", 1, DL + "ADDHLO_DCP_0000.rdat", 1, note="0 vs 5 mM adenine"),
    P("adenine_DCP_DMS", "adenine", "extension", "DCP", "DMS", DL + "ADDAPO_DCP_0000.rdat", 2, DL + "ADDHLO_DCP_0000.rdat", 2),
    P("adenine_K2011_SHAPE", "adenine", "extension", "K2011", "SHAPE", LC + "ADDRSW_SHP_0001.rdat", 2, LC + "ADDRSW_SHP_0001.rdat", 1),
    P("glycine_DCP_1M7", "glycine", "extension", "DCP", "SHAPE", DL + "GLYAPO_DCP_0000.rdat", 1, DL + "GLYHLO_DCP_0000.rdat", 1, note="0 vs 10 mM glycine"),
    P("glycine_DCP_DMS", "glycine", "extension", "DCP", "DMS", DL + "GLYAPO_DCP_0000.rdat", 2, DL + "GLYHLO_DCP_0000.rdat", 2),
    P("glycine_K2011_SHAPE", "glycine", "extension", "K2011", "SHAPE", LC + "GLYCFN_SHP_0001.rdat", 2, LC + "GLYCFN_SHP_0001.rdat", 1),
    P("fmn_DCP_1M7", "fmn", "extension", "DCP", "SHAPE", DL + "FMNAPO_DCP_0000.rdat", 1, DL + "FMNHLO_DCP_0000.rdat", 1, note="0 vs 200 uM FMN"),
    P("fmn_DCP_DMS", "fmn", "extension", "DCP", "DMS", DL + "FMNAPO_DCP_0000.rdat", 2, DL + "FMNHLO_DCP_0000.rdat", 2),
    P("tpp_M2_DMS", "tpp", "extension", "M2", "DMS", LC + "TPPRSW_DMS_0000.rdat", 1, LC + "TPPRSW_DMS_0001.rdat", 1, note="mutate-and-map WT rows; cacodylate"),
    P("ydao_M2_DMS", "ydao", "extension", "M2", "DMS", LC + "YDAORS_DMS_0000.rdat", 1, LC + "YDAORS_DMS_0001.rdat", 1, note="mutate-and-map WT rows; cacodylate"),
    P("cdigmp_DCP_1M7_classes3IRW", "cdigmp", "structure_sensitivity", "DCP", "SHAPE", DL + "CDGAPO_DCP_0000.rdat", 1, DL + "CDGHLO_DCP_0000.rdat", 1, structure="3IRW"),
    P("adocbl_DCP_1M7_withB1Z302", "adocbl", "structure_sensitivity", "DCP", "SHAPE", LC + "CBLAPO_DCP_0000.rdat", 1, LC + "CBLHLO_DCP_0000.rdat", 1, structure="4GXY_with_B1Z302"),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def header_check(pair):
    a = read_rdat(ROOT / pair["apo"]["file"], load_values=False)
    h = read_rdat(ROOT / pair["holo"]["file"], load_values=False)
    ann_a = " ".join(a["annotation"] + a["rows"][pair["apo"]["row"]].get("annotation", []))
    ann_h = " ".join(h["annotation"] + h["rows"][pair["holo"]["row"]].get("annotation", []))
    lig = LIG[pair["system"]].lower()
    ok = {"same_sequence": a["sequence"].upper() == h["sequence"].upper(),
          "same_offset_and_seqpos": a["offset"] == h["offset"] and a["seqpos"] == h["seqpos"],
          "ligand_in_holo_only": lig in ann_h.lower() and lig not in ann_a.lower()}
    extra = {"tpp": ["pyrophosphate"]}.get(pair["system"], [])
    strip = lambda s: sorted(t for t in s.split() if lig not in t.lower() and not any(x in t.lower() for x in extra)
                             and not t.lower().startswith("mutation"))
    ok["identical_annotations_apart_from_ligand"] = strip(ann_a) == strip(ann_h)
    ok["same_comment_lines"] = a["comments"] == h["comments"]
    want = {"SHAPE": ("modifier:1m7", "modifier:shape", "modifier:nmia"), "DMS": ("modifier:dms",), "CMCT": ("modifier:cmct",)}[pair["chemistry"]]
    ok["modifier_matches_chemistry"] = any(w in ann_a.lower() for w in want) and any(w in ann_h.lower() for w in want)
    if not all(v for k, v in ok.items() if k != "same_comment_lines"):
        raise SystemExit(f"{pair['id']}: header mismatch {ok}\n{ann_a}\n{ann_h}")
    return {"apo_annotation": ann_a, "holo_annotation": ann_h, "checks": ok,
            "apo_comments": a["comments"], "holo_comments": h["comments"]}


def main():
    for p in PAIRS:
        p["header_match"] = header_check(p)
    files = {"scripts/analysis_a.py", "scripts/rdat.py", "scripts/pdbtools.py", "scripts/structure_features.py",
             "scripts/build_classes_A.py", "scripts/make_plan_A.py", "results/analysisA_classes.json"}
    files |= {p["apo"]["file"] for p in PAIRS} | {p["holo"]["file"] for p in PAIRS}
    plan = {
        "schema": "feasibility-analysisA-plan-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "title": "Ligand-response localisation of chemical-probing changes (apo vs holo) against a-priori crystal contact classes",
        "status": "FROZEN before any apo/holo reactivity difference was computed or inspected",
        "classes_file": "results/analysisA_classes.json",
        "class_definitions": "See results/analysisA_classes.json 'definitions' (structure_features.py docstring): L = any heavy atom within 4.0 A of the designated ligand heavy atoms; T = not L and a <=4.0 A heavy-atom contact to a non-local residue (not a sequence neighbour within 3, not its nested canonical partner, not in the same nested helix, not in the same hairpin or internal/bulge loop unit; pseudoknot/kissing canonical pairs and junction contacts count as tertiary); R = rest. Engineered crystallisation-module residues are excluded from mapping and are never contact partners. Canonical pairs and helices are detected from the crystal geometry alone (no chemical-mapping-derived annotation enters the classes).",
        "matching_rule": "A pair is admitted only if apo and holo share the identical construct sequence, SEQPOS/offset, modifier, buffer, Mg2+ and temperature annotations, and differ only by the ligand annotation (checked from RDAT headers, recorded per pair). Pairs come either from the same RDAT (rows of one experiment) or from the Deep Chemical Profiling apo/holo deposit pairs that share one experiment identifier.",
        "structures": {"cdigmp": "3MXH chain R + C2E R 1 (2.3 A; U1A loop 660-669 engineered); sensitivity 3IRW chain R",
                       "adocbl": "4GXY chain A + B1Z A 301 (3.05 A); sensitivity adds the 18-atom B1Z A 302 fragment",
                       "adenine": "1Y26 chain X + ADE X 90", "glycine": "3P49 chain A + GLY A 745, 746 (U1A loop engineered)",
                       "fmn": "3F2Q chain X + FMN X 200", "tpp": "2GDI chain X + TPP X 100", "ydao": "4QLM chain A + 2BA A 203, 204"},
        "profile_pairs": PAIRS,
        "scored_positions": "Construct positions mapped to the crystal chain (identity mapping, see classes file), chemically eligible (SHAPE/1M7: all; DMS: A and C; CMCT: G and U), with finite reactivity in BOTH apo and holo rows. Ordered by construct index.",
        "statistics": {
            "seed": 20261018,
            "normalisation": {"method": "2-8% normalisation of each profile separately over its scored positions: drop the top round(0.02 n) values, divide by the mean of the next max(1, round(0.08 n)) values",
                              "drop_top_fraction": 0.02, "average_next_fraction": 0.08,
                              "justification": "apo and holo profiles were processed as separate profiles (separate RDAT files for DCP); a robust per-profile scale removes global intensity/scale differences so that |delta| reflects positional changes. As-deposited delta is a pre-specified sensitivity."},
            "response": "r_i = |holo_norm_i - apo_norm_i|",
            "error_scaled_floor": 0.05,
            "error_scaled_sensitivity": "r_i / sqrt(sd_apo^2 + sd_holo^2 + 0.05^2) with deposited REACTIVITY_ERROR rescaled by the same normalisation factors (only pairs where both rows report errors)",
            "contrasts": ["LT_vs_R", "L_vs_R", "T_vs_R"],
            "primary_statistic": "AUC (Mann-Whitney, ties 0.5) of r for L+T versus R; AUC>0.5 means larger ligand-induced changes at structural-contact nucleotides",
            "null": "Exact circular-shift null: the class-label vector is rotated over the ordered scored positions by every shift k=0..n-1 (k=0 is the observed labelling); one-sided p = fraction of shifts with AUC >= observed. Preserves the autocorrelation of both the response and the class runs.",
            "block_length": 5, "n_block_permutations": 10000,
            "block_null": "Sensitivity: label blocks of 5 consecutive scored positions permuted (within-block order kept), 10,000 permutations, p=(1+#>=)/(1+N)",
            "n_bootstrap": 10000,
            "interval": "Descriptive 95% moving-block bootstrap interval (block 5) for the AUC",
            "top_decile": "Enrichment of L+T among the top ceil(0.1 n) |delta| positions relative to their overall fraction, circular-shift p",
            "min_scored": 20, "min_class": 5},
        "multiplicity": {"primary_family": ["cdigmp_DCP_1M7", "adocbl_DCP_1M7"], "primary_correction": "Holm over the two primary tests, alpha 0.05",
                         "secondary": "Benjamini-Hochberg over all replication and extension pair-level LT_vs_R tests (q reported); structure-sensitivity and variant (as-deposited, error-scaled, block-null) results are descriptive robustness checks, not additional tests"},
        "pooled_endpoint": {"description": "Mean over the five Deep Chemical Profiling 1M7 apo/holo pairs (one per riboswitch) of (AUC_LT - 0.5); null by independent random circular shifts in each system; Fisher combination reported alongside",
                            "pair_ids": ["cdigmp_DCP_1M7", "adocbl_DCP_1M7", "adenine_DCP_1M7", "glycine_DCP_1M7", "fmn_DCP_1M7"], "n_draws": 100000},
        "interpretation": {"supported": "A primary system supports ligand-response localisation if AUC_LT > 0.5 with Holm-adjusted circular-shift p < 0.05.",
                           "null": "Otherwise the system is reported as null for this endpoint (not as evidence of absence).",
                           "scope": "Crystal holo contacts are a structural proxy for the solution holo state; the test asks whether ligand-induced reactivity changes concentrate at a-priori structural-contact nucleotides. It does not test a learned model and does not establish mechanism."},
        "not_executed": [{"id": "adocbl_A124U_A125U_mutant_response",
                          "reason": "No public 1D chemical-mapping profiles exist for the A124U/A125U double mutant (RMDB holds only MOHCA-seq 2D data, RNAPZ6_MCA_0004-0007, metadata checked). Those MOHCA data are the application's withheld response endpoint, so they were not opened. The mutant-versus-parent pocket-response comparison is therefore not testable here and is deferred to the award (G1 admission)."},
                         {"id": "other_RMDB_holo_only_profiles", "reason": "CIDGMP_DMS_0001, CIDGMP_CMC_0001, ADDRSW_DMS_0001/CMC_0001, GLYCFN_DMS_0001/CMC_0001 contain a ligand-present row only (no matched apo row): excluded."},
                         {"id": "37C_1M7_series", "reason": "CIDGMP/ADDRSW/TPPRSW_1M7_0001-0003 (37 C, 100 mM HEPES/NaCl) are ligand-present replicates without a matched apo deposit in the same series: excluded."},
                         {"id": "SAM-IV, glycine tandem (VCKT/FNKT), CLILIG", "reason": "No holo crystal structure used here (cryo-EM / not riboswitch / tandem constructs not matched to an on-disk crystal); not admitted."}],
        "disclosures": ["Before this plan was frozen, the first ~40 deposited values of CIDGMP_SHP_0001 rows 1 and 2 (5' flank and first P1 nucleotides) were displayed once during a file-format inspection; no apo/holo difference was computed or examined for any system before freezing.",
                        "A plumbing test with seeded random (lognormal) reactivities was run before freezing (results/synthetic_test_analysisA.json); it contains no real data.",
                        "Base-pair (3.8 A / 60 deg) and stacking thresholds were set on the crystal structures alone (agreement with deposited RDAT secondary structures; helix-neighbour stacking rate), before any reactivity was read."],
        "frozen_files_sha256": {f: sha(ROOT / f) for f in sorted(files)},
    }
    out = ROOT / "plans" / "analysisA_plan.json"
    out.write_text(json.dumps(plan, indent=1) + "\n")
    print("plan written", out, sha(out))
    for p in PAIRS:
        print(p["id"], p["header_match"]["checks"])


if __name__ == "__main__":
    main()
