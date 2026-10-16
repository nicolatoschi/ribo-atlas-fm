#!/usr/bin/env python3
"""Wave F v2 (audit point 4): rebuild every prepared array from raw local bytes.

Hash pinning proves the raw bytes are unchanged; it does not prove the prepared
values were extracted correctly. This script re-derives, with parsers written
independently of prepare_cohort_v2.py (different tokenisation, SEQPOS-label-driven
mapping, separate PDB reader and separate pair-table code), the core sequence,
reactivity, reported error, DMS A/C observation mask, C4' coordinates and the
restricted secondary annotation of each admitted profile, and asserts exact
equality with paired_core_cohort.json. It additionally cross-checks:
  * the four retained v1 profiles are value-identical to the frozen v1 cohort;
  * the ydaO core equals the earlier independent exact-core audit payload;
  * raw-file SHA-256 equals the hash recorded at download time and any other
    local copy of the same file in the project tree;
  * C4'-C4' consecutive distances (no hidden chain break inside a core);
  * organism labels for the ydaO RDAT and 4QLM (audit point 7), local files only.
No network access. No model fitting. Read-only on inputs except the two JSON outputs.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources"
COHORT = HERE / "paired_core_cohort.json"
V1_COHORT = HERE.parents[1] / "biological_validation_pilot" / "cohort" / "paired_core_cohort.json"
YDAO_PAYLOAD = HERE.parent / "benign_cohort" / "ydao_contiguous_core85.json"
YDAO_AUDIT = HERE.parent / "benign_cohort" / "ydao_source_mapping_audit.json"
PROJECT = HERE.parents[3]  # .../RIBO_ATLAS_FM
MANIFESTS = ["source_download_manifest.json", "expanded_source_download_manifest.json",
             "liganded_additional_source_manifest.json"]
KEY = re.compile(r"^([A-Z_]+(?::\d+)?)(.*)$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rdat(path: Path) -> dict:
    """Independent RDAT reader: key = leading [A-Z_](:n)? token, tolerating v0.24
    lines where the value is glued to the key (e.g. 'STRUCTURE....((')."""
    out = {"profiles": {}}
    for raw in path.read_text().splitlines():
        m = KEY.match(raw)
        if not m:
            continue
        key, rest = m.group(1), m.group(2).strip()
        if key in ("SEQUENCE", "STRUCTURE", "OFFSET", "SEQPOS", "NAME"):
            out.setdefault(key, rest)
        elif ":" in key:
            field, n = key.split(":")
            if field in ("REACTIVITY", "REACTIVITY_ERROR", "ANNOTATION_DATA"):
                out["profiles"].setdefault(int(n), {})[field] = rest
    seq = out["SEQUENCE"].split()[0]
    offset = int(out["OFFSET"].split()[0])
    labels = out["SEQPOS"].split()
    columns = {}
    for col, label in enumerate(labels):
        m = re.fullmatch(r"([ACGUTacgut]?)(-?\d+)", label)
        assert m, label
        pos = int(m.group(2))
        if m.group(1):  # label letter must agree with SEQUENCE at pos-offset
            assert m.group(1).upper() == seq[pos - offset - 1].upper(), (path.name, label)
        assert pos not in columns
        columns[pos] = col
    return {"sequence": seq, "offset": offset, "columns": columns, "n_columns": len(labels),
            "structure": out["STRUCTURE"].split()[0] if "STRUCTURE" in out else None,
            "name": out.get("NAME"), "profiles": out["profiles"]}


def values(text: str, n: int) -> list:
    v = [float(x) for x in text.split()]
    assert len(v) == n
    return v


def read_c4(path: Path, chain: str, window=None) -> list:
    """Independent PDB reader for C4' atoms; refuses alternate locations/insertions."""
    seen = {}
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  ") or line[21] != chain:
            continue
        if line[12:16].strip() not in ("C4'", "C4*"):
            continue
        res = line[17:20].strip()
        if res not in ("A", "C", "G", "U"):
            continue
        num = int(line[22:26])
        assert line[26] == " " and line[16] == " ", (path.name, num, "altloc/insertion present")
        if window and not (window[0] <= num <= window[1]):
            continue
        assert num not in seen
        seen[num] = (res, [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    nums = sorted(seen)
    return [(n, *seen[n]) for n in nums]


def core_secondary(structure: str, start: int, length: int):
    """Independent pair table; unbalanced bracket families become unknown (no repair)."""
    families = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closers = {v: k for k, v in families.items()}
    bad = {o for o, c in families.items() if structure.count(o) != structure.count(c)}
    table = [None] * len(structure)
    unknown = set()
    for o, c in families.items():
        if o in bad:
            unknown |= {i for i, ch in enumerate(structure) if ch in (o, c)}
            continue
        stack = []
        for i, ch in enumerate(structure):
            if ch == o:
                stack.append(i)
            elif ch == c:
                j = stack.pop()
                table[i], table[j] = (j, o), (i, o)
        assert not stack
    assert set(structure) <= set(".()[]{}<>")
    out = []
    pairs = []
    for k in range(length):
        i = start + k
        if i in unknown:
            out.append("x")
        elif table[i] is None:
            out.append(".")
        else:
            j, o = table[i]
            if start <= j < start + length:
                out.append(o if j > i else families[o])
                if j > i:
                    pairs.append([k, j - start])
            else:
                out.append(".")
    return "".join(out), sorted(pairs), sorted(i - start for i in unknown if start <= i < start + length)


def main() -> int:
    cohort = json.loads(COHORT.read_text())
    checks, failures = [], []

    def check(name, ok, detail=None):
        checks.append({"check": name, "passed": bool(ok), **({"detail": detail} if detail is not None else {})})
        if not ok:
            failures.append(name)

    recorded = {}
    for m in MANIFESTS:
        for r in json.loads((HERE / m).read_text())["files"]:
            if r.get("sha256"):
                recorded.setdefault(Path(r["path"]).name, set()).add(r["sha256"])

    per_example = []
    for e in cohort["examples"]:
        p = e["provenance"]
        eid = e["example_id"]
        rd = read_rdat(HERE / p["rdat_path"])
        window = tuple(p["pdb_residue_window"]) if p.get("pdb_residue_window") else None
        c4 = read_c4(HERE / p["pdb_path"], p["pdb_chain"], window)
        pdb_seq = "".join(b for _, b, _ in c4)
        full = rd["sequence"].upper()
        hits = [i for i in range(len(full) - len(pdb_seq) + 1) if full[i:i + len(pdb_seq)] == pdb_seq]
        check(f"{eid}: PDB core occurs exactly once in RDAT SEQUENCE", len(hits) == 1, hits)
        start = hits[0]
        if window:
            check(f"{eid}: residue window complete/contiguous", [n for n, _, _ in c4] == list(range(window[0], window[1] + 1)))
        check(f"{eid}: sequence", pdb_seq == e["sequence"])
        check(f"{eid}: coordinates (exact)", [xyz for _, _, xyz in c4] == e["coordinates"])
        steps = [math.dist(c4[k][2], c4[k + 1][2]) for k in range(len(c4) - 1)]
        check(f"{eid}: consecutive C4'-C4' <= 8 A (no hidden chain break)", max(steps) <= 8.0, round(max(steps), 3))
        prof = rd["profiles"][1]
        n = rd["n_columns"]
        react = values(prof["REACTIVITY"], n)
        err = values(prof["REACTIVITY_ERROR"], n) if "REACTIVITY_ERROR" in prof else None
        ys, es, mask = [], [], []
        for k, base in enumerate(pdb_seq):
            col = rd["columns"].get(start + k + 1 + rd["offset"])
            y = react[col] if col is not None and math.isfinite(react[col]) else None
            s = err[col] if (err is not None and col is not None and math.isfinite(err[col]) and err[col] >= 0) else None
            ys.append(y)
            es.append(s)
            mask.append(y is not None and (e["chemistry"] != "DMS" or base in "AC"))
        check(f"{eid}: reactivity (exact)", ys == e["reactivity"])
        check(f"{eid}: reported error (exact; null when absent)", es == e["stderr"])
        check(f"{eid}: observation mask (finite and DMS A/C rule)", mask == e["observed_mask"])
        structure = rd["structure"]
        if structure is None or p["secondary_annotation_rmdb_source"] != p["rmdb_id"]:
            other = read_rdat(HERE / f"sources/{p['secondary_annotation_rmdb_source']}.rdat")
            check(f"{eid}: secondary-source sequence identical", other["sequence"] == rd["sequence"])
            if structure is not None:
                check(f"{eid}: own (glued v0.24) STRUCTURE equals secondary-source STRUCTURE", structure == other["structure"])
            structure = other["structure"]
        dot, pairs, unknown = core_secondary(structure, start, len(pdb_seq))
        check(f"{eid}: restricted secondary dot-bracket", dot == e["secondary_dot_bracket"])
        check(f"{eid}: restricted secondary pair list", pairs == e["secondary_pairs_0based"])
        for key in ("rdat", "metadata", "pdb"):
            path = HERE / p[f"{key}_path"]
            h = digest(path)
            check(f"{eid}: {key} pin", h == p[f"{key}_sha256"])
            if path.name in recorded:
                check(f"{eid}: {path.name} equals download-time hash", h in recorded[path.name], sorted(recorded[path.name]))
        per_example.append({"example_id": eid, "core_construct_1based": [start + 1, start + len(pdb_seq)],
                            "n": len(pdb_seq), "n_observed_mask": sum(mask), "secondary_unknown_core_0based": unknown,
                            "max_consecutive_c4_distance_A": round(max(steps), 3)})

    # Retained v1 profiles must be value-identical (only Zika removed, ydaO added).
    v1 = {x["example_id"]: x for x in json.loads(V1_COHORT.read_text())["examples"]}
    for e in cohort["examples"]:
        if e["example_id"] in v1:
            old = v1[e["example_id"]]
            same = all(old[k] == e[k] for k in ("sequence", "reactivity", "stderr", "observed_mask", "coordinates",
                                                "secondary_dot_bracket", "secondary_pairs_0based", "lineage_id", "study_id", "chemistry"))
            check(f"{e['example_id']}: identical to frozen v1 cohort arrays", same)
    check("Zika removed and ydaO added relative to v1",
          "Zika_xrRNA_DMS_WT" in v1 and all(not x["example_id"].startswith("Zika") for x in cohort["examples"])
          and any(x["lineage_id"] == "ydao_cdiamp_riboswitch" for x in cohort["examples"]))

    # ydaO: equality with the earlier independent exact-core audit payload.
    payload = json.loads(YDAO_PAYLOAD.read_text())
    y = next(x for x in cohort["examples"] if x["lineage_id"] == "ydao_cdiamp_riboswitch")
    ep = next(x for x in payload["endpoints"] if x["source_filename"] == "YDAORS_DMS_0001.rdat")
    check("ydaO: sequence equals mapping-audit payload", y["sequence"] == payload["sequence"])
    check("ydaO: C4' coordinates equal mapping-audit payload", y["coordinates"] == payload["c4prime_coordinates"])
    check("ydaO: reactivity equals mapping-audit payload", y["reactivity"] == ep["reactivity_raw"])
    check("ydaO: A/C evaluation mask equals mapping-audit payload", y["observed_mask"] == ep["evaluation_mask"])
    check("ydaO: PDB residues 39-123 equal mapping audit", [m["pdb_resnum"] for m in y["mapping"]] == payload["pdb_residue_numbers"])
    check("ydaO: RDAT positions 37-121 equal mapping audit", [m["deposited_coordinate"] for m in y["mapping"]] == payload["rdat_numeric_positions"])

    # Other local copies of the same raw files (cross-copy byte identity).
    used = sorted({Path(x["provenance"][f"{k}_path"]).name for x in cohort["examples"] for k in ("rdat", "metadata", "pdb")})
    copies = {}
    for name in used:
        mine = digest(SOURCES / name)
        others = [q for q in PROJECT.rglob(name) if q.resolve() != (SOURCES / name).resolve()
                  and "biological_validation_pilot_v2" not in str(q)]
        copies[name] = {"sha256": mine, "other_local_copies": len(others),
                        "all_identical": all(digest(q) == mine for q in others)}
        check(f"cross-copy identity {name}", copies[name]["all_identical"], copies[name]["other_local_copies"])

    # Audit point 7: organism labels, from local metadata only.
    ydao_rdat = read_rdat(SOURCES / "YDAORS_DMS_0001.rdat")
    md = (SOURCES / "YDAORS_DMS_0001.md").read_text()
    pdb_lines = (SOURCES / "4QLM.pdb").read_text().splitlines()
    source = [l[10:].strip() for l in pdb_lines if l.startswith("SOURCE")]
    seqres = "".join("".join(l[19:].split()) for l in pdb_lines if l.startswith("SEQRES") and l[11] == "A")
    rfam_pdb = [l for l in gzip.open(PROJECT / "work/resource_costing/Rfam.pdb.gz", "rt") if "\t4qlm\t" in l]
    tpp = read_rdat(SOURCES / "TPPRSW_DMS_0001.rdat")["name"]
    tpp_src = [l[10:].strip() for l in (SOURCES / "2GDI.pdb").read_text().splitlines() if l.startswith("SOURCE")]
    full = ydao_rdat["sequence"].upper()
    organism = {
        "schema": "organism_label_check_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Local files only; no download, no web lookup.",
        "ydaO": {
            "rdat_NAME": ydao_rdat["name"],
            "rmdb_metadata_name": re.search(r'^name:\s*"(.*)"', md, re.M).group(1),
            "pdb_4QLM_SOURCE": source,
            "rfam_pdb_mapping_4qlm": [l.strip() for l in rfam_pdb],
            "seqres_A_length": len(seqres),
            "seqres_A_exact_substring_of_rdat_sequence": full.count(seqres) == 1,
            "locally_verified": [
                "RDAT/RMDB label is 'C. subtarraneus' (sic, misspelled 'subterraneus'); no strain or taxid is given in the RDAT or RMDB metadata.",
                "4QLM SOURCE is Thermoanaerobacter tengcongensis, ORGANISM_TAXID 273068, strain MB4.",
                "All 125 SEQRES residues of 4QLM chain A occur exactly once, contiguously and without substitution, in the probed RDAT construct; Rfam maps 4QLM chain A to RF00379 (c-di-AMP riboswitch, ydaO/yuaA leader).",
            ],
            "not_locally_verifiable": "No NCBI taxonomy or LPSN dump exists locally, so the synonymy itself cannot be confirmed from local files. From domain knowledge (not checked here): Thermoanaerobacter tengcongensis was reclassified as Caldanaerobacter subterraneus subsp. tengcongensis (Fardeau et al. 2004, Int J Syst Evol Microbiol 54:467-474), and NCBI taxid 273068 denotes strain MB4 of that subspecies.",
            "conclusion": "The two labels are most likely the same organism under old and current nomenclature (taxonomic synonym), not a construct mismatch. Admission does not depend on the label: exact 125/125 SEQRES identity with the probed construct establishes that the crystallised and probed RNAs share the sequence. Status: 'likely synonym; sequence identity verified locally; nomenclature not verified locally'.",
        },
        "TPP": {
            "rdat_NAME": tpp,
            "pdb_2GDI_SOURCE": tpp_src,
            "conclusion": "The C. subterraneus / T. tengcongensis caveat does not apply to TPP: the RDAT names E. coli and 2GDI is an in vitro transcript with no organism conflict recorded.",
        },
    }
    (HERE / "organism_label_check.json").write_text(json.dumps(organism, indent=2) + "\n")
    check("ydaO 4QLM SEQRES exactly contained in RDAT construct", organism["ydaO"]["seqres_A_exact_substring_of_rdat_sequence"])

    report = {
        "schema": "raw_extraction_verification_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "passed" if not failures else "FAILED",
        "cohort_path": "paired_core_cohort.json",
        "cohort_sha256": digest(COHORT),
        "verifier_sha256": digest(Path(__file__)),
        "v1_cohort_sha256": digest(V1_COHORT),
        "ydao_mapping_audit_sha256": digest(YDAO_AUDIT),
        "ydao_payload_sha256": digest(YDAO_PAYLOAD),
        "organism_label_check_sha256": digest(HERE / "organism_label_check.json"),
        "coordinate_atom": "C4'", "coordinate_units": "angstrom (PDB columns 31-54, as deposited)",
        "n_checks": len(checks), "n_failed": len(failures), "failures": failures,
        "examples": per_example, "raw_file_cross_copies": copies, "checks": checks,
        "scope": "Value-level equality of prepared arrays with independently parsed raw local bytes. This does not validate the scientific suitability of the sources, the deposited secondary annotations, or crystal-to-solution compatibility.",
    }
    (HERE / "raw_extraction_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "n_checks", "n_failed", "failures", "cohort_sha256")}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
