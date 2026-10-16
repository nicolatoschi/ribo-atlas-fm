"""Audit only the authorized public, nonviral ydaO source/coordinate candidates.

This is passive data preparation. It does not infer, alter, or design sequences.
Only the WT profile is read for endpoint admission; mutation profiles are excluded.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCES = HERE.parent / "cohort" / "sources"
RDAT_FILES = ("YDAORS_DMS_0000.rdat", "YDAORS_DMS_0001.rdat")
PDB_FILES = ("4OQU.pdb", "4QLM.pdb", "4P8Z.pdb")


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "filename": path.name,
        "source_path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def parse_rdat(filename: str) -> dict:
    path = SOURCES / filename
    lines = path.read_text().splitlines()
    fields = {}
    line_numbers = {}
    for n, line in enumerate(lines, 1):
        if line.strip():
            key, *values = line.split()
            fields.setdefault(key, []).append(values)
            line_numbers.setdefault(key, []).append(n)
    assert len(fields["SEQUENCE"]) == 1
    assert fields["ANNOTATION_DATA:1"] == [["mutation:WT"]]
    sequence = fields["SEQUENCE"][0][0]
    labels = fields["SEQPOS"][0]
    reactivity = [float(v) for v in fields["REACTIVITY:1"][0]]
    assert len(labels) == len(sequence) == len(reactivity)
    errors = fields.get("REACTIVITY_ERROR:1")
    errors = [float(v) for v in errors[0]] if errors else [None] * len(sequence)
    assert len(errors) == len(sequence)
    offset = int(fields["OFFSET"][0][0])
    for i, label in enumerate(labels):
        match = re.fullmatch(r"([ACGUacgu])(-?\d+)", label)
        assert match is not None
        assert match[1] == sequence[i]
        assert int(match[2]) == i + offset + 1
    return {
        **identity(path),
        "name": " ".join(fields["NAME"][0]),
        "sequence": sequence,
        "seqpos_labels": labels,
        "reactivity": reactivity,
        "errors": errors,
        "annotation": fields["ANNOTATION"][0],
        "comments": [" ".join(x) for x in fields.get("COMMENT", [])],
        "rdat_version": fields["RDAT_VERSION"][0][0],
        "offset": offset,
        "wt_profile_id": 1,
        "total_profiles": sum(k.startswith("REACTIVITY:") for k in fields),
        "reported_error_profile_count": sum(k.startswith("REACTIVITY_ERROR:") for k in fields),
        "line_numbers": {
            key: line_numbers.get(key, [])
            for key in ("NAME", "SEQUENCE", "ANNOTATION", "ANNOTATION_DATA:1", "SEQPOS", "REACTIVITY:1", "REACTIVITY_ERROR:1")
        },
    }


def parse_pdb(filename: str) -> dict:
    path = SOURCES / filename
    lines = path.read_text().splitlines()
    seqres = defaultdict(list)
    c4 = defaultdict(dict)
    for n, line in enumerate(lines, 1):
        if line.startswith("SEQRES"):
            seqres[line[11]].extend(line[19:].split())
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM  ") and line[12:16].strip() in ("C4'", "C4*"):
            chain, number, insertion = line[21], int(line[22:26]), line[26].strip()
            base = line[17:20].strip()
            if base not in {"A", "C", "G", "U"}:
                continue
            assert insertion == "", (filename, chain, number, insertion)
            # These authorized candidates contain one coordinate per C4' site.
            assert number not in c4[chain], (filename, chain, number)
            xyz = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
            assert all(math.isfinite(v) for v in xyz)
            c4[chain][number] = {
                "base": base,
                "xyz": xyz,
                "source_line": n,
                "atom_name": line[12:16].strip(),
                "alternate_location": line[16].strip(),
                "occupancy": float(line[54:60]),
                "b_factor": float(line[60:66]),
            }
    return {
        **identity(path),
        "pdb_id": filename.removesuffix(".pdb"),
        "title": " ".join(line[10:].strip() for line in lines if line.startswith("TITLE")),
        "compound_records": [line[10:].strip() for line in lines if line.startswith("COMPND")],
        "source_records": [line[10:].strip() for line in lines if line.startswith("SOURCE")],
        "journal_records": [line[12:].strip() for line in lines if line.startswith("JRNL")],
        "resolution_record": next(line.strip() for line in lines if line.startswith("REMARK   2 RESOLUTION.")),
        "missing_residue_records": [line.strip() for line in lines if line.startswith("REMARK 465") and re.search(r"[ACGU] A\s+\d+", line)],
        "seqres": {chain: "".join(bases) for chain, bases in seqres.items()},
        "c4": c4,
    }


def main() -> None:
    rdat = {filename: parse_rdat(filename) for filename in RDAT_FILES}
    pdb = {filename: parse_pdb(filename) for filename in PDB_FILES}
    source_sequence = rdat[RDAT_FILES[0]]["sequence"]
    assert source_sequence == rdat[RDAT_FILES[1]]["sequence"]
    target = pdb["4QLM.pdb"]
    seqres = target["seqres"]["A"]
    hits = [i for i in range(len(source_sequence) - len(seqres) + 1) if source_sequence[i:i + len(seqres)].upper() == seqres]
    assert hits == [23], hits
    start = hits[0]
    observed = target["c4"]["A"]
    assert len(observed) == 108
    for number, atom in observed.items():
        assert atom["base"] == seqres[number - 1] == source_sequence[start + number - 1].upper()

    blocks = []
    for number in sorted(observed):
        if not blocks or number != blocks[-1][-1] + 1:
            blocks.append([])
        blocks[-1].append(number)
    chosen = max(blocks, key=len)
    assert chosen == list(range(39, 124))
    indices = [start + number - 1 for number in chosen]
    sequence = "".join(source_sequence[i].upper() for i in indices)
    coordinates = [observed[number]["xyz"] for number in chosen]
    chemical_mask = [base in "AC" for base in sequence]
    endpoints = []
    for filename, record in rdat.items():
        values = [record["reactivity"][i] for i in indices]
        errors = [record["errors"][i] for i in indices]
        available = [math.isfinite(value) and value >= 0 for value in values]
        error_mask = [error is not None and math.isfinite(error) and error > 0 for error in errors]
        assert all(available)
        endpoints.append({
            "endpoint_id": filename.removesuffix(".rdat") + "_WT_core85",
            "lineage_id": "ydao_cdiamp_riboswitch",
            "source_filename": filename,
            "profile_id": 1,
            "profile_annotation": "mutation:WT",
            "modifier": "DMS",
            "ligand_state": "10uM_c-di-AMP" if filename.endswith("0001.rdat") else "no_c-di-AMP_recorded",
            "source_annotations": record["annotation"],
            "reactivity_raw": values,
            "reactivity_error_raw": errors,
            "reactivity_available_mask": available,
            "reported_positive_error_mask": error_mask,
            "chemical_A_C_mask": chemical_mask,
            "evaluation_mask": [a and c for a, c in zip(available, chemical_mask)],
            "counts": {
                "finite_nonnegative_raw": sum(available),
                "A_C_evaluation_sites": sum(a and c for a, c in zip(available, chemical_mask)),
                "reported_positive_error_sites": sum(error_mask),
            },
        })

    payload = {
        "schema": "benign_rna_structure_probing_exact_core_candidate_v1",
        "lineage_id": "ydao_cdiamp_riboswitch",
        "role": "candidate_fourth_independent_lineage",
        "geometry_provenance": "X-ray crystal C4prime coordinates used as a solution-geometry proxy",
        "coordinate_file": "4QLM.pdb",
        "chain": "A",
        "coordinate_atom": "C4prime",
        "coordinate_units": "angstrom",
        "sequence": sequence,
        "length": len(sequence),
        "pdb_residue_numbers": chosen,
        "rdat_sequence_indices_0based": indices,
        "rdat_sequence_positions_1based": [i + 1 for i in indices],
        "rdat_seqpos_labels": [rdat[RDAT_FILES[0]]["seqpos_labels"][i] for i in indices],
        "rdat_numeric_positions": [int(re.sub(r"^[ACGUacgu]", "", rdat[RDAT_FILES[0]]["seqpos_labels"][i])) for i in indices],
        "c4prime_coordinates": coordinates,
        "coordinate_available_mask": [True] * len(sequence),
        "coordinate_source_lines": [observed[number]["source_line"] for number in chosen],
        "endpoints": endpoints,
        "endpoint_selection_recommendation": "Use the ligand-present WT endpoint for comparison with the ligand-bound crystal; treat the ligand-absent profile as a same-lineage condition, not another independent lineage.",
        "uncertainty_status": "Source RDAT files provide no REACTIVITY_ERROR fields. Missing values remain null with false masks; no measurement errors were inferred or substituted.",
        "admission": {
            "exact_sequence_and_finite_contiguous_coordinates": True,
            "finite_raw_WT_reactivity": True,
            "reported_measurement_error_available": False,
            "eligible_if_reported_measurement_error_is_required": False,
            "eligible_for_explicitly_unweighted_exploratory_geometry_proxy_analysis": True,
        },
    }

    audit = {
        "schema": "ydao_source_mapping_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Only the two authorized ydaO RDAT files and three authorized PDB candidates were read. Only source WT profiles were selected; no mutation profiles were admitted.",
        "source_records": [
            {key: value for key, value in record.items() if key not in ("sequence", "seqpos_labels", "reactivity", "errors")}
            for record in rdat.values()
        ],
        "coordinate_candidates": [
            {
                **{key: value for key, value in record.items() if key not in ("seqres", "c4")},
                "seqres_lengths": {chain: len(seq) for chain, seq in record["seqres"].items()},
                "observed_c4prime_counts": {chain: len(atoms) for chain, atoms in record["c4"].items()},
                "full_seqres_exact_source_matches_0based": {
                    chain: [i for i in range(len(source_sequence) - len(seq) + 1) if source_sequence[i:i + len(seq)].upper() == seq]
                    for chain, seq in record["seqres"].items()
                },
                "decision": "Exact ydaO source match; retain the longest finite contiguous C4prime block." if record["pdb_id"] == "4QLM" else "Exclude: PDB primary title identifies a different RNA; no whole-SEQRES exact source match.",
            }
            for record in pdb.values()
        ],
        "exact_mapping": {
            "pdb_id": "4QLM",
            "chain": "A",
            "seqres_length": len(seqres),
            "source_substring_indices_0based_half_open": [start, start + len(seqres)],
            "sequence_identity": 1.0,
            "source_positions": "g-1 through C123",
            "pdb_to_rdat_numeric_position": "RDAT position = PDB residue number - 2",
            "all_observed_c4prime_exact_matches": len(observed),
            "coordinate_contiguous_blocks": [{"pdb_start": block[0], "pdb_end": block[-1], "length": len(block)} for block in blocks],
            "selected_pdb_residues": [chosen[0], chosen[-1]],
            "selected_rdat_numeric_positions": [37, 121],
            "selected_length": len(chosen),
            "selected_coordinate_mask_all_true": True,
            "selected_sequence_identity": 1.0,
            "selected_insertion_deletion_or_substitution_count": 0,
        },
        "endpoint_counts": [dict(endpoint_id=endpoint["endpoint_id"], **endpoint["counts"]) for endpoint in endpoints],
        "source_context_caveats": [
            "RDAT NAME says C. subtarraneus; PDB SOURCE says Thermoanaerobacter tengcongensis. These labels are not silently reconciled. Exact SEQRES/source-substring identity is established independently.",
            "PDB primary metadata flag an engineered experimental construct. This preparation only reuses its deposited coordinates; it does not propose or perform engineering or sequence design.",
            "The RDAT experiment is MutateAndMap, but only mutation:WT profile 1 is admitted here; the 191 remaining profiles per file are excluded.",
            "Neither RDAT file provides reactivity errors or biological replicate identifiers. Finite measurements are not evidence of known uncertainty or independent replication.",
            "The crystal is ligand bound. The ligand-present RDAT profile is the primary condition-matched candidate; ligand-absent data have an additional state mismatch.",
            "The exact 85-nt core is a partial RNA domain. Crystal-to-solution proxy and omitted-coordinate/context limitations remain.",
            "A fourth distinct RNA lineage supports exploratory leave-one-lineage-out measurement-model checks only. It does not establish blind efficacy, convergence, power, or general biological prediction.",
        ],
        "admission": payload["admission"],
        "payload_file": str(HERE / "ydao_contiguous_core85.json"),
    }

    (HERE / "ydao_contiguous_core85.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (HERE / "ydao_source_mapping_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    with (HERE / "ydao_contiguous_core85_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["endpoint_id", "lineage_id", "core_index_0based", "base", "pdb_chain", "pdb_residue", "rdat_seqpos", "rdat_sequence_index_0based", "c4prime_x", "c4prime_y", "c4prime_z", "reactivity_raw", "reactivity_error_raw", "reactivity_available", "reported_positive_error_available", "dms_A_C_evaluation_mask"])
        writer.writeheader()
        for endpoint in endpoints:
            for j, number in enumerate(chosen):
                writer.writerow({
                    "endpoint_id": endpoint["endpoint_id"], "lineage_id": payload["lineage_id"], "core_index_0based": j,
                    "base": sequence[j], "pdb_chain": "A", "pdb_residue": number,
                    "rdat_seqpos": payload["rdat_seqpos_labels"][j], "rdat_sequence_index_0based": indices[j],
                    "c4prime_x": coordinates[j][0], "c4prime_y": coordinates[j][1], "c4prime_z": coordinates[j][2],
                    "reactivity_raw": endpoint["reactivity_raw"][j], "reactivity_error_raw": endpoint["reactivity_error_raw"][j],
                    "reactivity_available": endpoint["reactivity_available_mask"][j], "reported_positive_error_available": endpoint["reported_positive_error_mask"][j],
                    "dms_A_C_evaluation_mask": endpoint["evaluation_mask"][j],
                })
    print(json.dumps({"core_length": payload["length"], "endpoints": audit["endpoint_counts"], "admission": audit["admission"]}, indent=2))


if __name__ == "__main__":
    main()
