#!/usr/bin/env python3
"""Copy the already-on-disk inputs (read-only originals elsewhere in the RIBO tree) into
inputs/local_copies/, verify byte identity, and write inputs/INPUT_MANIFEST.json covering both
local copies (original path + SHA-256) and public downloads (URL + SHA-256, from download_log.jsonl)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RIBO = Path.home() / "Documents" / "RIBO_ATLAS_FM"
SRC = RIBO / "work/geometry_pilot/biological_validation_pilot_v2/cohort/sources"
ADM = RIBO / "work/geometry_pilot/scientific_pilot_admission"
V2 = RIBO / "work/geometry_pilot/biological_validation_pilot_v2"
REPO = RIBO / "repo/ribo-atlas-fm"

LOCAL = [
    (SRC / "3MXH.pdb", "c-di-GMP holo crystal (Analysis A primary)"),
    (SRC / "1Y26.pdb", "adenine riboswitch holo crystal (A extension; B cohort)"),
    (SRC / "3P49.pdb", "glycine riboswitch holo crystal (A extension)"),
    (SRC / "2GDI.pdb", "TPP riboswitch holo crystal (A extension; B cohort)"),
    (SRC / "4QLM.pdb", "ydaO riboswitch holo crystal (A extension; B cohort)"),
    (SRC / "1GID.pdb", "P4-P6 crystal (B cohort)"),
    (SRC / "CIDGMP_SHP_0001.rdat", "c-di-GMP SHAPE apo/holo rows (Kladwang 2011)"),
    (SRC / "CIDGMP_SHP_0001.md", "metadata"),
    (SRC / "ADDRSW_SHP_0001.rdat", "adenine SHAPE apo/holo rows (Kladwang 2011)"),
    (SRC / "ADDRSW_SHP_0001.md", "metadata"),
    (SRC / "GLYCFN_SHP_0001.rdat", "glycine SHAPE apo/holo rows (Kladwang 2011)"),
    (SRC / "GLYCFN_SHP_0001.md", "metadata"),
    (SRC / "TPPRSW_DMS_0000.rdat", "TPP M2 DMS, no ligand (WT row used)"),
    (SRC / "TPPRSW_DMS_0001.rdat", "TPP M2 DMS, 10 mM TPP (WT row used)"),
    (SRC / "TPPRSW_DMS_0001.md", "metadata"),
    (SRC / "YDAORS_DMS_0000.rdat", "ydaO M2 DMS, no ligand (WT row used)"),
    (SRC / "YDAORS_DMS_0001.rdat", "ydaO M2 DMS, 10 uM c-di-AMP (WT row used)"),
    (SRC / "YDAORS_DMS_0001.md", "metadata"),
    (ADM / "CBLAPO_DCP_0000.rdat", "AdoCbl DCP apo (1M7, DMS)"),
    (ADM / "CBLAPO_DCP_0000.md", "metadata"),
    (ADM / "CBLHLO_DCP_0000.rdat", "AdoCbl DCP holo, 70 uM AdoCbl (1M7, DMS)"),
    (ADM / "CBLHLO_DCP_0000.md", "metadata"),
    (ADM / "ADOCBL_DMS_0000.md", "metadata (AdoCbl M2 no ligand)"),
    (ADM / "ADOCBL_DMS_0001.md", "metadata (AdoCbl M2 with ligand)"),
    (ADM / "RNAPZ6_MCA_0004.md", "metadata only: A124U/A125U mutant MOHCA (2D) - data NOT opened (withheld endpoint)"),
    (ADM / "RNAPZ6_MCA_0005.md", "metadata only (mutant MOHCA)"),
    (ADM / "RNAPZ6_MCA_0006.md", "metadata only (mutant MOHCA)"),
    (ADM / "RNAPZ6_MCA_0007.md", "metadata only (mutant MOHCA)"),
    (V2 / "cohort/paired_core_cohort.json", "Wave F v2 frozen cohort (Analysis B)"),
    (V2 / "cohort/frozen_lineage_splits.json", "Wave F v2 frozen lineage splits (Analysis B)"),
    (V2 / "model/forward_head_pilot.py", "Wave F v2 code (imported read-only by Analysis B)"),
    (V2 / "model/frozen_analysis_plan.json", "Wave F v2 frozen plan (reference)"),
    (V2 / "model/results/forward_head_results.json", "Wave F v2 results (reproduction check)"),
    (REPO / "results/waveF_forward_head/forward_head_results.json", "repo copy of Wave F v2 results"),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    dest = ROOT / "inputs" / "local_copies"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for src, role in LOCAL:
        name = src.name if src.parent.name != "waveF_forward_head" else "repo_" + src.name
        if src.name == "forward_head_results.json" and "model/results" in str(src):
            name = "v2_" + src.name
        tgt = dest / name
        shutil.copyfile(src, tgt)
        a, b = sha(src), sha(tgt)
        assert a == b
        rows.append({"local_copy": str(tgt.relative_to(ROOT)), "original_path": str(src), "sha256": a,
                     "bytes": tgt.stat().st_size, "role": role})
    downloads = []
    log = ROOT / "inputs" / "download_log.jsonl"
    for line in log.read_text().splitlines():
        r = json.loads(line)
        p = ROOT / r["path"]
        r["sha256_recheck_ok"] = sha(p) == r["sha256"]
        downloads.append(r)
    manifest = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "note": "Local copies are byte-identical to read-only originals elsewhere in the RIBO tree (no original was modified). "
                        "Public downloads: RMDB detail pages (rmdb.stanford.edu), RDATs via RMDB's own download links "
                        "(GitHub release assets of DasLab/rmdb.github.io linked from each RMDB entry), PDB files from files.rcsb.org.",
                "local_inputs": rows, "public_downloads": downloads}
    out = ROOT / "inputs" / "INPUT_MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(len(rows), "local copies;", len(downloads), "downloads; all rechecked:", all(d["sha256_recheck_ok"] for d in downloads))


if __name__ == "__main__":
    main()
