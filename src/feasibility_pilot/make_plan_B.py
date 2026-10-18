#!/usr/bin/env python3
"""Write the Analysis B pre-registration (plans/analysisB_plan.json). Reads no reactivity values."""
import datetime as dt, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
desc = json.loads((ROOT / "results/analysisB_descriptors.json").read_text())
files = ["scripts/analysis_b.py", "scripts/base_descriptors_B.py", "scripts/structure_features.py", "scripts/pdbtools.py",
         "scripts/make_plan_B.py", "results/analysisB_descriptors.json", "inputs/local_copies/paired_core_cohort.json",
         "inputs/local_copies/frozen_lineage_splits.json", "inputs/local_copies/forward_head_pilot.py",
         "inputs/local_copies/frozen_analysis_plan.json", "inputs/local_copies/v2_forward_head_results.json",
         "inputs/local_copies/1Y26.pdb", "inputs/local_copies/1GID.pdb", "inputs/local_copies/2GDI.pdb", "inputs/local_copies/4QLM.pdb"]
plan = {
 "schema": "feasibility-analysisB-plan-v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
 "status": "FROZEN before any base-resolved arm was fitted",
 "title": "Base-resolved observation head on the frozen Wave F v2 four-lineage cohort",
 "cohort": {"path": "inputs/local_copies/paired_core_cohort.json", "sha256_must_equal": "16daa486d622ff039a5b020e3061253f11a42fb1e598ef1f42a810c9dbd2a8b8",
            "content": "4 non-viral lineages, 5 profiles, 382 scored nucleotides (adenine SHAPE 71, P4-P6 SHAPE 158, P4-P6 DMS 78, TPP DMS 36, ydaO DMS 39)"},
 "protocol_reused_unchanged": "forward_head_pilot.py v2 (sha256 1adceaa9...) imported read-only: same features for the sequence and secondary arms, same scored rows (observed & C4'-valid & secondary-valid), same chemistry-specific slope blocks, same precision weights (error floor 0.1; unweighted refit), same outer leave-one-lineage-out and inner leave-one-training-lineage-out alpha selection over [1e-4..10], same equal-lineage metrics, same paired lineage percentile bootstrap (10,000, seed 20261014) and the same study-holdout sensitivity.",
 "arms": {"sequence_context": "unchanged", "secondary_context": "unchanged", "geometry_context": "unchanged C4' arm (reproduction reference)",
          "shuffled_geometry_context": "unchanged", "base_context": "sequence + secondary + base-resolved block (C4' block replaced)",
          "shuffled_base_context": "base-resolved block rows permuted within profile among chemically eligible scored positions using exactly the Wave F v2 permutation (seed 20261014 + first 32 bits of SHA-256(example_id))"},
 "base_resolved_descriptors": {"features": desc["features"], "scales": desc["scales"], "definition": desc["definition"],
                               "ligands_in_sasa": desc["ligands_in_sasa"],
                               "structure_context": "whole deposited RNA chain (not only the probed core) plus assay ligand heavy atoms; no waters, ions, other chains or symmetry mates",
                               "file": "results/analysisB_descriptors.json"},
 "comparisons": "Wave F v2 four comparisons plus secondary->base, shuffled_base->base, sequence->base, geometry(C4')->base; reference minus candidate MSE per held-out lineage (positive favours base), for error_weighted (error-weighted and unweighted MSE) and the unweighted refit",
 "interval": "Descriptive paired percentile 95% bootstrap over the 4 lineages (identical to Wave F v2); no p-values",
 "interpretation_gate": "Base-resolved geometry is described as adding held-out information only if, in BOTH the error-weighted primary and the unweighted refit, the mean lineage difference favours base_context over secondary_context AND over shuffled_base_context, with at least 3 of 4 held-out lineages favouring base_context in each of those four comparisons. Otherwise the result is reported as null or mixed. Base-versus-sequence and base-versus-C4' are descriptive.",
 "shuffle_calibration": {"n_permutations": 500,
   "rationale": "In the pre-freeze plumbing test on seeded random targets the inherited Wave F gate was met by chance (base and C4' arms both beat secondary on 4/4 lineages by ~1% MSE), so the single seeded shuffle is not a calibrated control. A pre-registered calibration therefore refits 500 independent within-profile shuffles of the base block (and, for reference, of the C4' block) with the full nested procedure; one-sided permutation p = (1 + #{shuffled equal-lineage MSE <= observed}) / 501, per weighting scheme.",
   "role": "Base-resolved geometry is described as carrying information beyond its shuffled marginals only if the frozen gate is met AND the calibration p < 0.05 in both schemes. The gate alone is reported but is not sufficient."},
 "reproduction_requirement": "The run must reproduce the published Wave F v2 lineage metrics and selected alphas for the four original arms (|difference| < 1e-10) or it stops without reporting the base arm.",
 "disclosures": ["Descriptor thresholds are shared with Analysis A and were fixed on crystal structures only.",
                 "Wave F v2 results for the original four arms were known before this plan (they are the published null). No base-resolved arm had been fitted.",
                 "A plumbing test on seeded random targets was run before freezing (results/synthetic_test_analysisB.json); it contains no real data. On those random targets the inherited Wave F gate was met (base and C4' arms both beat secondary and their single shuffle), while the 500-shuffle calibration gave p = 0.08-0.14; this motivated making the calibration a required part of the interpretation."],
 "frozen_files_sha256": {f: sha(ROOT / f) for f in files},
}
assert plan["frozen_files_sha256"]["inputs/local_copies/paired_core_cohort.json"] == plan["cohort"]["sha256_must_equal"]
out = ROOT / "plans/analysisB_plan.json"
out.write_text(json.dumps(plan, indent=1) + "\n")
print("plan written", sha(out))
