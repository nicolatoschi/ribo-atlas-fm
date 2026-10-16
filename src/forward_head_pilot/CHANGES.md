# Wave F v2: changes from biological_validation_pilot (v1)

Date: 2026-10-15. CPU only (scratch venv, Python 3.13.13, NumPy 2.5.0; v1 contract
checks were recorded under NumPy 2.3.5). No network downloads, no SSH, no GPU.
v1 files were not edited. All 79 v1 files were hashed before work began
(`scratchpad/waveF/v1_hashes_before.txt`) and re-hashed afterwards: unchanged.

Audit references: points 1-7 of the Wave F brief, which correspond to the issues
in `model/independent_benign_pilot_audit.txt` (copied unchanged, sha256 `d3033a85…`)
plus the later review.

## File hashes (SHA-256, before = v1, after = v2)

| File | v1 (before) | v2 (after) |
|---|---|---|
| cohort/paired_core_cohort.json | 05d0f8f9fc3eb27b4d5272548c6a605b36162b0ce21a014524c40011998d52df | 16daa486d622ff039a5b020e3061253f11a42fb1e598ef1f42a810c9dbd2a8b8 |
| cohort/frozen_lineage_splits.json | 51cc0c205d0bc6cce18e1236c528d557ac44dc4e427365899a68829d5af5f16c | 202df0934fbdad803908782176a692d07b703453812f2408a3e6ccc4f1c69b44 |
| cohort/admission_summary.json | 23a27b537db95cdadcc654477ad584596ad420e9bd1d2c9d1f3e3210454393ea | e70b883d7c8e850a32b357cb386ffb5e3593f71ac33f19c749efb3e96b79c7e5 |
| cohort/excluded_primary_candidates.json | 149cba64215f4c0927c108d848a44791eb8ee28f243acfe09f8e40de5f3ef811 | 921879f3ae7beb0d319c328089ed232419b7d5ecdf41528ca672fcb5eb1087d0 |
| cohort/paired_core_observations.csv | 6855ed6807003d84b601b3e546099a745e9f166f40a67fa2697225b3af64a3d8 | cbea0a9dbe7246ecce92e51063f562bc4d65b8a8599e30b2ecf3c3a3b5554561 |
| cohort/prepare_cohort.py → prepare_cohort_v2.py | 61edff28de767005f8fc9be51403b1c1bdef526a35d09ea125da18ef89205d45 | 51611314cb9b2e86e39ba29b24ff1937840e7abb30456fd353b3ebed1563dfbe |
| model/forward_head_pilot.py | ac3c4d85e8aea4cf16f2ef0ea6b52059c58f1c66a3f6cb50b1c2dfcc5880d9b2 | 1adceaa9d89b26d11a07ffeceac7394bb4e181c44ea2a26b73337402077a8ee4 |
| model/test_forward_head_contract.py | f24c05db33c51c506c78a86d096812d16ac0a78337cf520f1ebabe0a8f36bfe3 | 37d33f125bae0063c8818f3a851312906d77434ebf7da1a62933c85da120df9c |
| model/README.txt | 7d854cbd7291698ce9aa623176e74c72e262c9daf435c83359d9b71ab79f4837 | 0b535f033bccf587be4ce345c6296ac00aff108256a8e74b2943ad230e351c20 |
| model/code_contract_checks.json | 23d34eb74b47d6f175f9217a7f275f25abc318af6b4aaca27a396deeb050459b | 759b2500b1a1fb49e41ac79cadc761d1f787d49b9b7d65beb3642afccfb0a914 |

New in v2 (no v1 counterpart):

| File | SHA-256 |
|---|---|
| cohort/verify_extraction_from_raw.py | 186348c021e37070d72ecde141fc8ad11622c78e31d23453ecf08b4440298918 |
| cohort/raw_extraction_verification.json | 87a72846461f36d472f5381083a259775134cbb61c191ab33af8d340fc06febb |
| cohort/organism_label_check.json | a2a01b1b2d2fe05ad41e4d8042d118d6e209b02605ca4ba9d4d091a43a60f90d |
| cohort/verifier_negative_controls.json | 2e495a85e907db7538221c408777b78ebdbe9eb2be45bb0b5828df50f0f7a172 |
| model/frozen_analysis_plan.json | d5dbdc94e518c1b8e2d3d6c8e6edf223578b892d83a374fe268eb75fb3a23b6a |
| model/results/forward_head_results.json | e6b96b27139fb18e6222101b155fc0c66d11f9912f4bb454dbbf4554d90a62c5 |
| model/results_table.txt | 153ff138d2de1b7493076acbc439358fdab5ed2b17f6965bd88f473122f113fb |

Byte-identical copies: `cohort/sources/*` (all 59 files), `benign_cohort/*`
(4 files), `cohort/acquire_sources.py` and the three download manifests,
`model/independent_benign_pilot_audit.txt`. `acquire_sources.py` was copied for
provenance only and was not run. v1's `prepared_forward_head_manifest.json` is
superseded by `model/frozen_analysis_plan.json` and was not carried over.
Unified diffs of every edited script are in `CHANGES_code_diffs.patch`.

## Point 1: non-viral cohort (Zika replaced by ydaO)

- **Change.** `prepare_cohort_v2.py` drops `Zika_xrRNA_DMS_WT` (ZIKAXR_DMS_0000 / 5TPY A)
  and adds `ydaO_DMS_holo_WT`: YDAORS_DMS_0001 profile 1 (mutation:WT, 10 µM c-di-AMP),
  mapped to 4QLM chain A residues 39-123 = RDAT positions 37-121, as in
  `benign_cohort/ydao_source_mapping_audit.json`. `pdb_chain` gains an explicit
  residue window, asserted complete and contiguous. Splits, admission summary,
  observation CSV and the exclusion list were all regenerated.
- **Zika exclusion recorded** in `excluded_primary_candidates.json`. It is a scope
  decision under the non-viral audit, made before any fit: v1 was never fitted
  (`fitted:false`, no v1 results exist). It is not a data-quality or outcome-based
  exclusion. The v1 ydaO exclusion is marked SUPERSEDED. The ligand-absent
  YDAORS_DMS_0000, the mutate-map rows, 4OQU/4P8Z and the 4QLM residues outside
  39-123 are listed with their reasons.
- **Resulting cohort.** 4 lineages, 5 profiles, 3 studies, 382 scored nucleotides:
  adenine 71, P4-P6 SHAPE 158, P4-P6 DMS 78, TPP 36, ydaO 39. The four retained
  profiles are value-identical to v1 (checked by the verifier). The ydaO caveats
  are stored with the example: partial 85-nt domain, truncated packing context,
  high B factors (2.72 Å, core mean ~90 Å²) and no reported errors.
- **New finding that needed a decision.** The deposited ydaO STRUCTURE is internally
  unbalanced (7 `[` vs 6 `]`). Under the pre-specified rule, no repair is made and all
  13 positions of the unbalanced bracket family are set unknown (`x`). That masks
  them in every arm and removes 7 of 46 A/C sites. The rule was fixed from the
  annotation string alone, with no reactivities or fits inspected. For reference
  only, sequence complementarity suggests the deposit intended 96-143…101-138
  with 102 unpaired, and the naive stack rule would give two C·A "pairs". The
  repair was deliberately not applied.

## Point 2: shuffle control and masked marginals

- **Fixed in code, not just reworded.** v1 permuted every structurally valid row and
  only afterwards applied the DMS A/C mask, so its marginal claim failed on the
  scored subset. In v2, `feature_matrices` permutes within
  `structural_valid & chemical_eligibility`. Eligibility is A/C for DMS and all
  positions for SHAPE, and it comes from sequence and chemistry only. Features
  therefore still never depend on observations, and the existing leakage test is
  extended to DMS. `prepared_rows` rejects any observation at a chemically
  ineligible position.
- At freeze, `shuffle_marginal_audit` checks per profile that the geometry block
  of the scored rows is a row permutation of the shuffled block. Freeze refuses to
  proceed if it is not. All 5 profiles pass, with 92-100% of rows changed. Every
  eligible row is observed in this cohort, so preservation is exact. The plan
  states the claim in exactly these terms.
- New test: the A/C-subset marginal is preserved, and the v1 all-row scheme
  detectably fails the same check.

## Point 3: REACTIVITY_ERROR semantics, the 0.1 floor, and the unweighted sensitivity

- The plan, the cohort `protocol.reported_error_semantics` and the README now record:
  REACTIVITY_ERROR is used as deposited, in reactivity units. Whether it is an SD
  or an SEM is not stated in any local file, so its semantics are unknown. The
  0.1 floor is arbitrary, not a calibration. Most reported errors fall at or below
  it: 67/71 adenine, 154/158 P4-P6 SHAPE, 69/78 scored P4-P6 DMS. There are zero
  zero-valued or sentinel errors. TPP and ydaO report no errors.
- **Sensitivity added.** The whole nested analysis is re-run with unit precision
  (`unweighted` scheme), covering inner alpha selection, training weights and
  evaluation. Both schemes are in the frozen plan and the results, and the frozen
  interpretation gate requires agreement across them.

## Point 4: rebuilding values from raw sources

- **New `cohort/verify_extraction_from_raw.py`.** It uses independent parsers
  (different RDAT tokeniser, SEQPOS-label-driven mapping, separate PDB C4′ reader,
  separate pair table). It re-derives sequence, reactivity, error, DMS A/C mask, C4′
  coordinates and restricted secondary for all 5 profiles and asserts exact
  equality. It also checks: the 4 retained profiles against frozen v1, ydaO
  against the earlier mapping-audit payload, raw-file hashes against download-time
  manifests and other local copies, and C4′–C4′ continuity (max 7.19 Å).
  Result: 103/103 passed (`raw_extraction_verification.json`). Of the "other local
  copies", most are the v1 directory itself; the download-time hashes are the
  meaningful external anchor.
- **Negative controls** (`verifier_negative_controls.json`). On scratch copies, a
  1e-9 reactivity change, a 0.001 Å coordinate change and a one-token secondary
  change were each detected.
- **Binding.** `freeze` now requires `--extraction-audit`, checks that the audit
  passed and matches the exact cohort hash, and records the audit's hash. `run`
  aborts if the audit changes afterwards. Both behaviours are tested.

## Point 5: near-identity guard

- The comment claimed a near-identity check that did not exist; it has now been
  implemented. Freeze rejects any cross-lineage pair that meets one of these
  conditions:
  - exactly identical sequences;
  - global Needleman-Wunsch identity ≥ 0.80;
  - Smith-Waterman local identity ≥ 0.90 over ≥ 50% of the shorter sequence.
  Scoring is +1/−1/−2 (match/mismatch/linear gap). All pairwise values are stored
  in the plan. The maximum global identity in this cohort is 0.53 (TPP–ydaO), and
  no pair is flagged. Tests cover exact, single-substitution and contained-subsequence
  near-identity, plus an unrelated pair passing.
- The plan states that this is a guard, not proof of lineage independence.

## Point 6: deposited secondary structure may derive from probing data

- Flagged, and the arm is kept. Wording appears in the cohort
  (`protocol.deposited_secondary_annotation`), the plan's `features` text, the
  results `limitations`, and the README: the secondary arm is an
  annotation-informed baseline whose derivation may have used chemical-mapping
  data. It may therefore be advantaged over a blind sequence-only structure.

## Point 7: organism label for ydaO (and TPP)

- `organism_label_check.json` uses local files only.
- **Verified locally:**
  - The RDAT/RMDB label is "C. subtarraneus" (sic), with no strain or taxid.
  - 4QLM SOURCE is Thermoanaerobacter tengcongensis, taxid 273068, strain MB4.
  - All 125 SEQRES residues occur exactly once in the probed construct.
  - Rfam maps 4QLM A to RF00379 (c-di-AMP / ydaO).
- **Not verifiable locally (no taxonomy dump):** the synonymy itself. Domain
  knowledge says T. tengcongensis was reclassified as Caldanaerobacter subterraneus
  subsp. tengcongensis (Fardeau et al. 2004).
- **Status:** likely a taxonomic synonym; sequence identity verified, nomenclature not.
- The caveat does not apply to TPP: its RDAT says E. coli, and 2GDI is a synthetic
  transcript.

## Other code changes (all pre-specified before the run)

- `VERSION` bumped to `forward-head-lineage-ridge-v2`.
- A sequence-vs-geometry comparison was added, so the frozen set is: sequence→geometry,
  secondary→geometry, shuffled→geometry, and sequence→secondary.
- Comparisons take a `metric` argument.
- The plan records chemistry-by-lineage, study-by-lineage and lineages-per-chemistry
  counts.
- Study holdouts flag chemistry extrapolation. Holding out Kladwang2011 leaves
  DMS-only training against SHAPE test profiles.
- A frozen interpretation gate was added (see README).
- Contract tests went from 11 to 17 checks and all pass (`code_contract_checks.json`,
  code sha `1adceaa9…`).

## Order of operations

1. `prepare_cohort_v2.py` built the cohort (sha `16daa486…`).
2. `verify_extraction_from_raw.py` passed 103/103, plus the negative controls.
3. `test_forward_head_contract.py` passed 17/17.
4. `freeze` produced plan sha `d5dbdc94e518c1b8e2d3d6c8e6edf223578b892d83a374fe268eb75fb3a23b6a`.
5. `run` executed once against that plan (`results/execution_provenance.json`).

The README and this file were written after the run. They are not hash-bound and
they changed no analysis choice.

## Outcome (descriptive; see model/results_table.txt)

The frozen gate is **not met** in either weighting scheme. Geometry does not beat
the secondary arm or the shuffled-geometry control. Against sequence it is
essentially tied: +0.0035 error-weighted (3/4 lineages, interval spans 0) and
0.0000 unweighted (2/4). The pilot result is null for an added value of crystal
C4′ geometry. Any correlation gain over sequence comes from the
(annotation-informed) secondary features, and even that is not consistent across
lineages: ydaO goes the other way.
