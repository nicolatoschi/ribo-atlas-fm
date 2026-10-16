LEARNED STRUCTURE-TO-PROBE MEASUREMENT-OPERATOR PILOT (Wave F, v2)

What changed from v1 is listed, point by point against the audit, in ../CHANGES.md.

Scope
This CPU pilot tests whether local tertiary-coordinate descriptors add held-out
probe-prediction information beyond sequence, observable assay context and
supplied secondary structure. It does not train the RNAbpFlow inverse model,
predict blind structures, establish causal condition responses, or use a
previously random systems-test head as biological evidence.

v2 cohort (non-viral only, per independent_benign_pilot_audit.txt): adenine
riboswitch SHAPE (1Y26 X), P4-P6 SHAPE + DMS (1GID A), TPP riboswitch DMS (2GDI X),
ydaO c-di-AMP riboswitch DMS (YDAORS_DMS_0001 profile 1 on 4QLM A residues 39-123).
Zika xrRNA was removed for scope, before any fit (see
../cohort/excluded_primary_candidates.json). 382 scored nucleotides, 5 profiles,
4 lineages, 3 studies.

The admitted crystal-to-solution source set is a structural PROXY. Construct
mapping alone does not establish identical solution conditions or conformational
populations. Four lineages are an exploratory readiness test, not a powered
efficacy study; profiles, residues and crystal copies cannot inflate that count.

Models and endpoints
All arms use a fixed five-position nucleotide window, chemistry and recorded
numeric assay covariates with missing-value indicators. The secondary arm adds
supplied base-pair descriptors. The geometry arm adds C4-prime local distances,
packing and angle invariants. Fixed DMS- and SHAPE-specific slope blocks allow
chemistry-specific effects. No observations, errors, observed masks, profile,
study or lineage accession enter the predictor. These are bounded linear ridge
models, not large pretrained neural heads.

Secondary-structure caveat (audit point 6). The secondary arm uses the
source-deposited RDAT STRUCTURE annotation restricted to each core. Those
annotations may themselves have been derived or refined using chemical-mapping
data from the same laboratory or experiment class; their derivation is not
documented in the local files. The fit never learns them from y, but the secondary
arm must be read as an ANNOTATION-INFORMED baseline that may carry target-derived
information, not as a blind sequence-only structure. This can favour the
secondary arm relative to geometry. The arm is retained, not dropped. Where a
deposited bracket family is unbalanced (ydaO pseudoknot: 7 '[' vs 6 ']'), all
positions of that family are set unknown ('x') rather than repaired; unknown
positions are masked in every arm.

Shuffled-geometry control (audit point 2). A fourth arm permutes geometry rows
within each profile, restricted to positions that are chemically eligible
(DMS: A/C; SHAPE: all; decided from sequence and chemistry, never observations)
and have known geometry and secondary annotation. Because every scored row lies in
that set, the geometry-feature marginal over the scored rows is preserved; freeze
verifies this per profile and refuses to proceed otherwise (all five profiles
pass; every eligible row is observed in this cohort). The control exactly
matches the geometry arm's coefficient count, alpha grid and selection procedure.
It is a single seeded diagnostic negative control, not a randomisation p-value.
(v1 permuted all structurally valid rows before the A/C mask, so its stated
marginal claim did not hold on the scored DMS subset.)

Folds. Outer folds hold out an entire lineage, including all chemistry/replicate
profiles. Inner folds hold out a training lineage to select a penalty from the
same six-value grid in every arm. Features and intercept are standardized/fitted
only on the outer/inner training rows. Targets are never normalized against a
held-out profile.

Lineage-independence guard (audit point 5). Freeze rejects any cross-lineage pair
that is exactly identical, has global Needleman-Wunsch identity >= 0.80, or a
Smith-Waterman local alignment with identity >= 0.90 spanning >= 50% of the
shorter sequence (match +1, mismatch -1, gap -2). All pairwise values are stored
in the plan (max global identity in this cohort 0.53). This is a guard, not proof
of independence: lineage labels remain a scientific-curation responsibility.

Weights and errors (audit point 3). REACTIVITY_ERROR is used as deposited, in
deposited reactivity units. The local RDAT/RMDB files do not state whether it is
a standard deviation across replicates or a standard error of the mean, so its
semantics are UNKNOWN. The error floor of 0.1 reactivity units is an arbitrary
pre-specified constant, not a calibration; most reported errors in this cohort
are at or below it (67/71 adenine, 154/158 P4-P6 SHAPE, 69/78 scored P4-P6 DMS),
so error weighting mainly down-weights a minority of high-error nucleotides.
TPP and ydaO report no errors and get uniform weight within profile. No zero or
sentinel errors occur. Profiles have equal weight within lineage; lineages have
equal total weight. The primary estimate is the equal-lineage error-weighted MSE
(error_weighted scheme). The COMPLETE nested analysis is repeated with unit
precision everywhere (unweighted scheme) as a pre-specified sensitivity.

Interpretation gate (frozen before the run). Geometry is described as adding
held-out information only if, in BOTH weighting schemes, the mean lineage
difference favours geometry over the secondary arm AND over the shuffled-geometry
control, with at least 3 of 4 held-out lineages favouring geometry in each.
Otherwise the result is reported as null or mixed. Intervals resample entire
lineages (4 units) and are descriptive only. No nucleotide-level p-value.

Raw-source extraction audit (audit point 4). ../cohort/verify_extraction_from_raw.py
re-derives every prepared array from the raw local RDAT/PDB bytes with independent
parsers and asserts exact equality (103 checks). Freeze requires this audit to have
passed for the exact cohort hash; run aborts if it changes afterwards.

Organism label (audit point 7). See ../cohort/organism_label_check.json: the ydaO
RDAT says "C. subtarraneus" (sic), 4QLM says Thermoanaerobacter tengcongensis
(taxid 273068, MB4). Likely a taxonomic synonym (T. tengcongensis =
Caldanaerobacter subterraneus subsp. tengcongensis); the synonymy is from domain
knowledge and could not be confirmed from local files, but 125/125 SEQRES identity
with the probed construct is verified locally. The caveat does not apply to TPP
(RDAT: E. coli; 2GDI: synthetic transcript).

Commands (run from this directory, CPU only, NumPy only)
1. python test_forward_head_contract.py
2. python forward_head_pilot.py freeze --cohort ../cohort/paired_core_cohort.json \
       --plan frozen_analysis_plan.json \
       --extraction-audit ../cohort/raw_extraction_verification.json
3. python forward_head_pilot.py run --cohort ../cohort/paired_core_cohort.json \
       --plan frozen_analysis_plan.json --output results

A code or cohort change invalidates the plan. Review and freeze a new version
before running it; never silently patch a previously frozen analysis.

Outputs
results/forward_head_results.json (both weighting schemes, all arms, folds,
selected penalties, per-profile/per-lineage metrics, chemistry strata, study
holdouts, comparisons and the gate outcome), held_out_predictions.json,
fitted_outer_models.json and execution_provenance.json (SHA-256 of plan, cohort,
code and outputs). results_table.txt is a human-readable digest of the results
file. Synthetic toy checks (code_contract_checks.json) are code tests only.
