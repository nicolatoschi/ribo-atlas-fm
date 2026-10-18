# RIBO-ATLAS-FM feasibility pilot (18 October 2026): results summary

This pilot ran on CPU only, on the Mac (Python 3.13.13, NumPy 2.5.0, SciPy 1.18.0; Matplotlib 3.11.1 with Pillow 12.2.0 in `.venv`, which was used only for the figure). No HPC job was run, no SSH, no git push and nothing was published. Every file is in this folder, and no file elsewhere in the RIBO tree was modified (see `inputs/INPUT_MANIFEST.json`). Both analyses were pre-registered: each plan JSON was hashed and logged in `plans/FREEZE_LOG.txt` before the outcome was computed.

| Plan | SHA-256 | Frozen (UTC) |
|---|---|---|
| `plans/analysisA_plan.json` | `010e2f22cb8a376f7c82903b09d516be42706f70d827d93a46c115a5f2c8cd48` | 2026-10-17 22:59:19 |
| `plans/analysisB_plan.json` | `29712078efd5f5bdc4b4dd96e62675add37b83fac76368fff01e72d891e2899c` | 2026-10-17 23:07:33 |

Figure: `figures/feasibility_pilot_figure.png` (3.5 in wide; panel A = Analysis A, panel B = Analysis B).

---

## Analysis A (primary): where does the ligand response fall?

### What was tested

**Data.** We used public RMDB apo/holo chemical-mapping pairs. A pair was admitted only if the two profiles had the identical construct sequence, SEQPOS/offset, modifier, buffer, Mg²⁺ and temperature, and differed only in the ligand annotation. This was checked from the RDAT headers and is recorded per pair in the plan. The primary pairs are the two named in the application, both from the Deep Chemical Profiling (DCP) deposits. Each apo/holo pair in that series shares one experiment identifier. The modifier is 1M7, in 50 mM HEPES pH 8.0, 10 mM MgCl₂, 24 °C:

* **c-di-GMP-I riboswitch (V. cholerae Vc2):** CDGAPO/CDGHLO_DCP_0000, 0 vs 10 µM c-di-GMP. Holo crystal 3MXH, chain R, ligand C2E.
* **AdoCbl riboswitch (S. thermophilum):** CBLAPO/CBLHLO_DCP_0000, 0 vs 70 µM AdoCbl. Holo crystal 4GXY, chain A, ligand B1Z 301.

**Classes, defined from the holo crystal before any change was computed.** No reactivity value was read when these were built (`results/analysisA_classes.json`).

* **L (ligand contact):** any heavy atom within 4.0 Å of a ligand heavy atom.
* **T (tertiary contact):** not L, and a heavy-atom contact of 4.0 Å or less with a *non-local* residue. A residue is local if it is any of the following:
  * a sequence neighbour within 3;
  * its nested canonical partner;
  * in the same nested helix;
  * in the same hairpin or internal/bulge loop unit.

  Canonical pairs and helices come from the crystal geometry alone. Pairs need H-bonds of 3.8 Å or less, a C1′–C1′ distance of 9–12 Å and a base-plane angle of 60° or less. The nested set is the maximum non-crossing helix subset. Pseudoknot or kissing pairs and junction contacts therefore count as tertiary. No chemical-mapping-derived secondary structure enters the classes.
* **R:** the rest.

Crystal residues are mapped to the construct by a single numbering offset with identity checks. Engineered U1A crystallisation loops are excluded and never count as contact partners.

**Statistic.** We normalised each profile by the 2–8% rule over its scored positions and took r = |holo − apo|. The test statistic is the AUC (Mann–Whitney) of r for L+T versus R. The p-value is a one-sided exact circular-shift null: the class labels are rotated over all n shifts, which preserves the autocorrelation of both the response and the class runs. Sensitivities were a block-permutation null (blocks of 5), a moving-block bootstrap 95% interval, as-deposited values, and an error-scaled response using the RMDB REACTIVITY_ERROR.

**Multiplicity.** Holm correction over the 2 primary tests. Benjamini–Hochberg over the 17 replication and extension pairs. One pre-registered pooled endpoint: the mean AUC − 0.5 over the five DCP 1M7 riboswitch pairs, with independent circular shifts per system and 100,000 draws.

**Other pairs.** Further RMDB pairs that met the same matching rule were added and named in the plan:

* c-di-GMP: DCP DMS, and Kladwang 2011 SHAPE (CIDGMP_SHP_0001, same RDAT, 14–15 replicates).
* AdoCbl: DCP DMS; RNA-Puzzle 6 1M7, DMS and CMCT (RNAPZ6_STD_0001, same RDAT); M2-seq DMS wild-type rows (ADOCBL_DMS_0000/0001, cacodylate).
* Extension riboswitches: adenine (1Y26; DCP 1M7 and DMS, K2011 SHAPE), glycine (3P49; DCP 1M7 and DMS, K2011 SHAPE), FMN (3F2Q; DCP 1M7 and DMS), TPP (2GDI; M2 DMS) and ydaO (4QLM; M2 DMS).

Holo-only deposits were excluded because they have no matched apo row: CIDGMP/ADDRSW/GLYCFN DMS and CMC entries, and the 37 °C 1M7 series.

### Results

| Pair | n (L/T/R) | AUC L+T vs R [95% block bootstrap] | circular p | block-perm p | AUC L only (p) | AUC T only (p) |
|---|---|---|---|---|---|---|
| **c-di-GMP DCP 1M7 (primary)** | 79 (13/32/34) | **0.665 [0.55, 0.81]** | **0.013** (= 1/79, the minimum) | 0.013 | 0.704 (0.063) | 0.649 (0.038) |
| **AdoCbl DCP 1M7 (primary)** | 154 (15/46/93) | **0.480 [0.35, 0.59]** | 0.64 | 0.61 | 0.499 (0.55) | 0.474 (0.68) |
| c-di-GMP DCP DMS | 43 (9/18/16) | 0.808 [0.68, 0.92] | 0.023 | 0.0005 | 0.806 | 0.809 |
| c-di-GMP K2011 SHAPE | 79 | 0.629 [0.49, 0.76] | 0.025 | 0.052 | 0.708 | 0.597 |
| AdoCbl DCP DMS | 72 | 0.643 [0.53, 0.77] | 0.083 | 0.013 | 0.738 | 0.614 |
| AdoCbl PZ6 1M7 | 158 | 0.598 [0.47, 0.71] | 0.146 | 0.063 | 0.576 | 0.605 |
| AdoCbl PZ6 DMS | 74 | 0.769 [0.68, 0.85] | 0.027 | 0.0001 | 0.831 | 0.750 |
| AdoCbl PZ6 CMCT | 84 | 0.432 [0.24, 0.62] | 0.82 | 0.80 | 0.224 | 0.505 |
| AdoCbl M2 DMS (cacodylate) | 72 | 0.569 [0.45, 0.69] | 0.17 | 0.15 | 0.485 | 0.594 |
| adenine DCP 1M7 / DMS / K2011 SHAPE | 71 / 32 / 71 | 0.741 / 0.727 / 0.754 | 0.042 / 0.031 / 0.028 | 0.003 / 0.026 / 0.0007 | 0.92 / 0.88 / 0.95 | 0.67 / 0.68 / 0.68 |
| glycine DCP 1M7 / DMS / K2011 SHAPE | 155 / 85 / 155 | 0.644 / 0.721 / 0.647 | 0.0065 / 0.012 / 0.013 | 0.007 / 0.002 / 0.005 | 0.72 / 0.73 / 0.69 | 0.63 / 0.72 / 0.64 |
| FMN DCP 1M7 / DMS | 101 / 47 | 0.463 / 0.735 | 0.67 / 0.043 | 0.72 / 0.004 | 0.69 / 1.00 | 0.41 / 0.70 |
| TPP M2 DMS / ydaO M2 DMS | 37 / 58 | 0.527 / 0.556 | 0.46 / 0.28 | 0.39 / 0.27 | 0.41 / 0.46 | 0.56 / 0.62 |

* **Primary family (Holm).** c-di-GMP: adjusted p = **0.025**, so it is supported under the frozen rule. AdoCbl: adjusted p = **0.64**, so it is null.
* **c-di-GMP robustness.**
  * As-deposited values: AUC 0.720, p = 0.013.
  * Classes from the 3IRW crystal instead of 3MXH: 0.657, p = 0.039.
  * Error-scaled response: 0.612, p = 0.063.
  * 12 of 13 ligand-contact nucleotides lose reactivity on binding.
  * The classes include the known c-di-GMP contacts G20, A47 and C92.
* **AdoCbl sensitivity.** Adding the 18-atom B1Z 302 fragment to the ligand set gives AUC 0.500, still null.
* **Pooled endpoint (pre-registered; five riboswitches, DCP 1M7).** Mean AUC **0.599**, p = **0.0034** (circular shifts), Fisher χ² = 26.8, df = 10, p ≈ 0.003. This pool includes the c-di-GMP primary pair.
* **Secondary pairs.** 15 of 17 have AUC > 0.5. 10 of 17 reach BH q ≤ 0.072; none reach q < 0.05, because a circular-shift p cannot go below 1/n.
* **AdoCbl A124U/A125U mutant: not tested.** RMDB holds no 1D chemical-mapping profiles for the double mutant. Its only public data are MOHCA-seq (RNAPZ6_MCA_0004–0007), which the application reserves as the withheld response endpoint. We read their metadata only and did not open the data. The test that the mutant's pocket response is smaller is therefore not possible with public 1D data and is stopped here, as the plan specified.
* **Post-hoc and descriptive, not pre-registered** (`results/analysisA_posthoc_response_magnitude.json`). The AdoCbl primary pair shows almost no global ligand response: median |Δ| is 0.009 normalised units and the apo–holo Spearman ρ is 0.93. For c-di-GMP the same figures are 0.165 and 0.54. The AdoCbl M2 (ρ = 0.997) and ydaO M2 (ρ = 0.999) pairs are near-identical as well. Their nulls therefore reflect no measurable response to localise, not a response at the wrong nucleotides. We did not establish why. Possibilities include pre-organisation at 10 mM Mg²⁺ and AdoCbl photolysis: only the RNA-Puzzle 6 deposit records a darkened room.

### What Analysis A supports and what it does not

* **Supports.** In matched public data, ligand-induced reactivity changes concentrate at nucleotides that the holo crystal places in ligand or tertiary contact. The contacts were defined before the changes were computed. This holds for the c-di-GMP riboswitch in three profiles: the primary 1M7 profile, the DMS profile from the same deposit, and an independent 2011 SHAPE dataset. It also holds under both class structures (3MXH and 3IRW). It also holds across five riboswitches probed with one protocol. A structure-aware observation model can therefore have a measurable, localised response signal to explain.
* **Does not support.**
  * Any AdoCbl conclusion: the primary 1M7 pair is null and shows almost no response. All three AdoCbl DMS profiles have AUC > 0.5, but only one (PZ6) reaches p < 0.05.
  * The mutant null-control prediction: untested.
  * Any claim about a learned model: Analysis A uses no model.
  * Causality or mechanism.

---

## Analysis B (secondary): base-resolved observation head

### What was tested

The frozen Wave F v2 protocol was reused unchanged. Its code (`forward_head_pilot.py`, SHA-256 `1adceaa9…`) was imported read-only. The cohort was the same: 4 non-viral lineages, 5 profiles, 3 studies and 382 scored nucleotides. Everything else matched too:

* outer leave-one-lineage-out folds, with inner leave-one-training-lineage-out alpha selection;
* the error-weighted primary and the unweighted refit;
* the metrics and the paired 4-lineage percentile bootstrap.

The run first reproduced all four published Wave F arms exactly: 96 lineage metrics, maximum |difference| = 0.0, identical alphas.

**Change.** The C4′ block was replaced by 8 base-resolved descriptors. They were computed from the whole deposited crystal chain plus the assay ligand (`results/analysisB_descriptors.json`, frozen before fitting):

* SASA of the DMS-reactive atom (A N1 or C N3);
* O2′ SASA;
* base heavy-atom SASA;
* a canonical-pair indicator;
* the base–base H-bond partner count;
* stacking on the 5′ neighbour, on the 3′ neighbour, and on any non-neighbour.

SASA used Shrake–Rupley with a 1.4 Å probe, checked against analytic values. The control was the same within-profile row permutation as the Wave F shuffled-geometry arm.

**Calibration added before the freeze.** The plumbing test used seeded random targets. On those, the inherited Wave F gate was met by chance, for both the base and the C4′ arms. The plan therefore also requires a calibration: 500 independent shuffles of the base block, each refitted with the full nested procedure.

### Results (equal-lineage held-out MSE; lower is better)

| Arm | error-weighted (primary) | unweighted refit |
|---|---|---|
| sequence | 0.0923 | 0.2196 |
| secondary (annotation-informed) | 0.0859 | 0.2151 |
| C4′ geometry (Wave F) | 0.0889 | 0.2197 |
| shuffled C4′ | 0.0881 | 0.2135 |
| **base-resolved** | **0.0853** | **0.2111** |
| shuffled base-resolved | 0.0897 | 0.2207 |

Comparisons are reference minus base, so positive favours base. Intervals are the Wave F paired lineage bootstrap.

* **Secondary → base.**
  * Error-weighted: +0.0006 (+0.7%), [−0.0014, +0.0025], 2 of 4 lineages favour base.
  * Unweighted refit: +0.0039 (+1.8%), [−0.0031, +0.0133], 2 of 4.
* **Shuffled base → base.**
  * Error-weighted: +0.0044 (+4.9%), [−0.0023, +0.0153], 2 of 4.
  * Unweighted refit: +0.0095 (+4.3%), [−0.0041, +0.0232], 2 of 4.
* **C4′ → base (descriptive).** +0.0035 (+4.0%), [−0.0004, +0.0092], 3 of 4; unweighted +0.0085 (+3.9%), 3 of 4.
* **Frozen gate: not met.** 0 of 4 criteria pass.
* **Pre-registered shuffle calibration (500 shuffles).**
  * Error-weighted: p = 0.14 (0.0853 vs a null mean of 0.0907).
  * Unweighted: p = 0.052 (0.2111 vs 0.2389).
  * C4′ for reference: p = 0.68 and 0.85.
* **Descriptive patterns.** The chemistry-stratified view is a Wave F secondary endpoint. Error fell for the two SHAPE profiles: P4-P6 from 0.160 to 0.150, and adenine from 0.0337 to 0.0310. It did not fall for the three DMS profiles: P4-P6 0.381 to 0.386, TPP 0.0169 to 0.0172, ydaO 0.0227 to 0.0251. The study holdout also showed no advantage over secondary.
* **Post-hoc held-out correlation** (`results/analysisB_posthoc_correlation_calibration.json`). Mean within-profile Pearson r is a pre-specified Wave F secondary endpoint, but this calibration of it was not in plan B.
  * Base-resolved: r = **0.53**, against 0.42 for secondary, 0.40 for C4′ and 0.41 for the single shuffle.
  * Base beats secondary in **all 4 lineages**:

    | Lineage | secondary r | base r |
    |---|---|---|
    | P4-P6 | 0.41 | 0.47 |
    | TPP | 0.24 | 0.37 |
    | adenine | 0.54 | 0.72 |
    | ydaO | 0.50 | 0.56 |

  * Against the same 500 shuffles the observed r exceeds every one: null mean 0.38, 95% range 0.33–0.43, **p = 0.002**, which is the minimum. The unweighted refit gives the same (0.52 vs 0.37, p = 0.002).
  * C4′ does not: p = 0.22 and 0.13.

### What Analysis B supports and what it does not

* **Does not support** a claim that the base-resolved head beats the secondary-structure baseline on held-out error. The frozen gate failed, the MSE gain is 0.7–1.8% with 2 of 4 lineages, and the pre-registered calibration p is 0.14 and 0.05.
* **Supports, as feasibility only.**
  * Base-resolved descriptors carry held-out pattern information that C4′ geometry did not. The ranking/correlation improved in all four held-out lineages and exceeded every shuffle, but this is post hoc.
  * The descriptor pipeline runs end to end inside the frozen protocol.
  * The remaining gap is in absolute scale across lineages and in DMS. The cohort applies no target normalisation by design.

---

## Caveats

1. **Crystal vs solution.** The classes and descriptors come from single holo crystal structures, which stand in for solution ensembles. Crystal and assay differ in buffer and ions. Packing differs, and engineered U1A loops are present in 3MXH and 3P49. No apo crystal was used, so the classes describe the bound state. 3P49 is at 3.55 Å and 4GXY at 3.05 Å, and their pair detection is less complete (31/40 and 43/55 deposited pairs recovered).
2. **Small n.**
   * Analysis A has 2 primary systems and 7 riboswitches in total. Profiles from the same riboswitch are not independent. The circular-shift null treats positions within a profile as dependent, which makes the minimum attainable p equal to 1/n (0.006–0.03).
   * Analysis B has 4 lineages. Its bootstrap intervals are coarse, with at most 35 distinct resample means.
3. **Multiple testing.** There are 21 pair-level analyses in A, with Holm on the primaries and BH on the rest. Each pair also has several variants (normalisation, error-scaled, block null, structure), which are reported as robustness, not as extra tests. The Analysis B correlation calibration is post hoc and should be read as hypothesis-generating.
4. **Class breadth and thresholds.** L+T covers 37–64% of scored nucleotides, depending on the pair. The structural thresholds (base pairs 3.8 Å / 60°, stacking) were set on structures alone, before any reactivity was read, and are documented in `scripts/structure_features.py`.
5. **Deposit heterogeneity.**
   * In the c-di-GMP DCP 1M7 pair the holo 2–8% scale is 0.13× the apo scale. The result holds both normalised and as deposited.
   * REACTIVITY_ERROR semantics are not documented, and the M2 rows have no errors.
   * DCP data were collected in 2013 and deposited in 2024.
6. **Disclosures.**
   * The first ~40 deposited values of CIDGMP_SHP_0001 were displayed once during a file-format check before freezing. No difference was computed.
   * The Wave F results for the original four arms were known before plan B; it is their published null.
   * Plumbing tests on random values were run before each freeze (`results/synthetic_test_*.json`).
7. **The secondary baseline is annotation-informed** (Wave F caveat): deposited RDAT structures may derive partly from chemical mapping.

---

## Suggested application text (plain, factual)

> In a pre-registered CPU pilot on public RMDB data, ligand-induced reactivity changes were concentrated at nucleotides that the holo crystal structure places in ligand or tertiary contact, with contact classes fixed before the changes were computed. For the c-di-GMP riboswitch the area under the curve was 0.67 (Holm-adjusted circular-shift p = 0.025; also positive with DMS and in an independent SHAPE dataset), and across five riboswitches probed with one 1M7 protocol the mean was 0.60 (p = 0.003). The AdoCbl pair showed almost no global ligand response and was null (0.48), and the A124U/A125U mutant could not be tested because its only public data are the withheld MOHCA maps. Replacing C4′ geometry with base-resolved descriptors in the frozen four-lineage forward-head protocol did not meet the pre-registered error gate (held-out error within 2% of the secondary-structure baseline); held-out correlation rose in all four lineages (0.53 vs 0.42; post hoc, above all 500 shuffled controls), and calibrating this head on more lineages and testing the response endpoints are in-award work at gate G1.

---

## Files

* **Scripts** (`scripts/`):
  * `fetch_public.py`: downloads, logging URL and SHA-256;
  * `collect_inputs.py`: byte-identical local copies and the manifest;
  * `pdbtools.py`, `rdat.py`, `structure_features.py`: frozen structure definitions;
  * Analysis A: `build_classes_A.py`, `make_plan_A.py`, `analysis_a.py`, `analysis_a_posthoc.py`;
  * Analysis B: `base_descriptors_B.py`, `make_plan_B.py`, `analysis_b.py`, `analysis_b_posthoc.py`;
  * `make_figure.py`, `finalize.py`;
  * inspection helpers: `rdat_header.py`, `summarize_detail.py`.
* **Plans:** `plans/analysisA_plan.json` and `plans/analysisB_plan.json`, each with a `.sha256` file, plus `plans/FREEZE_LOG.txt`.
* **Inputs:**
  * `inputs/INPUT_MANIFEST.json`: 34 local copies with original path and SHA-256, and every public download with URL and SHA-256;
  * `inputs/download_log.jsonl`;
  * `inputs/downloaded/`: RMDB detail pages, RDATs, and the 4GXY, 3F2Q and 3IRW coordinate files;
  * `inputs/local_copies/`.
* **Results:**
  * `results/RESULTS_INDEX.json`: key numbers and hashes;
  * Analysis A: `analysisA_classes.json`, `analysisA_results.json`, `analysisA_per_position.json` (Δ with propagated RMDB error), `analysisA_posthoc_response_magnitude.json`;
  * Analysis B: `analysisB_descriptors.json`, `analysisB_results.json`, `analysisB_held_out_predictions.json`, `analysisB_posthoc_correlation_calibration.json`;
  * `synthetic_test_analysisA.json` and `synthetic_test_analysisB.json`, which are not results.
* **Figure:** `figures/feasibility_pilot_figure.png`.
* **Output hashes:** `OUTPUT_MANIFEST.json`.
