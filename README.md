# RIBO-ATLAS-FM

This repository is the code and measured evidence behind the EuroHPC AI-for-Science proposal RIBO-ATLAS-FM. The proposal requests 83,400 JUPITER node-hours = 333,600 GPU-hours, from a model total of 83,239.0. It contains:

- the training, restart, evaluation and analysis drivers that ran on JUPITER (GH200, Arm64, PyTorch 2.10, CUDA 12.9, NCCL);
- the data-admission parsers;
- frozen-packet manifests, batch scripts and submission records;
- compact result summaries and a ledger of every RIBO job;
- the resource-plan builder, together with the facts file from which every computed number in the proposal documents is read.

The full raw evidence is in `RIBO_ATLAS_FM_Execution_Pack_v2.zip`, attached to release v2.0. That pack holds every rank report, log and sacct record, and the complete frozen packets.

**What is not included:** trained models, checkpoints and generated structures. Large third-party inputs (RMDB RDAT files, the RNAbpFlow checkpoint, RNA-FM weights) are listed by accession, URL and sha256 in [docs/DATA.md](docs/DATA.md) instead.

## The scientific question

Can chemical probing (SHAPE, DMS, 2A3 reactivities) reveal how an RNA's tertiary structure responds to ligands, ions and mutations, beyond what sequence and probing-informed secondary structure already explain?

Three matched arms address this:

- sequence-derived base pairs;
- probing-guided base pairs;
- direct encoding of reactivities and experimental conditions.

All three share one sequence prior (RNA-FM, 99.5M parameters, frozen) and one geometric generator (RNAbpFlow, an SE(3) flow-matching decoder with 17.2M parameters). Training differentiates through the complete 50-step sampler with a K-sample ensemble observation loss.

## What was measured on JUPITER

All jobs except 2319595 and 2305296–98 use the corrected code; see the anomaly finding below.

| Job (nodes) | What ran | Measured result |
|---|---|---|
| 2339575 (64), Wave B | Same-node A/B gate; N32/K32/B128 updates with restart; length ladder 50–720 nt; microbatch 1–16; 256-GPU weak-scaling point | The update was 16.0× faster with anomaly mode off (110.0 s against 1,756.9 s with it on), with identical gradient norms. A K32 update on 32 nodes took 482.7, 483.4 and 473.7 s (the last resumed), i.e. 120 node-s per case (previously 1,340.7). Cost per case scales as L^0.12: flat to about 400 nt, 1.42× at 720 nt (26 GiB). Microbatch 8 at K8 and 175 nt costs 4.22 node-s per case (12.8 GiB) |
| 2342412 (128), Wave C | Four concurrent 32-node trajectories with cross-node restart; 16 concurrent 8-node microbatch-8 updates; one 512-GPU update | 4/4 cross-node restarts qualified (4 × 384 reports). Under 16-way concurrency: 4.28 (K8) and 17.24 (K32) node-s per case. Weak scaling from 32 to 128 nodes reached 0.81, because of one slow node |
| 2340563 (32), Wave D | Pretrained RNAbpFlow K32 generation, OL3/GBn2 refinement and metrics on 256 planned cases | 248/256 cases completed. 126.9 node-s per K32 case at 177 nt with perfect packing (95% CI 122.5–131.5). The static partition packed at 22%. Refinement failed on 2.9% of samples. P4–P6 anchor: 32/32 PDBs bit-identical to pilot job 2297413 |
| 2340325 (64), Wave E | RNA-FM masked-LM continued pretraining on Rfam 15.1, 8–64 nodes | Strong-scaling efficiency 0.77, 0.31 and 0.19 at 16, 32 and 64 nodes. Held-out-family loss changed by +0.0084 nats (95% CI −0.0010 to +0.0158, 210 families): **no transfer** |
| Wave F (local CPU) | Nested ridge forward-head pilot on 4 non-viral lineages (382 scored nucleotides) | **Null**: C4′ geometry did not beat secondary structure or a shuffled-geometry control under the frozen gate |
| 2319595 (32) | Two persistent AdamW updates plus fresh-process restart | 384/384 reports qualified. Timing invalid (anomaly mode on) |
| 2305296–98 (8–32) | First full-objective 175-nt measurements | 192 reports, numerics qualified. Timing superseded (anomaly mode on) |
| 2297211–16, 2297399–403 | Systems prototype to 192 GPUs; geometry-teacher scaling (execution pack) | Geometry teacher strong-scaling efficiency 0.89–0.91 at 48 nodes (0.907 for the baseline); synthetic targets |

**Probing data.** Sixty hash-verified RMDB DMS/2A3 RDAT files were parsed with the unchanged admission parser: 8 from 12 Oct plus 52 acquired on 16 Oct, 10.9 GB in total. They yield:

- 3,985,378 profiles;
- 1,708,211 unique 177- and 207-nt candidates from 17 libraries;
- 485,824 candidates with at least one clean chemistry, and 300,166 with both.

About 80% of profiles are flagged, almost all with SNR below 1, and are not used for training. After a 15% holdout, the clean pools hold 412,950 and 255,141 cases. These are above the 300,000 broad and 50,000 clean cohorts that the design uses.

**Preliminary compute.** Preliminary RIBO compute was 2,258.24 GPU-hours; the facts file gives 2,258.3, the sum of its rounded components. All of it is recorded in [`results/job_ledger/RIBO_JOB_LEDGER.json`](results/job_ledger/RIBO_JOB_LEDGER.json). That ledger is built from every shipped sacct record and contains 46 job attempts:

- 393.17 GPU-h on FL01-991;
- 1,865.07 GPU-h on FL01-993.

## How the numbers in the proposal are derived

Two deterministic scripts produce the numbers, both using the standard library only.

1. [`resource_plan/build_resource_plan_v2.py`](resource_plan/build_resource_plan_v2.py) reads [`design_v2.json`](resource_plan/design_v2.json) (arms, seeds, passes, cohort sizes, batch, microbatch, inference cohorts, development checks, packing) and the measured Wave B, C and D analyses. It writes [`resource_plan_v2.json`](resource_plan/resource_plan_v2.json).
   - **Training blocks.** Each training block costs trajectories × cases × passes × c ÷ 3,600, where c is the measured node-seconds per case: c8 = 4.29 for the broad K = 8 stage and c32 = 17.26 for the clean K = 32 stage. These are the Wave C microbatch-8 rates, scaled from 175 to 177 nt with the measured exponent 0.12.
   - **Inference.** 126,300 K32 cases (25,500 final cohorts plus 100,800 development-check cases, one check per pass boundary of every trajectory) are priced at 158.6 node-s each. That is the measured 126.9 node-s per case at 80% packing with a dynamic queue. A fixed 1,500 NH for references and preprocessing is added.
   - **Total.** The model total is **83,239.0 node-hours**.
2. [`resource_plan/build_facts_v2.py`](resource_plan/build_facts_v2.py) reads the ledger, the analyses, the data-admission evidence and the rank reports. It writes [`facts_v2.json`](resource_plan/facts_v2.json), which the proposal documents import, rounding each block up to the next 100 node-hours.

| Block | Model NH | Request NH |
|---|---:|---:|
| Matched core: 3 arms × 5 seeds; 3 broad K8 passes over 300,000 cases + up to 8 clean K32 passes over 50,000 (7,032 + 3,125 updates per trajectory) | 44,854.4 | 44,900 |
| Mechanism ablations: 3 × 3 seeds, clean stage | 17,260.6 | 17,300 |
| Locked reproduction: 1 per arm | 8,970.9 | 9,000 |
| Data-scaling study: 10% and 30% × 3 seeds | 3,588.4 | 3,600 |
| Frozen evaluation (inference) | 7,064.7 | 7,100 (8.5%) |
| Shared fitting | 1,000.0 | 1,000 |
| Operations | 500.0 | 500 |
| **Total** | **83,239.0** | **83,400 NH = 333,600 GPU-h** |

Rebuild both from this repository:

```bash
python3 -I resource_plan/build_resource_plan_v2.py --design resource_plan/design_v2.json \
  --waveB results/waveB_job2339575/waveB_analysis.json --waveC results/waveC_job2342412/waveC_analysis.json \
  --waveD results/waveD_job2340563/waveD_findings.json --out /tmp/resource_plan_v2.json
cmp /tmp/resource_plan_v2.json resource_plan/resource_plan_v2.json          # byte-identical
python3 -I resource_plan/rebuild_facts.py --plan /tmp/resource_plan_v2.json --out /tmp/facts_v2.json \
  --compare resource_plan/facts_v2.json                                       # all values identical
```

`build_facts_v2.py` is shipped unchanged, so it still hard-codes the build machine's project root. `rebuild_facts.py` imports it and remaps only its paths. The facts builder also globs 1,508 rank reports and two `steps.tsv` files, which are shipped under `results/rank_reports/`. It uses them for host-memory, gradient-spike and slow-node facts.

Some numbers are not computed by the builders: a few constants typed inside `build_facts_v2.py`, and the data, model and hardware numbers that the proposal text states directly. [`tools/check_number_sources.py`](tools/check_number_sources.py) recomputes each of them from the execution pack (66 checks), and [docs/NUMBER_SOURCES.md](docs/NUMBER_SOURCES.md) maps every hand-typed number to its source or class. [docs/REPRODUCE.md](docs/REPRODUCE.md) explains how to rerun each wave.

## The anomaly-detection finding

RNAbpFlow's `src/models/flow_module.py` calls `torch.autograd.set_detect_anomaly(True)` at import. This PyTorch debugging mode records a Python stack trace for every forward operation and NaN-checks every backward output.

Our 157-nt driver switched the mode off; the batched and reliability drivers used for every 175-nt measurement (2305296–98, 2319595) did not. As a result, every full-objective update ran about 10–16× slower. Only the gradient phases slowed; the gradient norms were identical.

The fix is one statement after model construction, shown in [docs/anomaly_fix.diff](docs/anomaly_fix.diff):

```python
model=Workload().to(device);torch.autograd.set_detect_anomaly(False);assert not torch.is_anomaly_enabled()
```

`Workload` and the update body are otherwise AST-identical to the qualified packets. On one GH200 node with identical inputs (gate B0 in job 2339575), the update took 110.0 s with the mode off against 1,756.9 s with it on.

The cost diagnosis is in [docs/DIAGNOSIS.md](docs/DIAGNOSIS.md).

## Limits of this evidence

- **Cost runs.** They use one physical RNA (5zwm_B, 175 nt) as 128 synthetic stochastic instances, with synthetic observations and a random frozen head. They measure cost, scaling, memory and restart, not learning. The length-ladder cases are RNAbpFlow training structures.
- **Wave D.** This is pipeline QA, not an accuracy claim: every case is in RNAbpFlow's training metadata, and the bpRNA entries are distillation targets. Eight cases are missing because one node (jpbo-090-25) hit a CUDA out-of-memory error at start-up. The 80% packing used for pricing is an assumption of production scheduling; the measured static partition reached 22%. Generation reproduces the pilot bit for bit, but refinement is not bit-reproducible. `results/waveD_job2340563/SUPERSEDED_FIELDS_NOTE.json` marks the findings fields that refer to an earlier internal costing.
- **Wave E.** Null for transfer. RNA-FM was pretrained on RNAcentral, which very likely contains these sequences, so the plan keeps the prior frozen.
- **Wave F.** Null, on four lineages, with crystals as proxies for solution states, one backbone atom per nucleotide and a possibly probe-informed secondary baseline. The plan responds with base-resolved descriptors and a G1 skill gate that must pass before inverse training.
- **Gradient spikes.** In the 512-GPU update, 3 of 512 cases had local gradient norms above 10³ (median 0.086). Production uses per-microbatch clipping.
- **Node screening.** The unit costs come from healthy, screened nodes. Five nodes were excluded: jpbo-048-33, jpbo-049-19 and jpbo-061-16 were 23–35% slower than the median node (facts `scaling.slow_node_excess`), while jpbo-055-37 and jpbo-090-25 were faulty. The ledger contains no straggler contingency.
- **Data counts.** The data-admission counts are parse results. Family, lineage and study holdouts are applied at G1. The 15% holdout is the design's planning fraction. The 16 Oct parser writes a fixed header that still describes the 8-file run. `results/data_expansion/SCOPE_NOTE.json` records the actual scope: 60 files, 17 groups, VIRWIN included.
- **Frozen files.** They keep the paths they ran with: JUPITER site paths (`/e/project1/...`) and, in some records and in two builder scripts, the build machine's absolute paths. They are byte-identical because their sha256 values are recorded in packet manifests and submission records. The wrappers here only remap paths.

## Layout

| Path | Contents |
|---|---|
| `src/training/` | Anomaly-off batched full-trajectory driver and core modules; production reliability driver with `reliability_state.py`; native and site-limit launchers; `rank_launch.py`; `step_launch.py` (concurrent-step-safe); teacher-fixture and packet builders; `RNAbpFlow_training/` (code and config); packet manifests |
| `src/evaluation/` | Wave D packet code: generation, refinement, metrics, aggregation, packet builder, and the inference copy of RNAbpFlow |
| `src/sequence_prior/` | Wave E RNA-FM continued-pretraining code, split builder, config and manifest (no weights, no dataset) |
| `src/forward_head_pilot/` | Wave F cohort builder, raw-source verifier, ridge pilot, contract tests, `CHANGES.md` |
| `src/data_admission/` | RMDB RDAT parser and audits, Rfam inventory curation, the 16 Oct expansion download list, manifest and fetch script |
| `jobs/` | sbatch scripts for Waves B–E and `submission_records.json` |
| `analysis/` | Analysis and aggregation scripts for Waves B–E and the Figure 1 script |
| `resource_plan/` | Resource-plan builder, design, ledger, facts builder, facts file, `rebuild_facts.py` |
| `results/` | Per-job summaries; data-admission and expansion evidence; the rank reports read by the facts builder; the job ledger |
| `docs/` | Cost diagnosis, anomaly-fix diff, reproduction guide, external data, number sources |
| `tools/` | Pack verifier, provenance checker, job-ledger builder, number-source checker, end-to-end reproduction |
| `PROVENANCE.json` | Origin path and sha256 of every copied or generated file, with the frozen-packet entries it matches |

The READMEs inside `src/` and `docs/DIAGNOSIS.md` are copied unchanged. They were written while each packet was being prepared, before submission, so they describe planned runs as "not submitted". The outcomes are in `results/` and in the tables above.

## Verifying

```bash
python3 -I tools/check_provenance.py                       # every copied file matches PROVENANCE.json
# with the execution pack unzipped next to it:
P=RIBO_ATLAS_FM_Execution_Pack_v2
python3 -I tools/verify_pack.py --pack $P
python3 -I tools/check_provenance.py --evidence-root $P/evidence
python3 -I tools/reproduce_from_pack.py --pack $P --work /tmp/ribo_rebuild
python3 -I tools/check_number_sources.py --evidence-root $P/evidence
```

## Licence and attribution

This repository is licensed under GPL-3.0 ([LICENSE](LICENSE)) because it contains code derived from RNAbpFlow (https://github.com/Bhattacharya-Lab/RNAbpFlow, GPL-3.0).

- **Upstream files.** These are byte-identical to the RNAbpFlow sources, except two modified files listed with their hashes in [NOTICE.md](NOTICE.md).
- **Feature pickles.** The RNAbpFlow training feature files in the execution pack are redistributed under CC-BY-4.0, with the attribution given in NOTICE.md.
- **RNA-FM.** RNA-FM (https://github.com/ml4bio/RNA-FM, MIT) is used as an installed package and is not redistributed.
