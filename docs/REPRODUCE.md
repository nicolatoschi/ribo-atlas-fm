# Reproducing the numbers and rerunning the waves

## 1. Rebuild the ledger and facts file

There are two ways to do this:

- **From this repository alone:** Python 3 standard library only, under a second.
- **From the execution pack:** the inputs are the frozen evidence copies.

### From the repository

```bash
python3 -I resource_plan/build_resource_plan_v2.py \
  --design resource_plan/design_v2.json \
  --waveB  results/waveB_job2339575/waveB_analysis.json \
  --waveC  results/waveC_job2342412/waveC_analysis.json \
  --waveD  results/waveD_job2340563/waveD_findings.json \
  --out /tmp/resource_plan_v2.json
cmp /tmp/resource_plan_v2.json resource_plan/resource_plan_v2.json           # byte-identical
python3 -I resource_plan/rebuild_facts.py --plan /tmp/resource_plan_v2.json \
  --out /tmp/facts_v2.json --compare resource_plan/facts_v2.json
```

The second command prints `values_identical_except_sources: true`, `request_NH: 83400` and `model_total_NH: 83238.96883557357`. In repository mode the facts builder also reads the 1,508 rank reports and `steps.tsv` files under `results/rank_reports/` (host RSS, gradient spikes and slow-node excess) and the data-admission evidence under `results/data_admission/` and `results/data_expansion/`.

To rebuild the GPU-hour ledger behind `spend_GPUh`, which needs the pack's scheduler records, run:

```bash
python3 -I tools/build_job_ledger.py --evidence-root $P/evidence --facts resource_plan/facts_v2.json --out /tmp/RIBO_JOB_LEDGER.json
```

The ledger covers 46 attempts: 393.17 GPU-h on FL01-991 and 1,865.07 GPU-h on FL01-993. It reproduces every `spend_GPUh` value.

### From an unzipped pack

Set `P=RIBO_ATLAS_FM_Execution_Pack_v2`, then run:

```bash
python3 -I $P/repository/ribo-atlas-fm/tools/verify_pack.py --pack $P                 # hashes, packets, freezes
python3 -I $P/repository/ribo-atlas-fm/tools/reproduce_from_pack.py --pack $P --work /tmp/ribo_rebuild
python3 -I $P/repository/ribo-atlas-fm/tools/check_number_sources.py --evidence-root $P/evidence
```

`reproduce_from_pack.py` reruns everything from the pack's frozen copies:

- the resource plan;
- the facts file;
- the Wave B and C analyses;
- the B0 gate;
- the strict restart qualifications (B1, Wave C trajectories 0–3, job 2319595);
- the Wave D aggregation;
- the Wave E analysis;
- the job ledger.

It compares each output with the shipped file and needs NumPy and Matplotlib for Waves D and E.

### Changing the design

Edit `design_v2.json` (arms, seeds, passes, cohort sizes, microbatch, inference cohorts, development panel, packing efficiency), then rerun the two commands. The facts builder asserts that the broad and clean cohorts fit inside the admitted clean pools after the 15% holdout. The script prints the total and checks the call bounds: 20,000–90,000 node-hours, with inference at most 10%. It does not use those bounds as inputs.

### Why `rebuild_facts.py` exists

`build_facts_v2.py` hard-codes the build machine's project root as `ROOT` and derives `ET`, `AO` and `SRC` from it at import. It is shipped unchanged. The wrapper imports it and replaces exactly these module-level paths: `SRC` (13 inputs), `AO` (the root of the globbed rank reports) and the output location. If the builder ever gains another path, the wrapper refuses to run. The `sources` field of the rebuilt file therefore names the new paths; every other value is compared for equality.

## 2. Rerun the waves on JUPITER

### Common setup

- **Environment.** Build an isolated venv over the site's CUDA PyTorch 2.10 environment with `src/training/setup_isolated_env.sh`. This uses `requirements-added.in`; the frozen result is `environment_freeze.txt`. The script refuses to replace the working CUDA Torch.
- **Checkpoint.** Fetch `RNA3DB.ckpt` (sha256 `bb598a73…e213`; see `docs/DATA.md`).
- **Packets.** Unzip the execution pack and restore the third-party feature pickles listed in `docs/DATA.md` into each training packet at their recorded paths. Then upload the packet directories, check every file against its manifest, and check the batch script's sha256 against `jobs/submission_records.json`.
- **Site paths.** The batch scripts hard-code the paths used on 15–16 Oct, at the top of each script:

  | Variable | Value |
  |---|---|
  | `ROOT` | `/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015` |
  | `PY` | `.../ribo_geometry_pilot_20261012/env/bin/python` |
  | `CKPT` | `.../pretrained_checkpoint/RNA3DB.ckpt` |

  They also hard-code the account. For a new site or account, change only these variables and record the script's new sha256.
- **Launching.** Steps are launched with `srun --exact -N n --ntasks-per-node=4 -c 8 --gres=gpu:4 ... < /dev/null &`, and `step_launch.py` takes rendezvous from `SLURM_STEP_NODELIST` with a port derived from the job and step IDs. This is what makes the concurrent steps safe.
- **Bad nodes.** Exclude known-bad nodes with `--exclude=jpbo-055-37,jpbo-048-33,jpbo-049-19,jpbo-061-16,jpbo-090-25`.
- **Retrieval.** Retrieve only the reports, logs, `steps.tsv` and `sacct`. Checkpoint states (about 56 GB per restart test) and structures stay on the cluster.

### The waves

| Wave | Script | Shape | Packets | After retrieval |
|---|---|---|---|---|
| B: cost and scaling | `jobs/waveB_cost_scaling.sbatch` | 64 nodes, 2 h 30 | `native_anomaly_off_b63550bd42ee`, `reliability_anomaly_off_32e2cc0f8974`, `native_campaign_e8ff330ee775`; the B0 "anomaly on" arm needs the original `native_fixed_frozen_670450638bf1` (in the pack) | `analysis/analyse_waveB.py --results <job> --output waveB_analysis.json`; `analysis/evaluate_gate.py --results <job>/B0 --output gate_B0.json`; `analysis/verify_reliability_multistep.py` for B1 (step IDs as recorded in `B1_reliability_qualification.json`) |
| C: production shape | `jobs/waveC_production_shape.sbatch` | 128 nodes, 1 h 45 | reliability and campaign packets | `analysis/analyse_waveC.py`; `verify_reliability_multistep.py` per trajectory `C/traj0..3` |
| D: evaluation pipeline | `jobs/waveD_eval_cost.sbatch` | 32 nodes, 1 h 30 | `waveD_efec09f1317a`; restore `cases/*/case_input.npz` from the RNAbpFlow-Train archive with `provenance/prepare/build_packet.py`. Check with `src/evaluation/verify_packet.py --packet … --checkpoint … --sbatch … --expected-sbatch-sha256 …` | `analysis/aggregate_waveD.py --results <job> --packet <packet> --out <dir> --sacct sacct.txt --pilot-refinement-manifest <packet>/provenance/refinement_packet/refinement_manifest.json --pilot-refinement-results <job_2298016 JSONs> --full-cohorts 25500` |
| E: RNA-FM prior | `jobs/waveE_rnafm_mlm.sbatch` (optional 1-node gate `waveE_gate_1node_OPTIONAL.sbatch`) | 64 nodes, 50 min | `waveE_05ec1ab6b438`; `bash setup_waveE_env.sh` in the packet installs hash-pinned `rna-fm` 0.2.2 and `ptflops`; then `verify_packet.py --expect-torch 2.10 --weights RNA-FM_pretrained.pth` | `analysis/analyze_waveE.py --results <job> --out <dir> --config waveE_config.json` |
| F: forward-head pilot | local CPU | — | `src/forward_head_pilot/` with the shipped cohort | `python test_forward_head_contract.py`; `python forward_head_pilot.py freeze --cohort ../cohort/paired_core_cohort.json --plan frozen_analysis_plan.json --extraction-audit ../cohort/raw_extraction_verification.json`; `python forward_head_pilot.py run --cohort … --plan … --output results/` (see `model/README.txt`) |

Run the analysis scripts with `python3 -B` rather than `-I`, because `analyse_waveC.py` imports its sibling `analyse_waveB.py`.

### Probing-data admission (local CPU)

The 60 RMDB files can be re-fetched and checked:
- the 52 files of 16 Oct with `src/data_admission/expansion_20261016/fetch_expansion.sh <dest> download_list.tsv`, which checks size and published sha256;
- the 8 files of 12 Oct from the URLs in `docs/DATA.md`.

Then run the unchanged `src/data_admission/parse_actual_rdat.py` over them. It reads `selected_sources.json` and `raw/<id>.rdat` from its own directory, so copy it next to the pack's `admission_run/selected_sources.json` with the fetched files in `raw/`. It builds `raw/admission_audit.sqlite`, about 6 GB. The shipped `admission_run/parse_time.txt` shows one 201 s run with a 62 MB peak RSS. That run's outputs are `actual_admission_evidence.json`, `expansion_breakdown.json` and the length distribution, all in the pack.

### Superseded scripts

`run0_ab_gate.sbatch`, `run1_reliability_N32.sbatch` and `run2_N8_repeats.sbatch` were prepared first:

- Run 0 was submitted as job 2339203 and cancelled before it started (0 GPU-h).
- Runs 1 and 2 were never submitted.

All three were folded into Wave B (steps B0, B1 and B2). They are kept because `submission_records.json` refers to them.

## 3. Expected cost of a full rerun

| Wave | Measured cost |
|---|---|
| B | 280 GPU-h |
| C | 489 GPU-h |
| D | 154 GPU-h |
| E | 14 GPU-h |
| F | 0 (CPU only) |

These are the measured costs on screened nodes, from sacct and the facts file.
