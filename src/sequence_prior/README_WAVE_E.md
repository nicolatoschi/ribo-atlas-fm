# Wave E: RNA-FM sequence-prior adaptation on JUPITER (prepared, not submitted)

Prepared by Claude on 15 Oct 2026. Nothing has been uploaded or submitted, and no SSH connection was made.

**What it replaces.** The only preliminary evidence for "RIBO-SeqPair adapts RNA-FM" so far is the synthetic transformer prototype in `work/hpc_pilot/` (60M/475M parameters, synthetic targets). Wave E runs the real RNA-FM, unmodified, on real Rfam 15.1 sequences:

- masked-LM continued pretraining with DDP and BF16 autocast;
- strong scaling at 8/16/32 nodes, then a 64-node run;
- held-out MLM loss before and after training, on families never seen in Wave E training.

## Files

| Path | Content |
|---|---|
| `waveE_05ec1ab6b438/` | **Frozen packet** (read-only). Freeze sha256 `05ec1ab6b43869854595aa8aa3818a4caee62960e842f376bd6a0a36741542ba` is computed over the configuration plus the sha256 of all 13 files, which `waveE_manifest.json` lists. |
| — `waveE_mlm.py` | Trainer/evaluator. Modes: `scaling` (phase 1) and `train` (phase 2). |
| — `waveE_config.json` | All hyperparameters, plus the dataset and weights hashes. |
| — `waveE_dataset.tsv`, `waveE_split_summary.json` | Split with hashes. The freezer rebuilds it and compares it byte for byte. |
| — `build_waveE_split.py` | Deterministic split builder. |
| — `step_launch.py` | Verbatim copy of the Wave B launcher (sha `eb0d5677…`). |
| — `verify_packet.py` | Pre-flight check: manifest, freeze sha, weights sha, versions. |
| — `analyze_waveE.py` | Analysis script. |
| — `smoke_checks.py` | Component checks. |
| — `setup_waveE_env.sh`, `requirements-waveE.txt` | JUPITER setup (not run) and hash-pinned requirements. |
| — `rnafm_sources.json` | Sources record. |
| — `freeze_waveE_packet.py` | The freezer itself. |
| `sbatch/waveE_rnafm_mlm.sbatch` | The 64-node job, sha256 `4ecd1e273692237e67dd95bdf95a656092fccc6ffaa5c38361e6dd32949a56e4`. |
| `sbatch/waveE_gate_1node_OPTIONAL.sbatch` | Optional 1-node GPU gate, sha256 `0d496b00…`; at most 1.33 GPU-h. |
| `dryrun/` | Fake `scontrol`/`srun`, probe, `run_dryrun.sh`, `check_dryrun.py`. Results are in `out/CHECKS.txt`. |
| `smoke/waveE_05ec1ab6b438/` | CPU smoke-test outputs (`run_cpu_smoke.sh`). |
| `src/`, `staging/`, `sources/` | Editable sources, split staging, and the RNA-FM source record. |
| `waveE_submission_prep.json` | Hashes, cost and the upload/submit procedure. |

## RNA-FM sources

All files were verified locally.

- **Code:** PyPI `rna-fm` 0.2.2.
  - Wheel sha256 `4fff72892ad02559be969b7b3263483e7f3730132025ea05c9927c3675a5e0b4`.
  - Its `fm/` package is byte-identical to GitHub `ml4bio/RNA-FM` HEAD `348951516e0963d22bbb33b3c9fc18c89081d38e` (2025-05-27).
  - Installed with `--no-deps --require-hashes` together with `ptflops` 0.7.5 (sha `c47543d5…`). `import fm` needs ptflops; nothing in `fm` uses scikit-learn.
- **Weights:** `RNA-FM_pretrained.pth`, 1,194,424,423 bytes, sha256 `5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99`.
  - Source: the Hugging Face repo that the RNA-FM README names, `cuhkaih/rnafm`, pinned at revision `91d4a46d28d8054a7b429955e8fc0c253ba0afd6`.
  - The package's own hub URL (`proj.cse.cuhk.edu.hk/rnafm/api/download`) returned HTTP 403 on 15 Oct. It is never used.
- **Model:**
  - Architecture: 12 layers, d = 640, FFN 5,120, 20 heads, 1,024 positions, vocabulary of 25 tokens.
  - Parameters: 99,521,546, of which 99,521,305 are trained (`contact_head` is frozen because MLM does not use it).
  - Loading: `torch.load(weights_only=True, mmap=True)`, with only `argparse.Namespace` allow-listed, then the official `fm.pretrained.load_model_and_alphabet_core`.
  - Fidelity check: the loaded model equals the checkpoint exactly. The fm loader zeroes the `<mask>` embedding row, but the tied `lm_head.weight` that is loaded afterwards restores it.
  - Initial-state digest: `24c733cc…`. The setup script asserts the same digest on JUPITER.

## Data and split

The source is Rfam 15.1 seed: 96,910 unique canonical sequences of 30–1,014 nt in 4,185 families. All inputs are verified against the hashes recorded on 12 Oct.

- **Clans.** The family CSV has no clan column, so clans come from the `#=GF CL` lines of the same `Rfam.seed.gz`: 482 families in 151 clans.
- **Grouping.** A clan, plus any families that share an exact sequence, form one group; there are 3,866 groups. Whole groups are held out in salted-hash order until at least 5% of families are held out, with held-out sequences capped at 8%.
- **Resulting split.** No held-out sequence belongs to any training family (asserted).

| Split | Sequences | Families | Groups | Clans |
|---|---:|---:|---:|---:|
| `train` | 92,684 | 3,975 | — | — |
| `heldout_family` | 3,332 | 210 | 198 | 6 |
| `val_in_family` | 894 | — | — | — |

`val_in_family` is a random 1% of sequences whose families stay in training. It contrasts in-distribution change with held-out change.

- **Length policy.** RNA-FM takes at most 1,022 nt (1,024 positions including `<cls>`/`<eos>`). The longest sequence is 1,014 nt, so nothing is cropped; the trainer refuses longer input.
- **Hashes.**
  - Dataset sha256: `fc34fe0aba27a57036a79890dc26dc5a3d905a3873c23f8747696c8aa240efb2`.
  - Per-split sha256 of the sorted IDs is in `waveE_split_summary.json`.

## Design

- **Global batch.** 8,192 sequences per step, identical at every scale: about 905k tokens and 134k masked positions per step. RNA-FM's own pretraining update was 10,240 sequences.
- **Masking.** 15% of positions: 80% `<mask>`, 10% random nucleotide, 10% unchanged. Masks come from a counter-based hash of (seed, stream position, residue).
- **World-size invariance.**
  - Sequences are snake-dealt over ranks. Each rank back-propagates sum-CE × W / N_global, so every world size computes the same gradient.
  - On CPU, world size 1 and world size 2 (gloo DDP) agree to 2.2e-7 in loss per step.
  - Phase-1 and phase-2 losses therefore must agree step by step. The analysis reports this as a correctness check of the 256-GPU DDP.
- **Optimizer.** AdamW with lr 1e-4, betas (0.9, 0.98), weight decay 0.01 and gradient clip 1.0. The lr warms up linearly over 10 steps, then follows a cosine to 1e-5 at step 72.
  - Precision: BF16 autocast, with fp32 master weights and fp32 loss.
  - Micro-batches are capped at 32,768 tokens and n·T² ≤ 16.8M.
- **Phase 1** (concurrent, disjoint nodes): 10 untimed steps, then 50 timed steps. Each step records the rank-max time, data-prep time, tokens/s, peak HBM (allocated/reserved) and host RSS.

| Run | Nodes | Node indices | Attention | Role |
|---|---:|---|---|---|
| P1_N8 | 8 | 0–7 | official | primary |
| P1_N16 | 16 | 8–23 | official | primary |
| P1_N32 | 32 | 24–55 | official | primary |
| P1_N8_sdpa | 8 | 56–63 | SDPA | supplementary |

  - P1_N8_sdpa uses nodes that would otherwise be idle. It gives the same function on CPU: logits agree to 2.6e-5.
- **Phase 2** (all 64 nodes, gated on at least one primary phase-1 success):
  - evaluation at step 0;
  - the same 60 timing steps;
  - continuation to step 72;
  - evaluations at every epoch boundary (steps 12, 23, 34, 46, 57, 68) and at 72;
  - final model saved (398 MB).
- **Evaluation.**
  - Held-out and in-family sets, each with 2 fixed mask replicates (all positions `<mask>`, fp32), identical before and after.
  - **Primary endpoint** (pre-specified in `waveE_config.json`, 15 Oct coordinator review):
    - what: held-out-family token-weighted CE change at the **first epoch boundary**, versus step 0, with perplexity ratio;
    - when: the first evaluation with epochs_seen ≥ 1.0, which is **step 12 (1.061 epochs)**;
    - interval: 95% bootstrap CI (10,000 resamples over the 210 held-out families); a group-level bootstrap is the sensitivity analysis;
    - reported whatever its sign.
  - **Secondary:**
    - step 72 vs step 0 (about 6.4 epochs), explicitly labelled as a *multi-epoch result on a small, redundant corpus*;
    - the in-family contrast at the same steps;
    - the paired change at every epoch boundary (the learning curve).
  - **Caveat in every report:** RNA-FM was pretrained on RNAcentral, which very likely already contains these Rfam seed sequences, so the split controls leakage within Wave E only.
- **Analysis** (`analyze_waveE.py`, also run on the batch host at the end of the job):
  - efficiency E = (T₈/T_N)/(N/8) with bootstrap CIs;
  - tokens/s overall and per GPU;
  - HBM and RSS;
  - padding share and rank imbalance;
  - loss consistency across runs;
  - held-out and in-family CE/perplexity change.

## Expected wall time and GPU-hours

These are estimates. Nothing has run on a GPU.

- **Compute per step.** About 5.4e14 FLOP. At an assumed 150–300 TFLOP/s effective per GH200 in eager BF16:
  - about 0.08–0.15 s per step on 32 GPUs (8 nodes);
  - about 0.02–0.04 s per step on 256 GPUs.
- **Every timing block is seconds long.** Wall time is dominated by start-up: imports from Lustre, NCCL init across 224–256 ranks, and model load.

| Part | Estimate |
|---|---|
| Pre-flight (batch host) | 0.5–1.5 min |
| Phase 1 | 2–5 min |
| Phase 2 (start-up, 8 evaluations, 72 steps, save) | 3–6 min |
| Analysis | under 1 min |
| **Total** | **about 6–13 min, i.e. 26–55 GPU-h (central ~45)** |

- **Pessimistic but completing:** about 20 min, or 85 GPU-h.
- **Hard maximum:** 64 × 4 GPUs × 50 min = **213.3 GPU-h**, under the 320 GPU-h target. The 15 + 25 min step caps bound both phases.
- **Failure modes are cheap.**
  - A pre-flight failure or a wrong node count exits before any `srun`.
  - If all primary phase-1 steps fail, phase 2 is skipped.
- **Optional gate:** expected 0.3–0.5 GPU-h, at most 1.33.
- **Expected scaling** (an estimate, to be measured): E(16) ≈ 0.8–0.95, E(32) ≈ 0.6–0.85, E(64) ≈ 0.3–0.6. At 64 nodes each GPU has only about 3.5k tokens per step, while the fp32 gradient all-reduce is 398 MB.

## Local verification (15 Oct)

**CPU smoke test.** Ran the frozen packet with the real weights: scratch venv, torch 2.13, Apple M CPU, fp32.

- Pre-flight: OK.
- Step time: about 1.3–1.4 s for 8 sequences (about 600 tokens/s).
- Model load: 0.33 s with mmap. A train-mode run (10 steps, 3 evaluations, save) took 30 s.
- Held-out CE at step 0:
  - 1.113 nats (perplexity 3.05) on 16 sequences;
  - 1.071 nats on 24 sequences in train mode.
- bf16 vs fp32 on one micro-batch: losses 0.82739 vs 0.82725 (difference 1.4e-4). CPU bf16 is about 90× slower than fp32 on this Mac, which is why the CPU runs use fp32.
- Weights sha256 unchanged after the read-only mmap load.
- The analysis reports the PRIMARY endpoint at the first evaluation ≥ 1 epoch (step 8 in this 64-sequence smoke) and the last step as the SECONDARY multi-epoch result.
- The CPU learning numbers (batch 8, 10 steps) are not predictive of the 8,192-sequence run.

**Placement dry run.** `dryrun/out/CHECKS.txt`: 23 passed, 0 failed. It was executed with macOS bash 3.2, the real `step_launch.py` and real pre-flight hashes.

- **Normal scenario:**
  - four phase-1 steps run concurrently on disjoint sets covering all 64 nodes;
  - P2 starts after them, on all 64 nodes;
  - MASTER_ADDR is the first host of each step, with a unique port per step;
  - WORLD_SIZE is 32/64/128/32/256;
  - each step has its own node-local CUDA cache;
  - every launch-rule flag is present, stdin is `/dev/null`, and `--exclusive` is never used.
- **Refusal scenarios:**
  - 63 nodes: refused, exit 2;
  - all of phase 1 fails: phase 2 is skipped;
  - wrong weights: pre-flight exit 3 with no `srun`.
- **Optional gate:** OK.

## Procedure on JUPITER

The steps are also listed in `waveE_submission_prep.json`.

1. Upload `waveE_05ec1ab6b438/` and `sbatch/` to `/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/`, then `mkdir -p results`.
2. On a login node, from inside the packet: `bash setup_waveE_env.sh`. It:
   - creates the venv `waveE_env` on top of `ribo_geometry_pilot_20261012/env`;
   - adds a `.pth` file to expose that env's own site-packages, because a venv inherits only the base interpreter;
   - installs the hash-pinned wheels;
   - downloads the weights and verifies their sha256;
   - runs a CPU load test that asserts the parameter count and digest.
3. Run `verify_packet.py --expect-torch 2.10` and check that it reports `ok: true`. Check the sbatch sha256 and that there are no duplicate `ribo-*` jobs.
4. Optional: submit `waveE_gate_1node_OPTIONAL.sbatch` first. Its losses must later match P2_N64 step by step. Analyse it with `analyze_waveE.py --reference GATE_N1 --train-run GATE_N1`.
5. `sbatch sbatch/waveE_rnafm_mlm.sbatch`, then retrieve `results/waveE_job_<id>/`. Skip `final_model_state.pt` unless you need it.

## Risks and open points

1. **The GPU path has never run.** NCCL, CUDA BF16, fused AdamW and the GH200 memory budget were exercised only on CPU (gloo, fp32 and CPU-bf16). The guards are the optional gate, the phase-2 gate, fail-fast checks (duplicate-GPU binding, masked-count assertions) and the step caps.
2. **The sign of the learning result is not guaranteed.**
   - RNA-FM was pretrained on RNAcentral, which very likely contains these Rfam seed sequences. The family/clan hold-out prevents leakage within Wave E only.
   - 72 steps of 8,192 sequences cover 6.36 epochs of a small, redundant corpus. Held-out CE may rise, which would indicate overfitting.
   - The primary endpoint (step 12, the first epoch boundary, vs step 0) is pre-specified and reported whatever its sign. Step 72 is reported only as a labelled secondary multi-epoch result, together with the epoch curve.
3. **"About one epoch".** Because the global batch is 8,192, the 60 timing steps already cover 5.3 epochs; the continuation adds 1.06. The **primary** endpoint is therefore the step-12 evaluation (1.061 epochs), which falls inside the timing block. The evaluation is excluded from step timing, and the step right after it is flagged and can be excluded. Training simply continues to 72 for the secondary multi-epoch result.
4. **Modest efficiency at 64 nodes is expected.** A 100M-parameter model on a 10M-token corpus leaves little work per GPU at 256 GPUs. The 16- and 32-node points are the production-relevant ones.
5. **Official attention materialises T×T fp32 probabilities** (`need_weights=True`). The micro-batch caps aim for a peak below about 35 GiB of the 95 GiB HBM, but this has not been measured. SDPA is measured only as a supplement.
6. **CPU/GPU binding.** None is set by default, as in the user launch rule and Wave B's main steps. GPUs are assigned by LOCAL_RANK, and the trainer records CPU affinity and GPU UUID. If Wave B step B0 favours `--cpu-bind=ldoms --gpu-bind=closest`, set `BIND=` in the sbatch; this changes its sha.
7. **No repeat runs.** There is one run per scale, so confidence intervals reflect step-to-step variation only, not node-to-node variance.
8. **Remaining environment and data risks.**
   - The weights download from Hugging Face needs outbound HTTPS from the login node. The fallback is to copy the verified file; its sha256 is known.
   - Local tests used torch 2.13, while JUPITER has 2.10. The APIs used (`weights_only`, `safe_globals`, mmap, `device_id`) all exist in 2.10.
   - There is no sequence-identity filter across unrelated clans.
   - Licences: the Hugging Face model card says Apache-2.0; the code is MIT. Cite Chen et al. 2022 (arXiv:2204.00300).
   - Account 993 is shared with AERIS.
