# Cost diagnosis: 175-nt training-step cost and the 28% sustained slowdown

15 October 2026. Diagnosis comes from local code reading plus a local CPU A/B test. No JUPITER job was submitted.

## Finding

The upstream RNAbpFlow file `src/models/flow_module.py` (line 25) calls `torch.autograd.set_detect_anomaly(True)` at import time. Anomaly detection is a PyTorch debugging mode. For every autograd operation it records a Python stack trace in the forward pass and checks for NaN in the backward pass. That work is CPU-bound and runs once per operation.

| Driver | Disables anomaly mode after model construction? | Jobs |
|---|---|---|
| `full_trajectory_driver.py` | yes (line 123) | 2298974 (157 nt, 459 s/update) |
| `ensemble_training_driver.py` | yes (line 251) | 2298975 and others |
| `geometry_pilot/geometry_scaling_driver.py` | yes (line 160) | 2297399–403 (Figure 1, so Figure 1 is clean) |
| `batched_full_trajectory_driver.py` (native-fixed packet 670450638bf1) | **no** | 2305296/97/98; earlier timeouts 2300741, 2303509/10 |
| `production_reliability_driver.py` (packet aa8af773fd10) | **no** | 2319595 |
| batched micro2/4/8, full_weak, optimized, power drivers | **no** | 2299567–69, 2300345, 2300616, 2300716 |

So every 175-nt full-objective measurement, and every "batched" timeout before it, ran with debugging instrumentation switched on.

## Evidence

**1. Only the autograd-recorded phases are slow.** This is the per-phase median across ranks, K32/T50, 1,600 model calls per phase.

| Phase | 157 nt, anomaly off (2298974) | 175 nt, anomaly on (2305296) | Ratio |
|---|---:|---:|---:|
| No-grad mean pass (anomaly mode inactive) | 63.3 s | 62.3 s | 0.98× |
| Grad-enabled forward | 112.5 s | 2,675 s | 23.8× |
| Backward + recompute | 280.8 s | 2,156 s | 7.7× |
| Teacher (absorbs the DDP wait for the slowest rank) | 29.6 s | 512 s | — |

The no-grad pass is the same at 157 and 175 nt. The model calls themselves are therefore not length-limited at this size, and the gap does not come from sequence length.

**2. The 28% gap between 2305296 and 2319595 sits entirely in the instrumented phases.** The work was identical: same call counts and the same first-update gradient norm (88.58099).

| Phase (median over 128 ranks) | 2305296 | 2319595 | Ratio |
|---|---:|---:|---:|
| No-grad mean pass | 62.3 s | 62.8 s | 1.009 |
| Grad forward | 2,675 s | 3,464 s | 1.295 |
| Backward | 2,156 s | 2,837 s | 1.316 |

GPU-side speed was the same in both jobs. The CPU-bound debug overhead differed with the node set. Within one job, the instrumented phases vary 1.22–1.32× across ranks, against 1.14–1.16× for the no-grad pass. Under synchronous DDP the slowest rank sets the update time.

**3. Local CPU A/B test.**
- **Setup:** the frozen driver code, the real 175-nt 5zwm_B input, the official checkpoint, K2/T5, microbatch 1, one update. Run twice per mode on a local Apple CPU with torch 2.13. Files are in `cpu_anomaly_ab/`.

  | Phase | Anomaly off | Anomaly on | Ratio |
  |---|---:|---:|---:|
  | No-grad pass | 2.26 / 2.41 s | 2.48 / 2.42 s | ≈1.0× |
  | Grad forward | 2.34 / 2.44 s | 5.05 / 4.94 s | 2.1× |
  | Backward | 6.26 / 6.43 s | 11.56 / 11.38 s | 1.8× |
- **Numerics unchanged:** the gradient norm is 16.591383 in both modes, and model-call counts are identical. Anomaly mode changes speed only.
- **Why the ratio is smaller on CPU:** each operation already takes milliseconds there, so the fixed per-operation overhead is a small share. On a GH200, an unbatched 175-nt operation takes microseconds, so the same overhead dominates.
- **Patch used:** the scratch copy replaced only the CPU self-test block with one timed update (`scratch_driver_timing_patch.diff`). The update body was unchanged.

## Consequences

- **Correctness:** the numerical results of 2305296/97/98 and 2319595 still stand. Their gates, replay checks, gradients and restart reproduction are unaffected. **Their timings are not a valid cost basis.**
- **Likely corrected cost at 175 nt (to be measured):**

  | Assumption | c32 at N32 (node-s/case) | c8 at N8 (node-s/case) |
  |---|---:|---:|
  | Current budget basis (anomaly on) | 1,340.7 | 323.5 |
  | 157-nt timings applied unchanged | ≈114.8 | ≈28.1 |
  | 157-nt timings × quadratic length factor (1.242) | ≈142.6 | ≈34.9 |

  The no-grad evidence favours the length-insensitive end of that range.
- **Effect on the planned curriculum:** the current curriculum (8,000 warm / 4,000 fine cores, 3 arms × 3 seeds) would cost about **6,400–7,900 trajectory node-hours, not 74,437**. That is roughly 9–12× less.
- **The resource plan must be re-derived.** At corrected costs, the earlier internal costing (85,500 NH) is about 6× more than this curriculum needs. Shrinking the request has its own limits:
  - The JUPITER minimum is 20,000 NH.
  - Inference is capped at 10% of the allocation. The current 5,000 NH evaluation block would then need a request of at least 50,000 NH, or the evaluation must shrink with the request.
  - The alternative is to restore the broader exposure that was cut to fit the inflated price (earlier drafts planned far more cores and passes; the clean pool has 111,703 cores). This also answers the "63 warm / 64 fine updates per trajectory" weakness.
  - This is a PI decision, to be taken once the corrected rates are measured.
- **Statements to retire:** "Native measurements support eight-node warm and 32-node fine stages" at 1.47/1.52 h per update. The 97.3% efficiency (measured on a CPU-bound, perfectly parallel workload) and the 74.4k forecast must be re-measured. Fixing a debug flag inherited from upstream that cost about 10× is itself a citable engineering result.

## Proposed JUPITER runs (not submitted; awaiting approval)

**The fix.** Add one statement, `torch.autograd.set_detect_anomaly(False)`, right after `model=Workload().to(device)`. This is the same convention the 157-nt and ensemble drivers already use. It goes into the batched driver (new native-fixed packet) and into the reliability driver (new reliability packet). No other change; the new freeze hashes are recorded and the numerical update body stays identical. Each report also records `torch.is_anomaly_enabled()`.

| Run | Purpose | Shape | Expected wall | Expected GPU-h | Limit / max GPU-h |
|---|---|---|---:|---:|---:|
| 0. A/B gate | Same node, anomaly off then on, K8/T50, 4 ranks × 1 case; confirms the ratio on GH200 and matching gradient norms | 1 node, 2 sruns | ~30 min | ~2 | 1 h / 4 |
| 1. Sustained N32/K32 + restart, anomaly off | Replaces c32 and the 5,363 s basis. 2 consecutive updates plus fresh-process resume (3 timed updates); restart gates as in 2319595 | 32 nodes, 128 ranks, B128 | ~35–45 min | ~85 | 1 h 30 / 192 |
| 2. N8 K8 and N8 K32, anomaly off, ×2 repeats each | Replaces c8 and the N8/K32 reference; restores the strong-scaling number from repeated measurements | 8 nodes, 4 sruns | ~1 h 30–1 h 50 | ~55 | 3 h / 96 |
| 3. (optional) Profiler trace | One K8/T50 update under `torch.profiler`. Measures GPU utilisation, and so whether batching the K members (one RNA per GPU now) could give a further, separately qualified speed-up | 1 node | ~20 min | ~1.5 | 45 min / 3 |

- **Total for runs 0–2:** about 140 GPU-h expected (about 35 node-hours), at most 292 GPU-h reserved. Run 3 adds up to 3 GPU-h.
- **Order:** Run 0 first, as a gate. Runs 1 and 2 go only if it confirms a ratio of at least 5× and identical gradient norms.
- **Account:** e-aif-2026fl01-993, as for 2319595. Account 991 has about 2% left (67,654 core-h) and is in use by AERIS. Account 993 is shared with the AERIS 15,000 GPU-h preliminary-data cap, and RIBO has used 927 GPU-h there so far.
- **Retrieval:** reports and sacct only. If Run 1's checkpoints are kept, they are about 56 GB again; that run can write them to a scratch path to be deleted after verification.
