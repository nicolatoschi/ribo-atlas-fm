# Batch scripts and submission records

`submission_records.json` records, for each submission:

- the job ID, account, node count, time limit and GPU-hour cap;
- the sha256 of the script;
- the packets used;
- the checks performed before submission.

Each script here is the exact submitted file. Its sha256 matches the record, as `PROVENANCE.json` and `tools/check_provenance.py` confirm.

| Script | sha256 (prefix) | Job | Status |
|---|---|---|---|
| `waveB_cost_scaling.sbatch` | `6ebee98f34e6` | 2339575 | COMPLETED 0:0, 64 nodes, 3,942 s |
| `waveC_production_shape.sbatch` | `a1a18ddbdc88` | 2342412 | COMPLETED 0:0, 128 nodes, 3,440 s |
| `waveD_eval_cost.sbatch` | `b53add7614b0` | 2340563 | COMPLETED, 32 nodes, 4,345 s (248/256 cases) |
| `waveE_rnafm_mlm.sbatch` | `4ecd1e273692` | 2340325 | COMPLETED 0:0, 64 nodes, 199 s |
| `waveE_gate_1node_OPTIONAL.sbatch` | `0d496b00de6f` | — | prepared, not submitted |
| `run0_ab_gate.sbatch` | `b7002b16e9c4` | 2339203 | cancelled before start (0 GPU-h); folded into Wave B step B0 |
| `run1_reliability_N32.sbatch`, `run2_N8_repeats.sbatch` | `3a5ea07f78d1`, `a03180c82c69` | — | never submitted; replaced by Wave B steps B1 and B2 |

The scripts use JUPITER site paths (`/e/project1/...`) and the project accounts that were in use on 15–16 Oct 2026. See `docs/REPRODUCE.md` for what to change at another site. Scheduler records (`sacct`, `steps.tsv`, job stdout and stderr) for these jobs are in the execution pack, together with the sacct records of every earlier RIBO attempt. `results/job_ledger/RIBO_JOB_LEDGER.json` summarises all 46 attempts.
