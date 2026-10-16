#!/bin/bash
#SBATCH --job-name=ribo-waveE-rnafm
#SBATCH --account=e-aif-2026fl01-993
#SBATCH --partition=booster
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:50:00
#SBATCH --exclude=jpbo-055-37
#SBATCH --output=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/waveE-%j.out
#SBATCH --error=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/waveE-%j.err
# Wave E (PI-approved 15 Oct 2026): masked-LM continued pretraining of the official RNA-FM
# (rna-fm 0.2.2, RNA-FM_pretrained.pth) on Rfam 15.1 seeds with a family/clan-held-out split.
# Packet waveE_05ec1ab6b438 (freeze 05ec1ab6b43869854595aa8aa3818a4caee62960e842f376bd6a0a36741542ba).
# Phase 1, concurrent on disjoint nodes, identical global batch (8,192 sequences/step),
#   10 untimed + 50 timed steps: P1_N8 (nodes 0-7), P1_N16 (8-23), P1_N32 (24-55), and the
#   supplementary P1_N8_sdpa (56-63, SDPA attention) on the otherwise idle nodes.
# Phase 2, all 64 nodes (only if a primary phase-1 step succeeded): held-out evaluation at
#   step 0, the same 60 timing steps, continue to step 72 (~1 more epoch), evaluations at
#   epoch boundaries and at step 72, final model saved. PRIMARY endpoint (pre-specified in
#   waveE_config.json): held-out-family CE at the first epoch boundary (step 12) vs step 0;
#   step 72 (~6.4 epochs) is a SECONDARY multi-epoch result.
# Max cost: 64 nodes x 4 GPUs x 50 min = 213.3 GPU-h. Step caps: 15 + 25 min.
# bash 3.2 compatible (no mapfile) so it can be dry-run on macOS with fake scontrol/srun.
set -uo pipefail
ROOT=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015
PKT=$ROOT/waveE_05ec1ab6b438
PY=$ROOT/waveE_env/bin/python
WEIGHTS=$ROOT/waveE_assets/RNA-FM_pretrained.pth
LAUNCH=$PKT/step_launch.py
CAP_P1=15
CAP_P2=25
BIND=""   # optional, e.g. "--cpu-bind=ldoms --gpu-bind=closest" if Wave B step B0 shows a benefit
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 PYTHONNOUSERSITE=1
export CUDA_CACHE_PATH=/tmp/${SLURM_JOB_ID}_nvcache
RES=$ROOT/results/waveE_job_${SLURM_JOB_ID}
mkdir -p "$RES"
LEDGER=$RES/steps.tsv
printf 'label\tnodes\tminutes_cap\tstart_utc\tend_utc\texit\n' > "$LEDGER"
echo "wave E job $SLURM_JOB_ID start $(date -u +%FT%TZ) on $SLURM_JOB_NODELIST"

# ---- pre-flight on the batch host (no GPU step): packet hashes + freeze sha, weights sha256, env
if ! "$PY" -I "$PKT/verify_packet.py" --packet "$PKT" --weights "$WEIGHTS" --expect-torch 2.10 --out "$RES/preflight.json" > "$RES/preflight.log" 2>&1; then
  echo "PREFLIGHT FAILED; no GPU step launched"; cat "$RES/preflight.log"; exit 3
fi

N=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [ "${#N[@]}" -ne 64 ]; then echo "expected 64 nodes, got ${#N[@]}"; exit 2; fi
nodes() { local IFS=,; echo "${N[*]:$1:$2}"; }

# run <label> <first-node-index> <node-count> <minutes> [trainer args...]
run() {
  local label=$1 first=$2 count=$3 minutes=$4; shift 4
  local list; list=$(nodes "$first" "$count")
  if [ -z "$list" ] || [ "$(echo "$list" | tr ',' '\n' | grep -c .)" -ne "$count" ]; then
    printf '%s\t%s\t%s\t-\t-\tREFUSED_bad_nodelist\n' "$label" "$list" "$minutes" >> "$LEDGER"; return 97
  fi
  local start; start=$(date -u +%FT%TZ)
  srun --exact -N "$count" --ntasks-per-node=4 -c 8 --gres=gpu:4 --mem=0 -t "$minutes" $BIND --kill-on-bad-exit=1 \
       -w "$list" --chdir="$PKT" \
       "$PY" "$LAUNCH" waveE_mlm.py --config waveE_config.json --dataset waveE_dataset.tsv --weights "$WEIGHTS" \
       --verify-dataset --label "$label" --out "$RES/$label" "$@" < /dev/null > "$RES/$label.log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$list" "$minutes" "$start" "$(date -u +%FT%TZ)" "$rc" >> "$LEDGER"
  return $rc
}

# ---- phase 1: concurrent strong scaling on disjoint node subsets ------------------------
P1ARGS="--mode scaling --stop-after 60"
run P1_N8       0  8 "$CAP_P1" $P1ARGS &
run P1_N16      8 16 "$CAP_P1" $P1ARGS &
run P1_N32     24 32 "$CAP_P1" $P1ARGS &
run P1_N8_sdpa 56  8 "$CAP_P1" $P1ARGS --attention sdpa &
wait
echo "phase 1 done $(date -u +%FT%TZ)"

# ---- phase 2: all 64 nodes, gated on at least one primary phase-1 success ----------------
OK=$(awk -F'\t' '($1=="P1_N8" || $1=="P1_N16" || $1=="P1_N32") && $6=="0"' "$LEDGER" | wc -l | tr -d ' ')
if [ "$OK" -ge 1 ]; then
  run P2_N64 0 64 "$CAP_P2" --mode train
else
  echo "no primary phase-1 step succeeded: phase 2 skipped"
fi
echo "wave E steps done $(date -u +%FT%TZ)"

# ---- analysis on the batch host (CPU, seconds) -------------------------------------------
"$PY" -I "$PKT/analyze_waveE.py" --results "$RES" --out "$RES/analysis" > "$RES/analysis.log" 2>&1 || echo "analysis failed; rerun locally"
cat "$RES/analysis.log"
cat "$LEDGER"
