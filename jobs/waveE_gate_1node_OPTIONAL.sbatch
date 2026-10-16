#!/bin/bash
#SBATCH --job-name=ribo-waveE-gate
#SBATCH --account=e-aif-2026fl01-993
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --exclude=jpbo-055-37
#SBATCH --output=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/waveE-gate-%j.out
#SBATCH --error=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/waveE-gate-%j.err
# OPTIONAL 1-node GPU gate for Wave E (not part of the 64-node design; max 4 GPUs x 20 min = 1.33 GPU-h).
# The CUDA/NCCL/bf16/fused-AdamW path was only exercised on CPU locally. This runs the phase-2
# protocol (mode train: eval, 60 timing steps, continue to 72, eval, save) on one node. The
# computation is world-size invariant, so its losses must match P2_N64 step by step.
# Packet waveE_05ec1ab6b438. bash 3.2 compatible.
set -uo pipefail
ROOT=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015
PKT=$ROOT/waveE_05ec1ab6b438
PY=$ROOT/waveE_env/bin/python
WEIGHTS=$ROOT/waveE_assets/RNA-FM_pretrained.pth
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 PYTHONNOUSERSITE=1
export CUDA_CACHE_PATH=/tmp/${SLURM_JOB_ID}_nvcache
RES=$ROOT/results/waveE_gate_job_${SLURM_JOB_ID}
mkdir -p "$RES"
LEDGER=$RES/steps.tsv
printf 'label\tnodes\tminutes_cap\tstart_utc\tend_utc\texit\n' > "$LEDGER"
if ! "$PY" -I "$PKT/verify_packet.py" --packet "$PKT" --weights "$WEIGHTS" --expect-torch 2.10 --out "$RES/preflight.json" > "$RES/preflight.log" 2>&1; then
  echo "PREFLIGHT FAILED"; cat "$RES/preflight.log"; exit 3
fi
N=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [ "${#N[@]}" -ne 1 ] || [ -z "${N[0]}" ]; then echo "expected 1 node, got ${#N[@]}"; exit 2; fi
start=$(date -u +%FT%TZ)
srun --exact -N 1 --ntasks-per-node=4 -c 8 --gres=gpu:4 --mem=0 -t 15 --kill-on-bad-exit=1 -w "${N[0]}" --chdir="$PKT" \
     "$PY" "$PKT/step_launch.py" waveE_mlm.py --config waveE_config.json --dataset waveE_dataset.tsv --weights "$WEIGHTS" \
     --verify-dataset --label GATE_N1 --out "$RES/GATE_N1" --mode train < /dev/null > "$RES/GATE_N1.log" 2>&1
rc=$?
printf '%s\t%s\t%s\t%s\t%s\t%s\n' GATE_N1 "${N[0]}" 15 "$start" "$(date -u +%FT%TZ)" "$rc" >> "$LEDGER"
tail -n 25 "$RES/GATE_N1.log"
cat "$LEDGER"
