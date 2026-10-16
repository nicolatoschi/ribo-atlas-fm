#!/bin/bash
#SBATCH --job-name=ribo-waveB-cost-scaling
#SBATCH --account=e-aif-2026fl01-993
#SBATCH --partition=booster
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=02:30:00
#SBATCH --exclude=jpbo-055-37
#SBATCH --output=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/waveB-%j.out
#SBATCH --error=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/waveB-%j.err
# Wave B (PI-approved 15 Oct): cost and scaling with anomaly mode off. Phase 1 runs
# independent measurements concurrently on disjoint node subsets; phase 2 uses all
# 64 nodes. Each step has its own Slurm step time cap (srun -t, minutes).
set -uo pipefail
ROOT=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015
NEW=$ROOT/native_anomaly_off_b63550bd42ee
REL=$ROOT/reliability_anomaly_off_32e2cc0f8974
CAMP=$ROOT/native_campaign_e8ff330ee775
OLD=/e/project1/e-aif-2026fl01-991/ribo_native_fixed_training_20261013/native_fixed_frozen_670450638bf1
PY=/e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/env/bin/python
CKPT=/e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/pretrained_checkpoint/RNA3DB.ckpt
LAUNCH=$CAMP/step_launch.py
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export CUDA_CACHE_PATH=/tmp/ribo_${SLURM_JOB_ID}_nvcache
RES=$ROOT/results/job_${SLURM_JOB_ID}
mkdir -p "$RES"
N=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [ "${#N[@]}" -ne 64 ]; then echo "expected 64 nodes, got ${#N[@]}"; exit 2; fi
nodes() { local IFS=,; echo "${N[*]:$1:$2}"; }
LEDGER=$RES/steps.tsv
printf 'step\tnodes\tminutes_cap\tstart_utc\tend_utc\texit\n' > "$LEDGER"

# run <label> <first-node-index> <node-count> <minutes> <chdir> <script> [args...]
run() {
  local label=$1 first=$2 count=$3 minutes=$4 dir=$5; shift 5
  local list; list=$(nodes "$first" "$count")
  if [ -z "$list" ] || [ "$(echo "$list" | tr ',' '\n' | grep -c .)" -ne "$count" ]; then
    printf '%s\t%s\t%s\t-\t-\tREFUSED_bad_nodelist\n' "$label" "$list" "$minutes" >> "$LEDGER"; return 97
  fi
  local start; start=$(date -u +%FT%TZ)
  mkdir -p "$(dirname "$RES/$label")"
  srun --exact -N "$count" --ntasks-per-node=4 -c 8 --gres=gpu:4 --mem=0 --kill-on-bad-exit=1 \
       -t "$minutes" ${EXTRA:-} -w "$list" --chdir="$dir" \
       "$PY" "$LAUNCH" "$@" < /dev/null > "$RES/$label.log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$list" "$minutes" "$start" "$(date -u +%FT%TZ)" "$rc" >> "$LEDGER"
  return $rc
}
# nat <label> <first> <count> <minutes> <chdir> <case> <micro> <local> <K> <fixture>
nat() {
  mkdir -p "$RES/$1"
  run "$1" "$2" "$3" "$4" "$5" native_fixed_site_limit_launcher.py --source RNAbpFlow_training --checkpoint "$CKPT" \
      --fixture "${10}" --case "$6" --microbatch "$7" --local-constructs "$8" --ensemble-size "$9" --grid-points 50 --out "$RES/$1"
}
FIX=native_fixed_teacher_fixture.json
CFIX=campaign_teacher_fixture.json

# ---- Phase 1 -------------------------------------------------------------
# B0 (node 0): A/B gate. Fixed packet vs untouched original (anomaly on), then binding variant.
( nat B0/anomaly_off 0 1 25 "$NEW" 5zwm_B 1 1 8 $FIX
  nat B0/anomaly_on  0 1 45 "$OLD" 5zwm_B 1 1 8 $FIX
  EXTRA="--cpu-bind=ldoms --gpu-bind=closest" nat B0/anomaly_off_ldoms 0 1 25 "$NEW" 5zwm_B 1 1 8 $FIX ) &

# B1 (nodes 1-32): approved Run 1, N32/K32/B128 two persistent updates + fresh-process resume.
( RARGS=(--source RNAbpFlow_training --checkpoint "$CKPT" --fixture $FIX --case 5zwm_B --microbatch 1 --local-constructs 1
         --ensemble-size 32 --grid-points 50 --out "$RES/B1" --state-root "$RES/B1/checkpoint_states"
         --campaign-manifest "$REL/campaign_manifest.json")
  mkdir -p "$RES/B1/checkpoint_states"
  run B1/continuous_step 1 32 50 "$REL" production_reliability_driver.py "${RARGS[@]}" --stage continuous &&
  run B1/resume_step     1 32 35 "$REL" production_reliability_driver.py "${RARGS[@]}" --stage resume ) &

# B2 (nodes 33-48): approved Run 2 as two concurrent 8-node replicas, K8 then K32, local 4.
for r in 1 2; do
  first=$((33 + 8 * (r - 1)))
  ( nat B2/r${r}_K8  $first 8 25 "$NEW" 5zwm_B 1 4 8  $FIX
    nat B2/r${r}_K32 $first 8 60 "$NEW" 5zwm_B 1 4 32 $FIX ) &
done

# B3 (nodes 49-60): length ladder, 4 ranks x 1 case; K8 twice, K32 once.
i=0
for c in 1e8o_E 6cc1_B 1grz_B 6t3n_A 6chr_A 7ptk_B; do
  k8=$((49 + 2 * i)); k32=$((50 + 2 * i))
  ( nat B3/${c}_K8_rep1 $k8 1 30 "$CAMP" $c 1 1 8 $CFIX
    nat B3/${c}_K8_rep2 $k8 1 30 "$CAMP" $c 1 1 8 $CFIX ) &
  ( nat B3/${c}_K32 $k32 1 80 "$CAMP" $c 1 1 32 $CFIX ) &
  i=$((i + 1))
done

# B4 (nodes 61-63): microbatch sweep at fixed work, K8.
( nat B4/175_m1 61 1 35 "$CAMP" 5zwm_B 1 8 8 $CFIX
  nat B4/175_m8 61 1 20 "$CAMP" 5zwm_B 8 8 8 $CFIX ) &
( nat B4/175_m2 62 1 25 "$CAMP" 5zwm_B 2 8 8 $CFIX
  nat B4/175_m4 62 1 20 "$CAMP" 5zwm_B 4 8 8 $CFIX ) &
( nat B4/75_m1  63 1 45 "$CAMP" 1g59_B 1 16 8 $CFIX
  nat B4/75_m4  63 1 20 "$CAMP" 1g59_B 4 16 8 $CFIX
  nat B4/75_m16 63 1 15 "$CAMP" 1g59_B 16 16 8 $CFIX ) &

wait
echo "phase 1 done $(date -u +%FT%TZ)"

# ---- Phase 2: all 64 nodes, weak-scaling point K32/B256 (1 case per GPU) --
nat P2/weak_N64_K32 0 64 30 "$CAMP" 5zwm_B 1 1 32 $CFIX
echo "wave B done $(date -u +%FT%TZ)"
cat "$LEDGER"
