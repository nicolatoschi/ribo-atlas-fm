#!/bin/bash
#SBATCH --job-name=ribo-waveC-production-shape
#SBATCH --account=e-aif-2026fl01-993
#SBATCH --partition=booster
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=01:45:00
#SBATCH --exclude=jpbo-055-37,jpbo-048-33,jpbo-049-19
#SBATCH --output=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/waveC-%j.out
#SBATCH --error=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/waveC-%j.err
# Wave C (PI-approved 15 Oct, 128-node option): production job shape with anomaly mode off.
# C1 four concurrent 32-node trajectories (2 persistent updates + checkpoints);
# C2 each trajectory resumes on the NEXT block's nodes (cross-node restart);
# C3 sixteen concurrent 8-node trajectories with the efficient microbatch (8 cases/GPU at once):
#    groups 0-7 K8, groups 8-15 K32; C4 one 128-node K32/B512 update (1 case/GPU).
# Node health screen: jpbo-048-33 and jpbo-049-19 were 35-38% slower than peers in every Wave B
# step they ran (job 2339575); they are excluded like the known-faulty jpbo-055-37.
# No new numerical code: verified packets 32e2cc0f8974 (reliability) and e8ff330ee775 (campaign).
# Step caps from Wave B (job 2339575): B1 continuous 17.9 min, resume 8.4 min; B2 K8 local4 9.5-12.2 min;
# B4 m8 2.7 min/update; P2 N64 10.6 min. Caps ~1.6-2x those.
set -uo pipefail
ROOT=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015
REL=$ROOT/reliability_anomaly_off_32e2cc0f8974
CAMP=$ROOT/native_campaign_e8ff330ee775
PY=/e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/env/bin/python
CKPT=/e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/pretrained_checkpoint/RNA3DB.ckpt
LAUNCH=$CAMP/step_launch.py
CAP_CONTINUOUS=${CAP_CONTINUOUS:-30}; CAP_RESUME=${CAP_RESUME:-20}; CAP_WARM=${CAP_WARM:-25}; CAP_WIDE=${CAP_WIDE:-20}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export CUDA_CACHE_PATH=/tmp/ribo_${SLURM_JOB_ID}_nvcache
RES=$ROOT/results/job_${SLURM_JOB_ID}
mkdir -p "$RES"
N=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [ "${#N[@]}" -ne 128 ]; then echo "expected 128 nodes, got ${#N[@]}"; exit 2; fi
nodes() { local IFS=,; echo "${N[*]:$1:$2}"; }
LEDGER=$RES/steps.tsv
printf 'step\tnodes\tminutes_cap\tstart_utc\tend_utc\texit\n' > "$LEDGER"
run() {
  local label=$1 first=$2 count=$3 minutes=$4 dir=$5; shift 5
  local list; list=$(nodes "$first" "$count")
  if [ -z "$list" ] || [ "$(echo "$list" | tr ',' '\n' | grep -c .)" -ne "$count" ]; then
    printf '%s\t%s\t%s\t-\t-\tREFUSED_bad_nodelist\n' "$label" "$list" "$minutes" >> "$LEDGER"; return 97
  fi
  local start; start=$(date -u +%FT%TZ)
  mkdir -p "$(dirname "$RES/$label")"
  srun --exact -N "$count" --ntasks-per-node=4 -c 8 --gres=gpu:4 --mem=0 --kill-on-bad-exit=1 \
       -t "$minutes" -w "$list" --chdir="$dir" "$PY" "$LAUNCH" "$@" < /dev/null > "$RES/$label.log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$list" "$minutes" "$start" "$(date -u +%FT%TZ)" "$rc" >> "$LEDGER"
  return $rc
}
rel() {  # rel <label> <first> <trajectory> <stage> <cap>
  local t=$3
  mkdir -p "$RES/C/traj$t/checkpoint_states"
  run "$1" "$2" 32 "$5" "$REL" production_reliability_driver.py --source RNAbpFlow_training --checkpoint "$CKPT" \
      --fixture native_fixed_teacher_fixture.json --case 5zwm_B --microbatch 1 --local-constructs 1 --ensemble-size 32 \
      --grid-points 50 --out "$RES/C/traj$t" --state-root "$RES/C/traj$t/checkpoint_states" \
      --campaign-manifest "$REL/campaign_manifest.json" --stage "$4"
}
nat() {  # nat <label> <first> <count> <minutes> <case> <micro> <local> <K>
  mkdir -p "$RES/$1"
  run "$1" "$2" "$3" "$4" "$CAMP" native_fixed_site_limit_launcher.py --source RNAbpFlow_training --checkpoint "$CKPT" \
      --fixture campaign_teacher_fixture.json --case "$5" --microbatch "$6" --local-constructs "$7" --ensemble-size "$8" \
      --grid-points 50 --out "$RES/$1"
}

# C1: four concurrent continuous trajectories, trajectory t on block t.
for t in 0 1 2 3; do ( rel C1/continuous_traj$t $((32 * t)) $t continuous "$CAP_CONTINUOUS" ) & done
wait
# C2: trajectory t resumes on block (t+1) mod 4 (different nodes), only if its continuous stage succeeded.
for t in 0 1 2 3; do
  b=$(( (t + 1) % 4 ))
  if awk -F'\t' -v s="C1/continuous_traj$t" '$1==s && $6==0 {ok=1} END {exit !ok}' "$LEDGER"; then
    ( rel C2/resume_traj${t}_on_block$b $((32 * b)) $t resume "$CAP_RESUME" ) &
  else printf 'C2/resume_traj%s\t-\t-\t-\t-\tSKIPPED_continuous_failed\n' "$t" >> "$LEDGER"; fi
done
wait
# C3: sixteen concurrent 8-node trajectories, microbatch 8, local 8 (B256 each), one update each.
for g in $(seq 0 7); do ( nat C3/K8_m8_g$g $((8 * g)) 8 "$CAP_WARM" 5zwm_B 8 8 8 ) & done
for g in $(seq 8 15); do ( nat C3/K32_m8_g$g $((8 * g)) 8 "$CAP_WARM" 5zwm_B 8 8 32 ) & done
wait
# C4: all 128 nodes, K32/B512 (1 case per GPU), one update.
nat C4/wide_N128_K32 0 128 "$CAP_WIDE" 5zwm_B 1 1 32
echo "wave C done $(date -u +%FT%TZ)"
cat "$LEDGER"
