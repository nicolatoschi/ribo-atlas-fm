#!/bin/bash
#SBATCH --job-name=ribo-anomaly-off-reliability-N32
#SBATCH --account=e-aif-2026fl01-993
#SBATCH --partition=booster
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/ribo-anomaly-off-reliability-N32-%j.out
#SBATCH --error=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/ribo-anomaly-off-reliability-N32-%j.err
set -euo pipefail
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
RES=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/job_${SLURM_JOB_ID}
STATE="$RES/checkpoint_states"; mkdir -p "$RES" "$STATE"
srun --kill-on-bad-exit=1 --chdir=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/reliability_anomaly_off_32e2cc0f8974 /e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/env/bin/python rank_launch.py production_reliability_driver.py --source RNAbpFlow_training --checkpoint /e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/pretrained_checkpoint/RNA3DB.ckpt --fixture native_fixed_teacher_fixture.json --case 5zwm_B --microbatch 1 --grid-points 50 --local-constructs 1 --ensemble-size 32 --out "$RES" --state-root "$STATE" --campaign-manifest /e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/reliability_anomaly_off_32e2cc0f8974/campaign_manifest.json --stage continuous
srun --kill-on-bad-exit=1 --chdir=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/reliability_anomaly_off_32e2cc0f8974 /e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/env/bin/python rank_launch.py production_reliability_driver.py --source RNAbpFlow_training --checkpoint /e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/pretrained_checkpoint/RNA3DB.ckpt --fixture native_fixed_teacher_fixture.json --case 5zwm_B --microbatch 1 --grid-points 50 --local-constructs 1 --ensemble-size 32 --out "$RES" --state-root "$STATE" --campaign-manifest /e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/reliability_anomaly_off_32e2cc0f8974/campaign_manifest.json --stage resume
