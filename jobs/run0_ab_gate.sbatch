#!/bin/bash
#SBATCH --job-name=ribo-anomaly-ab-gate
#SBATCH --account=e-aif-2026fl01-993
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/ribo-anomaly-ab-gate-%j.out
#SBATCH --error=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/ribo-anomaly-ab-gate-%j.err
set -euo pipefail
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
RES=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/results/job_${SLURM_JOB_ID}
mkdir -p "$RES/anomaly_off" "$RES/anomaly_on"
# Arm 1: fixed packet (anomaly off). Arm 2: untouched original 670450638bf1 (anomaly on). Same node, same inputs.
srun --kill-on-bad-exit=1 --chdir=/e/project1/e-aif-2026fl01-991/ribo_anomaly_off_20261015/native_anomaly_off_b63550bd42ee /e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/env/bin/python rank_launch.py native_fixed_site_limit_launcher.py --source RNAbpFlow_training --checkpoint /e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/pretrained_checkpoint/RNA3DB.ckpt --fixture native_fixed_teacher_fixture.json --case 5zwm_B --microbatch 1 --grid-points 50 --local-constructs 1 --ensemble-size 8 --out "$RES/anomaly_off"
srun --kill-on-bad-exit=1 --chdir=/e/project1/e-aif-2026fl01-991/ribo_native_fixed_training_20261013/native_fixed_frozen_670450638bf1 /e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/env/bin/python rank_launch.py native_fixed_site_limit_launcher.py --source RNAbpFlow_training --checkpoint /e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/pretrained_checkpoint/RNA3DB.ckpt --fixture native_fixed_teacher_fixture.json --case 5zwm_B --microbatch 1 --grid-points 50 --local-constructs 1 --ensemble-size 8 --out "$RES/anomaly_on"
