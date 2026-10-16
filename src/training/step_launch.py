#!/usr/bin/env python3
"""Step-safe rank launcher for concurrent srun steps inside one wide allocation.

rank_launch.py derives MASTER_ADDR from SLURM_JOB_NODELIST and one port per job,
so concurrent steps on disjoint node subsets would rendezvous on the same host
and port. This launcher uses the step's own node list, a port unique per
(job, step), and a node-local CUDA cache (a shared ~/.nv cache hung 128-node
jobs). It only sets environment variables; no numerical code is touched.
"""
import os
import subprocess
import sys

env = os.environ.copy()
job, step = int(env['SLURM_JOB_ID']), int(env['SLURM_STEP_ID'])
env['RANK'] = env['SLURM_PROCID']
env['LOCAL_RANK'] = env['SLURM_LOCALID']
env['WORLD_SIZE'] = env.get('SLURM_STEP_NUM_TASKS', env['SLURM_NTASKS'])
hosts = subprocess.check_output(['scontrol', 'show', 'hostnames', env['SLURM_STEP_NODELIST']], text=True).split()
env['MASTER_ADDR'] = hosts[0]
env['MASTER_PORT'] = str(20000 + (job * 97 + step) % 20000)
env['OMP_NUM_THREADS'] = '8'
env['CUDA_CACHE_PATH'] = f'/tmp/ribo_{job}_{step}_nvcache'
os.makedirs(env['CUDA_CACHE_PATH'], exist_ok=True)
os.execve(sys.executable, [sys.executable, '-u', *sys.argv[1:]], env)
