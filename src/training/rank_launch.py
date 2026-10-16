#!/usr/bin/env python3
import os
import subprocess
import sys

env=os.environ.copy()
env['RANK']=env.get('SLURM_PROCID','0')
env['LOCAL_RANK']=env.get('SLURM_LOCALID','0')
env['WORLD_SIZE']=env.get('SLURM_NTASKS','1')
hosts=subprocess.check_output(['scontrol','show','hostnames',env['SLURM_JOB_NODELIST']],text=True).splitlines()
env['MASTER_ADDR']=hosts[0]
env['MASTER_PORT']=str(20000+int(env['SLURM_JOB_ID'])%20000)
env['OMP_NUM_THREADS']='8'
os.execve(sys.executable,[sys.executable,'-u',*sys.argv[1:]],env)
