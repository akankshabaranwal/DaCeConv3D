#!/bin/bash -l
#SBATCH --job-name="daceconv"
#SBATCH --account="g34"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akanksha.baranwal@inf.ethz.ch
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

srun /users/abaranwa/my_venv/bin/python /users/abaranwa/dacelocal/samples/simple/conv3D.py
