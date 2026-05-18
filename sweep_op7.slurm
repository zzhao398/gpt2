#!/bin/bash
#SBATCH --job-name="sweep_op7"
#SBATCH --output="sweep_op7.out"
#SBATCH --error="sweep_op7.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:30:00

module reset
module load cudatoolkit

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

python kernels_op_7/sweep.py --out kernels_op_7/sweep_results.csv
