#!/bin/bash
#SBATCH --job-name="generate_tokens"
#SBATCH --output="generate_tokens_output.out"
#SBATCH --error="generate_tokens_errors.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# run model
srun ./next_token_generation "The meaning of life is"
