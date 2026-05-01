#!/bin/bash
#SBATCH --job-name="verify_kv_cache"
#SBATCH --output="verify_kv_cache_output.out"
#SBATCH --error="verify_kv_cache_errors.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# Deterministic Test
echo -e "================ DETERMINISTIC VERIFICATION ================\n"
srun ./verify_kv_cache

# Random Performance Grid
echo -e "\n\n================ RANDOM PERFORMANCE GRID ================\n"

srun ./verify_kv_cache --random --batch 80 --prompt-len 32 --gen-len 32
srun ./verify_kv_cache --random --batch 80 --prompt-len 32 --gen-len 64
srun ./verify_kv_cache --random --batch 80 --prompt-len 32 --gen-len 128

# srun ./verify_kv_cache --random --batch 80 --prompt-len 64 --gen-len 32
# srun ./verify_kv_cache --random --batch 80 --prompt-len 64 --gen-len 64
# srun ./verify_kv_cache --random --batch 80 --prompt-len 64 --gen-len 128

# srun ./verify_kv_cache --random --batch 80 --prompt-len 128 --gen-len 32
# srun ./verify_kv_cache --random --batch 80 --prompt-len 128 --gen-len 64
# srun ./verify_kv_cache --random --batch 80 --prompt-len 128 --gen-len 128

# srun ./verify_kv_cache --random --batch 80 --prompt-len 256 --gen-len 32
# srun ./verify_kv_cache --random --batch 80 --prompt-len 256 --gen-len 64
# srun ./verify_kv_cache --random --batch 80 --prompt-len 256 --gen-len 128

# srun ./verify_kv_cache --random --batch 80 --prompt-len 512 --gen-len 32
# srun ./verify_kv_cache --random --batch 80 --prompt-len 512 --gen-len 64
# srun ./verify_kv_cache --random --batch 80 --prompt-len 512 --gen-len 128
