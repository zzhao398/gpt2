#!/bin/bash
#SBATCH --account=bche-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10g
#SBATCH --time=55
#SBATCH --job-name=competition_baseline2
#SBATCH --output=competition_baseline2.out

echo "--- Running Remaining Baseline Benchmarks ---"
srun ./competition_metrics competition_baseline2_results.txt

echo ""
echo "--- Results ---"
cat competition_baseline2_results.txt
