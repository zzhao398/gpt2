#!/bin/bash
#SBATCH --account=bche-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10g
#SBATCH --time=10
#SBATCH --job-name=competition_metrics
#SBATCH --output=competition.out

echo "--- Running Competition Benchmarks ---"
rm -f competition_results.txt
srun ./competition_metrics

echo ""
echo "--- Results ---"
cat competition_results.txt
