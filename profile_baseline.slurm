#!/bin/bash
#SBATCH --job-name="profile_baseline"
#SBATCH --output="profile_baseline.out"
#SBATCH --error="profile_baseline.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cudatoolkit

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

REPO_ROOT=$(pwd)
BUILD_DIR=$(mktemp -d /tmp/build_baseline_XXXXXX)
cp -r . "$BUILD_DIR"
cd "$BUILD_DIR"

# 用 M1 未改动的原始 kernels（README 1 版本）
cp kernels_unmodified/matmul.cuh kernels/matmul.cuh

make clean
make test_gpt2

echo "=== baseline: nsys profile ==="
srun nsys profile --stats=true --force-overwrite=true -o baseline_nsys ./test_gpt2 > baseline_nsys_prof.out 2>&1

echo "=== baseline: ncu profile ==="
ncu --set full -f -o baseline_ncu ./test_gpt2 > baseline_ncu.out 2>&1

cp baseline_nsys_prof.out baseline_ncu.out baseline_ncu.ncu-rep baseline_nsys.nsys-rep "$REPO_ROOT/"
rm -rf "$BUILD_DIR"
