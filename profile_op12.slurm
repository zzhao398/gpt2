#!/bin/bash
#SBATCH --job-name="profile_op12"
#SBATCH --output="profile_op12.out"
#SBATCH --error="profile_op12.err"
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

# 每个 job 在独立的临时目录里 build，避免并行 job 互相覆盖 kernels/matmul.cuh
REPO_ROOT=$(pwd)
BUILD_DIR=$(mktemp -d /tmp/build_op12_XXXXXX)
cp -r . "$BUILD_DIR"
cd "$BUILD_DIR"

cp kernels_op_12/matmul.cuh kernels/matmul.cuh

make clean
make test_gpt2

echo "=== op_12: nsys profile ==="
srun nsys profile --stats=true --force-overwrite=true -o op12_nsys ./test_gpt2 > op12_nsys_prof.out 2>&1

echo "=== op_12: ncu profile ==="
ncu --set full -f -o op12_ncu ./test_gpt2 > op12_ncu.out 2>&1

# 把结果文件拷回原目录
cp op12_nsys_prof.out op12_ncu.out op12_ncu.ncu-rep op12_nsys.nsys-rep "$REPO_ROOT/"

# 清理临时目录
rm -rf "$BUILD_DIR"
