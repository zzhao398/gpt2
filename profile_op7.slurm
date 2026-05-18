#!/bin/bash
#SBATCH --job-name="profile_op7"
#SBATCH --output="profile_op7.out"
#SBATCH --error="profile_op7.err"
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
BUILD_DIR=$(mktemp -d /tmp/build_op7_XXXXXX)
cp -r . "$BUILD_DIR"
cd "$BUILD_DIR"

cp kernels_op_7/matmul.cuh kernels/matmul.cuh

make clean
make test_gpt2

echo "=== op_7: nsys profile ==="
srun nsys profile --stats=true --force-overwrite=true -o op7_nsys ./test_gpt2 > op7_nsys_prof.out 2>&1

echo "=== op_7: ncu profile ==="
ncu --set full -f -o op7_ncu ./test_gpt2 > op7_ncu.out 2>&1

# 把结果文件拷回原目录
cp op7_nsys_prof.out op7_ncu.out op7_ncu.ncu-rep op7_nsys.nsys-rep "$REPO_ROOT/"

# 清理临时目录
rm -rf "$BUILD_DIR"
