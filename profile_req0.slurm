#!/bin/bash
#SBATCH --job-name="profile_req0"
#SBATCH --output="profile_req0.out"
#SBATCH --error="profile_req0.err"
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

REPO_ROOT=$(pwd)
BUILD_DIR=$(mktemp -d /tmp/build_req0_XXXXXX)
cp -r . "$BUILD_DIR"
cd "$BUILD_DIR"

cp kernels_req_0/matmul.cuh kernels/matmul.cuh

make clean
make test_gpt2

echo "=== req_0: nsys profile ==="
srun nsys profile --stats=true --force-overwrite=true -o req0_nsys ./test_gpt2 > req0_nsys_prof.out 2>&1

echo "=== req_0: ncu profile ==="
ncu --set full -f -o req0_ncu ./test_gpt2 > req0_ncu.out 2>&1

cp req0_nsys_prof.out req0_ncu.out req0_ncu.ncu-rep req0_nsys.nsys-rep "$REPO_ROOT/"
rm -rf "$BUILD_DIR"
