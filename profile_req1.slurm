#!/bin/bash
#SBATCH --job-name="profile_req1"
#SBATCH --output="profile_req1.out"
#SBATCH --error="profile_req1.err"
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
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# swap in req_1 matmul
cp kernels/matmul.cuh kernels/matmul.cuh.bak
cp kernels_req_1/matmul.cuh kernels/matmul.cuh

make clean
make test_gpt2

echo "=== req_1: nsys profile ==="
srun nsys profile --stats=true -o req1_nsys ./test_gpt2 > req1_nsys_prof.out 2>&1

echo "=== req_1: ncu profile ==="
ncu --set full -f -o req1_ncu ./test_gpt2 > req1_ncu.out 2>&1

# restore
mv kernels/matmul.cuh.bak kernels/matmul.cuh
