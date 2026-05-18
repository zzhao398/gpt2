#!/bin/bash
#SBATCH --job-name="profile_req2"
#SBATCH --output="profile_req2.out"
#SBATCH --error="profile_req2.err"
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

# swap in req_2 matmul
cp kernels/matmul.cuh kernels/matmul.cuh.bak
cp kernels_req_2/matmul.cuh kernels/matmul.cuh

make clean
make test_gpt2

echo "=== req_2: nsys profile ==="
srun nsys profile --stats=true -o req2_nsys ./test_gpt2 > req2_nsys_prof.out 2>&1

echo "=== req_2: ncu profile ==="
srun ncu --set full -f -o req2_ncu ./test_gpt2 > req2_ncu.out 2>&1

# restore
mv kernels/matmul.cuh.bak kernels/matmul.cuh
