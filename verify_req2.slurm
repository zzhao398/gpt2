#!/bin/bash
#SBATCH --job-name="verify_req2"
#SBATCH --output="verify_req2.out"
#SBATCH --error="verify_req2.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# copy req_2 kernels in
cp kernels/matmul.cuh kernels/matmul.cuh.bak
cp kernels_req_2/matmul.cuh kernels/matmul.cuh

make clean
make test_gpt2_kernels

echo "=== req_2: test_gpt2_kernels ==="
srun ./test_gpt2_kernels

echo "=== req_2: test_gpt2 ==="
make test_gpt2
srun ./test_gpt2

# restore
mv kernels/matmul.cuh.bak kernels/matmul.cuh
