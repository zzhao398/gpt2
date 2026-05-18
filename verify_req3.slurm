#!/bin/bash
#SBATCH --job-name="verify_req3"
#SBATCH --output="verify_req3.out"
#SBATCH --error="verify_req3.err"
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

# copy req_3 kernels in
cp kernels/layernorm.cuh kernels/layernorm.cuh.bak
cp kernels/softmax.cuh kernels/softmax.cuh.bak
cp kernels/attention.cuh kernels/attention.cuh.bak
cp kernels_req_3/layernorm.cuh kernels/layernorm.cuh
cp kernels_req_3/softmax.cuh kernels/softmax.cuh
cp kernels_req_3/attention.cuh kernels/attention.cuh

make clean
make test_gpt2_kernels

echo "=== req_3: test_gpt2_kernels ==="
srun ./test_gpt2_kernels

echo "=== req_3: test_gpt2 ==="
make test_gpt2
srun ./test_gpt2

# restore
mv kernels/layernorm.cuh.bak kernels/layernorm.cuh
mv kernels/softmax.cuh.bak kernels/softmax.cuh
mv kernels/attention.cuh.bak kernels/attention.cuh
