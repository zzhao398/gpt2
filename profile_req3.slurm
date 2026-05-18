#!/bin/bash
#SBATCH --job-name="profile_req3"
#SBATCH --output="profile_req3.out"
#SBATCH --error="profile_req3.err"
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

# swap in req_3 kernels
cp kernels/layernorm.cuh kernels/layernorm.cuh.bak
cp kernels/softmax.cuh kernels/softmax.cuh.bak
cp kernels/attention.cuh kernels/attention.cuh.bak
cp kernels_req_3/layernorm.cuh kernels/layernorm.cuh
cp kernels_req_3/softmax.cuh kernels/softmax.cuh
cp kernels_req_3/attention.cuh kernels/attention.cuh

make clean
make test_gpt2

echo "=== req_3: nsys profile ==="
srun nsys profile --stats=true -o req3_nsys ./test_gpt2 > req3_nsys_prof.out 2>&1

echo "=== req_3: ncu profile ==="
ncu --set full -f -o req3_ncu ./test_gpt2 > req3_ncu.out 2>&1

# restore
mv kernels/layernorm.cuh.bak kernels/layernorm.cuh
mv kernels/softmax.cuh.bak kernels/softmax.cuh
mv kernels/attention.cuh.bak kernels/attention.cuh
