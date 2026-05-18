#!/bin/bash
#SBATCH --job-name="profile_ops"
#SBATCH --output="profile_ops.out"
#SBATCH --error="profile_ops.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 01:00:00

module reset
module load cudatoolkit

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# Save original kernel
cp kernels/matmul.cuh kernels/matmul.cuh.orig

# ──────────────────────────────────────────────────────
# op_7: Configuration Sweep best params (BM=32,BN=64,BK=32,TM=4,TN=8)
# Uses shared memory + register tiling, no bank-conflict padding
# ──────────────────────────────────────────────────────
echo "=== Profiling op_7 ==="
cp kernels_op_7/matmul.cuh kernels/matmul.cuh
make clean && make test_gpt2

echo "=== op_7: nsys profile ==="
srun nsys profile --stats=true --force-overwrite=true -o op7_nsys ./test_gpt2 > op7_nsys_prof.out 2>&1

echo "=== op_7: ncu profile ==="
ncu --set full -f -o op7_ncu ./test_gpt2 > op7_ncu.out 2>&1

# ──────────────────────────────────────────────────────
# op_12: Shared memory padding (BM=64,BN=64,BK=8,PAD=1)
# Eliminates 4-way bank conflicts via +1 float row padding
# ──────────────────────────────────────────────────────
echo "=== Profiling op_12 ==="
cp kernels_op_12/matmul.cuh kernels/matmul.cuh
make clean && make test_gpt2

echo "=== op_12: nsys profile ==="
srun nsys profile --stats=true --force-overwrite=true -o op12_nsys ./test_gpt2 > op12_nsys_prof.out 2>&1

echo "=== op_12: ncu profile ==="
ncu --set full -f -o op12_ncu ./test_gpt2 > op12_ncu.out 2>&1

# Restore original kernel
mv kernels/matmul.cuh.orig kernels/matmul.cuh

echo "=== Done ==="
