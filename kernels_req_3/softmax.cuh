#ifndef __SOFTMAX_KERNEL_CUH__
#define __SOFTMAX_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "../utils/cuda_utils.cuh"

// One block per row, parallel reduction for max and sum
__global__ void softmax_forward_kernel(float* out, float inv_temperature, const float* inp, int N, int T) {
    int row = blockIdx.x;
    if (row >= N * T) return;

    int own_pos = row % T;
    const float* x = inp + row * T;
    float* y = out + row * T;
    int tid = threadIdx.x;

    __shared__ float smem[256];

    // max reduction
    float maxval = -FLT_MAX;
    for (int i = tid; i <= own_pos; i += blockDim.x)
        maxval = fmaxf(maxval, x[i]);
    smem[tid] = maxval;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float global_max = smem[0];

    // sum reduction
    float sumval = 0.0f;
    for (int i = tid; i <= own_pos; i += blockDim.x) {
        float e = expf(inv_temperature * (x[i] - global_max));
        y[i] = e;
        sumval += e;
    }
    smem[tid] = sumval;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];

    //normalize
    for (int i = tid; i <= own_pos; i += blockDim.x)
        y[i] *= inv_sum;
}

#endif // __SOFTMAX_KERNEL_CUH__
