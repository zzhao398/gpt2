#ifndef __LAYERNORM_KERNEL_CUH__
#define __LAYERNORM_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "../utils/cuda_utils.cuh"

// One block per token row, parallel reduction over C for mean and variance
__global__ void layernorm_forward_kernel(float* out, float* mean, float* rstd, const float* inp,
                                         const float* weight, const float* bias, int B, int T, int C) {
    int row = blockIdx.x; // which token (0..B*T-1)
    if (row >= B * T) return;

    const float* x = inp + row * C;
    int tid = threadIdx.x;

    __shared__ float smem[256];

    // mean
    float sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x)
        sum += x[i];
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float m = smem[0] / C;

    // variance
    float var = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float d = x[i] - m;
        var += d * d;
    }
    smem[tid] = var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float s = rsqrtf(smem[0] / C + 1e-5f);

    // normalize
    for (int i = tid; i < C; i += blockDim.x) {
        float n = s * (x[i] - m);
        out[row * C + i] = n * weight[i] + bias[i];
    }
}

void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    layernorm_forward_kernel<<<B * T, 256>>>(out, mean, rstd, inp, weight, bias, B, T, C);
}

#endif // __LAYERNORM_KERNEL_CUH__
