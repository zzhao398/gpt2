#ifndef __LAYERNORM_KERNEL_CUH__
#define __LAYERNORM_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "../utils/cuda_utils.cuh"

__global__ void layernorm_forward_kernel(float* __restrict__ out, float* __restrict__ mean,
                                         float* __restrict__ rstd, const float* __restrict__ inp,
                                         const float* __restrict__ weight, const float* __restrict__ bias,
                                         int B, int T, int C) {
    // Implement this

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < B * T) {
        float sum = 0.0f;
        for (int i = 0; i < C; i++) {
            sum += inp[index * C + i];
        }
        float m = sum / C;
        float sigma2 = 0.0f;
        for (int i = 0; i < C; i++) {
            sigma2 += (inp[index * C + i] - m) * (inp[index * C + i] - m);
        }
        float s = rsqrtf(sigma2/ C + 1e-5f);
        for (int i = 0; i < C; i++) {
            float n = s * (inp[index * C + i] - m);
            out[index * C + i] = n * weight[i] + bias[i];
        }

    }
}

// Launch kernel here
void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // Implement this
    const unsigned int numThreadsPerBlock = 256;
    const unsigned int numBlocks = (B * T + numThreadsPerBlock - 1) / numThreadsPerBlock;
    layernorm_forward_kernel<<<numBlocks, numThreadsPerBlock>>>(out, mean, rstd, inp, weight, bias, B, T, C);
}

#endif // __LAYERNORM_KERNEL_CUH__