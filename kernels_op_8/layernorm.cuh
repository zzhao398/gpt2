#ifndef __LAYERNORM_KERNEL_CUH__
#define __LAYERNORM_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include "../utils/cuda_utils.cuh"


#define LAYERNORM_MAX 1600
__constant__ float cons_layernorm_weight[LAYERNORM_MAX];
__constant__ float cons_layernorm_bias[LAYERNORM_MAX];

__global__ void layernorm_forward_kernel(float* out, float* mean, float* rstd, const float* inp,
                                         int B, int T, int C) {
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
        float s = rsqrtf(sigma2 / C + 1e-5f);
        for (int i = 0; i < C; i++) {
            float n = s * (inp[index * C + i] - m);
            out[index * C + i] = n * cons_layernorm_weight[i] + cons_layernorm_bias[i];
        }
    }
}


void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    assert(C <= LAYERNORM_MAX && "op_8 layernorm: C exceeds LAYERNORM_MAX");
    cudaCheck(cudaMemcpyToSymbol(cons_layernorm_weight, weight, C * sizeof(float),
                                 0, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpyToSymbol(cons_layernorm_bias, bias, C * sizeof(float),
                                 0, cudaMemcpyDeviceToDevice));
    const unsigned int numThreadsPerBlock = 256;
    const unsigned int numBlocks = (B * T + numThreadsPerBlock - 1) / numThreadsPerBlock;
    layernorm_forward_kernel<<<numBlocks, numThreadsPerBlock>>>(out, mean, rstd, inp, B, T, C);
}

#endif // __LAYERNORM_KERNEL_CUH__
