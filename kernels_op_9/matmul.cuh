#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void matmul_forward_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                      const float* __restrict__ weight, const float* __restrict__ bias,
                                      int B, int T, int C, int OC) {
    // Implement this
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < B * T * OC) {
        int oc = index % OC;
        int t = (index / OC) % T;
        int b = index / (OC * T);
        float sum = 0.0f;
        for (int i = 0; i < C; i++) {
            sum += inp[b * T * C + t * C + i] * weight[oc * C + i];
        }
        if (bias)
            sum += bias[oc];
        out[index] = sum;
    }
}

// Launch kernel here
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // Implement this
    const unsigned int numThreadsPerBlock = 256;
    const unsigned int numBlocks = (B * T * OC + numThreadsPerBlock - 1) / numThreadsPerBlock;
    matmul_forward_kernel<<<numBlocks, numThreadsPerBlock>>>(out, inp, weight, bias, B, T, C, OC);
    
}

#endif // __MATMUL_KERNEL_CUH__