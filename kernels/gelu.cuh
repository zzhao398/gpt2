#ifndef GELU_KERNEL_CUH_
#define GELU_KERNEL_CUH_

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    // Implement this
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        float val = inp[index];
        float cube = 0.044715f * val * val * val;
        // Think about what GELU is, are we computing the exact gelu function here?
        out[index] = 0.5f * val * (1.0f + tanhf(GELU_SCALING_FACTOR * (val + cube)));
    }
}

// Launch kernel here
void gelu_forward(float* out, const float* inp, int N) {
    // Implement this
    const unsigned int numThreadsPerBlock = 256;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    gelu_forward_kernel<<<numBlocks, numThreadsPerBlock>>>(out, inp, N);
}

#endif // GELU_KERNEL_CUH_