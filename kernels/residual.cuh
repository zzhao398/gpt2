#ifndef RESIDUAL_KERNEL_CUH_
#define RESIDUAL_KERNEL_CUH_

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    // Implement this
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {        
        out[index] = inp1[index] + inp2[index];
    }
}

// Launch kernel here
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    // Implement this
    const unsigned int numThreadsPerBlock = 256;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    residual_forward_kernel<<<numBlocks, numThreadsPerBlock>>>(out, inp1, inp2, N);
}

#endif // RESIDUAL_KERNEL_CUH_