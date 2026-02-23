#ifndef RESIDUAL_KERNEL_CUH_
#define RESIDUAL_KERNEL_CUH_

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    // Implement this

}

// Launch kernel here
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    // Implement this
    
}

#endif // RESIDUAL_KERNEL_CUH_