#ifndef GELU_KERNEL_CUH_
#define GELU_KERNEL_CUH_

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    // Implement this

}

// Launch kernel here
void gelu_forward(float* out, const float* inp, int N) {
    // Implement this
    
}

#endif // GELU_KERNEL_CUH_