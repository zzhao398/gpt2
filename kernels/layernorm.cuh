#ifndef __LAYERNORM_KERNEL_CUH__
#define __LAYERNORM_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "../utils/cuda_utils.cuh"

__global__ void layernorm_forward_kernel(float* out, float* mean, float* rstd, const float* inp, const float* weight,
                                         const float* bias, int B, int T, int C) {
    // Implement this

}

// Launch kernel here
void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // Implement this
    
}

#endif // __LAYERNORM_KERNEL_CUH__