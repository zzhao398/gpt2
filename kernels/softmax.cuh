#ifndef __SOFTMAX_KERNEL_CUH__
#define __SOFTMAX_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "../utils/cuda_utils.cuh"

__global__ void softmax_forward_kernel(float* out, float inv_temperature, const float* inp, int N, int T) {
    // Implement this

}

#endif // __SOFTMAX_KERNEL_CUH__