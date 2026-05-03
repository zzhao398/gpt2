#ifndef __SOFTMAX_KERNEL_CUH__
#define __SOFTMAX_KERNEL_CUH__

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "../utils/cuda_utils.cuh"

__global__ void softmax_forward_kernel(float* __restrict__ out, float inv_temperature,
                                       const float* __restrict__ inp, int N, int T) {
    // Implement this
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= N * T) return;
    int own_pos = index % T;
    const float* x = inp + index * T;

    float maxval = -FLT_MAX;
    // Think about what this for loop condition is doing (Note the significance of i <= own_pos)
    for (int i = 0; i <= own_pos; ++i) {
        maxval = fmaxf(maxval, x[i]);
    }

    // Compute softmax
    float sumval = 0.0f;
    for (int i = 0; i <= own_pos; ++i) {
            float ev = expf(inv_temperature * (x[i] - maxval));
            sumval += ev;
            out[index * T + i] = ev;
    }
    // Normalize
    float norm = 1.0f / sumval;
    for (int i = 0; i <= own_pos; ++i) {
        out[index * T + i] *= norm;
    }
}

#endif // __SOFTMAX_KERNEL_CUH__