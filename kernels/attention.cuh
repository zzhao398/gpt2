#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void permute_kernel(float* q, float* k, float* v, const float* inp, int B, int N, int NH, int d) {
    // Implement this

}


__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {
    // Implement this

}


// Launch all kernels related to attention here 
void attention_forward(float* out, float* qkvr, float* att, float* inp, int B, int T, int C, int NH) {
    // Implement this
    
}

#endif // __ATTENTION_CUH__