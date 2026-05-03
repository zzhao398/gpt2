#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"
#include "../kernels_op_9/softmax.cuh"

__global__ void permute_kernel(float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
                               const float* __restrict__ inp, int B, int N, int NH, int d) {
    // Implement this
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * N * NH * d) return;
    int d_ = index % d;
    int n = (index / d) % N;
    int nh = (index / (d * N)) % NH;
    int b = (index / (d * N * NH));

    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh * d) + d_;
    q[index] = inp[inp_idx];
    k[index] = inp[inp_idx + NH * d];
    v[index] = inp[inp_idx + 2 * (NH * d)];
}


__global__ void unpermute_kernel(float* __restrict__ inp, float* __restrict__ out, int B, int N, int NH, int d) {
    // Implement this
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * N * NH * d) return ;
    int d_ = index % d;
    int nh = (index / d) % NH;
    int n = (index / (d * NH)) % N;
    int b = (index / (d * N * NH));
    out[index] = inp[b * NH * N * d + nh * N * d + n * d + d_];
}

__global__ void preatt_kernel(float* __restrict__ preatt, const float* __restrict__ k,
                              const float* __restrict__ q, int B, int NH, int T, int HS) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * NH * T * T) return;
    int t2 = index % T;
    int t1 = (index / T) % T;
    int nh = (index / (T * T)) % NH;
    int b = index / (T * T * NH);
    float sum = 0.0f;
    for (int hs = 0; hs < HS; hs++) {
        sum += k[b * NH * T * HS + nh * T * HS + t2 * HS + hs] * 
                q[b * NH * T * HS + nh * T * HS + t1 * HS + hs];
    }
    preatt[index] = sum;
}

__global__ void vaccum_kernel(float* __restrict__ vaccum, const float* __restrict__ att,
                              const float* __restrict__ v, int B, int NH, int T, int HS) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * NH * T * HS) return;
    int hs = index % HS;
    int t = (index / HS) % T;
    int nh = (index / (T * HS)) % NH;
    int b = index / (T * HS * NH);
    float sum = 0.0f;
    for (int t2 = 0; t2 < T; ++t2) {
        sum += att[b * NH * T * T + nh * T * T + t * T + t2] *
                v[b * NH * T * HS + nh * T * HS + t2 * HS + hs];
    }
    vaccum[index] = sum;
}

// Launch all kernels related to attention here 
void attention_forward(float* out, float* qkvr, float* att, float* inp, int B, int T, int C, int NH) {
    // Implement this
    int HS = C / NH; // head size
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;

    const unsigned int numThreadsPerBlock = 256;
    const unsigned int Permute_numBlocks = (B * T * NH * HS + numThreadsPerBlock - 1) / numThreadsPerBlock;
    permute_kernel<<<Permute_numBlocks, numThreadsPerBlock>>>(q, k, v, inp, B, T, NH, HS);

    float* preatt = inp;
    const unsigned int Preatt_numBlocks = (B * T * NH * T + numThreadsPerBlock - 1) / numThreadsPerBlock;
    preatt_kernel<<<Preatt_numBlocks, numThreadsPerBlock>>>(preatt, k, q, B, NH, T, HS);

    float scale = 1.0 / sqrtf(HS);
    const unsigned int Softmax_numBlocks = (B * T * NH + numThreadsPerBlock - 1) / numThreadsPerBlock;
    softmax_forward_kernel<<<Softmax_numBlocks, numThreadsPerBlock>>>(att, scale, preatt, B * NH, T);

    float* vaccum = inp;
    const unsigned int Vaccum_numBlocks = (B * T * NH * HS + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vaccum_kernel<<<Vaccum_numBlocks, numThreadsPerBlock>>>(vaccum, att, v, B, NH, T, HS);

    const unsigned int Unpermute_numBlocks = (B * T * NH * HS + numThreadsPerBlock- 1) / numThreadsPerBlock;    
    unpermute_kernel<<<Unpermute_numBlocks, numThreadsPerBlock>>>(vaccum, out, B, T, NH, HS);
}

#endif // __ATTENTION_CUH__