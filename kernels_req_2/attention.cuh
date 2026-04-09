#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils/cuda_utils.cuh"
#include "../kernels/softmax.cuh"

__global__ void permute_kernel(float* q, float* k, float* v, const float* inp, int B, int N, int NH, int d) {
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


__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {
    // Implement this
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * N * NH * d) return ;
    int d_ = index % d;
    int nh = (index / d) % NH;
    int n = (index / (d * NH)) % N;
    int b = (index / (d * N * NH));
    out[index] = inp[b * NH * N * d + nh * N * d + n * d + d_];
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
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));
    int batchCount = B * NH;
    int M = T;
    int K = HS;
    int N = T;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha, k, HS,
                                  T * HS, q, HS,
                                  T * HS, &beta,
                                  preatt, T,
                                  T * T, batchCount));

    float scale = 1.0 / sqrtf(HS);
    const unsigned int Softmax_numBlocks = (B * T * NH + numThreadsPerBlock - 1) / numThreadsPerBlock;
    softmax_forward_kernel<<<Softmax_numBlocks, numThreadsPerBlock>>>(att, scale, preatt, B * NH, T);

    float* vaccum = inp;
    M = T;
    K = T;
    N = HS;
    cublasCheck(cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K, &alpha, v, HS,
                                  T * HS, att, T,
                                  T * T, &beta, vaccum, HS,
                                  T * HS, batchCount));


    cublasCheck(cublasDestroy(handle));
    

    const unsigned int Unpermute_numBlocks = (B * T * NH * HS + numThreadsPerBlock- 1) / numThreadsPerBlock;    
    unpermute_kernel<<<Unpermute_numBlocks, numThreadsPerBlock>>>(vaccum, out, B, T, NH, HS);
}

#endif // __ATTENTION_CUH__