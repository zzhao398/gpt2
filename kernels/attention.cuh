#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils/cuda_utils.cuh"
#include "../kernels/softmax.cuh"

// req_2: cuBLAS StridedBatched for Q*K^T and Att*V, reusing global cublas_handle.
// softmax uses req_3 parallel-reduction kernel (one block per row).

__global__ void permute_kernel(float* q, float* k, float* v, const float* inp, int B, int N, int NH, int d) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * N * NH * d) return;
    int d_ = index % d;
    int n  = (index / d) % N;
    int nh = (index / (d * N)) % NH;
    int b  = (index / (d * N * NH));

    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh * d) + d_;
    q[index] = inp[inp_idx];
    k[index] = inp[inp_idx + NH * d];
    v[index] = inp[inp_idx + 2 * (NH * d)];
}

__global__ void unpermute_kernel(float* inp, float* out, int B, int N, int NH, int d) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * N * NH * d) return;
    int d_ = index % d;
    int nh = (index / d) % NH;
    int n  = (index / (d * NH)) % N;
    int b  = (index / (d * N * NH));
    out[index] = inp[b * NH * N * d + nh * N * d + n * d + d_];
}

void attention_forward(float* out, float* qkvr, float* att, float* inp, int B, int T, int C, int NH) {
    int HS = C / NH;
    float *q = qkvr + 0 * B * T * C;
    float *k = qkvr + 1 * B * T * C;
    float *v = qkvr + 2 * B * T * C;

    const int numThreadsPerBlock = 256;
    const int Permute_numBlocks = (B * T * NH * HS + numThreadsPerBlock - 1) / numThreadsPerBlock;
    permute_kernel<<<Permute_numBlocks, numThreadsPerBlock>>>(q, k, v, inp, B, T, NH, HS);

    // Q*K^T: (B*NH) batches of (T,HS) x (HS,T) -> (T,T)
    float* preatt = inp;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                         CUBLAS_OP_T, CUBLAS_OP_N,
                                         T, T, HS,
                                         &alpha,
                                         k, HS, T * HS,
                                         q, HS, T * HS,
                                         &beta,
                                         preatt, T, T * T,
                                         B * NH));

    // softmax (req_3: one block per row for parallel reduction)
    float scale = 1.0f / sqrtf((float)HS);
    softmax_forward_kernel<<<B * NH * T, numThreadsPerBlock>>>(att, scale, preatt, B * NH, T);

    // Att*V: (B*NH) batches of (T,T) x (T,HS) -> (T,HS)
    float* vaccum = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         HS, T, T,
                                         &alpha,
                                         v,   HS, T * HS,
                                         att, T,  T * T,
                                         &beta,
                                         vaccum, HS, T * HS,
                                         B * NH));

    const int Unpermute_numBlocks = (B * T * NH * HS + numThreadsPerBlock - 1) / numThreadsPerBlock;
    unpermute_kernel<<<Unpermute_numBlocks, numThreadsPerBlock>>>(vaccum, out, B, T, NH, HS);
}

#endif // __ATTENTION_CUH__
