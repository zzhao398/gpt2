#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils/cuda_utils.cuh"

__global__ void add_bias_kernel(float* out, const float* bias, int BT, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BT * OC) return;
    out[idx] += bias[idx % OC];
}

//   out^T (OC, B*T) = weight (OC, C) @ inp^T (C, B*T)
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    int M = OC;
    int N = B * T;
    int K = C;
    float alpha = 1.0f, beta = 0.0f;

    cublasCheck(cublasSgemm(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        weight, K,
        inp,    K,
        &beta,
        out,    M));

    if (bias) {
        int total = B * T * OC;
        add_bias_kernel<<<(total + 255) / 256, 256>>>(out, bias, B * T, OC);
    }
}

#endif // __MATMUL_KERNEL_CUH__
