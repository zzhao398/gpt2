#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils/cuda_utils.cuh"


__global__ void bias_kernel(const float* bias, float* out, int B, int T, int OC) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < B * T * OC) {
        int oc = index % OC;
        out[index] += bias[oc];
    }
}


// Launch kernel here
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // Input is row-major (B * T, C), weight is row-major (OC, C),
    // and cuBLAS expects column-major. We compute out^T = weight * inp^T.
    int M = B * T;
    int K = C;
    int N = OC;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    cublasCheck(cublasSgemm(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            weight, K,
                            inp, K,
                            &beta,
                            out, N));

    if (bias) {
        int numThreadsPerBlock = 256;
        int numBlocks = (B * T * OC - 1) / numThreadsPerBlock + 1;
        bias_kernel<<<numBlocks, numThreadsPerBlock>>>(bias, out, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
    cublasCheck(cublasDestroy(handle));
    
}

#endif // __MATMUL_KERNEL_CUH__
