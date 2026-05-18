#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils/cuda_utils.cuh"

// req_2: cuBLAS Sgemm, reuses the global cublas_handle from cuda_utils.cuh
// to avoid per-call cublasCreate/Destroy overhead.

__global__ void bias_kernel(const float* bias, float* out, int B, int T, int OC) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < B * T * OC) {
        int oc = index % OC;
        out[index] += bias[oc];
    }
}

void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasCheck(cublasSgemm(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            OC, B * T, C,
                            &alpha,
                            weight, C,
                            inp,    C,
                            &beta,
                            out,    OC));

    if (bias) {
        const int numThreadsPerBlock = 256;
        const int numBlocks = (B * T * OC - 1) / numThreadsPerBlock + 1;
        bias_kernel<<<numBlocks, numThreadsPerBlock>>>(bias, out, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}

#endif // __MATMUL_KERNEL_CUH__
