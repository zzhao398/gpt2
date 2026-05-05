#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

#define U 16
#define BLOCK_SIZE 256
#define TILE_WIDTH 16

__global__ void matmul_forward_kernel(float* out, const float* inp, const float* weight, 
                                      const float* bias, int B, int T, int C, int OC) {
    // Implement this
    // Input (B,T,C) @ Weight (OC, C) + Bias is (OC) = Output (B,T,OC)
    int bt = blockDim.x * blockIdx.x + threadIdx.x;
    int oc_start = blockIdx.y * U;
    // oc_offset and c_offset are used for weight_s memory mapping
    // it has no actual meaning
    int oc_offset = threadIdx.x % U;
    int C_offset = threadIdx.x / U;
    __shared__ float input_s[BLOCK_SIZE][TILE_WIDTH];
    __shared__ float weight_s[TILE_WIDTH][U];
    float sum[U];
    for (int i = 0; i < U; i++) {
        sum[i] = 0.0f;
    }
    for (int i = 0; i < (C - 1) / TILE_WIDTH + 1; i++) {
        if (bt < B * T) {
            for (int j = 0; j < TILE_WIDTH; j++) {
                if (i * TILE_WIDTH + j < C)
                    input_s[threadIdx.x][j] = inp[bt * C + i * TILE_WIDTH + j];
                else
                    input_s[threadIdx.x][j] = 0.0f;
            }
        } else {
            for (int j = 0; j < TILE_WIDTH; j++) {
                input_s[threadIdx.x][j] = 0.0f;
            }
        }
        if (i * TILE_WIDTH + C_offset >= C || oc_offset + oc_start >= OC) {
            weight_s[C_offset][oc_offset] = 0.0f;
        } else {
            weight_s[C_offset][oc_offset] = weight[(oc_offset + oc_start) * C + i * TILE_WIDTH + C_offset];
        }
        __syncthreads();

        for (int j = 0; j < U; j++) {
            for (int k = 0; k < TILE_WIDTH; k++) {
                sum[j] += input_s[threadIdx.x][k] * weight_s[k][j];
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < U; i++) {
        if (bias && oc_start + i < OC)
            sum[i] += bias[oc_start + i];
        if (bt < B * T && oc_start + i < OC)
            out[bt * OC + oc_start + i] = sum[i];
    }
}

// Launch kernel here
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // Implement this
    const unsigned int numThreadsPerBlock = BLOCK_SIZE;
    dim3 numBlocks((B * T - 1) / numThreadsPerBlock + 1, (OC - 1) / U + 1);
    matmul_forward_kernel<<<numBlocks, numThreadsPerBlock>>>(out, inp, weight, bias, B, T, C, OC);
    
}

#endif // __MATMUL_KERNEL_CUH__