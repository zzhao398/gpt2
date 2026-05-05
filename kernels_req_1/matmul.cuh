#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include <mma.h>
#include "../utils/cuda_utils.cuh"

using namespace nvcuda;

#define WARP_SIZE 32

__global__ void weight_transpose_kernel(const float* weight, half* weight_transpose, int OC, int C) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int c = index % C;
    int oc = index / C;
    if (index < C * OC) {
        weight_transpose[c * OC + oc] = __float2half(weight[index]);
    }
}

__global__ void input_half_kernel(const float* inp, half* input_half, int B, int T, int C) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < B * T * C) {
        input_half[index] = __float2half(inp[index]);
    }
}

__global__ void matmul_forward_kernel(float* out, half* input_half, half* weight_transpose, 
                                      const float* bias, int B, int T, int C, int OC) {
    // Implement this
    // Input (B,T,C) @ Weight_transpose (C, OC) + Bias is (OC) = Output (B,T,OC)
    int warpId = threadIdx.x / 32;
    int in_warp_id = threadIdx.x % 32;
    int warpM_start = blockIdx.x * 128 + warpId * 16;
    int warpN_start = blockIdx.y * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> inp_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> weight_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_frag;

    wmma::fill_fragment(out_frag, 0.0f);
    // for padding reason, we still use shared memory
    // we know that gpt2-small model is 16-aligned
    // but we want our code handle general cases
    __shared__ half inp_s[8][16][16];
    __shared__ half weight_s[16][16];
    

    for (int i = 0; i < C; i += 16) {
        for (int j = 0; j < 8; j++) {
            if (warpM_start + in_warp_id / 2 < B * T && i + (in_warp_id % 2) * 8 + j < C)
                inp_s[warpId][in_warp_id / 2][(in_warp_id % 2) * 8 + j] = input_half[(warpM_start + in_warp_id / 2) * C + (i + (in_warp_id % 2) * 8 + j)];
            else 
                inp_s[warpId][in_warp_id / 2][(in_warp_id % 2) * 8 + j] = __float2half(0.0f);
        }

        if (i + threadIdx.x / 16 < C && warpN_start + threadIdx.x % 16 < OC)
            weight_s[threadIdx.x / 16][threadIdx.x % 16] = weight_transpose[(i + threadIdx.x / 16) * OC + warpN_start + threadIdx.x % 16];
        else 
            weight_s[threadIdx.x / 16][threadIdx.x % 16] = __float2half(0.0f); 

        __syncthreads();

        // Load the inputs
        wmma::load_matrix_sync(inp_frag, &inp_s[warpId][0][0], 16);
        wmma::load_matrix_sync(weight_frag, &weight_s[0][0], 16);

        // Perform the matrix multiplication
        wmma::mma_sync(out_frag, inp_frag, weight_frag, out_frag);
        __syncthreads();

    }

    __shared__ float tmp_s[8][16][16];
    wmma::store_matrix_sync(&tmp_s[warpId][0][0], out_frag, 16, wmma::mem_row_major);
    __syncthreads();
    for (int i = 0; i < 8; i++) {
        //(warpM_start + in_warp_id / 2) * OC + (warpN_start + (in_warp_id % 2) * 8 + i)
        if (warpM_start + in_warp_id / 2 < B * T && warpN_start + (in_warp_id % 2) * 8 + i < OC) {
            out[(warpM_start + in_warp_id / 2) * OC + (warpN_start + (in_warp_id % 2) * 8 + i)] = tmp_s[warpId][(in_warp_id / 2)][(in_warp_id % 2) * 8 + i];
            if (bias)
                out[(warpM_start + in_warp_id / 2) * OC + (warpN_start + (in_warp_id % 2) * 8 + i)] += bias[(warpN_start + (in_warp_id % 2) * 8 + i)];
        }
    }
}

// Launch kernel here
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // Implement this
    // Input (B,T,C) @ Weight (OC, C) + Bias is (OC) = Output (B,T,OC)
    int M = B * T;
    int K = C;
    int N = OC;

    half* weight_transpose;
    half* input_half;
    cudaMalloc(&weight_transpose, C * OC * sizeof(half));
    cudaMalloc(&input_half, B * T * C * sizeof(half));

    
    const unsigned int numThreadsPerBlock = 256;
    const int transpose_blocks = (C * OC - 1) / numThreadsPerBlock + 1;
    weight_transpose_kernel<<<transpose_blocks, numThreadsPerBlock>>> (weight, weight_transpose, OC, C);

    const int input_half_blocks = (B * T * C - 1) / numThreadsPerBlock + 1;
    input_half_kernel<<<input_half_blocks, numThreadsPerBlock>>>(inp, input_half, B, T, C);

    // 256 threads per block => 8 warps per block
    // the warp organization is 8 * 1
    // it handles 128 rows at BT dimension and 16 cols at OC dimention
    dim3 numBlocks((B * T - 1) / 128 + 1, (OC - 1) / 16 + 1);
    matmul_forward_kernel<<<numBlocks, numThreadsPerBlock>>>(out, input_half, weight_transpose, bias, B, T, C, OC);

    cudaFree(weight_transpose);
    cudaFree(input_half);
    
}

#endif // __MATMUL_KERNEL_CUH__