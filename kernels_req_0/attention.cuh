#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"
#include "../kernels/softmax.cuh"

#define U 16
#define TILE_WIDTH 16
#define BLOCK_SIZE 256

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

__global__ void preatt_kernel(float* preatt, float* k, float* q, int B, int NH, int T, int HS) {
    // Attention matmul: Q @ K^T
    // Compute pre-attention scores (B, NH, T, T)
    int b_nh = blockIdx.x;
    int t_1 = blockIdx.y * blockDim.x + threadIdx.x;
    int t2_start = blockIdx.z * U;
    int hs_offset = threadIdx.x / U;
    int t2_offset = threadIdx.x % U;
    float sum[U];
    for (int i = 0; i < U; i++) {
        sum[i] = 0.0f;
    }
    __shared__ float q_s[BLOCK_SIZE][TILE_WIDTH];
    __shared__ float k_s[TILE_WIDTH][U];
    for (int i = 0; i < (HS - 1) / TILE_WIDTH + 1; i++) {
        if (t_1 < T) {
            for (int j = 0; j < TILE_WIDTH; j++) {
                if (i * TILE_WIDTH + j < HS) {
                    q_s[threadIdx.x][j] = q[b_nh * T * HS + t_1 * HS + i * TILE_WIDTH + j];
                }
                else
                    q_s[threadIdx.x][j] = 0.0f;
            }
        } else {
            for (int j = 0; j < TILE_WIDTH; j++) {
                q_s[threadIdx.x][j] = 0.0f;
            }
        }
        if (i * TILE_WIDTH + hs_offset >= HS || t2_start + t2_offset >= T)
            k_s[hs_offset][t2_offset] = 0.0f;
        else 
            k_s[hs_offset][t2_offset] = k[b_nh * T * HS + (t2_start + t2_offset) * HS + i * TILE_WIDTH + hs_offset];

        __syncthreads();
        for (int j = 0; j < U; j++) {
            for (int jj = 0; jj < TILE_WIDTH; jj++) {
                sum[j] += q_s[threadIdx.x][jj] * k_s[jj][j];
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < U; i++) {
        if (t_1 < T && t2_start + i < T)
            preatt[b_nh * T * T + t_1 * T + t2_start + i] = sum[i];
    }
}

__global__ void vaccum_kernel(float* vaccum, float* att, float* v, int B, int NH, int T, int HS) {

    // Attention matmul: P @ V, where P holds the attention probabilities
    // (B, NH, T, T) @ (B, NH, T, HS) -> (B, NH, T, HS)
    int b_nh = blockIdx.x;
    int t1 = blockIdx.y * blockDim.x + threadIdx.x;
    int hs_start = blockIdx.z * U;
    int t2_offset = threadIdx.x / U;
    int hs_offset = threadIdx.x % U;
    float sum[U];
    for (int i = 0; i < U; i++) {
        sum[i] = 0.0f;
    }
    __shared__ float att_s[BLOCK_SIZE][TILE_WIDTH];
    __shared__ float v_s[TILE_WIDTH][U];
    for (int i = 0; i < (T - 1) / TILE_WIDTH + 1; i++) {
        if (t1 < T) {
            for (int j = 0; j < TILE_WIDTH; j++) {
                if (i * TILE_WIDTH + j < T) {
                    att_s[threadIdx.x][j] = att[b_nh * T * T + t1 * T + i * TILE_WIDTH + j];
                } else {
                    att_s[threadIdx.x][j] = 0.0f;
                }
            }
        } else {
            for (int j = 0; j < TILE_WIDTH; j++) {
                att_s[threadIdx.x][j] = 0.0f;
            }
        }
        if (hs_start + hs_offset >= HS || i * TILE_WIDTH + t2_offset >= T) {
            v_s[t2_offset][hs_offset] = 0.0f;
        } else {
            v_s[t2_offset][hs_offset] = v[b_nh * T * HS + (i * TILE_WIDTH + t2_offset) * HS + hs_start + hs_offset];
        }
        __syncthreads();
        for (int j = 0; j < U; j++) {
            for (int jj = 0; jj < TILE_WIDTH; jj++) {
                sum[j] += att_s[threadIdx.x][jj] * v_s[jj][j];
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < U; i++) {
        if (t1 < T && hs_start + i < HS) {
            vaccum[b_nh * T * HS + t1 * HS + hs_start + i] = sum[i];
        }
    }
}

// Launch all kernels related to attention here 
void attention_forward(float* out, float* qkvr, float* att, float* inp, int B, int T, int C, int NH) {
    // Implement this
    int HS = C / NH; // head size
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;

    const unsigned int numThreadsPerBlock = BLOCK_SIZE;
    const unsigned int Permute_numBlocks = (B * T * NH * HS + numThreadsPerBlock - 1) / numThreadsPerBlock;
    permute_kernel<<<Permute_numBlocks, numThreadsPerBlock>>>(q, k, v, inp, B, T, NH, HS);

    float* preatt = inp;
    dim3 Preatt_numBlocks(B * NH, (T - 1) / numThreadsPerBlock + 1, (T - 1) / U + 1);
    preatt_kernel<<<Preatt_numBlocks, numThreadsPerBlock>>>(preatt, k, q, B, NH, T, HS);

    float scale = 1.0 / sqrtf(HS);
    const unsigned int Softmax_numBlocks = (B * T * NH + numThreadsPerBlock - 1) / numThreadsPerBlock;
    softmax_forward_kernel<<<Softmax_numBlocks, numThreadsPerBlock>>>(att, scale, preatt, B * NH, T);

    float* vaccum = inp;
    dim3 Vaccum_numBlocks(B * NH, (T - 1) / numThreadsPerBlock + 1, (HS - 1) / U + 1);
    vaccum_kernel<<<Vaccum_numBlocks, numThreadsPerBlock>>>(vaccum, att, v, B, NH, T, HS);

    const unsigned int Unpermute_numBlocks = (B * T * NH * HS + numThreadsPerBlock- 1) / numThreadsPerBlock;    
    unpermute_kernel<<<Unpermute_numBlocks, numThreadsPerBlock>>>(vaccum, out, B, T, NH, HS);
}

#endif // __ATTENTION_CUH__