#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include <mma.h>
#include "../utils/cuda_utils.cuh"
#include "../kernels/softmax.cuh"

using namespace nvcuda;

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
    // Attention matmul: Q (B, NH, T, HS) @ K^T, K is(B, NH, T, HS)
    // Compute pre-attention scores (B, NH, T, T)
    int warpId = threadIdx.x / 32;
    int in_warp_id = threadIdx.x % 32;
    int b_nh = blockIdx.x;
    int t1_start = blockIdx.y * 128 + warpId * 16;
    int t2_start = blockIdx.z * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> K_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> preatt_frag;

    wmma::fill_fragment(preatt_frag, 0.0f);
    __shared__ half Q_s[8][16][16];
    __shared__ half K_s[16][16];
    __shared__ float tmp_s[8][16][16];

    for (int i = 0; i < HS; i += 16) {
        for (int j = 0; j < 8; j++) {
            if (b_nh < B * NH && t1_start + (in_warp_id / 2) < T && i + (in_warp_id % 2) * 8 + j < HS)
                Q_s[warpId][in_warp_id / 2][(in_warp_id % 2) * 8 + j] = __float2half(q[b_nh * T * HS + (t1_start + (in_warp_id / 2)) * HS + i + (in_warp_id % 2) * 8 + j]);
            else
                Q_s[warpId][in_warp_id / 2][(in_warp_id % 2) * 8 + j] = __float2half(0.0f);
        }


        if (b_nh < B * NH && t2_start + threadIdx.x / 16 < T && i + threadIdx.x % 16 < HS) {
            K_s[threadIdx.x / 16][threadIdx.x % 16] = __float2half(k[b_nh * T * HS + (t2_start + threadIdx.x / 16) * HS + i + threadIdx.x % 16]);
        } else {
            K_s[threadIdx.x / 16][threadIdx.x % 16] = __float2half(0.0f);
        }
        __syncthreads();

        // Load the inputs
        wmma::load_matrix_sync(Q_frag, &Q_s[warpId][0][0], 16);
        wmma::load_matrix_sync(K_frag, &K_s[0][0], 16);

        // Perform the matrix multiplication
        wmma::mma_sync(preatt_frag, Q_frag, K_frag, preatt_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(&tmp_s[warpId][0][0], preatt_frag, 16, wmma::mem_row_major);
    __syncthreads();
    for (int i = 0; i < 8; i++) {
        if (b_nh < B * NH && t1_start + (in_warp_id / 2) < T && t2_start + (in_warp_id % 2) * 8 + i < T) {
            preatt[b_nh * T * T + (t1_start + (in_warp_id / 2)) * T + t2_start + (in_warp_id % 2) * 8 + i] = tmp_s[warpId][in_warp_id / 2][(in_warp_id % 2) * 8 + i];
        }
    }
}

__global__ void vaccum_kernel(float* vaccum, float* att, float* v, int B, int NH, int T, int HS) {
    // Attention matmul: P @ V, where P holds the attention probabilities
    // (B, NH, T, T) @ (B, NH, T, HS) -> (B, NH, T, HS)
    int warpId = threadIdx.x / 32;
    int in_warp_id = threadIdx.x % 32;
    int b_nh = blockIdx.x;
    int t1_start = blockIdx.y * 128 + warpId * 16;
    int hs_start = blockIdx.z * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Att_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> V_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> vaccum_frag;

    wmma::fill_fragment(vaccum_frag, 0.0f);
    __shared__ half att_s[8][16][16];
    __shared__ half v_s[16][16];
    __shared__ float tmp_s[8][16][16];
    for (int i = 0; i < T; i += 16) {
        for (int j = 0; j < 8; j++) {
            if (b_nh < B * NH && t1_start + (in_warp_id / 2) < T && i + (in_warp_id % 2) * 8 + j < T) {
                att_s[warpId][in_warp_id / 2][(in_warp_id % 2) * 8 + j] = __float2half(att[b_nh * T * T + (t1_start + in_warp_id / 2) * T + i + (in_warp_id % 2) * 8 + j]);
            } else {
                att_s[warpId][in_warp_id / 2][(in_warp_id % 2) * 8 + j] = __float2half(0.0f);
            }
        }

        if (i + threadIdx.x / 16 < T && hs_start + threadIdx.x % 16 < HS) {
            v_s[threadIdx.x / 16][threadIdx.x % 16] = __float2half(v[b_nh * T * HS + (i + threadIdx.x / 16) * HS + hs_start + threadIdx.x % 16]);
        } else {
            v_s[threadIdx.x / 16][threadIdx.x % 16] = __float2half(0.0f);
        }
        __syncthreads();

        // Load the inputs
        wmma::load_matrix_sync(Att_frag, &att_s[warpId][0][0], 16);
        wmma::load_matrix_sync(V_frag, &v_s[0][0], 16);

        // Perform the matrix multiplication
        wmma::mma_sync(vaccum_frag, Att_frag, V_frag, vaccum_frag);
        __syncthreads();

    }

    wmma::store_matrix_sync(&tmp_s[warpId][0][0], vaccum_frag, 16, wmma::mem_row_major);
    __syncthreads();
    for (int i = 0; i < 8; i++) {
        if (b_nh < B * NH && t1_start + in_warp_id / 2 < T && hs_start + (in_warp_id % 2) * 8 + i < HS) {
            vaccum[b_nh * T * HS + (t1_start + in_warp_id / 2) * HS + hs_start + (in_warp_id % 2) * 8 + i] = tmp_s[warpId][in_warp_id / 2][(in_warp_id % 2) * 8 + i];
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

    const unsigned int numThreadsPerBlock = 256;
    const unsigned int Permute_numBlocks = (B * T * NH * HS + numThreadsPerBlock - 1) / numThreadsPerBlock;
    permute_kernel<<<Permute_numBlocks, numThreadsPerBlock>>>(q, k, v, inp, B, T, NH, HS);

    float* preatt = inp;
    dim3 Preatt_numBlocks(B * NH, (T - 1) / 128 + 1, (T - 1) / 16 + 1);
    preatt_kernel<<<Preatt_numBlocks, numThreadsPerBlock>>>(preatt, k, q, B, NH, T, HS);

    float scale = 1.0 / sqrtf(HS);
    const unsigned int Softmax_numBlocks = (B * T * NH + numThreadsPerBlock - 1) / numThreadsPerBlock;
    softmax_forward_kernel<<<Softmax_numBlocks, numThreadsPerBlock>>>(att, scale, preatt, B * NH, T);

    float* vaccum = inp;
    dim3 Vaccum_numBlocks(B * NH, (T - 1) / 128 + 1, (HS - 1) / 16 + 1);
    vaccum_kernel<<<Vaccum_numBlocks, numThreadsPerBlock>>>(vaccum, att, v, B, NH, T, HS);

    const unsigned int Unpermute_numBlocks = (B * T * NH * HS + numThreadsPerBlock- 1) / numThreadsPerBlock;    
    unpermute_kernel<<<Unpermute_numBlocks, numThreadsPerBlock>>>(vaccum, out, B, T, NH, HS);
}

#endif // __ATTENTION_CUH__