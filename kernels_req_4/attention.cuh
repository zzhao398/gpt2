#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
#include "../utils/cuda_utils.cuh"
#include "../kernels/softmax.cuh"

#define BR 16
#define BC 16
#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define HS_fixed 64


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

__global__ void attention_kernel(float* vaccum, float* q, float* k, float* v, int B, int NH, int T,int HS, float inv_temperature) {
    const int T_r = (T + BR - 1) / BR;
    const int T_c = (T + BC - 1) / BC;
    const int b = blockIdx.x;
    const int nh = blockIdx.y;
    const int t_r = blockIdx.z;
    const int row_local = threadIdx.x;
    const int col_local = threadIdx.y;
    const int row = t_r * BR + row_local;

    __shared__ float q_shared[BR][HS_fixed];
    __shared__ float k_shared[BC][TILE_WIDTH];
    __shared__ float v_shared[BC][TILE_WIDTH];
    // we only need to store previous m and l but not entire history of T_c length
    __shared__ float m[BR];
    __shared__ float l[BR];
    __shared__ float prev_m[BR];
    __shared__ float att_shared[BR][BC];
    __shared__ float vaccum_shared[BR][HS_fixed];

    if (b < B && nh < NH && t_r < T_r) {

        // load q to sram and initialize vaccum_shared to 0
        for (int hs_tile = 0; hs_tile < HS; hs_tile += TILE_WIDTH) {
            const int hs = hs_tile + col_local;
            if (row < T && hs < HS) {
                q_shared[row_local][hs] = q[b * NH * T * HS + nh * T * HS + row * HS + hs];
                vaccum_shared[row_local][hs] = 0.0f;
            } else if (hs < HS_fixed) {
                q_shared[row_local][hs] = 0.0f;
                vaccum_shared[row_local][hs] = 0.0f;
            }
        }
        if (col_local == 0) {
            m[row_local] = -FLT_MAX;
            l[row_local] = 0.0f;
        }
        __syncthreads();

        for (int t_c = 0; t_c < T_c; t_c++) {
            att_shared[row_local][col_local] = 0.0f;

            // QK^T, but tiling K, since every K tile is used once
            for (int hs_tile = 0; hs_tile < HS; hs_tile += TILE_WIDTH) {
                const int hs = hs_tile + col_local;
                const int col = t_c * BC + row_local;
                if (col < T && hs < HS) {
                    k_shared[row_local][col_local] = k[b * NH * T * HS + nh * T * HS + col * HS + hs];
                } else {
                    k_shared[row_local][col_local] = 0.0f;
                }
                __syncthreads();

                for (int j = 0; j < TILE_WIDTH && hs_tile + j < HS; j++) {
                    att_shared[row_local][col_local] += q_shared[row_local][hs_tile + j] * k_shared[col_local][j];
                }
                __syncthreads();
            }

            if (col_local == 0) {
                float tile_max = -FLT_MAX;
                for (int j = 0; j < BC; j++) {
                    const int col = t_c * BC + j;
                    if (row < T && col < T && col <= row) {
                        tile_max = fmaxf(tile_max, att_shared[row_local][j]);
                    }
                }

                prev_m[row_local] = m[row_local];
                const float new_max = fmaxf(m[row_local], tile_max);
                float new_sum = 0.0f;
                if (l[row_local] > 0.0f && prev_m[row_local] > -FLT_MAX) {
                    new_sum = expf(inv_temperature * (prev_m[row_local] - new_max)) * l[row_local];
                }
                for (int j = 0; j < BC; j++) {
                    const int col = t_c * BC + j;
                    if (row < T && col < T && col <= row) {
                        new_sum += expf(inv_temperature * (att_shared[row_local][j] - new_max));
                    }
                }
                m[row_local] = new_max;
                l[row_local] = new_sum;
            }
            __syncthreads();

            const float rescale =
                (col_local == 0 && prev_m[row_local] > -FLT_MAX) ? expf(inv_temperature * (prev_m[row_local] - m[row_local])) : 0.0f;

            // tiling V to calculat PV
            for (int hs_tile = 0; hs_tile < HS; hs_tile += TILE_WIDTH) {
                const int hs = hs_tile + col_local;
                const int col = t_c * BC + row_local;
                if (col < T && hs < HS) {
                    v_shared[row_local][col_local] = v[b * NH * T * HS + nh * T * HS + col * HS + hs];
                } else {
                    v_shared[row_local][col_local] = 0.0f;
                }
                __syncthreads();

                if (col_local == 0 && row < T) {
                    for (int hs_offset = 0; hs_offset < TILE_WIDTH && hs_tile + hs_offset < HS; hs_offset++) {
                        float acc = vaccum_shared[row_local][hs_tile + hs_offset];
                        if (t_c > 0) {
                            acc *= rescale;
                        }
                        for (int j = 0; j < BC; j++) {
                            const int col = t_c * BC + j;
                            if (col < T && col <= row) {
                                const float p_tilde = expf(inv_temperature * (att_shared[row_local][j] - m[row_local]));
                                acc += p_tilde * v_shared[j][hs_offset];
                            }
                        }
                        vaccum_shared[row_local][hs_tile + hs_offset] = acc;
                    }
                }
                __syncthreads();
            }
        }

        if (col_local == 0 && row < T) {
            const float norm = l[row_local];
            for (int hs = 0; hs < HS; hs++) {
                vaccum[b * NH * T * HS + nh * T * HS + row * HS + hs] = vaccum_shared[row_local][hs] / norm;
            }
        }
    }
}


// Launch all kernels related to attention here 
void attention_forward(float* out, float* qkvr, float* att, float* inp, int B, int T, int C, int NH) {
    // Implement this
    const int HS = C / NH; // head size
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;


    const unsigned int numThreadsPerBlock = BLOCK_SIZE;
    const unsigned int Permute_numBlocks = (B * T * NH * HS + numThreadsPerBlock - 1) / numThreadsPerBlock;
    permute_kernel<<<Permute_numBlocks, numThreadsPerBlock>>>(q, k, v, inp, B, T, NH, HS);

    const int T_r = (T + BR - 1) / BR;
    const int T_c = (T + BC - 1) / BC;
    const float inv_temperature = 1.0f / sqrtf((float)HS);

    const dim3 attention_numBlocks(B, NH, T_r);
    const dim3 attention_blockSize(BR, BC);
    float* vaccum = att;
    attention_kernel<<<attention_numBlocks, attention_blockSize>>>(vaccum, q, k, v, B, NH, T, HS, inv_temperature);

    const unsigned int Unpermute_numBlocks = (B * T * NH * HS + numThreadsPerBlock- 1) / numThreadsPerBlock;    
    unpermute_kernel<<<Unpermute_numBlocks, numThreadsPerBlock>>>(vaccum, out, B, T, NH, HS);
}

#endif // __ATTENTION_CUH__
