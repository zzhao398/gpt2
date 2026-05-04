#ifndef GPU_LOCAL_ATTENTION_CUH
#define GPU_LOCAL_ATTENTION_CUH

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#include "../utils/cuda_utils.cuh"

#define BLOCK_SIZE 256
#define WINDOW_SIZE 128

__global__ void permute_kernel_local(float* q, float* k, float* v, const float* inp, int B, int T, int NH, int HS) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * T * NH * HS) return;

    int hs = index % HS;
    int t = (index / HS) % T;
    int nh = (index / (HS * T)) % NH;
    int b = index / (HS * T * NH);

    int inp_idx = b * T * 3 * NH * HS + t * 3 * NH * HS + nh * HS + hs;
    q[index] = inp[inp_idx];
    k[index] = inp[inp_idx + NH * HS];
    v[index] = inp[inp_idx + 2 * NH * HS];
}

__global__ void unpermute_kernel_local(const float* inp, float* out, int B, int T, int NH, int HS) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * T * NH * HS) return;

    int hs = index % HS;
    int nh = (index / HS) % NH;
    int t = (index / (HS * NH)) % T;
    int b = index / (HS * NH * T);

    out[index] = inp[b * NH * T * HS + nh * T * HS + t * HS + hs];
}

__global__ void local_attention_kernel(float* inp, const float* q, const float* k, const float* v,
                                       int B, int T, int NH, int HS, float scale) {
    int bnhq = blockIdx.x;
    if (bnhq >= B * NH * T || threadIdx.x >= WINDOW_SIZE + 1) return;

    // local_k = 128 => k_t = q_t 
    int local_k = threadIdx.x;
    int q_t = bnhq % T;
    int nh = (bnhq / T) % NH;
    int b = bnhq / (T * NH);

    __shared__ float max_scores[BLOCK_SIZE];
    __shared__ float sum_weights[BLOCK_SIZE];

    max_scores[threadIdx.x] = -FLT_MAX;
    sum_weights[threadIdx.x] = 0.0f;
    __syncthreads();

    const int base = b * NH * T * HS + nh * T * HS;
    const int q_offset = base + q_t * HS;
    const int k_start = (q_t >= WINDOW_SIZE) ? 0 : (WINDOW_SIZE - q_t);

    float score = 0.0f;
    if (local_k >= k_start) {
        
        const int k_offset = base + (q_t - WINDOW_SIZE + local_k) * HS;
        for (int hs = 0; hs < HS; ++hs) {
            score += q[q_offset + hs] * k[k_offset + hs];
        }
        max_scores[local_k] = score;
    }

    __syncthreads();

    float maxval = -FLT_MAX;
    for (int k_t = k_start; k_t <= WINDOW_SIZE; ++k_t) {
        maxval = fmaxf(maxval, max_scores[k_t]);
    }

    if (threadIdx.x == 0) {
        for (int hs = 0; hs < HS; ++hs) {
            inp[q_offset + hs] = 0.0f;
        }
    }
    float sumval = 0.0f;
    __syncthreads();

    if (local_k >= k_start) {
        float weight = expf(scale * (score - maxval));
        sum_weights[local_k] = weight;
        for (int hs = 0; hs < HS; ++hs) {
            atomicAdd(&inp[q_offset + hs], weight * v[base + (q_t - WINDOW_SIZE + local_k) * HS + hs]);
        }
    }
    __syncthreads();

    for (int k_t = k_start; k_t <= local_k; ++k_t) {
        sumval += sum_weights[k_t];
    }

    if (local_k == 128) {
        float inv_sum = 1.0f / sumval;
        for (int hs = 0; hs < HS; ++hs) {
            inp[q_offset + hs] *= inv_sum;
        }
    }
}

void local_attention_forward_gpu(float* out, float* qkvr, float* inp, int B, int T, int NH, int HS) {
    float* q = qkvr + 0 * B * T * NH * HS;
    float* k = qkvr + 1 * B * T * NH * HS;
    float* v = qkvr + 2 * B * T * NH * HS;

    const int qkv_elems = B * T * NH * HS;
    const int out_elems = B * NH * T ;
    const int num_blocks_qkv = (qkv_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_blocks_out = out_elems;
    const float scale = 1.0f / sqrtf((float)HS);

    permute_kernel_local<<<num_blocks_qkv, BLOCK_SIZE>>>(q, k, v, inp, B, T, NH, HS);
    local_attention_kernel<<<num_blocks_out, WINDOW_SIZE + 1>>>(inp, q, k, v, B, T, NH, HS, scale);
    unpermute_kernel_local<<<num_blocks_qkv, BLOCK_SIZE>>>(inp, out, B, T, NH, HS);
}


#endif // GPU_LOCAL_ATTENTION_CUH
