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
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * NH * T) return;

    int q_t = index % T;
    int nh = (index / T) % NH;
    int b = index / (T * NH);

    const int base = b * NH * T * HS + nh * T * HS;
    const int q_offset = base + q_t * HS;
    const int k_start = (q_t > WINDOW_SIZE) ? (q_t - WINDOW_SIZE) : 0;

    float maxval = -FLT_MAX;
    for (int k_t = k_start; k_t <= q_t; ++k_t) {
        float score = 0.0f;
        const int k_offset = base + k_t * HS;
        for (int hs = 0; hs < HS; ++hs) {
            score += q[q_offset + hs] * k[k_offset + hs];
        }
        maxval = fmaxf(maxval, score);
    }

    float sumval = 0.0f;
    for (int hs = 0; hs < HS; ++hs) {
        inp[q_offset + hs] = 0.0f;
    }

    // we compute score twice since thread level don't have enough resources to stor a weight array of 128 elements
    for (int k_t = k_start; k_t <= q_t; ++k_t) {
        float score = 0.0f;
        const int kv_offset = base + k_t * HS;
        for (int hs = 0; hs < HS; ++hs) {
            score += q[q_offset + hs] * k[kv_offset + hs];
        }

        float weight = expf(scale * (score - maxval));
        sumval += weight;
        for (int hs = 0; hs < HS; ++hs) {
            inp[q_offset + hs] += weight * v[kv_offset + hs];
        }
    }

    float inv_sum = 1.0f / sumval;
    for (int hs = 0; hs < HS; ++hs) {
        inp[q_offset + hs] *= inv_sum;
    }
}

void local_attention_forward_gpu(float* out, float* qkvr, float* inp, int B, int T, int NH, int HS) {
    float* q = qkvr + 0 * B * T * NH * HS;
    float* k = qkvr + 1 * B * T * NH * HS;
    float* v = qkvr + 2 * B * T * NH * HS;

    const int qkv_elems = B * T * NH * HS;
    const int out_elems = B * NH * T;
    const int num_blocks_qkv = (qkv_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_blocks_out = (out_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const float scale = 1.0f / sqrtf((float)HS);

    permute_kernel_local<<<num_blocks_qkv, BLOCK_SIZE>>>(q, k, v, inp, B, T, NH, HS);
    local_attention_kernel<<<num_blocks_out, BLOCK_SIZE>>>(inp, q, k, v, B, T, NH, HS, scale);
    unpermute_kernel_local<<<num_blocks_qkv, BLOCK_SIZE>>>(inp, out, B, T, NH, HS);
}


#endif // GPU_LOCAL_ATTENTION_CUH
