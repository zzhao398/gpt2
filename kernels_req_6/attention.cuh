#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils/cuda_utils.cuh"
#include "../kernels/softmax.cuh"

extern bool g_enable_kv_cache; // Toggles KV-caching logic
extern float* g_kv_cache;      // Pointer to the GPU-allocated KV cache buffer
extern int g_layer_idx;        // Tracks the current transformer layer, useful for knowing when to switch to decode
extern bool g_is_prefill;      // True during the initial prompt, False during token generation
extern int g_current_pos;      // Tracks the absolute sequence position during Decode, helps index into the KV cache

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
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * NH * T * T) return;
    int t2 = index % T;
    int t1 = (index / T) % T;
    int nh = (index / (T * T)) % NH;
    int b = index / (T * T * NH);
    float sum = 0.0f;
    for (int hs = 0; hs < HS; hs++) {
        sum += k[b * NH * T * HS + nh * T * HS + t2 * HS + hs] * 
                q[b * NH * T * HS + nh * T * HS + t1 * HS + hs];
    }
    preatt[index] = sum;
}

__global__ void vaccum_kernel(float* vaccum, float* att, float* v, int B, int NH, int T, int HS) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * NH * T * HS) return;
    int hs = index % HS;
    int t = (index / HS) % T;
    int nh = (index / (T * HS)) % NH;
    int b = index / (T * HS * NH);
    float sum = 0.0f;
    for (int t2 = 0; t2 < T; ++t2) {
        sum += att[b * NH * T * T + nh * T * T + t * T + t2] *
                v[b * NH * T * HS + nh * T * HS + t2 * HS + hs];
    }
    vaccum[index] = sum;
}

// Launch all kernels related to attention here 
void baseline_attention_forward(float* out, float* qkvr, float* att, float* inp, int B, int T, int C, int NH) {
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
    const unsigned int Preatt_numBlocks = (B * T * NH * T + numThreadsPerBlock - 1) / numThreadsPerBlock;
    preatt_kernel<<<Preatt_numBlocks, numThreadsPerBlock>>>(preatt, k, q, B, NH, T, HS);

    float scale = 1.0 / sqrtf(HS);
    const unsigned int Softmax_numBlocks = (B * T * NH + numThreadsPerBlock - 1) / numThreadsPerBlock;
    softmax_forward_kernel<<<Softmax_numBlocks, numThreadsPerBlock>>>(att, scale, preatt, B * NH, T);

    float* vaccum = inp;
    const unsigned int Vaccum_numBlocks = (B * T * NH * HS + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vaccum_kernel<<<Vaccum_numBlocks, numThreadsPerBlock>>>(vaccum, att, v, B, NH, T, HS);

    const unsigned int Unpermute_numBlocks = (B * T * NH * HS + numThreadsPerBlock- 1) / numThreadsPerBlock;    
    unpermute_kernel<<<Unpermute_numBlocks, numThreadsPerBlock>>>(vaccum, out, B, T, NH, HS);
}


/**
 * Kernel to store the newly computed Key and Value into the global KV cache. 
 * CAN BE FUSED WITH QKV MATMUL
 * In Prefill: Stores T tokens for the current layer.
 * In Decode: Stores 1 token at g_current_pos for the current layer.
 */
__global__ void store_kv_kernel(float* kv_cache, const float* inp, int B, int T, int C, int layer_idx, int current_pos, int max_seq_len) {
    // Implement logic to copy K and V from the input QKV buffer into g_kv_cache
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= 2 * B * T * C) return;

    int c = index % C;
    int t = (index / C) % T;
    int b = (index / (C * T)) % B;
    int kv = index / (C * T * B);
    int pos = (T == 1) ? current_pos : t;

    int inp_idx = b * T * 3 * C + t * 3 * C + (kv + 1) * C + c;
    int cache_idx = (((layer_idx * 2 + kv) * B + b) * max_seq_len + pos) * C + c;
    kv_cache[cache_idx] = inp[inp_idx];
}

/**
 * Kernel to compute the attention for the Decode phase (T=1).
 * This involves:
 * 1. Extracting the current Query (Q)
 * 2. Gathering all previous Keys (K) and Values (V) from the cache
 * 3. Computing attention using the new Q and cached KV values.
 */
__global__ void decode_attention_kernel(float* out, const float* kv_cache, const float* inp, int B, int C, int NH, int layer_idx, int current_pos, int max_seq_len) {
    // Implement T=1 attention logic using the KV cache
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= B * C) return;

    int HS = C / NH;
    int hs = index % HS;
    int nh = (index / HS) % NH;
    int b = index / C;
    float scale = 1.0f / sqrtf((float)HS);

    float max_score = -FLT_MAX;
    for (int pos = 0; pos <= current_pos; pos++) {
        float score = 0.0f;
        for (int i = 0; i < HS; i++) {
            int c = nh * HS + i;
            int q_idx = b * 3 * C + c;
            int k_idx = (((layer_idx * 2 + 0) * B + b) * max_seq_len + pos) * C + c;
            score += inp[q_idx] * kv_cache[k_idx];
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
    }

    float denom = 0.0f;
    float sum = 0.0f;
    for (int pos = 0; pos <= current_pos; pos++) {
        float score = 0.0f;
        for (int i = 0; i < HS; i++) {
            int c = nh * HS + i;
            int q_idx = b * 3 * C + c;
            int k_idx = (((layer_idx * 2 + 0) * B + b) * max_seq_len + pos) * C + c;
            score += inp[q_idx] * kv_cache[k_idx];
        }
        float weight = expf(score * scale - max_score);
        int v_idx = (((layer_idx * 2 + 1) * B + b) * max_seq_len + pos) * C + nh * HS + hs;
        denom += weight;
        sum += weight * kv_cache[v_idx];
    }

    out[index] = sum / denom;
}


void attention_forward(float* out, float* qkvr, float* att, float* inp, int B, int T, int C, int NH) {
    // 1. Check if KV-caching is enabled, used by the verification script
    if (!g_enable_kv_cache) {
        // TODO: Call your standard Baseline Attention here (Milestone 1/2)
        baseline_attention_forward(out, qkvr, att, inp, B, T, C, NH);
        return;
    }

    // 2. Initialize KV Cache if necessary
    if (g_kv_cache == nullptr) {
        // TODO: Allocate the KV cache, based accordingly to the model parameters
        // Option 1: Allocate a very big kv cache to store all the tokens
        // Option 2: Dynamically resize the kv cache repeatedly for each new token

        // 12 layers, each with separate K and V, batch size B, 1024 tokens, C channels per token
        cudaCheck(cudaMalloc((void**)&g_kv_cache, 12 * 2 * B * 1024 * C * sizeof(float)));
    }

    if (g_is_prefill) {
        // --- PREFILL PHASE ---
        // During prefill, we process the entire input sequence (T > 1).
        
        // A. Store the K and V results from this layer into the cache.
        //    The projections are available in 'inp' (the QKV matmul output).
        const unsigned int numThreadsPerBlock = 256;
        const unsigned int Store_numBlocks = (2 * B * T * C + numThreadsPerBlock - 1) / numThreadsPerBlock;
        store_kv_kernel<<<Store_numBlocks, numThreadsPerBlock>>>(g_kv_cache, inp, B, T, C, g_layer_idx, g_current_pos, 1024);
        
        // B. Run standard baseline attention for the initial prompt tokens.
        baseline_attention_forward(out, qkvr, att, inp, B, T, C, NH);

        // C. Update the global layer index to track progress through the model.
        g_layer_idx++;
        if (g_layer_idx == 12) {
            g_layer_idx = 0;
            g_is_prefill = false; // Transition to Decode phase for the next token
        }
    } else {
        // --- DECODE PHASE ---
        // During decode, we only process the single newest token (T = 1).
        
        // A. Store the single new Key and Value for this step into the cache at g_current_pos.
        const unsigned int numThreadsPerBlock = 256;
        const unsigned int Store_numBlocks = (2 * B * T * C + numThreadsPerBlock - 1) / numThreadsPerBlock;
        store_kv_kernel<<<Store_numBlocks, numThreadsPerBlock>>>(g_kv_cache, inp, B, T, C, g_layer_idx, g_current_pos, 1024);
        
        // B. Run specialized Decode Attention.
        //    This requires reading the history of Keys and Values from the cache
        //    to compute the output for the current token.
        //    TODO: Don't forget to scale your attention scores
        //    TODO: Ensure the output shapes match the next layer
        const unsigned int Decode_numBlocks = (B * C + numThreadsPerBlock - 1) / numThreadsPerBlock;
        decode_attention_kernel<<<Decode_numBlocks, numThreadsPerBlock>>>(out, g_kv_cache, inp, B, C, NH, g_layer_idx, g_current_pos, 1024);
        
        // C. Update the global state.
        g_layer_idx++;
        if (g_layer_idx == 12) {
            g_layer_idx = 0;
            g_current_pos++; // Advance the sequence position
        }
    }
}


#endif // __ATTENTION_CUH__
