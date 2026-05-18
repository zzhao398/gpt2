#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

// ── 参数（与 req_0 相同，额外加 PAD）────────────────────────────────────────
// U=16, TILE_WIDTH=16, BLOCK_SIZE=256 是 sweep 在此内核架构下找到的最优配置
// PAD=1: shared memory bank conflict 消除补丁
#define U 16
#define BLOCK_SIZE 256
#define TILE_WIDTH 16
#define PAD 1

// ── Bank Conflict 分析与 PAD 的作用 ─────────────────────────────────────────
//
// 共享内存有 32 个 bank，每 bank 4 字节，每行 128 字节。
//
// input_s[BLOCK_SIZE][TILE_WIDTH]（无 PAD）：
//   行步长 = TILE_WIDTH = 16 floats = 64 字节。
//   线程 i 读 input_s[i][k] 时，bank 号 = (i * 16 + k) % 32。
//   同 warp 内 32 个连续线程（i = base, base+1, ..., base+31）
//   读同一列 k 时：bank = (i*16 + k) % 32；因 gcd(32,16)=16，
//   每 16 个线程映射到同一 bank → 2-way conflict。
//
// input_s[BLOCK_SIZE][TILE_WIDTH + PAD]（PAD=1）：
//   行步长 = 17 floats = 68 字节。gcd(32,17)=1，
//   32 个连续线程各落不同 bank → 0 conflict。
//
// weight_s[TILE_WIDTH][U]：
//   16 个线程同时读 weight_s[k][j]（j 固定）= broadcast 读取，
//   不产生 bank conflict，无需 PAD。

__global__ void matmul_forward_kernel(float* out, const float* inp, const float* weight,
                                      const float* bias, int B, int T, int C, int OC) {
    int bt = blockDim.x * blockIdx.x + threadIdx.x;
    int oc_start = blockIdx.y * U;
    int oc_offset = threadIdx.x % U;
    int C_offset = threadIdx.x / U;

    // PAD 把行步长从 16→17，消除 2-way bank conflict
    __shared__ float input_s[BLOCK_SIZE][TILE_WIDTH + PAD];
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

void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    const unsigned int numThreadsPerBlock = BLOCK_SIZE;
    dim3 numBlocks((B * T - 1) / numThreadsPerBlock + 1, (OC - 1) / U + 1);
    matmul_forward_kernel<<<numBlocks, numThreadsPerBlock>>>(out, inp, weight, bias, B, T, C, OC);
}

#undef U
#undef BLOCK_SIZE
#undef TILE_WIDTH
#undef PAD

#endif // __MATMUL_KERNEL_CUH__
