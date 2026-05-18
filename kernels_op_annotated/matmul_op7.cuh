#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

// ── 参数说明（Python format 占位符，由 sweep.py 替换）─────────────────────────
// U          : 每个线程负责的输出列数（OC 方向寄存器 tile）
// TILE_WIDTH : K 方向 tile 宽度（shared memory 深度）
// BLOCK_SIZE : 每个 block 的线程数 = U × TILE_WIDTH（硬约束）
//
// 设计要点：1D thread block，每个线程完整负责一行 B*T 中的一行输入，
// 沿 K 方向分段加载进 shared memory，然后对 U 个 OC 输出做点积。
//
// shared memory 布局：
//   input_s[BLOCK_SIZE][TILE_WIDTH]  ← 每个线程行读取 TILE_WIDTH 个 inp 元素
//   weight_s[TILE_WIDTH][U]          ← 协作加载 weight tile
//   总 smem = (BLOCK_SIZE × TILE_WIDTH + TILE_WIDTH × U) × 4 字节
//
// 有效配置（BLOCK_SIZE ≤ 1024，smem ≤ 48 KB）:
//   U=8  TW=8  BS=64   smem=2304 B
//   U=8  TW=16 BS=128  smem=8704 B
//   U=8  TW=32 BS=256  smem=33792 B
//   U=16 TW=8  BS=128  smem=4608 B
//   U=16 TW=16 BS=256  smem=17408 B  ← req_0 默认配置
//   U=32 TW=8  BS=256  smem=9216 B
//   U=32 TW=16 BS=512  smem=34816 B
#define U {U}
#define BLOCK_SIZE {BLOCK_SIZE}
#define TILE_WIDTH {TILE_WIDTH}

// 计算 out[B*T, OC] = inp[B*T, C] @ weight[OC, C]^T + bias[OC]
// grid  : (ceil(OC/U), ceil(B*T/BLOCK_SIZE))
// block : BLOCK_SIZE 个线程（1D）
// 每线程：负责 inp 中一行（bt 行），输出 U 个 OC 值
__global__ void matmul_forward_kernel(float* out, const float* inp, const float* weight,
                                      const float* bias, int B, int T, int C, int OC) {{
    // bt: 当前线程对应的输入行（inp[bt, :]）
    int bt = blockDim.x * blockIdx.x + threadIdx.x;
    // oc_start: 当前 block 负责的 OC 起始列
    int oc_start = blockIdx.y * U;
    // oc_offset: 该线程在 U 个 OC 输出中负责加载 weight 的列偏移
    int oc_offset = threadIdx.x % U;
    // C_offset: 该线程负责加载 weight 的行偏移（沿 TILE_WIDTH 方向）
    int C_offset = threadIdx.x / U;

    // input_s[BLOCK_SIZE][TILE_WIDTH]: 每个线程存入自己 bt 行的当前 K tile
    // weight_s[TILE_WIDTH][U]: 所有线程协作填入当前 K×U weight 子块
    __shared__ float input_s[BLOCK_SIZE][TILE_WIDTH];
    __shared__ float weight_s[TILE_WIDTH][U];

    // 每个线程用寄存器累加 U 个输出，避免频繁写 global memory
    float sum[U];
    for (int i = 0; i < U; i++) {{
        sum[i] = 0.0f;
    }}

    // 主循环：沿 C 维度逐 TILE_WIDTH 宽 tile 推进
    for (int i = 0; i < (C - 1) / TILE_WIDTH + 1; i++) {{
        // 加载 input_s：每个线程独立读取自己 bt 行的 TILE_WIDTH 个元素
        if (bt < B * T) {{
            for (int j = 0; j < TILE_WIDTH; j++) {{
                if (i * TILE_WIDTH + j < C)
                    input_s[threadIdx.x][j] = inp[bt * C + i * TILE_WIDTH + j];
                else
                    input_s[threadIdx.x][j] = 0.0f;
            }}
        }} else {{
            for (int j = 0; j < TILE_WIDTH; j++) {{
                input_s[threadIdx.x][j] = 0.0f;
            }}
        }}

        // 加载 weight_s：利用 C_offset/oc_offset 把 BLOCK_SIZE 个线程
        // 映射到 TILE_WIDTH×U 个 weight 元素（每线程恰好写一个位置）
        if (i * TILE_WIDTH + C_offset >= C || oc_offset + oc_start >= OC) {{
            weight_s[C_offset][oc_offset] = 0.0f;
        }} else {{
            weight_s[C_offset][oc_offset] = weight[(oc_offset + oc_start) * C + i * TILE_WIDTH + C_offset];
        }}

        __syncthreads();  // 等待 smem 全部就绪

        // 寄存器级点积：对当前 TILE_WIDTH 个 K 值，更新 U 个输出
        for (int j = 0; j < U; j++) {{
            for (int k = 0; k < TILE_WIDTH; k++) {{
                sum[j] += input_s[threadIdx.x][k] * weight_s[k][j];
            }}
        }}

        __syncthreads();  // 下一 tile 开始前同步
    }}

    // 写回 global memory，加 bias
    for (int i = 0; i < U; i++) {{
        if (bias && oc_start + i < OC)
            sum[i] += bias[oc_start + i];
        if (bt < B * T && oc_start + i < OC)
            out[bt * OC + oc_start + i] = sum[i];
    }}
}}

void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {{
    const unsigned int numThreadsPerBlock = BLOCK_SIZE;
    dim3 numBlocks((B * T - 1) / numThreadsPerBlock + 1, (OC - 1) / U + 1);
    matmul_forward_kernel<<<numBlocks, numThreadsPerBlock>>>(out, inp, weight, bias, B, T, C, OC);
}}

#undef U
#undef BLOCK_SIZE
#undef TILE_WIDTH

#endif // __MATMUL_KERNEL_CUH__
