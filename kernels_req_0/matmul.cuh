#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8

// out[B*T, OC] = inp[B*T, C] @ weight[OC, C]^T + bias[OC]
__global__ void matmul_forward_kernel(float* out, const float* inp, const float* weight,
                                      const float* bias, int B, int T, int C, int OC) {
    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    int ty = threadIdx.y; // thread row
    int tx = threadIdx.x; // thread col

    __shared__ float smem_inp[BM][BK];
    __shared__ float smem_w[BK][BN];

    float reg_out[TM][TN] = {};
    float reg_inp[TM];
    float reg_w[TN];

    int total_threads = (BM / TM) * (BN / TN); // 64
    int tid = ty * (BN / TN) + tx;

    for (int k = 0; k < C; k += BK) {
        // inp tile: BM x BK
        for (int i = tid; i < BM * BK; i += total_threads) {
            int r = i / BK, c = i % BK;
            int grow = row_start + r, gk = k + c;
            smem_inp[r][c] = (grow < B * T && gk < C) ? inp[grow * C + gk] : 0.0f;
        }
        // weight tile: BK x BN
        for (int i = tid; i < BK * BN; i += total_threads) {
            int r = i / BN, c = i % BN;
            int gk = k + r, gcol = col_start + c;
            smem_w[r][c] = (gk < C && gcol < OC) ? weight[gcol * C + gk] : 0.0f;
        }
        __syncthreads();

        for (int bk = 0; bk < BK; bk++) {
            for (int tm = 0; tm < TM; tm++)
                reg_inp[tm] = smem_inp[ty * TM + tm][bk];
            for (int tn = 0; tn < TN; tn++)
                reg_w[tn] = smem_w[bk][tx * TN + tn];
            for (int tm = 0; tm < TM; tm++)
                for (int tn = 0; tn < TN; tn++)
                    reg_out[tm][tn] += reg_inp[tm] * reg_w[tn];
        }
        __syncthreads();
    }

    for (int tm = 0; tm < TM; tm++) {
        int grow = row_start + ty * TM + tm;
        if (grow >= B * T) continue;
        for (int tn = 0; tn < TN; tn++) {
            int gcol = col_start + tx * TN + tn;
            if (gcol >= OC) continue;
            float val = reg_out[tm][tn];
            if (bias) val += bias[gcol];
            out[grow * OC + gcol] = val;
        }
    }
}

void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    dim3 grid((OC + BN - 1) / BN, (B * T + BM - 1) / BM);
    dim3 block(BN / TN, BM / TM); // 64 threads
    matmul_forward_kernel<<<grid, block>>>(out, inp, weight, bias, B, T, C, OC);
}

#undef BM
#undef BN
#undef BK
#undef TM
#undef TN

#endif // __MATMUL_KERNEL_CUH__
