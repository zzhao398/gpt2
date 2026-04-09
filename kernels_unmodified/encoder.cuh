#ifndef ENCODER_FORWARD_KERNEL_CUH
#define ENCODER_FORWARD_KERNEL_CUH

#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

__global__ void encoder_forward_kernel(float* out, const int* inp, const float* wte, const float* wpe,
                                       int B, int T, int C) {
    // Implement this
    // done
    int pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos < B * T * C) {
        int channelId = pos % C;
        int tokenId = pos / C;
        int tokenPos = tokenId % T;
        int token = inp[tokenId];
        out[pos] = wte[token * C + channelId] + wpe[tokenPos * C + channelId];
    }
}

// Launch kernel here
void encoder_forward(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C) {
    // Implement this
    // done
    const unsigned int numThreadsPerBlock = 256;
    const unsigned int numBlocks = (B * T * C + numThreadsPerBlock - 1)/numThreadsPerBlock;

    encoder_forward_kernel<<<numBlocks, numThreadsPerBlock>>>(out, inp, wte, wpe, B, T, C);
}


#endif // ENCODER_FORWARD_KERNEL_CUH