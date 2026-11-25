/* SPDX-License-Identifier: Apache-2.0 */
#include <cuda_fp16.h>

// One warp owns one projected Q or K head. V and the other heads are disjoint.
extern "C" __global__ void pointdit_qk(__half* qkv, const float* weight, const float* table,
                                      int tokens, int heads, long long rows) {
    const int lane = threadIdx.x % 32;
    for (long long row = (long long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
         row < rows; row += (long long)gridDim.x * (blockDim.x / 32)) {
        const long long token = row / (2 * heads);
        const int kind = (row / heads) % 2;
        __half* ptr = qkv + (token * 3 * heads + kind * heads + row % heads) * 64 + lane;
        const float a = __half2float(ptr[0]), b = __half2float(ptr[32]);
        float sum = a * a + b * b;
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        const float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) * (1.0f / 64) + 1e-6f);
        const float* scale = weight + kind * 64 + lane;
        // PyTorch casts normalized Q/K back to fp16 before rotary embedding.
        float x = __half2float(__float2half_rn((a * inv) * scale[0]));
        float y = __half2float(__float2half_rn((b * inv) * scale[32]));
        const float* angle = table + (token % tokens) * 64 + lane;
        const float c = angle[0], s = angle[32], m = kind ? 1.f : 0.125f;
        ptr[0] = __float2half_rn((x * c - y * s) * m);
        ptr[32] = __float2half_rn((y * c + x * s) * m);
    }
}
