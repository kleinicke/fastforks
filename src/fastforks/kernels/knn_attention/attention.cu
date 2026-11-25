#include <cuda_fp16.h>
#include <stdint.h>
#include <math_constants.h>

constexpr unsigned MASK = 0xffffffffu;

// A warp attends one query. Each 8-lane subgroup sums over a different
// neighbour, with eight contiguous channels per lane in the 64-wide head.
struct __align__(16) Channels { half v[8]; };

template<int K>
__device__ void indexed(const Channels* qkv, const int64_t* indices, Channels* out, int64_t rows) {
    int lane = threadIdx.x & 31, group = lane / 8, channel = lane % 8;
    for (int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
         row < rows; row += int64_t(gridDim.x) * (blockDim.x / 32)) {
        Channels query = qkv[row * 24 + channel];  // 3 projections x 8 chunks
        int64_t neighbor[K / 4];
        float logits[K / 4], largest = -CUDART_INF_F;
        #pragma unroll
        for (int j = 0; j < K / 4; ++j) {
            neighbor[j] = indices[row * K + j * 4 + group];
            Channels key = qkv[neighbor[j] * 24 + 8 + channel];
            float dot = 0;
            #pragma unroll
            for (int c = 0; c < 8; ++c) dot = fmaf(float(query.v[c]), float(key.v[c]), dot);
            #pragma unroll
            for (int s = 4; s; s /= 2) dot += __shfl_xor_sync(MASK, dot, s, 8);
            // Matmul returns fp16, the exact scale for C=64 is 1/8.
            logits[j] = float(__float2half_rn(float(__float2half_rn(dot)) * .125f));
            largest = fmaxf(largest, logits[j]);
        }
        largest = fmaxf(largest, __shfl_xor_sync(MASK, largest, 8));
        largest = fmaxf(largest, __shfl_xor_sync(MASK, largest, 16));
        float sum = 0;
        #pragma unroll
        for (int j = 0; j < K / 4; ++j) sum += (logits[j] = expf(logits[j] - largest));
        sum += __shfl_xor_sync(MASK, sum, 8);
        sum += __shfl_xor_sync(MASK, sum, 16);
        float result[8] = {};
        #pragma unroll
        for (int j = 0; j < K / 4; ++j) {
            float probability = float(__float2half_rn(logits[j] / sum));
            Channels value = qkv[neighbor[j] * 24 + 16 + channel];
            #pragma unroll
            for (int c = 0; c < 8; ++c) result[c] = fmaf(probability, float(value.v[c]), result[c]);
        }
        Channels packed;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            float y = result[c];
            y += __shfl_down_sync(MASK, y, 16);
            y += __shfl_down_sync(MASK, y, 8);
            packed.v[c] = __float2half_rn(y);
        }
        if (group == 0) out[row * 8 + channel] = packed;
    }
}

extern "C" __global__ void knn8(const Channels* qkv, const int64_t* indices, Channels* out, int64_t rows) {
    indexed<8>(qkv, indices, out, rows);
}
extern "C" __global__ void knn16(const Channels* qkv, const int64_t* indices, Channels* out, int64_t rows) {
    indexed<16>(qkv, indices, out, rows);
}
