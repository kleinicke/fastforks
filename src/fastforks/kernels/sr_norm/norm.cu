// FP16 row LayerNorm for Swin-style restoration models, with FP32 moments and affine coefficients.
#include <cuda_fp16.h>

template <int VPT, int VW, typename Vec>
__device__ __forceinline__ void norm_impl(const __half* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                        __half* __restrict__ y, long long M, int C, float eps, int G) {
    int lane_in_row = threadIdx.x % G;
    long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / G;
    bool valid = row < M;
    float v[VPT][VW];
    float s = 0.f;
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        int c = (lane_in_row + u * G) * VW;
        if (valid && c < C) {
            Vec q = __ldg(reinterpret_cast<const Vec*>(x + row * C + c));
            const __half* hq = reinterpret_cast<const __half*>(&q);
#pragma unroll
            for (int k = 0; k < VW; ++k) { v[u][k] = __half2float(hq[k]); s += v[u][k]; }
        } else {
#pragma unroll
            for (int k = 0; k < VW; ++k) v[u][k] = 0.f;
        }
    }
    for (int o = G / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o, G);
    float mean = s / C, ss = 0.f;
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        int c = (lane_in_row + u * G) * VW;
        if (c < C)
#pragma unroll
            for (int k = 0; k < VW; ++k) { float d = v[u][k] - mean; ss += d * d; }
    }
    for (int o = G / 2; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o, G);
    float rstd = rsqrtf(ss / C + eps);
    if (!valid) return;
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        int c = (lane_in_row + u * G) * VW;
        if (c >= C) continue;
        __align__(16) __half packed[VW];
#pragma unroll
        for (int k = 0; k < VW; ++k) {
            float r = (v[u][k] - mean) * rstd * w[c + k] + b[c + k];
            packed[k] = __float2half(r);
        }
        *reinterpret_cast<Vec*>(y + row * C + c) = *reinterpret_cast<const Vec*>(packed);
    }
}
#define LN_K(V) extern "C" __global__ void __launch_bounds__(256) sr_norm8_v##V(const __half* x, const float* w, \
    const float* b, __half* y, long long M, int C, float eps, int G) { norm_impl<V, 8, uint4>(x, w, b, y, M, C, eps, G); }
LN_K(1)
LN_K(2)
LN_K(4)
LN_K(8)
#undef LN_K

#define LN4_K(V) extern "C" __global__ void __launch_bounds__(256) sr_norm4_v##V(const __half* x, const float* w, \
    const float* b, __half* y, long long M, int C, float eps, int G) { norm_impl<V, 4, uint2>(x, w, b, y, M, C, eps, G); }
LN4_K(1)
LN4_K(2)
LN4_K(4)
LN4_K(8)
#undef LN4_K
