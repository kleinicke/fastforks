// FP16 row LayerNorm for Swin-style restoration models, with FP32 moments.
#include <cuda_fp16.h>

// SPDX-License-Identifier: BSD-3-Clause
// Reduction order follows PyTorch at 5c4886908584 (aten/src/ATen/native/cuda/layer_norm_kernel.cu).
// PyTorch copyright and licence are reproduced in THIRD_PARTY_NOTICES.md.
// `float3` stores the mean, sum of squared deviations and count, in that order.
__device__ __forceinline__ float3 torch_moments(uint2 values) {
    const __half* p = reinterpret_cast<const __half*>(&values);
    float3 m{0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x = __half2float(p[k]);
        const float old_delta = x - m.x;
        const float total = m.z + 1.f;
        const float next_avg = m.x + old_delta * (1.f / total);
        m = {next_avg, m.y + old_delta * (x - next_avg), total};
    }
    return m;
}

// The tree gives the left and right inputs the same roles as Torch's shuffle reduction.
__device__ __forceinline__ float3 torch_merge(float3 left, float3 right) {
    const float count = left.z + right.z;
    if (count == 0.f)
        return {0.f, 0.f, 0.f};
    const float inverse = 1.f / count;
    const float right_fraction = right.z * inverse;
    const float left_fraction = left.z * inverse;
    const float delta = left.x - right.x;
    return {
        right_fraction * right.x + left_fraction * left.x,
        right.y + left.y + delta * delta * right.z * left_fraction,
        count,
    };
}

__device__ __forceinline__ float3 torch_warp_moments(float3 m) {
#pragma unroll
    for (int distance = 16; distance > 0; distance /= 2) {
        const float3 other{
            __shfl_down_sync(0xffffffff, m.x, distance),
            __shfl_down_sync(0xffffffff, m.y, distance),
            __shfl_down_sync(0xffffffff, m.z, distance),
        };
        m = torch_merge(m, other);
    }
    return m;
}

__device__ __forceinline__ void torch_output(uint2* out, uint2 values, const uint2* weight, const uint2* bias,
                                           int i, float avg, float inv_std) {
    const uint2 alpha4 = weight[i], beta4 = bias[i];
    const __half* x = reinterpret_cast<const __half*>(&values);
    const __half* alpha = reinterpret_cast<const __half*>(&alpha4);
    const __half* beta = reinterpret_cast<const __half*>(&beta4);
    uint2 packed;
    __half* p = reinterpret_cast<__half*>(&packed);
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const float normalized = inv_std * (__half2float(x[v]) - avg);
        p[v] = __float2half_rn(__half2float(alpha[v]) * normalized + __half2float(beta[v]));
    }
    out[i] = packed;
}

template <int channels>
__device__ __forceinline__ float torch_inverse_std(float3 m, float eps) {
    // PyTorch's channel count is dynamic and its division rounds to FP32.
    return rsqrtf(__fdiv_rn(m.y, static_cast<float>(channels)) + eps);
}

template <int channels>
__device__ __forceinline__ void
torch_norm_two_warps(const uint2* __restrict__ x, const uint2* __restrict__ alpha, const uint2* __restrict__ beta,
                     uint2* __restrict__ y, long long rows, float eps, float3* parts) {
    constexpr int vectors = channels / 4;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int index = (warp % 2) * 32 + lane;
    const long long row = blockIdx.x * 4LL + warp / 2;
    const bool valid = row < rows && index < vectors;
    const uint2 values = valid ? x[row * vectors + index] : uint2{};
    const float3 part = torch_warp_moments(valid ? torch_moments(values) : float3{});
    // Two Torch warps are empty. Preserve their merges before combining the occupied pair.
    if (lane == 0)
        parts[warp] = torch_merge(part, {});
    __syncthreads();
    if (!valid)
        return;
    const float3 total = torch_merge(parts[warp & ~1], parts[warp | 1]);
    const float inv = torch_inverse_std<channels>(total, eps);
    torch_output(y + row * vectors, values, alpha, beta, index, total.x, inv);
}

extern "C" __global__ __launch_bounds__(256) void
sr_norm_torch_240(const uint2* __restrict__ x, const uint2* __restrict__ alpha, const uint2* __restrict__ beta,
                  uint2* __restrict__ y, long long rows, float eps) {
    __shared__ float3 parts[8];
    torch_norm_two_warps<240>(x, alpha, beta, y, rows, eps, parts);
}

extern "C" __global__ __launch_bounds__(256) void
sr_norm_torch_180(const uint2* __restrict__ x, const uint2* __restrict__ alpha, const uint2* __restrict__ beta,
                  uint2* __restrict__ y, long long rows, float eps) {
    __shared__ float3 parts[8];
    torch_norm_two_warps<180>(x, alpha, beta, y, rows, eps, parts);
}

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
