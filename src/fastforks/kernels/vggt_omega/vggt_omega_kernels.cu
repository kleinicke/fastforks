// VGGT-Omega max-path fusions (from exp 011 vggt-omega-opt attempt/fast 6a78bb5 vggt_omega/fast_kernels.py, Triton,
// rewritten in CUDA C with extern "C" entry points):
//   omega_ln_f2h:       LayerNorm of an fp32 residual stream (rows [R, T] with strides) -> contiguous fp16 [R * T, C]
//   omega_add_ln:       x1 = x + o * g (fp32, new tensor) and LayerNorm(x1) -> fp16, one pass
//   omega_qk_norm_rope: in place on the packed fp16 qkv [B, N, 3, H, 64]: q / k LayerNorm (optional), rotate-half 2D
//                       RoPE (fp32 tables [N - P, 64], the first P tokens not rotated, optional), q times qscale
// LayerNorms: one 256-thread block per row, float4 per thread (C % 4 == 0, C <= 4096, 16-byte aligned rows).
#include <cuda_fp16.h>
#include <stdint.h>

#define LN_NT 256
#define LN_V 4  // float4 chunks per thread at most

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();  // sh is reused by the next reduction
    if (l == 0) sh[w] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < LN_NT / 32; i++) t += sh[i];
    return t;
}

// v[LN_V] float4 values of one row (zeros past C) -> fp16 LayerNorm written to y (row base)
__device__ __forceinline__ void ln_store(float4 (&v)[LN_V], float s, __half* __restrict__ y, const float* __restrict__ w,
                                         const float* __restrict__ b, int C, float eps, float* sh) {
    const int C4 = C >> 2;
    const float mean = block_sum(s, sh) / C;
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < LN_V; i++) {
        const int c = threadIdx.x + i * LN_NT;
        if (c < C4) {
            v[i].x -= mean; v[i].y -= mean; v[i].z -= mean; v[i].w -= mean;
            q += v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z + v[i].w * v[i].w;
        }
    }
    const float rstd = rsqrtf(block_sum(q, sh) / C + eps);
#pragma unroll
    for (int i = 0; i < LN_V; i++) {
        const int c = threadIdx.x + i * LN_NT;
        if (c < C4) {
            const float4 ww = reinterpret_cast<const float4*>(w)[c], bb = reinterpret_cast<const float4*>(b)[c];
            __half2 h0 = __floats2half2_rn(v[i].x * rstd * ww.x + bb.x, v[i].y * rstd * ww.y + bb.y);
            __half2 h1 = __floats2half2_rn(v[i].z * rstd * ww.z + bb.z, v[i].w * rstd * ww.w + bb.w);
            uint2 u;
            u.x = *reinterpret_cast<uint32_t*>(&h0);
            u.y = *reinterpret_cast<uint32_t*>(&h1);
            reinterpret_cast<uint2*>(y)[c] = u;
        }
    }
}

extern "C" __global__ void __launch_bounds__(LN_NT)
    omega_ln_f2h(const float* __restrict__ x, __half* __restrict__ y, const float* __restrict__ w,
                 const float* __restrict__ b, int T, long long sR, long long sT, int C, float eps) {
    __shared__ float sh[LN_NT / 32];
    const long long row = blockIdx.x;
    const float4* xp = reinterpret_cast<const float4*>(x + (row / T) * sR + (row % T) * sT);
    const int C4 = C >> 2;
    float4 v[LN_V];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < LN_V; i++) {
        const int c = threadIdx.x + i * LN_NT;
        v[i] = c < C4 ? xp[c] : make_float4(0.f, 0.f, 0.f, 0.f);
        s += v[i].x + v[i].y + v[i].z + v[i].w;
    }
    ln_store(v, s, y + row * C, w, b, C, eps, sh);
}

extern "C" __global__ void __launch_bounds__(LN_NT)
    omega_add_ln(const float* __restrict__ x, const __half* __restrict__ o, const float* __restrict__ g,
                 float* __restrict__ xo, __half* __restrict__ y, const float* __restrict__ w, const float* __restrict__ b,
                 int C, float eps) {
    __shared__ float sh[LN_NT / 32];
    const long long row = blockIdx.x;
    const int C4 = C >> 2;
    const float4* xp = reinterpret_cast<const float4*>(x + row * C);
    const uint2* op = reinterpret_cast<const uint2*>(o + row * C);
    float4* xop = reinterpret_cast<float4*>(xo + row * C);
    float4 v[LN_V];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < LN_V; i++) {
        const int c = threadIdx.x + i * LN_NT;
        if (c < C4) {
            const float4 xx = xp[c], gg = reinterpret_cast<const float4*>(g)[c];
            const uint2 u = op[c];
            const float2 o0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
            const float2 o1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
            v[i] = make_float4(fmaf(o0.x, gg.x, xx.x), fmaf(o0.y, gg.y, xx.y), fmaf(o1.x, gg.z, xx.z),
                               fmaf(o1.y, gg.w, xx.w));
            xop[c] = v[i];
        } else {
            v[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        s += v[i].x + v[i].y + v[i].z + v[i].w;
    }
    ln_store(v, s, y + row * C, w, b, C, eps, sh);
}

// One warp per (token, q / k, head) row; lane d holds elements d and d + 32 (the rotate-half pair).
extern "C" __global__ void __launch_bounds__(256)
    omega_qk_norm_rope(__half* __restrict__ qkv, const float* __restrict__ sin_t, const float* __restrict__ cos_t,
                       const float* __restrict__ qw, const float* __restrict__ qb, const float* __restrict__ kw,
                       const float* __restrict__ kb, long long nrows, int N, int H, int P, float eps, float qscale,
                       int has_norm, int has_rope) {
    const long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= nrows) return;  // warp-uniform
    const int d = threadIdx.x & 31;
    const int h = (int)(r % H);
    const long long t = r / H;
    const int isk = (int)(t & 1);
    const long long tok = t >> 1;
    const int n = (int)(tok % N);
    __half* p = qkv + tok * (3LL * H * 64) + (long long)isk * (H * 64) + h * 64;
    float x1 = __half2float(p[d]), x2 = __half2float(p[d + 32]);
    if (has_norm) {
        const float mean = warp_sum(x1 + x2) * (1.f / 64.f);
        x1 -= mean;
        x2 -= mean;
        const float rstd = rsqrtf(warp_sum(x1 * x1 + x2 * x2) * (1.f / 64.f) + eps);
        const float* W = isk ? kw : qw;
        const float* B = isk ? kb : qb;
        x1 = fmaf(x1 * rstd, W[d], B[d]);
        x2 = fmaf(x2 * rstd, W[d + 32], B[d + 32]);
    }
    if (has_rope && n >= P) {
        const long long tp = (long long)(n - P) * 64;
        const float c1 = cos_t[tp + d], c2 = cos_t[tp + d + 32], s1 = sin_t[tp + d], s2 = sin_t[tp + d + 32];
        const float y1 = x1 * c1 - x2 * s1, y2 = x2 * c2 + x1 * s2;
        x1 = y1;
        x2 = y2;
    }
    const float sc = isk ? 1.f : qscale;
    p[d] = __float2half_rn(x1 * sc);
    p[d + 32] = __float2half_rn(x2 * sc);
}
