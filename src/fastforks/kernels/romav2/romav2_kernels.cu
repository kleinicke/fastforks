// RoMa v2 kernels.
//
// Local correlation of the refiners without the sampled window tensor: for every pixel p of image A and every offset
// (dy, dx) of the (2R+1)^2 window, <f0[p], grid_sample(f1, warp[p] + offset)> / sqrt(C) (bilinear, zeros outside,
// align_corners = False), as upstream `local_correlation`. The window steps are exactly one pixel of f1, so all
// samples of a pixel share the bilinear weights: the kernel takes the (2R+2)^2 dot products of the integer window
// (fp32) and combines them with those weights. A corner outside f1 contributes zero, as grid_sample's zero padding.
// One warp per pixel: lane k holds channels k, k + 32, ... of f0 (coalesced loads); 32 window positions at a time are
// reduced with a butterfly reduce-scatter (31 shuffles per 32 sums), after which lane k holds position k.
// f0, f1: [B, H, W, C] fp32 contiguous; warp: [B, H, W, 2] fp32 (x, y normalised to [-1, 1]); out: [B, H, W, K] fp32
// with K = (2R+1)^2, channel j * (2R+1) + i = y offset j - R, x offset i - R (upstream's row-major window order).

#include <cuda_runtime.h>

template <int C, int R>
__device__ __forceinline__ void local_corr_body(const float* __restrict__ f0, const float* __restrict__ f1,
                                                const float* __restrict__ warp, float* __restrict__ out, int b, int h,
                                                int w, float scale) {
    constexpr int V = (C + 31) / 32, S = 2 * R + 2, N = S * S, SIDE = 2 * R + 1, NC = (N + 31) / 32;
    __shared__ float sd[8][NC * 32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const long long hw = (long long)h * w;
    const long long p = (long long)blockIdx.x * (blockDim.x >> 5) + wid;
    if (p >= b * hw) return;  // whole warps
    float a[V];
#pragma unroll
    for (int k = 0; k < V; ++k) {
        const int c = lane + 32 * k;
        a[k] = (c < C) ? f0[p * C + c] * scale : 0.f;
    }
    const float* base = f1 + (p / hw) * hw * C;
    // grid_sample's unnormalisation (align_corners = False): pixel = ((g + 1) * size - 1) / 2
    const float ix = ((warp[2 * p] + 1.f) * w - 1.f) * 0.5f, iy = ((warp[2 * p + 1] + 1.f) * h - 1.f) * 0.5f;
    const bool finite = fabsf(ix) < 1e7f && fabsf(iy) < 1e7f;  // far outside (or NaN): every sample is zero
    const float fx = finite ? floorf(ix) : 0.f, fy = finite ? floorf(iy) : 0.f;
    const float wx = ix - fx, wy = iy - fy;
    const int x0 = (int)fx - R, y0 = (int)fy - R;
    float* d = sd[wid];
#pragma unroll
    for (int ch = 0; ch < NC; ++ch) {
        float part[32];
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const int pos = ch * 32 + k;
            part[k] = 0.f;
            if (pos < N) {
                const int gx = x0 + pos % S, gy = y0 + pos / S;
                if (finite && gx >= 0 && gx < w && gy >= 0 && gy < h) {
                    const float* src = base + ((long long)gy * w + gx) * C + lane;
                    float acc = 0.f;
#pragma unroll
                    for (int c = 0; c < V; ++c)
                        if (lane + 32 * c < C) acc = fmaf(a[c], __ldg(src + 32 * c), acc);
                    part[k] = acc;
                }
            }
        }
#pragma unroll
        for (int s = 16; s >= 1; s >>= 1) {
            const bool up = (lane & s) != 0;
#pragma unroll
            for (int k = 0; k < s; ++k) {
                const float send = up ? part[k] : part[k + s];
                const float keep = up ? part[k + s] : part[k];
                part[k] = keep + __shfl_xor_sync(0xffffffffu, send, s);
            }
        }
        d[ch * 32 + lane] = part[0];
    }
    __syncwarp();
    float* o = out + p * (SIDE * SIDE);
    for (int e = lane; e < SIDE * SIDE; e += 32) {
        const int j = e / SIDE, i = e % SIDE;  // y offset j - R, x offset i - R
        const float* q = d + j * S + i;
        o[e] = (1.f - wy) * ((1.f - wx) * q[0] + wx * q[1]) + wy * ((1.f - wx) * q[S] + wx * q[S + 1]);
    }
}

// The two refiners with a local correlation: stride 4 (C = 192, radius 3) and stride 2 (C = 48, radius 1).
// blockDim = 256 (8 pixels per block).
extern "C" __global__ void __launch_bounds__(256)
    romav2_local_corr_c192_r3(const float* f0, const float* f1, const float* warp, float* out, int b, int h, int w,
                              float scale) {
    local_corr_body<192, 3>(f0, f1, warp, out, b, h, w, scale);
}
extern "C" __global__ void __launch_bounds__(256)
    romav2_local_corr_c48_r1(const float* f0, const float* f1, const float* warp, float* out, int b, int h, int w,
                             float scale) {
    local_corr_body<48, 1>(f0, f1, warp, out, b, h, w, scale);
}

// Refiner blocks: depthwise 5x5 conv (padding 2, zeros outside) + bias + ReLU of a channels-last bf16 map, one pass
// instead of cuDNN's conv, the bias add and the ReLU. Rounding as those three ops: fp32 sum over (ky, kx) in row-major
// order, rounded to bf16 (the conv's output), + bias in fp32, rounded to bf16, ReLU.
// x, out: [B, H, W, C] bf16 contiguous (NHWC); wt: [25, C] bf16 (tap-major, the conv weight transposed); bias: [C].
// One thread: 8 channels (16-byte loads) x TX consecutive x of one row; consecutive threads take consecutive channel
// groups (coalesced). C must be a multiple of 8.

#include <cuda_bf16.h>

template <int TX>
__device__ __forceinline__ void dw5_body(const uint4* __restrict__ x, const uint4* __restrict__ wt,
                                         const uint4* __restrict__ bias, uint4* __restrict__ out, int b, int h, int w,
                                         int c) {
    constexpr int K = 5, P = 2, L = TX + K - 1;
    const int g8 = c >> 3, nxt = (w + TX - 1) / TX;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)b * h * nxt * g8) return;
    const int cg = (int)(t % g8);
    t /= g8;
    const int x0 = (int)(t % nxt) * TX;
    t /= nxt;
    const int y = (int)(t % h);
    const long long n = t / h;
    float acc[TX][8];
#pragma unroll
    for (int i = 0; i < TX; ++i)
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[i][k] = 0.f;
#pragma unroll
    for (int ky = 0; ky < K; ++ky) {
        const int yy = y + ky - P;
        if (yy < 0 || yy >= h) continue;  // zero row: adds nothing (exact, as + 0 * w)
        const uint4* row = x + ((n * h + yy) * w) * g8 + cg;
        uint4 v[L];
#pragma unroll
        for (int j = 0; j < L; ++j) {
            const int xx = x0 + j - P;
            v[j] = (xx >= 0 && xx < w) ? __ldg(row + (long long)xx * g8) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int kx = 0; kx < K; ++kx) {
            const uint4 wq = __ldg(wt + (ky * K + kx) * g8 + cg);
            const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wq);
            float wf[8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 f = __bfloat1622float2(wp[k]);
                wf[2 * k] = f.x;
                wf[2 * k + 1] = f.y;
            }
#pragma unroll
            for (int i = 0; i < TX; ++i) {
                const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(&v[i + kx]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float2 f = __bfloat1622float2(vp[k]);
                    acc[i][2 * k] = fmaf(f.x, wf[2 * k], acc[i][2 * k]);
                    acc[i][2 * k + 1] = fmaf(f.y, wf[2 * k + 1], acc[i][2 * k + 1]);
                }
            }
        }
    }
    const uint4 bq = __ldg(bias + cg);
    const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&bq);
    uint4* o = out + ((n * h + y) * w + x0) * g8 + cg;
#pragma unroll
    for (int i = 0; i < TX; ++i) {
        if (x0 + i >= w) break;
        uint4 r;
        __nv_bfloat162* rp = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 bf = __bfloat1622float2(bp[k]);
            const float a0 = __bfloat162float(__float2bfloat16_rn(acc[i][2 * k])) + bf.x;
            const float a1 = __bfloat162float(__float2bfloat16_rn(acc[i][2 * k + 1])) + bf.y;
            rp[k] = __floats2bfloat162_rn(a0 > 0.f ? a0 : 0.f, a1 > 0.f ? a1 : 0.f);  // ReLU: +0 for <= 0 and -0
        }
        o[(long long)i * g8] = r;
    }
}

extern "C" __global__ void __launch_bounds__(256)
    romav2_dw5_bias_relu_bf16(const uint4* x, const uint4* wt, const uint4* bias, uint4* out, int b, int h, int w,
                              int c) {
    dw5_body<4>(x, wt, bias, out, b, h, w, c);
}
