// RAFT optical-flow inference kernels (everything except the convolutions).
// Layout: fp16 NHWC activations; a "slice" is (pointer to its first channel, pixel stride in elements). The flow state
// is the fp32 match position `coords` [P][2] (x, y) at 1/8 resolution, as RAFT's coords1.
// Numerics follow the portable fp16 runner (fp16 autocast for the convolutions, fp32 correlation, lookup and coords).
#include <cuda_fp16.h>
#include <stdint.h>

#define GRID_STRIDE(i, n) for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (long long)blockDim.x * gridDim.x)

// ---------------------------------------------------------------------------------------------------------------
// Input: float32 planar [3, H, W] (already scaled to [-1, 1] and padded by the runner) -> fp16 NHWC, 3 channels.
extern "C" __global__ void prep_image(const float* __restrict__ src, __half* __restrict__ dst, long long P) {
    GRID_STRIDE(i, P) {
#pragma unroll
        for (int c = 0; c < 3; ++c) dst[i * 3 + c] = __float2half(src[c * P + i]);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// InstanceNorm (as the `raft` library). Pass 1: per (pixel chunk, channel) fp32 sums of (x - a) and (x - a)^2 around
// a = the chunk's first pixel -> (mean, M2) per chunk.
extern "C" __global__ void in_stats_partial(const __half* __restrict__ x, long long P, int C, int chunk,
                                            float* __restrict__ partial) {
    __shared__ float red[2][256 * 8];
    const int CV = C >> 3, rows = blockDim.x / CV;
    const int t = threadIdx.x, c8 = t % CV, row = t / CV;
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, ss[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    long long p0 = (long long)blockIdx.x * chunk, p1 = min(P, p0 + chunk);
    float a[8];
    {
        uint4 v = *reinterpret_cast<const uint4*>(x + p0 * C + c8 * 8);
        const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
        for (int k = 0; k < 4; ++k) { float2 f = __half22float2(h2[k]); a[2 * k] = f.x; a[2 * k + 1] = f.y; }
    }
    if (row < rows) {
        for (long long p = p0 + row; p < p1; p += rows) {
            uint4 v = *reinterpret_cast<const uint4*>(x + p * C + c8 * 8);
            const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __half22float2(h2[k]);
                float dx = f.x - a[2 * k], dy = f.y - a[2 * k + 1];
                s[2 * k] += dx; ss[2 * k] = fmaf(dx, dx, ss[2 * k]);
                s[2 * k + 1] += dy; ss[2 * k + 1] = fmaf(dy, dy, ss[2 * k + 1]);
            }
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            red[0][row * C + c8 * 8 + k] = s[k];
            red[1][row * C + c8 * 8 + k] = ss[k];
        }
    }
    __syncthreads();
    const float inv = 1.f / (float)(p1 - p0);
    for (int c = t; c < C; c += blockDim.x) {
        float sa = 0.f, sb = 0.f;
        for (int r = 0; r < rows; ++r) { sa += red[0][r * C + c]; sb += red[1][r * C + c]; }
        const float m = sa * inv;
        const float ac = __half2float(x[p0 * C + c]);
        partial[(long long)blockIdx.x * 2 * C + c] = ac + m;
        partial[(long long)blockIdx.x * 2 * C + C + c] = fmaxf(sb - sa * m, 0.f);
    }
}

// Pass 2: merge the chunks' (mean, M2) with Chan's formula in a fixed order -> mean and 1/sqrt(var + eps).
__device__ __forceinline__ void chan_merge(float& na, float& ma, float& qa, float nb, float mb, float qb) {
    if (nb == 0.f) return;
    const float n = na + nb, d = mb - ma, f = nb / n;
    ma += d * f;
    qa += qb + d * d * na * f;
    na = n;
}
extern "C" __global__ void in_stats_final(const float* __restrict__ partial, int nblk, int C, long long P, int chunk,
                                          float eps, float* __restrict__ mean_rstd) {
    int c = blockIdx.x, lane = threadIdx.x;
    float n = 0.f, m = 0.f, q = 0.f;
    for (int bk = lane; bk < nblk; bk += 32)
        chan_merge(n, m, q, (float)min((long long)chunk, P - (long long)bk * chunk), partial[(long long)bk * 2 * C + c],
                   partial[(long long)bk * 2 * C + C + c]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float nb = __shfl_down_sync(0xffffffffu, n, o), mb = __shfl_down_sync(0xffffffffu, m, o),
                    qb = __shfl_down_sync(0xffffffffu, q, o);
        if (lane < o) chan_merge(n, m, q, nb, mb, qb);
    }
    if (lane == 0) {
        mean_rstd[c] = m;
        mean_rstd[C + c] = rsqrtf(q / (float)P + eps);
    }
}

__device__ __forceinline__ __half hrelu(__half v) { return __hgt(v, __float2half(0.f)) ? v : __float2half(0.f); }

// out = relu(IN(x))                              mode 0
// out = relu(relu(IN(x)) + IN(r))                mode 1 (residual through a normalized 1x1 downsample conv)
// out = relu(relu(IN(x)) + r)                    mode 2 (identity residual)
extern "C" __global__ void in_apply(const __half* x, const float* __restrict__ st, const __half* __restrict__ r,
                                    const float* __restrict__ st_r, __half* out, long long n, int C, int mode) {
    GRID_STRIDE(v, n / 8) {
        long long i = v * 8;
        int c0 = (int)(i % C);
        uint4 xv = *reinterpret_cast<const uint4*>(x + i);
        const __half* xh = reinterpret_cast<const __half*>(&xv);
        uint4 rv = make_uint4(0, 0, 0, 0);
        if (mode != 0) rv = *reinterpret_cast<const uint4*>(r + i);
        const __half* rh = reinterpret_cast<const __half*>(&rv);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int c = c0 + k;
            __half a = hrelu(__float2half((__half2float(xh[k]) - st[c]) * st[C + c]));
            if (mode == 1) a = hrelu(__hadd(a, __float2half((__half2float(rh[k]) - st_r[c]) * st_r[C + c])));
            else if (mode == 2) a = hrelu(__hadd(a, rh[k]));
            o[k] = a;
        }
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(o);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Per-channel affine + activation of an NHWC slice y (C % 8 == 0), written to slice o and optionally also to slice o2:
//   v = y * s[c] + t[c]   (s == nullptr: scale 1; t == nullptr: shift 0), fp32 math, rounded to fp16
//   mode 0: v      1: relu(v)      2: tanh(v)      3: relu(fp16(relu(v)) + r)   (residual r: a slice, fp16 add)
extern "C" __global__ void affine_act(const __half* y, int ystride, __half* o, int ostride, __half* o2, int o2stride,
                                      const float* __restrict__ s, const float* __restrict__ t,
                                      const __half* __restrict__ r, int rstride, long long P, int C, int mode) {
    const int CV = C / 8;
    GRID_STRIDE(i, P * CV) {
        long long p = i / CV;
        int c0 = (int)(i % CV) * 8;
        uint4 yv = *reinterpret_cast<const uint4*>(y + p * ystride + c0);
        const __half* yh = reinterpret_cast<const __half*>(&yv);
        uint4 rv = make_uint4(0, 0, 0, 0);
        if (mode == 3) rv = *reinterpret_cast<const uint4*>(r + p * rstride + c0);
        const __half* rh = reinterpret_cast<const __half*>(&rv);
        __align__(16) __half out[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int c = c0 + k;
            float v = __half2float(yh[k]);
            if (s) v *= s[c];
            if (t) v += t[c];
            if (mode == 1) v = fmaxf(v, 0.f);
            else if (mode == 2) v = tanhf(v);
            __half hv = __float2half(mode == 3 ? fmaxf(v, 0.f) : v);
            if (mode == 3) hv = hrelu(__hadd(hv, rh[k]));
            out[k] = hv;
        }
        *reinterpret_cast<uint4*>(o + p * ostride + c0) = *reinterpret_cast<const uint4*>(out);
        if (o2) *reinterpret_cast<uint4*>(o2 + p * o2stride + c0) = *reinterpret_cast<const uint4*>(out);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Correlation pyramid without the volume: corr_l(p, q) = <f1(p), avgpool^l(f2)(q)> / sqrt(C) (average pooling commutes
// with the dot product), so only the pooled fmap2 pyramid is stored. 2 x 2 average pooling (floor) of an NHWC map.
extern "C" __global__ void avgpool2(const __half* __restrict__ src, int Hs, int Ws, __half* __restrict__ dst, int Hd,
                                    int Wd, int C) {
    const int CV = C / 8;
    GRID_STRIDE(i, (long long)Hd * Wd * CV) {
        int c = (int)(i % CV) * 8;
        long long pd = i / CV;
        int y = (int)(pd / Wd), x = (int)(pd % Wd);
        float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            uint4 v = *reinterpret_cast<const uint4*>(src + ((long long)(2 * y + d / 2) * Ws + 2 * x + d % 2) * C + c);
            const __half* h = reinterpret_cast<const __half*>(&v);
#pragma unroll
            for (int k = 0; k < 8; ++k) s[k] += __half2float(h[k]);
        }
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __float2half(0.25f * s[k]);
        *reinterpret_cast<uint4*>(dst + pd * C + c) = *reinterpret_cast<const uint4*>(o);
    }
}

// Correlation lookup on the fly: for every pixel and level, the (2R+1)^2 bilinear samples (grid_sample, align_corners,
// zeros outside) of the correlation around coords / 2^l. All samples share the fractional weights, so the kernel
// takes the (2R+2)^2 dot products of the integer window (fp16 operands, fp32 accumulation) and combines them.
// One warp per pixel: lane k holds channels [k*V, k*V+V) of f1; 32 window positions at a time are reduced with a
// butterfly reduce-scatter (31 shuffles), after which lane k holds the dot product of position k.
// Output channel l*(2R+1)^2 + i*(2R+1) + j samples x offset i - R and y offset j - R (torchvision's order); channels
// up to ostride are zero padding. Level l of the L-level pyramid is [h0 >> l, w0 >> l, C].
template <int C, int R, int L>
__device__ __forceinline__ void corr_lookup_body(const __half* __restrict__ f1, const __half* __restrict__ l0,
                                                 const __half* __restrict__ l1, const __half* __restrict__ l2,
                                                 const __half* __restrict__ l3, int h0, int w0,
                                                 const float* __restrict__ coords, __half* __restrict__ out,
                                                 int ostride, float scale) {
    constexpr int V = C / 32, S = 2 * R + 2, N = S * S, SIDE = 2 * R + 1, NC = (N + 31) / 32;
    __shared__ float sd[8][NC * 32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const long long P = (long long)h0 * w0;
    const long long p = (long long)blockIdx.x * (blockDim.x >> 5) + wid;
    if (p >= P) return;  // whole warps
    float a[V];
    {
        const __half* src = f1 + p * C + lane * V;
        __align__(16) __half v[V];
        if constexpr (V == 8) *reinterpret_cast<uint4*>(v) = *reinterpret_cast<const uint4*>(src);
        else *reinterpret_cast<uint2*>(v) = *reinterpret_cast<const uint2*>(src);
#pragma unroll
        for (int k = 0; k < V; ++k) a[k] = __half2float(v[k]) * scale;
    }
    const float cx0 = coords[2 * p], cy0 = coords[2 * p + 1];
    const __half* lv[4] = {l0, l1, l2, l3};
    float* d = sd[wid];
#pragma unroll 1
    for (int l = 0; l < L; ++l) {
        const int hl = h0 >> l, wl = w0 >> l;
        const float inv = 1.f / (float)(1 << l);
        const float cx = cx0 * inv, cy = cy0 * inv;
        const float fx = floorf(cx), fy = floorf(cy);
        const float wx = cx - fx, wy = cy - fy;
        const int x0 = (int)fx - R, y0 = (int)fy - R;
        const __half* base = lv[l] + lane * V;
#pragma unroll
        for (int ch = 0; ch < NC; ++ch) {
            float part[32];
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const int pos = ch * 32 + k;
                part[k] = 0.f;
                if (pos < N) {
                    const int gx = x0 + pos % S, gy = y0 + pos / S;
                    if (gx >= 0 && gx < wl && gy >= 0 && gy < hl) {
                        const __half* src = base + ((long long)gy * wl + gx) * C;
                        __align__(16) __half v[V];
                        if constexpr (V == 8) *reinterpret_cast<uint4*>(v) = __ldg(reinterpret_cast<const uint4*>(src));
                        else *reinterpret_cast<uint2*>(v) = __ldg(reinterpret_cast<const uint2*>(src));
                        float acc = 0.f;
#pragma unroll
                        for (int c = 0; c < V; ++c) acc = fmaf(a[c], __half2float(v[c]), acc);
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
        __half* o = out + p * ostride + l * SIDE * SIDE;
        for (int e = lane; e < SIDE * SIDE; e += 32) {
            const int i = e / SIDE, j = e % SIDE;  // x offset i, y offset j
            const float* q = d + j * S + i;
            const float v = (1.f - wy) * ((1.f - wx) * q[0] + wx * q[1]) + wy * ((1.f - wx) * q[S] + wx * q[S + 1]);
            o[e] = __float2half(v);
        }
        __syncwarp();
    }
    for (int c = L * SIDE * SIDE + lane; c < ostride; c += 32) out[p * ostride + c] = __float2half(0.f);
}

// raft-large: C = 256, radius 4 (324 channels); raft-small: C = 128, radius 3 (196 channels). blockDim = 256.
extern "C" __global__ void __launch_bounds__(256) corr_lookup_c256_r4(const __half* f1, const __half* l0, const __half* l1,
                                                                      const __half* l2, const __half* l3, int h0, int w0,
                                                                      const float* coords, __half* out, int ostride,
                                                                      float scale) {
    corr_lookup_body<256, 4, 4>(f1, l0, l1, l2, l3, h0, w0, coords, out, ostride, scale);
}
extern "C" __global__ void __launch_bounds__(256) corr_lookup_c128_r3(const __half* f1, const __half* l0, const __half* l1,
                                                                      const __half* l2, const __half* l3, int h0, int w0,
                                                                      const float* coords, __half* out, int ostride,
                                                                      float scale) {
    corr_lookup_body<128, 3, 4>(f1, l0, l1, l2, l3, h0, w0, coords, out, ostride, scale);
}
// NeuFlow v2: one level, C = 128, radius 4 (81 channels); f1 is the query image's features, l0 the other image's.
extern "C" __global__ void __launch_bounds__(256) corr_lookup_c128_r4_l1(const __half* f1, const __half* l0, int h0, int w0,
                                                                         const float* coords, __half* out, int ostride,
                                                                         float scale) {
    corr_lookup_body<128, 4, 1>(f1, l0, l0, l0, l0, h0, w0, coords, out, ostride, scale);
}

// ---------------------------------------------------------------------------------------------------------------
extern "C" __global__ void init_coords(float* __restrict__ coords, int W, long long P) {
    GRID_STRIDE(p, P) {
        coords[2 * p] = (float)(p % W);
        coords[2 * p + 1] = (float)(p / W);
    }
}

// The flow (coords - grid, rounded to fp16 as the network input) into channels [0, n) of one or two slices, padded
// with zeros after the two flow channels (n = 2 or 8).
extern "C" __global__ void write_flow(const float* __restrict__ coords, int W, long long P, __half* dst, __half* dst2,
                                      int dstride, int n) {
    GRID_STRIDE(p, P) {
        const __half fx = __float2half(coords[2 * p] - (float)(p % W)), fy = __float2half(coords[2 * p + 1] - (float)(p / W));
        if (n == 8) {
            __align__(16) __half v[8] = {fx, fy};
#pragma unroll
            for (int k = 2; k < 8; ++k) v[k] = __float2half(0.f);
            *reinterpret_cast<uint4*>(dst + p * dstride) = *reinterpret_cast<const uint4*>(v);
            if (dst2) *reinterpret_cast<uint4*>(dst2 + p * dstride) = *reinterpret_cast<const uint4*>(v);
        } else {
            const __half2 v = __halves2half2(fx, fy);
            *reinterpret_cast<__half2*>(dst + p * dstride) = v;
            if (dst2) *reinterpret_cast<__half2*>(dst2 + p * dstride) = v;
        }
    }
}

// convflow1: 7x7 conv (padding 3) of the 2-channel flow -> relu, K outputs (64 per blockIdx.z), fp16-rounded operands
// and fp32 accumulation (as cuDNN under autocast). w: fp32 [K][2][7][7], b: fp32 [K]; out: packed [P][K].
#define F1_TILE 16
extern "C" __global__ void convflow1(const float* __restrict__ coords, int H, int W, const float* __restrict__ w,
                                     const float* __restrict__ bias, __half* __restrict__ out, int K) {
    __shared__ float tile[2][F1_TILE + 6][F1_TILE + 6];
    __shared__ float wsh[98][64];  // [channel * 49 + tap][output channel]
    __shared__ float bsh[64];
    const int tx = threadIdx.x, ty = threadIdx.y, t = ty * F1_TILE + tx;
    const int x0 = blockIdx.x * F1_TILE, y0 = blockIdx.y * F1_TILE, k0 = blockIdx.z * 64;
    for (int i = t; i < 64 * 98; i += F1_TILE * F1_TILE) {
        const int k = i / 98, j = i % 98;
        wsh[j][k] = __half2float(__float2half(w[(long long)(k0 + k) * 98 + j]));
    }
    if (t < 64) bsh[t] = __half2float(__float2half(bias[k0 + t]));
    for (int i = t; i < (F1_TILE + 6) * (F1_TILE + 6); i += F1_TILE * F1_TILE) {
        const int ly = i / (F1_TILE + 6), lx = i % (F1_TILE + 6);
        const int gy = y0 + ly - 3, gx = x0 + lx - 3;
        float vx = 0.f, vy = 0.f;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const long long q = (long long)gy * W + gx;
            vx = __half2float(__float2half(coords[2 * q] - (float)gx));
            vy = __half2float(__float2half(coords[2 * q + 1] - (float)gy));
        }
        tile[0][ly][lx] = vx;
        tile[1][ly][lx] = vy;
    }
    __syncthreads();
    const int y = y0 + ty, x = x0 + tx;
    if (y >= H || x >= W) return;
    uint4* dst = reinterpret_cast<uint4*>(out + ((long long)y * W + x) * K + k0);
#pragma unroll 1
    for (int kb = 0; kb < 64; kb += 8) {
        float acc[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] = bsh[kb + k];
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int j = 0; j < 49; ++j) {
                const float in = tile[c][ty + j / 7][tx + j % 7];
                const float4 w0 = *reinterpret_cast<const float4*>(&wsh[c * 49 + j][kb]);
                const float4 w1 = *reinterpret_cast<const float4*>(&wsh[c * 49 + j][kb + 4]);
                acc[0] += w0.x * in; acc[1] += w0.y * in; acc[2] += w0.z * in; acc[3] += w0.w * in;
                acc[4] += w1.x * in; acc[5] += w1.y * in; acc[6] += w1.z * in; acc[7] += w1.w * in;
            }
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __float2half(fmaxf(acc[k], 0.f));
        dst[kb / 8] = *reinterpret_cast<const uint4*>(o);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// ConvGRU gates for any hidden size HD (% 8 == 0), after a raw z|r conv (zr: [P][zs], z in [0, HD), r in [HD, 2HD)):
// z = sigmoid(zr + b) written back into zr[:HD]; rh = sigmoid(r + b) * h written into the rh slice (stride hs, as h).
extern "C" __global__ void gru_zr(__half* zr, int zs, const float* __restrict__ b, const __half* __restrict__ h,
                                  __half* __restrict__ rh, int hs, long long P, int HD) {
    const int CV = HD / 8;
    GRID_STRIDE(i, P * CV) {
        const long long p = i / CV;
        const int c = (int)(i % CV) * 8;
        const uint4 zv = *reinterpret_cast<const uint4*>(zr + p * zs + c);
        const uint4 rv = *reinterpret_cast<const uint4*>(zr + p * zs + HD + c);
        const uint4 hv = *reinterpret_cast<const uint4*>(h + p * hs + c);
        const __half *zh = reinterpret_cast<const __half*>(&zv), *rr = reinterpret_cast<const __half*>(&rv);
        const __half* hh = reinterpret_cast<const __half*>(&hv);
        __align__(16) __half oz[8], orh[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float zp = __half2float(__float2half(__half2float(zh[k]) + b[c + k]));
            const float rp = __half2float(__float2half(__half2float(rr[k]) + b[HD + c + k]));
            oz[k] = __float2half(1.f / (1.f + __expf(-zp)));
            orh[k] = __hmul(__float2half(1.f / (1.f + __expf(-rp))), hh[k]);
        }
        *reinterpret_cast<uint4*>(zr + p * zs + c) = *reinterpret_cast<const uint4*>(oz);
        *reinterpret_cast<uint4*>(rh + p * hs + c) = *reinterpret_cast<const uint4*>(orh);
    }
}

// State update after a raw q conv (q: [P][HD]): h = h + z (tanh(q + b) - h) in place in the h slice, optionally also
// into a packed copy h2 ([P][HD]).
extern "C" __global__ void gru_update(const __half* __restrict__ q, const float* __restrict__ b, const __half* __restrict__ zr,
                                      int zs, __half* h, int hs, __half* h2, long long P, int HD) {
    const int CV = HD / 8;
    GRID_STRIDE(i, P * CV) {
        const long long p = i / CV;
        const int c = (int)(i % CV) * 8;
        const uint4 qv = *reinterpret_cast<const uint4*>(q + p * HD + c);
        const uint4 zv = *reinterpret_cast<const uint4*>(zr + p * zs + c), hv = *reinterpret_cast<const uint4*>(h + p * hs + c);
        const __half *qh = reinterpret_cast<const __half*>(&qv), *zh = reinterpret_cast<const __half*>(&zv);
        const __half* hh = reinterpret_cast<const __half*>(&hv);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float qq = tanhf(__half2float(__float2half(__half2float(qh[k]) + b[c + k])));
            const float hf = __half2float(hh[k]), z = __half2float(zh[k]);
            o[k] = __float2half(hf + z * (qq - hf));
        }
        *reinterpret_cast<uint4*>(h + p * hs + c) = *reinterpret_cast<const uint4*>(o);
        if (h2) *reinterpret_cast<uint4*>(h2 + p * HD + c) = *reinterpret_cast<const uint4*>(o);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Flow head output conv (3x3, FH -> 2, padding 1) fused with the coords update: coords[p] += fp16(b + conv(fh)).
// fh: packed [P][FH] (relu already applied); w: KRSC fp16 [2][3][3][FH]; b: fp16 [2]. One warp per pixel, each lane
// covers FH / 32 channels (8 or 4), then a warp reduction.
template <int FH>
__device__ __forceinline__ void flow_head_out_body(const __half* __restrict__ fh, int H, int W, const __half* __restrict__ w,
                                                   const __half* __restrict__ b, float* __restrict__ coords) {
    constexpr int V = FH / 32;
    const int lane = threadIdx.x & 31;
    const long long p = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (p >= (long long)H * W) return;
    const int y = (int)(p / W), x = (int)(p % W);
    float ax = 0.f, ay = 0.f;
#pragma unroll
    for (int t = 0; t < 9; ++t) {
        const int ny = y + t / 3 - 1, nx = x + t % 3 - 1;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
        __align__(16) __half f[V], w0[V], w1[V];
        const __half* fs = fh + ((long long)ny * W + nx) * FH + lane * V;
        const __half* ws0 = w + t * FH + lane * V;
        const __half* ws1 = w + 9 * FH + t * FH + lane * V;
        if constexpr (V == 8) {
            *reinterpret_cast<uint4*>(f) = *reinterpret_cast<const uint4*>(fs);
            *reinterpret_cast<uint4*>(w0) = *reinterpret_cast<const uint4*>(ws0);
            *reinterpret_cast<uint4*>(w1) = *reinterpret_cast<const uint4*>(ws1);
        } else {
            *reinterpret_cast<uint2*>(f) = *reinterpret_cast<const uint2*>(fs);
            *reinterpret_cast<uint2*>(w0) = *reinterpret_cast<const uint2*>(ws0);
            *reinterpret_cast<uint2*>(w1) = *reinterpret_cast<const uint2*>(ws1);
        }
#pragma unroll
        for (int k = 0; k < V; ++k) {
            const float v = __half2float(f[k]);
            ax = fmaf(v, __half2float(w0[k]), ax);
            ay = fmaf(v, __half2float(w1[k]), ay);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ax += __shfl_xor_sync(0xffffffffu, ax, o);
        ay += __shfl_xor_sync(0xffffffffu, ay, o);
    }
    if (lane == 0) {
        coords[2 * p] += __half2float(__float2half(ax + __half2float(b[0])));
        coords[2 * p + 1] += __half2float(__float2half(ay + __half2float(b[1])));
    }
}
extern "C" __global__ void flow_head_out_256(const __half* fh, int H, int W, const __half* w, const __half* b, float* coords) {
    flow_head_out_body<256>(fh, H, W, w, b, coords);
}
extern "C" __global__ void flow_head_out_128(const __half* fh, int H, int W, const __half* w, const __half* b, float* coords) {
    flow_head_out_body<128>(fh, H, W, w, b, coords);
}

// ---------------------------------------------------------------------------------------------------------------
// Convex 8x upsampling of the flow with the learned mask: mask [h*w][576] fp16 raw conv output (scaled by 0.25 in fp16
// here, as the reference), channel = k*64 + a*8 + b; softmax over the 9 neighbours k in fp32.
// out: float32 [2][8h][8w].
extern "C" __global__ void upsample_convex(const float* __restrict__ coords, const __half* __restrict__ mask, int h,
                                           int w, float* __restrict__ out) {
    const int H = 8 * h, W = 8 * w;
    GRID_STRIDE(i, (long long)H * W) {
        const int Y = (int)(i / W), X = (int)(i % W);
        const int yy = Y >> 3, a = Y & 7, xx = X >> 3, bb = X & 7;
        const __half* m = mask + ((long long)yy * w + xx) * 576 + a * 8 + bb;
        float mk[9], mx = -1e30f;
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            mk[k] = __half2float(__hmul(__float2half(0.25f), m[k * 64]));
            mx = fmaxf(mx, mk[k]);
        }
        float den = 0.f;
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            mk[k] = expf(mk[k] - mx);
            den += mk[k];
        }
        float ax = 0.f, ay = 0.f;
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            const int ny = yy + k / 3 - 1, nx = xx + k % 3 - 1;
            if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                const long long q = (long long)ny * w + nx;
                const float wk = mk[k] / den;
                ax += wk * (8.f * (coords[2 * q] - (float)nx));
                ay += wk * (8.f * (coords[2 * q + 1] - (float)ny));
            }
        }
        out[i] = ax;
        out[(long long)H * W + i] = ay;
    }
}

// Bilinear 8x upsampling (align_corners=True, PyTorch's formula) of 8 * flow: out float32 [2][8h][8w].
extern "C" __global__ void upsample_bilinear(const float* __restrict__ coords, int h, int w, float* __restrict__ out) {
    const int H = 8 * h, W = 8 * w;
    const float rh = H > 1 ? (float)(h - 1) / (float)(H - 1) : 0.f;
    const float rw = W > 1 ? (float)(w - 1) / (float)(W - 1) : 0.f;
    GRID_STRIDE(i, (long long)H * W) {
        const int Y = (int)(i / W), X = (int)(i % W);
        const float h1r = rh * (float)Y, w1r = rw * (float)X;
        const int h1 = (int)h1r, w1 = (int)w1r;
        const int h1p = h1 < h - 1 ? 1 : 0, w1p = w1 < w - 1 ? 1 : 0;
        const float h1l = h1r - (float)h1, w1l = w1r - (float)w1, h0l = 1.f - h1l, w0l = 1.f - w1l;
        const long long q00 = (long long)h1 * w + w1, q01 = q00 + w1p, q10 = q00 + (long long)h1p * w, q11 = q10 + w1p;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            auto f = [&](long long q) { return coords[2 * q + c] - (float)(c == 0 ? q % w : q / w); };
            const float v = h0l * (w0l * f(q00) + w1l * f(q01)) + h1l * (w0l * f(q10) + w1l * f(q11));
            out[(long long)c * H * W + i] = 8.f * v;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// SEA-RAFT ConvNeXt block front: 7 x 7 depthwise conv (zero padding 3, fp32 bias) of C channels of a slice x (pixel
// stride xs), rounded to fp16 as the autocast conv output, then LayerNorm over the C channels in fp32 (two-pass,
// weight g, bias be, eps) -> packed fp16 out [P][C]. w: fp32 [49][C] (tap-major). Block (C / 8, DW_ROWS): one thread
// per 8 channels of DW_TX consecutive pixels of one row, so each input vector is loaded once per tap row.
#define DW_TX 8
#define DW_ROWS 4
template <int C>
__device__ __forceinline__ void dwconv7_ln_body(const __half* __restrict__ x, int xs, int H, int W,
                                                const float* __restrict__ w, const float* __restrict__ b,
                                                const float* __restrict__ g, const float* __restrict__ be, float eps,
                                                __half* __restrict__ out) {
    constexpr int CV = C / 8;
    __shared__ float red[DW_ROWS][DW_TX][CV];
    __shared__ float stat[DW_ROWS][DW_TX];
    const int cv = threadIdx.x, ty = threadIdx.y, c0 = cv * 8;
    const int spr = (W + DW_TX - 1) / DW_TX;
    const long long seg = (long long)blockIdx.x * DW_ROWS + ty;
    const bool live = seg < (long long)H * spr;
    const int y = live ? (int)(seg / spr) : 0, x0 = live ? (int)(seg % spr) * DW_TX : 0;
    float acc[DW_TX][8];
#pragma unroll
    for (int o = 0; o < DW_TX; ++o)
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[o][k] = b[c0 + k];
    if (live) {
#pragma unroll 1
        for (int ky = 0; ky < 7; ++ky) {
            const int iy = y + ky - 3;
            if (iy < 0 || iy >= H) continue;
            float wr[7][8];
#pragma unroll
            for (int kx = 0; kx < 7; ++kx) {
                const float4 w0 = *reinterpret_cast<const float4*>(w + (ky * 7 + kx) * C + c0);
                const float4 w1 = *reinterpret_cast<const float4*>(w + (ky * 7 + kx) * C + c0 + 4);
                wr[kx][0] = w0.x; wr[kx][1] = w0.y; wr[kx][2] = w0.z; wr[kx][3] = w0.w;
                wr[kx][4] = w1.x; wr[kx][5] = w1.y; wr[kx][6] = w1.z; wr[kx][7] = w1.w;
            }
#pragma unroll
            for (int ix = 0; ix < DW_TX + 6; ++ix) {
                const int gx = x0 + ix - 3;
                if (gx < 0 || gx >= W) continue;
                const uint4 v = *reinterpret_cast<const uint4*>(x + ((long long)iy * W + gx) * xs + c0);
                const __half* hv = reinterpret_cast<const __half*>(&v);
                float f[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) f[k] = __half2float(hv[k]);
#pragma unroll
                for (int kx = 0; kx < 7; ++kx) {
                    const int o = ix - kx;
                    if (o < 0 || o >= DW_TX) continue;
#pragma unroll
                    for (int k = 0; k < 8; ++k) acc[o][k] = fmaf(f[k], wr[kx][k], acc[o][k]);
                }
            }
        }
    }
    // fp16 rounding of the conv output, then the mean
#pragma unroll
    for (int o = 0; o < DW_TX; ++o) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc[o][k] = __half2float(__float2half(acc[o][k]));
            s += acc[o][k];
        }
        red[ty][o][cv] = s;
    }
    __syncthreads();
    if (cv < DW_TX) {
        float s = 0.f;
        for (int i = 0; i < CV; ++i) s += red[ty][cv][i];
        stat[ty][cv] = s / (float)C;
    }
    __syncthreads();
#pragma unroll
    for (int o = 0; o < DW_TX; ++o) {
        const float m = stat[ty][o];
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) s += (acc[o][k] - m) * (acc[o][k] - m);
        red[ty][o][cv] = s;
    }
    __syncthreads();
    float mean[DW_TX];
#pragma unroll
    for (int o = 0; o < DW_TX; ++o) mean[o] = stat[ty][o];
    __syncthreads();
    if (cv < DW_TX) {
        float s = 0.f;
        for (int i = 0; i < CV; ++i) s += red[ty][cv][i];
        stat[ty][cv] = rsqrtf(s / (float)C + eps);
    }
    __syncthreads();
    if (!live) return;
    float gg[8], bb[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        gg[k] = g[c0 + k];
        bb[k] = be[c0 + k];
    }
#pragma unroll
    for (int o = 0; o < DW_TX; ++o) {
        if (x0 + o >= W) break;
        const float r = stat[ty][o];
        __align__(16) __half v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = __float2half((acc[o][k] - mean[o]) * r * gg[k] + bb[k]);
        *reinterpret_cast<uint4*>(out + ((long long)y * W + x0 + o) * C + c0) = *reinterpret_cast<const uint4*>(v);
    }
}
extern "C" __global__ void __launch_bounds__(384 / 8 * DW_ROWS) dwconv7_ln_384(
    const __half* x, int xs, int H, int W, const float* w, const float* b, const float* g, const float* be, float eps,
    __half* out) {
    dwconv7_ln_body<384>(x, xs, H, W, w, b, g, be, eps, out);
}
