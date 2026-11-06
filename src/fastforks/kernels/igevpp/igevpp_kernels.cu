// IGEV++ / GREAT-IGEV max-path kernels (size-independent; built into one cubin per target by tools/build_kernels.py).
// New for fastforks: group-wise correlation volume, padded / single-level geometry lookups + the geometry-encoder
// blend (IGEV++); LayerNorm over rows, fused volume attention + feed-forward, final volume softmax + regression,
// outlook attention + fold, the split excitive-attention concat-volume conv (GREAT-IGEV). fp16 storage, fp32 math.
#include <cuda_fp16.h>
#include <stdint.h>

#define GRID_STRIDE(i, n) \
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (n); i += (long long)gridDim.x * blockDim.x)

__device__ __forceinline__ void load8(const __half* p, float* f) {
    uint4 v = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float2 t = __half22float2(h[k]);
        f[2 * k] = t.x;
        f[2 * k + 1] = t.y;
    }
}

__device__ __forceinline__ void store8(__half* p, const float* f) {
    uint4 v;
    __half2* h = reinterpret_cast<__half2*>(&v);
#pragma unroll
    for (int k = 0; k < 4; ++k) h[k] = __floats2half2_rn(f[2 * k], f[2 * k + 1]);
    *reinterpret_cast<uint4*>(p) = v;
}

template <int C>
__device__ __forceinline__ void loadc(const __half* p, float* f) {
#pragma unroll
    for (int k = 0; k < C / 8; ++k) load8(p + 8 * k, f + 8 * k);
}

template <int C>
__device__ __forceinline__ void storec(__half* p, const float* f) {
#pragma unroll
    for (int k = 0; k < C / 8; ++k) store8(p + 8 * k, f + 8 * k);
}

// (x - mean) * rstd in place, fp32, biased variance (two-pass)
template <int C>
__device__ __forceinline__ void norm_rows(float* x, float eps) {
    float m = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) m += x[c];
    m *= 1.f / C;
    float v = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float d = x[c] - m;
        v += d * d;
    }
    float r = rsqrtf(v * (1.f / C) + eps);
#pragma unroll
    for (int c = 0; c < C; ++c) x[c] = (x[c] - m) * r;
}

// ------------------------------------------------------------------------------------------------ IGEV++
// Group-wise correlation volume (8 groups): out[b, d, y, x, g] = mean_c L[b, y, x, g*cpg + c] * R[b, y, x - d, ...],
// 0 for x < d. L, R fp16 NHWC rows of C = 96 channels (row stride ls / rs); out fp16 NDHWC [B, D, H, W, 8].
// One thread per (b, chunk of dchunk disparities, y, x): L in registers, R rows read along d.
extern "C" __global__ void gwc_volume96(const __half* __restrict__ L, int ls, const __half* __restrict__ R, int rs,
                                       __half* __restrict__ out, int B, int H, int W, int D, int dchunk) {
    constexpr int C = 96, G = 8, CPG = C / G;
    const int nch = (D + dchunk - 1) / dchunk;
    GRID_STRIDE(i, (long long)B * nch * H * W) {
        int x = (int)(i % W);
        long long t = i / W;
        int y = (int)(t % H);
        t /= H;
        int ch = (int)(t % nch);
        int b = (int)(t / nch);
        long long row = ((long long)b * H + y) * W;
        float l[C];
        loadc<C>(L + (row + x) * ls, l);
        int d0 = ch * dchunk, d1 = min(D, d0 + dchunk);
        for (int d = d0; d < d1; ++d) {
            float acc[G];
#pragma unroll
            for (int g = 0; g < G; ++g) acc[g] = 0.f;
            if (x >= d) {
                const __half* r = R + (row + x - d) * rs;
#pragma unroll
                for (int k = 0; k < C / 8; ++k) {
                    float f[8];
                    load8(r + 8 * k, f);
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[(8 * k + j) / CPG] += l[8 * k + j] * f[j];
                }
#pragma unroll
                for (int g = 0; g < G; ++g) acc[g] *= 1.f / CPG;
            }
            store8(out + ((((long long)b * D + d) * H + y) * W + x) * G, acc);
        }
    }
}

__device__ __forceinline__ float lvl_val(const __half* v, int i, int n, int l) {
    if (l == 0) return (i >= 0 && i < n) ? __half2float(v[i]) : 0.f;
    return (i >= 0 && i < n / 2) ? 0.5f * (__half2float(v[2 * i]) + __half2float(v[2 * i + 1])) : 0.f;
}

// Geometry lookup as igev's geo_lookup (same values), with `nl` levels (1 or 2), the correlation rows optional
// (`wcorr`), the lookup at disp * dscale, and rows of `ostride` >= nl * (C + wcorr) * K channels (rest zero).
// Per level: geo (c*K + k), then corr (k). g0 [P, C, D0] fp16, c0 [P, W0] fp16, disp fp32 [P], out fp16 [P, ostride].
extern "C" __global__ void geo_lookup8(const __half* __restrict__ g0, int C, int D0, const __half* __restrict__ c0,
                                      int W0, int Wimg, const float* __restrict__ disp, float dscale,
                                      __half* __restrict__ out, int ostride, long long P, int r, int nl, int wcorr) {
    const int K = 2 * r + 1;
    const int per = C + wcorr, used = nl * per;
    GRID_STRIDE(i, P * (used + 1)) {
        long long p = i / (used + 1);
        int jj = (int)(i - p * (used + 1));
        if (jj == used) {
            for (int c = used * K; c < ostride; ++c) out[p * ostride + c] = __float2half(0.f);
            continue;
        }
        int l = jj / per, j = jj - l * per;
        float sc = l ? 0.5f : 1.f;
        float d = disp[p] * dscale * sc;
        __half* o = out + p * ostride + l * per * K + j * K;
        const __half* v;
        int n;
        float base;
        if (j < C) {
            v = g0 + (p * C + j) * D0;
            n = D0;
            base = d;
        } else {
            v = c0 + p * W0;
            n = W0;
            base = (float)(p % Wimg) * sc - d;
        }
        for (int k = 0; k < K; ++k) {
            float x = base + (float)(k - r);
            float xf = floorf(x);
            int x0 = (int)xf;
            float w = x - xf;
            o[k] = __float2half(lvl_val(v, x0, n, l) * (1.f - w) + lvl_val(v, x0 + 1, n, l) * w);
        }
    }
}

// Selective blend of the three geometry encodings + the motion-encoder input row:
// xb[p, :ce] = e0 * sw0 + e1 * sw1 + e2 * sw2 (fp32), xb[p, ce + k] = lk[p, cl + k], xb[p, ce + K + k] =
// lk[p, 2*cl + K + k] (the two correlation levels of the small-range lookup). ce % 8 == 0, row strides in elements.
extern "C" __global__ void geo_blend(const __half* __restrict__ e0, const __half* __restrict__ e1,
                                    const __half* __restrict__ e2, int es, int ce, const __half* __restrict__ sw,
                                    int sws, const __half* __restrict__ lk, int ls, int cl, int K, __half* xb, int xs,
                                    long long P) {
    const int nv = ce / 8 + 1;
    GRID_STRIDE(i, P * nv) {
        long long p = i / nv;
        int q = (int)(i - p * nv);
        if (q < ce / 8) {
            float w0 = __half2float(sw[p * sws]), w1 = __half2float(sw[p * sws + 1]), w2 = __half2float(sw[p * sws + 2]);
            float a[8], b[8], c[8];
            load8(e0 + p * es + 8 * q, a);
            load8(e1 + p * es + 8 * q, b);
            load8(e2 + p * es + 8 * q, c);
#pragma unroll
            for (int k = 0; k < 8; ++k) a[k] = a[k] * w0 + b[k] * w1 + c[k] * w2;
            store8(xb + p * xs + 8 * q, a);
        } else {
            for (int k = 0; k < K; ++k) {
                xb[p * xs + ce + k] = lk[p * ls + cl + k];
                xb[p * xs + ce + K + k] = lk[p * ls + 2 * cl + K + k];
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ GREAT-IGEV
// LayerNorm over the last dim of rows (fp16 in / out, fp32 statistics + affine, C % 8 == 0).
extern "C" __global__ void layer_norm_rows(const __half* __restrict__ x, int xs, __half* __restrict__ y, int ys,
                                          const float* __restrict__ w, const float* __restrict__ b, long long P, int C,
                                          float eps) {
    GRID_STRIDE(p, P) {
        const __half* xr = x + p * xs;
        float f[8], m = 0.f, v = 0.f;
        for (int k = 0; k < C; k += 8) {
            load8(xr + k, f);
#pragma unroll
            for (int j = 0; j < 8; ++j) m += f[j];
        }
        m /= C;
        for (int k = 0; k < C; k += 8) {
            load8(xr + k, f);
#pragma unroll
            for (int j = 0; j < 8; ++j) v += (f[j] - m) * (f[j] - m);
        }
        float r = rsqrtf(v / C + eps);
        for (int k = 0; k < C; k += 8) {
            load8(xr + k, f);
#pragma unroll
            for (int j = 0; j < 8; ++j) f[j] = (f[j] - m) * r * w[k + j] + b[k + j];
            store8(y + p * ys + k, f);
        }
    }
}

// Volume attention, pass 1: per pixel the online softmax statistics (max, sum exp) over D of the logits
// LN(t[b, d, pix]) . g[b, pix] (LN without affine). t fp16 [B, D, HW, C], g fp32 [B, HW, C].
template <int C>
__device__ void vol_stats_body(const __half* __restrict__ t, const float* __restrict__ g, float* __restrict__ st,
                               int B, int D, long long HW, float ln_eps) {
    GRID_STRIDE(i, (long long)B * HW) {
        long long b = i / HW, pix = i - b * HW;
        float gg[C];
#pragma unroll
        for (int c = 0; c < C; ++c) gg[c] = g[i * C + c];
        float mx = -INFINITY, s = 0.f;
        for (int d = 0; d < D; ++d) {
            float x[C];
            loadc<C>(t + ((b * D + d) * HW + pix) * C, x);
            norm_rows<C>(x, ln_eps);
            float lg = 0.f;
#pragma unroll
            for (int c = 0; c < C; ++c) lg += x[c] * gg[c];
            if (lg > mx) {
                s = s * __expf(mx - lg) + 1.f;
                mx = lg;
            } else {
                s += __expf(lg - mx);
            }
        }
        st[2 * i] = mx;
        st[2 * i + 1] = s;
    }
}

// Volume attention, pass 2 + feed-forward, per token, in place on t:
//   y = fp16(x + p * vo[b, pix] + bo), p = (softmax_D + eps) / (1 + D * eps);  t = y + W2 silu(W1 LN'(y) + b1) + b2
// (LN' with affine lw / lb, eps ln2). prm (fp32): bo[C] | lw[C] | lb[C] | b1[2C] | b2[C] | W1[2C, C] | W2[C, 2C].
template <int C>
__device__ void vol_update_body(__half* t, const float* __restrict__ g, const float* __restrict__ st,
                                const float* __restrict__ vo, const float* __restrict__ prm, int B, int D,
                                long long HW, float ln_eps, float ln2, float eps) {
    extern __shared__ float sm[];
    constexpr int NP = 5 * C + 4 * C * C + C;  // bo lw lb b1(2C) b2 W1 W2
    for (int k = threadIdx.x; k < NP; k += blockDim.x) sm[k] = prm[k];
    __syncthreads();
    const float *bo = sm, *lw = sm + C, *lb = sm + 2 * C, *b1 = sm + 3 * C, *b2 = sm + 5 * C, *W1 = sm + 6 * C,
                *W2 = sm + 6 * C + 2 * C * C;
    const float pe = 1.f / (1.f + D * eps);
    GRID_STRIDE(i, (long long)B * D * HW) {
        long long pix = i % HW, bd = i / HW, b = bd / D;
        long long q = b * HW + pix;
        __half* tp = t + i * C;
        float x[C], z[C];
        loadc<C>(tp, x);
#pragma unroll
        for (int c = 0; c < C; ++c) z[c] = x[c];
        norm_rows<C>(z, ln_eps);
        float lg = 0.f;
#pragma unroll
        for (int c = 0; c < C; ++c) lg += z[c] * g[q * C + c];
        float p = (__expf(lg - st[2 * q]) / st[2 * q + 1] + eps) * pe;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            x[c] = __half2float(__float2half(x[c] + p * vo[q * C + c] + bo[c]));
            z[c] = x[c];
        }
        norm_rows<C>(z, ln2);
#pragma unroll
        for (int c = 0; c < C; ++c) z[c] = z[c] * lw[c] + lb[c];
        float o[C];
#pragma unroll
        for (int c = 0; c < C; ++c) o[c] = b2[c];
        for (int j = 0; j < 2 * C; ++j) {
            float h = b1[j];
#pragma unroll
            for (int c = 0; c < C; ++c) h += W1[j * C + c] * z[c];
            h = h / (1.f + __expf(-h));
#pragma unroll
            for (int c = 0; c < C; ++c) o[c] += W2[c * 2 * C + j] * h;
        }
#pragma unroll
        for (int c = 0; c < C; ++c) x[c] += o[c];
        storec<C>(tp, x);
    }
}

#define VOL_KERNELS(NC)                                                                                                \
    extern "C" __global__ void vol_stats_c##NC(const __half* t, const float* g, float* st, int B, int D, long long HW, \
                                              float ln_eps) {                                                          \
        vol_stats_body<NC>(t, g, st, B, D, HW, ln_eps);                                                                \
    }                                                                                                                  \
    extern "C" __global__ void vol_update_c##NC(__half* t, const float* g, const float* st, const float* vo,           \
                                               const float* prm, int B, int D, long long HW, float ln_eps, float ln2,  \
                                               float eps) {                                                            \
        vol_update_body<NC>(t, g, st, vo, prm, B, D, HW, ln_eps, ln2, eps);                                            \
    }
VOL_KERNELS(8)
VOL_KERNELS(16)
VOL_KERNELS(32)
VOL_KERNELS(48)

// Final volume softmax over D (+ eps, renormalised) of the logits t . g and the disparity regression sum_d p_d * d:
// disp fp32 [B, HW]. C == 8.
extern "C" __global__ void vol_regress_c8(const __half* __restrict__ t, const float* __restrict__ g,
                                         float* __restrict__ disp, int B, int D, long long HW, float eps) {
    constexpr int C = 8;
    GRID_STRIDE(i, (long long)B * HW) {
        long long b = i / HW, pix = i - b * HW;
        float gg[C];
#pragma unroll
        for (int c = 0; c < C; ++c) gg[c] = g[i * C + c];
        float mx = -INFINITY, s = 0.f, sd = 0.f;  // sum exp, sum exp * d (rescaled with the running max)
        for (int d = 0; d < D; ++d) {
            float x[C];
            load8(t + ((b * D + d) * HW + pix) * C, x);
            float lg = 0.f;
#pragma unroll
            for (int c = 0; c < C; ++c) lg += x[c] * gg[c];
            if (lg > mx) {
                float k = __expf(mx - lg);
                s = s * k + 1.f;
                sd = sd * k + (float)d;
                mx = lg;
            } else {
                float e = __expf(lg - mx);
                s += e;
                sd += e * (float)d;
            }
        }
        // sum_d ((e_d / s) + eps) * d / (1 + D eps)
        disp[i] = (sd / s + eps * 0.5f * (float)D * (float)(D - 1)) / (1.f + D * eps);
    }
}

// Outlook attention (stride 1, one head), per (b, pixel m, window row r): logits of "pixel" m are the flat elements
// [m*KK*KK, (m+1)*KK*KK) of the NCHW [B, KK*KK, H, W] to_attn output (upstream's reshape): element f = channel f / N
// at pixel f % N, = wa[c] . xn[b, f % N] + ba[c]; p = softmax(lg * scale) + eps renormalised; O[b, m, r] =
// sum_j p_j v[b, window pixel j of m] (zero outside). xn, v fp16 [B, N, C] rows (strides xs, vs); wa fp32 [KK*KK, C].
template <int K, int C>
__device__ void outlook_body(const __half* __restrict__ xn, int xs, const __half* __restrict__ v, int vs,
                             const float* __restrict__ wa, const float* __restrict__ ba, __half* __restrict__ O, int B,
                             int H, int W, float scale, float eps) {
    constexpr int KK = K * K, pad = K / 2;
    const long long N = (long long)H * W;
    GRID_STRIDE(i, (long long)B * N * KK) {
        long long bm = i / KK;
        int r = (int)(i - bm * KK);
        long long b = bm / N, m = bm - b * N;
        long long f = m * KK * KK + (long long)r * KK;
        float lg[KK], mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < KK; ++j) {
            long long c = (f + j) / N, q = (f + j) - c * N;
            float x[C];
            loadc<C>(xn + (b * N + q) * xs, x);
            float s = ba[c];
#pragma unroll
            for (int k = 0; k < C; ++k) s += wa[c * C + k] * x[k];
            lg[j] = s * scale;
            mx = fmaxf(mx, lg[j]);
        }
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < KK; ++j) {
            lg[j] = __expf(lg[j] - mx);
            sum += lg[j];
        }
        const float pe = 1.f / (1.f + KK * eps), inv = 1.f / sum;
        int y = (int)(m / W), x0 = (int)(m - (long long)y * W);
        float o[C];
#pragma unroll
        for (int k = 0; k < C; ++k) o[k] = 0.f;
#pragma unroll
        for (int j = 0; j < KK; ++j) {
            int yy = y + j / K - pad, xx = x0 + j % K - pad;
            if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
            float p = __half2float(__float2half((lg[j] * inv + eps) * pe));  // fp16 weights, as upstream's matmul
            float vv[C];
            loadc<C>(v + (b * N + (long long)yy * W + xx) * vs, vv);
#pragma unroll
            for (int k = 0; k < C; ++k) o[k] += p * vv[k];
        }
        storec<C>(O + i * C, o);
    }
}

// Fold: out[b, y, x] = sum over window positions (i, j) of O[b, (y - i + pad, x - j + pad), i*K + j] (fp32 sum).
template <int K, int C>
__device__ void outlook_fold_body(const __half* __restrict__ O, __half* __restrict__ out, int os, int B, int H, int W) {
    constexpr int KK = K * K, pad = K / 2;
    const long long N = (long long)H * W;
    GRID_STRIDE(i, (long long)B * N * (C / 8)) {
        long long bp = i / (C / 8);
        int q = (int)(i - bp * (C / 8));
        long long b = bp / N, p = bp - b * N;
        int y = (int)(p / W), x = (int)(p - (long long)y * W);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int t = 0; t < KK; ++t) {
            int sy = y - t / K + pad, sx = x - t % K + pad;
            if (sy < 0 || sy >= H || sx < 0 || sx >= W) continue;
            float f[8];
            load8(O + ((b * N + (long long)sy * W + sx) * KK + t) * C + 8 * q, f);
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[k] += f[k];
        }
        store8(out + bp * os + 8 * q, acc);
    }
}

// one-parameter instantiation macro (kernel name suffix KC; OUTLOOK_<KC> holds the template arguments K, C)
#define OUTLOOK_k7_c8 7, 8
#define OUTLOOK_k5_c16 5, 16
#define OUTLOOK_k3_c32 3, 32
#define OUTLOOK_k3_c48 3, 48
#define OUTLOOK_KERNELS(KC)                                                                                            \
    extern "C" __global__ void outlook_##KC(const __half* xn, int xs, const __half* v, int vs, const float* wa,        \
                                            const float* ba, __half* O, int B, int H, int W, float scale, float eps) { \
        outlook_body<OUTLOOK_##KC>(xn, xs, v, vs, wa, ba, O, B, H, W, scale, eps);                                     \
    }                                                                                                                  \
    extern "C" __global__ void outlook_fold_##KC(const __half* O, __half* out, int os, int B, int H, int W) {          \
        outlook_fold_body<OUTLOOK_##KC>(O, out, os, B, H, W);                                                          \
    }
OUTLOOK_KERNELS(k7_c8)
OUTLOOK_KERNELS(k5_c16)
OUTLOOK_KERNELS(k3_c32)
OUTLOOK_KERNELS(k3_c48)

// Excitive-attention concat volume conv (1x3x3, [L | R shifted by d] -> C) split by linearity: plane d (a + k) =
// leaky(s * (A[y, x] + Bm[y, x - d]) + t), A = conv_L(L) [H, W, C], Bm [H, W + 1, 2C] = conv_R(R) evaluated at
// columns -1..W-1 (index u + 1), second half with the kx = +1 taps dropped (used at x = W - 1, where the shifted R
// column W - d lies outside the image); x - d < -1: no right term. out fp16 NDHWC [nd, H, W, C]. Batch 1.
extern "C" __global__ void cc_stem0(const __half* __restrict__ A, int as, const __half* __restrict__ Bm, int bs,
                                   const float* __restrict__ s, const float* __restrict__ t, __half* __restrict__ out,
                                   int a, int nd, int H, int W, int C) {
    const int CV = C / 8;
    GRID_STRIDE(i, (long long)nd * H * W * CV) {
        int q = (int)(i % CV);
        long long r = i / CV;
        int x = (int)(r % W);
        r /= W;
        int y = (int)(r % H);
        int d = a + (int)(r / H);
        float v[8];
        load8(A + ((long long)y * W + x) * as + 8 * q, v);
        int u = x - d;
        if (u >= -1) {
            float f[8];
            load8(Bm + ((long long)y * (W + 1) + u + 1) * bs + (x == W - 1 ? C : 0) + 8 * q, f);
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] += f[k];
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float z = v[k] * s[8 * q + k] + t[8 * q + k];
            v[k] = z >= 0.f ? z : 0.01f * z;
        }
        store8(out + i * 8, v);
    }
}

// FeedForwardLayer depthwise stage: the three depthwise convs (3x3, 5x5, 7x7, zero padding, + bias) of y written as
// one concatenated row [dw3 | dw5 | dw7] (3E channels), one 7x7 window read per pixel. y fp16 NHWC rows (stride ys),
// weights fp32 tap-major [k*k, E], out fp16 [B, H, W, 3E]. E % 8 == 0. One thread per (pixel, 8 channels).
extern "C" __global__ void dw357(const __half* __restrict__ y, int ys, const float* __restrict__ w3,
                                const float* __restrict__ w5, const float* __restrict__ w7, const float* __restrict__ b3,
                                const float* __restrict__ b5, const float* __restrict__ b7, __half* __restrict__ out,
                                int B, int H, int W, int E) {
    const int EV = E / 8;
    GRID_STRIDE(i, (long long)B * H * W * EV) {
        int q = (int)(i % EV);
        long long p = i / EV;
        int x = (int)(p % W);
        long long r = p / W;
        int yy0 = (int)(r % H);
        long long b = r / H;
        const int c = 8 * q;
        float a3[8], a5[8], a7[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            a3[k] = b3[c + k];
            a5[k] = b5[c + k];
            a7[k] = b7[c + k];
        }
        for (int di = 0; di < 7; ++di) {
            int sy = yy0 + di - 3;
            if (sy < 0 || sy >= H) continue;
            for (int dj = 0; dj < 7; ++dj) {
                int sx = x + dj - 3;
                if (sx < 0 || sx >= W) continue;
                float v[8];
                load8(y + ((b * H + sy) * W + sx) * ys + c, v);
                const float* t7 = w7 + (di * 7 + dj) * E + c;
#pragma unroll
                for (int k = 0; k < 8; ++k) a7[k] += t7[k] * v[k];
                if (di >= 1 && di <= 5 && dj >= 1 && dj <= 5) {
                    const float* t5 = w5 + ((di - 1) * 5 + dj - 1) * E + c;
#pragma unroll
                    for (int k = 0; k < 8; ++k) a5[k] += t5[k] * v[k];
                    if (di >= 2 && di <= 4 && dj >= 2 && dj <= 4) {
                        const float* t3 = w3 + ((di - 2) * 3 + dj - 2) * E + c;
#pragma unroll
                        for (int k = 0; k < 8; ++k) a3[k] += t3[k] * v[k];
                    }
                }
            }
        }
        __half* o = out + p * 3 * E + c;
        store8(o, a3);
        store8(o + E, a5);
        store8(o + 2 * E, a7);
    }
}
