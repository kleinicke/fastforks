// Fused bilinear resize (+ per-pixel argmax) of low-resolution segmentation scores, one or two resize steps composed.
//
// Each step is PyTorch's `upsample_bilinear2d` with align_corners=False and no explicit scale (source index
// max((dst + 0.5) * in / out - 0.5, 0), second tap clamped at the border, identity at equal size). Two steps
// (scores h x w -> M x N -> H x W, e.g. mmseg: network input size, then image size) compose into up to 3 consecutive
// source rows / columns per output pixel when the first step does not shrink (h <= M, w <= N; the caller checks).
// The math is fp32 in a fixed order (explicit fmaf / __fmul_rn), shared by `up_*` (resized scores) and
// `up_argmax_*` (their argmax), so the labels-only call is exactly the argmax of the logits call.
// scores: [B, C, h, w] with element strides (sb, sc, sy, sx) (NCHW or NHWC views); fp16 (`_h`) or fp32 (`_f`).
#include <cuda_fp16.h>
#include <stdint.h>

struct Taps {
    int i[3];
    float w[3];
};

__device__ __forceinline__ void step(int o, int in, int out, int& i0, int& i1, float& l1) {
    if (in == out) {
        i0 = i1 = o;
        l1 = 0.f;
        return;
    }
    float r = (float)in / (float)out;
    float src = fmaxf(__fmul_rn(r, (float)o + 0.5f) - 0.5f, 0.f);
    i0 = (int)src;
    i1 = i0 + (i0 < in - 1);
    l1 = fminf(fmaxf(src - (float)i0, 0.f), 1.f);
}

// taps of output index o: in -> mid -> out (steps == 2) or in -> out (steps == 1, mid ignored)
__device__ __forceinline__ Taps taps(int o, int in, int mid, int out, int steps) {
    Taps t;
    if (steps == 1) {
        int i0, i1;
        float l1;
        step(o, in, out, i0, i1, l1);
        t.i[0] = i0;
        t.i[1] = i1;
        t.i[2] = i1;
        t.w[0] = 1.f - l1;
        t.w[1] = i1 == i0 ? 0.f : l1;
        t.w[2] = 0.f;
        if (i1 == i0) t.w[0] = 1.f;  // both taps on one row: weights sum to 1 there
        return t;
    }
    int m0, m1;
    float a1;
    step(o, mid, out, m0, m1, a1);
    float a0 = 1.f - a1;
    int p0, p1, q0, q1;
    float b1, c1;
    step(m0, in, mid, p0, p1, b1);
    step(m1, in, mid, q0, q1, c1);
    float w[3] = {0.f, 0.f, 0.f};
    // same order of additions for every pixel: (m0, tap 0), (m0, tap 1), (m1, tap 0), (m1, tap 1)
    w[0] = __fmul_rn(a0, 1.f - b1);
    w[p1 - p0] = __fadd_rn(w[p1 - p0], __fmul_rn(a0, b1));
    w[q0 - p0] = __fadd_rn(w[q0 - p0], __fmul_rn(a1, 1.f - c1));
    w[q1 - p0] = __fadd_rn(w[q1 - p0], __fmul_rn(a1, c1));
    // a row PyTorch's two resizes never read (weight 0) loads row p0 instead: NaN / inf there stays out, as in PyTorch
    bool used[3] = {true, p1 != p0 || q0 == p0 + 1 || q1 == p0 + 1, q0 == p0 + 2 || q1 == p0 + 2};
    for (int k = 0; k < 3; ++k) {
        t.i[k] = used[k] ? min(p0 + k, in - 1) : p0;
        t.w[k] = w[k];
    }
    return t;
}

template <typename T>
__device__ __forceinline__ float ld(const T* p);
template <>
__device__ __forceinline__ float ld<__half>(const __half* p) { return __half2float(__ldg(p)); }
template <>
__device__ __forceinline__ float ld<float>(const float* p) { return __ldg(p); }

// NT = 2: one resize step (the third tap has weight 0 and repeats the second: not loaded); NT = 3: two steps
template <int NT, typename T>
__device__ __forceinline__ float interp(const T* p, const long long* ro, const Taps& ty, const long long* co,
                                        const Taps& tx) {
    float r[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) {
        const T* q = p + ro[k];
        float v = __fmul_rn(tx.w[0], ld(q + co[0]));
        v = fmaf(tx.w[1], ld(q + co[1]), v);
        r[k] = NT == 3 ? fmaf(tx.w[2], ld(q + co[2]), v) : v;
    }
    float v = __fmul_rn(ty.w[0], r[0]);
    v = fmaf(ty.w[1], r[1], v);
    return NT == 3 ? fmaf(ty.w[2], r[2], v) : v;
}

template <typename T, int NT>
__device__ void up_impl(const T* __restrict__ s, float* __restrict__ out, int B, int C, int h, int w, int M, int N,
                        int H, int W, int steps, long long sb, long long sc, long long sy, long long sx) {
    long long n = (long long)B * H * W;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int x = (int)(i % W);
        long long t = i / W;
        int y = (int)(t % H);
        int b = (int)(t / H);
        Taps ty = taps(y, h, M, H, steps), tx = taps(x, w, N, W, steps);
        long long ro[3], co[3];
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            ro[k] = ty.i[k] * sy;
            co[k] = tx.i[k] * sx;
        }
        const T* p = s + b * sb;
        float* o = out + (long long)b * C * H * W + (long long)y * W + x;
        for (int c = 0; c < C; ++c) o[(long long)c * H * W] = interp<NT>(p + c * sc, ro, ty, co, tx);
    }
}

// Labels: one block per TH x TW output tile, classes in chunks of CH. Pass 1: thread (row, j) resizes the tile's source
// row rlo + row along x at output column x0 + j for all CH classes of the chunk (independent loads, half2 pairs of
// classes when they are adjacent in memory) into shared memory; pass 2 along y per pixel: 3 shared loads per class
// instead of 9 global ones, the same fp32 operations in the same order as `interp` (bit-identical labels). A tile whose
// rows span more than RMAX source rows (strong downscaling) takes the per-pixel path. Launch: 256 threads, one block per
// tile (grid-stride over tiles).
constexpr int TW = 32, TH = 8, CH = 16, RMAX = 8, RS = RMAX * TW + 1;  // RMAX = 256 / TW: one pass-1 row per thread row

template <typename T>
__device__ __forceinline__ float2 ld2(const T* p);
template <>
__device__ __forceinline__ float2 ld2<__half>(const __half* p) { return __half22float2(__ldg((const __half2*)p)); }
template <>
__device__ __forceinline__ float2 ld2<float>(const float* p) { return __ldg((const float2*)p); }

template <typename T, int NT>
__device__ void up_argmax_tiled(const T* __restrict__ s, long long* __restrict__ out, int B, int C, int h, int w,
                                int M, int N, int H, int W, int steps, long long sb, long long sc, long long sy,
                                long long sx, float* __restrict__ hs) {
    int tx_ = threadIdx.x % TW, ty_ = threadIdx.x / TW;
    int nx = (W + TW - 1) / TW, ny = (H + TH - 1) / TH;
    long long ntiles = (long long)B * nx * ny;
    // adjacent classes adjacent in memory and pairs aligned: one 2-wide load per pair (uniform per launch)
    bool pairs = sc == 1 && ((sx | sy | sb) & 1) == 0 && ((uintptr_t)s % (2 * sizeof(T))) == 0;
    for (long long tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        int bx = (int)(tile % nx);
        long long t = tile / nx;
        int by = (int)(t % ny), b = (int)(t / ny);
        int x0 = bx * TW, y0 = by * TH;
        int x = x0 + tx_, y = y0 + ty_;
        int ylast = min(y0 + TH, H) - 1;
        int rlo = taps(y0, h, M, H, steps).i[0];
        int rhi = min(taps(ylast, h, M, H, steps).i[0] + 2, h - 1);
        int span = rhi - rlo + 1;
        bool inside = x < W && y < H;
        Taps ty = taps(min(y, H - 1), h, M, H, steps), tx = taps(min(x, W - 1), w, N, W, steps);
        long long co[3];
#pragma unroll
        for (int k = 0; k < 3; ++k) co[k] = tx.i[k] * sx;
        const T* p = s + b * sb;
        float best = 0.f;
        long long arg = -1;
        if (span > RMAX) {  // uniform per block
            long long ro[3];
#pragma unroll
            for (int k = 0; k < 3; ++k) ro[k] = ty.i[k] * sy;
            for (int c = 0; c < C; ++c) {
                float v = interp<NT>(p + c * sc, ro, ty, co, tx);
                if (arg < 0 || v > best || (isnan(v) && !isnan(best))) {
                    best = v;
                    arg = c;
                }
            }
        } else {
            int r[3];
#pragma unroll
            for (int k = 0; k < 3; ++k) r[k] = ty.i[k] - rlo;
            const T* q = p + (rlo + min(ty_, span - 1)) * sy;  // pass-1 source row of this thread
            for (int c0 = 0; c0 < C; c0 += CH) {
                int cn = min(CH, C - c0);
                __syncthreads();  // the previous chunk's (or tile's) pass 2 done
                if (ty_ < span) {
                    float v[CH];
                    if (pairs) {
#pragma unroll
                        for (int c = 0; c < CH; c += 2) {
                            if (c + 1 < cn) {
                                const T* qc = q + (c0 + c);
                                float2 a = ld2(qc + co[0]), e = ld2(qc + co[1]);
                                float vx = __fmul_rn(tx.w[0], a.x), vy = __fmul_rn(tx.w[0], a.y);
                                vx = fmaf(tx.w[1], e.x, vx);
                                vy = fmaf(tx.w[1], e.y, vy);
                                if (NT == 3) {
                                    float2 g = ld2(qc + co[2]);
                                    vx = fmaf(tx.w[2], g.x, vx);
                                    vy = fmaf(tx.w[2], g.y, vy);
                                }
                                v[c] = vx;
                                v[c + 1] = vy;
                            } else {  // the odd last class of the chunk (or past the end: not read in pass 2)
                                const T* qc = q + (c0 + min(c, cn - 1));
                                float a = __fmul_rn(tx.w[0], ld(qc + co[0]));
                                a = fmaf(tx.w[1], ld(qc + co[1]), a);
                                if (NT == 3) a = fmaf(tx.w[2], ld(qc + co[2]), a);
                                v[c] = v[c + 1] = a;
                            }
                        }
                    } else {
#pragma unroll
                        for (int c = 0; c < CH; ++c) {
                            const T* qc = q + (c0 + min(c, cn - 1)) * sc;
                            float a = __fmul_rn(tx.w[0], ld(qc + co[0]));
                            a = fmaf(tx.w[1], ld(qc + co[1]), a);
                            if (NT == 3) a = fmaf(tx.w[2], ld(qc + co[2]), a);
                            v[c] = a;
                        }
                    }
#pragma unroll
                    for (int c = 0; c < CH; ++c) hs[c * RS + ty_ * TW + tx_] = v[c];
                }
                __syncthreads();
                for (int c = 0; c < cn; ++c) {
                    const float* hc = hs + c * RS + tx_;
                    float v = __fmul_rn(ty.w[0], hc[r[0] * TW]);
                    v = fmaf(ty.w[1], hc[r[1] * TW], v);
                    if (NT == 3) v = fmaf(ty.w[2], hc[r[2] * TW], v);
                    if (arg < 0 || v > best || (isnan(v) && !isnan(best))) {
                        best = v;
                        arg = c0 + c;
                    }
                }
            }
        }
        if (inside) out[((long long)b * H + y) * W + x] = arg;
    }
}

// Labels, row variant: one block per BW x RPT output tile, one thread per column and all RPT rows of the tile (no
// barriers: each thread reads back only what it wrote). Per chunk of 2 * CP classes the thread resizes the tile's source
// rows rlo .. rlo + span - 1 along x at its column into its own shared slots (half2 / float2 loads of class pairs when
// adjacent in memory), then combines them along y for its RPT pixels: the same fp32 operations in the same order as
// `interp` (bit-identical labels). Tiles spanning more than RM source rows take the per-pixel path. Launch: 128 threads,
// one block per tile (grid-stride over tiles). Faster than the 32 x 8 tiles for two-step resizes (1024 x 2048 from
// 1/8 scores: 0.92 vs 1.33 ms), slower for one step (0.43 vs 0.20 ms at 512 x 683): the caller picks by steps.
constexpr int BW = 128, RPT = 8, RM = 4, CP = 4;

__device__ __forceinline__ bool better(float v, float best) { return !(v <= best) && !isnan(best); }

template <typename T, int NT>
__device__ void up_argmax_rows(const T* __restrict__ s, long long* __restrict__ out, int B, int C, int h, int w,
                               int M, int N, int H, int W, int steps, long long sb, long long sc, long long sy,
                               long long sx, float2* __restrict__ hs) {
    int tid = threadIdx.x;
    int nx = (W + BW - 1) / BW, ny = (H + RPT - 1) / RPT;
    long long ntiles = (long long)B * nx * ny;
    bool pairs = sc == 1 && ((sx | sy | sb) & 1) == 0 && ((uintptr_t)s % (2 * sizeof(T))) == 0;
    for (long long tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        int bx = (int)(tile % nx);
        long long t = tile / nx;
        int by = (int)(t % ny), b = (int)(t / ny);
        int x = bx * BW + tid, y0 = by * RPT;
        Taps tx = taps(min(x, W - 1), w, N, W, steps);
        long long co[3];
#pragma unroll
        for (int k = 0; k < 3; ++k) co[k] = tx.i[k] * sx;
        const T* p = s + b * sb;
        long long* o = out + ((long long)b * H + y0) * W + x;
        int rlo = taps(y0, h, M, H, steps).i[0];
        int rhi = min(taps(min(y0 + RPT, H) - 1, h, M, H, steps).i[0] + 2, h - 1);
        int span = rhi - rlo + 1;
        if (span > RM) {  // uniform per block
            for (int j = 0; j < RPT; ++j) {
                int y = y0 + j;
                if (y >= H) break;
                Taps ty = taps(y, h, M, H, steps);
                long long ro[3];
#pragma unroll
                for (int k = 0; k < 3; ++k) ro[k] = ty.i[k] * sy;
                float best = -INFINITY;
                int arg = 0;
                for (int c = 0; c < C; ++c) {
                    float v = interp<NT>(p + c * sc, ro, ty, co, tx);
                    if (better(v, best)) {
                        best = v;
                        arg = c;
                    }
                }
                if (x < W) o[(long long)j * W] = arg;
            }
            continue;
        }
        float wy[RPT][NT];
        int oy[RPT][NT];
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            Taps ty = taps(min(y0 + j, H - 1), h, M, H, steps);
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                wy[j][k] = ty.w[k];
                oy[j][k] = (ty.i[k] - rlo) * CP * BW + tid;
            }
        }
        float best[RPT];
        int arg[RPT];
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            best[j] = -INFINITY;
            arg[j] = 0;
        }
        for (int c0 = 0; c0 < C; c0 += 2 * CP) {
#pragma unroll
            for (int r = 0; r < RM; ++r) {
                if (r >= span) break;
                const T* q = p + (rlo + r) * sy;
#pragma unroll
                for (int cp = 0; cp < CP; ++cp) {
                    int c = c0 + 2 * cp;
                    float vx, vy;
                    if (pairs && c + 1 < C) {
                        const T* qc = q + c;
                        float2 a = ld2(qc + co[0]), e = ld2(qc + co[1]);
                        vx = __fmul_rn(tx.w[0], a.x);
                        vy = __fmul_rn(tx.w[0], a.y);
                        vx = fmaf(tx.w[1], e.x, vx);
                        vy = fmaf(tx.w[1], e.y, vy);
                        if (NT == 3) {
                            float2 g = ld2(qc + co[2]);
                            vx = fmaf(tx.w[2], g.x, vx);
                            vy = fmaf(tx.w[2], g.y, vy);
                        }
                    } else {  // classes one by one (the second of a pair past the end repeats the last: not used)
                        const T* qa = q + min(c, C - 1) * sc;
                        const T* qb = q + min(c + 1, C - 1) * sc;
                        vx = __fmul_rn(tx.w[0], ld(qa + co[0]));
                        vy = __fmul_rn(tx.w[0], ld(qb + co[0]));
                        vx = fmaf(tx.w[1], ld(qa + co[1]), vx);
                        vy = fmaf(tx.w[1], ld(qb + co[1]), vy);
                        if (NT == 3) {
                            vx = fmaf(tx.w[2], ld(qa + co[2]), vx);
                            vy = fmaf(tx.w[2], ld(qb + co[2]), vy);
                        }
                    }
                    hs[(r * CP + cp) * BW + tid] = make_float2(vx, vy);
                }
            }
            int cn = min(2 * CP, C - c0);
#pragma unroll
            for (int cp = 0; cp < CP; ++cp) {
#pragma unroll
                for (int j = 0; j < RPT; ++j) {
                    float2 a = hs[oy[j][0] + cp * BW], e = hs[oy[j][1] + cp * BW];
                    float vx = __fmul_rn(wy[j][0], a.x), vy = __fmul_rn(wy[j][0], a.y);
                    vx = fmaf(wy[j][1], e.x, vx);
                    vy = fmaf(wy[j][1], e.y, vy);
                    if (NT == 3) {
                        float2 g = hs[oy[j][NT - 1] + cp * BW];
                        vx = fmaf(wy[j][NT - 1], g.x, vx);
                        vy = fmaf(wy[j][NT - 1], g.y, vy);
                    }
                    if (2 * cp < cn && better(vx, best[j])) {
                        best[j] = vx;
                        arg[j] = c0 + 2 * cp;
                    }
                    if (2 * cp + 1 < cn && better(vy, best[j])) {
                        best[j] = vy;
                        arg[j] = c0 + 2 * cp + 1;
                    }
                }
            }
        }
        if (x < W) {
#pragma unroll
            for (int j = 0; j < RPT; ++j)
                if (y0 + j < H) o[(long long)j * W] = arg[j];
        }
    }
}

// Labels, pixel variant for channels-last scores (class pairs adjacent in memory and aligned; the caller checks): one
// thread per output pixel reads the classes of its 2 x 2 (two steps: 3 x 3) source pixels in half2 / float2 pairs
// straight from global memory (neighbouring output pixels share source pixels: L1 hits), the same fp32 operations in
// the same order as `interp` (bit-identical labels). Launch: one thread per output pixel (grid-stride).
template <typename T, int NT>
__device__ void up_argmax_pix(const T* __restrict__ s, long long* __restrict__ out, int B, int C, int h, int w, int M,
                              int N, int H, int W, int steps, long long sb, long long sy, long long sx) {
    long long n = (long long)B * H * W;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int x = (int)(i % W);
        long long t = i / W;
        int y = (int)(t % H);
        int b = (int)(t / H);
        Taps ty = taps(y, h, M, H, steps), tx = taps(x, w, N, W, steps);
        const T* q[NT][NT];
#pragma unroll
        for (int ky = 0; ky < NT; ++ky)
#pragma unroll
            for (int kx = 0; kx < NT; ++kx) q[ky][kx] = s + b * sb + ty.i[ky] * sy + tx.i[kx] * sx;
        float best = -INFINITY;
        int arg = 0;
        int c = 0;
        for (; c + 1 < C; c += 2) {
            float rx[NT], ry[NT];
#pragma unroll
            for (int ky = 0; ky < NT; ++ky) {
                float2 a = ld2(q[ky][0] + c), e = ld2(q[ky][1] + c);
                float vx = __fmul_rn(tx.w[0], a.x), vy = __fmul_rn(tx.w[0], a.y);
                vx = fmaf(tx.w[1], e.x, vx);
                vy = fmaf(tx.w[1], e.y, vy);
                if (NT == 3) {
                    float2 g = ld2(q[ky][NT - 1] + c);
                    vx = fmaf(tx.w[NT - 1], g.x, vx);
                    vy = fmaf(tx.w[NT - 1], g.y, vy);
                }
                rx[ky] = vx;
                ry[ky] = vy;
            }
            float vx = __fmul_rn(ty.w[0], rx[0]), vy = __fmul_rn(ty.w[0], ry[0]);
            vx = fmaf(ty.w[1], rx[1], vx);
            vy = fmaf(ty.w[1], ry[1], vy);
            if (NT == 3) {
                vx = fmaf(ty.w[NT - 1], rx[NT - 1], vx);
                vy = fmaf(ty.w[NT - 1], ry[NT - 1], vy);
            }
            if (better(vx, best)) {
                best = vx;
                arg = c;
            }
            if (better(vy, best)) {
                best = vy;
                arg = c + 1;
            }
        }
        if (c < C) {  // odd C: the last class alone
            float r[NT];
#pragma unroll
            for (int ky = 0; ky < NT; ++ky) {
                float v = __fmul_rn(tx.w[0], ld(q[ky][0] + c));
                v = fmaf(tx.w[1], ld(q[ky][1] + c), v);
                r[ky] = NT == 3 ? fmaf(tx.w[NT - 1], ld(q[ky][NT - 1] + c), v) : v;
            }
            float v = __fmul_rn(ty.w[0], r[0]);
            v = fmaf(ty.w[1], r[1], v);
            if (NT == 3) v = fmaf(ty.w[NT - 1], r[NT - 1], v);
            if (better(v, best)) arg = c;
        }
        out[i] = arg;
    }
}

#define UP_ARGS(T) const T *__restrict__ s, void *__restrict__ out, int B, int C, int h, int w, int M, int N, int H, \
                   int W, int steps, long long sb, long long sc, long long sy, long long sx
#define UP_DISPATCH(T) \
    if (steps == 1) up_impl<T, 2>(s, (float*)out, B, C, h, w, M, N, H, W, steps, sb, sc, sy, sx); \
    else up_impl<T, 3>(s, (float*)out, B, C, h, w, M, N, H, W, steps, sb, sc, sy, sx);
// the shared arrays live in the kernels (one per kernel, not one per template instance)
#define UP_TILED(T) \
    __shared__ float hs[CH * RS]; \
    if (steps == 1) up_argmax_tiled<T, 2>(s, (long long*)out, B, C, h, w, M, N, H, W, steps, sb, sc, sy, sx, hs); \
    else up_argmax_tiled<T, 3>(s, (long long*)out, B, C, h, w, M, N, H, W, steps, sb, sc, sy, sx, hs);

extern "C" __global__ void __launch_bounds__(256) up_argmax_h(UP_ARGS(__half)) { UP_TILED(__half) }
extern "C" __global__ void __launch_bounds__(256) up_argmax_f(UP_ARGS(float)) { UP_TILED(float) }
#define UP_ROWS(T) \
    __shared__ float2 hs[RM * CP * BW]; \
    if (steps == 1) up_argmax_rows<T, 2>(s, (long long*)out, B, C, h, w, M, N, H, W, steps, sb, sc, sy, sx, hs); \
    else up_argmax_rows<T, 3>(s, (long long*)out, B, C, h, w, M, N, H, W, steps, sb, sc, sy, sx, hs);
extern "C" __global__ void __launch_bounds__(128) up_argmax_rh(UP_ARGS(__half)) { UP_ROWS(__half) }
extern "C" __global__ void __launch_bounds__(128) up_argmax_rf(UP_ARGS(float)) { UP_ROWS(float) }
#define UP_PIX(T) \
    if (steps == 1) up_argmax_pix<T, 2>(s, (long long*)out, B, C, h, w, M, N, H, W, steps, sb, sy, sx); \
    else up_argmax_pix<T, 3>(s, (long long*)out, B, C, h, w, M, N, H, W, steps, sb, sy, sx);
extern "C" __global__ void up_argmax_ph(UP_ARGS(__half)) { UP_PIX(__half) }
extern "C" __global__ void up_argmax_pf(UP_ARGS(float)) { UP_PIX(float) }
extern "C" __global__ void up_h(UP_ARGS(__half)) { UP_DISPATCH(__half) }
extern "C" __global__ void up_f(UP_ARGS(float)) { UP_DISPATCH(float) }

// HRNet / SegFormer fuse term: out = relu?(base + half(resize(z))), all fp16 dense NHWC [B, H, W, C] (z: [B, h, w, C]),
// z resized through one or two steps as above (h x w -> M x N -> H x W). The resized value is rounded to fp16 before
// the add, as PyTorch's fp16 resize + add. One thread per V channels (`up_add_h`: V = 2, C even; `up_add8_h`: V = 8,
// C % 8 == 0 and 16-byte aligned tensors); one step reads 2 x 2 taps, two steps 3 x 3. out may alias base.
template <int V>
struct Vec {
    __half2 h[V / 2];
};

template <int NT, int V>
__device__ void up_add_impl(const __half* __restrict__ z, const __half* base, __half* out, int B, int C, int h, int w,
                            int M, int N, int H, int W, int steps, int relu) {
    using VT = Vec<V>;
    int cvn = C / V;
    long long n = (long long)B * H * W * cvn;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int cv = (int)(i % cvn);
        long long p = i / cvn;
        int x = (int)(p % W);
        long long t = p / W;
        int y = (int)(t % H);
        int b = (int)(t / H);
        Taps ty = taps(y, h, M, H, steps), tx = taps(x, w, N, W, steps);
        const VT* zb = (const VT*)z + (long long)b * h * w * cvn + cv;
        float r[NT][V];
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            const VT* q = zb + (long long)ty.i[k] * w * cvn;
            VT a[NT];
#pragma unroll
            for (int l = 0; l < NT; ++l) a[l] = q[(long long)tx.i[l] * cvn];
#pragma unroll
            for (int e = 0; e < V / 2; ++e) {
                float2 f0 = __half22float2(a[0].h[e]), f1 = __half22float2(a[1].h[e]);
                float vx = fmaf(tx.w[1], f1.x, __fmul_rn(tx.w[0], f0.x));
                float vy = fmaf(tx.w[1], f1.y, __fmul_rn(tx.w[0], f0.y));
                if (NT == 3) {
                    float2 f2 = __half22float2(a[NT - 1].h[e]);
                    vx = fmaf(tx.w[2], f2.x, vx);
                    vy = fmaf(tx.w[2], f2.y, vy);
                }
                r[k][2 * e] = vx;
                r[k][2 * e + 1] = vy;
            }
        }
        VT sb = ((const VT*)base)[i], o;
#pragma unroll
        for (int e = 0; e < V / 2; ++e) {
            float u[2];
#pragma unroll
            for (int d = 0; d < 2; ++d) {
                float v = fmaf(ty.w[1], r[1][2 * e + d], __fmul_rn(ty.w[0], r[0][2 * e + d]));
                u[d] = NT == 3 ? fmaf(ty.w[2], r[NT - 1][2 * e + d], v) : v;
            }
            float2 uh = __half22float2(__floats2half2_rn(u[0], u[1]));
            float2 s = __half22float2(sb.h[e]);
            float ox = __half2float(__float2half_rn(s.x + uh.x)), oy = __half2float(__float2half_rn(s.y + uh.y));
            if (relu) {
                ox = fmaxf(ox, 0.f);
                oy = fmaxf(oy, 0.f);
            }
            o.h[e] = __floats2half2_rn(ox, oy);
        }
        ((VT*)out)[i] = o;
    }
}

#define ADD_ARGS const __half *__restrict__ z, const __half *base, __half *out, int B, int C, int h, int w, int M, \
                 int N, int H, int W, int steps, int relu
#define ADD_DISPATCH(V) \
    if (steps == 1) up_add_impl<2, V>(z, base, out, B, C, h, w, M, N, H, W, steps, relu); \
    else up_add_impl<3, V>(z, base, out, B, C, h, w, M, N, H, W, steps, relu);

extern "C" __global__ void up_add_h(ADD_ARGS) { ADD_DISPATCH(2) }
extern "C" __global__ void up_add8_h(ADD_ARGS) { ADD_DISPATCH(8) }
