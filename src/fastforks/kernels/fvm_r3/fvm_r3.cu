// Round-3 classification kernels (fast-vision-models `r3.cu`, exp 008): grouped / small-C dense tensor-core conv
// (gconv2), SE scale, depthwise conv with half2 accumulation (dwconv2), fused stem conv + max pool (stem_pool) and the
// cuBLASLt 1x1 conv (pw). Kernel bodies unchanged; the torch host functions became `extern "C"` launchers and the
// Python wrapper (`__init__.py`) does the checks, allocations and the fallback decisions the host code did.
// cuBLASLt is not linked: the library binds to the libcublasLt the process already loaded (PyTorch's) at run time.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <library_types.h>
#include <mutex>
#include <unordered_map>

namespace {

__device__ __forceinline__ float actf(float v, int act) {
    switch (act) {
        case 1: return fmaxf(v, 0.f);
        case 2: return fminf(fmaxf(v, 0.f), 6.f);
        case 3: return __fdividef(v, 1.f + __expf(-v));
        case 4: return v * fminf(fmaxf(v + 3.f, 0.f), 6.f) * (1.f / 6.f);
    }
    return v;
}


__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm2(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(unsigned* d, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ __half2 u2h(unsigned u) { return *reinterpret_cast<__half2*>(&u); }

// ---------------------------------------------------------------------------------------------------------------------
// gconv2: grouped conv (group width GW = C/G = K/G, GW % 16 == 0), NHWC fp16, w [K, R, S, GW], fp16 accumulation, + bias
// + act (+ optional fp32 per-(image, channel) output sums for a following global average pool, e.g. SE).
// Round-3 rewrite of clfmb.gconv: CTA = 128 output pixels x BN = 16 NT channels of one group, 8 warps (4 M x 2 N),
// KB k16-blocks per stage (one tap, 16 channels each), ST-stage cp.async pipeline, no divisions in the main loop
// (per-thread tap / channel-block counters), C tile staged in smem -> 16-B coalesced stores.
template <int NT, int KB, int ST>
__global__ void __launch_bounds__(256) gconv2_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                                                     const __half* __restrict__ bias, __half* __restrict__ y,
                                                     float* __restrict__ pool, int N, int H, int W, int C, int K, int GW, int KG, int ldy,
                                                     int R, int S, int sh, int sw, int ph, int pw, int P, int Q, int act) {
    constexpr int BM = 128, BN = 16 * NT, SK = 16 * KB, CH = 2 * KB;   // smem row: SK halfs = CH 16-B chunks
    constexpr int APT = BM * CH / 256, ROWS_STEP = 256 / CH;            // A chunks per thread, row step
    constexpr int CS = BN + 8;                                          // C staging row stride (halfs)
    extern __shared__ __align__(16) unsigned char smem[];
    __half* As = reinterpret_cast<__half*>(smem);          // [ST][BM][SK]
    __half* Bs = As + ST * BM * SK;                          // [ST][BN][SK]
    auto swz = [](int row, int ch) { return KB == 4 ? (ch ^ (row & 7)) : KB == 2 ? (ch ^ ((row >> 1) & 3)) : (ch ^ ((row >> 2) & 1)); };
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long M = (long long)N * P * Q;
    const long long m0 = (long long)blockIdx.x * BM;
    const int k0 = blockIdx.y * BN;
    const int cbase = (k0 / KG) * GW;     // grouped: KG = GW; dense: KG = K -> 0
    const int kpb = GW / 16, nkb = R * S * kpb, nst = (nkb + KB - 1) / KB;
    const int part = tid % CH, kbo = part / 2, hf = part & 1;
    // A rows of this thread: tid / CH + i * ROWS_STEP
    const __half* rowp[APT];
    int ih0[APT], iw0[APT];
#pragma unroll
    for (int i = 0; i < APT; ++i) {
        long long m = m0 + tid / CH + (long long)ROWS_STEP * i;
        bool v = m < M;
        long long mm = v ? m : 0;
        int q = mm % Q; long long t = mm / Q; int p = t % P; int n = t / P;
        ih0[i] = v ? p * sh - ph : -100000;
        iw0[i] = q * sw - pw;
        rowp[i] = x + (long long)n * H * W * C + cbase + hf * 8;
    }
    // per-thread k-block state for A (all rows of the thread share kb = st * KB + kbo)
    int a_cb = kbo % kpb, a_tap = kbo / kpb;
    int a_r = a_tap / S, a_s = a_tap % S;
    const int RS = R * S;
    auto load = [&](int st, int buf) {
        __half* a_ = As + buf * BM * SK;
        __half* b_ = Bs + buf * BN * SK;
        const int kb = st * KB + kbo;
        const bool kok = kb < nkb;
#pragma unroll
        for (int i = 0; i < APT; ++i) {
            int row = tid / CH + ROWS_STEP * i;
            int ih = ih0[i] + a_r, iw = iw0[i] + a_s;
            bool ok = kok && ih >= 0 && ih < H && iw >= 0 && iw < W;
            const __half* src = ok ? rowp[i] + ((long long)ih * W + iw) * C + a_cb * 16 : x;
            cp16(a_ + row * SK + swz(row, part) * 8, src, ok);
        }
        for (int i = tid; i < BN * CH; i += 256) {
            int row = i / CH, pt = i % CH;
            int kb2 = st * KB + pt / 2;
            bool ok = kb2 < nkb;
            const __half* src = ok ? w + (long long)(k0 + row) * RS * GW + kb2 * 16 + (pt & 1) * 8 : w;
            cp16(b_ + row * SK + swz(row, pt) * 8, src, ok);
        }
        // advance the A state by KB k-blocks
        a_cb += KB;
        while (a_cb >= kpb) { a_cb -= kpb; if (++a_s == S) { a_s = 0; ++a_r; } }
    };
    const int wm = warp & 3, wn = warp >> 2;
    unsigned acc[2][NT][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = 0u;
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) { if (st < nst) load(st, st); cp_commit(); }
    for (int st = 0; st < nst; ++st) {
        cp_wait<ST - 2>();
        __syncthreads();
        if (st + ST - 1 < nst) load(st + ST - 1, (st + ST - 1) % ST);
        cp_commit();
        const __half* a_ = As + (st % ST) * BM * SK;
        const __half* b_ = Bs + (st % ST) * BN * SK;
#pragma unroll
        for (int kk = 0; kk < KB; ++kk) {
            unsigned af[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int row = wm * 32 + i * 16 + (lane & 15), ch = kk * 2 + (lane >> 4);
                ldsm4(af[i][0], af[i][1], af[i][2], af[i][3], a_ + row * SK + swz(row, ch) * 8);
            }
#pragma unroll
            for (int j = 0; j < NT; j += 2) {
                if (j + 1 < NT) {
                    int brow = wn * NT * 8 + j * 8 + (lane >> 4) * 8 + (lane & 7), bch = kk * 2 + ((lane >> 3) & 1);
                    unsigned b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, b_ + brow * SK + swz(brow, bch) * 8);
#pragma unroll
                    for (int i = 0; i < 2; ++i) { mma16816(acc[i][j], af[i], b0, b1); mma16816(acc[i][j + 1], af[i], b2, b3); }
                } else {
                    int r2 = wn * NT * 8 + j * 8 + (lane & 7), bch = kk * 2 + ((lane >> 3) & 1);
                    unsigned b0, b1;
                    ldsm2(b0, b1, b_ + r2 * SK + swz(r2, bch) * 8);
#pragma unroll
                    for (int i = 0; i < 2; ++i) mma16816(acc[i][j], af[i], b0, b1);
                }
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
    // epilogue: bias + act -> smem C tile [BM][CS] -> coalesced 16-B stores (+ pool sums)
    __half* Cs = reinterpret_cast<__half*>(smem);
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        int col = wn * NT * 8 + j * 8 + t4 * 2;
        __half2 hb = bias ? *reinterpret_cast<const __half2*>(bias + k0 + col) : __float2half2_rn(0.f);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int row = wm * 32 + i * 16 + g + h * 8;
                __half2 v = __hadd2(u2h(acc[i][j][h]), hb);
                if (act == 1) v = __hmax2(v, __float2half2_rn(0.f));
                else if (act) { float2 f = __half22float2(v); v = __floats2half2_rn(actf(f.x, act), actf(f.y, act)); }
                *reinterpret_cast<__half2*>(Cs + row * CS + col) = v;
            }
    }
    __syncthreads();
    for (int i = tid; i < BM * (BN / 8); i += 256) {
        int row = i / (BN / 8), c8 = i % (BN / 8);
        long long m = m0 + row;
        if (m < M) *reinterpret_cast<uint4*>(y + m * ldy + k0 + c8 * 8) = *reinterpret_cast<const uint4*>(Cs + row * CS + c8 * 8);
    }
    if (pool) {   // per-(image, channel) sums over the tile's rows: thread = column pair x row group
        constexpr int NCP = BN / 2, NRG = 256 / NCP;
        const int PQ = P * Q;
        int cp = tid % NCP, rg = tid / NCP;
        if (rg < NRG) {
            constexpr int RPG = (BM + NRG - 1) / NRG;
            int r0 = rg * RPG, r1 = min(r0 + RPG, BM);
            float sx = 0.f, sy = 0.f;
            int cur = -1;
            for (int r = r0; r < r1; ++r) {
                long long m = m0 + r;
                if (m >= M) break;
                int n = (int)(m / PQ);
                if (n != cur) {
                    if (cur >= 0) { atomicAdd(pool + (long long)cur * K + k0 + 2 * cp, sx); atomicAdd(pool + (long long)cur * K + k0 + 2 * cp + 1, sy); }
                    cur = n; sx = sy = 0.f;
                }
                float2 f = __half22float2(*reinterpret_cast<const __half2*>(Cs + r * CS + 2 * cp));
                sx += f.x; sy += f.y;
            }
            if (cur >= 0) { atomicAdd(pool + (long long)cur * K + k0 + 2 * cp, sx); atomicAdd(pool + (long long)cur * K + k0 + 2 * cp + 1, sy); }
        }
    }
}

template <int NT, int KB, int ST>
static void gconv2_launch(dim3 grid, cudaStream_t st, const __half* x, const __half* w, const __half* b, __half* y, float* pool,
                          int N, int H, int W, int C, int K, int GW, int KG, int ldy, int R, int S, int sh, int sw, int ph, int pw,
                          int P, int Q, int act) {
    int smem = std::max(ST * (128 + 16 * NT) * 16 * KB * 2, 128 * (16 * NT + 8) * 2);
    static int attr[64] = {};
    int d = 0;
    cudaGetDevice(&d);
    if (d < 64 && smem > attr[d]) {
        cudaFuncSetAttribute(gconv2_kernel<NT, KB, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr[d] = smem;
    }
    gconv2_kernel<NT, KB, ST><<<grid, 256, smem, st>>>(x, w, b, y, pool, N, H, W, C, K, GW, KG, ldy, R, S, sh, sw, ph, pw, P, Q, act);
}

template <int NT>
static void gconv2_cfg(dim3 grid, cudaStream_t st, const __half* x, const __half* w, const __half* b, __half* y, float* pool,
                       int N, int H, int W, int C, int K, int GW, int KG, int ldy, int R, int S, int sh, int sw, int ph, int pw,
                       int P, int Q, int act, int cfg) {
    switch (cfg) {
        case 1: gconv2_launch<NT, 4, 2>(grid, st, x, w, b, y, pool, N, H, W, C, K, GW, KG, ldy, R, S, sh, sw, ph, pw, P, Q, act); break;
        case 2: gconv2_launch<NT, 2, 4>(grid, st, x, w, b, y, pool, N, H, W, C, K, GW, KG, ldy, R, S, sh, sw, ph, pw, P, Q, act); break;
        case 3: gconv2_launch<NT, 4, 3>(grid, st, x, w, b, y, pool, N, H, W, C, K, GW, KG, ldy, R, S, sh, sw, ph, pw, P, Q, act); break;
        default: gconv2_launch<NT, 2, 3>(grid, st, x, w, b, y, pool, N, H, W, C, K, GW, KG, ldy, R, S, sh, sw, ph, pw, P, Q, act);
    }
}

// SE scale: y[n, p, c] = x[n, p, c] * sigmoid(s[n, c]); x channels-last [N, C, H, W], s [N, C, 1, 1]; 8 channels / thread
__global__ void se_scale_k(const __half* __restrict__ x, const __half* __restrict__ s, __half* __restrict__ y, long long HW, int C8,
                           long long total) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = i % C8; long long n = i / C8 / HW;
    uint4 v = __ldg(reinterpret_cast<const uint4*>(x) + i);
    uint4 g = __ldg(reinterpret_cast<const uint4*>(s) + n * C8 + c);
    __half2* hv = reinterpret_cast<__half2*>(&v);
    const __half2* hg = reinterpret_cast<const __half2*>(&g);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float2 f = __half22float2(hg[k]);
        hv[k] = __hmul2(hv[k], __floats2half2_rn(1.f / (1.f + __expf(-f.x)), 1.f / (1.f + __expf(-f.y))));
    }
    reinterpret_cast<uint4*>(y)[i] = v;
}

// ---------------------------------------------------------------------------------------------------------------------
// dwconv2: depthwise conv (groups == C == K), NHWC fp16, weight [R, S, C], + bias + act, HALF2 accumulation (as the
// tensor-core convs; round-2 lesson: fp32 conversions per element made the dw kernels issue-bound). One thread = 8
// channels x TW consecutive output columns; compile-time taps for 3x3 / 5x5 (RT = 0: runtime).
template <int TW, int RT, int MODE>   // MODE 0: half2 accumulation, 1: fp32, 2: half2 per kernel row + fp32 across rows
__global__ void __launch_bounds__(128) dwconv2_k(const __half* __restrict__ x, const __half* __restrict__ w,
                                                 const __half* __restrict__ b, __half* __restrict__ y, int N, int H, int W,
                                                 int C8, int P, int Q, int R_, int S_, int sh, int sw, int ph, int pw, int act) {
    const int R = RT ? RT : R_, S = RT ? RT : S_;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int QT = (Q + TW - 1) / TW;
    long long total = (long long)N * P * QT * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int qt = t % QT; t /= QT; int p = t % P; int n = t / P;
    int q0 = qt * TW;
    __half2 bb[4];
    if (b) { uint4 bv = __ldg(reinterpret_cast<const uint4*>(b) + c); const __half2* hb = reinterpret_cast<const __half2*>(&bv);
#pragma unroll
             for (int k = 0; k < 4; ++k) bb[k] = hb[k]; }
    else {
#pragma unroll
        for (int k = 0; k < 4; ++k) bb[k] = __float2half2_rn(0.f);
    }
    __half2 acc[TW][4];
    constexpr bool F32 = MODE != 0;
    float2 af[F32 ? TW : 1][4];
#pragma unroll
    for (int j = 0; j < TW; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) { acc[j][k] = bb[k]; if (F32) af[F32 ? j : 0][k] = __half22float2(bb[k]); }
    const int L = (TW - 1) * sw + S;
    const int iw0 = q0 * sw - pw;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        const uint4* xrow = reinterpret_cast<const uint4*>(x + ((long long)n * H + h) * W * C8 * 8) + c;
        uint4 wr[RT ? RT : 7];
#pragma unroll
        for (int s2 = 0; s2 < S; ++s2) wr[s2] = __ldg(reinterpret_cast<const uint4*>(w + ((long long)r * S + s2) * C8 * 8) + c);
        if (MODE == 2) {
#pragma unroll
            for (int j = 0; j < TW; ++j)
#pragma unroll
                for (int k = 0; k < 4; ++k) acc[j][k] = __float2half2_rn(0.f);
        }
#pragma unroll
        for (int l = 0; l < L; ++l) {
            int ww = iw0 + l; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(xrow + (long long)ww * C8);
            const __half2* hv = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int j = 0; j < TW; ++j) {
                int s2 = l - j * sw;
                if (s2 < 0 || s2 >= S) continue;
                const __half2* hg = reinterpret_cast<const __half2*>(&wr[s2]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (MODE == 1) {
                        float2 a = __half22float2(hv[k]), b2 = __half22float2(hg[k]);
                        af[F32 ? j : 0][k].x += a.x * b2.x; af[F32 ? j : 0][k].y += a.y * b2.y;
                    } else {
                        acc[j][k] = __hfma2(hv[k], hg[k], acc[j][k]);
                    }
                }
            }
        }
        if (MODE == 2) {
#pragma unroll
            for (int j = 0; j < TW; ++j)
#pragma unroll
                for (int k = 0; k < 4; ++k) { float2 f = __half22float2(acc[j][k]); af[F32 ? j : 0][k].x += f.x; af[F32 ? j : 0][k].y += f.y; }
        }
    }
#pragma unroll
    for (int j = 0; j < TW; ++j) {
        int q = q0 + j; if (q >= Q) break;
        __align__(16) __half2 o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            __half2 v = F32 ? __floats2half2_rn(af[F32 ? j : 0][k].x, af[F32 ? j : 0][k].y) : acc[j][k];
            if (act == 1) v = __hmax2(v, __float2half2_rn(0.f));
            else if (act == 2) v = __hmin2(__hmax2(v, __float2half2_rn(0.f)), __float2half2_rn(6.f));
            else if (act) { float2 f = __half22float2(v); v = __floats2half2_rn(actf(f.x, act), actf(f.y, act)); }
            o[k] = v;
        }
        reinterpret_cast<uint4*>(y + (((long long)n * P + p) * Q + q) * C8 * 8)[c] = *reinterpret_cast<uint4*>(o);
    }
}

// One CTA = TP x TQ pooled output pixels of one image.
//  phase 0: input patch (C x IH x IW, zero padded) -> smem as fp16
//  phase 1: stem tile SH x SW (= the pool windows of the CTA) x K channels -> smem (row stride KS halfs, conflict-free);
//           items = (8-channel group, stem pixel), group-major so a warp shares its weights (smem broadcast);
//           positions outside the stem output get -inf (max-pool padding semantics)
//  phase 2: max over the pool window, 8 channels per thread, 16-B coalesced stores (NHWC)
constexpr int TP = 8, TQ = 8, NTH = 256;

template <int K, int KG, int RT, int CT, int SST, int PKT, class T>   // KG channels per item; RT/CT/SST/PKT: compile-time
__global__ void __launch_bounds__(NTH) stem_pool_kernel(const T* __restrict__ x, const __half* __restrict__ w,  // taps / C /
                                                        const __half* __restrict__ b, __half* __restrict__ y, int N, int C_,  // stride /
                                                        int H, int W, int P, int Q, int R_, int S_, int sh_, int sw_, int ph,  // pool k
                                                        int pw, int act, int PK_, int PS, int PP, int P2, int Q2, int SH,      // (0: runtime)
                                                        int SW, int IH, int IW, int tq_n) {
    const int R = RT ? RT : R_, S = RT ? RT : S_, C = CT ? CT : C_, sh = SST ? SST : sh_, sw = SST ? SST : sw_;
    const int PK = PKT ? PKT : PK_;
    // K == 64: 8 chunks XOR-swizzled by pixel; else row padding with (KS / 8) odd. Both: 16-B accesses conflict-free
    constexpr int KS = K == 64 ? 64 : (K / 8) % 2 ? K + 16 : K + 8;
    auto sidx = [](int px, int c8) { return px * KS + (K == 64 ? (c8 ^ (px & 7)) : c8) * 8; };
    extern __shared__ __align__(16) unsigned char smem[];
    __half* st = reinterpret_cast<__half*>(smem);                    // [SH * SW][KS]
    __half* ws = st + SH * SW * KS;                                  // [R * S * C][K]
    __half* xs = ws + ((R * S * C * K + 7) / 8) * 8;                 // [C][IH][IW]
    const int tid = threadIdx.x;
    const int n = blockIdx.y;
    const int tp0 = (blockIdx.x / tq_n) * TP, tq0 = (blockIdx.x % tq_n) * TQ;
    const int sp0 = tp0 * PS - PP, sq0 = tq0 * PS - PP;              // stem tile origin (stem output coords)
    const int ih0 = sp0 * sh - ph, iw0 = sq0 * sw - pw;              // input patch origin
    const int nw = R * S * C * K;
    for (int i = tid * 8; i < nw; i += NTH * 8) *reinterpret_cast<uint4*>(ws + i) = __ldg(reinterpret_cast<const uint4*>(w + i));
    const long long HW = (long long)H * W;
    const T* xn = x + (long long)n * C * HW;
    {   // all loads first (independent, in flight together), then convert + store: no serialized load latency
        constexpr int MAXL = 32;
        float v[MAXL];
        const int tot = C * IH * IW;
#pragma unroll
        for (int j = 0; j < MAXL; ++j) {
            int i = tid + j * NTH;
            v[j] = 0.f;
            if (i < tot) {
                int c = i / (IH * IW), r = i - c * (IH * IW), yy = r / IW, xx = r - yy * IW;
                int ih = ih0 + yy, iw = iw0 + xx;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) v[j] = (float)__ldg(xn + c * HW + (long long)ih * W + iw);
            }
        }
#pragma unroll
        for (int j = 0; j < MAXL; ++j) {
            int i = tid + j * NTH;
            if (i < tot) xs[i] = __float2half_rn(v[j]);
        }
        for (int i = tid + MAXL * NTH; i < tot; i += NTH) {      // large patches (big stem strides): plain loop
            int c = i / (IH * IW), r = i - c * (IH * IW), yy = r / IW, xx = r - yy * IW;
            int ih = ih0 + yy, iw = iw0 + xx;
            xs[i] = __float2half_rn((ih >= 0 && ih < H && iw >= 0 && iw < W) ? (float)xn[c * HW + (long long)ih * W + iw] : 0.f);
        }
    }
    __syncthreads();
    const int npx = SH * SW;
    const __half2 ninf = __half2(__ushort_as_half(0xfc00), __ushort_as_half(0xfc00));
    constexpr int NG = K / KG;
    for (int it = tid; it < npx * NG; it += NTH) {
        int kg = it / npx, px = it % npx;
        int ty = px / SW, tx = px % SW;
        int sp = sp0 + ty, sq = sq0 + tx;
        if (sp < 0 || sp >= P || sq < 0 || sq >= Q) {
            __half2 o[4] = {ninf, ninf, ninf, ninf};
#pragma unroll
            for (int u = 0; u < KG / 8; ++u) *reinterpret_cast<uint4*>(st + sidx(px, kg * (KG / 8) + u)) = *reinterpret_cast<uint4*>(o);
            continue;
        }
        __half2 acc[KG / 2];
#pragma unroll
        for (int j = 0; j < KG / 2; ++j) acc[j] = __float2half2_rn(0.f);
        const __half* xp = xs + (ty * sh) * IW + tx * sw;
        const __half* wp = ws + kg * KG;
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int s2 = 0; s2 < S; ++s2)
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    __half2 xv = __half2half2(xp[(c * IH + r) * IW + s2]);
                    const __half* wt = wp + ((r * S + s2) * C + c) * K;
#pragma unroll
                    for (int u = 0; u < KG / 8; ++u) {
                        uint4 wv = *reinterpret_cast<const uint4*>(wt + u * 8);
                        const __half2* hw = reinterpret_cast<const __half2*>(&wv);
#pragma unroll
                        for (int j = 0; j < 4; ++j) acc[u * 4 + j] = __hfma2(xv, hw[j], acc[u * 4 + j]);
                    }
                }
#pragma unroll
        for (int u = 0; u < KG / 8; ++u) {
            __align__(16) __half o[8];
            int k0 = kg * KG + u * 8;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 v = __half22float2(acc[u * 4 + j]);
                float b0 = b ? __half2float(b[k0 + 2 * j]) : 0.f, b1 = b ? __half2float(b[k0 + 2 * j + 1]) : 0.f;
                o[2 * j] = __float2half(actf(v.x + b0, act));
                o[2 * j + 1] = __float2half(actf(v.y + b1, act));
            }
            *reinterpret_cast<uint4*>(st + sidx(px, kg * (KG / 8) + u)) = *reinterpret_cast<const uint4*>(o);
        }
    }
    __syncthreads();
    for (int it = tid; it < TP * TQ * (K / 8); it += NTH) {
        int c8 = it % (K / 8), op = it / (K / 8);
        int pp = tp0 + op / TQ, pq = tq0 + op % TQ;
        if (pp >= P2 || pq >= Q2) continue;
        int ly = (op / TQ) * PS, lx = (op % TQ) * PS;
        __half2 m[4] = {ninf, ninf, ninf, ninf};
#pragma unroll
        for (int r = 0; r < PK; ++r)
#pragma unroll
            for (int s2 = 0; s2 < PK; ++s2) {
                uint4 v = *reinterpret_cast<const uint4*>(st + sidx((ly + r) * SW + lx + s2, c8));
                const __half2* hv = reinterpret_cast<const __half2*>(&v);
#pragma unroll
                for (int j = 0; j < 4; ++j) m[j] = __hmax2(m[j], hv[j]);
            }
        reinterpret_cast<uint4*>(y + (((long long)n * P2 + pp) * Q2 + pq) * K)[c8] = *reinterpret_cast<uint4*>(m);
    }
}
// ---------------------------------------------------------------------------------------------------------------------
// pw: 1x1 / stride-1 conv as a cuBLASLt GEMM with fused epilogue  y = act(x . w^T + bias (+ res)),  act: 0 none, 1 relu.
// fp16 in / out / accumulate (CUBLAS_COMPUTE_16F). x [M, ldx] (C channels used), w [K, C], res [M, ldr], y [M, ldy].
// Column-major view: D[K, M] = op(W)[K, C] . X[C, M] (+ C[K, M]) with M = N * H * W.
// Plans (descriptors + heuristic algo, no timing) are cached per (device, M, K, C, ldx, ldy, ldr, epilogue).

// The few cuBLASLt declarations used (values from cublasLt.h / cublas_api.h; the C ABI is stable across CUDA 12 / 13).
typedef int lt_status;
typedef struct LtCtx* lt_handle;
typedef struct LtOpaque* lt_desc;
typedef struct LtOpaque* lt_layout;
typedef struct LtOpaque* lt_pref;
struct lt_algo { uint64_t data[8]; };
struct lt_heur { lt_algo algo; size_t workspaceSize; lt_status state; float wavesCount; int reserved[4]; };
constexpr int LT_COMPUTE_16F = 64, LT_OP_N = 0, LT_OP_T = 1;
constexpr int LT_DESC_TRANSA = 3, LT_DESC_TRANSB = 4, LT_DESC_EPILOGUE = 7, LT_DESC_BIAS_POINTER = 8, LT_DESC_BIAS_TYPE = 26;
constexpr int LT_EPI_DEFAULT = 1, LT_EPI_RELU = 2, LT_EPI_BIAS = 4, LT_EPI_RELU_BIAS = 6, LT_PREF_MAX_WORKSPACE = 1;

struct Lt {
    lt_status (*create)(lt_handle*);
    lt_status (*desc_create)(lt_desc*, int, cudaDataType_t);
    lt_status (*desc_set)(lt_desc, int, const void*, size_t);
    lt_status (*layout_create)(lt_layout*, cudaDataType_t, uint64_t, uint64_t, int64_t);
    lt_status (*pref_create)(lt_pref*);
    lt_status (*pref_set)(lt_pref, int, const void*, size_t);
    lt_status (*pref_destroy)(lt_pref);
    lt_status (*heuristic)(lt_handle, lt_desc, lt_layout, lt_layout, lt_layout, lt_layout, lt_pref, int, lt_heur*, int*);
    lt_status (*matmul)(lt_handle, lt_desc, const void*, const void*, lt_layout, const void*, lt_layout, const void*,
                        const void*, lt_layout, void*, lt_layout, const lt_algo*, void*, size_t, cudaStream_t);
    bool ok = false;
};

// bind to the libcublasLt already in the process (PyTorch's), else load one from the library search path
static const Lt& lt() {
    static Lt l = [] {
        Lt r{};
        void* h = nullptr;
        for (const char* n : {"libcublasLt.so.13", "libcublasLt.so.12"}) {
            h = dlopen(n, RTLD_NOW | RTLD_NOLOAD);
            if (h) break;
        }
        for (const char* n : {"libcublasLt.so.13", "libcublasLt.so.12"}) {
            if (h) break;
            h = dlopen(n, RTLD_NOW | RTLD_LOCAL);
        }
        if (!h) return r;
#define LT_SYM(f, s) r.f = reinterpret_cast<decltype(r.f)>(dlsym(h, s)); if (!r.f) return r;
        LT_SYM(create, "cublasLtCreate") LT_SYM(desc_create, "cublasLtMatmulDescCreate")
        LT_SYM(desc_set, "cublasLtMatmulDescSetAttribute") LT_SYM(layout_create, "cublasLtMatrixLayoutCreate")
        LT_SYM(pref_create, "cublasLtMatmulPreferenceCreate") LT_SYM(pref_set, "cublasLtMatmulPreferenceSetAttribute")
        LT_SYM(pref_destroy, "cublasLtMatmulPreferenceDestroy") LT_SYM(heuristic, "cublasLtMatmulAlgoGetHeuristic")
        LT_SYM(matmul, "cublasLtMatmul")
#undef LT_SYM
        r.ok = true;
        return r;
    }();
    return l;
}

struct LtPlan {
    lt_desc op = nullptr;
    lt_layout a = nullptr, b = nullptr, c = nullptr, d = nullptr;
    lt_algo algo;
    bool ok = false;
};
struct LtKey {
    long long M; int K, C, ldx, ldy, ldr, epi, dev; size_t ws;
    bool operator==(const LtKey& o) const {
        return M == o.M && K == o.K && C == o.C && ldx == o.ldx && ldy == o.ldy && ldr == o.ldr && epi == o.epi &&
               dev == o.dev && ws == o.ws;
    }
};
struct LtKeyHash {
    size_t operator()(const LtKey& k) const {
        size_t h = std::hash<long long>()(k.M) ^ std::hash<size_t>()(k.ws);
        for (int v : {k.K, k.C, k.ldx, k.ldy, k.ldr, k.epi, k.dev}) h = h * 1000003u ^ std::hash<int>()(v);
        return h;
    }
};
static std::unordered_map<LtKey, LtPlan, LtKeyHash> g_plans;
static std::mutex g_plans_mu;

static LtPlan& lt_plan(const LtKey& k, lt_handle h) {
    const Lt& L = lt();
    std::lock_guard<std::mutex> g(g_plans_mu);
    auto it = g_plans.find(k);
    if (it != g_plans.end()) return it->second;
    LtPlan& p = g_plans[k];
    bool ok = L.desc_create(&p.op, LT_COMPUTE_16F, CUDA_R_16F) == 0;
    int ta = LT_OP_T, tb = LT_OP_N;
    ok = ok && L.desc_set(p.op, LT_DESC_TRANSA, &ta, sizeof(ta)) == 0;
    ok = ok && L.desc_set(p.op, LT_DESC_TRANSB, &tb, sizeof(tb)) == 0;
    int ep = k.epi == 0 ? LT_EPI_DEFAULT : k.epi == 1 ? LT_EPI_RELU : k.epi == 2 ? LT_EPI_BIAS : LT_EPI_RELU_BIAS;
    ok = ok && L.desc_set(p.op, LT_DESC_EPILOGUE, &ep, sizeof(ep)) == 0;
    if (k.epi >= 2) {
        cudaDataType_t bt = CUDA_R_16F;
        ok = ok && L.desc_set(p.op, LT_DESC_BIAS_TYPE, &bt, sizeof(bt)) == 0;
    }
    ok = ok && L.layout_create(&p.a, CUDA_R_16F, k.C, k.K, k.C) == 0;     // W^T as [C, K]
    ok = ok && L.layout_create(&p.b, CUDA_R_16F, k.C, k.M, k.ldx) == 0;   // X^T [C, M]
    ok = ok && L.layout_create(&p.c, CUDA_R_16F, k.K, k.M, k.ldr > 0 ? k.ldr : k.ldy) == 0;
    ok = ok && L.layout_create(&p.d, CUDA_R_16F, k.K, k.M, k.ldy) == 0;
    lt_pref pref = nullptr;
    size_t ws = k.ws;
    ok = ok && L.pref_create(&pref) == 0;
    ok = ok && L.pref_set(pref, LT_PREF_MAX_WORKSPACE, &ws, sizeof(ws)) == 0;
    lt_heur res[1];
    int nres = 0;
    ok = ok && L.heuristic(h, p.op, p.a, p.b, p.c, p.d, pref, 1, res, &nres) == 0 && nres > 0;
    if (pref) L.pref_destroy(pref);
    if (ok) p.algo = res[0].algo;
    p.ok = ok;
    return p;
}

static lt_handle lt_handle_for(int dev) {
    static lt_handle hs[64] = {};
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    if (!hs[dev] && lt().create(&hs[dev]) != 0) hs[dev] = nullptr;
    return hs[dev];
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// C API. Pointers are device pointers, `st` a cudaStream_t; launchers return 0 on success, 1 when they do not support
// the case (the caller then runs its documented fallback), else a cudaError_t (or 1000 + cublasStatus_t for pw).

// grouped / dense conv: x [N, H, W, C] fp16, w [K, R, S, GW] (groups = C / GW), bias [K] or null, y [N, P, Q, ldy];
// needs GW % 16 == 0 and (K / groups) % 16 == 0; cfg < 0 picks the pipeline config
extern "C" int fvm_gconv2(const void* x, const void* w, const void* bias, void* y, int N, int H, int W, int C, int K, int GW,
                          int R, int S, int sh, int sw, int ph, int pw, int act, int ldy, int cfg, cudaStream_t st) {
    int groups = C / GW, KG = K / groups;
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    int NT = KG % 128 == 0 ? 8 : KG % 112 == 0 ? 7 : KG % 64 == 0 ? 4 : KG % 32 == 0 ? 2 : 1;
    long long M = (long long)N * P * Q;
    dim3 grid((M + 127) / 128, K / (16 * NT));
    const __half* xp = (const __half*)x; const __half* wp = (const __half*)w;
    const __half* bp = (const __half*)bias; __half* yp = (__half*)y;
    // KB2/ST3, except stride 2 with >= 4 channel blocks per tap (RegNet GW 112): KB4/ST3
    if (cfg < 0) cfg = sh > 1 && GW >= 64 ? 3 : 0;
#define GC2(NN) gconv2_cfg<NN>(grid, st, xp, wp, bp, yp, nullptr, N, H, W, C, K, GW, KG, ldy, R, S, sh, sw, ph, pw, P, Q, act, cfg)
    switch (NT) { case 8: GC2(8); break; case 7: GC2(7); break; case 4: GC2(4); break; case 2: GC2(2); break; default: GC2(1); }
#undef GC2
    return (int)cudaGetLastError();
}

// y[n, p, c] = x[n, p, c] * sigmoid(s[n, c]); x / y [N, HW, C] fp16, s [N, C]
extern "C" int fvm_se_scale(const void* x, const void* s, void* y, int N, long long HW, int C, cudaStream_t st) {
    long long total = (long long)N * HW * (C / 8);
    se_scale_k<<<(total + 255) / 256, 256, 0, st>>>((const __half*)x, (const __half*)s, (__half*)y, HW, C / 8, total);
    return (int)cudaGetLastError();
}

// depthwise 3x3 / 5x5 / 7x7 (half2 per kernel row, fp32 across rows): x [N, H, W, C], w [R, S, C], y [N, P, Q, C];
// returns 1 for other kernel shapes
extern "C" int fvm_dwconv2(const void* x, const void* w, const void* bias, void* y, int N, int H, int W, int C, int R, int S,
                           int sh, int sw, int ph, int pw, int act, cudaStream_t st) {
    int rt = (R == S && (R == 3 || R == 5 || R == 7)) ? R : 0;
    if (rt == 0) return 1;
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    long long total = (long long)N * P * Q * (C / 8);
    const __half* xp = (const __half*)x; const __half* wp = (const __half*)w;
    const __half* bp = (const __half*)bias; __half* yp = (__half*)y;
#define DW2(RTV) dwconv2_k<1, RTV, 2><<<(total + 127) / 128, 128, 0, st>>>(xp, wp, bp, yp, N, H, W, C / 8, P, Q, R, S, sh, sw, ph, pw, act)
    if (rt == 3) DW2(3); else if (rt == 5) DW2(5); else DW2(7);
#undef DW2
    return (int)cudaGetLastError();
}

// direct stem conv (C <= 4, fp32 (f32 = 1) or fp16 NCHW input) + act + max pool (square pk, stride ps, pad pp) -> y
// [N, P2, Q2, K] fp16; w_t [R, S, C, K]; K in {16, 24, 32, 40, 48, 64}, else returns 1
extern "C" int fvm_stem_pool(const void* x, int f32, const void* w_t, const void* bias, void* y, int N, int C, int H, int W,
                             int K, int R, int S, int sh, int sw, int ph, int pw, int act, int pk, int ps, int pp, int P2,
                             int Q2, cudaStream_t st) {
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    int SH = (TP - 1) * ps + pk, SW = (TQ - 1) * ps + pk;
    int IH = (SH - 1) * sh + R, IW = (SW - 1) * sw + S;
    int KS = K == 64 ? 64 : (K / 8) % 2 ? K + 16 : K + 8;
    int smem = (SH * SW * KS + ((R * S * C * K + 7) / 8) * 8 + C * IH * IW) * 2;
    smem = (smem + 15) / 16 * 16;
    int tp_n = (P2 + TP - 1) / TP, tq_n = (Q2 + TQ - 1) / TQ;
    dim3 grid(tp_n * tq_n, N);
    const __half* wp = (const __half*)w_t; const __half* bp = (const __half*)bias;
    __half* yp = (__half*)y;
    bool fast = R == 3 && S == 3 && C == 3 && sh == 2 && sw == 2 && pk == 3;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 64) return 1;
#define SP_L(KK, KG, RT, CT, SST, PKT, TT)                                                                               \
    {                                                                                                                     \
        auto kern = stem_pool_kernel<KK, KG, RT, CT, SST, PKT, TT>;                                                       \
        static int attr[64] = {};                                                                                         \
        if (smem > attr[dev]) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr[dev] = smem; } \
        kern<<<grid, NTH, smem, st>>>((const TT*)x, wp, bp, yp, N, C, H, W, P, Q, R, S, sh, sw, ph, pw, act, pk, ps, pp, \
                                      P2, Q2, SH, SW, IH, IW, tq_n);                                                      \
    }
#define SP_T(KK, KG, RT, CT, SST, PKT)                                                                                    \
    if (f32) SP_L(KK, KG, RT, CT, SST, PKT, float)                                                                        \
    else SP_L(KK, KG, RT, CT, SST, PKT, __half)
#define SP_K(KK, KG)                                                                                                      \
    if (K == KK) {                                                                                                        \
        if (fast) SP_T(KK, KG, 3, 3, 2, 3) else SP_T(KK, KG, 0, 0, 0, 0)                                                  \
        return (int)cudaGetLastError();                                                                                   \
    }
    SP_K(16, 16) SP_K(24, 8) SP_K(32, 16) SP_K(40, 8) SP_K(48, 16) SP_K(64, 16)
#undef SP_K
#undef SP_T
#undef SP_L
    return 1;
}

// 1x1 conv on cuBLASLt (see above): act 0 / 1; bias [K] / res [M, ldr] or null; workspace of ws bytes. Returns 1 if
// cuBLASLt is not loadable or has no algorithm for the plan (the caller then runs the CUTLASS conv).
extern "C" int fvm_pw(const void* x, const void* w, const void* bias, const void* res, void* y, long long M, int K, int C,
                      int ldx, int ldy, int ldr, int act, void* workspace, size_t ws, cudaStream_t st) {
    if (!lt().ok) return 1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev >= 64) return 1;
    lt_handle h = lt_handle_for(dev);
    if (!h) return 1;
    int epi = (bias ? 2 : 0) + (act == 1 ? 1 : 0);
    LtPlan& p = lt_plan(LtKey{M, K, C, ldx, ldy, res ? ldr : 0, epi, dev, ws}, h);
    if (!p.ok) return 1;
    if (bias) lt().desc_set(p.op, LT_DESC_BIAS_POINTER, &bias, sizeof(bias));
    __half one = __float2half(1.f), beta = __float2half(res ? 1.f : 0.f);
    lt_status s = lt().matmul(h, p.op, &one, w, p.a, x, p.b, &beta, res ? res : y, p.c, y, p.d, &p.algo, workspace, ws, st);
    return s == 0 ? 0 : 1000 + s;
}
