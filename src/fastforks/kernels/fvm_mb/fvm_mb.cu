// MBConv-family classification kernels (fast-vision-models `mbconv.cu`, exp 008): the fused inverted-residual block
// (mbconv, used by the ShuffleNetV2 units), grouped conv (gconv), expand 1x1 + depthwise (xdw), depthwise with channel
// multiplier (dwmult), channel gather, EfficientViT ReLU linear attention on tensor cores (linattn2) and the direct stem
// conv (stem_direct). Kernel bodies unchanged; the torch host functions became `extern "C"` launchers and the Python
// wrapper (`__init__.py`) does the checks and allocations the host code did.
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int KC = 32;          // expanded channels per chunk
constexpr int NWARP = 8;
constexpr int NTH = NWARP * 32;
constexpr int SKC = KC + 8;     // smem row stride (halfs) of KC-wide buffers
constexpr int NTW_MAX = 16;     // n8 tiles per warp in the project GEMM (32 fp16x2 accumulator registers)

struct Args {
    const __half* x; const __half* w1; const __half* b1; const __half* wd; const __half* bd; const __half* w2;
    const __half* b2; const __half* res; __half* y;
    const __half* xb;       // input buffer base (channel 0; x = xb + cin_off)
    const int* colmap;      // [Cout/8] physical output channel of each 8-column block (null: identity)
    const int* gsrc;        // [G] passthrough gather: y[p, gdst-block j] = x[p, gsrc[j]] (-1: zero), stride-1 blocks only
    const int* gdst;        // [G/8] physical output channel of each gathered 8-block
    int G, ldx, ldy, gproj, dwf, ldwd, dbg;   // dwf: dw first (dw on x -> smem, then FFN: expand GEMM -> project GEMM)   // gproj: block-diagonal project (groups of KC channels, Cout == Ep)        // pixel strides of x (input buffer) and y
    int N, H, W, Cin, Cin_p, Ep, Cout, R, S, sh, sw, ph, pw, P, Q;
    int TH, TW, HH, HW, tp_n, tq_n, MT, cout_ch, nip;   // nip: max compact halo rows (padded to 16)
    int act1, act2, act3;
};

__device__ __forceinline__ float actf(float v, int act) {
    switch (act) {
        case 1: return fmaxf(v, 0.f);
        case 2: return fminf(fmaxf(v, 0.f), 6.f);
        case 3: return __fdividef(v, 1.f + __expf(-v));
        case 4: return v * fminf(fmaxf(v + 3.f, 0.f), 6.f) * (1.f / 6.f);
        case 5: return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
        case 6: return 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
    return v;
}

__device__ __forceinline__ __half2 acth2(__half2 v, int act) {   // relu / relu6 / none in half2, others via fp32
    if (act == 0) return v;
    if (act == 1) return __hmax2(v, __float2half2_rn(0.f));
    if (act == 2) return __hmin2(__hmax2(v, __float2half2_rn(0.f)), __float2half2_rn(6.f));
    float2 f = __half22float2(v);
    return __floats2half2_rn(actf(f.x, act), actf(f.y, act));
}

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred) {
    unsigned d = smem_u32(dst);
    int n = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
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

__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }
__device__ __forceinline__ __half2 u2h(unsigned u) { return *reinterpret_cast<__half2*>(&u); }

__global__ void __launch_bounds__(NTH) mbconv_kernel(const Args a) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const bool expand = a.w1 != nullptr;
    // ---- tile coordinates
    int t = blockIdx.x;
    const int tq = t % a.tq_n; t /= a.tq_n;
    const int tp = t % a.tp_n; const int n = t / a.tp_n;
    const int oy0 = tp * a.TH, ox0 = tq * a.TW;
    const int th = min(a.TH, a.P - oy0), tw = min(a.TW, a.Q - ox0);
    const int TP = th * tw;                                   // valid output pixels of this tile
    const int iy0 = oy0 * a.sh - a.ph, ix0 = ox0 * a.sw - a.pw;
    const int y0 = max(iy0, 0), x0 = max(ix0, 0);
    const int y1 = min(iy0 + (th - 1) * a.sh + a.R, a.H), x1 = min(ix0 + (tw - 1) * a.sw + a.S, a.W);
    const int cw = x1 - x0;
    const int NI = a.dwf ? a.MT * 16 : (y1 - y0) * cw;        // GEMM1 rows: compact in-image halo rows (dwf: tile pixels)
    const int NIp = (NI + 15) & ~15;
    const int co0 = blockIdx.y * a.cout_ch;
    const int cch = min(a.cout_ch, a.Cout - co0);             // output channels of this CTA (multiple of 8)
    const int sx = a.Cin_p + 8;
    // ---- shared memory carve
    __half* Xs = reinterpret_cast<__half*>(smem);                              // [nip][sx]  (expand only)
    __half* Ec = Xs + (expand ? a.nip * sx : 0);                                // [nip][SKC] (x2 without expand)
    __half* Dc = Ec + a.nip * SKC * (expand ? 1 : 2);                           // [MT*16][SKC]
    __half* W1s = Dc + a.MT * 16 * SKC;                                         // [2][KC][sx] (expand only)
    __half* W2s = W1s + (expand ? 2 * KC * sx : 0);                             // [2][cout_ch][SKC]
    const int nchunk = a.Ep / KC;
    const long long img = (long long)n * a.H * a.W;

    auto load_w = [&](int c, int buf) {
        if (expand) {
            const __half* src = a.w1 + (long long)c * KC * a.Cin_p;
            __half* dst = W1s + buf * KC * sx;
            const int cpr = a.Cin_p / 8;
            for (int i = tid; i < KC * cpr; i += NTH) {
                int r = i / cpr, k = i % cpr;
                cp16(dst + r * sx + k * 8, src + (long long)r * a.Cin_p + k * 8, true);
            }
        } else {   // input chunk (compact halo rows) -> Ec[buf]
            __half* dst = Ec + buf * a.nip * SKC;
            for (int i = tid; i < NI * (KC / 8); i += NTH) {
                int r = i / (KC / 8), k = i % (KC / 8);
                int gy = y0 + r / cw, gx = x0 + r % cw;
                int ch = c * KC + k * 8;
                bool ok = ch < a.Cin;
                const __half* src = a.x + (img + (long long)gy * a.W + gx) * a.ldx + (ok ? ch : 0);
                cp16(dst + r * SKC + k * 8, src, ok);
            }
        }
        const __half* src2 = a.gproj ? a.w2 + (long long)c * KC * KC : a.w2 + (long long)co0 * a.Ep + c * KC;
        const int ld2 = a.gproj ? KC : a.Ep;
        __half* dst2 = W2s + buf * a.cout_ch * SKC;
        for (int i = tid; i < (a.gproj ? KC : cch) * (KC / 8); i += NTH) {
            int r = i / (KC / 8), k = i % (KC / 8);
            cp16(dst2 + r * SKC + k * 8, src2 + (long long)r * ld2 + k * 8, true);
        }
    };

    // ---- dw first: Xs[px][Cin_p] = act2(dw(x) + bd) for the tile pixels (from global / L2), zero rows / pad channels
    if (a.dwf) {
        const int cpr = a.Cin_p / 8;
        for (int i = tid; i < NIp * cpr; i += NTH) {
            int px = i / cpr, cg = i % cpr;
            __align__(16) __half o[8];
            if (px < TP && cg * 8 < a.Cin) {
                float sacc[8];
                uint4 bv = __ldg(reinterpret_cast<const uint4*>(a.bd + cg * 8));
                const __half* hb = reinterpret_cast<const __half*>(&bv);
#pragma unroll
                for (int k = 0; k < 8; ++k) sacc[k] = __half2float(hb[k]);
                int oy = oy0 + px / tw, ox = ox0 + px % tw;
                for (int r = 0; r < a.R; ++r) {
                    int gy = oy * a.sh - a.ph + r;
                    if (gy < 0 || gy >= a.H) continue;
                    for (int q = 0; q < a.S; ++q) {
                        int gx = ox * a.sw - a.pw + q;
                        if (gx < 0 || gx >= a.W) continue;
                        uint4 v = __ldg(reinterpret_cast<const uint4*>(a.x + (img + (long long)gy * a.W + gx) * a.ldx + cg * 8));
                        uint4 w = __ldg(reinterpret_cast<const uint4*>(a.wd + (long long)(r * a.S + q) * a.ldwd + cg * 8));
                        const __half2* hv = reinterpret_cast<const __half2*>(&v);
                        const __half2* hw = reinterpret_cast<const __half2*>(&w);
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            float2 fv = __half22float2(hv[k]), fw = __half22float2(hw[k]);
                            sacc[2 * k] += fv.x * fw.x; sacc[2 * k + 1] += fv.y * fw.y;
                        }
                    }
                }
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __float2half(actf(sacc[k], a.act2));
            } else {
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __float2half(0.f);
            }
            *reinterpret_cast<uint4*>(Xs + px * sx + cg * 8) = *reinterpret_cast<const uint4*>(o);
        }
    } else if (expand) {
        const int cpr = a.Cin_p / 8;
        for (int i = tid; i < NIp * cpr; i += NTH) {
            int r = i / cpr, k = i % cpr;
            bool ok = r < NI && k * 8 < a.Cin;
            int rr = ok ? r : 0;
            int gy = y0 + rr / cw, gx = x0 + rr % cw;
            const __half* src = a.x + (img + (long long)gy * a.W + gx) * a.ldx + (ok ? k * 8 : 0);
            cp16(Xs + r * sx + k * 8, src, ok);
        }
    }
    load_w(0, 0);
    cp_commit();

    // project GEMM accumulators: warp -> m-tile (warp % MT), n-group (warp / MT) of ntw n8 tiles
    const int mt2 = warp % a.MT, ng = warp / a.MT, ngroups = NWARP / a.MT;
    const int ntiles = cch / 8;
    const int ntw = (ntiles + ngroups - 1) / ngroups;
    const int nt0 = ng * ntw;
    unsigned acc[NTW_MAX][2];
#pragma unroll
    for (int j = 0; j < NTW_MAX; ++j) acc[j][0] = acc[j][1] = 0u;

    const int g = lane >> 2, tq4 = lane & 3;
    for (int c = 0; c < nchunk; ++c) {
        const int buf = c & 1;
        if (c + 1 < nchunk) { load_w(c + 1, buf ^ 1); cp_commit(); cp_wait<1>(); } else { cp_wait<0>(); }
        __syncthreads();
        const __half* E;                       // expanded chunk [rows][SKC]
        if (expand) {
            // ---- expand GEMM: Ec[NI][KC] = act1(Xs . W1c^T + b1)
            const __half* W1c = W1s + buf * KC * sx;
            for (int mt = warp; mt < NIp / 16; mt += NWARP) {
                unsigned e[4][2] = {{0u, 0u}, {0u, 0u}, {0u, 0u}, {0u, 0u}};
                for (int k = 0; k < a.Cin_p; k += 16) {
                    unsigned af[4];
                    ldsm4(af[0], af[1], af[2], af[3], Xs + (mt * 16 + (lane & 15)) * sx + k + (lane >> 4) * 8);
#pragma unroll
                    for (int nb = 0; nb < 2; ++nb) {
                        unsigned b0, b1, b2, b3;
                        ldsm4(b0, b1, b2, b3, W1c + (nb * 16 + (lane >> 4) * 8 + (lane & 7)) * sx + k + ((lane >> 3) & 1) * 8);
                        mma16816(e[nb * 2], af, b0, b1);
                        mma16816(e[nb * 2 + 1], af, b2, b3);
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int col = j * 8 + tq4 * 2;
                    __half2 bb = *reinterpret_cast<const __half2*>(a.b1 + c * KC + col);
                    float2 fb = __half22float2(bb);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        int r = mt * 16 + g + h * 8;
                        float2 v = __half22float2(u2h(e[j][h]));
                        __half2 o = __floats2half2_rn(actf(v.x + fb.x, a.act1), actf(v.y + fb.y, a.act1));
                        if (r < NI) *reinterpret_cast<__half2*>(Ec + r * SKC + col) = o;
                    }
                }
            }
            __syncthreads();
            E = Ec;
        } else {
            E = Ec + buf * a.nip * SKC;
        }
        const __half* Ad = a.dwf ? Ec : Dc;    // A operand of the project GEMM
        if (!a.dwf) {
        // ---- depthwise RxS (+ bd, act2): Dc[px][KC], fp32 accumulation; rows >= TP are zero
        for (int i = tid; i < a.MT * 16 * (KC / 8); i += NTH) {
            int px = i / (KC / 8), cg = i % (KC / 8);
            float s[8];
            int ch = c * KC + cg * 8;
            __align__(16) __half o[8];
            if (px < TP) {
                uint4 bv = __ldg(reinterpret_cast<const uint4*>(a.bd + ch));
                const __half* hb = reinterpret_cast<const __half*>(&bv);
#pragma unroll
                for (int k = 0; k < 8; ++k) s[k] = __half2float(hb[k]);
                int oy = oy0 + px / tw, ox = ox0 + px % tw;
                for (int r = 0; r < a.R; ++r) {
                    int gy = oy * a.sh - a.ph + r;
                    if (gy < 0 || gy >= a.H) continue;
                    for (int q = 0; q < a.S; ++q) {
                        int gx = ox * a.sw - a.pw + q;
                        if (gx < 0 || gx >= a.W) continue;
                        uint4 v = *reinterpret_cast<const uint4*>(E + ((gy - y0) * cw + (gx - x0)) * SKC + cg * 8);
                        uint4 w = __ldg(reinterpret_cast<const uint4*>(a.wd + (long long)(r * a.S + q) * a.Ep + ch));
                        const __half2* hv = reinterpret_cast<const __half2*>(&v);
                        const __half2* hw = reinterpret_cast<const __half2*>(&w);
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            float2 fv = __half22float2(hv[k]), fw = __half22float2(hw[k]);
                            s[2 * k] += fv.x * fw.x; s[2 * k + 1] += fv.y * fw.y;
                        }
                    }
                }
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __float2half(actf(s[k], a.act2));
            } else {
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __float2half(0.f);
            }
            *reinterpret_cast<uint4*>(Dc + px * SKC + cg * 8) = *reinterpret_cast<const uint4*>(o);
        }
        __syncthreads();
        }
        if (a.gproj) {   // ---- grouped project: chunk c -> output channels [c*KC, c*KC + KC), written directly
            const __half* W2c = W2s + buf * a.cout_ch * SKC;
            for (int tl = warp; tl < a.MT * (KC / 8); tl += NWARP) {
                int mt = tl / (KC / 8), nt = tl % (KC / 8);
                unsigned d[2] = {0u, 0u};
#pragma unroll
                for (int kk = 0; kk < KC; kk += 16) {
                    unsigned af[4], b0, b1;
                    ldsm4(af[0], af[1], af[2], af[3], Dc + (mt * 16 + (lane & 15)) * SKC + kk + (lane >> 4) * 8);
                    ldsm2(b0, b1, W2c + (nt * 8 + (lane & 7)) * SKC + kk + ((lane >> 3) & 1) * 8);
                    mma16816(d, af, b0, b1);
                }
                int co = c * KC + nt * 8 + tq4 * 2;
                int cph = a.colmap ? a.colmap[co >> 3] + (co & 7) : co;
                float2 fb = a.b2 ? __half22float2(*reinterpret_cast<const __half2*>(a.b2 + co)) : make_float2(0.f, 0.f);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int px = mt * 16 + g + h * 8;
                    if (px >= TP) continue;
                    int oy = oy0 + px / tw, ox = ox0 + px % tw;
                    long long pix = ((long long)n * a.P + oy) * a.Q + ox;
                    float2 v = __half22float2(u2h(d[h]));
                    *reinterpret_cast<__half2*>(a.y + pix * a.ldy + cph) =
                        __floats2half2_rn(actf(v.x + fb.x, a.act3), actf(v.y + fb.y, a.act3));
                }
            }
        } else
        // ---- project GEMM: acc[TP][cch] += Dc . W2c^T
        {
            const __half* W2c = W2s + buf * a.cout_ch * SKC;
#pragma unroll
            for (int kk = 0; kk < KC; kk += 16) {
                unsigned af[4];
                ldsm4(af[0], af[1], af[2], af[3], Ad + (mt2 * 16 + (lane & 15)) * SKC + kk + (lane >> 4) * 8);
#pragma unroll
                for (int j = 0; j < NTW_MAX; ++j) {
                    if (j < ntw && nt0 + j < ntiles) {
                        unsigned b0, b1;
                        ldsm2(b0, b1, W2c + ((nt0 + j) * 8 + (lane & 7)) * SKC + kk + ((lane >> 3) & 1) * 8);
                        mma16816(acc[j], af, b0, b1);
                    }
                }
            }
        }
        __syncthreads();
    }
    // ---- epilogue: + b2 (+ res), act3 -> y
#pragma unroll
    for (int j = 0; j < NTW_MAX; ++j) {
        if (a.gproj) break;
        if (j < ntw && nt0 + j < ntiles) {
            int co = co0 + (nt0 + j) * 8 + tq4 * 2;
            int cph = a.colmap ? a.colmap[co >> 3] + (co & 7) : co;   // physical output channel
            float2 fb = a.b2 ? __half22float2(*reinterpret_cast<const __half2*>(a.b2 + co)) : make_float2(0.f, 0.f);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int px = mt2 * 16 + g + h * 8;
                if (px >= TP) continue;
                int oy = oy0 + px / tw, ox = ox0 + px % tw;
                long long pix = ((long long)n * a.P + oy) * a.Q + ox;
                float2 v = __half22float2(u2h(acc[j][h]));
                v.x += fb.x; v.y += fb.y;
                if (a.res) {
                    float2 r = __half22float2(*reinterpret_cast<const __half2*>(a.res + pix * a.Cout + co));
                    v.x += r.x; v.y += r.y;
                }
                *reinterpret_cast<__half2*>(a.y + pix * a.ldy + cph) = __floats2half2_rn(actf(v.x, a.act3), actf(v.y, a.act3));
            }
        }
    }
    // ---- passthrough gather (ShuffleNet identity half; stride 1: output pixel == input pixel)
    if (a.G && blockIdx.y == 0) {
        for (int i = tid; i < TP * (a.G / 8); i += NTH) {
            int px = i / (a.G / 8), b = i % (a.G / 8);
            int oy = oy0 + px / tw, ox = ox0 + px % tw;
            const __half* xr = a.xb + (img + (long long)oy * a.W + ox) * a.ldx;
            __align__(16) __half o[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) { int sc = a.gsrc[b * 8 + k]; o[k] = sc >= 0 ? xr[sc] : __float2half(0.f); }
            *reinterpret_cast<uint4*>(a.y + (((long long)n * a.P + oy) * a.Q + ox) * a.ldy + a.gdst[b]) =
                *reinterpret_cast<const uint4*>(o);
        }
    }
}

int smem_bytes(bool expand, int nip, int Cin_p, int MT, int cout_ch) {
    int sx = Cin_p + 8;
    long long h = (expand ? (long long)nip * sx : 0) + (long long)nip * SKC * (expand ? 1 : 2) + MT * 16 * SKC +
                  (expand ? 2 * KC * sx : 0) + 2LL * cout_ch * SKC;
    return int(h * 2);
}

const int SMEM_MAX = 99 * 1024;
const int SMEM_TOTAL = 100 * 1024;

}  // namespace

// per-device caches (the torch host code used process-wide statics bound to the first device)
static int sm_count() {
    static int n[64] = {};
    int d = 0;
    cudaGetDevice(&d);
    if (d >= 64) return 36;
    if (!n[d]) {
        int v = 36;
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d);
        n[d] = v;
    }
    return n[d];
}

static bool smem_attr(const void* f, int smem, int* done) {
    int d = 0;
    cudaGetDevice(&d);
    if (d < 64 && done[d] >= smem) return true;
    if (cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return false;
    if (d < 64) done[d] = smem;
    return true;
}

// x = input buffer base (channel 0, pixel stride ldx); w1 [Ep, Cin_p] or null; b1 [Ep]; wd [R, S, ldwd]; bd [Ep];
// w2 [Cout, Ep] (or [Ep / KC, KC, KC] if gproj); b2 [Cout] or null; res [N, P, Q, Cout] or null; y pixel stride ldy;
// colmap [Cout / 8] / gsrc [G] / gdst [G / 8] int32 or null. Same tile rule as the torch op.
// Returns 0, 2 (smem plan too large / attribute refused) or a CUDA error.
extern "C" int fvm_mbconv(const void* x, const void* w1, const void* b1, const void* wd, const void* bd, const void* w2,
                          const void* b2, const void* res, void* y, const int* colmap, const int* gsrc, const int* gdst,
                          int G, int N, int H, int W, int ldx, int cin_off, int Cin, int Cin_p, int R, int S, int ldwd, int Ep,
                          int Cout, int gproj, int ldy, int sh, int sw, int ph, int pw, int act1, int act2, int act3, int tile,
                          int dwf, cudaStream_t st) {
    Args a{};
    a.N = N; a.H = H; a.W = W; a.ldx = ldx; a.Cin = Cin; a.R = R; a.S = S; a.ldwd = ldwd; a.Ep = Ep; a.dwf = dwf;
    a.gproj = gproj; a.Cout = Cout; a.sh = sh; a.sw = sw; a.ph = ph; a.pw = pw;
    a.P = (H + 2 * ph - R) / sh + 1; a.Q = (W + 2 * pw - S) / sw + 1;
    bool expand = w1 != nullptr;
    a.Cin_p = Cin_p;
    a.xb = (const __half*)x; a.x = a.xb + cin_off;
    a.w1 = (const __half*)w1; a.b1 = (const __half*)b1; a.wd = (const __half*)wd; a.bd = (const __half*)bd;
    a.w2 = (const __half*)w2; a.b2 = (const __half*)b2; a.res = (const __half*)res; a.y = (__half*)y;
    a.ldy = ldy; a.colmap = colmap; a.gsrc = gsrc; a.gdst = gdst; a.G = gsrc ? G : 0;
    a.act1 = act1; a.act2 = act2; a.act3 = act3;
    int tgt = tile > 0 ? tile : 8;
    auto side = [&](int L) { int nt = (L + tgt - 1) / tgt; return (L + nt - 1) / nt; };
    a.TH = side(a.P); a.TW = side(a.Q);
    for (;;) {
        int TP = a.TH * a.TW;
        a.MT = TP > 64 ? 8 : TP > 32 ? 4 : TP > 16 ? 2 : 1;
        int maxch = 8 * NTW_MAX * (NWARP / a.MT);
        int nch = (a.Cout + maxch - 1) / maxch;
        a.cout_ch = a.gproj ? KC : ((a.Cout / 8 + nch - 1) / nch) * 8;
        a.HH = (a.TH - 1) * a.sh + a.R; a.HW = (a.TW - 1) * a.sw + a.S;
        a.nip = dwf ? a.MT * 16 : (a.HH * a.HW + 15) & ~15;
        if (smem_bytes(expand, a.nip, a.Cin_p, a.MT, a.cout_ch) <= SMEM_MAX || (a.TH == 1 && a.TW == 1)) break;
        if (a.TH >= a.TW) a.TH = (a.TH + 1) / 2; else a.TW = (a.TW + 1) / 2;
    }
    int smem = smem_bytes(expand, a.nip, a.Cin_p, a.MT, a.cout_ch);
    if (smem > SMEM_MAX) return 2;
    a.tp_n = (a.P + a.TH - 1) / a.TH; a.tq_n = (a.Q + a.TW - 1) / a.TW;
    static int attr[64] = {};
    if (!smem_attr((const void*)mbconv_kernel, SMEM_MAX, attr)) return 2;
    dim3 grid(a.N * a.tp_n * a.tq_n, a.gproj ? 1 : (a.Cout + a.cout_ch - 1) / a.cout_ch);
    mbconv_kernel<<<grid, NTH, smem, st>>>(a);
    return (int)cudaGetLastError();
}

// grouped conv (group width GW = C/G = K/G, GW % 16 == 0), NHWC fp16, weight [K, R, S, GW], fp16 accumulation (mma.sync),
// + bias + act. Multistage cp.async implicit GEMM: CTA = 128 output pixels x BN = 16 * NT output channels of one group,
// 8 warps = 4 (M, 32 rows) x 2 (N, 8 * NT cols); k blocks of 16 channels of one tap, 2 per stage.
template <int NT, int MI>     // MI m16 tiles per warp: warps = (128 / (16 MI)) x 2
__global__ void __launch_bounds__(32 * (8 / MI) * 2) gconv_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                                                    const __half* __restrict__ bias, __half* __restrict__ y, int N, int H,
                                                    int W, int C, int K, int GW, int R, int S, int sh, int sw, int ph, int pw,
                                                    int P, int Q, int act) {
    constexpr int BM = 128, BN = 16 * NT, ST = 3, SK = 32;   // 64-B rows, 16-B chunks XOR-swizzled by (row >> 1) & 3
    constexpr int WMN = 8 / MI, NTH = 32 * WMN * 2, RPT = BM * 4 / NTH;   // warps along M, threads, A chunks per thread
    extern __shared__ __align__(16) unsigned char smem[];
    __half* As = reinterpret_cast<__half*>(smem);          // [ST][BM][SK]
    __half* Bs = As + ST * BM * SK;                          // [ST][BN][SK]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long M = (long long)N * P * Q;
    const long long m0 = (long long)blockIdx.x * BM;
    const int k0 = blockIdx.y * BN;                          // first output channel
    const int grp = k0 / GW, cbase = grp * GW;
    const int kpb = GW / 16, nkb = R * S * kpb, nst = (nkb + 1) / 2;
    // A rows of this thread: tid/4 + j * NTH/4
    int an[RPT], ap[RPT], aq[RPT]; bool av[RPT];
#pragma unroll
    for (int j = 0; j < RPT; ++j) {
        long long m = m0 + tid / 4 + (NTH / 4) * j;
        av[j] = m < M;
        long long mm = av[j] ? m : 0;
        aq[j] = mm % Q; long long t = mm / Q; ap[j] = t % P; an[j] = t / P;
    }
    auto load = [&](int st, int buf) {
        __half* a_ = As + buf * BM * SK;
        __half* b_ = Bs + buf * BN * SK;
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            int row = tid / 4 + (NTH / 4) * j, part = tid % 4;
            int kb = st * 2 + part / 2;
            bool ok = av[j] && kb < nkb;
            const __half* src = x;
            if (ok) {
                int tap = kb / kpb, cb = kb % kpb, r = tap / S, s2 = tap % S;
                int ih = ap[j] * sh - ph + r, iw = aq[j] * sw - pw + s2;
                ok = ih >= 0 && ih < H && iw >= 0 && iw < W;
                if (ok) src = x + (((long long)an[j] * H + ih) * W + iw) * C + cbase + cb * 16 + (part & 1) * 8;
            }
            cp16(a_ + row * SK + ((part ^ ((row >> 1) & 3)) * 8), src, ok);
        }
        for (int i = tid; i < BN * 4; i += NTH) {
            int row = i / 4, part = i % 4;
            int kb = st * 2 + part / 2;
            bool ok = kb < nkb;
            const __half* src = w;
            if (ok) {
                int tap = kb / kpb, cb = kb % kpb;
                src = w + ((long long)(k0 + row) * R * S + tap) * GW + cb * 16 + (part & 1) * 8;
            }
            cp16(b_ + row * SK + ((part ^ ((row >> 1) & 3)) * 8), src, ok);
        }
    };
    const int wm = warp % WMN, wn = warp / WMN;
    unsigned acc[MI][NT][2];
#pragma unroll
    for (int i = 0; i < MI; ++i)
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
        for (int kk = 0; kk < 32; kk += 16) {
            unsigned af[MI][4];
#pragma unroll
            for (int i = 0; i < MI; ++i)
            {
                int row = wm * MI * 16 + i * 16 + (lane & 15), ch = kk / 8 + (lane >> 4);
                ldsm4(af[i][0], af[i][1], af[i][2], af[i][3], a_ + row * SK + ((ch ^ ((row >> 1) & 3)) * 8));
            }
#pragma unroll
            for (int j = 0; j < NT; j += 2) {
                int brow = wn * NT * 8 + j * 8 + (lane >> 4) * 8 + (lane & 7), bch = kk / 8 + ((lane >> 3) & 1);
                const __half* bp = b_ + brow * SK + ((bch ^ ((brow >> 1) & 3)) * 8);
                if (j + 1 < NT) {
                    unsigned b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, bp);
#pragma unroll
                    for (int i = 0; i < MI; ++i) { mma16816(acc[i][j], af[i], b0, b1); mma16816(acc[i][j + 1], af[i], b2, b3); }
                } else {
                    unsigned b0, b1;
                    int r2 = wn * NT * 8 + j * 8 + (lane & 7);
                    ldsm2(b0, b1, b_ + r2 * SK + (((kk / 8 + ((lane >> 3) & 1)) ^ ((r2 >> 1) & 3)) * 8));
#pragma unroll
                    for (int i = 0; i < MI; ++i) mma16816(acc[i][j], af[i], b0, b1);
                }
            }
        }
    }
    cp_wait<0>();
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        int co = k0 + wn * NT * 8 + j * 8 + t4 * 2;
        float2 fb = bias ? __half22float2(*reinterpret_cast<const __half2*>(bias + co)) : make_float2(0.f, 0.f);
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                long long m = m0 + wm * MI * 16 + i * 16 + g + h * 8;
                if (m >= M) continue;
                float2 v = __half22float2(u2h(acc[i][j][h]));
                *reinterpret_cast<__half2*>(y + m * K + co) = __floats2half2_rn(actf(v.x + fb.x, act), actf(v.y + fb.y, act));
            }
    }
}

template <int NT, int MI>
static int gconv_launch1(dim3 grid, cudaStream_t st, const __half* x, const __half* w, const __half* b, __half* y, int N, int H,
                         int W, int C, int K, int GW, int R, int S, int sh, int sw, int ph, int pw, int P, int Q, int act) {
    int smem = 3 * (128 + 16 * NT) * 32 * 2;
    static int attr[64] = {};
    if (!smem_attr((const void*)gconv_kernel<NT, MI>, smem, attr)) return 2;
    gconv_kernel<NT, MI><<<grid, 32 * (8 / MI) * 2, smem, st>>>(x, w, b, y, N, H, W, C, K, GW, R, S, sh, sw, ph, pw, P, Q, act);
    return (int)cudaGetLastError();
}

// x [N, H, W, C] fp16; w [K, R, S, GW]; groups = C / GW (K / groups == GW, GW % 16 == 0); y [N, P, Q, K]
extern "C" int fvm_gconv(const void* x, const void* w, const void* bias, void* y, int N, int H, int W, int C, int K, int GW,
                         int R, int S, int sh, int sw, int ph, int pw, int act, cudaStream_t st) {
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    int NT = GW % 128 == 0 ? 8 : GW == 112 ? 7 : GW % 64 == 0 ? 4 : 2;   // BN = 16 * NT divides GW
    long long M = (long long)N * P * Q;
    dim3 grid((M + 127) / 128, K / (16 * NT));
    const __half* xp = (const __half*)x; const __half* wp = (const __half*)w;
    const __half* bp = (const __half*)bias; __half* yp = (__half*)y;
    switch (NT) {
        case 8: return gconv_launch1<8, 2>(grid, st, xp, wp, bp, yp, N, H, W, C, K, GW, R, S, sh, sw, ph, pw, P, Q, act);
        case 7: return gconv_launch1<7, 2>(grid, st, xp, wp, bp, yp, N, H, W, C, K, GW, R, S, sh, sw, ph, pw, P, Q, act);
        case 4: return gconv_launch1<4, 2>(grid, st, xp, wp, bp, yp, N, H, W, C, K, GW, R, S, sh, sw, ph, pw, P, Q, act);
        default: return gconv_launch1<2, 2>(grid, st, xp, wp, bp, yp, N, H, W, C, K, GW, R, S, sh, sw, ph, pw, P, Q, act);
    }
}

// depthwise conv with channel multiplier (groups == C, K = C * mult; FastViT patch embeddings), NHWC fp16, w [R, S, K]
__global__ void dwmult_kernel(const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
                              __half* __restrict__ y, int N, int H, int W, int C, int K8, int mult, int P, int Q, int R, int S,
                              int sh, int sw, int ph, int pw, int act) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * K8;
    if (i >= total) return;
    int c = i % K8; long long t = i / K8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = b ? __half2float(b[c * 8 + k]) : 0.f;
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int ww = q * sw - pw + s2; if (ww < 0 || ww >= W) continue;
            const __half* xp = x + (((long long)n * H + h) * W + ww) * C;
            uint4 g = __ldg(reinterpret_cast<const uint4*>(w + ((long long)r * S + s2) * K8 * 8) + c);
            const __half* hg = reinterpret_cast<const __half*>(&g);
            if (mult == 2) {     // 8 outputs <- 4 consecutive inputs (8-byte load)
                uint2 u = __ldg(reinterpret_cast<const uint2*>(xp + c * 4));
                const __half* hx = reinterpret_cast<const __half*>(&u);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] += __half2float(hx[k >> 1]) * __half2float(hg[k]);
            } else {
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] += __half2float(__ldg(xp + (c * 8 + k) / mult)) * __half2float(hg[k]);
            }
        }
    }
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(actf(acc[k], act));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

// x [N, H, W, C] fp16; w [R, S, K] (K = C * mult, K % 8 == 0); y [N, P, Q, K]
extern "C" int fvm_dwmult(const void* x, const void* w, const void* bias, void* y, int N, int H, int W, int C, int K, int R,
                          int S, int sh, int sw, int ph, int pw, int act, cudaStream_t st) {
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    long long total = (long long)N * P * Q * (K / 8);
    dwmult_kernel<<<(total + 127) / 128, 128, 0, st>>>((const __half*)x, (const __half*)w, (const __half*)bias, (__half*)y, N,
                                                       H, W, C, K / 8, K / C, P, Q, R, S, sh, sw, ph, pw, act);
    return (int)cudaGetLastError();
}

// expand 1x1 (+b1, act1) + depthwise RxS (+bd, act2) in one kernel; the expanded tensor stays in smem, the dw output D
// [N, P, Q, E] goes to global (the project 1x1 stays a CUTLASS conv with fused residual).
// Persistent streaming CTAs: grid.y = chunk of XC expanded channels (its W1 rows, dw weights and biases stay in smem), the
// CTA loops over output tiles; the next tile's input halo is prefetched (cp.async, double buffer) while the current tile
// runs expand GEMM (mma.sync) -> smem -> dw -> global. w1 [Ep][Cin_p], b1 [Ep], wd [R][S][Ep], bd [Ep].
constexpr int XC = 32, XNT = 256;
template <int RT, int ST_>   // compile-time taps (0 = runtime)
__global__ void __launch_bounds__(XNT) xdw_kernel(const Args a) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int R = RT ? RT : a.R, S = ST_ ? ST_ : a.S;
    const int c0 = blockIdx.y * XC;
    const int sx = a.Cin_p + 8;
    constexpr int SE = XC + 8;
    __half* W1s = reinterpret_cast<__half*>(smem);   // [XC][sx]
    __half* Wds = W1s + XC * sx;                     // [R*S][XC]
    __half* B1s = Wds + R * S * XC;                  // [XC]
    __half* Bds = B1s + XC;                          // [XC]
    __half* Ec = Bds + XC;                           // [nip][SE]
    __half* Xs0 = Ec + a.nip * SE;                   // [2][nip][sx]
    const int cpr = a.Cin_p / 8;
    for (int i = tid; i < XC * cpr; i += XNT) {
        int r = i / cpr, k = i % cpr;
        cp16(W1s + r * sx + k * 8, a.w1 + (long long)(c0 + r) * a.Cin_p + k * 8, true);
    }
    for (int i = tid; i < R * S * (XC / 8); i += XNT) {
        int tap = i / (XC / 8), k = i % (XC / 8);
        cp16(Wds + tap * XC + k * 8, a.wd + (long long)tap * a.Ep + c0 + k * 8, true);
    }
    if (tid < XC / 8) cp16(B1s + tid * 8, a.b1 + c0 + tid * 8, true);
    else if (tid < XC / 4) cp16(Bds + (tid - XC / 8) * 8, a.bd + c0 + (tid - XC / 8) * 8, true);
    const int ntiles = a.N * a.tp_n * a.tq_n;
    struct Tile { int n, oy0, ox0, th, tw, y0, x0, cw, NI; };
    auto tile_of = [&](int t) {
        Tile T;
        int tq = t % a.tq_n; t /= a.tq_n;
        int tp = t % a.tp_n; T.n = t / a.tp_n;
        T.oy0 = tp * a.TH; T.ox0 = tq * a.TW;
        T.th = min(a.TH, a.P - T.oy0); T.tw = min(a.TW, a.Q - T.ox0);
        int iy0 = T.oy0 * a.sh - a.ph, ix0 = T.ox0 * a.sw - a.pw;
        T.y0 = max(iy0, 0); T.x0 = max(ix0, 0);
        int y1 = min(iy0 + (T.th - 1) * a.sh + R, a.H), x1 = min(ix0 + (T.tw - 1) * a.sw + S, a.W);
        T.cw = x1 - T.x0; T.NI = (y1 - T.y0) * T.cw;
        return T;
    };
    auto load_x = [&](const Tile& T, __half* Xs) {
        const long long img = (long long)T.n * a.H * a.W;
        int NIp = (T.NI + 15) & ~15;
        for (int i = tid; i < NIp * cpr; i += XNT) {
            int r = i / cpr, k = i % cpr;
            bool ok = r < T.NI && k * 8 < a.Cin;
            int rr = ok ? r : 0;
            int gy = T.y0 + rr / T.cw, gx = T.x0 + rr % T.cw;
            cp16(Xs + r * sx + k * 8, a.x + (img + (long long)gy * a.W + gx) * a.ldx + (ok ? k * 8 : 0), ok);
        }
    };
    int t = blockIdx.x;
    Tile cur;
    if (t < ntiles) { cur = tile_of(t); load_x(cur, Xs0); }
    cp_commit();
    const int g = lane >> 2, tq4 = lane & 3;
    for (int it = 0; t < ntiles; ++it, t += gridDim.x) {
        __half* Xs = Xs0 + (it & 1) * a.nip * sx;
        Tile nxt;
        bool more = t + (int)gridDim.x < ntiles;
        if (more) { nxt = tile_of(t + gridDim.x); if (!(a.dbg & 4)) load_x(nxt, Xs0 + ((it + 1) & 1) * a.nip * sx); }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        // ---- expand GEMM: Ec[NI][XC] = act1(Xs . W1^T + b1)
        const int NIp = (a.dbg & 1) ? 0 : (cur.NI + 15) & ~15;
        for (int mt = warp; mt < NIp / 16; mt += XNT / 32) {
            unsigned e[4][2] = {{0u, 0u}, {0u, 0u}, {0u, 0u}, {0u, 0u}};
            for (int k = 0; k < a.Cin_p; k += 16) {
                unsigned af[4];
                ldsm4(af[0], af[1], af[2], af[3], Xs + (mt * 16 + (lane & 15)) * sx + k + (lane >> 4) * 8);
#pragma unroll
                for (int nb = 0; nb < 2; ++nb) {
                    unsigned b0, b1, b2, b3;
                    ldsm4(b0, b1, b2, b3, W1s + (nb * 16 + (lane >> 4) * 8 + (lane & 7)) * sx + k + ((lane >> 3) & 1) * 8);
                    mma16816(e[nb * 2], af, b0, b1);
                    mma16816(e[nb * 2 + 1], af, b2, b3);
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int col = j * 8 + tq4 * 2;
                __half2 hb = *reinterpret_cast<const __half2*>(B1s + col);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int r = mt * 16 + g + h * 8;
                    *reinterpret_cast<__half2*>(Ec + r * SE + col) = acth2(__hadd2(u2h(e[j][h]), hb), a.act1);
                }
            }
        }
        __syncthreads();
        // ---- depthwise (+bd, act2) -> D (global)
        const int TP = cur.th * cur.tw;
        for (int i = tid; i < TP * (XC / 8); i += XNT) {
            int px = i / (XC / 8), cg = i % (XC / 8);
            int ch = c0 + cg * 8;
            if (ch >= a.Cout) continue;
            if (a.dbg & 2) {
                int oy = cur.oy0 + px / cur.tw, ox = cur.ox0 + px % cur.tw;
                *reinterpret_cast<uint4*>(a.y + (((long long)cur.n * a.P + oy) * a.Q + ox) * a.Cout + ch) = make_uint4(0, 0, 0, 0);
                continue;
            }
            __half2 hacc[4];   // fp16 accumulation (as the tensor-core convs): 4 HFMA2 per tap, no conversions
            {
                uint4 bv = *reinterpret_cast<const uint4*>(Bds + cg * 8);
                const __half2* hb = reinterpret_cast<const __half2*>(&bv);
#pragma unroll
                for (int k = 0; k < 4; ++k) hacc[k] = hb[k];
            }
            int oy = cur.oy0 + px / cur.tw, ox = cur.ox0 + px % cur.tw;
            int gy0 = oy * a.sh - a.ph, gx0 = ox * a.sw - a.pw;
#pragma unroll
            for (int r = 0; r < (RT ? RT : 8); ++r) {
                if (!RT && r >= R) break;
                int gy = gy0 + r;
                if (gy < 0 || gy >= a.H) continue;
#pragma unroll
                for (int q = 0; q < (ST_ ? ST_ : 8); ++q) {
                    if (!ST_ && q >= S) break;
                    int gx = gx0 + q;
                    if (gx < 0 || gx >= a.W) continue;
                    uint4 v = *reinterpret_cast<const uint4*>(Ec + ((gy - cur.y0) * cur.cw + (gx - cur.x0)) * SE + cg * 8);
                    uint4 w = *reinterpret_cast<const uint4*>(Wds + (r * S + q) * XC + cg * 8);
                    const __half2* hv = reinterpret_cast<const __half2*>(&v);
                    const __half2* hw = reinterpret_cast<const __half2*>(&w);
#pragma unroll
                    for (int k = 0; k < 4; ++k) hacc[k] = __hfma2(hv[k], hw[k], hacc[k]);
                }
            }
            __align__(16) __half2 o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) o[k] = acth2(hacc[k], a.act2);
            *reinterpret_cast<uint4*>(a.y + (((long long)cur.n * a.P + oy) * a.Q + ox) * a.Cout + ch) =
                *reinterpret_cast<const uint4*>(o);
        }
        __syncthreads();
        cur = nxt;
    }
    cp_wait<0>();
}

template <int RT, int ST_>
static int xdw_go(const Args& a, dim3 grid, int smem, cudaStream_t st) {
    static int attr[64] = {};
    if (!smem_attr((const void*)xdw_kernel<RT, ST_>, SMEM_MAX, attr)) return 2;
    xdw_kernel<RT, ST_><<<grid, XNT, smem, st>>>(a);
    return (int)cudaGetLastError();
}

// x [N, H, W, Cin] fp16; w1 [Ep, Cin_p]; b1 [Ep]; wd [R, S, Ep]; bd [Ep]; y [N, P, Q, E]
extern "C" int fvm_xdw(const void* x, const void* w1, const void* b1, const void* wd, const void* bd, void* y, int N, int H,
                       int W, int Cin, int Cin_p, int Ep, int E, int R, int S, int sh, int sw, int ph, int pw, int act1,
                       int act2, int tile, cudaStream_t st) {
    Args a{};
    a.N = N; a.Cin = Cin; a.H = H; a.W = W; a.ldx = Cin;
    a.R = R; a.S = S; a.Ep = Ep; a.Cin_p = Cin_p; a.Cout = E;
    a.sh = sh; a.sw = sw; a.ph = ph; a.pw = pw;
    a.P = (H + 2 * ph - R) / sh + 1; a.Q = (W + 2 * pw - S) / sw + 1;
    a.x = (const __half*)x; a.w1 = (const __half*)w1; a.b1 = (const __half*)b1;
    a.wd = (const __half*)wd; a.bd = (const __half*)bd;
    a.act1 = act1; a.act2 = act2;
    a.y = (__half*)y;
    int tgt = tile > 0 ? tile : 8;
    auto side = [&](int L) { int nt = (L + tgt - 1) / tgt; return (L + nt - 1) / nt; };
    a.TH = side(a.P); a.TW = side(a.Q);
    const int XBUDGET = 48 * 1024;      // aim for >= 2 CTAs (16 warps) per SM
    int smem = 0;
    for (;;) {
        a.HH = (a.TH - 1) * a.sh + a.R; a.HW = (a.TW - 1) * a.sw + a.S;
        a.nip = (a.HH * a.HW + 15) & ~15;
        smem = 2 * (XC * (a.Cin_p + 8) + a.R * a.S * XC + 2 * XC + a.nip * (XC + 8) + 2 * a.nip * (a.Cin_p + 8));
        if (smem <= XBUDGET || (a.TH * a.TW <= 16)) break;
        if (a.TH >= a.TW) a.TH = (a.TH + 1) / 2; else a.TW = (a.TW + 1) / 2;
    }
    if (smem > SMEM_MAX) return 2;
    a.tp_n = (a.P + a.TH - 1) / a.TH; a.tq_n = (a.Q + a.TW - 1) / a.TW;
    int ntiles = a.N * a.tp_n * a.tq_n, nch = (E + XC - 1) / XC;
    int per_sm = std::max(1, std::min(4, SMEM_TOTAL / std::max(smem, 1)));
    int gx = std::max(1, std::min(ntiles, (sm_count() * per_sm + nch - 1) / nch));
    dim3 grid(gx, nch);
    if (a.R == 3 && a.S == 3) return xdw_go<3, 3>(a, grid, smem, st);
    if (a.R == 5 && a.S == 5) return xdw_go<5, 5>(a, grid, smem, st);
    return xdw_go<0, 0>(a, grid, smem, st);
}

// out[p, gdst[b] + k] = x[p, gsrc[8 b + k]] (-1: 0) for every pixel (ShuffleNet passthrough channels), x / out channels-last
__global__ void chan_gather_kernel(const __half* __restrict__ x, __half* __restrict__ y, const int* __restrict__ gsrc,
                                   const int* __restrict__ gdst, long long npix, int nb, int ldx, int ldy) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= npix * nb) return;
    int b = i % nb; long long p = i / nb;
    const __half* xr = x + p * ldx;
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { int sc = __ldg(gsrc + b * 8 + k); o[k] = sc >= 0 ? xr[sc] : __float2half(0.f); }
    *reinterpret_cast<uint4*>(y + p * ldy + __ldg(gdst + b)) = *reinterpret_cast<const uint4*>(o);
}

// out[p, gdst[b] + k] = x[p, gsrc[8 b + k]] for npix pixels; ldx / ldy = channel counts of x / out
extern "C" int fvm_chan_gather(const void* x, void* y, const int* gsrc, const int* gdst, long long npix, int nb, int ldx,
                               int ldy, cudaStream_t st) {
    chan_gather_kernel<<<(npix * nb + 255) / 256, 256, 0, st>>>((const __half*)x, (__half*)y, gsrc, gdst, npix, nb, ldx, ldy);
    return (int)cudaGetLastError();
}

__device__ __forceinline__ void ldsm4t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816f(float* d, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ReLU linear attention (EfficientViT LiteMLA, head dim 32) on tensor cores, fp32 accumulation. Heads 0..HA-1 come from
// xa [B, H, W, HA*96], heads HA.. from xb [B, H, W, HB*96] (per head [q | k | v]: the qkv conv output and the multi-scale
// aggregate, no concatenation). One CTA (4 warps) per (head, image):
//   phase A: KV[32][40] = sum_p relu(k_p)^T [v_p, 1, 0..] / HW   (mma, A = K^T via ldmatrix.trans, warps split the pixels)
//   phase B: out_p = relu(q_p) KV[:, :32] / (relu(q_p) KV[:, 32] + eps / HW)  -> y[b, p, h*32 + d]
__global__ void __launch_bounds__(128) linattn_tc(const __half* __restrict__ xa, const __half* __restrict__ xb,
                                                  __half* __restrict__ y, int HW, int HA, int HB, float eps) {
    constexpr int SK = 40, SV = 48;
    __shared__ __align__(16) __half Ks[64 * SK];
    __shared__ __align__(16) __half Vs[64 * SV];
    __shared__ __align__(16) float red[4][32 * 40];
    __shared__ __align__(16) __half KVt[40 * SK];
    const int h = blockIdx.x, b = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ld = h < HA ? HA * 96 : HB * 96;
    const __half* xh = (h < HA ? xa + (long long)b * HW * ld + h * 96 : xb + (long long)b * HW * ld + (h - HA) * 96);
    const int g = lane >> 2, t4 = lane & 3;
    float acc[2][5][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 5; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
    for (int p0 = 0; p0 < HW; p0 += 64) {
        for (int i = tid; i < 64 * 8; i += 128) {
            int p = i / 8, part = i % 8;
            uint4 u = make_uint4(0, 0, 0, 0);
            if (p0 + p < HW) u = __ldg(reinterpret_cast<const uint4*>(xh + (long long)(p0 + p) * ld + 32) + part);
            __half2* hv = reinterpret_cast<__half2*>(&u);
            if (part < 4) {
#pragma unroll
                for (int k = 0; k < 4; ++k) hv[k] = __hmax2(hv[k], __float2half2_rn(0.f));
                *reinterpret_cast<uint4*>(Ks + p * SK + part * 8) = u;
            } else {
                *reinterpret_cast<uint4*>(Vs + p * SV + (part - 4) * 8) = u;
            }
            if (part == 0) {   // ones column (32) for valid pixels, zeros 33..39
                __align__(16) __half o[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) o[k] = __float2half(k == 0 && p0 + p < HW ? 1.f : 0.f);
                *reinterpret_cast<uint4*>(Vs + p * SV + 32) = *reinterpret_cast<const uint4*>(o);
            }
        }
        __syncthreads();
        {
            const int kp = warp * 16;     // this warp's k16 slice of the 64 pixels
            unsigned af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int p = kp + (lane & 7) + ((lane >> 4) & 1) * 8, i = mi * 16 + ((lane >> 3) & 1) * 8;
                ldsm4t(af[mi][0], af[mi][1], af[mi][2], af[mi][3], Ks + p * SK + i);
            }
#pragma unroll
            for (int nj = 0; nj < 5; nj += 2) {
                int p = kp + (lane & 7) + ((lane >> 3) & 1) * 8, d = nj * 8 + ((lane >> 4) & 1) * 8;
                unsigned b0, b1, b2, b3;
                ldsm4t(b0, b1, b2, b3, Vs + p * SV + d);     // nj = 4: second pair reads cols 40..47 (unused)
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    mma16816f(acc[mi][nj], af[mi], b0, b1);
                    if (nj + 1 < 5) mma16816f(acc[mi][nj + 1], af[mi], b2, b3);
                }
            }
        }
        __syncthreads();
    }
    // reduce the 4 warps' partial KV, scale by 1 / HW, store transposed fp16 KVt[d][i]
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int nj = 0; nj < 5; ++nj)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                int i = mi * 16 + g + (k >> 1) * 8, d = nj * 8 + t4 * 2 + (k & 1);
                red[warp][i * 40 + d] = acc[mi][nj][k];
            }
    __syncthreads();
    const float inv = 1.f / HW;
    for (int e = tid; e < 32 * 40; e += 128) {
        int i = e / 40, d = e % 40;
        float v = (red[0][e] + red[1][e] + red[2][e] + red[3][e]) * inv;
        KVt[d * SK + i] = __float2half(v);
    }
    __syncthreads();
    // phase B
    unsigned bf[5][2][2];   // B fragments of KV (k = i 0..31, n = d), constant over pixels
#pragma unroll
    for (int nj = 0; nj < 5; ++nj)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
            ldsm2(bf[nj][kk][0], bf[nj][kk][1], KVt + (nj * 8 + (lane & 7)) * SK + kk * 16 + ((lane >> 3) & 1) * 8);
    const float epsn = eps * inv;
    for (int p0 = warp * 16; p0 < HW; p0 += 64) {
        // A = relu(q) rows p0..p0+15 straight from global into fragments (row g / g+8, cols 2t.. per k16 half)
        unsigned af[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                int p = p0 + g + (r & 1) * 8, i = kk * 16 + t4 * 2 + (r >> 1) * 8;
                __half2 v = p < HW ? *reinterpret_cast<const __half2*>(xh + (long long)p * ld + i) : __float2half2_rn(0.f);
                v = __hmax2(v, __float2half2_rn(0.f));
                af[kk][r] = h2u(v);
            }
        float o[5][4];
#pragma unroll
        for (int nj = 0; nj < 5; ++nj) {
#pragma unroll
            for (int k = 0; k < 4; ++k) o[nj][k] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) mma16816f(o[nj], af[kk], bf[nj][kk][0], bf[nj][kk][1]);
        }
        float den0 = __shfl_sync(0xffffffffu, o[4][0], lane & ~3), den1 = __shfl_sync(0xffffffffu, o[4][2], lane & ~3);
        float i0 = 1.f / (den0 + epsn), i1 = 1.f / (den1 + epsn);
        int pa = p0 + g, pb = p0 + g + 8;
        __half* ya = y + ((long long)b * HW + pa) * (HA + HB) * 32 + h * 32;
        __half* yb = y + ((long long)b * HW + pb) * (HA + HB) * 32 + h * 32;
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
            int d = nj * 8 + t4 * 2;
            if (pa < HW) *reinterpret_cast<__half2*>(ya + d) = __floats2half2_rn(o[nj][0] * i0, o[nj][1] * i0);
            if (pb < HW) *reinterpret_cast<__half2*>(yb + d) = __floats2half2_rn(o[nj][2] * i1, o[nj][3] * i1);
        }
    }
}

// xa [B, H, W, HA * 96], xb [B, H, W, HB * 96] fp16; y [B, H, W, (HA + HB) * 32]
extern "C" int fvm_linattn2(const void* xa, const void* xb, void* y, int B, int HW, int HA, int HB, float eps, cudaStream_t st) {
    linattn_tc<<<dim3(HA + HB, B), 128, 0, st>>>((const __half*)xa, (const __half*)xb, (__half*)y, HW, HA, HB, eps);
    return (int)cudaGetLastError();
}

// direct stem conv straight from the network input: x fp32 (or fp16) NCHW [N, C <= 4, H, W] -> y fp16 NHWC [N, K, P, Q]
// (replaces the fp32 -> NHWC8 conversion kernel + the implicit-GEMM stem). One thread = one output pixel x all K channels,
// half2 accumulation (as the tensor-core convs), weights [R*S*C][K] in smem (8-channel vector loads, broadcast).
template <int K, class T>
__global__ void __launch_bounds__(128) stem_direct_kernel(const T* __restrict__ x, const __half* __restrict__ w,
                                                          const __half* __restrict__ b, __half* __restrict__ y, int N, int C,
                                                          int H, int W, int P, int Q, int R, int S, int sh, int sw, int ph,
                                                          int pw, int act) {
    extern __shared__ __align__(16) __half ws[];
    const int nw = R * S * C * K;
    for (int i = threadIdx.x * 8; i < nw; i += blockDim.x * 8)
        *reinterpret_cast<uint4*>(ws + i) = __ldg(reinterpret_cast<const uint4*>(w + i));
    __syncthreads();
    long long pix = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (pix >= (long long)N * P * Q) return;
    int q = pix % Q; long long t = pix / Q; int p = t % P; int n = t / P;
    __half2 acc[K / 2];
#pragma unroll
    for (int k = 0; k < K / 2; ++k) acc[k] = __float2half2_rn(0.f);
    const long long HW = (long long)H * W;
    const T* xn = x + (long long)n * C * HW;
    for (int r = 0; r < R; ++r) {
        int ih = p * sh - ph + r;
        if (ih < 0 || ih >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int iw = q * sw - pw + s2;
            if (iw < 0 || iw >= W) continue;
            const __half* wt = ws + ((r * S + s2) * C) * K;
            for (int c = 0; c < C; ++c) {
                __half2 xv = __float2half2_rn((float)xn[c * HW + (long long)ih * W + iw]);
#pragma unroll
                for (int k8 = 0; k8 < K / 8; ++k8) {
                    uint4 wv = *reinterpret_cast<const uint4*>(wt + c * K + k8 * 8);
                    const __half2* hw = reinterpret_cast<const __half2*>(&wv);
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[k8 * 4 + j] = __hfma2(xv, hw[j], acc[k8 * 4 + j]);
                }
            }
        }
    }
    __half* yo = y + pix * K;
#pragma unroll
    for (int k8 = 0; k8 < K / 8; ++k8) {
        __align__(16) __half o[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 v = __half22float2(acc[k8 * 4 + j]);
            float b0 = b ? __half2float(b[k8 * 8 + 2 * j]) : 0.f, b1 = b ? __half2float(b[k8 * 8 + 2 * j + 1]) : 0.f;
            o[2 * j] = __float2half(actf(v.x + b0, act));
            o[2 * j + 1] = __float2half(actf(v.y + b1, act));
        }
        *reinterpret_cast<uint4*>(yo + k8 * 8) = *reinterpret_cast<const uint4*>(o);
    }
}

// x [N, C <= 4, H, W] contiguous fp32 (f32 = 1) or fp16; w_t [R, S, C, K] fp16; y [N, P, Q, K] fp16.
// Returns 1 if K is not one of 16, 24, 32, 40, 48, 64.
extern "C" int fvm_stem_direct(const void* x, int f32, const void* w_t, const void* bias, void* y, int N, int C, int H, int W,
                               int K, int R, int S, int sh, int sw, int ph, int pw, int act, cudaStream_t st) {
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    long long total = (long long)N * P * Q;
    int smem = R * S * C * K * 2;
    dim3 grid((total + 127) / 128);
    const __half* wp = (const __half*)w_t; const __half* bp = (const __half*)bias;
    __half* yp = (__half*)y;
#define STEM_K(KK)                                                                                                         \
    if (K == KK) {                                                                                                         \
        if (f32)                                                                                                           \
            stem_direct_kernel<KK, float><<<grid, 128, smem, st>>>((const float*)x, wp, bp, yp, N, C, H, W, P, Q, R, S, sh, \
                                                                   sw, ph, pw, act);                                       \
        else                                                                                                               \
            stem_direct_kernel<KK, __half><<<grid, 128, smem, st>>>((const __half*)x, wp, bp, yp, N, C, H, W, P, Q, R, S,  \
                                                                    sh, sw, ph, pw, act);                                  \
        return (int)cudaGetLastError();                                                                                    \
    }
    STEM_K(16) STEM_K(24) STEM_K(32) STEM_K(40) STEM_K(48) STEM_K(64)
#undef STEM_K
    return 1;
}
