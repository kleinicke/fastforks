// Fused convolutions and NHWC helper kernels for classification CNNs (CUTLASS 2.x implicit GEMM, sm_80-style
// mma.sync, fp16 accumulation). Ported from fast-vision-models `fused_conv.cu` (exp 008): the kernel bodies are
// unchanged; the torch host functions became `extern "C"` launchers (pointers, sizes, stream -> status), and the
// Python wrapper (`__init__.py`) does the checks, allocations and layout handling the torch host code did.
//
//   y = act(conv(x, w) + bias[k] (+ res[n, p, q, k]))        x, res, y: NHWC (= channels-last NCHW) fp16, w: KRSC fp16
//   act: 0 none, 1 relu, 2 relu6, 3 silu, 4 hardswish, 5 gelu(erf), 6 gelu(tanh)
// Requires C % 8 == 0 and K % 8 == 0 (16-byte vectors); dilation 1, groups 1.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using Hf = cutlass::half_t;
using NHWC = cutlass::layout::TensorNHWC;

struct EpiArgs {
    const __half* bias;   // [K] or null
    const __half* res;    // [M, K] or null
    __half* y;            // [M, ldy] (ldy >= K: output into a channel slice of a wider buffer)
    int act;
    int ldy;
    int ldx;              // input pixel stride (>= C: input may be a channel slice of a wider channels-last buffer)
};

__device__ __forceinline__ float act_f(float v, int act) {
    switch (act) {
        case 1: return fmaxf(v, 0.f);
        case 2: return fminf(fmaxf(v, 0.f), 6.f);
        case 3: return v / (1.f + __expf(-v));
        case 4: return v * fminf(fmaxf(v + 3.f, 0.f), 6.f) * (1.f / 6.f);
        case 5: return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
        case 6: return 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
    return v;
}

template <class OutIt>
struct Visitor {
    static int const kIterations = OutIt::kIterations;
    static int const kElementsPerAccess = OutIt::kElementsPerAccess;
    static_assert(kElementsPerAccess == 8, "8 channels per access");
    using ElementAccumulator = Hf;
    using AccumulatorFragment = cutlass::Array<Hf, kElementsPerAccess>;
    using ThreadMap = typename OutIt::ThreadMap;
    struct SharedStorage {};
    static int const kFrags = ThreadMap::Iterations::kCount;

    EpiArgs const& a;
    OutIt it;
    int M, N;
    uint4 pr[kFrags];

    CUTLASS_DEVICE Visitor(EpiArgs const& args, typename OutIt::Params const& prm, cutlass::MatrixCoord extent,
                           int thread_idx, cutlass::MatrixCoord tb_offset)
        : a(args), it(prm, nullptr, extent, thread_idx, tb_offset), M(extent.row()), N(extent.column()) {}

    CUTLASS_DEVICE void begin_epilogue() {}
    CUTLASS_DEVICE void begin_step(int) {
        if (!a.res) return;
#pragma unroll
        for (int i = 0; i < kFrags; ++i) {
            cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(i);
            long long p = co.row();
            int n = co.column();
            if (p < M && n < N) pr[i] = __ldg(reinterpret_cast<const uint4*>(a.res + p * N + n));
        }
    }
    CUTLASS_DEVICE void begin_row(int) {}
    CUTLASS_DEVICE void end_row(int) {}
    CUTLASS_DEVICE void end_step(int) { ++it; }
    CUTLASS_DEVICE void end_epilogue() {}

    CUTLASS_DEVICE void visit(int, int, int, int frag_idx, AccumulatorFragment const& acc) {
        cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(frag_idx);
        long long p = co.row();
        int n = co.column();
        if (p >= M || n >= N) return;
        const __half* av = reinterpret_cast<const __half*>(&acc);
        __align__(16) __half bv[8];
        if (a.bias) *reinterpret_cast<uint4*>(bv) = __ldg(reinterpret_cast<const uint4*>(a.bias + n));
        const __half* rv = reinterpret_cast<const __half*>(&pr[frag_idx]);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __half2float(av[k]);
            if (a.bias) v += __half2float(bv[k]);
            if (a.res) v += __half2float(rv[k]);
            o[k] = __float2half(act_f(v, a.act));
        }
        *reinterpret_cast<uint4*>(a.y + p * a.ldy + n) = *reinterpret_cast<const uint4*>(o);
    }
};

template <class TB, class WS, int Stages, int Swz,
          cutlass::conv::IteratorAlgorithm Alg = cutlass::conv::IteratorAlgorithm::kOptimized>
struct Fused {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;  // only shapes the epilogue
    using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Hf, NHWC, Hf, NHWC, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
        cutlass::arch::OpMultiplyAdd, Alg>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using V = Visitor<OutIt>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        V, typename Base::Epilogue>::Epilogue;
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
    struct Params {
        typename Base::Params base;
        EpiArgs e;
    };
};

template <class F>
__global__ void __launch_bounds__(F::Base::kThreadCount) fused_conv_kernel(typename F::Params const P) {
    using Mma = typename F::Mma;
    extern __shared__ int smem_[];
    auto& ss = *reinterpret_cast<typename F::SharedStorage*>(smem_);
    auto const& prm = P.base;
    typename F::Base::ThreadblockSwizzle sw;
    cutlass::gemm::GemmCoord tile = sw.get_tile_offset(prm.swizzle_log_tile);
    if (prm.grid_tiled_shape.m() <= tile.m() || prm.grid_tiled_shape.n() <= tile.n()) return;
    int thread_idx = threadIdx.x;
    typename Mma::IteratorA iA(prm.iterator_A, prm.problem_size, prm.ptr_A, thread_idx,
                               cutlass::MatrixCoord(tile.m() * Mma::Shape::kM, tile.k() * Mma::Shape::kK));
    typename Mma::IteratorB iB(prm.iterator_B, prm.problem_size, prm.ptr_B, thread_idx,
                               cutlass::MatrixCoord(tile.k() * Mma::Shape::kK, tile.n() * Mma::Shape::kN));
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    Mma mma(ss.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC acc;
    acc.clear();
    mma(prm.gemm_k_iterations, acc, iA, iB, acc, prm.gemm_k_iterations_per_channel);
    cutlass::MatrixCoord tb_off(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    cutlass::MatrixCoord extent(prm.problem_size.N * prm.problem_size.P * prm.problem_size.Q, prm.problem_size.K);
    typename F::V v(P.e, prm.iterator_D, extent, thread_idx, tb_off);
    typename F::Epilogue epi(ss.epilogue, thread_idx, warp_idx, lane_idx);
    epi(v, acc);
}

// cudaFuncSetAttribute is per device: remember the dynamic smem size set for each (kernel, device).
static bool smem_attr(const void* f, int smem, int* done) {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d >= 64) return false;
    if (done[d] >= smem) return true;
    if (cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return false;
    done[d] = smem;
    return true;
}

template <class F>
static int run(cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e, cudaStream_t s) {
    using Base = typename F::Base;
    long long lx = e.ldx > 0 ? e.ldx : p.C;
    NHWC xl(cutlass::make_Coord(int(lx), int(lx * p.W), int(lx * p.W * p.H)));
    typename Base::Arguments args(p, {(Hf*)x, xl}, {(Hf*)w, NHWC::packed({p.K, p.R, p.S, p.C})},
                                  {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})}, {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})},
                                  {Hf(1), Hf(0)});
    if (cutlass::conv::device::ImplicitGemmConvolution<Base>::can_implement(args) != cutlass::Status::kSuccess) return 1;
    typename F::Params P{typename Base::Params(args, nullptr), e};
    typename Base::ThreadblockSwizzle sw;
    dim3 grid = sw.get_grid_shape(P.base.grid_tiled_shape);
    int smem = int(sizeof(typename F::SharedStorage));
    static int attr[64] = {};
    if (!smem_attr((const void*)fused_conv_kernel<F>, smem, attr)) return 2;
    fused_conv_kernel<F><<<grid, Base::kThreadCount, smem, s>>>(P);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <int M, int N, int K>
using GS = cutlass::gemm::GemmShape<M, N, K>;

static int dispatch(int cfg, cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e,
                    cudaStream_t s) {
    switch (cfg) {
        case 0: return run<Fused<GS<128, 128, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 1: return run<Fused<GS<128, 64, 32>, GS<64, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 2: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 5, 1>>(p, x, w, e, s);
        case 3: return run<Fused<GS<64, 128, 32>, GS<32, 64, 32>, 4, 4>>(p, x, w, e, s);
        case 4: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 4, 4>>(p, x, w, e, s);
        case 5: return run<Fused<GS<128, 256, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 6: return run<Fused<GS<256, 128, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 7: return run<Fused<GS<64, 128, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 8: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 9: return run<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        // stem (C == 8, zero-padded RGB): fixed-channels iterator, no wasted K
        case 10: return run<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4, cutlass::conv::IteratorAlgorithm::kFixedChannels>>(p, x, w, e, s);
        case 11: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4, cutlass::conv::IteratorAlgorithm::kFixedChannels>>(p, x, w, e, s);
        case 12: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 4, 1, cutlass::conv::IteratorAlgorithm::kFixedChannels>>(p, x, w, e, s);
        // narrow-K (output channel) tiles for mobile nets (K = 16 .. 160)
        case 13: return run<Fused<GS<128, 32, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 14: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 3, 4>>(p, x, w, e, s);
        case 15: return run<Fused<GS<256, 32, 32>, GS<64, 32, 32>, 3, 4>>(p, x, w, e, s);
        case 16: return run<Fused<GS<128, 32, 64>, GS<32, 32, 64>, 3, 4>>(p, x, w, e, s);
        case 17: return run<Fused<GS<128, 96, 32>, GS<32, 32, 32>, 3, 4>>(p, x, w, e, s);
    }
    return -1;
}

// fixed rule (no timing; from sweeps of ResNet-50 / MobileNet V2 / EfficientNet-B0 convs at batch 1 / 8 / 64): 128x128
// tiles (8 warps) when K fits 128-wide tiles and they fill the GPU, else 128x64 (8 warps), 64x64 when there are too few
// CTAs (batch 1). cfg 17 (128x96, 12 warps) computes garbage (NaN) -> never picked.
static int pick(long long M, int K, int C, int SMS) {
    auto tiles = [&](int tm, int tn) { return ((M + tm - 1) / tm) * (long long)((K + tn - 1) / tn); };
    auto fits = [&](int tn) { return ((K + tn - 1) / tn) * tn <= 1.1 * K; };   // <= 10 % padded output channels
    if (C == 8) return tiles(128, 64) >= SMS ? 10 : 12;   // 128x128 fixed-channel tiles (cfg 11) measured slower everywhere
    if (fits(128)) {
        if (tiles(128, 128) >= SMS) return 8;
        if (tiles(64, 128) >= SMS) return 7;
        return 2;
    }
    if (tiles(128, 64) >= SMS) return 9;     // also for K not a multiple of 64 (mobile nets): ~= 64x64 3-stage, >> 5-stage
    return 2;
}

// y = act(x * scale[c] + shift[c]) for an NHWC fp16 tensor that may be a channel slice (pixel stride ldx); packed output.
// (pre-activation BatchNorm + ReLU of DenseNet / unfoldable BNs)
__global__ void affine_act_k(const __half* __restrict__ x, const float* __restrict__ sc, const float* __restrict__ sh,
                             __half* __restrict__ y, long long npix, int C8, long long ldx, int act) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= npix * C8) return;
    int c = i % C8; long long p = i / C8;
    uint4 v = __ldg(reinterpret_cast<const uint4*>(x + p * ldx) + c);
    const __half* h = reinterpret_cast<const __half*>(&v);
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(act_f(__half2float(h[k]) * sc[c * 8 + k] + sh[c * 8 + k], act));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

// max pool, NHWC fp16, 8 channels per thread (C % 8 == 0), -inf padding (PyTorch semantics, no ceil_mode / dilation)
__global__ void maxpool_nhwc8(const __half* __restrict__ x, __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q,
                              int R, int S, int sh, int sw, int ph, int pw) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    __half2 m[4];
    for (int k = 0; k < 4; ++k) m[k] = __half2(__ushort_as_half(0xfc00), __ushort_as_half(0xfc00));
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int w = q * sw - pw + s2; if (w < 0 || w >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (((long long)n * H + h) * W + w) * C8 * 8) + c);
            const __half2* hv = reinterpret_cast<const __half2*>(&v);
            for (int k = 0; k < 4; ++k) m[k] = __hmax2(m[k], hv[k]);
        }
    }
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(m);
}

// register-tiled variant: one thread = 8 channels x TW consecutive output columns; weights and the input row segment are
// loaded once per thread and reused across the TW outputs (fp32 accumulation). R, S <= 7.
template <int TW>
__global__ void dwconv_tw(const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
                          __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q, int R, int S, int sh, int sw,
                          int ph, int pw, int act) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int QT = (Q + TW - 1) / TW;
    long long total = (long long)N * P * QT * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int qt = t % QT; t /= QT; int p = t % P; int n = t / P;
    int q0 = qt * TW;
    float acc[TW][8];
    float bb[8];
    if (b) {
        uint4 bv = __ldg(reinterpret_cast<const uint4*>(b) + c);
        const __half* hb = reinterpret_cast<const __half*>(&bv);
#pragma unroll
        for (int k = 0; k < 8; ++k) bb[k] = __half2float(hb[k]);
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) bb[k] = 0.f;
    }
#pragma unroll
    for (int j = 0; j < TW; ++j)
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[j][k] = bb[k];
    const int L = (TW - 1) * sw + S;          // input columns touched by the TW outputs
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        const __half* xrow = x + ((long long)n * H + h) * W * C8 * 8;
        for (int l = 0; l < L; ++l) {
            int ww = q0 * sw - pw + l; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(xrow + (long long)ww * C8 * 8) + c);
            float xv[8];
            const __half2* hv = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int k = 0; k < 4; ++k) { float2 f = __half22float2(hv[k]); xv[2 * k] = f.x; xv[2 * k + 1] = f.y; }
#pragma unroll
            for (int j = 0; j < TW; ++j) {
                int s2 = l - j * sw;                       // tap index of input column l for output j
                if (s2 < 0 || s2 >= S) continue;
                uint4 g = __ldg(reinterpret_cast<const uint4*>(w + ((long long)r * S + s2) * C8 * 8) + c);
                const __half2* hg = reinterpret_cast<const __half2*>(&g);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float2 f = __half22float2(hg[k]);
                    acc[j][2 * k] += xv[2 * k] * f.x; acc[j][2 * k + 1] += xv[2 * k + 1] * f.y;
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TW; ++j) {
        int q = q0 + j; if (q >= Q) break;
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __float2half(act_f(acc[j][k], act));
        reinterpret_cast<uint4*>(y + (((long long)n * P + p) * Q + q) * C8 * 8)[c] = *reinterpret_cast<uint4*>(o);
    }
}

// direct conv for the zero-padded RGB stem (C == 8) with few output channels (K <= 64): one thread = 8 output channels of
// one pixel; weights [K/8][R][S][8 in][8 out] in shared memory (implicit GEMM wastes most of a 64-wide tile at K = 16 / 32)
__global__ void stem_c8_k(const __half* __restrict__ x, const __half* __restrict__ wt, const __half* __restrict__ b,
                          __half* __restrict__ y, int N, int H, int W, int K8, int P, int Q, int R, int S, int sh, int sw,
                          int ph, int pw, int act) {
    extern __shared__ __half ws[];
    int nw = K8 * R * S * 64;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) ws[i] = wt[i];
    __syncthreads();
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * K8;
    if (i >= total) return;
    int kg = i % K8; long long t = i / K8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = b ? __half2float(b[kg * 8 + k]) : 0.f;
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int ww = q * sw - pw + s2; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (((long long)n * H + h) * W + ww) * 8));
            const __half* hv = reinterpret_cast<const __half*>(&v);
            const __half* wp = ws + ((kg * R + r) * S + s2) * 64;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float xv = __half2float(hv[c]);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] += xv * __half2float(wp[c * 8 + k]);
            }
        }
    }
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(act_f(acc[k], act));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

// avg pool NHWC fp16 (count_include_pad as PyTorch), 8 channels per thread
__global__ void avgpool_nhwc8(const __half* __restrict__ x, __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q,
                              int R, int S, int sh, int sw, int ph, int pw, bool cip) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int cnt = 0;
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int ww = q * sw - pw + s2; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (((long long)n * H + h) * W + ww) * C8 * 8) + c);
            const __half* hv = reinterpret_cast<const __half*>(&v);
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[k] += __half2float(hv[k]);
            ++cnt;
        }
    }
    int h0 = p * sh - ph, w0 = q * sw - pw;
    int hend = min(h0 + R, H + ph), wend = min(w0 + S, W + pw);
    float div = cip ? (float)((hend - h0) * (wend - w0)) : (float)max(cnt, 1);
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(acc[k] / div);
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

// network input: fp32 / fp16 NCHW (any strides via contiguous) -> fp16 NHWC with channels zero-padded to 8 (stem conv C 3 -> 8)
template <class T>
__global__ void input_nhwc8_k(const T* __restrict__ x, __half* __restrict__ y, int C, long long HW, long long total) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;   // one output pixel (8 channels) per thread
    if (i >= total) return;
    long long n = i / HW, hw = i % HW;
    __align__(16) __half o[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) o[c] = c < C ? __float2half((float)x[(n * C + c) * HW + hw]) : __float2half(0.f);
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}
// ---------------------------------------------------------------------------------------------------------------------
// C API. Pointers are device pointers, `st` a cudaStream_t; every launcher returns 0 on success. fvm_conv_fused also
// returns 1 if CUTLASS cannot implement the geometry, 2 if the shared-memory attribute cannot be set, 3 on a launch
// error and -1 for an unknown cfg; the other launchers return the cudaError_t of the launch.

static int sm_count() {
    static int sms[64] = {};
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d >= 64) return 36;
    if (!sms[d]) {
        int n = 36;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
        sms[d] = n;
    }
    return sms[d];
}

// x [N, H, W, ldx] (C channels used, ldx 0 = C), w [K, R, S, C], bias [K] / res [N, P, Q, K] or null, y [N, P, Q, ldy]
// (ldy 0 = K); cfg < 0 picks the tile config by the fixed rule
extern "C" int fvm_conv_fused(int cfg, const void* x, const void* w, const void* bias, const void* res, void* y, int N, int H, int W,
                   int C, int K, int R, int S, int ph, int pw, int sh, int sw, int act, int ldx, int ldy, cudaStream_t st) {
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    cutlass::conv::Conv2dProblemSize p({N, H, W, C}, {K, R, S, C}, {ph, ph, pw, pw}, {sh, sw}, {1, 1}, {N, P, Q, K},
                                       cutlass::conv::Mode::kCrossCorrelation, 1);
    EpiArgs e{(const __half*)bias, (const __half*)res, (__half*)y, act, ldy > 0 ? ldy : K, ldx};
    if (cfg < 0) cfg = pick((long long)N * P * Q, K, C, sm_count());
    return dispatch(cfg, p, x, w, e, st);
}

// x [N, H, W, 8] fp16, wt [K/8][R][S][8 in][8 out], bias [K] or null, y [N, P, Q, K]
extern "C" int fvm_stem_c8(const void* x, const void* wt, const void* bias, void* y, int N, int H, int W, int K, int R, int S, int ph,
                int pw, int sh, int sw, int act, cudaStream_t st) {
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    long long total = (long long)N * P * Q * (K / 8);
    int smem = K * R * S * 8 * 2;
    stem_c8_k<<<(total + 127) / 128, 128, smem, st>>>((const __half*)x, (const __half*)wt, (const __half*)bias, (__half*)y,
                                                       N, H, W, K / 8, P, Q, R, S, sh, sw, ph, pw, act);
    return (int)cudaGetLastError();
}

// x [npix, ldx] fp16 (C channels used), scale / shift [C] fp32, y [npix, C]
extern "C" int fvm_affine_act(const void* x, const void* scale, const void* shift, void* y, long long npix, int C, long long ldx, int act,
                   cudaStream_t st) {
    long long total = npix * (C / 8);
    affine_act_k<<<(total + 255) / 256, 256, 0, st>>>((const __half*)x, (const float*)scale, (const float*)shift, (__half*)y,
                                                       npix, C / 8, ldx, act);
    return (int)cudaGetLastError();
}

// x [N, H, W, C] fp16, y [N, P, Q, C] (P, Q from the caller: ceil_mode is an output-size rule only)
extern "C" int fvm_maxpool(const void* x, void* y, int N, int H, int W, int C, int P, int Q, int R, int S, int sh, int sw, int ph, int pw,
                cudaStream_t st) {
    long long total = (long long)N * P * Q * (C / 8);
    maxpool_nhwc8<<<(total + 255) / 256, 256, 0, st>>>((const __half*)x, (__half*)y, N, H, W, C / 8, P, Q, R, S, sh, sw, ph, pw);
    return (int)cudaGetLastError();
}

// depthwise: x [N, H, W, C] fp16, w [R, S, C], bias [C] or null, y [N, P, Q, C]
extern "C" int fvm_dwconv(const void* x, const void* w, const void* bias, void* y, int N, int H, int W, int C, int R, int S, int sh, int sw,
               int ph, int pw, int act, cudaStream_t st) {
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    long long total = (long long)N * P * ((Q + 1) / 2) * (C / 8);
    dwconv_tw<2><<<(total + 127) / 128, 128, 0, st>>>((const __half*)x, (const __half*)w, (const __half*)bias, (__half*)y, N, H,
                                                       W, C / 8, P, Q, R, S, sh, sw, ph, pw, act);
    return (int)cudaGetLastError();
}

// x [N, H, W, C] fp16, y [N, P, Q, C]; cip = count_include_pad
extern "C" int fvm_avgpool(const void* x, void* y, int N, int H, int W, int C, int R, int S, int sh, int sw, int ph, int pw, int cip,
                cudaStream_t st) {
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    long long total = (long long)N * P * Q * (C / 8);
    avgpool_nhwc8<<<(total + 255) / 256, 256, 0, st>>>((const __half*)x, (__half*)y, N, H, W, C / 8, P, Q, R, S, sh, sw, ph, pw,
                                                       cip != 0);
    return (int)cudaGetLastError();
}

// x [N, C <= 8, H, W] contiguous fp32 (f32 = 1) or fp16 -> y [N, H, W, 8] fp16, channels zero-padded
extern "C" int fvm_input_nhwc8(const void* x, void* y, int f32, int N, int C, long long HW, cudaStream_t st) {
    long long total = (long long)N * HW;
    if (f32)
        input_nhwc8_k<float><<<(total + 255) / 256, 256, 0, st>>>((const float*)x, (__half*)y, C, HW, total);
    else
        input_nhwc8_k<__half><<<(total + 255) / 256, 256, 0, st>>>((const __half*)x, (__half*)y, C, HW, total);
    return (int)cudaGetLastError();
}

