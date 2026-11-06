// Fused convolutions for the segmentation CNNs (mmseg ResNet-D8 / HRNet; CUTLASS 2.x implicit GEMM, sm_80-style
// mma.sync, fp16 accumulation). The conv of fvm_conv.cu (fast-vision-models fused_conv.cu) with the dilation our
// segmentation fork added (exp 009 segx_conv.cu, conv_fused_d: DeepLab / PSP / FCN D8 backbones) and the BatchNorm as
// an fp32 per-channel affine in the epilogue (the fork folded it into fp16 weights / bias):
//
//   y = act(conv(x, w) * s[k] + t[k] (+ res[n, p, q, k]))     x, res, y: NHWC (= channels-last NCHW) fp16, w: KRSC fp16
//   s, t: fp32 [K] or null (1 / 0); act: 0 none, 1 relu; the sum in fp32, one rounding to fp16
// Requires C % 8 == 0 and K % 8 == 0 (16-byte vectors), groups 1.
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
    const float* s;       // [K] or null
    const float* t;       // [K] or null
    const __half* res;    // [M, K] or null
    __half* y;            // [M, ldy]
    int act;
    int ldy;
};

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
    CUTLASS_DEVICE void begin_step(int) {  // residual loads issued before the accumulators arrive (fvm_conv)
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
        __align__(16) float sv[8], tv[8];
        if (a.s) {
            *reinterpret_cast<float4*>(sv) = __ldg(reinterpret_cast<const float4*>(a.s + n));
            *reinterpret_cast<float4*>(sv + 4) = __ldg(reinterpret_cast<const float4*>(a.s + n + 4));
        }
        if (a.t) {
            *reinterpret_cast<float4*>(tv) = __ldg(reinterpret_cast<const float4*>(a.t + n));
            *reinterpret_cast<float4*>(tv + 4) = __ldg(reinterpret_cast<const float4*>(a.t + n + 4));
        }
        const __half* rv = reinterpret_cast<const __half*>(&pr[frag_idx]);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __half2float(av[k]);
            if (a.s) v *= sv[k];
            if (a.t) v += tv[k];
            if (a.res) v += __half2float(rv[k]);
            if (a.act == 1) v = fmaxf(v, 0.f);
            o[k] = __float2half(v);
        }
        *reinterpret_cast<uint4*>(a.y + p * a.ldy + n) = *reinterpret_cast<const uint4*>(o);
    }
};

template <class TB, class WS, int Stages, int Swz>
struct Fused {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;  // only shapes the epilogue
    using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Hf, NHWC, Hf, NHWC, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
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
    typename Base::Arguments args(p, {(Hf*)x, NHWC::packed({p.N, p.H, p.W, p.C})}, {(Hf*)w, NHWC::packed({p.K, p.R, p.S, p.C})},
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

// the tile configs the fixed rule picks (fvm_conv cfg 2 / 7 / 8 / 9, same numbering)
static int dispatch(int cfg, cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e,
                    cudaStream_t s) {
    switch (cfg) {
        case 2: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 5, 1>>(p, x, w, e, s);
        case 7: return run<Fused<GS<64, 128, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 8: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 9: return run<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
    }
    return -1;
}

// fixed rule of fvm_conv / the fork (no timing; from sweeps of ResNet-50 / MobileNet V2 / EfficientNet-B0 convs at
// batch 1 / 8 / 64): 128x128 tiles (8 warps) when K fits 128-wide tiles and they fill the GPU, else 128x64 (8 warps),
// 64x64 when there are too few CTAs
static int pick(long long M, int K, int SMS) {
    auto tiles = [&](int tm, int tn) { return ((M + tm - 1) / tm) * (long long)((K + tn - 1) / tn); };
    auto fits = [&](int tn) { return ((K + tn - 1) / tn) * tn <= 1.1 * K; };   // <= 10 % padded output channels
    if (fits(128)) {
        if (tiles(128, 128) >= SMS) return 8;
        if (tiles(64, 128) >= SMS) return 7;
        return 2;
    }
    if (tiles(128, 64) >= SMS) return 9;
    return 2;
}

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

// ---------------------------------------------------------------------------------------------------------------------
// C API. Pointers are device pointers, `st` a cudaStream_t. Returns 0 on success, 1 if CUTLASS cannot implement the
// geometry, 2 if the shared-memory attribute cannot be set, 3 on a launch error, -1 for an unknown cfg.
// x [N, H, W, C], w [K, R, S, C], s / t fp32 [K] / res [N, P, Q, K] or null, y [N, P, Q, ldy] (ldy 0 = K);
// cfg < 0 picks the tile config by the fixed rule
extern "C" int seg_conv(int cfg, const void* x, const void* w, const void* s, const void* t, const void* res, void* y,
                        int N, int H, int W, int C, int K, int R, int S, int ph, int pw, int sh, int sw, int dh, int dw,
                        int act, int ldy, cudaStream_t st) {
    int P = (H + 2 * ph - dh * (R - 1) - 1) / sh + 1, Q = (W + 2 * pw - dw * (S - 1) - 1) / sw + 1;
    cutlass::conv::Conv2dProblemSize p({N, H, W, C}, {K, R, S, C}, {ph, ph, pw, pw}, {sh, sw}, {dh, dw}, {N, P, Q, K},
                                       cutlass::conv::Mode::kCrossCorrelation, 1);
    EpiArgs e{(const float*)s, (const float*)t, (const __half*)res, (__half*)y, act, ldy > 0 ? ldy : K};
    if (cfg < 0) cfg = pick((long long)N * P * Q, K, sm_count());
    return dispatch(cfg, p, x, w, e, st);
}

// ---------------------------------------------------------------------------------------------------------------------
// Depthwise conv (groups = C, any taps / stride / dilation): y = act(dwconv(x, w) * s + t), fp32 accumulation, one
// fp16 rounding. x [N, H, W, C], w [R, S, C], y [N, P, Q, C], C % 8 == 0 (one thread: 8 channels of one output pixel,
// 16-byte loads); s / t fp32 [C] or null. Memory-bound: cuDNN's channels-last fp16 depthwise kernels are 4-5x slower
// than NCHW on sm_120 for the dilated 3x3 of DeepLabV3+ (exp 009), this kernel reads each input once per tap.
static __global__ void dw_kernel(const __half* __restrict__ x, const __half* __restrict__ w, const float* __restrict__ s,
                          const float* __restrict__ t, __half* __restrict__ y, int N, int H, int W, int C, int R, int S,
                          int ph, int pw, int sh, int sw, int dh, int dw, int P, int Q, int act) {
    int c8n = C / 8;
    long long n = (long long)N * P * Q * c8n;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int c8 = (int)(i % c8n);
        long long pix = i / c8n;
        int q = (int)(pix % Q);
        long long tq = pix / Q;
        int p = (int)(tq % P);
        int b = (int)(tq / P);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int r = 0; r < R; ++r) {
            int hi = p * sh - ph + r * dh;
            if (hi < 0 || hi >= H) continue;
            for (int k = 0; k < S; ++k) {
                int wi = q * sw - pw + k * dw;
                if (wi < 0 || wi >= W) continue;
                uint4 xv = __ldg((const uint4*)(x + (((long long)b * H + hi) * W + wi) * C) + c8);
                uint4 wv = __ldg((const uint4*)(w + (long long)(r * S + k) * C) + c8);
                const __half2* xh = (const __half2*)&xv;
                const __half2* wh = (const __half2*)&wv;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 a = __half22float2(xh[j]), c = __half22float2(wh[j]);
                    acc[2 * j] = fmaf(a.x, c.x, acc[2 * j]);
                    acc[2 * j + 1] = fmaf(a.y, c.y, acc[2 * j + 1]);
                }
            }
        }
        uint4 out;
        __half2* oh = (__half2*)&out;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                int c = c8 * 8 + 2 * j + u;
                float a = acc[2 * j + u];
                if (s) a *= s[c];
                if (t) a += t[c];
                if (act == 1) a = fmaxf(a, 0.f);
                v[u] = a;
            }
            oh[j] = __floats2half2_rn(v[0], v[1]);
        }
        ((uint4*)y)[i] = out;
    }
}

// Returns 0 on success, 1 for an unsupported geometry (C % 8), 3 on a launch error.
extern "C" int seg_dwconv(const void* x, const void* w, const void* s, const void* t, void* y, int N, int H, int W,
                          int C, int R, int S, int ph, int pw, int sh, int sw, int dh, int dw, int act, cudaStream_t st) {
    if (C % 8) return 1;
    int P = (H + 2 * ph - dh * (R - 1) - 1) / sh + 1, Q = (W + 2 * pw - dw * (S - 1) - 1) / sw + 1;
    long long n = (long long)N * P * Q * (C / 8);
    if (n <= 0) return 0;
    long long blocks = (n + 255) / 256;
    if (blocks > 65535LL * 8) blocks = 65535LL * 8;
    dw_kernel<<<(unsigned)blocks, 256, 0, st>>>((const __half*)x, (const __half*)w, (const float*)s, (const float*)t,
                                               (__half*)y, N, H, W, C, R, S, ph, pw, sh, sw, dh, dw, P, Q, act);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
