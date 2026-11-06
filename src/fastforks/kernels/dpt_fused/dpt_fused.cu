// Fused fp16 GEMMs / convolutions for DINOv2 ViT + DPT head networks (Depth Anything V2), ported from our mono-depth fork
// (exp 004 attempt/trtgap mfs_fused.cu / mono_kernels.cu) to a plain C API.
// Build needs CUTLASS v4.7.1 (cb42473) headers (tools/build_kernels.py --cutlass).
//
// CUTLASS 2.x threadblock GEMM / implicit-GEMM fprop (sm_80-style mma.sync, fp16 accumulation), compiled once: the
// problem size is a runtime argument, nothing is compiled or tuned per size. The caller picks a tile configuration by a
// fixed rule; a few configurations accumulate in fp32 instead (precision-critical layers). A coordinate-aware visitor epilogue finishes each 8-channel accumulator vector from registers:
//
//   v = act((acc + bias[n]) * scale[n] + res[p, n] + res2[p, n])      (every term optional)
//   y[p, n] = v;  y2[p, n] = relu(v) if y2                            (row strides ldy / ldy2 / lres / lres2)
//   act: 0 none, 1 relu, 5 gelu (erf)
//   shuffle (sh > 0): deconvolution with kernel == stride as a GEMM: row p = (b, h, w) of a [B, H, W] grid, column
//   n = (a, c, k) with a < sh, c < sw, k < cout -> y[((b * H + h) * sh + a) * (W * sw) + w * sw + c, k] (NHWC, ldy)
//
// y may alias res / res2 (in-place residual): each thread loads the residual vectors of a step before writing them.
// GEMM: x [M, K] (row stride lda), w [N, K] row-major. Conv: x NHWC [N, H, W, C], w [K, R, S, C], output NHWC.
// Requirements (checked, status 1 otherwise): C, K, lda, row strides % 8 == 0, 16-byte aligned pointers.
// Status: 0 ran, 1 geometry not supported, 2 / 3 CUDA error, -1 unknown configuration.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using Hf = cutlass::half_t;
using NHWC = cutlass::layout::TensorNHWC;

struct EpiArgs {
    const __half* bias;  // [N] or null
    const float* scale;  // [N] or null (applied to acc + bias, before the residuals)
    const __half* res;   // [M, lres] or null
    int lres;
    const __half* res2;  // [M, lres2] or null
    int lres2;
    __half* y;
    int ldy;
    __half* y2;  // relu(v) or null
    int ldy2;
    int act;
    int sh, sw, gh, gw, cout;  // pixel shuffle (sh > 0): sh x sw blocks of a gh x gw grid, cout channels
};

__device__ __forceinline__ float act_f(float v, int act) {
    switch (act) {
        case 1: return fmaxf(v, 0.f);
        case 5: return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    }
    return v;
}

static bool al16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

static bool epi_ok(EpiArgs const& e, int N) {
    if (!e.y || !al16(e.y) || !al16(e.bias) || !al16(e.scale) || !al16(e.res) || !al16(e.res2) || !al16(e.y2)) return false;
    if (N % 8 || e.ldy % 8 || (e.res && e.lres % 8) || (e.res2 && e.lres2 % 8) || (e.y2 && e.ldy2 % 8)) return false;
    if (e.sh > 0 && (e.cout % 8 || N != e.sh * e.sw * e.cout || e.res || e.res2 || e.y2)) return false;
    return e.sh > 0 || e.ldy >= N;
}

template <class OutIt, class Acc>
struct Visitor {
    static int const kIterations = OutIt::kIterations;
    static int const kElementsPerAccess = OutIt::kElementsPerAccess;
    static_assert(kElementsPerAccess == 8, "8 channels per access");
    using ElementAccumulator = Acc;
    using AccumulatorFragment = cutlass::Array<Acc, kElementsPerAccess>;
    using ThreadMap = typename OutIt::ThreadMap;
    struct SharedStorage {};
    static int const kFrags = ThreadMap::Iterations::kCount;

    EpiArgs const& a;
    OutIt it;
    int M, N;
    uint4 pr[kFrags], pr2[kFrags];

    CUTLASS_DEVICE Visitor(EpiArgs const& args, typename OutIt::Params const& prm, cutlass::MatrixCoord extent,
                           int thread_idx, cutlass::MatrixCoord tb_offset)
        : a(args), it(prm, nullptr, extent, thread_idx, tb_offset), M(extent.row()), N(extent.column()) {}

    CUTLASS_DEVICE void begin_epilogue() {}
    CUTLASS_DEVICE void begin_step(int) {
        if (!a.res && !a.res2) return;
#pragma unroll
        for (int i = 0; i < kFrags; ++i) {
            cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(i);
            long long p = co.row();
            int n = co.column();
            if (p < M && n < N) {
                if (a.res) pr[i] = *reinterpret_cast<const uint4*>(a.res + p * a.lres + n);
                if (a.res2) pr2[i] = *reinterpret_cast<const uint4*>(a.res2 + p * a.lres2 + n);
            }
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
        __align__(16) __half bv[8];
        if (a.bias) *reinterpret_cast<uint4*>(bv) = __ldg(reinterpret_cast<const uint4*>(a.bias + n));
        float sv[8];
        if (a.scale) {
            float4 s0 = __ldg(reinterpret_cast<const float4*>(a.scale + n));
            float4 s1 = __ldg(reinterpret_cast<const float4*>(a.scale + n) + 1);
            sv[0] = s0.x; sv[1] = s0.y; sv[2] = s0.z; sv[3] = s0.w; sv[4] = s1.x; sv[5] = s1.y; sv[6] = s1.z; sv[7] = s1.w;
        }
        const __half* rv = reinterpret_cast<const __half*>(&pr[frag_idx]);
        const __half* rv2 = reinterpret_cast<const __half*>(&pr2[frag_idx]);
        __align__(16) __half o[8];
        __align__(16) __half o2[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = float(acc[k]);
            if (a.bias) v += __half2float(bv[k]);
            if (a.scale) v *= sv[k];
            if (a.res) v += __half2float(rv[k]);
            if (a.res2) v += __half2float(rv2[k]);
            o[k] = __float2half(act_f(v, a.act));
            o2[k] = __hmax(o[k], __float2half(0.f));
        }
        long long q = p;
        int c = n;
        if (a.sh > 0) {  // deconvolution pixel shuffle
            int g = a.gh * a.gw;
            int b = int(p / g), r = int(p - (long long)b * g);
            int h = r / a.gw, w = r - h * a.gw;
            int blk = n / a.cout;
            c = n - blk * a.cout;
            int i = blk / a.sw, j = blk - i * a.sw;
            q = ((long long)(b * a.gh + h) * a.sh + i) * (a.gw * a.sw) + w * a.sw + j;
        }
        *reinterpret_cast<uint4*>(a.y + q * a.ldy + c) = *reinterpret_cast<const uint4*>(o);
        if (a.y2) *reinterpret_cast<uint4*>(a.y2 + p * a.ldy2 + n) = *reinterpret_cast<const uint4*>(o2);
    }
};

template <int M, int N, int K>
using GS = cutlass::gemm::GemmShape<M, N, K>;

// ---------------------------------------------------------------- implicit-GEMM convolution (NHWC, groups 1, dilation 1)
template <class TB, class WS, int Stages, int Swz, class Acc = Hf>
struct Fused {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Acc, Acc>;  // only shapes the epilogue
    using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Hf, NHWC, Hf, NHWC, Hf, NHWC, Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using V = Visitor<OutIt, Acc>;
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
static int run_conv(cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e,
                    cudaStream_t s) {
    using Base = typename F::Base;
    typename Base::Arguments args(p, {(Hf*)x, NHWC::packed({p.N, p.H, p.W, p.C})},
                                  {(Hf*)w, NHWC::packed({p.K, p.R, p.S, p.C})},
                                  {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})},
                                  {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})}, {Hf(1), Hf(0)});
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

// g: N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w; output N x P x Q x K through the epilogue
extern "C" int dpt_conv(int cfg, const void* x, const void* w, const int* g, const EpiArgs* e, void* stream) {
    int N = g[0], H = g[1], W = g[2], C = g[3], K = g[4], R = g[5], S = g[6], ph = g[7], pw = g[8], sh = g[9], sw = g[10];
    if (C % 8 || K % 8 || !al16(x) || !al16(w) || e->sh > 0 || !epi_ok(*e, K)) return 1;
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    cutlass::conv::Conv2dProblemSize p({N, H, W, C}, {K, R, S, C}, {ph, ph, pw, pw}, {sh, sw}, {1, 1}, {N, P, Q, K},
                                       cutlass::conv::Mode::kCrossCorrelation, 1);
    cudaStream_t s = (cudaStream_t)stream;
    switch (cfg) {
        case 0: return run_conv<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4>>(p, x, w, *e, s);
        case 1: return run_conv<Fused<GS<64, 128, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, *e, s);
        case 2: return run_conv<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, *e, s);
        case 3: return run_conv<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 5, 1>>(p, x, w, *e, s);
        case 4: return run_conv<Fused<GS<128, 32, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, *e, s);
        case 5: return run_conv<Fused<GS<128, 128, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, *e, s);
        // fp32 accumulation (precision-critical layers)
        case 6: return run_conv<Fused<GS<128, 32, 32>, GS<32, 32, 32>, 4, 4, float>>(p, x, w, *e, s);
        case 7: return run_conv<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 4, 1, float>>(p, x, w, *e, s);
    }
    return -1;
}

// ---------------------------------------------------------------- GEMM y[M, N] = epilogue(x[M, K] @ w[N, K]^T)
template <class TB, class WS, int Stages, int Swz, class Acc = Hf>
struct FusedGemm {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Acc, Acc>;
    using Base = typename cutlass::gemm::kernel::DefaultGemm<
        Hf, cutlass::layout::RowMajor, 8, Hf, cutlass::layout::ColumnMajor, 8, Hf, cutlass::layout::RowMajor, Acc,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS, cutlass::gemm::GemmShape<16, 8, 16>, EpiOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages, false,
        cutlass::arch::OpMultiplyAdd>::GemmKernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using V = Visitor<OutIt, Acc>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        V, typename Base::Epilogue>::Epilogue;
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
    struct Params {
        cutlass::gemm::GemmCoord problem, grid_tiled;
        int swizzle_log_tile, gemm_k_iterations;
        typename Mma::IteratorA::Params pA;
        typename Mma::IteratorB::Params pB;
        typename OutIt::Params pD;
        const Hf* A;
        const Hf* B;
        EpiArgs e;
    };
};

template <class F>
__global__ void __launch_bounds__(F::Mma::WarpCount::kCount * 32) fused_gemm_kernel(typename F::Params const P) {
    using Mma = typename F::Mma;
    extern __shared__ int smem_[];
    auto& ss = *reinterpret_cast<typename F::SharedStorage*>(smem_);
    typename F::Base::ThreadblockSwizzle sw;
    cutlass::gemm::GemmCoord tile = sw.get_tile_offset(P.swizzle_log_tile);
    if (P.grid_tiled.m() <= tile.m() || P.grid_tiled.n() <= tile.n()) return;
    int thread_idx = threadIdx.x;
    cutlass::MatrixCoord tbA(tile.m() * Mma::Shape::kM, 0), tbB(0, tile.n() * Mma::Shape::kN);
    typename Mma::IteratorA iA(P.pA, const_cast<Hf*>(P.A), {P.problem.m(), P.problem.k()}, thread_idx, tbA);
    typename Mma::IteratorB iB(P.pB, const_cast<Hf*>(P.B), {P.problem.k(), P.problem.n()}, thread_idx, tbB);
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    Mma mma(ss.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC acc;
    acc.clear();
    mma(P.gemm_k_iterations, acc, iA, iB, acc);
    cutlass::MatrixCoord tb_off(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    typename F::V v(P.e, P.pD, cutlass::MatrixCoord(P.problem.m(), P.problem.n()), thread_idx, tb_off);
    typename F::Epilogue epi(ss.epilogue, thread_idx, warp_idx, lane_idx);
    epi(v, acc);
}

template <class F>
static int run_gemm(int M, int N, int K, const void* x, int lda, const void* w, EpiArgs const& e, cudaStream_t st) {
    using Mma = typename F::Mma;
    typename F::Params P;
    P.problem = cutlass::gemm::GemmCoord(M, N, K);
    typename F::Base::ThreadblockSwizzle sw;
    P.grid_tiled = sw.get_tiled_shape(P.problem, {Mma::Shape::kM, Mma::Shape::kN, Mma::Shape::kK}, 1);
    P.swizzle_log_tile = sw.get_log_tile(P.grid_tiled);
    P.gemm_k_iterations = (K + Mma::Shape::kK - 1) / Mma::Shape::kK;
    P.pA = typename Mma::IteratorA::Params(cutlass::layout::RowMajor(lda));
    P.pB = typename Mma::IteratorB::Params(cutlass::layout::ColumnMajor(K));
    P.pD = typename F::OutIt::Params(cutlass::layout::RowMajor(N));
    P.A = (const Hf*)x;
    P.B = (const Hf*)w;
    P.e = e;
    dim3 grid = sw.get_grid_shape(P.grid_tiled);
    int smem = int(sizeof(typename F::SharedStorage));
    static int attr[64] = {};
    if (!smem_attr((const void*)fused_gemm_kernel<F>, smem, attr)) return 2;
    fused_gemm_kernel<F><<<grid, Mma::WarpCount::kCount * 32, smem, st>>>(P);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

extern "C" int dpt_gemm(int cfg, int M, int N, int K, const void* x, int lda, const void* w, const EpiArgs* e,
                        void* stream) {
    if (M <= 0 || K % 8 || lda % 8 || lda < K || !al16(x) || !al16(w) || !epi_ok(*e, N)) return 1;
    cudaStream_t s = (cudaStream_t)stream;
    switch (cfg) {
        case 0: return run_gemm<FusedGemm<GS<128, 128, 32>, GS<32, 64, 32>, 3, 8>>(M, N, K, x, lda, w, *e, s);
        case 1: return run_gemm<FusedGemm<GS<128, 128, 32>, GS<64, 64, 32>, 4, 8>>(M, N, K, x, lda, w, *e, s);
        case 2: return run_gemm<FusedGemm<GS<128, 64, 32>, GS<32, 32, 32>, 4, 8>>(M, N, K, x, lda, w, *e, s);
        case 3: return run_gemm<FusedGemm<GS<64, 64, 32>, GS<32, 32, 32>, 5, 8>>(M, N, K, x, lda, w, *e, s);
        case 4: return run_gemm<FusedGemm<GS<128, 256, 32>, GS<64, 64, 32>, 3, 8>>(M, N, K, x, lda, w, *e, s);
        // fp32 accumulation (precision-critical layers)
        case 5: return run_gemm<FusedGemm<GS<128, 64, 32>, GS<32, 32, 32>, 4, 8, float>>(M, N, K, x, lda, w, *e, s);
        case 6: return run_gemm<FusedGemm<GS<64, 64, 32>, GS<32, 32, 32>, 4, 8, float>>(M, N, K, x, lda, w, *e, s);
    }
    return -1;
}

// ---------------------------------------------------------------- small kernels
// Bicubic resize (PyTorch upsample_bicubic2d, align_corners=False, a = -0.75, replicated borders; = OpenCV INTER_CUBIC)
// of a float [B, 3, H, W] image, (v - mean[c]) / std[c], written as fp16 patch rows: out[b, gy * gw + gx, k] with
// k = (c * P + ky) * P + kx (the patch-embedding weight order), k in [3 P^2, kp) zero.
__device__ __forceinline__ float cc1(float x) { return ((1.25f * x - 2.25f) * x) * x + 1.f; }               // |x| <= 1
__device__ __forceinline__ float cc2(float x) { return ((-0.75f * x + 3.75f) * x - 6.f) * x + 3.f; }        // 1 < |x| < 2

__global__ void patches_k(const float* __restrict__ img, int B, int H, int W, int h, int w, float scy, float scx, int P,
                          int kp, float m0, float m1, float m2, float s0, float s1, float s2, __half* __restrict__ out) {
    int gw = w / P, gh = h / P;
    long long total = (long long)B * gh * gw * kp;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        int k = int(i % kp);
        long long r = i / kp;
        int pix = int(r % ((long long)gh * gw));
        int b = int(r / ((long long)gh * gw));
        if (k >= 3 * P * P) {
            out[i] = __float2half(0.f);
            continue;
        }
        int c = k / (P * P), kk = k - c * P * P, ky = kk / P, kx = kk - ky * P;
        int y = (pix / gw) * P + ky, x = (pix % gw) * P + kx;
        float ry = scy * (y + 0.5f) - 0.5f, rx = scx * (x + 0.5f) - 0.5f;
        int iy = (int)floorf(ry), ix = (int)floorf(rx);
        float ty = ry - iy, tx = rx - ix;
        float cy[4] = {cc2(ty + 1.f), cc1(ty), cc1(1.f - ty), cc2(2.f - ty)};
        float cx[4] = {cc2(tx + 1.f), cc1(tx), cc1(1.f - tx), cc2(2.f - tx)};
        int xs[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) xs[j] = min(max(ix - 1 + j, 0), W - 1);
        const float* src = img + ((long long)b * 3 + c) * H * W;
        float rows[4];
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const float* row = src + (long long)min(max(iy - 1 + a, 0), H - 1) * W;
            rows[a] = __ldg(row + xs[0]) * cx[0] + __ldg(row + xs[1]) * cx[1] + __ldg(row + xs[2]) * cx[2] +
                      __ldg(row + xs[3]) * cx[3];
        }
        float v = rows[0] * cy[0] + rows[1] * cy[1] + rows[2] * cy[2] + rows[3] * cy[3];
        float m = c == 0 ? m0 : (c == 1 ? m1 : m2), s = c == 0 ? s0 : (c == 1 ? s1 : s2);
        out[i] = __float2half((v - m) / s);
    }
}

static dim3 grid_for(long long n) {
    long long b = (n + 255) / 256;
    return dim3((unsigned)(b < 65535 * 8 ? b : 65535 * 8));
}

// ms: mean[3], std[3]; scy / scx: input / output size (PyTorch's fp32 scale)
extern "C" int dpt_patches(const void* img, int B, int H, int W, int h, int w, float scy, float scx, int P, int kp,
                           const float* ms, void* out, void* stream) {
    if (h % P || w % P || kp < 3 * P * P || kp % 8) return 1;
    long long n = (long long)B * (h / P) * (w / P) * kp;
    patches_k<<<grid_for(n), 256, 0, (cudaStream_t)stream>>>((const float*)img, B, H, W, h, w, scy, scx, P, kp, ms[0], ms[1],
                                                             ms[2], ms[3], ms[4], ms[5], (__half*)out);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// bilinear resize of a dense NHWC fp16 tensor (C % 8 == 0); src index = sc * dst (align_corners) or
// max(sc * (dst + 0.5) - 0.5, 0), scales computed by the caller as PyTorch does (fp32): upsample_bilinear2d arithmetic
__global__ void bilerp_k(const __half* __restrict__ src, int N, int Hs, int Ws, __half* __restrict__ dst, int Hd, int Wd,
                         int C, float sh, float sw, int align) {
    const int CV = C / 8;
    const long long total = (long long)N * Hd * Wd * CV;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long pix = i / CV;
        int c0 = (int)(i - pix * CV) * 8;
        int x = (int)(pix % Wd);
        long long r = pix / Wd;
        int y = (int)(r % Hd), n = (int)(r / Hd);
        float fy = align ? sh * y : fmaxf(sh * (y + 0.5f) - 0.5f, 0.f);
        float fx = align ? sw * x : fmaxf(sw * (x + 0.5f) - 0.5f, 0.f);
        int y0 = (int)fy, x0 = (int)fx;
        int y1 = y0 + (y0 < Hs - 1), x1 = x0 + (x0 < Ws - 1);
        float ly = fy - y0, lx = fx - x0;
        float hy = 1.f - ly, hx = 1.f - lx;
        const __half* s = src + (long long)n * Hs * Ws * C + c0;
        uint4 a = __ldg(reinterpret_cast<const uint4*>(s + ((long long)y0 * Ws + x0) * C));
        uint4 b = __ldg(reinterpret_cast<const uint4*>(s + ((long long)y0 * Ws + x1) * C));
        uint4 c = __ldg(reinterpret_cast<const uint4*>(s + ((long long)y1 * Ws + x0) * C));
        uint4 d = __ldg(reinterpret_cast<const uint4*>(s + ((long long)y1 * Ws + x1) * C));
        const __half *a8 = reinterpret_cast<const __half*>(&a), *b8 = reinterpret_cast<const __half*>(&b);
        const __half *c8 = reinterpret_cast<const __half*>(&c), *d8 = reinterpret_cast<const __half*>(&d);
        uint4 o;
        __half* o8 = reinterpret_cast<__half*>(&o);
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o8[k] = __float2half(hy * (hx * __half2float(a8[k]) + lx * __half2float(b8[k])) +
                                 ly * (hx * __half2float(c8[k]) + lx * __half2float(d8[k])));
        *reinterpret_cast<uint4*>(dst + pix * C + c0) = o;
    }
}

extern "C" int dpt_bilinear(const void* src, int N, int Hs, int Ws, void* dst, int Hd, int Wd, int C, float sh, float sw,
                            int align, void* stream) {
    if (C % 8 || !al16(src) || !al16(dst)) return 1;
    bilerp_k<<<grid_for((long long)N * Hd * Wd * (C / 8)), 256, 0, (cudaStream_t)stream>>>(
        (const __half*)src, N, Hs, Ws, (__half*)dst, Hd, Wd, C, sh, sw, align);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// DPT output tail: out[p] = relu(sum_c x[p, c] * w[c] + b) in fp32 from a dense fp16 NHWC [P, C] tensor (C % 8 == 0)
__global__ void head_dot_k(const __half* __restrict__ x, long long P, int C, const float* __restrict__ w, float b,
                           float* __restrict__ out) {
    long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (p >= P) return;
    const uint4* r = reinterpret_cast<const uint4*>(x + p * C);
    float acc = b;
    for (int c8 = 0; c8 < C / 8; ++c8) {
        uint4 q = __ldg(r + c8);
        const __half* h = reinterpret_cast<const __half*>(&q);
#pragma unroll
        for (int t = 0; t < 8; ++t) acc += __half2float(h[t]) * __ldg(w + c8 * 8 + t);
    }
    out[p] = fmaxf(acc, 0.f);
}

extern "C" int dpt_head_dot(const void* x, long long P, int C, const float* w, float b, void* out, void* stream) {
    if (C % 8 || !al16(x)) return 1;
    head_dot_k<<<(unsigned)((P + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const __half*)x, P, C, w, b, (float*)out);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
