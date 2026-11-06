// HITNet convolutions: conv + bias (+ residual) + LeakyReLU(0.2) as one CUTLASS implicit-GEMM kernel (sm_80-style
// mma.sync, fp16 accumulation), dilation supported (analytic iterator for dilated convolutions). The problem size is a
// runtime argument: nothing is compiled or tuned per input size, one shared library per target (tools/build_kernels.py).
// Origin: our HITNet fork (exp 003 hitnet-official efe9fb7, csrc/fused_conv.cu), itself a trimmed copy of exp 006
// fastbb/csrc/fused_conv.cu (exp 008 clf_fast), with the torch glue removed; only activations 0 (none) and 7 (LeakyReLU 0.2).
//
//   y = act(conv(x, w) + bias[k] (+ res[n, p, q, k]))     x, res, y: NHWC fp16, w: KRSC fp16, C % 8 == 0 and K % 8 == 0
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
};

__device__ __forceinline__ float act_f(float v, int act) { return act == 7 ? (v > 0.f ? v : 0.2f * v) : v; }

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
using cutlass::conv::IteratorAlgorithm;

static int dispatch(int cfg, cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e,
                    cudaStream_t s) {
    switch (cfg) {
        case 2: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 5, 1>>(p, x, w, e, s);
        case 7: return run<Fused<GS<64, 128, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 8: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 9: return run<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 13: return run<Fused<GS<128, 32, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 15: return run<Fused<GS<256, 32, 32>, GS<64, 32, 32>, 3, 4>>(p, x, w, e, s);
        // dilated convs: analytic iterator
        case 102: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 4, 1, IteratorAlgorithm::kAnalytic>>(p, x, w, e, s);
        case 109: return run<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4, IteratorAlgorithm::kAnalytic>>(p, x, w, e, s);
        case 113: return run<Fused<GS<128, 32, 32>, GS<32, 32, 32>, 4, 4, IteratorAlgorithm::kAnalytic>>(p, x, w, e, s);
    }
    return -1;
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

// fixed rule, no timing (narrow K: 32-wide N tiles; HITNet K is 16..72)
static int pick(long long M, int K, int dil) {
    int SMS = sm_count();
    auto tiles = [&](int tm, int tn) { return ((M + tm - 1) / tm) * (long long)((K + tn - 1) / tn); };
    int base;
    if (K <= 32) base = tiles(128, 32) >= SMS ? 13 : 2;
    else if (K <= 64) base = tiles(128, 64) >= SMS ? 9 : 2;
    else base = tiles(128, 128) >= SMS ? 8 : (tiles(64, 128) >= SMS ? 7 : 2);
    if (dil > 1) return base == 13 ? 113 : (base == 9 ? 109 : 102);
    return base;
}

// Returns 0 (ran), 1 (geometry not supported by the tile config), -1 (unknown config), or 2 / 3 (CUDA error).
// cfg < 0: pick by the fixed rule. x [N,H,W,C], w [K,R,S,C], res / y [N,P,Q,K] (all fp16, packed); bias [K] or null.
extern "C" int hitnet_conv(int cfg, const void* x, const void* w, int N, int H, int W, int C, int K, int R, int S, int ph,
                           int pw, int sh, int sw, int dh, int dw, const void* bias, const void* res, void* y, int act,
                           void* stream) {
    int P = (H + 2 * ph - dh * (R - 1) - 1) / sh + 1, Q = (W + 2 * pw - dw * (S - 1) - 1) / sw + 1;
    cutlass::conv::Conv2dProblemSize p({N, H, W, C}, {K, R, S, C}, {ph, ph, pw, pw}, {sh, sw}, {dh, dw}, {N, P, Q, K},
                                       cutlass::conv::Mode::kCrossCorrelation, 1);
    EpiArgs e{(const __half*)bias, (const __half*)res, (__half*)y, act, K};
    if (cfg < 0) cfg = pick((long long)N * P * Q, K, dh > dw ? dh : dw);
    return dispatch(cfg, p, x, w, e, (cudaStream_t)stream);
}
