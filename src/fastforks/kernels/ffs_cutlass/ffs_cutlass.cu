// Fast-FoundationStereo max kernels: fused 2-D / 3-D convolutions (CUTLASS implicit GEMM, sm_80-style mma.sync, fp16
// accumulation, bias + activation + residual in the epilogue, output into a channel slice or a sub-pixel voxel map) and
// the network's elementwise / reduction kernels (cost volume, geometry lookup, GRU gates, classifier soft-argmax,
// trilinear upsample-add, context upsample, row LayerNorm, depthwise conv, stem conv, input packing). The problem size
// is a runtime argument: nothing is compiled or tuned per input size, one shared library per target
// (tools/build_kernels.py).
// Origin: our Fast-FoundationStereo fork (exp 003 attempt/trtgap c9f070b, csrc/ffs_fused.cu; its conv from exp 006
// fastbb / exp 008 clf_fast csrc/fused_conv.cu), with the torch glue removed (C entry points, the caller allocates).
//
//   conv: y = act(conv(x, w) + bias[k] (+ res[n, p, q, k]))   x, res: NHWC / NDHWC fp16 (packed), w: K(T)RSC fp16,
//         y: [M, ldy] (ldy >= K, channel slice of a wider buffer), C % 8 == 0 and K % 8 == 0; 3-D convs optionally
//         multiply by a depth-broadcast gate[n, p, q, k] after the activation
//   act: 0 none, 1 relu, 2 relu6, 3 silu, 4 hardswish, 5 gelu (erf), 6 gelu (tanh), 7 sigmoid, 8 tanh, 9 leaky 0.01
// Entry points return 0 (ran), 1 (geometry not supported by the tile config), -1 (unknown config), 2 / 3 (CUDA error).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv3d_fprop.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using Hf = cutlass::half_t;
using NHWC = cutlass::layout::TensorNHWC;
using NDHWC = cutlass::layout::TensorNDHWC;

struct EpiArgs {
    const __half* bias;   // [K] or null
    const __half* res;    // [M, K] or null
    __half* y;            // [M, ldy]
    int act;
    int ldy;
    // optional output index map (sub-pixel transposed convs): row (n, z', p', q') of the sub-conv -> output voxel
    // (n, 2 z' + bz, 2 p' + bp, 2 q' + bq) of an [N, Do, Ho, Wo, ldy] buffer; out-of-range voxels are skipped
    int map = 0, Zs = 1, Ps = 1, Qs = 1, bz = 0, bp = 0, bq = 0, Do = 1, Ho = 1, Wo = 1;
    // optional gate (3-D convs, no map): y = act(...) * gate[n, p, q, k], gate [N, P, Q, K] broadcast over the depth;
    // output row r = (n Z + z) P Q + p Q + q -> gate row n P Q + p Q + q
    const __half* gate = nullptr;
    long long gZPQ = 1, gPQ = 1;
};

__device__ __forceinline__ float act_f(float v, int act) {
    switch (act) {
        case 1: return fmaxf(v, 0.f);
        case 2: return fminf(fmaxf(v, 0.f), 6.f);
        case 3: return v / (1.f + __expf(-v));
        case 4: return v * fminf(fmaxf(v + 3.f, 0.f), 6.f) * (1.f / 6.f);
        case 5: return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
        case 6: return 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        case 7: return 1.f / (1.f + __expf(-v));
        case 8: return tanhf(v);
        case 9: return v > 0.f ? v : 0.01f * v;
    }
    return v;
}

static int launch_status() { return cudaGetLastError() == cudaSuccess ? 0 : 3; }

// ------------------------------------------------------------------------------------------------ fused convolutions
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
        long long dst = p * a.ldy;
        if (a.map) {
            long long t = p;
            int q = t % a.Qs; t /= a.Qs; int pp = t % a.Ps; t /= a.Ps; int z = t % a.Zs; long long nb = t / a.Zs;
            int zo = 2 * z + a.bz, yo = 2 * pp + a.bp, xo = 2 * q + a.bq;
            if (zo < 0 || zo >= a.Do || yo < 0 || yo >= a.Ho || xo < 0 || xo >= a.Wo) return;
            dst = (((nb * a.Do + zo) * a.Ho + yo) * a.Wo + xo) * a.ldy;
        }
        const __half* av = reinterpret_cast<const __half*>(&acc);
        __align__(16) __half bv[8];
        if (a.bias) *reinterpret_cast<uint4*>(bv) = __ldg(reinterpret_cast<const uint4*>(a.bias + n));
        const __half* rv = reinterpret_cast<const __half*>(&pr[frag_idx]);
        __align__(16) __half gv[8];
        if (a.gate)
            *reinterpret_cast<uint4*>(gv) =
                __ldg(reinterpret_cast<const uint4*>(a.gate + ((p / a.gZPQ) * a.gPQ + p % a.gPQ) * N + n));
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __half2float(av[k]);
            if (a.bias) v += __half2float(bv[k]);
            if (a.res) v += __half2float(rv[k]);
            v = act_f(v, a.act);
            if (a.gate) v *= __half2float(gv[k]);
            o[k] = __float2half(v);
        }
        *reinterpret_cast<uint4*>(a.y + dst + n) = *reinterpret_cast<const uint4*>(o);
    }
};

template <class Base>
struct FusedFrom {
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

using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;  // only shapes the epilogue

template <class TB, class WS, int Stages, int Swz,
          cutlass::conv::IteratorAlgorithm Alg = cutlass::conv::IteratorAlgorithm::kOptimized>
struct Fused2 : FusedFrom<typename cutlass::conv::kernel::DefaultConv2dFprop<
                    Hf, NHWC, Hf, NHWC, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
                    cutlass::gemm::GemmShape<16, 8, 16>, EpiOp,
                    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
                    cutlass::arch::OpMultiplyAdd, Alg>::Kernel> {
    using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Hf, NHWC, Hf, NHWC, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>,
        Stages, cutlass::arch::OpMultiplyAdd, Alg>::Kernel;
    CUTLASS_HOST_DEVICE static long long rows(typename Base::Params const& p) {
        return (long long)p.problem_size.N * p.problem_size.P * p.problem_size.Q;
    }
};

template <class TB, class WS, int Stages, int Swz>
struct Fused3 : FusedFrom<typename cutlass::conv::kernel::DefaultConv3dFprop<
                    Hf, NDHWC, Hf, NDHWC, Hf, NDHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
                    cutlass::gemm::GemmShape<16, 8, 16>, EpiOp,
                    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
                    cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel> {
    using Base = typename cutlass::conv::kernel::DefaultConv3dFprop<
        Hf, NDHWC, Hf, NDHWC, Hf, NDHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>,
        Stages, cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    CUTLASS_HOST_DEVICE static long long rows(typename Base::Params const& p) {
        auto const& s = p.problem_size;
        return (long long)s.N * s.Z * s.P * s.Q;
    }
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
    cutlass::MatrixCoord extent(int(F::rows(prm)), prm.problem_size.K);
    typename F::V v(P.e, prm.iterator_D, extent, thread_idx, tb_off);
    typename F::Epilogue epi(ss.epilogue, thread_idx, warp_idx, lane_idx);
    epi(v, acc);
}

// CUTLASS's device-level can_implement also caps every operand at 2 GiB. The 3-D fprop operand iterators (kOptimized)
// add int64 offsets (int32 layout strides times int32 coordinates, each product below the tensor's element count) and
// the epilogue visitor writes with int64 offsets, so a 3-D conv needs only < 2^31 elements per tensor (a 12 MP volume).
template <class Base>
static bool can_implement3(cutlass::conv::Conv3dProblemSize const& p) {
    constexpr long long kMax = 1ll << 31;
    return Base::Mma::IteratorA::can_implement(p) == cutlass::Status::kSuccess &&
           Base::Mma::IteratorB::can_implement(p) == cutlass::Status::kSuccess && p.groups == 1 &&
           p.split_k_slices == 1 && p.K % 8 == 0 && p.activation_size() < kMax && p.filter_size() < kMax &&
           p.output_size() < kMax;
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

template <class F, class Args>
static int launch_fused(Args const& args, EpiArgs const& e, cudaStream_t s, bool checked = false) {
    using Base = typename F::Base;
    if (!checked &&
        cutlass::conv::device::ImplicitGemmConvolution<Base>::can_implement(args) != cutlass::Status::kSuccess)
        return 1;
    typename F::Params P{typename Base::Params(args, nullptr), e};
    typename Base::ThreadblockSwizzle sw;
    dim3 grid = sw.get_grid_shape(P.base.grid_tiled_shape);
    int smem = int(sizeof(typename F::SharedStorage));
    static int attr[64] = {};
    if (!smem_attr((const void*)fused_conv_kernel<F>, smem, attr)) return 2;
    fused_conv_kernel<F><<<grid, Base::kThreadCount, smem, s>>>(P);
    return launch_status();
}

template <class F>
static int run2(cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e,
                cudaStream_t s) {
    typename F::Base::Arguments args(
        p, {(Hf*)x, NHWC::packed({p.N, p.H, p.W, p.C})}, {(Hf*)w, NHWC::packed({p.K, p.R, p.S, p.C})},
        {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})}, {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})}, {Hf(1), Hf(0)});
    return launch_fused<F>(args, e, s);
}

template <class F>
static int run3(cutlass::conv::Conv3dProblemSize const& p, const void* x, const void* w, EpiArgs const& e,
                cudaStream_t s) {
    typename F::Base::Arguments args(
        p, {(Hf*)x, NDHWC::packed({p.N, p.D, p.H, p.W, p.C})}, {(Hf*)w, NDHWC::packed({p.K, p.T, p.R, p.S, p.C})},
        {nullptr, NDHWC::packed({p.N, p.Z, p.P, p.Q, p.K})}, {nullptr, NDHWC::packed({p.N, p.Z, p.P, p.Q, p.K})},
        {Hf(1), Hf(0)});
    if (!can_implement3<typename F::Base>(p)) return 1;
    return launch_fused<F>(args, e, s, true);
}

template <int M, int N, int K>
using GS = cutlass::gemm::GemmShape<M, N, K>;
using cutlass::conv::IteratorAlgorithm;

static int dispatch2(int cfg, cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w,
                     EpiArgs const& e, cudaStream_t s) {
    switch (cfg) {
        case 2: return run2<Fused2<GS<64, 64, 32>, GS<32, 32, 32>, 5, 1>>(p, x, w, e, s);
        case 7: return run2<Fused2<GS<64, 128, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 8: return run2<Fused2<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 9: return run2<Fused2<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        // C == 8 (zero-padded RGB): fixed-channels iterator, no wasted reduction
        case 10: return run2<Fused2<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4, IteratorAlgorithm::kFixedChannels>>(p, x, w, e, s);
        case 12: return run2<Fused2<GS<64, 64, 32>, GS<32, 32, 32>, 4, 1, IteratorAlgorithm::kFixedChannels>>(p, x, w, e, s);
        // narrow output (K <= 32)
        case 13: return run2<Fused2<GS<128, 32, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
    }
    return -1;
}

static int dispatch3(int cfg, cutlass::conv::Conv3dProblemSize const& p, const void* x, const void* w,
                     EpiArgs const& e, cudaStream_t s) {
    switch (cfg) {
        case 0: return run3<Fused3<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 1: return run3<Fused3<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 2: return run3<Fused3<GS<64, 64, 32>, GS<32, 32, 32>, 5, 1>>(p, x, w, e, s);
        // K <= 32: 3 stages (30 KB shared memory, 3 CTAs per SM on sm_120; 4 stages: 2 CTAs, ~20 % slower)
        case 3: return run3<Fused3<GS<128, 32, 32>, GS<32, 32, 32>, 3, 4>>(p, x, w, e, s);
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

// fixed rules, no timing (exp 008 sweeps of ResNet / MobileNet / EfficientNet convs, exp 003 Fast-FS layers):
// 128x128 tiles when K fits 128-wide tiles (<= 10 % padding) and they fill the GPU, else 128x64, 64x64 when there are
// too few CTAs; C == 8: the fixed-channels tiles.
static int pick2(long long M, int K, int C) {
    int SMS = sm_count();
    auto tiles = [&](int tm, int tn) { return ((M + tm - 1) / tm) * (long long)((K + tn - 1) / tn); };
    auto fits = [&](int tn) { return ((K + tn - 1) / tn) * tn <= 1.1 * K; };
    if (C == 8) return tiles(128, 64) >= SMS ? 10 : 12;
    if (fits(128)) {
        if (tiles(128, 128) >= SMS) return 8;
        if (tiles(64, 128) >= SMS) return 7;
        return 2;
    }
    if (tiles(128, 64) >= SMS) return 9;
    return 2;
}

// output-channel tile 32 for K <= 32, 128x128 when K fits it (<= 15 % padding: K 112 of FFS) and fills the GPU, else
// 128x64; 64x64 with too few CTAs
static int pick3(long long M, int K) {
    int SMS = sm_count();
    auto tiles = [&](int tm, int tn) { return ((M + tm - 1) / tm) * (long long)((K + tn - 1) / tn); };
    if (K <= 32) return tiles(128, 32) >= SMS ? 3 : 2;
    bool fit128 = ((K + 127) / 128) * 128 <= 1.15 * K;
    if (fit128 && tiles(128, 128) >= SMS) return 0;
    if (tiles(128, 64) >= SMS) return 1;
    return 2;
}

// x [N,H,W,C], w [K,R,S,C], res [N,P,Q,K] (packed) or null, y [N,P,Q,ldy]; cfg < 0: pick by the fixed rule
extern "C" int ffs_conv2d(int cfg, const void* x, const void* w, int N, int H, int W, int C, int K, int R, int S,
                          int ph, int pw, int sh, int sw, const void* bias, const void* res, void* y, int ldy, int act,
                          void* stream) {
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    cutlass::conv::Conv2dProblemSize p({N, H, W, C}, {K, R, S, C}, {ph, ph, pw, pw}, {sh, sw}, {1, 1}, {N, P, Q, K},
                                       cutlass::conv::Mode::kCrossCorrelation, 1);
    EpiArgs e{(const __half*)bias, (const __half*)res, (__half*)y, act, ldy};
    if (cfg < 0) cfg = pick2((long long)N * P * Q, K, C);
    return dispatch2(cfg, p, x, w, e, (cudaStream_t)stream);
}

// x [N,D,H,W,C], w [K,T,R,S,C], res [N,Z,P,Q,K] (packed) or null, gate [N,P,Q,K] (packed) or null, y [N,Z,P,Q,ldy];
// map (sub-pixel class of a stride-2 transposed conv): y is [N,Do,Ho,Wo,ldy] and row (n,z,p,q) goes to voxel
// (n, 2z+bz, 2p+bp, 2q+bq) (out-of-range voxels skipped; no residual)
extern "C" int ffs_conv3d(int cfg, const void* x, const void* w, int N, int D, int H, int W, int C, int K, int T, int R,
                          int S, int pd, int ph, int pw, int sd, int sh, int sw, const void* bias, const void* res,
                          const void* gate, void* y, int ldy, int act, int map, int bz, int bp, int bq, int Do, int Ho,
                          int Wo, void* stream) {
    int Z = (D + 2 * pd - T) / sd + 1, P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    cutlass::conv::Conv3dProblemSize p(N, D, H, W, C, K, T, R, S, Z, P, Q, pd, ph, pw, sd, sh, sw, 1, 1, 1,
                                       cutlass::conv::Mode::kCrossCorrelation, 1);
    EpiArgs e{(const __half*)bias, (const __half*)res, (__half*)y, act, ldy};
    if (gate) {
        if (map || ldy != K) return 1;
        e.gate = (const __half*)gate; e.gZPQ = (long long)Z * P * Q; e.gPQ = (long long)P * Q;
    }
    if (map) {
        if (res) return 1;
        e.map = 1; e.Zs = Z; e.Ps = P; e.Qs = Q; e.bz = bz; e.bp = bp; e.bq = bq; e.Do = Do; e.Ho = Ho; e.Wo = Wo;
    }
    if (cfg < 0) cfg = pick3((long long)N * Z * P * Q, K);
    return dispatch3(cfg, p, x, w, e, (cudaStream_t)stream);
}

// ------------------------------------------------------------------------------------------------ feature kernels
// network input: fp32 / fp16 NCHW (packed) -> fp16 NHWC with channels zero-padded to 8 (stem conv C 3 -> 8)
template <class T>
__global__ void input_nhwc8_k(const T* __restrict__ x, __half* __restrict__ y, int C, long long HW, long long total) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    long long n = i / HW, hw = i % HW;
    __align__(16) __half o[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) o[c] = c < C ? __float2half((float)x[(n * C + c) * HW + hw]) : __float2half(0.f);
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

extern "C" int ffs_input_nhwc8(const void* x, int fp32, void* y, int N, int C, int H, int W, void* stream) {
    if (C > 8) return 1;
    long long total = (long long)N * H * W;
    cudaStream_t st = (cudaStream_t)stream;
    unsigned blocks = (unsigned)((total + 255) / 256);
    if (fp32)
        input_nhwc8_k<float><<<blocks, 256, 0, st>>>((const float*)x, (__half*)y, C, (long long)H * W, total);
    else
        input_nhwc8_k<__half><<<blocks, 256, 0, st>>>((const __half*)x, (__half*)y, C, (long long)H * W, total);
    return launch_status();
}

// direct conv for a C == 8 input with few output channels (K <= 16): one thread = 8 output channels of one pixel;
// weights [K/8][R][S][8 in][8 out] in shared memory
__global__ void stem_c8_k(const __half* __restrict__ x, const __half* __restrict__ wt, const __half* __restrict__ b,
                          __half* __restrict__ y, int N, int H, int W, int K8, int P, int Q, int R, int S, int sh,
                          int sw, int ph, int pw, int act) {
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

// wt: [K/8][R][S][8 in][8 out]; y [N,P,Q,K] packed
extern "C" int ffs_stem_c8(const void* x, const void* wt, const void* b, void* y, int N, int H, int W, int K, int R,
                           int S, int ph, int pw, int sh, int sw, int act, void* stream) {
    if (K % 8 || K > 16) return 1;
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    long long total = (long long)N * P * Q * (K / 8);
    int smem = K * R * S * 8 * 2;
    if (smem > 48 * 1024) return 1;
    stem_c8_k<<<(unsigned)((total + 127) / 128), 128, smem, (cudaStream_t)stream>>>(
        (const __half*)x, (const __half*)wt, (const __half*)b, (__half*)y, N, H, W, K / 8, P, Q, R, S, sh, sw, ph, pw,
        act);
    return launch_status();
}

// depthwise conv (groups == C == K), NHWC fp16, weight [R, S, C] fp16, + bias + act; one thread = 8 channels x 2
// consecutive output columns (weights / input columns reused across both), fp32 accumulation
template <int TW>
__global__ void dwconv_tw(const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
                          __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q, int R, int S, int sh,
                          int sw, int ph, int pw, int act) {
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
    const int L = (TW - 1) * sw + S;
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
                int s2 = l - j * sw;
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

// x [N,H,W,C], w [R,S,C], y [N,P,Q,C] (packed), C % 8 == 0
extern "C" int ffs_dwconv(const void* x, const void* w, const void* b, void* y, int N, int H, int W, int C, int R,
                          int S, int ph, int pw, int sh, int sw, int act, void* stream) {
    if (C % 8) return 1;
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    long long total = (long long)N * P * ((Q + 1) / 2) * (C / 8);
    dwconv_tw<2><<<(unsigned)((total + 127) / 128), 128, 0, (cudaStream_t)stream>>>(
        (const __half*)x, (const __half*)w, (const __half*)b, (__half*)y, N, H, W, C / 8, P, Q, R, S, sh, sw, ph, pw,
        act);
    return launch_status();
}

// LayerNorm over the last dim of a packed [R, C] fp16 tensor (C % 8 == 0, C <= 512): 8 lanes per row, fp32 statistics
// (two-pass in registers), affine (= F.layer_norm); l2: y = x / max(||x||_2, eps) (= F.normalize), no affine
__global__ void ln_rows_k(const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
                          __half* __restrict__ y, long long R, int C, float eps, int l2) {
    long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / 8;
    int lane = threadIdx.x % 8;
    bool ok = row < R;
    const uint4* xr = reinterpret_cast<const uint4*>(x + (ok ? row : 0) * C);
    int n8 = C / 8;
    float v[64];
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int c8 = lane + j * 8;
        if (c8 < n8 && ok) {
            uint4 q = __ldg(xr + c8);
            const __half* h = reinterpret_cast<const __half*>(&q);
#pragma unroll
            for (int t = 0; t < 8; ++t) { v[j * 8 + t] = __half2float(h[t]); s += v[j * 8 + t]; }
        } else {
#pragma unroll
            for (int t = 0; t < 8; ++t) v[j * 8 + t] = 0.f;
        }
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o, 8);
    float mean = l2 ? 0.f : s / C, q2 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int c8 = lane + j * 8;
        if (c8 < n8) {
#pragma unroll
            for (int t = 0; t < 8; ++t) { float d = v[j * 8 + t] - mean; q2 += d * d; }
        }
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) q2 += __shfl_xor_sync(0xffffffff, q2, o, 8);
    float rs = l2 ? 1.f / fmaxf(sqrtf(q2), eps) : rsqrtf(q2 / C + eps);
    if (!ok) return;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int c8 = lane + j * 8;
        if (c8 < n8) {
            __align__(16) __half o[8];
            if (l2) {
#pragma unroll
                for (int t = 0; t < 8; ++t) o[t] = __float2half(v[j * 8 + t] * rs);
            } else {
                uint4 wq = __ldg(reinterpret_cast<const uint4*>(w) + c8), bq = __ldg(reinterpret_cast<const uint4*>(b) + c8);
                const __half* wh = reinterpret_cast<const __half*>(&wq);
                const __half* bh = reinterpret_cast<const __half*>(&bq);
#pragma unroll
                for (int t = 0; t < 8; ++t)
                    o[t] = __float2half((v[j * 8 + t] - mean) * rs * __half2float(wh[t]) + __half2float(bh[t]));
            }
            reinterpret_cast<uint4*>(y + row * C)[c8] = *reinterpret_cast<uint4*>(o);
        }
    }
}

extern "C" int ffs_ln_rows(const void* x, const void* w, const void* b, void* y, long long R, int C, float eps, int l2,
                           void* stream) {
    if (C % 8 || C > 512 || (!l2 && (!w || !b))) return 1;
    long long th = R * 8;
    ln_rows_k<<<(unsigned)((th + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (const __half*)x, (const __half*)w, (const __half*)b, (__half*)y, R, C, eps, l2);
    return launch_status();
}

// ------------------------------------------------------------------------------------------------ cost volume
// cat([group-wise correlation volume, concat volume], channels) in NDHWC: out[b, d, h, w, :] =
//   [ sum_c nl[b,h,w,g*Cg+c] * nr[b,h,w-d,g*Cg+c] (g < G) | L[b,h,w,:] | R[b,h,w-d,:] ]   (w - d < 0: gwc and R parts 0)
// nl / nr: features [B, H, W, G*Cg] fp16 (norm: group-normalized in the kernel, F.normalize in fp32 rounded to fp16);
// L / R: [B, H, W, Cc] fp16. fp32 accumulation. Block = (b, h, tile of TW columns); feature rows staged in smem.
template <int TW>
__global__ void build_volume_k(const __half* __restrict__ nl, const __half* __restrict__ nr,
                               const __half* __restrict__ L, const __half* __restrict__ R, __half* __restrict__ out,
                               int B, int H, int W, int D, int G, int Cg, int Cc, int norm) {
    extern __shared__ __half sm[];
    int CF = G * Cg;
    int w0 = blockIdx.x * TW; int h = blockIdx.y; int b = blockIdx.z;
    int nr_cols = TW + D - 1;
    __half* sl = sm;
    __half* sr = sm + TW * CF;
    long long rowbase = ((long long)b * H + h) * W;
    for (int i = threadIdx.x; i < TW * CF / 2; i += blockDim.x) {
        int wl = (i * 2) / CF, c = (i * 2) % CF; int w = w0 + wl;
        __half2 v = w < W ? *reinterpret_cast<const __half2*>(nl + (rowbase + w) * CF + c) : __float2half2_rn(0.f);
        *reinterpret_cast<__half2*>(sl + wl * CF + c) = v;
    }
    for (int i = threadIdx.x; i < nr_cols * CF / 2; i += blockDim.x) {
        int wr = (i * 2) / CF, c = (i * 2) % CF; int w = w0 - D + 1 + wr;
        __half2 v = (w >= 0 && w < W) ? *reinterpret_cast<const __half2*>(nr + (rowbase + w) * CF + c)
                                      : __float2half2_rn(0.f);
        *reinterpret_cast<__half2*>(sr + wr * CF + c) = v;
    }
    __syncthreads();
    if (norm) {
        for (int j = threadIdx.x; j < (TW + nr_cols) * G; j += blockDim.x) {
            int row = j / G, g = j % G;
            __half* v = (row < TW ? sl + row * CF : sr + (row - TW) * CF) + g * Cg;
            float ss = 0.f;
            for (int c = 0; c < Cg; ++c) { float f = __half2float(v[c]); ss += f * f; }
            float inv = 1.f / fmaxf(sqrtf(ss), 1e-12f);
            for (int c = 0; c < Cg; ++c) v[c] = __float2half(__half2float(v[c]) * inv);
        }
        __syncthreads();
    }
    int Ctot = G + 2 * Cc;
    for (int j = threadIdx.x; j < TW * D; j += blockDim.x) {
        int wl = j % TW, d = j / TW; int w = w0 + wl;
        if (w >= W) continue;
        __half* o = out + ((((long long)b * D + d) * H + h) * W + w) * Ctot;
        bool valid = w - d >= 0;
        const __half* a = sl + wl * CF;
        const __half* r = sr + (wl - d + D - 1) * CF;
        __align__(16) __half ob[64];
        for (int g = 0; g < G; ++g) {
            float acc = 0.f;
            if (valid) {
                for (int c = 0; c < Cg; c += 2) {
                    float2 x = __half22float2(*reinterpret_cast<const __half2*>(a + g * Cg + c));
                    float2 y = __half22float2(*reinterpret_cast<const __half2*>(r + g * Cg + c));
                    acc += x.x * y.x + x.y * y.y;
                }
            }
            ob[g] = __float2half(acc);
        }
        const __half* lp = L + (rowbase + w) * Cc;
        const __half* rp = R + (rowbase + w - d) * Cc;
        for (int c = 0; c < Cc; ++c) {
            ob[G + c] = lp[c];
            ob[G + Cc + c] = valid ? rp[c] : __float2half(0.f);
        }
        for (int c = 0; c < Ctot; c += 8) *reinterpret_cast<uint4*>(o + c) = *reinterpret_cast<const uint4*>(ob + c);
    }
}

// out [B, D, H, W, G + 2 Cc] packed; (G + 2 Cc) % 8 == 0, <= 64; Cg even
extern "C" int ffs_build_volume(const void* nl, const void* nr, const void* L, const void* R, void* out, int B, int H,
                                int W, int D, int G, int Cg, int Cc, int norm, void* stream) {
    constexpr int TW = 32;
    if (Cg % 2 || (G + 2 * Cc) % 8 || G + 2 * Cc > 64) return 1;
    size_t smem = (size_t)(TW + TW + D - 1) * G * Cg * 2;
    static int attr[64] = {};
    if (!smem_attr((const void*)build_volume_k<TW>, (int)smem, attr)) return 2;
    dim3 grid((W + TW - 1) / TW, H, B);
    build_volume_k<TW><<<grid, 256, smem, (cudaStream_t)stream>>>((const __half*)nl, (const __half*)nr,
                                                                  (const __half*)L, (const __half*)R, (__half*)out, B,
                                                                  H, W, D, G, Cg, Cc, norm);
    return launch_status();
}

// y = trilinear_upsample(x, size of r, align_corners=False) + r   (NDHWC fp16, C % 8 == 0; PyTorch source-index rule)
__device__ __forceinline__ void src_idx(int o, int in, int out, int& i0, int& i1, float& l) {
    float sc = (float)in / (float)out;
    float s = fmaxf(sc * (o + 0.5f) - 0.5f, 0.f);
    i0 = min((int)s, in - 1); i1 = min(i0 + 1, in - 1); l = s - (float)i0;
}

__global__ void up3_add_k(const __half* __restrict__ x, const __half* __restrict__ r, __half* __restrict__ y, int B,
                          int C8, int Di, int Hi, int Wi, int Do, int Ho, int Wo) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long tot = (long long)B * Do * Ho * Wo * C8;
    if (i >= tot) return;
    int c = i % C8; long long p = i / C8;
    int w = p % Wo; int h = (p / Wo) % Ho; int d = (p / ((long long)Wo * Ho)) % Do;
    int b = p / ((long long)Wo * Ho * Do);
    int d0, d1, h0, h1, w0, w1; float ld, lh, lw;
    src_idx(d, Di, Do, d0, d1, ld); src_idx(h, Hi, Ho, h0, h1, lh); src_idx(w, Wi, Wo, w0, w1, lw);
    float acc[8];
    uint4 rv = __ldg(reinterpret_cast<const uint4*>(r) + i);
    const __half* rh = reinterpret_cast<const __half*>(&rv);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        int dd = (t & 4) ? d1 : d0, hh = (t & 2) ? h1 : h0, ww = (t & 1) ? w1 : w0;
        float wt = ((t & 4) ? ld : 1.f - ld) * ((t & 2) ? lh : 1.f - lh) * ((t & 1) ? lw : 1.f - lw);
        uint4 v = __ldg(reinterpret_cast<const uint4*>(x + ((((long long)b * Di + dd) * Hi + hh) * Wi + ww) * C8 * 8) + c);
        const __half* vh = reinterpret_cast<const __half*>(&v);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += wt * __half2float(vh[k]);
    }
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(__half2float(__float2half(acc[k])) + __half2float(rh[k]));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

// x [B, Di, Hi, Wi, C], r / y [B, Do, Ho, Wo, C] packed (y may alias r)
extern "C" int ffs_up3_add(const void* x, const void* r, void* y, int B, int C, int Di, int Hi, int Wi, int Do, int Ho,
                           int Wo, void* stream) {
    if (C % 8) return 1;
    long long tot = (long long)B * Do * Ho * Wo * (C / 8);
    up3_add_k<<<(unsigned)((tot + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (const __half*)x, (const __half*)r, (__half*)y, B, C / 8, Di, Hi, Wi, Do, Ho, Wo);
    return launch_status();
}

// classifier tail: cost[d] = bias + o[d][1] + o[d-1][0] + o[d+1][2] (the 3 disparity taps of a 3x3x3 conv computed as
// output channels of one 2-D conv, o: [B, D, H, W, 8] fp16), then softmax over d and soft-argmax (fp32):
// disp[b, h, w] = sum_d softmax(cost)_d * d
__global__ void cls_regress_k(const __half* __restrict__ o, float* __restrict__ disp, int B, int D, long long HW,
                              float bias) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= B * HW) return;
    long long b = i / HW, p = i % HW;
    const uint4* base = reinterpret_cast<const uint4*>(o) + b * D * HW + p;
    float mx = -1e30f, se = 0.f, acc = 0.f;
    uint4 prev, cur = __ldg(base), nxt;
    for (int d = 0; d < D; ++d) {
        if (d + 1 < D) nxt = __ldg(base + (long long)(d + 1) * HW);
        const __half* c = reinterpret_cast<const __half*>(&cur);
        float v = bias + __half2float(c[1]);
        if (d > 0) v += __half2float(reinterpret_cast<const __half*>(&prev)[0]);
        if (d + 1 < D) v += __half2float(reinterpret_cast<const __half*>(&nxt)[2]);
        if (v > mx) { float sc = __expf(mx - v); se *= sc; acc *= sc; mx = v; }
        float e = __expf(v - mx); se += e; acc += e * d;
        prev = cur; cur = nxt;
    }
    disp[i] = acc / se;
}

extern "C" int ffs_cls_regress(const void* o, void* disp, int B, int D, long long HW, float bias, void* stream) {
    long long n = B * HW;
    cls_regress_k<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const __half*)o, (float*)disp, B, D,
                                                                                  HW, bias);
    return launch_status();
}

// context_upsample(disp_low * scale, softmax(logits, 1)) (= unfold 3x3 zero-pad, nearest upsample, weighted sum):
// logits [B, 9, H4, W4] fp16 with strides (sb, sc, sy, sx), disp [B, h, w] fp32 -> out [B, H4, W4] fp32;
// low pixel = (y * h / H4, x * w / W4)
__global__ void ctx_up_k(const __half* __restrict__ lg, long long sb, long long sc, long long sy, long long sx,
                         const float* __restrict__ disp, float* __restrict__ out, int B, int H4, int W4, int h, int w,
                         float scale) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)B * H4 * W4) return;
    int x = i % W4; int y = (i / W4) % H4; int b = i / ((long long)W4 * H4);
    int yl = (int)(((long long)y * h) / H4), xl = (int)(((long long)x * w) / W4);
    const __half* l = lg + b * sb + y * sy + x * sx;
    float v[9], mx = -1e30f;
#pragma unroll
    for (int k = 0; k < 9; ++k) { v[k] = __half2float(l[k * sc]); mx = fmaxf(mx, v[k]); }
    float se = 0.f, acc = 0.f;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        float e = __expf(v[k] - mx); se += e;
        int yy = yl + k / 3 - 1, xx = xl + k % 3 - 1;
        float dv = (yy >= 0 && yy < h && xx >= 0 && xx < w) ? disp[((long long)b * h + yy) * w + xx] : 0.f;
        acc += e * dv;
    }
    out[i] = acc / se * scale;
}

extern "C" int ffs_ctx_up(const void* logits, long long sb, long long sc, long long sy, long long sx, const void* disp,
                          void* out, int B, int H4, int W4, int h, int w, float scale, void* stream) {
    long long n = (long long)B * H4 * W4;
    ctx_up_k<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (const __half*)logits, sb, sc, sy, sx, (const float*)disp, (float*)out, B, H4, W4, h, w, scale);
    return launch_status();
}

// ------------------------------------------------------------------------------------------------ GRU iteration
// Combined geometry lookup (2 levels: 1-D linear interpolation with zero padding along D of the geometry volume and
// along W of the all-pairs correlation) written straight into the NHWC rows of the padded motion-encoder input:
//   [geo l0 (k, c) | geo l1 (k, c) | corr l0 (k) | corr l1 (k) | zero pad]   (k = tap 0..2r, c = 0..Cw-1)
// One thread per (pixel, level, tap). Also writes disp (fp16) into an 8-channel buffer and a channel of mf.
// fly: level 1 pooled from level 0 on the fly (pairwise means along D / W, rounded to fp16 like a stored level).
__global__ void geo_lookup_k(const __half* __restrict__ g0, const __half* __restrict__ g1,
                             const __half* __restrict__ c0, const __half* __restrict__ c1,
                             const float* __restrict__ disp, __half* __restrict__ out, __half* __restrict__ d8,
                             __half* __restrict__ mf, int B, int H, int W, int D0, int D1, int W0, int W1, int Cg,
                             int Cw, int K, int ldo, int ldmf, int mfoff, int fly) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long P = (long long)B * H * W;
    if (i >= P * 2 * K) return;
    int k = i % K; int l = (i / K) % 2; long long p = i / (2 * K);
    int r = K / 2;
    long long hw = p % ((long long)H * W); int b = p / ((long long)H * W); int w = hw % W;
    float d = disp[p];
    float s = l ? 0.5f : 1.f;
    int Dn = l ? D1 : D0;
    const __half* g = l ? g1 : g0;
    float pos = d * s + (float)(k - r);
    float x0 = floorf(pos); float f = pos - x0; int i0 = (int)x0, i1 = i0 + 1;
    float w0 = (i0 >= 0 && i0 <= Dn - 1) ? 1.f - f : 0.f;
    float w1 = (i1 >= 0 && i1 <= Dn - 1) ? f : 0.f;
    i0 = min(max(i0, 0), Dn - 1); i1 = min(max(i1, 0), Dn - 1);
    __half* o = out + p * ldo + (long long)(l * K + k) * Cw;
    if (l == 1 && fly) {
        const __half* r0 = g0 + (((long long)b * D0 + 2 * i0) * H * W + hw) * Cg;
        const __half* r1 = g0 + (((long long)b * D0 + 2 * i1) * H * W + hw) * Cg;
        long long st = (long long)H * W * Cg;
        for (int c = 0; c < Cw; c += 4) {
            uint2 a = __ldg(reinterpret_cast<const uint2*>(r0 + c)), a2 = __ldg(reinterpret_cast<const uint2*>(r0 + st + c));
            uint2 bb = __ldg(reinterpret_cast<const uint2*>(r1 + c)), b2 = __ldg(reinterpret_cast<const uint2*>(r1 + st + c));
            const __half* ah = reinterpret_cast<const __half*>(&a); const __half* ah2 = reinterpret_cast<const __half*>(&a2);
            const __half* bh = reinterpret_cast<const __half*>(&bb); const __half* bh2 = reinterpret_cast<const __half*>(&b2);
            __align__(8) __half v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float pa = __half2float(__float2half(0.5f * (__half2float(ah[j]) + __half2float(ah2[j]))));
                float pb = __half2float(__float2half(0.5f * (__half2float(bh[j]) + __half2float(bh2[j]))));
                v[j] = __float2half(pa * w0 + pb * w1);
            }
            *reinterpret_cast<uint2*>(o + c) = *reinterpret_cast<uint2*>(v);
        }
    } else {
        const __half* r0 = g + (((long long)b * Dn + i0) * H * W + hw) * Cg;
        const __half* r1 = g + (((long long)b * Dn + i1) * H * W + hw) * Cg;
        for (int c = 0; c < Cw; c += 4) {
            uint2 a = __ldg(reinterpret_cast<const uint2*>(r0 + c)), bb = __ldg(reinterpret_cast<const uint2*>(r1 + c));
            const __half* ah = reinterpret_cast<const __half*>(&a); const __half* bh = reinterpret_cast<const __half*>(&bb);
            __align__(8) __half v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = __float2half(__half2float(ah[j]) * w0 + __half2float(bh[j]) * w1);
            *reinterpret_cast<uint2*>(o + c) = *reinterpret_cast<uint2*>(v);
        }
    }
    int Wn = l ? W1 : W0;
    pos = (float)w * s - d * s + (float)(k - r);
    x0 = floorf(pos); f = pos - x0; i0 = (int)x0; i1 = i0 + 1;
    w0 = (i0 >= 0 && i0 <= Wn - 1) ? 1.f - f : 0.f;
    w1 = (i1 >= 0 && i1 <= Wn - 1) ? f : 0.f;
    i0 = min(max(i0, 0), Wn - 1); i1 = min(max(i1, 0), Wn - 1);
    float cv0, cv1;
    if (l == 1 && fly) {
        const __half* cc = c0 + p * W0;
        cv0 = __half2float(__float2half(0.5f * (__half2float(cc[2 * i0]) + __half2float(cc[2 * i0 + 1]))));
        cv1 = __half2float(__float2half(0.5f * (__half2float(cc[2 * i1]) + __half2float(cc[2 * i1 + 1]))));
    } else {
        const __half* cc = (l ? c1 : c0) + p * Wn;
        cv0 = __half2float(cc[i0]); cv1 = __half2float(cc[i1]);
    }
    out[p * ldo + 2 * K * Cw + l * K + k] = __float2half(cv0 * w0 + cv1 * w1);
    if (l == 0 && k == 0) {
        __half dh = __float2half(d);
        if (d8) d8[p * 8] = dh;
        if (mf) mf[p * ldmf + mfoff] = dh;
    }
}

// g0 [B, D0, H, W, Cg] (Cw of Cg channels used), g1 [B, D1, H, W, Cg] or null (fly), c0 [B, H, W, W0], c1 or null;
// disp [B, H, W] fp32; out [B, H, W, ldo] (ldo >= 2 K (Cw + 1)); d8 [B, H, W, 8] / mf [B, H, W, ldmf] or null
extern "C" int ffs_geo_lookup(const void* g0, const void* g1, const void* c0, const void* c1, const void* disp,
                              void* out, void* d8, void* mf, int B, int H, int W, int D0, int D1, int W0, int W1,
                              int Cg, int Cw, int K, int ldo, int ldmf, int mfoff, void* stream) {
    int fly = g1 == nullptr ? 1 : 0;
    if (Cg % 4 || Cw % 4 || Cw > Cg || ldo < 2 * K * (Cw + 1) || (fly && c1 != nullptr)) return 1;
    long long total = (long long)B * H * W * 2 * K;
    geo_lookup_k<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (const __half*)g0, (const __half*)g1, (const __half*)c0, (const __half*)c1, (const float*)disp, (__half*)out,
        (__half*)d8, (__half*)mf, B, H, W, D0, D1, W0, W1, Cg, Cw, K, ldo, ldmf, mfoff, fly);
    return launch_status();
}

// GRU step 1: q inputs [r*h | x] of the small (1x1) and large (3x3) GRU. zr*: [P, 2 Hd] (z | r, sigmoid applied),
// h [P, Hd], hx [P, ldhx] (x in channels 0..Cx-1), qs / ql [P, Hd + Cx]. 8 channels per thread.
__global__ void gru_rh_k(const __half* __restrict__ zrs, const __half* __restrict__ zrl, const __half* __restrict__ h,
                         const __half* __restrict__ hx, __half* __restrict__ qs, __half* __restrict__ ql, long long P,
                         int Hd, int Cx, int ldhx) {
    int n8 = (Hd + Cx) / 8;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= P * n8) return;
    int c = (i % n8) * 8; long long p = i / n8;
    int ldq = Hd + Cx;
    if (c < Hd) {
        uint4 hv = __ldg(reinterpret_cast<const uint4*>(h + p * Hd + c));
        uint4 rs = __ldg(reinterpret_cast<const uint4*>(zrs + p * 2 * Hd + Hd + c));
        uint4 rl = __ldg(reinterpret_cast<const uint4*>(zrl + p * 2 * Hd + Hd + c));
        __half2* a = reinterpret_cast<__half2*>(&hv);
        __half2* bs = reinterpret_cast<__half2*>(&rs);
        __half2* bl = reinterpret_cast<__half2*>(&rl);
        __align__(16) __half2 os[4], ol[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { os[j] = __hmul2(a[j], bs[j]); ol[j] = __hmul2(a[j], bl[j]); }
        *reinterpret_cast<uint4*>(qs + p * ldq + c) = *reinterpret_cast<uint4*>(os);
        *reinterpret_cast<uint4*>(ql + p * ldq + c) = *reinterpret_cast<uint4*>(ol);
    } else {
        uint4 xv = __ldg(reinterpret_cast<const uint4*>(hx + p * ldhx + c - Hd));
        *reinterpret_cast<uint4*>(qs + p * ldq + c) = xv;
        *reinterpret_cast<uint4*>(ql + p * ldq + c) = xv;
    }
}

extern "C" int ffs_gru_rh(const void* zrs, const void* zrl, const void* h, const void* hx, void* qs, void* ql,
                          long long P, int Hd, int Cx, int ldhx, void* stream) {
    if (Hd % 8 || Cx % 8 || ldhx % 8) return 1;
    long long total = P * ((Hd + Cx) / 8);
    gru_rh_k<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (const __half*)zrs, (const __half*)zrl, (const __half*)h, (const __half*)hx, (__half*)qs, (__half*)ql, P, Hd,
        Cx, ldhx);
    return launch_status();
}

// GRU step 2: h' = att * ((1-zs) h + zs qs) + (1-att) * ((1-zl) h + zl ql)  (qs / ql: tanh applied; each branch
// rounded to fp16); writes h (in place) and the h channels of hx (offset hxoff).
__global__ void gru_out_k(const __half* __restrict__ zrs, const __half* __restrict__ zrl,
                          const __half* __restrict__ qs, const __half* __restrict__ ql,
                          const __half* __restrict__ att, __half* __restrict__ h, __half* __restrict__ hx,
                          long long P, int Hd, int ldhx, int hxoff) {
    int n8 = Hd / 8;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= P * n8) return;
    int c = (i % n8) * 8; long long p = i / n8;
    float a = __half2float(att[p]);
    uint4 hv = *reinterpret_cast<const uint4*>(h + p * Hd + c);
    uint4 zs = __ldg(reinterpret_cast<const uint4*>(zrs + p * 2 * Hd + c));
    uint4 zl = __ldg(reinterpret_cast<const uint4*>(zrl + p * 2 * Hd + c));
    uint4 vs = __ldg(reinterpret_cast<const uint4*>(qs + p * Hd + c));
    uint4 vl = __ldg(reinterpret_cast<const uint4*>(ql + p * Hd + c));
    const __half* H_ = reinterpret_cast<const __half*>(&hv);
    const __half* ZS = reinterpret_cast<const __half*>(&zs);
    const __half* ZL = reinterpret_cast<const __half*>(&zl);
    const __half* QS = reinterpret_cast<const __half*>(&vs);
    const __half* QL = reinterpret_cast<const __half*>(&vl);
    __align__(16) __half o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float hh = __half2float(H_[j]);
        float s_ = __half2float(__float2half((1.f - __half2float(ZS[j])) * hh + __half2float(ZS[j]) * __half2float(QS[j])));
        float l_ = __half2float(__float2half((1.f - __half2float(ZL[j])) * hh + __half2float(ZL[j]) * __half2float(QL[j])));
        o[j] = __float2half(s_ * a + l_ * (1.f - a));
    }
    *reinterpret_cast<uint4*>(h + p * Hd + c) = *reinterpret_cast<uint4*>(o);
    *reinterpret_cast<uint4*>(hx + p * ldhx + hxoff + c) = *reinterpret_cast<uint4*>(o);
}

extern "C" int ffs_gru_out(const void* zrs, const void* zrl, const void* qs, const void* ql, const void* att, void* h,
                           void* hx, long long P, int Hd, int ldhx, int hxoff, void* stream) {
    if (Hd % 8 || ldhx % 8 || hxoff % 8) return 1;
    long long total = P * (Hd / 8);
    gru_out_k<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (const __half*)zrs, (const __half*)zrl, (const __half*)qs, (const __half*)ql, (const __half*)att, (__half*)h,
        (__half*)hx, P, Hd, ldhx, hxoff);
    return launch_status();
}
