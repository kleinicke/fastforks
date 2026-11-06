// fastbb: flash attention (head_dim 64, optional fp16 accumulation) and NHWC instance norm for fp16 / bf16, from the
// fastbb package (Apache-2.0). The chunked fp16-accumulate GEMMs of the original are not included: the ViT uses
// cuBLAS with fp16 accumulation. Nothing here depends on the input size.
#include <cuda_fp16.h>
#include <stdint.h>

// ------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    int n = valid ? 16 : 0;  // src-size 0 -> zero fill
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// D(f16) = A * B + C(f16)   m16n8k16
__device__ __forceinline__ void mma_f16(uint32_t* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// D(f32) = A * B + C(f32)
__device__ __forceinline__ void mma_f32(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float2 unpack_h2(uint32_t u) { return __half22float2(*reinterpret_cast<__half2*>(&u)); }
__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// swizzled [rows][64] fp16 tile: 8 chunks of 16 B per row, chunk index XOR (row & 7) -> conflict-free ldmatrix / cp.async
__device__ __forceinline__ int swz(int row, int chunk) { return row * 64 + ((chunk ^ (row & 7)) << 3); }

// ------------------------------------------------------------------------------------------------ flash attention
// O = softmax(scale * Q K^T) V, head_dim 64, no mask, any N (tails masked), fp16 in / out.
// Q, K, V, O: element (b, n, h, d) at ptr[b * sb + n * sn + h * sh + d] (d contiguous; strides in elements, multiples of 8,
// e.g. straight views into the packed [B, N, 3, H, 64] qkv GEMM output, and O written as [B, N, H*64] for the projection).
// FlashAttention-2 schedule: CTA = 4 warps x (16 * RM) query rows; K/V tiles of 64 keys double-buffered with cp.async.
// QK16 / PV16: fp16-accumulate mma for S = Q K^T / for each 64-key block of P V (the running O, the row max and the row sums
// stay fp32; an fp16 accumulation never spans more than 64 products: head_dim for S, one key block for P V).
template <int RM, bool QK16, bool PV16>
__device__ __forceinline__ void flash_fwd_impl(const __half* __restrict__ q, const __half* __restrict__ k,
                                               const __half* __restrict__ v, __half* __restrict__ o, int N, int H,
                                               long long qsb, int qsn, int qsh, long long ksb, int ksn, int ksh,
                                               long long vsb, int vsn, int vsh, long long osb, int osn, int osh,
                                               float scale_log2) {
    constexpr int BR = 64 * RM, BC = 64;
    __shared__ __align__(128) __half sQ[BR * 64];
    __shared__ __align__(128) __half sK[2][BC * 64];
    __shared__ __align__(128) __half sV[2][BC * 64];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int bh = blockIdx.y, b = bh / H, h = bh - b * H;
    const int q0 = blockIdx.x * BR;
    const __half* qb = q + b * qsb + (long long)h * qsh;
    const __half* kb = k + b * ksb + (long long)h * ksh;
    const __half* vb = v + b * vsb + (long long)h * vsh;

    // Q tile -> smem
#pragma unroll
    for (int i = 0; i < BR * 8 / 128; ++i) {
        int c = tid + i * 128, row = c >> 3, ch = c & 7;
        int n = q0 + row;
        cp_async16(smem_u32(&sQ[swz(row, ch)]), qb + (long long)min(n, N - 1) * qsn + ch * 8, n < N);
    }
    auto load_kv = [&](int buf, int k0) {
#pragma unroll
        for (int i = 0; i < BC * 8 / 128; ++i) {
            int c = tid + i * 128, row = c >> 3, ch = c & 7;
            int n = k0 + row;
            bool ok = n < N;
            int nn = min(n, N - 1);
            cp_async16(smem_u32(&sK[buf][swz(row, ch)]), kb + (long long)nn * ksn + ch * 8, ok);
            cp_async16(smem_u32(&sV[buf][swz(row, ch)]), vb + (long long)nn * vsn + ch * 8, ok);
        }
    };
    load_kv(0, 0);
    cp_async_commit();

    // running state: rows (lane>>2) and (lane>>2)+8 of each of the RM m-tiles of this warp
    float m_r[RM][2], l_r[RM][2];
    float oacc[RM][8][4];
#pragma unroll
    for (int mi = 0; mi < RM; ++mi) {
        m_r[mi][0] = m_r[mi][1] = -INFINITY;
        l_r[mi][0] = l_r[mi][1] = 0.f;
#pragma unroll
        for (int u = 0; u < 8; ++u) oacc[mi][u][0] = oacc[mi][u][1] = oacc[mi][u][2] = oacc[mi][u][3] = 0.f;
    }
    uint32_t qf[RM][4][4];  // Q A-fragments: [m-tile][k-step of 16 dims][4]

    const int nblk = (N + BC - 1) / BC;
    for (int j = 0; j < nblk; ++j) {
        cp_async_wait<0>();
        __syncthreads();
        if (j == 0) {
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    int row = warp * 16 * RM + mi * 16 + (lane & 15);
                    int ch = kk * 2 + (lane >> 4);
                    ldsm_x4(smem_u32(&sQ[swz(row, ch)]), qf[mi][kk][0], qf[mi][kk][1], qf[mi][kk][2], qf[mi][kk][3]);
                }
        }
        if (j + 1 < nblk) load_kv((j + 1) & 1, (j + 1) * BC);
        cp_async_commit();
        const __half* K_ = sK[j & 1];
        const __half* V_ = sV[j & 1];

        // ---- S = Q K^T for 64 keys: 8 n-tiles of 8 keys
        float s[RM][8][4];
        if (QK16) {
            uint32_t sh[RM][8][2];
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int t = 0; t < 8; ++t) sh[mi][t][0] = sh[mi][t][1] = 0u;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int t = 0; t < 8; t += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = (t + (lane >> 4)) * 8 + (lane & 7);
                    int ch = kk * 2 + ((lane >> 3) & 1);
                    ldsm_x4(smem_u32(&K_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f16(sh[mi][t], qf[mi][kk], b0, b1);
                        mma_f16(sh[mi][t + 1], qf[mi][kk], b2, b3);
                    }
                }
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    float2 a = unpack_h2(sh[mi][t][0]), c = unpack_h2(sh[mi][t][1]);
                    s[mi][t][0] = a.x; s[mi][t][1] = a.y; s[mi][t][2] = c.x; s[mi][t][3] = c.y;
                }
        } else {
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int t = 0; t < 8; ++t) s[mi][t][0] = s[mi][t][1] = s[mi][t][2] = s[mi][t][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int t = 0; t < 8; t += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = (t + (lane >> 4)) * 8 + (lane & 7);
                    int ch = kk * 2 + ((lane >> 3) & 1);
                    ldsm_x4(smem_u32(&K_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f32(s[mi][t], qf[mi][kk], b0, b1);
                        mma_f32(s[mi][t + 1], qf[mi][kk], b2, b3);
                    }
                }
        }
        // ---- tail mask (last block only)
        if ((j + 1) * BC > N) {
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                int key = j * BC + t * 8 + (lane & 3) * 2;
#pragma unroll
                for (int mi = 0; mi < RM; ++mi) {
                    if (key >= N) s[mi][t][0] = s[mi][t][2] = -INFINITY;
                    if (key + 1 >= N) s[mi][t][1] = s[mi][t][3] = -INFINITY;
                }
            }
        }
        // ---- online softmax (base 2), P packed as fp16 A-fragments
        uint32_t pf[RM][4][4];  // [m-tile][k-chunk of 16 keys][4]
#pragma unroll
        for (int mi = 0; mi < RM; ++mi) {
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                mx0 = fmaxf(mx0, fmaxf(s[mi][t][0], s[mi][t][1]));
                mx1 = fmaxf(mx1, fmaxf(s[mi][t][2], s[mi][t][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
            float mn0 = fmaxf(m_r[mi][0], mx0 * scale_log2), mn1 = fmaxf(m_r[mi][1], mx1 * scale_log2);
            float c0 = fast_exp2(m_r[mi][0] - mn0), c1 = fast_exp2(m_r[mi][1] - mn1);
            m_r[mi][0] = mn0;
            m_r[mi][1] = mn1;
            float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                float p0 = fast_exp2(fmaf(s[mi][t][0], scale_log2, -mn0));
                float p1 = fast_exp2(fmaf(s[mi][t][1], scale_log2, -mn0));
                float p2 = fast_exp2(fmaf(s[mi][t][2], scale_log2, -mn1));
                float p3 = fast_exp2(fmaf(s[mi][t][3], scale_log2, -mn1));
                sum0 += p0 + p1;
                sum1 += p2 + p3;
                // n-tile t -> k-chunk t/2: even t = a0 (row) / a1 (row+8), odd t = a2 / a3
                pf[mi][t >> 1][(t & 1) * 2 + 0] = pack_h2(p0, p1);
                pf[mi][t >> 1][(t & 1) * 2 + 1] = pack_h2(p2, p3);
            }
            l_r[mi][0] = l_r[mi][0] * c0 + sum0;
            l_r[mi][1] = l_r[mi][1] * c1 + sum1;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                oacc[mi][u][0] *= c0; oacc[mi][u][1] *= c0;
                oacc[mi][u][2] *= c1; oacc[mi][u][3] *= c1;
            }
        }
        // ---- O += P V
        if (PV16) {
            uint32_t ot[RM][8][2];
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int u = 0; u < 8; ++u) ot[mi][u][0] = ot[mi][u][1] = 0u;
#pragma unroll
            for (int kc = 0; kc < 4; ++kc)
#pragma unroll
                for (int u = 0; u < 8; u += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int ch = u + (lane >> 4);
                    ldsm_x4_t(smem_u32(&V_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f16(ot[mi][u], pf[mi][kc], b0, b1);
                        mma_f16(ot[mi][u + 1], pf[mi][kc], b2, b3);
                    }
                }
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    float2 a = unpack_h2(ot[mi][u][0]), c = unpack_h2(ot[mi][u][1]);
                    oacc[mi][u][0] += a.x; oacc[mi][u][1] += a.y; oacc[mi][u][2] += c.x; oacc[mi][u][3] += c.y;
                }
        } else {
#pragma unroll
            for (int kc = 0; kc < 4; ++kc)
#pragma unroll
                for (int u = 0; u < 8; u += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int ch = u + (lane >> 4);
                    ldsm_x4_t(smem_u32(&V_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f32(oacc[mi][u], pf[mi][kc], b0, b1);
                        mma_f32(oacc[mi][u + 1], pf[mi][kc], b2, b3);
                    }
                }
        }
    }
    // ---- epilogue: O / l -> fp16 (staged through smem for 16-byte stores)
    __syncthreads();
#pragma unroll
    for (int mi = 0; mi < RM; ++mi) {
        float l0 = l_r[mi][0], l1 = l_r[mi][1];  // per-thread partial row sums -> quad total
        l0 += __shfl_xor_sync(0xffffffff, l0, 1);
        l0 += __shfl_xor_sync(0xffffffff, l0, 2);
        l1 += __shfl_xor_sync(0xffffffff, l1, 1);
        l1 += __shfl_xor_sync(0xffffffff, l1, 2);
        float i0 = 1.f / l0, i1 = 1.f / l1;
        int r0 = warp * 16 * RM + mi * 16 + (lane >> 2);
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            int col = u * 8 + (lane & 3) * 2;
            *reinterpret_cast<uint32_t*>(&sQ[swz(r0, col >> 3) + (col & 7)]) = pack_h2(oacc[mi][u][0] * i0, oacc[mi][u][1] * i0);
            *reinterpret_cast<uint32_t*>(&sQ[swz(r0 + 8, col >> 3) + (col & 7)]) = pack_h2(oacc[mi][u][2] * i1, oacc[mi][u][3] * i1);
        }
    }
    __syncthreads();
    __half* ob = o + b * osb + (long long)h * osh;
#pragma unroll
    for (int i = 0; i < BR * 8 / 128; ++i) {
        int c = tid + i * 128, row = c >> 3, ch = c & 7;
        int n = q0 + row;
        if (n < N) *reinterpret_cast<uint4*>(ob + (long long)n * osn + ch * 8) = *reinterpret_cast<const uint4*>(&sQ[swz(row, ch)]);
    }
}


extern "C" __global__ void __launch_bounds__(128) flash64_r2_hh(const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v, __half *__restrict__ o, int N, int H, long long qsb, int qsn, int qsh, long long ksb, int ksn, int ksh, long long vsb, int vsn, int vsh, long long osb, int osn, int osh, float scale_log2) { flash_fwd_impl<2, true, true>(q, k, v, o, N, H, qsb, qsn, qsh, ksb, ksn, ksh, vsb, vsn, vsh, osb, osn, osh, scale_log2); }
extern "C" __global__ void __launch_bounds__(128) flash64_r2_fh(const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v, __half *__restrict__ o, int N, int H, long long qsb, int qsn, int qsh, long long ksb, int ksn, int ksh, long long vsb, int vsn, int vsh, long long osb, int osn, int osh, float scale_log2) { flash_fwd_impl<2, false, true>(q, k, v, o, N, H, qsb, qsn, qsh, ksb, ksn, ksh, vsb, vsn, vsh, osb, osn, osh, scale_log2); }
extern "C" __global__ void __launch_bounds__(128) flash64_r2_ff(const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v, __half *__restrict__ o, int N, int H, long long qsb, int qsn, int qsh, long long ksb, int ksn, int ksh, long long vsb, int vsn, int vsh, long long osb, int osn, int osh, float scale_log2) { flash_fwd_impl<2, false, false>(q, k, v, o, N, H, qsb, qsn, qsh, ksb, ksn, ksh, vsb, vsn, vsh, osb, osn, osh, scale_log2); }
extern "C" __global__ void __launch_bounds__(128) flash64_r1_hh(const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v, __half *__restrict__ o, int N, int H, long long qsb, int qsn, int qsh, long long ksb, int ksn, int ksh, long long vsb, int vsn, int vsh, long long osb, int osn, int osh, float scale_log2) { flash_fwd_impl<1, true, true>(q, k, v, o, N, H, qsb, qsn, qsh, ksb, ksn, ksh, vsb, vsn, vsh, osb, osn, osh, scale_log2); }
extern "C" __global__ void __launch_bounds__(128) flash64_r1_fh(const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v, __half *__restrict__ o, int N, int H, long long qsb, int qsn, int qsh, long long ksb, int ksn, int ksh, long long vsb, int vsn, int vsh, long long osb, int osn, int osh, float scale_log2) { flash_fwd_impl<1, false, true>(q, k, v, o, N, H, qsb, qsn, qsh, ksb, ksn, ksh, vsb, vsn, vsh, osb, osn, osh, scale_log2); }
extern "C" __global__ void __launch_bounds__(128) flash64_r1_ff(const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v, __half *__restrict__ o, int N, int H, long long qsb, int qsn, int qsh, long long ksb, int ksn, int ksh, long long vsb, int vsn, int vsh, long long osb, int osn, int osh, float scale_log2) { flash_fwd_impl<1, false, false>(q, k, v, o, N, H, qsb, qsn, qsh, ksb, ksn, ksh, vsb, vsn, vsh, osb, osn, osh, scale_log2); }

// ------------------------------------------------------------------------------------------------ instance norm NHWC
// x [N, HW, C] (channels-last), C % 8 == 0. Stats in one read: per block, sums of (x - a) and (x - a)^2 around the block's
// first value a (no cancellation) -> (mean, M2); blocks merged in a fixed order with Chan's formula (in_stats_final)
// -> st[N, C, 2] = (mean, rstd). Deterministic, no atomics. Apply: y = (x - mean) * rstd (* w + b) (relu) -> same layout, in place allowed.
#include <cuda_bf16.h>
template <typename T> __device__ __forceinline__ float2 to_f2(uint32_t v);
template <> __device__ __forceinline__ float2 to_f2<__half>(uint32_t v) { return __half22float2(*reinterpret_cast<__half2*>(&v)); }
template <> __device__ __forceinline__ float2 to_f2<__nv_bfloat16>(uint32_t v) { return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v)); }
template <typename T> __device__ __forceinline__ uint32_t from_f2(float a, float b);
template <> __device__ __forceinline__ uint32_t from_f2<__half>(float a, float b) { __half2 h = __floats2half2_rn(a, b); return *reinterpret_cast<uint32_t*>(&h); }
template <> __device__ __forceinline__ uint32_t from_f2<__nv_bfloat16>(float a, float b) { __nv_bfloat162 h = __floats2bfloat162_rn(a, b); return *reinterpret_cast<uint32_t*>(&h); }

// merge (nb, mb, qb) into (na, ma, qa): Chan et al. (count, mean, sum of squared deviations)
__device__ __forceinline__ void chan_merge(float& na, float& ma, float& qa, float nb, float mb, float qb) {
    const float n = na + nb;
    if (nb == 0.f) return;
    const float d = mb - ma, f = nb / n;
    ma += d * f;
    qa += qb + d * d * na * f;
    na = n;
}

// grid (blocks over HW, N), 256 threads; thread t: channel group g = t % G (G = C/8), row r0 = t / G, rows step 256/G.
// Sums of (x - a) and (x - a)^2 around a = the block's first row (a value of the same channel, so no cancellation), reduced
// in shared memory in a fixed order -> partial [N, gridDim.x, C, 2] = per block (mean, M2); the block's count is
// min(rpb, HW - blockIdx.x * rpb).
template <typename T>
__device__ void in_stats_impl(const T* __restrict__ x, float* __restrict__ partial, int HW, int C, int rows_per_block) {
    extern __shared__ float sh[];  // [256][16] partial sums
    const int G = C >> 3, t = threadIdx.x, n = blockIdx.y;
    const int R = 256 / G;  // rows in flight per iteration (G <= 256)
    const int g = t % G, r0 = t / G;
    const int beg = blockIdx.x * rows_per_block, end = min(HW, beg + rows_per_block);
    if (beg >= end) return;  // empty tail block (in_stats_final skips it)
    const T* xn = x + (size_t)n * HW * C + g * 8;
    float a[8], s[8], q[8];
    {
        uint4 v = *reinterpret_cast<const uint4*>(xn + (size_t)beg * C);
        uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; i++) { float2 f = to_f2<T>(w[i]); a[2 * i] = f.x; a[2 * i + 1] = f.y; }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) s[i] = q[i] = 0.f;
    if (r0 < R) {
        for (int r = beg + r0; r < end; r += R) {
            uint4 v = *reinterpret_cast<const uint4*>(xn + (size_t)r * C);
            uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 f = to_f2<T>(w[i]);
                const float dx = f.x - a[2 * i], dy = f.y - a[2 * i + 1];
                s[2 * i] += dx; s[2 * i + 1] += dy;
                q[2 * i] = fmaf(dx, dx, q[2 * i]); q[2 * i + 1] = fmaf(dy, dy, q[2 * i + 1]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) { sh[t * 16 + i] = s[i]; sh[t * 16 + 8 + i] = q[i]; }
    __syncthreads();
    if (t < G) {  // thread g sums the R partials of its group
        float sa[16];
#pragma unroll
        for (int i = 0; i < 16; i++) sa[i] = 0.f;
        for (int r = 0; r < R; r++) {
            const float* p = sh + (r * G + t) * 16;
#pragma unroll
            for (int i = 0; i < 16; i++) sa[i] += p[i];
        }
        const float inv = 1.f / (float)(end - beg);
        float* o = partial + (((size_t)n * gridDim.x + blockIdx.x) * C + t * 8) * 2;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float m = sa[i] * inv;
            o[2 * i] = a[i] + m;
            o[2 * i + 1] = fmaxf(sa[8 + i] - sa[i] * m, 0.f);
        }
    }
}

// one warp per (n, c): lane l merges blocks l, l + 32, ... in order, then a fixed shuffle tree -> st[n, c] = (mean, rstd)
// nc32 / nc16 (optional, batch 1): the same statistics in fastbb.normconv's layout, fp32 [2, C] (rstd; mean) and fp16 [2, C]
// (rstd; -mean * rstd)
extern "C" __global__ void __launch_bounds__(256) in_stats_final(const float* __restrict__ partial, float* __restrict__ st, int N,
                                                                 int nb, int HW, int C, int rpb, float eps,
                                                                 float* __restrict__ nc32, __half* __restrict__ nc16) {
    const int i = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (i >= N * C) return;
    const int n = i / C, c = i % C;
    float na = 0.f, ma = 0.f, qa = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const int cnt = min(rpb, HW - b * rpb);
        if (cnt <= 0) break;
        const float* p = partial + (((size_t)n * nb + b) * C + c) * 2;
        chan_merge(na, ma, qa, (float)cnt, p[0], p[1]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float nb2 = __shfl_down_sync(0xffffffffu, na, o), mb = __shfl_down_sync(0xffffffffu, ma, o),
                    qb = __shfl_down_sync(0xffffffffu, qa, o);
        if (lane < o) chan_merge(na, ma, qa, nb2, mb, qb);
    }
    if (lane == 0) {
        const float rstd = rsqrtf(qa / (float)HW + eps);
        st[2 * i] = ma;
        st[2 * i + 1] = rstd;
        if (nc32) {
            nc32[i] = rstd; nc32[C + i] = ma;
            nc16[i] = __float2half(rstd); nc16[C + i] = __float2half(-ma * rstd);
        }
    }
}

// grid-stride over N*HW*C/8 vectors
template <typename T>
__device__ void in_apply_impl(const T* __restrict__ x, T* __restrict__ y, const float* __restrict__ st, const float* __restrict__ w,
                              const float* __restrict__ b, const T* __restrict__ res, int N, int HW, int C, float eps, int relu) {
    const int G = C >> 3;
    const size_t total = (size_t)N * HW * G;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int g = (int)(i % G);
        const int n = (int)(i / ((size_t)HW * G));
        uint4 v = reinterpret_cast<const uint4*>(x)[i];
        uint32_t u[4] = {v.x, v.y, v.z, v.w};
        uint4 rv = make_uint4(0, 0, 0, 0);
        if (relu == 2) rv = reinterpret_cast<const uint4*>(res)[i];
        const uint32_t ru[4] = {rv.x, rv.y, rv.z, rv.w};
        const float* sp = st + ((size_t)n * C + g * 8) * 2;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            float2 f = to_f2<T>(u[k]);
            float2 fr = to_f2<T>(ru[k]);
            float r[2] = {f.x, f.y}, rr[2] = {fr.x, fr.y};
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int c = 2 * k + j;
                r[j] = (r[j] - sp[2 * c]) * sp[2 * c + 1];
                if (w) r[j] = r[j] * w[g * 8 + c] + b[g * 8 + c];
                if (relu) r[j] = fmaxf(r[j], 0.f);
                if (relu == 2) r[j] = fmaxf(r[j] + rr[j], 0.f);  // RAFT ResidualBlock tail: relu(res + relu(norm(y)))
            }
            u[k] = from_f2<T>(r[0], r[1]);
        }
        reinterpret_cast<uint4*>(y)[i] = make_uint4(u[0], u[1], u[2], u[3]);
    }
}
extern "C" __global__ void __launch_bounds__(256) in_stats_h(const __half* x, float* partial, int HW, int C, int rpb) { in_stats_impl<__half>(x, partial, HW, C, rpb); }
extern "C" __global__ void __launch_bounds__(256) in_stats_b(const __nv_bfloat16* x, float* partial, int HW, int C, int rpb) { in_stats_impl<__nv_bfloat16>(x, partial, HW, C, rpb); }
extern "C" __global__ void __launch_bounds__(256) in_apply_h(const __half* x, __half* y, const float* st, const float* w, const float* b, const __half* res, int N, int HW, int C, float eps, int relu) { in_apply_impl<__half>(x, y, st, w, b, res, N, HW, C, eps, relu); }
extern "C" __global__ void __launch_bounds__(256) in_apply_b(const __nv_bfloat16* x, __nv_bfloat16* y, const float* st, const float* w, const float* b, const __nv_bfloat16* res, int N, int HW, int C, float eps, int relu) { in_apply_impl<__nv_bfloat16>(x, y, st, w, b, res, N, HW, C, eps, relu); }

// ------------------------------------------------------------------------------------------------ window attention + RPE
// NMRF / BridgeDepth WindowAttention (Swin-style windows of ws x ws cells x n candidates = T tokens, head_dim 32) with
// relative-position embeddings on q, k and v, from our NMRF fork (nmrf_kernels.cu win_rpe_attn). Same math as upstream:
//   logit_ij = q_i . k_j + q_i . Ek[r(ci,cj)] + k_j . Eq[r(ci,cj)] + mask[w % nW][i][j]      (q, Eq scaled)
//   out_i    = sum_j softmax_j(logit_ij) * (v_j + Ev[r(ci,cj)])
// with r = (hi - hj + ws - 1) * (2 ws - 1) + (wi - wj + ws - 1) (Swin relative_position_index), cell c = token / n.
// fp16 in / out, fp32 math. Tokens are read straight from the packed qkv projection and written in place of the output
// (no window partition / cyclic shift copies): token i of window w = (b, wy, wx) is grid cell
//   ((wy * ws + cy + shift) % Hc, (wx * ws + cx + shift) % Wc) of image b, candidate i % n  (cell (cy, cx) = i / n),
// row ((b * Hc + y) * Wc + x) * n + cand of qkv [rows, ld_in] (q | k | v at columns h*32, C + h*32, 2C + h*32) and of
// out [rows, ld_out] (column h*32). Hc = Wc = ws, shift 0 gives windows stored one after another (row w * T + i).
// One CTA per (wpc windows, head), looping over `wloop` such groups (the head's embedding table is loaded once); one
// thread per query token; K, V and the table in shared memory, read as half2. mask: fp32 [nW, T, T] or null.
#define WRA_D2 16
extern "C" __global__ void win_rpe_attn(const __half* __restrict__ qkv, int ld_in, int C, const __half* __restrict__ table,
                                        int tstride, const float* __restrict__ mask, int nW, int heads, int T, int ws, int n,
                                        int Hc, int Wc, int shift, float scale, __half* __restrict__ out, int ld_out,
                                        int nwin, int wpc, int wloop) {
    extern __shared__ __half2 wra_smem[];
    const int R = (2 * ws - 1) * (2 * ws - 1);
    __half2* se = wra_smem;              // [R, 3 * D2]: Eq | Ek | Ev of this head
    __half2* sk = se + R * 3 * WRA_D2;   // [wpc, T, D2]
    __half2* sv = sk + wpc * T * WRA_D2; // [wpc, T, D2]
    const int h = blockIdx.y;
    for (int x = threadIdx.x; x < R * 3 * WRA_D2; x += blockDim.x) {
        int r = x / (3 * WRA_D2), c = x - r * 3 * WRA_D2;
        se[x] = reinterpret_cast<const __half2*>(table + (long long)r * tstride + h * 96)[c];
    }
    const int nwy = Hc / ws, nwx = Wc / ws;
    const int ld_in2 = ld_in / 2, ld_out2 = ld_out / 2, coff2 = C / 2, hoff2 = h * WRA_D2;
    auto row_of = [&](int w, int i) -> long long {
        int b = w / (nwy * nwx), r = w - b * nwy * nwx;
        int wy = r / nwx, wx = r - wy * nwx;
        int cell = i / n, cand = i - cell * n;
        int cy = cell / ws, cx = cell - cy * ws;
        int y = wy * ws + cy + shift, x = wx * ws + cx + shift;
        if (y >= Hc) y -= Hc;
        if (x >= Wc) x -= Wc;
        return (((long long)b * Hc + y) * Wc + x) * n + cand;
    };
    const __half2* qkv2 = reinterpret_cast<const __half2*>(qkv);
    for (int it = 0; it < wloop; ++it) {
        const int w0 = (blockIdx.x * wloop + it) * wpc;
        if (w0 >= nwin) break;
        if (it > 0) __syncthreads();  // previous group's K / V reads are done
        for (int x = threadIdx.x; x < wpc * T * WRA_D2; x += blockDim.x) {
            int wl = x / (T * WRA_D2), rem = x - wl * T * WRA_D2;
            int j = rem / WRA_D2, e = rem - j * WRA_D2;
            int w = w0 + wl;
            if (w < nwin) {
                long long base = row_of(w, j) * ld_in2 + hoff2 + e;
                sk[x] = qkv2[base + coff2];
                sv[x] = qkv2[base + 2 * coff2];
            }
        }
        __syncthreads();
        const int wl = threadIdx.x / T, i = threadIdx.x - wl * T;
        const int w = w0 + wl;
        if (wl >= wpc || w >= nwin) continue;
        const long long row = row_of(w, i);
        float2 qi[WRA_D2];
#pragma unroll
        for (int e = 0; e < WRA_D2; ++e) {
            float2 t = __half22float2(qkv2[row * ld_in2 + hoff2 + e]);
            qi[e] = make_float2(t.x * scale, t.y * scale);
        }
        const int ci = i / n, hi = ci / ws, wi = ci - hi * ws;
        const float* mrow = mask ? mask + ((long long)(w % nW) * T + i) * T : nullptr;
        const __half2* kw = sk + wl * T * WRA_D2;
        const __half2* vw = sv + wl * T * WRA_D2;
        float m = -INFINITY, s = 0.f;
        float2 acc[WRA_D2];
#pragma unroll
        for (int e = 0; e < WRA_D2; ++e) acc[e] = make_float2(0.f, 0.f);
        for (int j = 0; j < T; ++j) {
            float l = mrow ? mrow[j] : 0.f;
            if (l == -INFINITY) continue;
            const int cj = j / n, hj = cj / ws, wj = cj - hj * ws;
            const int r = (hi - hj + ws - 1) * (2 * ws - 1) + (wi - wj + ws - 1);
            const __half2* eq = se + r * 3 * WRA_D2;
            const __half2* ek = eq + WRA_D2;
            const __half2* kj = kw + j * WRA_D2;
            float a = 0.f, b = 0.f;
#pragma unroll
            for (int e = 0; e < WRA_D2; ++e) {
                float2 kv = __half22float2(kj[e]);
                float2 kk = __half22float2(ek[e]);
                float2 qq = __half22float2(eq[e]);
                a += qi[e].x * (kv.x + kk.x) + qi[e].y * (kv.y + kk.y);
                b += kv.x * qq.x + kv.y * qq.y;
            }
            l += a + b * scale;
            float mn = fmaxf(m, l);
            float c = __expf(m - mn), p = __expf(l - mn);
            s = s * c + p;
            const __half2* ev = eq + 2 * WRA_D2;
            const __half2* vj = vw + j * WRA_D2;
#pragma unroll
            for (int e = 0; e < WRA_D2; ++e) {
                float2 vv = __half22float2(vj[e]);
                float2 ee = __half22float2(ev[e]);
                acc[e].x = acc[e].x * c + p * (vv.x + ee.x);
                acc[e].y = acc[e].y * c + p * (vv.y + ee.y);
            }
            m = mn;
        }
        const float inv = 1.f / s;
        __half2* o = reinterpret_cast<__half2*>(out) + row * ld_out2 + hoff2;
#pragma unroll
        for (int e = 0; e < WRA_D2; ++e) o[e] = __floats2half2_rn(acc[e].x * inv, acc[e].y * inv);
    }
}

// ------------------------------------------------------------------------------------------------ Restormer blocks (NHWC)
// From fastbb.normconv / fastbb.fused (exp 006, used by our DLNR fork's Restormer extractor).
// LayerNorm over the last (channel) dim of a contiguous [M, C] fp16 tensor, C % 8 == 0, C <= 2048; fp32 two-pass in
// registers. mode 0: (x - mean) * rstd * w + b (b optional); mode 1 (Restormer BiasFree): x * rstd * w (rstd from the
// variance about the mean). G = threads per row (power of two <= 32, one 16-byte vector each, VPT vectors per thread);
// launch ceil(M * G / 256) blocks of 256.
template <int VPT>
__device__ __forceinline__ void ln_impl(const __half* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                        __half* __restrict__ y, long long M, int C, float eps, int mode, int G) {
    int lane_in_row = threadIdx.x % G;
    long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / G;
    bool valid = row < M;
    float v[VPT][8];
    float s = 0.f;
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        int c = (lane_in_row + u * G) * 8;
        if (valid && c < C) {
            uint4 q = __ldg(reinterpret_cast<const uint4*>(x + row * C + c));
            const __half* hq = reinterpret_cast<const __half*>(&q);
#pragma unroll
            for (int k = 0; k < 8; ++k) { v[u][k] = __half2float(hq[k]); s += v[u][k]; }
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) v[u][k] = 0.f;
        }
    }
    for (int o = G / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o, G);
    float mean = s / C, ss = 0.f;
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        int c = (lane_in_row + u * G) * 8;
        if (c < C)
#pragma unroll
            for (int k = 0; k < 8; ++k) { float d = v[u][k] - mean; ss += d * d; }
    }
    for (int o = G / 2; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o, G);
    float rstd = rsqrtf(ss / C + eps);
    if (!valid) return;
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        int c = (lane_in_row + u * G) * 8;
        if (c >= C) continue;
        __align__(16) __half o8[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float r = mode == 0 ? (v[u][k] - mean) * rstd * w[c + k] + (b ? b[c + k] : 0.f) : v[u][k] * rstd * w[c + k];
            o8[k] = __float2half(r);
        }
        *reinterpret_cast<uint4*>(y + row * C + c) = *reinterpret_cast<const uint4*>(o8);
    }
}
#define LN_K(V) extern "C" __global__ void __launch_bounds__(256) ln_lastdim_v##V(const __half* x, const float* w, \
    const float* b, __half* y, long long M, int C, float eps, int mode, int G) { ln_impl<V>(x, w, b, y, M, C, eps, mode, G); }
LN_K(1)
LN_K(2)
LN_K(4)
LN_K(8)
#undef LN_K

// Restormer GDFN core: depthwise 3x3 (stride 1, zero pad 1) over 2h channels + GELU(erf) gate
//   y[n, p, c] = gelu(dw(x)[c]) * dw(x)[c + h],  x [N, H, W, 2h] NHWC fp16, w fp32 [9][2h], bias fp32 [2h] or null
// one thread = V channels (h % V == 0) x TW = 4 consecutive output pixels of a row; consecutive threads = consecutive
// channel groups (coalesced); fp32 math. Launch ceil(N * H * ceil(W / 4) * h / V / 128) blocks of 128.
template <int V, int TW>
__device__ __forceinline__ void dwgate_impl(const __half* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                            __half* __restrict__ y, int N, int H, int W, int h) {
    int hg = h / V, Wt = (W + TW - 1) / TW;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * H * Wt * hg;
    if (i >= total) return;
    int c = V * (int)(i % hg);
    long long t = i / hg;
    int x0 = (int)(t % Wt) * TW;
    t /= Wt;
    int yh = (int)(t % H);
    long long n = t / H;
    int C2 = 2 * h;
    float wa[9][V], wb[9][V];
#pragma unroll
    for (int k = 0; k < 9; ++k)
#pragma unroll
        for (int v = 0; v < V; ++v) { wa[k][v] = __ldg(w + k * C2 + c + v); wb[k][v] = __ldg(w + k * C2 + c + h + v); }
    float acc_a[TW][V], acc_b[TW][V];
#pragma unroll
    for (int j = 0; j < TW; ++j)
#pragma unroll
        for (int v = 0; v < V; ++v) {
            acc_a[j][v] = b ? __ldg(b + c + v) : 0.f;
            acc_b[j][v] = b ? __ldg(b + c + h + v) : 0.f;
        }
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        int yy = yh + r - 1;
        if (yy < 0 || yy >= H) continue;
        const __half* row = x + ((n * H + yy) * W) * (long long)C2;
#pragma unroll
        for (int q = 0; q < TW + 2; ++q) {
            int xx = x0 + q - 1;
            if (xx < 0 || xx >= W) continue;
            float va[V], vb[V];
            if (V == 4) {
                uint2 ra = *reinterpret_cast<const uint2*>(row + (long long)xx * C2 + c);
                uint2 rb = *reinterpret_cast<const uint2*>(row + (long long)xx * C2 + c + h);
                const __half* ha = reinterpret_cast<const __half*>(&ra);
                const __half* hb = reinterpret_cast<const __half*>(&rb);
#pragma unroll
                for (int v = 0; v < V; ++v) { va[v] = __half2float(ha[v]); vb[v] = __half2float(hb[v]); }
            } else if (V == 2) {
                float2 fa = __half22float2(*reinterpret_cast<const __half2*>(row + (long long)xx * C2 + c));
                float2 fb = __half22float2(*reinterpret_cast<const __half2*>(row + (long long)xx * C2 + c + h));
                va[0] = fa.x; va[V - 1] = fa.y; vb[0] = fb.x; vb[V - 1] = fb.y;
            } else {
                va[0] = __half2float(row[(long long)xx * C2 + c]);
                vb[0] = __half2float(row[(long long)xx * C2 + c + h]);
            }
#pragma unroll
            for (int j = 0; j < TW; ++j) {
                int s = q - j;  // tap column of output x0 + j
                if (s < 0 || s > 2) continue;
#pragma unroll
                for (int v = 0; v < V; ++v) {
                    acc_a[j][v] += va[v] * wa[r * 3 + s][v];
                    acc_b[j][v] += vb[v] * wb[r * 3 + s][v];
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TW; ++j) {
        if (x0 + j >= W) break;
        long long po = ((n * H + yh) * W + x0 + j) * (long long)h + c;
        float o[V];
#pragma unroll
        for (int v = 0; v < V; ++v) {
            float a = acc_a[j][v];
            o[v] = 0.5f * a * (1.f + erff(a * 0.70710678118654752f)) * acc_b[j][v];
        }
        if (V == 4) {
            __align__(8) __half o4[4];
#pragma unroll
            for (int v = 0; v < V; ++v) o4[v] = __float2half(o[v]);
            *reinterpret_cast<uint2*>(y + po) = *reinterpret_cast<const uint2*>(o4);
        } else if (V == 2) {
            *reinterpret_cast<__half2*>(y + po) = __floats2half2_rn(o[0], o[V - 1]);
        } else {
            y[po] = __float2half(o[0]);
        }
    }
}
#define DWG_K(V) extern "C" __global__ void __launch_bounds__(128) dwgate_v##V(const __half* x, const float* w, \
    const float* b, __half* y, int N, int H, int W, int h) { dwgate_impl<V, 4>(x, w, b, y, N, H, W, h); }
DWG_K(1)
DWG_K(2)
DWG_K(4)
#undef DWG_K

// depthwise 3x3 conv (stride 1, zero pad 1, groups == C), NHWC fp16 [N, H, W, C], C % 8 == 0, weight fp16 [3, 3, C],
// bias fp16 [C] or null; one thread = 8 channels x 2 consecutive output columns, fp32 accumulation.
// Launch ceil(N * H * ceil(W / 2) * C / 8 / 128) blocks of 128.
extern "C" __global__ void __launch_bounds__(128) dwconv3_nhwc(const __half* __restrict__ x, const __half* __restrict__ w,
                                                               const __half* __restrict__ b, __half* __restrict__ y,
                                                               int N, int H, int W, int C8) {
    constexpr int TW = 2;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int QT = (W + TW - 1) / TW;
    long long total = (long long)N * H * QT * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int qt = t % QT; t /= QT; int p = t % H; int n = t / H;
    int q0 = qt * TW;
    float acc[TW][8], bb[8];
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
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        int hh = p - 1 + r; if (hh < 0 || hh >= H) continue;
        const __half* xrow = x + ((long long)n * H + hh) * W * C8 * 8;
#pragma unroll
        for (int l = 0; l < TW + 2; ++l) {
            int ww = q0 - 1 + l; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(xrow + (long long)ww * C8 * 8) + c);
            float xv[8];
            const __half2* hv = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int k = 0; k < 4; ++k) { float2 f = __half22float2(hv[k]); xv[2 * k] = f.x; xv[2 * k + 1] = f.y; }
#pragma unroll
            for (int j = 0; j < TW; ++j) {
                int s2 = l - j;
                if (s2 < 0 || s2 >= 3) continue;
                uint4 g = __ldg(reinterpret_cast<const uint4*>(w + (r * 3 + s2) * C8 * 8) + c);
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
        int q = q0 + j; if (q >= W) break;
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __float2half(acc[j][k]);
        reinterpret_cast<uint4*>(y + (((long long)n * H + p) * W + q) * C8 * 8)[c] = *reinterpret_cast<uint4*>(o);
    }
}

// DINOv3 RoPE, in place on q and k of a packed qkv buffer [B, N, 3, H, 64] fp16 (patch tokens only; the first P tokens
// = cls + registers are not rotated). cs: fp32 table [N - P, 2, 32] = cos | sin (dims j and j + 32 share an angle).
// qs: factor applied to q of every token (the softmax scale 2^-3: exact in fp16; call the attention with scale 1).
// out[j] = x[j] c - x[j + 32] s, out[j + 32] = x[j + 32] c + x[j] s (= x cos + rotate_half(x) sin), in fp32.
extern "C" __global__ void __launch_bounds__(256) rope_qk_h(__half* qkv, const float* cs, int N, int P, int H, int total, float qs) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        int jj = i & 15;  // 16 half2 pairs per half row
        int r = i >> 4;
        int h = r % H; r /= H;
        int t = r & 1; r >>= 1;  // 0 = q, 1 = k
        int n = r % N; int b = r / N; int np = n - P;
        long long base = ((((long long)b * N + n) * 3 + t) * H + h) * 64;
        __half2* p1 = reinterpret_cast<__half2*>(qkv + base) + jj;
        __half2* p2 = reinterpret_cast<__half2*>(qkv + base + 32) + jj;
        float2 x1 = __half22float2(*p1), x2 = __half22float2(*p2);
        if (np < 0) {
            if (t == 0) { *p1 = __floats2half2_rn(x1.x * qs, x1.y * qs); *p2 = __floats2half2_rn(x2.x * qs, x2.y * qs); }
            continue;
        }
        float2 c = *(reinterpret_cast<const float2*>(cs + (long long)np * 64) + jj);
        float2 s = *(reinterpret_cast<const float2*>(cs + (long long)np * 64 + 32) + jj);
        float m = t ? 1.f : qs;
        *p1 = __floats2half2_rn((x1.x * c.x - x2.x * s.x) * m, (x1.y * c.y - x2.y * s.y) * m);
        *p2 = __floats2half2_rn((x2.x * c.x + x1.x * s.x) * m, (x2.y * c.y + x1.y * s.y) * m);
    }
}
