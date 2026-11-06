// sparse_conv: MoGe-3's sparse 3D refiner in fp16 (our code). A level's voxels are rows of an [N, C] fp16 matrix; the
// neighbour table nbr [27, N] (int32, -1 = absent) holds, per tap v of product((-1, 0, 1), repeat=3) over (di, dj, dz),
// the row of the voxel at coord + d_v. Nothing here depends on the input size.
//
// subm_conv_n{32,64}: out[n] = act(bias + sum_v W[v]^T x[nbr[v, n]] (+ res[n])), W pre-transposed to [27, C_in, C_out].
// With z (int64 bins [B, H, W]) given, nbr is unused and a tile computes its grid-level neighbours as grid_nbr does.
// Gather implicit GEMM: a CTA takes 64 voxels (rows perm[m0 ..] if perm is given: voxels sorted by their neighbour
// mask, so a tile's voxels share their taps) x 32 / 64 output channels, loops over the taps any of its voxels has and
// the input channels in steps of 32 (cp.async double buffer, absent rows zero-filled), WMMA with float32 accumulation.
// Needs C_in % 32 == 0 and C_out % 32 (n32) / 64 (n64) == 0.
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

#define NTAP 27
#define BM 64
#define BK 32

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    int n = valid ? 16 : 0;  // src-size 0 -> zero fill
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int BN, int NW, int ST>
__device__ __forceinline__ void subm_conv(const half* __restrict__ x, const int* __restrict__ nbr,
                                          const int* __restrict__ perm, const half* __restrict__ w,
                                          const half* __restrict__ bias, const half* res, half* out,
                                          const long long* __restrict__ z, int N, int Cin, int Cout, int act, int H,
                                          int W) {
    constexpr int NTH = NW * 32;       // threads
    constexpr int WN = BN / 32;        // warps along the output channels (32 channels each)
    constexpr int WMW = NW / WN;       // warps along the voxels
    constexpr int FM = BM / WMW / 16;  // 16x16 fragments per warp along the voxels (2 along the channels)
    constexpr int LDA = BK + 8, LDB = BN + 8, LDC = BN + 4;
    constexpr int A_BYTES = ST * BM * LDA * 2, B_BYTES = ST * BK * LDB * 2, C_BYTES = BM * LDC * 4;
    constexpr int SM = A_BYTES + B_BYTES > C_BYTES ? A_BYTES + B_BYTES : C_BYTES;
    __shared__ __align__(128) unsigned char smem[SM];
    __shared__ int sNbr[NTAP][BM];
    __shared__ int sVox[BM];
    __shared__ int sTaps[NTAP];
    __shared__ unsigned sMask;
    __shared__ int sNT;
    auto sA = reinterpret_cast<half(*)[BM][LDA]>(smem);
    auto sB = reinterpret_cast<half(*)[BK][LDB]>(smem + A_BYTES);
    auto sC = reinterpret_cast<float(*)[LDC]>(smem);  // after the main loop

    const int tid = threadIdx.x, warp = tid >> 5;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    if (tid == 0) sMask = 0;
    if (tid < BM) {
        int r = m0 + tid;
        sVox[tid] = r < N ? (perm ? perm[r] : r) : -1;
    }
    __syncthreads();
    unsigned mine = 0;
    if (z) {  // grid level (nbr unused): voxel = (b H + i) W + j, the neighbour pixel's bin decides dz (as grid_nbr)
        for (int i = tid; i < NTAP * BM; i += NTH) sNbr[i / BM][i % BM] = -1;
        __syncthreads();
        for (int i = tid; i < 9 * BM; i += NTH) {
            const int t = i / BM, r = i % BM, vox = sVox[r];
            if (vox < 0) continue;
            const int di = t / 3 - 1, dj = t % 3 - 1, ii = (vox / W) % H + di, jj = vox % W + dj;
            if (ii < 0 || ii >= H || jj < 0 || jj >= W) continue;
            const int s = vox + di * W + dj;
            const long long d = z[s] - z[vox];
            if (d >= -1 && d <= 1) {
                const int v = t * 3 + (int)d + 1;
                sNbr[v][r] = s;
                mine |= 1u << v;
            }
        }
    } else {
        for (int i = tid; i < NTAP * BM; i += NTH) {
            int v = i / BM, r = i % BM, vox = sVox[r];
            int s = vox >= 0 ? nbr[(long long)v * N + vox] : -1;
            sNbr[v][r] = s;
            if (s >= 0) mine |= 1u << v;
        }
    }
    mine = __reduce_or_sync(0xffffffffu, mine);
    if ((tid & 31) == 0) atomicOr(&sMask, mine);
    __syncthreads();
    if (tid == 0) {
        int n = 0;
        for (int v = 0; v < NTAP; ++v)
            if (sMask >> v & 1) sTaps[n++] = v;
        sNT = n;
    }
    __syncthreads();

    const int KC = Cin / BK, T = sNT * KC;
    auto load = [&](int it, int buf) {
        const int tap = sTaps[it / KC], k0 = (it % KC) * BK;
#pragma unroll
        for (int c = tid; c < BM * (BK / 8); c += NTH) {
            int r = c / (BK / 8), ch = c % (BK / 8), s = sNbr[tap][r];
            cp_async16(smem_u32(&sA[buf][r][ch * 8]), x + (long long)(s < 0 ? 0 : s) * Cin + k0 + ch * 8, s >= 0);
        }
#pragma unroll
        for (int c = tid; c < BK * (BN / 8); c += NTH) {
            int k = c / (BN / 8), ch = c % (BN / 8);
            cp_async16(smem_u32(&sB[buf][k][ch * 8]), w + ((long long)tap * Cin + k0 + k) * Cout + n0 + ch * 8, true);
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][2];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);
    const int wm = warp / WN, wn = warp % WN;
#pragma unroll
    for (int st = 0; st < ST - 1; ++st) {  // prefetch ST - 1 stages (empty groups past the end keep the count)
        if (st < T) load(st, st);
        cp_async_commit();
    }
    for (int it = 0; it < T; ++it) {
        if (it + ST - 1 < T) load(it + ST - 1, (it + ST - 1) % ST);
        cp_async_commit();
        cp_async_wait<ST - 1>();
        __syncthreads();
        const int buf = it % ST;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[2];
#pragma unroll
            for (int i = 0; i < FM; ++i) wmma::load_matrix_sync(a[i], &sA[buf][(wm * FM + i) * 16][kk], LDA);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &sB[buf][kk][(wn * 2 + j) * 16], LDB);
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&sC[(wm * FM + i) * 16][(wn * 2 + j) * 16], acc[i][j], LDC, wmma::mem_row_major);
    __syncthreads();
    for (int c = tid; c < BM * (BN / 8); c += NTH) {
        const int r = c / (BN / 8), ch = c % (BN / 8), vox = sVox[r];
        if (vox < 0) continue;
        const int col = n0 + ch * 8;
        float v[8];
        uint4 bb = *reinterpret_cast<const uint4*>(bias + col);
        const half* bh = reinterpret_cast<const half*>(&bb);
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = sC[r][ch * 8 + e] + __half2float(bh[e]);
        if (res) {
            uint4 rr = *reinterpret_cast<const uint4*>(res + (long long)vox * Cout + col);
            const half* rh = reinterpret_cast<const half*>(&rr);
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] += __half2float(rh[e]);
        }
        uint4 o;
        half* oh = reinterpret_cast<half*>(&o);
#pragma unroll
        for (int e = 0; e < 8; ++e) oh[e] = __float2half_rn(act ? v[e] / (1.f + __expf(-v[e])) : v[e]);
        *reinterpret_cast<uint4*>(out + (long long)vox * Cout + col) = o;
    }
}

#define SUBM(name, BN, NW, ST)                                                                                       \
    extern "C" __global__ void __launch_bounds__(NW * 32)                                                            \
        name(const half* x, const int* nbr, const int* perm, const half* w, const half* bias, const half* res,     \
             half* out, const long long* z, int N, int Cin, int Cout, int act, int H, int W) {                       \
        subm_conv<BN, NW, ST>(x, nbr, perm, w, bias, res, out, z, N, Cin, Cout, act, H, W);                          \
    }
SUBM(subm_conv_n32, 32, 4, 2)
SUBM(subm_conv_n64, 64, 4, 2)
SUBM(subm_conv_n64s3, 64, 4, 3)
SUBM(subm_conv_n128, 128, 8, 2)

// grid_nbr: the finest level, one voxel per pixel of z [B, H, W] (int64 bins), voxel = (b H + i) W + j. The pixel
// (i + di, j + dj) is the neighbour of tap (di, dj, dz) iff its bin differs by dz. mask[n]: bit v = tap v present;
// nbr may be null (the convs then take z and find the same neighbours themselves: no [27, N] table at the largest level).
extern "C" __global__ void grid_nbr(const long long* z, int* nbr, int* mask, int B, int H, int W) {
    const long long N = (long long)B * H * W;
    for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < N; n += (long long)gridDim.x * blockDim.x) {
        const int j = n % W, i = (n / W) % H;
        const long long zc = z[n];
        unsigned m = 0;
#pragma unroll
        for (int di = -1; di <= 1; ++di)
#pragma unroll
            for (int dj = -1; dj <= 1; ++dj) {
                const int t = (di + 1) * 9 + (dj + 1) * 3;
                const int ii = i + di, jj = j + dj;
                int k = -1;
                if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                    long long d = z[n + di * W + dj] - zc;
                    if (d >= -1 && d <= 1) k = (int)d + 1;
                }
                if (nbr)
#pragma unroll
                    for (int kk = 0; kk < 3; ++kk) nbr[(t + kk) * N + n] = kk == k ? (int)(n + di * W + dj) : -1;
                if (k >= 0) m |= 1u << (t + k);
            }
        mask[n] = (int)m;
    }
}

// sparse_nbr: a level from coords [N, 4] (b, i, j, z; int64) sorted by key = ((b H + i) W + j) Z + z: neighbours by
// binary search over the keys.
extern "C" __global__ void sparse_nbr(const long long* coords, const long long* key, int* nbr, int* mask, int N, int H,
                                      int W, int Z) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N; n += gridDim.x * blockDim.x) {
        const long long i = coords[4 * n + 1], j = coords[4 * n + 2], zz = coords[4 * n + 3], kc = key[n];
        unsigned m = 0;
        for (int v = 0; v < NTAP; ++v) {
            const int di = v / 9 - 1, dj = v / 3 % 3 - 1, dz = v % 3 - 1;
            int hit = -1;
            if (v == 13) {
                hit = n;
            } else if (i + di >= 0 && i + di < H && j + dj >= 0 && j + dj < W && zz + dz >= 0 && zz + dz < Z) {
                const long long want = kc + ((long long)di * W + dj) * Z + dz;
                int lo = 0, hi = N;  // first key >= want
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (key[mid] < want) lo = mid + 1; else hi = mid;
                }
                if (lo < N && key[lo] == want) hit = lo;
            }
            nbr[(long long)v * N + n] = hit;
            if (hit >= 0) m |= 1u << v;
        }
        mask[n] = (int)m;
    }
}

// ln_silu: out = silu(LayerNorm(x)) per row of x [N, C] (C % 32 == 0, C <= 512), float32 statistics; one warp per row.
extern "C" __global__ void __launch_bounds__(128) ln_silu(const half* x, const half* g, const half* b, half* out, int N,
                                                          int C, float eps) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const half* xr = x + (long long)row * C;
    float v[16], s = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k)
        if (k * 32 < C) {
            v[k] = __half2float(xr[k * 32 + lane]);
            s += v[k];
        }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / C;
    float q = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k)
        if (k * 32 < C) {
            v[k] -= mean;
            q += v[k] * v[k];
        }
#pragma unroll
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rstd = rsqrtf(q / C + eps);
    half* orow = out + (long long)row * C;
#pragma unroll
    for (int k = 0; k < 16; ++k)
        if (k * 32 < C) {
            const int c = k * 32 + lane;
            float y = v[k] * rstd * __half2float(g[c]) + __half2float(b[c]);
            orow[c] = __float2half_rn(y / (1.f + __expf(-y)));
        }
}
