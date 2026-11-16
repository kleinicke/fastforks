// Forward-only 3D Gaussian splatting rasteriser (feed-forward Gaussian families): gsplat's EWA projection with the
// antialiased 2D blur, degree-3 SH colours, tile intersection keys, and the front-to-back tile compositing.
// Derived from gsplat (Apache-2.0, nerfstudio-project/gsplat: fully_fused_projection, isect_tiles,
// rasterize_to_pixels_3dgs_fwd) and our exp 001 fork's 4-pixels-per-thread forward (CTA 64).
//
// Pipeline (fastforks.kernels.gsraster.render): gsr_project (one thread per view x Gaussian) -> exclusive cumsum of
// the tile counts (torch) -> gsr_isect (one key per Gaussian x tile: tile id << 32 | depth bits) -> torch.sort ->
// per-tile offsets (torch.searchsorted) -> gsr_rasterize (one block per tile).
#include <stdint.h>

#define TILE 16
#define CTA 64                     // threads per tile block: 4 pixels per thread
#define PPT (TILE * TILE / CTA)    // pixels per thread
#define ROW_STRIDE (CTA / TILE)
#define SH 16
#define EPS2D 0.1f                 // upstream's eps2d (antialiased mode)
#define ALPHA_MIN (1.0f / 255.0f)  // gsplat ALPHA_THRESHOLD
#define ALPHA_MAX 0.99f            // gsplat MAX_ALPHA
#define T_MIN 1e-4f                // gsplat TRANSMITTANCE_THRESHOLD
#define EXTEND 3.33f               // gsplat GAUSSIAN_EXTEND
#define MIN_COMP 0.005f            // gsplat MIN_COMPENSATION

struct Bbox { int x0, y0, nx, ny; };

// Degree-3 real SH (gsplat _eval_sh_bases_fast) dotted with 16 coefficients.
__device__ __forceinline__ float sh_eval(const float* __restrict__ c, float x, float y, float z) {
    const float z2 = z * z;
    const float c1 = x * x - y * y, s1 = 2.f * x * y;
    const float c2 = x * c1 - y * s1, s2 = x * s1 + y * c1;
    const float a = -0.48860251190292f, b = -1.092548430592079f * z, cc = -2.285228997322329f * z2 + 0.4570457994644658f;
    const float e = 1.445305721320277f * z;
    float r = 0.2820947917738781f * c[0];
    r += a * y * c[1] - a * z * c[2] + a * x * c[3];
    r += 0.5462742152960395f * s1 * c[4] + b * y * c[5] + (0.9461746957575601f * z2 - 0.3153915652525201f) * c[6]
         + b * x * c[7] + 0.5462742152960395f * c1 * c[8];
    r += -0.5900435899266435f * s2 * c[9] + e * s1 * c[10] + cc * y * c[11]
         + z * (1.865881662950577f * z2 - 1.119528997770346f) * c[12] + cc * x * c[13] + e * c1 * c[14]
         - 0.5900435899266435f * c2 * c[15];
    return r;
}

// One thread per (view, Gaussian): EWA projection, colour, tile bounding box. Invalid Gaussians get cnt 0.
extern "C" __global__ void __launch_bounds__(256) gsr_project(
    const float* __restrict__ means,   // [N][3] world
    const float* __restrict__ cov,     // [N][3][3] world (upper triangle read)
    const float* __restrict__ opac,    // [N]
    const float* __restrict__ sh,      // [N][3][16]
    const float* __restrict__ w2c,     // [V][4][4]
    const float* __restrict__ kmat,    // [V][3][3] pixel units
    const float* __restrict__ campos,  // [V][3] camera centres (world)
    float* __restrict__ m2d,           // [V*N][2]
    float* __restrict__ conic,         // [V*N][3]
    float* __restrict__ opac_o,        // [V*N] compensated opacity
    float* __restrict__ color,         // [V*N][3]
    float* __restrict__ depth,         // [V*N]
    Bbox* __restrict__ bbox,           // [V*N]
    int* __restrict__ cnt,             // [V*N] tiles touched
    int n, int v, int h, int w, float near, float far)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n * v) return;
    const int i = gid / n, g = gid - i * n;
    const float* P = w2c + i * 16;
    const float mx = means[3 * g], my = means[3 * g + 1], mz = means[3 * g + 2];
    const float x = P[0] * mx + P[1] * my + P[2] * mz + P[3];
    const float y = P[4] * mx + P[5] * my + P[6] * mz + P[7];
    const float z = P[8] * mx + P[9] * my + P[10] * mz + P[11];
    const float* C = cov + 9 * g;
    const float c00 = C[0], c01 = C[1], c02 = C[2], c11 = C[4], c12 = C[5], c22 = C[8];
    // RC = R @ C, CC = RC @ R^T
    float rc[3][3];
#pragma unroll
    for (int r = 0; r < 3; r++) {
        const float r0 = P[4 * r], r1 = P[4 * r + 1], r2 = P[4 * r + 2];
        rc[r][0] = r0 * c00 + r1 * c01 + r2 * c02;
        rc[r][1] = r0 * c01 + r1 * c11 + r2 * c12;
        rc[r][2] = r0 * c02 + r1 * c12 + r2 * c22;
    }
    float ccm[3][3];
#pragma unroll
    for (int r = 0; r < 3; r++)
#pragma unroll
        for (int s = 0; s < 3; s++)
            ccm[r][s] = rc[r][0] * P[4 * s] + rc[r][1] * P[4 * s + 1] + rc[r][2] * P[4 * s + 2];
    const float* K = kmat + i * 9;
    const float fx = K[0], fy = K[4], cx = K[2], cy = K[5];
    const float tfx = 0.5f * w / fx, tfy = 0.5f * h / fy;
    const float tx = z * fminf(fmaxf(x / z, -(cx / fx + 0.3f * tfx)), (w - cx) / fx + 0.3f * tfx);
    const float ty = z * fminf(fmaxf(y / z, -(cy / fy + 0.3f * tfy)), (h - cy) / fy + 0.3f * tfy);
    const float rz = 1.f / z, rz2 = rz * rz;
    // J = [[fx/z, 0, -fx tx/z^2], [0, fy/z, -fy ty/z^2]]
    const float j00 = fx * rz, j02 = -fx * tx * rz2, j11 = fy * rz, j12 = -fy * ty * rz2;
    // c2 = J CC J^T
    const float t00 = j00 * ccm[0][0] + j02 * ccm[2][0], t01 = j00 * ccm[0][1] + j02 * ccm[2][1];
    const float t02 = j00 * ccm[0][2] + j02 * ccm[2][2];
    const float t10 = j11 * ccm[1][0] + j12 * ccm[2][0], t11 = j11 * ccm[1][1] + j12 * ccm[2][1];
    const float t12 = j11 * ccm[1][2] + j12 * ccm[2][2];
    float a = t00 * j00 + t02 * j02;
    const float b = t01 * j11 + t02 * j12;
    const float b2 = t10 * j00 + t12 * j02;
    float c = t11 * j11 + t12 * j12;
    const float det0 = a * c - b * b2;
    a += EPS2D;
    c += EPS2D;
    const float det = a * c - b * b2;
    const float comp = sqrtf(fmaxf(det0 / det, MIN_COMP * MIN_COMP));
    const float o = opac[g] * comp;
    const float px = fx * x * rz + cx, py = fy * y * rz + cy;
    const float ext = fminf(sqrtf(2.f * logf(fmaxf(o, ALPHA_MIN) / ALPHA_MIN)), EXTEND);
    const float rx = ceilf(ext * sqrtf(fmaxf(a, 0.f))), ry = ceilf(ext * sqrtf(fmaxf(c, 0.f)));
    bool ok = z >= near && z <= far && det > 0.f && o >= ALPHA_MIN && (rx > 0.f || ry > 0.f);
    ok = ok && px + rx > 0.f && px - rx < (float)w && py + ry > 0.f && py - ry < (float)h;
    const int tw = (w + TILE - 1) / TILE, th = (h + TILE - 1) / TILE;
    Bbox bb = {0, 0, 0, 0};
    if (ok) {
        const int x0 = max((int)floorf((px - rx) / TILE), 0), y0 = max((int)floorf((py - ry) / TILE), 0);
        const int x1 = min((int)ceilf((px + rx) / TILE), tw), y1 = min((int)ceilf((py + ry) / TILE), th);
        bb.x0 = x0;
        bb.y0 = y0;
        bb.nx = max(x1 - x0, 0);
        bb.ny = max(y1 - y0, 0);
    }
    cnt[gid] = bb.nx * bb.ny;
    bbox[gid] = bb;
    if (!ok) return;
    m2d[2 * gid] = px;
    m2d[2 * gid + 1] = py;
    conic[3 * gid] = c / det;
    conic[3 * gid + 1] = -(b + b2) * 0.5f / det;
    conic[3 * gid + 2] = a / det;
    opac_o[gid] = o;
    depth[gid] = z;
    float dx = mx - campos[3 * i], dy = my - campos[3 * i + 1], dz = mz - campos[3 * i + 2];
    const float inv = 1.f / sqrtf(dx * dx + dy * dy + dz * dz);
    dx *= inv;
    dy *= inv;
    dz *= inv;
#pragma unroll
    for (int k = 0; k < 3; k++) color[3 * gid + k] = fmaxf(sh_eval(sh + (3 * g + k) * SH, dx, dy, dz) + 0.5f, 0.f);
}

// One thread per (view, Gaussian): its intersection keys (tile id << 32 | depth bits) and values (view-Gaussian id).
extern "C" __global__ void __launch_bounds__(256) gsr_isect(
    const Bbox* __restrict__ bbox,      // [V*N]
    const float* __restrict__ depth,    // [V*N]
    const long long* __restrict__ off,  // [V*N] exclusive cumsum of cnt
    long long* __restrict__ keys,       // [n_isects]
    int* __restrict__ vals,             // [n_isects]
    int n, int v, int ntiles, int tw)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n * v) return;
    const Bbox bb = bbox[gid];
    if (bb.nx == 0 || bb.ny == 0) return;
    const int i = gid / n;
    const long long dbits = (long long)__float_as_uint(depth[gid]);
    long long o = off[gid];
    for (int y = 0; y < bb.ny; y++)
        for (int x = 0; x < bb.nx; x++) {
            const long long tile = (long long)i * ntiles + (bb.y0 + y) * tw + bb.x0 + x;
            keys[o] = (tile << 32) | dbits;
            vals[o] = gid;
            o++;
        }
}

// One block per (view, tile), 64 threads x 4 pixels; the Gaussians of the tile stream through shared memory in
// batches of 64, front to back.
extern "C" __global__ void __launch_bounds__(CTA) gsr_rasterize(
    const float* __restrict__ m2d,       // [V*N][2]
    const float* __restrict__ conic,     // [V*N][3]
    const float* __restrict__ opac,      // [V*N]
    const float* __restrict__ color,     // [V*N][3]
    const long long* __restrict__ offs,  // [V*ntiles] first intersection of every tile
    const int* __restrict__ vals,        // [n_isects] sorted view-Gaussian ids
    float* __restrict__ out,             // [V][3][h][w]
    long long n_isects, int ntiles, int tw, int h, int w)
{
    __shared__ int s_id[CTA];
    __shared__ float4 s_g[CTA];  // mean x, mean y, opacity, pad
    __shared__ float4 s_c[CTA];  // conic a, b, c, pad
    const int block = blockIdx.x;
    const int i = block / ntiles, tile = block - i * ntiles;
    const int tile_x = tile % tw, tile_y = tile / tw;
    const int tid = threadIdx.x;
    const int ox = tile_x * TILE + (tid & (TILE - 1));
    const float px = (float)ox + 0.5f;
    float py[PPT];
    int oy[PPT];
    unsigned done = ox >= w ? (1u << PPT) - 1u : 0u;
#pragma unroll
    for (int p = 0; p < PPT; p++) {
        oy[p] = tile_y * TILE + (tid >> 4) + p * ROW_STRIDE;
        py[p] = (float)oy[p] + 0.5f;
        if (oy[p] >= h) done |= 1u << p;
    }
    const long long start = offs[block];
    const long long end = block + 1 < gridDim.x ? offs[block + 1] : n_isects;
    const int nb = (int)((end - start + CTA - 1) / CTA);
    float T[PPT], acc[PPT][3];
#pragma unroll
    for (int p = 0; p < PPT; p++) {
        T[p] = 1.f;
        acc[p][0] = acc[p][1] = acc[p][2] = 0.f;
    }
    const unsigned all_done = (1u << PPT) - 1u;
#pragma unroll 1
    for (int b = 0; b < nb; b++) {
        const long long bs = start + (long long)b * CTA;
        const long long idx = bs + tid;
        if (idx < end) {
            const int g = vals[idx];
            s_id[tid] = g;
            s_g[tid] = make_float4(m2d[2 * g], m2d[2 * g + 1], opac[g], 0.f);
            s_c[tid] = make_float4(conic[3 * g], conic[3 * g + 1], conic[3 * g + 2], 0.f);
        }
        __syncthreads();
        const int bsz = (int)min((long long)CTA, end - bs);
        for (int t = 0; t < bsz && done != all_done; t++) {
            const float4 cn = s_c[t];
            const float4 gg = s_g[t];
            const float dx = gg.x - px;
#pragma unroll
            for (int p = 0; p < PPT; p++) {
                if (done & (1u << p)) continue;
                const float dy = gg.y - py[p];
                const float sigma = 0.5f * (cn.x * dx * dx + cn.z * dy * dy) + cn.y * dx * dy;
                const float alpha = fminf(ALPHA_MAX, gg.z * __expf(-sigma));
                if (sigma < 0.f || alpha < ALPHA_MIN) continue;
                const float nt = T[p] * (1.f - alpha);
                if (nt <= T_MIN) {
                    done |= 1u << p;
                    continue;
                }
                const float vis = alpha * T[p];
                const float* cp = color + 3 * s_id[t];
                acc[p][0] += cp[0] * vis;
                acc[p][1] += cp[1] * vis;
                acc[p][2] += cp[2] * vis;
                T[p] = nt;
            }
        }
        if (__syncthreads_count(done == all_done) >= CTA) break;
    }
    if (ox >= w) return;
    const long long plane = (long long)h * w;
    float* o = out + (long long)i * 3 * plane;
#pragma unroll
    for (int p = 0; p < PPT; p++) {
        if (oy[p] >= h) continue;
        const long long pix = (long long)oy[p] * w + ox;
        o[pix] = acc[p][0];
        o[plane + pix] = acc[p][1];
        o[2 * plane + pix] = acc[p][2];
    }
}

// Exact kNN of a point cloud with itself (the point transformers' neighbourhoods; was pointops.knn_query, then the
// pooled torch search of resplat's point_transformer.knn_query). The host sorts the points in z-order, packs them as
// (x, y, z, original index) and gives the bounding box of every KNN_TILE consecutive points. A block takes KNN_TILE
// consecutive queries (KNN_Q per thread, sorted lists of K in registers), scans its own tile and then the others
// alternately to the left and right, and skips a tile when its box is farther than every thread's current k-th
// distance. Squared distances are rounded like torch's (p - q).square().sum(-1) (no FMA contraction); rounding is
// monotonic, so the box bound never exceeds the rounded distance of a point inside and the skip is exact. Order is
// (distance, original index): the result is the unique k smallest, independent of the scan order.
#define KNN_Q 2
#define KNN_THREADS 128
#define KNN_TILE (KNN_Q * KNN_THREADS)

__device__ __forceinline__ float knn_d2(float dx, float dy, float dz) {
    return __fadd_rn(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)), __fmul_rn(dz, dz));
}

__device__ __forceinline__ float knn_gap(float q, float lo, float hi) {  // >= 0, rounded like the point differences
    return q < lo ? __fsub_rn(lo, q) : (q > hi ? __fsub_rn(q, hi) : 0.f);
}

template <int K>
__device__ __forceinline__ void knn_impl(
    const float4* __restrict__ pts,  // [N] z-ordered (x, y, z, original index as int bits)
    const float* __restrict__ box,   // [T, 6] lo xyz, hi xyz of each tile
    long long* __restrict__ idx,     // [N, K] by original index, nearest first
    int n) {
    __shared__ float4 sp[KNN_TILE];
    const int nt = (n + KNN_TILE - 1) / KNN_TILE, b = blockIdx.x;
    const int q0 = b * KNN_TILE + threadIdx.x;  // queries q0 + a * KNN_THREADS (coalesced)
    float qx[KNN_Q], qy[KNN_Q], qz[KNN_Q], bd[KNN_Q][K], worst[KNN_Q];
    int bi[KNN_Q][K], wi[KNN_Q];
#pragma unroll
    for (int a = 0; a < KNN_Q; a++) {
        const float4 q = pts[min(q0 + a * KNN_THREADS, n - 1)];
        qx[a] = q.x;
        qy[a] = q.y;
        qz[a] = q.z;
        worst[a] = __int_as_float(0x7f800000);
        wi[a] = 0x7fffffff;
#pragma unroll
        for (int s = 0; s < K; s++) {
            bd[a][s] = worst[a];
            bi[a][s] = 0x7fffffff;
        }
    }
    for (int step = 0; step < 2 * nt; step++) {  // b, b - 1, b + 1, b - 2, b + 2, ...
        const int t = (step & 1) ? b - (step + 1) / 2 : b + step / 2;
        if (t < 0 || t >= nt) continue;  // uniform over the block
        const float* bx = box + 6 * t;
        bool need = false;
#pragma unroll
        for (int a = 0; a < KNN_Q; a++) {
            const float g = knn_d2(knn_gap(qx[a], bx[0], bx[3]), knn_gap(qy[a], bx[1], bx[4]), knn_gap(qz[a], bx[2], bx[5]));
            need |= g <= worst[a];
        }
        if (!__syncthreads_or(need)) continue;
        const int t0 = t * KNN_TILE, m = min(KNN_TILE, n - t0);
        for (int j = threadIdx.x; j < m; j += KNN_THREADS) sp[j] = pts[t0 + j];
        __syncthreads();
        for (int j = 0; j < m; j++) {
            const float4 c = sp[j];
            const int ci = __float_as_int(c.w);
#pragma unroll
            for (int a = 0; a < KNN_Q; a++) {
                const float d = knn_d2(__fsub_rn(c.x, qx[a]), __fsub_rn(c.y, qy[a]), __fsub_rn(c.z, qz[a]));
                if (d < worst[a] || (d == worst[a] && ci < wi[a])) {
#pragma unroll
                    for (int s = K - 1; s > 0; s--) {
                        const bool after = bd[a][s - 1] > d || (bd[a][s - 1] == d && bi[a][s - 1] > ci);
                        const bool here = !after && (bd[a][s] > d || (bd[a][s] == d && bi[a][s] > ci));
                        if (after) {
                            bd[a][s] = bd[a][s - 1];
                            bi[a][s] = bi[a][s - 1];
                        } else if (here) {
                            bd[a][s] = d;
                            bi[a][s] = ci;
                        }
                    }
                    if (bd[a][0] > d || (bd[a][0] == d && bi[a][0] > ci)) {
                        bd[a][0] = d;
                        bi[a][0] = ci;
                    }
                    worst[a] = bd[a][K - 1];
                    wi[a] = bi[a][K - 1];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < KNN_Q; a++) {
        const int q = q0 + a * KNN_THREADS;
        if (q >= n) continue;
        long long* o = idx + (long long)__float_as_int(pts[q].w) * K;
#pragma unroll
        for (int s = 0; s < K; s++) o[s] = bi[a][s];
    }
}

extern "C" __global__ void __launch_bounds__(KNN_THREADS) gsr_knn8(
    const float4* __restrict__ pts, const float* __restrict__ box, long long* __restrict__ idx, int n) {
    knn_impl<8>(pts, box, idx, n);
}

extern "C" __global__ void __launch_bounds__(KNN_THREADS) gsr_knn16(
    const float4* __restrict__ pts, const float* __restrict__ box, long long* __restrict__ idx, int n) {
    knn_impl<16>(pts, box, idx, n);
}
