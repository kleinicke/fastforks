// Essential-matrix RANSAC for `fastforks.matching.estimate_pose`: Nister's 5-point solver, one hypothesis per thread,
// and fused Sampson scoring of many candidates against the matches of a pair. Launched on torch's current stream.
//
// Matches: pts [P, N, 4] fp32 (u1, v1, u2, v2) in normalised image coordinates, the valid ones first; a launch works on
// a subset `pairs` [A] of the pairs (the unfinished ones of a RANSAC round).
//
// Precision (measured against exact synthetic data): the null space and the 10 x 20 constraint matrix in fp32, the
// Gauss-Jordan elimination, the polynomial matrix B(z), its determinant and the root polish in fp64, the root search
// (Aberth-Ehrlich) in fp32 complex. All-fp32 loses 10-20 % of the hypotheses.
#include <cuda_runtime.h>
#include <math.h>

#define NSOL 10
#define ITER_ABERTH 48

// Monomial bases of polynomials in (x, y, z): linear (x, y, z, 1), quadratic (x^2, xy, xz, x, y^2, yz, y, z^2, z, 1),
// cubic in Nister's order (x^3 y^3 x^2y xy^2 x^2z x^2 y^2z y^2 xyz xy | xz^2 xz x yz^2 yz y z^3 z^2 z 1).
__constant__ int LL[4][4] = {{0, 1, 2, 3}, {1, 4, 5, 6}, {2, 5, 7, 8}, {3, 6, 8, 9}};
__constant__ int QL[10][4] = {{0, 2, 4, 5},   {2, 3, 8, 9},    {4, 8, 10, 11},  {5, 9, 11, 12},   {3, 1, 6, 7},
                              {8, 6, 13, 14}, {9, 7, 14, 15}, {10, 13, 16, 17}, {11, 14, 17, 18}, {12, 15, 18, 19}};

__device__ __forceinline__ void mul_ll(const float* a, const float* b, float* q) {  // q[10] += a[4] * b[4]
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) q[LL[i][j]] += a[i] * b[j];
}

__device__ __forceinline__ void mul_ql(const float* q, const float* b, float s, float* c) {  // c[20] += s q[10] b[4]
#pragma unroll
  for (int i = 0; i < 10; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) c[QL[i][j]] += s * q[i] * b[j];
}

// Polynomials in z, coefficients highest power first.
template <int DA, int DB>
__device__ __forceinline__ void pmul(const double* a, const double* b, double* out) {
#pragma unroll
  for (int i = 0; i < DA + DB - 1; ++i) out[i] = 0.0;
#pragma unroll
  for (int i = 0; i < DA; ++i)
#pragma unroll
    for (int j = 0; j < DB; ++j) out[i + j] += a[i] * b[j];
}

template <int D>
__device__ __forceinline__ double peval(const double* p, double z) {
  double v = p[0];
#pragma unroll
  for (int i = 1; i < D; ++i) v = v * z + p[i];
  return v;
}

// Null space of the 5 x 9 epipolar constraint rows: orthonormalise the rows (modified Gram-Schmidt), then the four
// unit vectors with the largest residual after projecting out the row span, orthonormalised (pivoted Gram-Schmidt).
__device__ void null_space(const float* pts, int n, const int* idx, float null[4][9]) {
  float q[5][9];
  for (int r = 0; r < 5; ++r) {
    const float* m = pts + (long long)idx[r] * 4;
    float u1 = m[0], v1 = m[1], u2 = m[2], v2 = m[3];
    float row[9] = {u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1.f};
    for (int s = 0; s < r; ++s) {
      float d = 0.f;
      for (int k = 0; k < 9; ++k) d += row[k] * q[s][k];
      for (int k = 0; k < 9; ++k) row[k] -= d * q[s][k];
    }
    float nrm = 0.f;
    for (int k = 0; k < 9; ++k) nrm += row[k] * row[k];
    nrm = rsqrtf(fmaxf(nrm, 1e-30f));
    for (int k = 0; k < 9; ++k) q[r][k] = row[k] * nrm;
  }
  // Residual norms of the unit vectors: 1 - sum_s q[s][j]^2 (and - sum of accepted null vectors' j-th squares).
  float res[9];
  for (int j = 0; j < 9; ++j) {
    float s = 1.f;
    for (int r = 0; r < 5; ++r) s -= q[r][j] * q[r][j];
    res[j] = s;
  }
  for (int t = 0; t < 4; ++t) {
    int best = 0;
    for (int j = 1; j < 9; ++j) best = res[j] > res[best] ? j : best;
    float v[9];
    for (int k = 0; k < 9; ++k) v[k] = 0.f;
    v[best] = 1.f;
    for (int r = 0; r < 5; ++r) {
      float d = q[r][best];
      for (int k = 0; k < 9; ++k) v[k] -= d * q[r][k];
    }
    for (int s = 0; s < t; ++s) {
      float d = 0.f;
      for (int k = 0; k < 9; ++k) d += v[k] * null[s][k];
      for (int k = 0; k < 9; ++k) v[k] -= d * null[s][k];
    }
    float nrm = 0.f;
    for (int k = 0; k < 9; ++k) nrm += v[k] * v[k];
    nrm = rsqrtf(fmaxf(nrm, 1e-30f));
    for (int k = 0; k < 9; ++k) null[t][k] = v[k] * nrm;
    for (int j = 0; j < 9; ++j) res[j] -= null[t][j] * null[t][j];
    res[best] = -1.f;
  }
}

// The ten cubic constraints (det E = 0, 2 E E^T E - tr(E E^T) E = 0) over the 20 cubic monomials, E = x E1 + y E2 +
// z E3 + E4 (null[0..3]); rows[r][c] fp32 -> written to m (double) [10][20].
__device__ void constraints(const float null[4][9], double m[10][20]) {
  float eet[3][3][10];  // E E^T, quadratic
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float* q = eet[i][j];
      for (int k = 0; k < 10; ++k) q[k] = 0.f;
      for (int k = 0; k < 3; ++k) {
        float a[4] = {null[0][3 * i + k], null[1][3 * i + k], null[2][3 * i + k], null[3][3 * i + k]};
        float b[4] = {null[0][3 * j + k], null[1][3 * j + k], null[2][3 * j + k], null[3][3 * j + k]};
        mul_ll(a, b, q);
      }
    }
  float tr[10];
  for (int k = 0; k < 10; ++k) tr[k] = eet[0][0][k] + eet[1][1][k] + eet[2][2][k];
  // rows 1..9: (2 E E^T - tr I) E
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float c[20];
      for (int k = 0; k < 20; ++k) c[k] = 0.f;
      for (int k = 0; k < 3; ++k) {
        float q[10];
        for (int t = 0; t < 10; ++t) q[t] = 2.f * eet[i][k][t] - (i == k ? tr[t] : 0.f);
        float b[4] = {null[0][3 * k + j], null[1][3 * k + j], null[2][3 * k + j], null[3][3 * k + j]};
        mul_ql(q, b, 1.f, c);
      }
      for (int k = 0; k < 20; ++k) m[1 + 3 * i + j][k] = c[k];
    }
  // row 0: det E = e00 (e11 e22 - e12 e21) - e01 (e10 e22 - e12 e20) + e02 (e10 e21 - e11 e20)
  {
    float c[20];
    for (int k = 0; k < 20; ++k) c[k] = 0.f;
    const int cof[3][4] = {{4, 8, 5, 7}, {3, 8, 5, 6}, {3, 7, 4, 6}};  // (a, d, b, c): a d - b c of the 3x3 index
    for (int j = 0; j < 3; ++j) {
      float q[10];
      for (int k = 0; k < 10; ++k) q[k] = 0.f;
      float a[4] = {null[0][cof[j][0]], null[1][cof[j][0]], null[2][cof[j][0]], null[3][cof[j][0]]};
      float d[4] = {null[0][cof[j][1]], null[1][cof[j][1]], null[2][cof[j][1]], null[3][cof[j][1]]};
      float b[4] = {null[0][cof[j][2]], null[1][cof[j][2]], null[2][cof[j][2]], null[3][cof[j][2]]};
      float e[4] = {null[0][cof[j][3]], null[1][cof[j][3]], null[2][cof[j][3]], null[3][cof[j][3]]};
      mul_ll(a, d, q);
      float nb[4] = {-b[0], -b[1], -b[2], -b[3]};
      mul_ll(nb, e, q);
      float e0[4] = {null[0][j], null[1][j], null[2][j], null[3][j]};
      mul_ql(q, e0, j == 1 ? -1.f : 1.f, c);
    }
    for (int k = 0; k < 20; ++k) m[0][k] = c[k];
  }
}

// Gauss-Jordan with partial pivoting: m -> [I | X]. Returns false if singular.
__device__ bool gauss_jordan(double m[10][20]) {
  for (int c = 0; c < 10; ++c) {
    int piv = c;
    for (int r = c + 1; r < 10; ++r) piv = fabs(m[r][c]) > fabs(m[piv][c]) ? r : piv;
    if (fabs(m[piv][c]) < 1e-14) return false;
    if (piv != c)
      for (int k = 0; k < 20; ++k) {
        double t = m[c][k];
        m[c][k] = m[piv][k];
        m[piv][k] = t;
      }
    double inv = 1.0 / m[c][c];
    for (int k = 0; k < 20; ++k) m[c][k] *= inv;
    for (int r = 0; r < 10; ++r) {
      if (r == c) continue;
      double f = m[r][c];
      if (f == 0.0) continue;
      for (int k = c; k < 20; ++k) m[r][k] -= f * m[c][k];
    }
  }
  return true;
}

// Complex fp32 helpers for the root search.
__device__ __forceinline__ float2 cmul(float2 a, float2 b) { return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
__device__ __forceinline__ float2 cdiv(float2 a, float2 b) {
  float d = fmaxf(b.x * b.x + b.y * b.y, 1e-30f);
  return make_float2((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}

// All roots of the degree-10 polynomial p (double, highest power first) by Aberth-Ehrlich in fp32 -> z[10].
__device__ void roots10(const double* pd, float2* z) {
  float p[11], dp[10], mx = 0.f;
  for (int i = 0; i < 11; ++i) mx = fmaxf(mx, (float)fabs(pd[i]));
  mx = fmaxf(mx, 1e-30f);
  for (int i = 0; i < 11; ++i) p[i] = (float)(pd[i]) / mx;
  for (int i = 0; i < 10; ++i) dp[i] = p[i] * (10 - i);
  float lead = fmaxf(fabsf(p[0]), 1e-30f), radius = 0.f;
  for (int k = 1; k <= 10; ++k) radius = fmaxf(radius, powf(fabsf(p[k]) / lead, 1.f / k));
  radius = fminf(fmaxf(2.f * radius, 1e-3f), 1e6f);
  for (int i = 0; i < 10; ++i) {
    float a = 6.2831853f * (i + 0.37f) / 10.f;
    z[i] = make_float2(radius * cosf(a), radius * sinf(a));
  }
  for (int it = 0; it < ITER_ABERTH; ++it) {
    float2 nz[10];
    float step = 0.f;
    for (int i = 0; i < 10; ++i) {
      float2 v = make_float2(p[0], 0.f), d = make_float2(dp[0], 0.f);
      for (int k = 1; k < 11; ++k) {
        v = cmul(v, z[i]);
        v.x += p[k];
      }
      for (int k = 1; k < 10; ++k) {
        d = cmul(d, z[i]);
        d.x += dp[k];
      }
      float2 ratio = cdiv(v, d), s = make_float2(0.f, 0.f);
      for (int j = 0; j < 10; ++j) {
        if (j == i) continue;
        float2 diff = make_float2(z[i].x - z[j].x, z[i].y - z[j].y);
        float2 inv = cdiv(make_float2(1.f, 0.f), diff);
        s.x += inv.x;
        s.y += inv.y;
      }
      float2 den = make_float2(1.f - (ratio.x * s.x - ratio.y * s.y), -(ratio.x * s.y + ratio.y * s.x));
      float2 upd = cdiv(ratio, den);
      nz[i] = make_float2(z[i].x - upd.x, z[i].y - upd.y);
      step = fmaxf(step, fabsf(upd.x) + fabsf(upd.y));
    }
    for (int i = 0; i < 10; ++i) z[i] = nz[i];
    if (step < 1e-7f * radius) break;
  }
}

// One thread per hypothesis (a, h): the up to ten essential matrices of its five matches.
// pts [P, N, 4]; pairs [A]; idx [A, H, 5] (into the first n_valid matches of the pair); out [A, H, 10, 9]; valid
// [A, H, 10] int32.
extern "C" __global__ void e5pt_hypotheses(const float* __restrict__ pts, const int* __restrict__ pairs,
                                           const int* __restrict__ idx, float* __restrict__ out,
                                           int* __restrict__ valid, int N, int H, int A) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)A * H) return;
  const int a = (int)(tid / H);
  const float* p = pts + (long long)pairs[a] * N * 4;
  const int* id = idx + tid * 5;
  float* o = out + tid * NSOL * 9;
  int* ok = valid + tid * NSOL;
  for (int s = 0; s < NSOL; ++s) {
    ok[s] = 0;
    for (int k = 0; k < 9; ++k) o[s * 9 + k] = 0.f;
  }
  for (int r = 0; r < 5; ++r)  // a repeated index: degenerate sample
    for (int s = r + 1; s < 5; ++s)
      if (id[r] == id[s]) return;
  float null[4][9];
  null_space(p, N, id, null);
  double m[10][20];
  constraints(null, m);
  if (!gauss_jordan(m)) return;
  // Rows e..j (4..9) over the tail monomials xz^2 xz x yz^2 yz y z^3 z^2 z 1 (columns 10..19):
  // k = e - z f, l = g - z h, m = i - z j, each x px(z) + y py(z) + pc(z), degrees 3, 3, 4.
  double px[3][4], py[3][4], pc[3][5];
  for (int t = 0; t < 3; ++t) {
    const double* e = m[4 + 2 * t] + 10;
    const double* f = m[5 + 2 * t] + 10;
    px[t][0] = -f[0]; px[t][1] = e[0] - f[1]; px[t][2] = e[1] - f[2]; px[t][3] = e[2];
    py[t][0] = -f[3]; py[t][1] = e[3] - f[4]; py[t][2] = e[4] - f[5]; py[t][3] = e[5];
    pc[t][0] = -f[6]; pc[t][1] = e[6] - f[7]; pc[t][2] = e[7] - f[8]; pc[t][3] = e[8] - f[9]; pc[t][4] = e[9];
  }
  // det B = kx (ly mc - lc my) - ky (lx mc - lc mx) + kc (lx my - ly mx)
  double t1[8], t2[8], t3[7], u[11], poly[11];
  pmul<4, 5>(py[1], pc[2], t1);
  pmul<5, 4>(pc[1], py[2], t2);
  for (int i = 0; i < 8; ++i) t1[i] -= t2[i];
  pmul<4, 8>(px[0], t1, poly);
  pmul<4, 5>(px[1], pc[2], t1);
  pmul<5, 4>(pc[1], px[2], t2);
  for (int i = 0; i < 8; ++i) t1[i] -= t2[i];
  pmul<4, 8>(py[0], t1, u);
  for (int i = 0; i < 11; ++i) poly[i] -= u[i];
  pmul<4, 4>(px[1], py[2], t3);
  pmul<4, 4>(py[1], px[2], t2);
  for (int i = 0; i < 7; ++i) t3[i] -= t2[i];
  pmul<5, 7>(pc[0], t3, u);
  for (int i = 0; i < 11; ++i) poly[i] += u[i];
  double dpoly[10];
  for (int i = 0; i < 10; ++i) dpoly[i] = poly[i] * (10 - i);
  float2 z[10];
  roots10(poly, z);
  for (int s = 0; s < NSOL; ++s) {
    if (fabsf(z[s].y) > 1e-3f * (1.f + fabsf(z[s].x))) continue;
    double zr = z[s].x;
    for (int it = 0; it < 3; ++it) {
      double d = peval<10>(dpoly, zr);
      if (d == 0.0) break;
      zr -= peval<11>(poly, zr) / d;
    }
    if (!isfinite(zr)) continue;
    // (x, y, 1) ~ cross product of the two best-conditioned rows of B(zr).
    double rx[3], ry[3], rc[3];
    for (int t = 0; t < 3; ++t) {
      rx[t] = peval<4>(px[t], zr);
      ry[t] = peval<4>(py[t], zr);
      rc[t] = peval<5>(pc[t], zr);
    }
    double bx = 0, by = 0, bw = 0, bq = -1.0;
    const int pr[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int t = 0; t < 3; ++t) {
      int i = pr[t][0], j = pr[t][1];
      double cx = ry[i] * rc[j] - rc[i] * ry[j], cy = rc[i] * rx[j] - rx[i] * rc[j], cw = rx[i] * ry[j] - ry[i] * rx[j];
      double sc = sqrt(cx * cx + cy * cy + cw * cw);
      double q = sc > 0 ? fabs(cw) / sc : -1.0;
      if (q > bq) { bq = q; bx = cx; by = cy; bw = cw; }
    }
    if (bq < 1e-8) continue;
    float x = (float)(bx / bw), y = (float)(by / bw), zz = (float)zr;
    float e[9], nrm = 0.f;
    for (int k = 0; k < 9; ++k) {
      e[k] = x * null[0][k] + y * null[1][k] + zz * null[2][k] + null[3][k];
      nrm += e[k] * e[k];
    }
    if (!(nrm > 0.f) || !isfinite(nrm)) continue;
    nrm = rsqrtf(nrm);
    for (int k = 0; k < 9; ++k) o[s * 9 + k] = e[k] * nrm;
    ok[s] = 1;
  }
}

// Sampson inlier counts of many candidates per pair: one thread per candidate, the pair's matches through shared
// memory in chunks. e [A, C, 9]; counts [P] valid matches per pair; thr2 [P]; out [A, C] int32; scores the first
// min(n, counts[pair]) matches. grid (ceil(C / 256), A), block 256.
#define CHUNK 1024
extern "C" __global__ void score_many(const float* __restrict__ pts, const int* __restrict__ counts,
                                      const float* __restrict__ thr2, const int* __restrict__ pairs,
                                      const float* __restrict__ e, int* __restrict__ out, int N, int C, int n) {
  __shared__ float4 sm[CHUNK];
  const int a = blockIdx.y, pair = pairs[a];
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int tot = min(n, counts[pair]);
  const float4* p = (const float4*)(pts + (long long)pair * N * 4);
  float E[9];
  if (c < C)
    for (int k = 0; k < 9; ++k) E[k] = e[((long long)a * C + c) * 9 + k];
  const float t2 = thr2[pair];
  int cnt = 0;
  for (int base = 0; base < tot; base += CHUNK) {
    const int len = min(CHUNK, tot - base);
    __syncthreads();
    for (int i = threadIdx.x; i < len; i += blockDim.x) sm[i] = p[base + i];
    __syncthreads();
    if (c < C) {
      for (int i = 0; i < len; ++i) {
        float4 m = sm[i];
        float l0 = E[0] * m.x + E[1] * m.y + E[2], l1 = E[3] * m.x + E[4] * m.y + E[5], l2 = E[6] * m.x + E[7] * m.y + E[8];
        float r0 = E[0] * m.z + E[3] * m.w + E[6], r1 = E[1] * m.z + E[4] * m.w + E[7];
        float num = m.z * l0 + m.w * l1 + l2;
        float den = l0 * l0 + l1 * l1 + r0 * r0 + r1 * r1;
        cnt += (num * num < t2 * den) ? 1 : 0;
      }
    }
  }
  if (c < C) out[(long long)a * C + c] = cnt;
}

// Sampson inlier counts of few candidates per pair: one block per (a, c), threads over the matches. Writes the
// inlier mask [A, C, N] (uint8) when `mask` is not null. grid (C, A), block 256.
extern "C" __global__ void score_few(const float* __restrict__ pts, const int* __restrict__ counts,
                                     const float* __restrict__ thr2, const int* __restrict__ pairs,
                                     const float* __restrict__ e, int* __restrict__ out, unsigned char* mask, int N,
                                     int C, int n) {
  __shared__ int red[256];
  const int c = blockIdx.x, a = blockIdx.y, pair = pairs[a];
  const int tot = min(n, counts[pair]);
  const float4* p = (const float4*)(pts + (long long)pair * N * 4);
  const float* E = e + ((long long)a * C + c) * 9;
  float E0 = E[0], E1 = E[1], E2 = E[2], E3 = E[3], E4 = E[4], E5 = E[5], E6 = E[6], E7 = E[7], E8 = E[8];
  const float t2 = thr2[pair];
  unsigned char* mk = mask ? mask + ((long long)a * C + c) * N : nullptr;
  int cnt = 0;
  for (int i = threadIdx.x; i < (mk ? N : tot); i += blockDim.x) {
    int in = 0;
    if (i < tot) {
      float4 m = p[i];
      float l0 = E0 * m.x + E1 * m.y + E2, l1 = E3 * m.x + E4 * m.y + E5, l2 = E6 * m.x + E7 * m.y + E8;
      float r0 = E0 * m.z + E3 * m.w + E6, r1 = E1 * m.z + E4 * m.w + E7;
      float num = m.z * l0 + m.w * l1 + l2;
      float den = l0 * l0 + l1 * l1 + r0 * r0 + r1 * r1;
      in = (num * num < t2 * den) ? 1 : 0;
    }
    cnt += in;
    if (mk) mk[i] = (unsigned char)in;
  }
  red[threadIdx.x] = cnt;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[(long long)a * C + c] = red[0];
}
