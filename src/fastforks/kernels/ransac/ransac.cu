// Essential-matrix RANSAC for `fastforks.matching.estimate_pose`: Nister's 5-point solver, one hypothesis per thread,
// and fused Sampson scoring of many candidates against the matches of a pair (the 7-point solver and the absolute
// pose, P3P, below share the scoring and refinement). Launched on torch's current stream.
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

// Null space of the R x 9 epipolar constraint rows (R = 5: E, 7: F): orthonormalise the rows (modified Gram-Schmidt), then the 9 - R
// unit vectors with the largest residual after projecting out the row span, orthonormalised (pivoted Gram-Schmidt).
template <int R>
__device__ void null_space(const float* pts, int n, const int* idx, float null[9 - R][9]) {
  float q[R][9];
  for (int r = 0; r < R; ++r) {
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
    for (int r = 0; r < R; ++r) s -= q[r][j] * q[r][j];
    res[j] = s;
  }
  for (int t = 0; t < 9 - R; ++t) {
    int best = 0;
    for (int j = 1; j < 9; ++j) best = res[j] > res[best] ? j : best;
    float v[9];
    for (int k = 0; k < 9; ++k) v[k] = 0.f;
    v[best] = 1.f;
    for (int r = 0; r < R; ++r) {
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
    // Rows 0..3 are not read after their own pivot (only rows 4..9 of X are used), so skip them from pivot 4 on.
    for (int r = (c >= 4 ? 4 : 0); r < 10; ++r) {
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
__device__ __forceinline__ float2 cdiv(float2 a, float2 b) {  // approximate reciprocal: the roots are polished later
  float inv = __fdividef(1.f, fmaxf(b.x * b.x + b.y * b.y, 1e-30f));
  return make_float2((a.x * b.x + a.y * b.y) * inv, (a.y * b.x - a.x * b.y) * inv);
}
__device__ __forceinline__ float2 cinv(float2 b) {
  float inv = __fdividef(1.f, fmaxf(b.x * b.x + b.y * b.y, 1e-30f));
  return make_float2(b.x * inv, -b.y * inv);
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
  // Fully unrolled so the roots stay in registers (a 5x of the kernel's latency otherwise).
  for (int it = 0; it < ITER_ABERTH; ++it) {
    float2 nz[10];
    float step = 0.f;
#pragma unroll
    for (int i = 0; i < 10; ++i) {
      float2 v = make_float2(p[0], 0.f), d = make_float2(dp[0], 0.f);
#pragma unroll
      for (int k = 1; k < 11; ++k) {
        v = cmul(v, z[i]);
        v.x += p[k];
      }
#pragma unroll
      for (int k = 1; k < 10; ++k) {
        d = cmul(d, z[i]);
        d.x += dp[k];
      }
      float2 ratio = cdiv(v, d), s = make_float2(0.f, 0.f);
#pragma unroll
      for (int j = 0; j < 10; ++j) {
        if (j == i) continue;
        float2 inv = cinv(make_float2(z[i].x - z[j].x, z[i].y - z[j].y));
        s.x += inv.x;
        s.y += inv.y;
      }
      float2 den = make_float2(1.f - (ratio.x * s.x - ratio.y * s.y), -(ratio.x * s.y + ratio.y * s.x));
      float2 upd = cdiv(ratio, den);
      nz[i] = make_float2(z[i].x - upd.x, z[i].y - upd.y);
      step = fmaxf(step, fabsf(upd.x) + fabsf(upd.y));
    }
#pragma unroll
    for (int i = 0; i < 10; ++i) z[i] = nz[i];
    if (step < 1e-6f * radius) break;
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
  null_space<5>(p, N, id, null);
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
    double bx = 0, by = 0, bw = 0;
    float bq = -1.f;
    const int pr[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int t = 0; t < 3; ++t) {
      int i = pr[t][0], j = pr[t][1];
      double cx = ry[i] * rc[j] - rc[i] * ry[j], cy = rc[i] * rx[j] - rx[i] * rc[j], cw = rx[i] * ry[j] - ry[i] * rx[j];
      float fx = (float)cx, fy = (float)cy, fw = (float)cw;  // the conditioning measure needs no fp64
      float sc = sqrtf(fx * fx + fy * fy + fw * fw);
      float q = sc > 0.f ? fabsf(fw) / sc : -1.f;
      if (q > bq) { bq = q; bx = cx; by = cy; bw = cw; }
    }
    if (bq < 1e-8f) continue;
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


// One thread per hypothesis (a, h): the up to three fundamental matrices of its seven matches (pts in normalised
// pixel coordinates). F = l F1 + (1 - l) F2 over the null space, det F = 0 a cubic in l (interpolated from four
// determinants, solved in fp64). out [A, H, 3, 9]; valid [A, H, 3] int32.
__device__ __forceinline__ double det3d(const double* m) {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}
extern "C" __global__ void f7pt_hypotheses(const float* __restrict__ pts, const int* __restrict__ pairs,
                                           const int* __restrict__ idx, float* __restrict__ out,
                                           int* __restrict__ valid, int N, int H, int A) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)A * H) return;
  const int a = (int)(tid / H);
  const float* p = pts + (long long)pairs[a] * N * 4;
  const int* id = idx + tid * 7;
  float* o = out + tid * 3 * 9;
  int* ok = valid + tid * 3;
  for (int s = 0; s < 3; ++s) {
    ok[s] = 0;
    for (int k = 0; k < 9; ++k) o[s * 9 + k] = 0.f;
  }
  for (int r = 0; r < 7; ++r)
    for (int s = r + 1; s < 7; ++s)
      if (id[r] == id[s]) return;
  float null[2][9];
  null_space<7>(p, N, id, null);
  double f1[9], f2[9], g[9];
  for (int k = 0; k < 9; ++k) { f1[k] = null[0][k]; f2[k] = null[1][k]; }
  const double d0 = det3d(f2), d1 = det3d(f1);
  for (int k = 0; k < 9; ++k) g[k] = 2.0 * f2[k] - f1[k];
  const double dm = det3d(g);
  for (int k = 0; k < 9; ++k) g[k] = 2.0 * f1[k] - f2[k];
  const double d2 = det3d(g);
  const double a0 = d0, a2 = 0.5 * (d1 + dm) - a0, a3 = (d2 - a0 - 4.0 * a2 - (d1 - dm)) / 6.0;
  const double a1 = 0.5 * (d1 - dm) - a3;
  double roots[3];
  int nr = 0;
  const double scale = fmax(fmax(fabs(a3), fabs(a2)), fmax(fabs(a1), fabs(a0)));
  if (!(scale > 0.0)) return;
  if (fabs(a3) > 1e-10 * scale) {  // cubic: depressed form, trigonometric / Cardano
    const double b = a2 / a3, c = a1 / a3, d = a0 / a3;
    const double q = (3.0 * c - b * b) / 9.0, r = (9.0 * b * c - 27.0 * d - 2.0 * b * b * b) / 54.0;
    const double disc = q * q * q + r * r;
    if (disc < 0.0) {
      const double th = acos(fmax(-1.0, fmin(1.0, r / sqrt(-q * q * q)))), m2 = 2.0 * sqrt(-q);
      roots[0] = m2 * cos(th / 3.0) - b / 3.0;
      roots[1] = m2 * cos((th + 2.0 * 3.14159265358979323846) / 3.0) - b / 3.0;
      roots[2] = m2 * cos((th + 4.0 * 3.14159265358979323846) / 3.0) - b / 3.0;
      nr = 3;
    } else {
      const double sd = sqrt(disc), s1 = cbrt(r + sd), s2 = cbrt(r - sd);
      roots[0] = s1 + s2 - b / 3.0;
      nr = 1;
    }
  } else if (fabs(a2) > 1e-10 * scale) {  // quadratic
    const double disc = a1 * a1 - 4.0 * a2 * a0;
    if (disc >= 0.0) {
      const double sd = sqrt(disc);
      roots[0] = (-a1 + sd) / (2.0 * a2);
      roots[1] = (-a1 - sd) / (2.0 * a2);
      nr = 2;
    }
  } else if (fabs(a1) > 1e-10 * scale) {
    roots[0] = -a0 / a1;
    nr = 1;
  }
  for (int s = 0; s < nr; ++s) {
    double l = roots[s];
    for (int it = 0; it < 2; ++it) {  // Newton polish
      const double v = ((a3 * l + a2) * l + a1) * l + a0, dv = (3.0 * a3 * l + 2.0 * a2) * l + a1;
      if (dv != 0.0) l -= v / dv;
    }
    if (!isfinite(l)) continue;
    float f[9], nrm = 0.f;
    for (int k = 0; k < 9; ++k) {
      f[k] = (float)(l * f1[k] + (1.0 - l) * f2[k]);
      nrm += f[k] * f[k];
    }
    if (!(nrm > 0.f) || !isfinite(nrm)) continue;
    nrm = rsqrtf(nrm);
    for (int k = 0; k < 9; ++k) o[s * 9 + k] = f[k] * nrm;
    ok[s] = 1;
  }
}

// MSAC score of a match: 1 - sampson / thr2 for inliers (truncated quadratic loss), 0 for outliers.
__device__ __forceinline__ float gain(const float4 m, const float* E, float t2) {
  float l0 = E[0] * m.x + E[1] * m.y + E[2], l1 = E[3] * m.x + E[4] * m.y + E[5], l2 = E[6] * m.x + E[7] * m.y + E[8];
  float r0 = E[0] * m.z + E[3] * m.w + E[6], r1 = E[1] * m.z + E[4] * m.w + E[7];
  float num = m.z * l0 + m.w * l1 + l2;
  float den = (l0 * l0 + l1 * l1 + r0 * r0 + r1 * r1) * t2;
  return (den > 0.f && num * num < den) ? 1.f - num * num / den : 0.f;
}

// Model types of the scoring kernels: the essential / fundamental matrix (9 floats, matches of one float4: u1, v1,
// u2, v2 normalised) and the absolute pose [R | t] (12 floats, matches of two float4: u, v, X, Y | Z, 0, 0, 0;
// reprojection error of X in normalised coordinates, points behind the camera are outliers).
struct Epipolar {
  static constexpr int NM = 9, NF = 1;
  __device__ static __forceinline__ float gain(const float4* m, const float* E, float t2) { return ::gain(m[0], E, t2); }
};
struct Absolute {
  static constexpr int NM = 12, NF = 2;
  __device__ static __forceinline__ float gain(const float4* m, const float* M, float t2) {
    const float X = m[0].z, Y = m[0].w, Z = m[1].x;
    float z = M[6] * X + M[7] * Y + M[8] * Z + M[11];
    if (!(z > 0.f)) return 0.f;
    float dx = (M[0] * X + M[1] * Y + M[2] * Z + M[9]) / z - m[0].x;
    float dy = (M[3] * X + M[4] * Y + M[5] * Z + M[10]) / z - m[0].y;
    float e = (dx * dx + dy * dy) / t2;
    return e < 1.f ? 1.f - e : 0.f;
  }
};

// MSAC scores and inlier counts of many candidates per pair: one thread per candidate, the pair's matches through
// shared memory in chunks. e [A, C, NM]; counts [P] valid matches per pair; thr2 [P]; out [A, C] float32 scores,
// cnt [A, C] int32 inlier counts; scores the first min(n, counts[pair]) matches. grid (ceil(C / 256), A), block 256.
#define CHUNK 1024
template <class Model>
__device__ __forceinline__ void score_many_t(const float* __restrict__ pts, const int* __restrict__ counts,
                                             const float* __restrict__ thr2, const int* __restrict__ pairs,
                                             const float* __restrict__ e, float* __restrict__ out,
                                             int* __restrict__ cnt, int N, int C, int n) {
  constexpr int NF = Model::NF, NM = Model::NM, PER = CHUNK / NF;
  __shared__ float4 sm[CHUNK];
  const int a = blockIdx.y, pair = pairs[a];
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int tot = min(n, counts[pair]);
  const float4* p = (const float4*)(pts + (long long)pair * N * 4 * NF);
  float E[NM];
  if (c < C)
    for (int k = 0; k < NM; ++k) E[k] = e[((long long)a * C + c) * NM + k];
  const float t2 = thr2[pair];
  float sc = 0.f;
  int in = 0;
  for (int base = 0; base < tot; base += PER) {
    const int len = min(PER, tot - base);
    __syncthreads();
    for (int i = threadIdx.x; i < len * NF; i += blockDim.x) sm[i] = p[base * NF + i];
    __syncthreads();
    if (c < C) {
      for (int i = 0; i < len; ++i) {
        float g = Model::gain(sm + i * NF, E, t2);
        sc += g;
        in += g > 0.f;
      }
    }
  }
  if (c < C) {
    out[(long long)a * C + c] = sc;
    cnt[(long long)a * C + c] = in;
  }
}
extern "C" __global__ void score_many(const float* __restrict__ pts, const int* __restrict__ counts,
                                      const float* __restrict__ thr2, const int* __restrict__ pairs,
                                      const float* __restrict__ e, float* __restrict__ out, int* __restrict__ cnt,
                                      int N, int C, int n) {
  score_many_t<Epipolar>(pts, counts, thr2, pairs, e, out, cnt, N, C, n);
}
extern "C" __global__ void score_many_abs(const float* __restrict__ pts, const int* __restrict__ counts,
                                          const float* __restrict__ thr2, const int* __restrict__ pairs,
                                          const float* __restrict__ e, float* __restrict__ out,
                                          int* __restrict__ cnt, int N, int C, int n) {
  score_many_t<Absolute>(pts, counts, thr2, pairs, e, out, cnt, N, C, n);
}

// MSAC scores and inlier counts of few candidates per pair: one block per (a, c), threads over the matches. Writes
// the inlier mask [A, C, N] (uint8) when `mask` is not null. grid (C, A), block 256.
template <class Model>
__device__ __forceinline__ void score_few_t(const float* __restrict__ pts, const int* __restrict__ counts,
                                            const float* __restrict__ thr2, const int* __restrict__ pairs,
                                            const float* __restrict__ e, float* __restrict__ out,
                                            int* __restrict__ cnt, unsigned char* mask, int N, int C, int n) {
  constexpr int NF = Model::NF, NM = Model::NM;
  __shared__ float red[256];
  __shared__ int redi[256];
  const int c = blockIdx.x, a = blockIdx.y, pair = pairs[a];
  const int tot = min(n, counts[pair]);
  const float4* p = (const float4*)(pts + (long long)pair * N * 4 * NF);
  const float* eg = e + ((long long)a * C + c) * NM;
  float E[NM];
  for (int k = 0; k < NM; ++k) E[k] = eg[k];
  const float t2 = thr2[pair];
  unsigned char* mk = mask ? mask + ((long long)a * C + c) * N : nullptr;
  float sc = 0.f;
  int in = 0;
  for (int i = threadIdx.x; i < (mk ? N : tot); i += blockDim.x) {
    float g = i < tot ? Model::gain(p + i * NF, E, t2) : 0.f;
    sc += g;
    in += g > 0.f;
    if (mk) mk[i] = (unsigned char)(g > 0.f);
  }
  red[threadIdx.x] = sc;
  redi[threadIdx.x] = in;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
      redi[threadIdx.x] += redi[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[(long long)a * C + c] = red[0];
    cnt[(long long)a * C + c] = redi[0];
  }
}
extern "C" __global__ void score_few(const float* __restrict__ pts, const int* __restrict__ counts,
                                     const float* __restrict__ thr2, const int* __restrict__ pairs,
                                     const float* __restrict__ e, float* __restrict__ out, int* __restrict__ cnt,
                                     unsigned char* mask, int N, int C, int n) {
  score_few_t<Epipolar>(pts, counts, thr2, pairs, e, out, cnt, mask, N, C, n);
}
extern "C" __global__ void score_few_abs(const float* __restrict__ pts, const int* __restrict__ counts,
                                         const float* __restrict__ thr2, const int* __restrict__ pairs,
                                         const float* __restrict__ e, float* __restrict__ out,
                                         int* __restrict__ cnt, unsigned char* mask, int N, int C, int n) {
  score_few_t<Absolute>(pts, counts, thr2, pairs, e, out, cnt, mask, N, C, n);
}

// Levenberg-Marquardt of (R, t), E = [t]_x R, on the truncated Sampson cost (PoseLib's refinement): one block per
// model, every iteration inside the kernel. rt [A, 12]: R (row-major) then t (unit), updated in place. The cost is
// sum(min(sampson^2, thr2)) over the first min(n, counts[pair]) matches; J runs over the inliers. grid A, block 256.
#define NT 46  // block_sum width: 45 + 1 terms of the 8-point normal matrix (LM uses 21)
#define NLM 21  // 5 pose parameters: 15 + 5 + 1 terms
#define NLMF 36  // 7 fundamental-matrix parameters: 28 + 7 + 1
__device__ __forceinline__ void cross_mat(const float* v, float* m) {
  m[0] = 0.f; m[1] = -v[2]; m[2] = v[1]; m[3] = v[2]; m[4] = 0.f; m[5] = -v[0]; m[6] = -v[1]; m[7] = v[0]; m[8] = 0.f;
}
__device__ __forceinline__ void mat_mul(const float* a, const float* b, float* c) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}
// Sum of acc[0..k) over the block into red[0][0..k); all threads call it.
__device__ void block_sum(float* acc, int k, float (*red)[NT]) {
  for (int i = 0; i < k; ++i)
    for (int off = 16; off > 0; off >>= 1) acc[i] += __shfl_down_sync(0xffffffffu, acc[i], off);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();  // red may still be read from the previous reduction
  if (lane == 0)
    for (int i = 0; i < k; ++i) red[warp][i] = acc[i];
  __syncthreads();
  if (threadIdx.x < k) {
    float v = 0.f;
    for (int w = 0; w < 8; ++w) v += red[w][threadIdx.x];
    red[0][threadIdx.x] = v;
  }
  __syncthreads();
}
// Per-thread J^T J (upper triangle), J^T r (NP) and cost (1) of this thread's matches; D [9, NP] = dE / dparams.
template <int NP>
__device__ void lm_accumulate(const float4* p, int tot, const float* E, const float* D, float t2, float* acc) {
  constexpr int NU = NP * (NP + 1) / 2;
  for (int k = 0; k < NU + NP + 1; ++k) acc[k] = 0.f;
  for (int i = threadIdx.x; i < tot; i += blockDim.x) {
    const float4 m = p[i];
    const float x1[3] = {m.x, m.y, 1.f}, x2[3] = {m.z, m.w, 1.f};
    float l0 = E[0] * m.x + E[1] * m.y + E[2], l1 = E[3] * m.x + E[4] * m.y + E[5], l2 = E[6] * m.x + E[7] * m.y + E[8];
    float r0 = E[0] * m.z + E[3] * m.w + E[6], r1 = E[1] * m.z + E[4] * m.w + E[7];
    float num = m.z * l0 + m.w * l1 + l2;
    float den = l0 * l0 + l1 * l1 + r0 * r0 + r1 * r1;
    if (!(den > 0.f)) { acc[NU + NP] += t2; continue; }
    float inv = rsqrtf(den), res = num * inv;
    if (res * res >= t2) { acc[NU + NP] += t2; continue; }
    acc[NU + NP] += res * res;
    float s3 = num * inv * inv * inv, J[NP];
    for (int k = 0; k < NP; ++k) J[k] = 0.f;
    for (int i3 = 0; i3 < 3; ++i3)
      for (int j3 = 0; j3 < 3; ++j3) {
        float dd = (i3 == 0 ? l0 * x1[j3] : i3 == 1 ? l1 * x1[j3] : 0.f) + (j3 == 0 ? r0 * x2[i3] : j3 == 1 ? r1 * x2[i3] : 0.f);
        float g = x2[i3] * x1[j3] * inv - s3 * dd;
        const float* d = D + (i3 * 3 + j3) * NP;
        for (int k = 0; k < NP; ++k) J[k] += g * d[k];
      }
    int q = 0;
    for (int u = 0; u < NP; ++u)
      for (int v = u; v < NP; ++v) acc[q++] += J[u] * J[v];
    for (int k = 0; k < NP; ++k) acc[NU + k] += J[k] * res;
  }
}
__device__ float thread_cost(const float4* p, int tot, const float* E, float t2) {
  float c = 0.f;
  for (int i = threadIdx.x; i < tot; i += blockDim.x) {
    const float4 m = p[i];
    float l0 = E[0] * m.x + E[1] * m.y + E[2], l1 = E[3] * m.x + E[4] * m.y + E[5], l2 = E[6] * m.x + E[7] * m.y + E[8];
    float r0 = E[0] * m.z + E[3] * m.w + E[6], r1 = E[1] * m.z + E[4] * m.w + E[7];
    float num = m.z * l0 + m.w * l1 + l2;
    float den = l0 * l0 + l1 * l1 + r0 * r0 + r1 * r1;
    float e2 = den > 0.f ? num * num / den : t2;
    c += fminf(e2, t2);
  }
  return c;
}
// Thread 0: E = [t]_x R and the derivatives D [9, 5] of E by (omega, alpha), the tangent basis b1, b2 of t.
__device__ void lm_linearise(const float* R, const float* t, float* E, float* D, float* b1, float* b2) {
  int k = fabsf(t[0]) <= fabsf(t[1]) ? (fabsf(t[0]) <= fabsf(t[2]) ? 0 : 2) : (fabsf(t[1]) <= fabsf(t[2]) ? 1 : 2);
  float ek[3] = {0.f, 0.f, 0.f};
  ek[k] = 1.f;
  float c[3] = {t[1] * ek[2] - t[2] * ek[1], t[2] * ek[0] - t[0] * ek[2], t[0] * ek[1] - t[1] * ek[0]};
  float inv = rsqrtf(fmaxf(c[0] * c[0] + c[1] * c[1] + c[2] * c[2], 1e-24f));
  for (int i = 0; i < 3; ++i) b1[i] = c[i] * inv;
  b2[0] = t[1] * b1[2] - t[2] * b1[1]; b2[1] = t[2] * b1[0] - t[0] * b1[2]; b2[2] = t[0] * b1[1] - t[1] * b1[0];
  float tx[9], tmp[9], gen[9];
  cross_mat(t, tx);
  mat_mul(tx, R, E);
  for (int k3 = 0; k3 < 3; ++k3) {
    float e3[3] = {0.f, 0.f, 0.f};
    e3[k3] = 1.f;
    cross_mat(e3, gen);
    mat_mul(E, gen, tmp);
    for (int i = 0; i < 9; ++i) D[i * 5 + k3] = tmp[i];
  }
  cross_mat(b1, gen);
  mat_mul(gen, R, tmp);
  for (int i = 0; i < 9; ++i) D[i * 5 + 3] = tmp[i];
  cross_mat(b2, gen);
  mat_mul(gen, R, tmp);
  for (int i = 0; i < 9; ++i) D[i * 5 + 4] = tmp[i];
}
// Thread 0: the damped normal equations from the accumulated terms; the NP-parameter step, or zero when singular.
template <int NP>
__device__ void lm_step(const float* terms, float lam, float* step) {
  constexpr int NU = NP * (NP + 1) / 2;
  float M[NP][NP + 1];
  int q = 0;
  for (int u = 0; u < NP; ++u)
    for (int v = u; v < NP; ++v) {
      M[u][v] = M[v][u] = terms[q++];
    }
  for (int u = 0; u < NP; ++u) {
    M[u][u] += lam * fmaxf(M[u][u], 1e-12f);
    M[u][NP] = -terms[NU + u];
  }
  for (int c = 0; c < NP; ++c) {  // Gaussian elimination with partial pivoting
    int piv = c;
    for (int r = c + 1; r < NP; ++r)
      if (fabsf(M[r][c]) > fabsf(M[piv][c])) piv = r;
    if (piv != c)
      for (int j = 0; j < NP + 1; ++j) { float x = M[c][j]; M[c][j] = M[piv][j]; M[piv][j] = x; }
    if (!(fabsf(M[c][c]) > 1e-30f)) {
      for (int k = 0; k < NP; ++k) step[k] = 0.f;
      return;
    }
    for (int r = c + 1; r < NP; ++r) {
      float f = M[r][c] / M[c][c];
      for (int j = c; j < NP + 1; ++j) M[r][j] -= f * M[c][j];
    }
  }
  for (int r = NP - 1; r >= 0; --r) {
    float x = M[r][NP];
    for (int j = r + 1; j < NP; ++j) x -= M[r][j] * step[j];
    step[r] = x / M[r][r];
  }
  for (int k = 0; k < NP; ++k)
    if (!isfinite(step[k])) {
      for (int j = 0; j < NP; ++j) step[j] = 0.f;
      return;
    }
}
// exp([w]_x) of the axis-angle vector w.
__device__ void exp_so3(const float* w, float* Rw) {
  float th = sqrtf(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  for (int i = 0; i < 9; ++i) Rw[i] = i % 4 == 0 ? 1.f : 0.f;
  if (th > 1e-12f) {
    float k[3] = {w[0] / th, w[1] / th, w[2] / th}, K[9], K2[9];
    cross_mat(k, K);
    mat_mul(K, K, K2);
    float s = sinf(th), c = 1.f - cosf(th);
    for (int i = 0; i < 9; ++i) Rw[i] += s * K[i] + c * K2[i];
  }
}
// Thread 0: apply the step: R_new = R exp([omega]_x), t_new = normalise(t + a1 b1 + a2 b2).
__device__ void lm_apply(const float* R, const float* t, const float* step, const float* b1, const float* b2,
                         float* Rn, float* tn) {
  float Rw[9];
  exp_so3(step, Rw);
  mat_mul(R, Rw, Rn);
  for (int i = 0; i < 3; ++i) tn[i] = t[i] + step[3] * b1[i] + step[4] * b2[i];
  float inv = rsqrtf(fmaxf(tn[0] * tn[0] + tn[1] * tn[1] + tn[2] * tn[2], 1e-24f));
  for (int i = 0; i < 3; ++i) tn[i] *= inv;
}
// The LM iterations on the pose in shared memory (sR, st); all threads call it. acc is the caller's NT-wide
// per-thread scratch, red the 8 x NT reduction buffer.
__device__ void lm_run(const float4* p, int tot, float t2, int iterations, float* sR, float* st, float* acc,
                       float (*red)[NT]) {
  __shared__ float sRn[9], stn[3], sE[9], sEn[9], sD[45], sb1[3], sb2[3], sStep[5];
  __shared__ float sLam, sBest;
  if (threadIdx.x == 0) {
    sLam = 1e-3f;
    lm_linearise(sR, st, sE, sD, sb1, sb2);
  }
  __syncthreads();
  acc[0] = thread_cost(p, tot, sE, t2);
  block_sum(acc, 1, red);
  if (threadIdx.x == 0) sBest = red[0][0];
  for (int it = 0; it < iterations; ++it) {
    lm_accumulate<5>(p, tot, sE, sD, t2, acc);
    block_sum(acc, NLM, red);
    if (threadIdx.x == 0) {
      lm_step<5>(red[0], sLam, sStep);
      lm_apply(sR, st, sStep, sb1, sb2, sRn, stn);
      float tx[9];
      cross_mat(stn, tx);
      mat_mul(tx, sRn, sEn);
    }
    __syncthreads();
    acc[0] = thread_cost(p, tot, sEn, t2);
    block_sum(acc, 1, red);
    if (threadIdx.x == 0) {
      if (red[0][0] < sBest) {
        sBest = red[0][0];
        for (int i = 0; i < 9; ++i) sR[i] = sRn[i];
        for (int i = 0; i < 3; ++i) st[i] = stn[i];
        sLam = fmaxf(sLam * 0.1f, 1e-6f);
        lm_linearise(sR, st, sE, sD, sb1, sb2);
      } else {
        sLam = fminf(sLam * 10.f, 1e6f);
      }
    }
    __syncthreads();
  }
}
// Thread 0: F = U diag(cos p, sin p, 0) V^T and its derivatives D [9, 7] by the rotations of U, V (right
// multiplied by exp([w]_x)) and the angle p.
__device__ void lm_linearise_f(const float* U, const float* V, float phi, float* F, float* D) {
  const float s[3] = {cosf(phi), sinf(phi), 0.f}, ds[3] = {-sinf(phi), cosf(phi), 0.f};
  float US[9], gen[9], tmp[9], tmp2[9], Vt[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) { US[i * 3 + j] = U[i * 3 + j] * s[j]; Vt[i * 3 + j] = V[j * 3 + i]; }
  mat_mul(US, Vt, F);
  for (int k = 0; k < 3; ++k) {
    float e3[3] = {0.f, 0.f, 0.f};
    e3[k] = 1.f;
    cross_mat(e3, gen);
    mat_mul(U, gen, tmp);  // U [e_k]_x S V^T
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) tmp2[i * 3 + j] = tmp[i * 3 + j] * s[j];
    mat_mul(tmp2, Vt, tmp);
    for (int i = 0; i < 9; ++i) D[i * 7 + k] = tmp[i];
    mat_mul(US, gen, tmp);  // -U S [e_k]_x V^T
    mat_mul(tmp, Vt, tmp2);
    for (int i = 0; i < 9; ++i) D[i * 7 + 3 + k] = -tmp2[i];
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) tmp[i * 3 + j] = U[i * 3 + j] * ds[j];
  mat_mul(tmp, Vt, tmp2);
  for (int i = 0; i < 9; ++i) D[i * 7 + 6] = tmp2[i];
}
// The LM iterations of a fundamental matrix in shared memory (sU, sV, *sPhi; sF receives the result); all
// threads call it.
__device__ void lm_run_f(const float4* p, int tot, float t2, int iterations, float* sU, float* sV, float* sPhi,
                         float* sF, float* acc, float (*red)[NT]) {
  __shared__ float sUn[9], sVn[9], sPhin, sFn[9], sD[63], sStep[7];
  __shared__ float sLam, sBest;
  if (threadIdx.x == 0) {
    sLam = 1e-3f;
    lm_linearise_f(sU, sV, *sPhi, sF, sD);
  }
  __syncthreads();
  acc[0] = thread_cost(p, tot, sF, t2);
  block_sum(acc, 1, red);
  if (threadIdx.x == 0) sBest = red[0][0];
  for (int it = 0; it < iterations; ++it) {
    lm_accumulate<7>(p, tot, sF, sD, t2, acc);
    block_sum(acc, NLMF, red);
    if (threadIdx.x == 0) {
      lm_step<7>(red[0], sLam, sStep);
      float Rw[9];
      exp_so3(sStep, Rw);
      mat_mul(sU, Rw, sUn);
      exp_so3(sStep + 3, Rw);
      mat_mul(sV, Rw, sVn);
      sPhin = *sPhi + sStep[6];
      float dummy[63];
      lm_linearise_f(sUn, sVn, sPhin, sFn, dummy);
    }
    __syncthreads();
    acc[0] = thread_cost(p, tot, sFn, t2);
    block_sum(acc, 1, red);
    if (threadIdx.x == 0) {
      if (red[0][0] < sBest) {
        sBest = red[0][0];
        for (int i = 0; i < 9; ++i) { sU[i] = sUn[i]; sV[i] = sVn[i]; }
        *sPhi = sPhin;
        sLam = fmaxf(sLam * 0.1f, 1e-6f);
        lm_linearise_f(sU, sV, *sPhi, sF, sD);
      } else {
        sLam = fminf(sLam * 10.f, 1e6f);
      }
    }
    __syncthreads();
  }
}
extern "C" __global__ void lm_refine(const float* __restrict__ pts, const int* __restrict__ counts,
                                     const float* __restrict__ thr2, const int* __restrict__ pairs,
                                     float* __restrict__ rt, int iterations, int N, int n) {
  __shared__ float sR[9], st[3];
  __shared__ float red[8][NT];
  const int a = blockIdx.x, pair = pairs[a];
  const int tot = min(n, counts[pair]);
  const float4* p = (const float4*)(pts + (long long)pair * N * 4);
  const float t2 = thr2[pair];
  if (threadIdx.x < 9) sR[threadIdx.x] = rt[a * 12 + threadIdx.x];
  else if (threadIdx.x < 12) st[threadIdx.x - 9] = rt[a * 12 + threadIdx.x];
  float acc[NT];
  lm_run(p, tot, t2, iterations, sR, st, acc, red);
  if (threadIdx.x < 9) rt[a * 12 + threadIdx.x] = sR[threadIdx.x];
  else if (threadIdx.x < 12) rt[a * 12 + threadIdx.x] = st[threadIdx.x - 9];
}

// ---------------------------------------------------------------------------------------------------------------------
// Final polish of one model per pair in one launch: weighted 8-point refits on the inliers, recoverPose (the
// decomposition with most inliers in front of both cameras), LM refinement, the final inlier mask. Replaces ~1300
// small torch ops (two SVDs, an eigendecomposition) on the single-pair path.

// Cyclic Jacobi eigendecomposition of the symmetric n x n matrix a (row-major, n <= 9, overwritten: eigenvalues on
// the diagonal); v receives the eigenvectors as columns. Thread-local.
__device__ void jacobi_eig(float* a, float* v, int n) {
  for (int i = 0; i < n * n; ++i) v[i] = (i % (n + 1) == 0) ? 1.f : 0.f;
  for (int sweep = 0; sweep < 20; ++sweep) {
    float off = 0.f, dg = 0.f;
    for (int i = 0; i < n; ++i) {
      dg += a[i * n + i] * a[i * n + i];
      for (int j = i + 1; j < n; ++j) off += a[i * n + j] * a[i * n + j];
    }
    if (off <= 1e-16f * dg) break;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        float apq = a[p * n + q];
        if (fabsf(apq) < 1e-30f) continue;
        float th = (a[q * n + q] - a[p * n + p]) / (2.f * apq);
        float t = (th >= 0.f ? 1.f : -1.f) / (fabsf(th) + sqrtf(th * th + 1.f));
        float c = rsqrtf(t * t + 1.f), s = t * c;
        for (int k = 0; k < n; ++k) {
          float akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {
          float apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; ++k) {
          float vkp = v[k * n + p], vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
  }
}
__device__ __forceinline__ float det3(const float* m) {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}
// SVD of the 3 x 3 matrix e (row-major): V (columns, det +1), singular values descending, U columns u0, u1 = E v_i /
// s_i and u2 = u0 x u1 (det +1; e has rank 2 for an essential matrix). False when e is (nearly) rank 1.
__device__ bool svd3(const float* e, float* U, float* V, float* sig) {
  float a[9], v[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) a[i * 3 + j] = e[i] * e[j] + e[3 + i] * e[3 + j] + e[6 + i] * e[6 + j];
  jacobi_eig(a, v, 3);
  int ord[3] = {0, 1, 2};
  for (int i = 0; i < 2; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (a[ord[j] * 3 + ord[j]] > a[ord[i] * 3 + ord[i]]) { int x = ord[i]; ord[i] = ord[j]; ord[j] = x; }
  for (int c = 0; c < 3; ++c) {
    sig[c] = sqrtf(fmaxf(a[ord[c] * 3 + ord[c]], 0.f));
    for (int k = 0; k < 3; ++k) V[k * 3 + c] = v[k * 3 + ord[c]];
  }
  if (det3(V) < 0.f)
    for (int k = 0; k < 3; ++k) V[k * 3 + 2] = -V[k * 3 + 2];
  for (int c = 0; c < 2; ++c) {
    if (!(sig[c] > 1e-12f * sig[0]) || !(sig[0] > 0.f)) return false;
    for (int k = 0; k < 3; ++k)
      U[k * 3 + c] = (e[k * 3] * V[c] + e[k * 3 + 1] * V[3 + c] + e[k * 3 + 2] * V[6 + c]) / sig[c];
  }
  U[2] = U[3] * U[7] - U[6] * U[4];
  U[5] = U[6] * U[1] - U[0] * U[7];
  U[8] = U[0] * U[4] - U[3] * U[1];
  bool fin = true;
  for (int k = 0; k < 9; ++k) fin = fin && isfinite(U[k]) && isfinite(V[k]);
  return fin;
}
// Thread 0: the two rotations U W V^T, U W^T V^T and t = u2 of e; false when e is degenerate.
__device__ bool decompose(const float* e, float* R1, float* R2, float* t) {
  float U[9], V[9], sig[3], tmp[9];
  if (!svd3(e, U, V, sig)) return false;
  const float W[9] = {0.f, -1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
  const float WT[9] = {0.f, 1.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
  float VT[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) VT[i * 3 + j] = V[j * 3 + i];
  mat_mul(U, W, tmp);
  mat_mul(tmp, VT, R1);
  mat_mul(U, WT, tmp);
  mat_mul(tmp, VT, R2);
  for (int k = 0; k < 3; ++k) t[k] = U[k * 3 + 2];
  return true;
}
// Thread 0: the essential matrix (equal singular values) or fundamental matrix (rank 2) closest to the 9-vector v,
// unit norm; false when v is degenerate.
__device__ bool project_rank2(const float* v, float* e, bool fundamental) {
  float U[9], V[9], sig[3];
  if (!svd3(v, U, V, sig)) return false;
  float s0 = rsqrtf(2.f), s1 = s0;
  if (fundamental) {
    const float inv = rsqrtf(sig[0] * sig[0] + sig[1] * sig[1]);
    s0 = sig[0] * inv;
    s1 = sig[1] * inv;
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) e[i * 3 + j] = s0 * U[i * 3] * V[j * 3] + s1 * U[i * 3 + 1] * V[j * 3 + 1];
  return true;
}
// Per-thread MSAC score (acc[0]) and inlier count (acc[1]) of E; writes the mask when given.
__device__ void score_pass(const float4* p, int tot, int N, const float* E, float t2, float* acc, unsigned char* mk) {
  acc[0] = acc[1] = 0.f;
  for (int i = threadIdx.x; i < (mk ? N : tot); i += blockDim.x) {
    float g = i < tot ? gain(p[i], E, t2) : 0.f;
    acc[0] += g;
    acc[1] += g > 0.f;
    if (mk) mk[i] = (unsigned char)(g > 0.f);
  }
}
// Per-thread upper triangle (45) of sum kron kron^T over the inliers of E, and their count (acc[45]).
__device__ void ata_pass(const float4* p, int tot, const float* E, float t2, float* acc) {
  for (int k = 0; k < 46; ++k) acc[k] = 0.f;
  for (int i = threadIdx.x; i < tot; i += blockDim.x) {
    const float4 m = p[i];
    if (!(gain(m, E, t2) > 0.f)) continue;
    const float row[9] = {m.z * m.x, m.z * m.y, m.z, m.w * m.x, m.w * m.y, m.w, m.x, m.y, 1.f};
    int q = 0;
    for (int u = 0; u < 9; ++u)
      for (int v = u; v < 9; ++v) acc[q++] += row[u] * row[v];
    acc[45] += 1.f;
  }
}
// Per-thread counts (acc[0..3]) of the inliers of E in front of both cameras for (R1, t), (R1, -t), (R2, t), (R2, -t)
// (midpoint triangulation depths).
__device__ void cheirality_pass(const float4* p, int tot, const float* E, float t2, const float* R1, const float* R2,
                                const float* t, float* acc) {
  for (int k = 0; k < 4; ++k) acc[k] = 0.f;
  for (int i = threadIdx.x; i < tot; i += blockDim.x) {
    const float4 m = p[i];
    if (!(gain(m, E, t2) > 0.f)) continue;
    const float xx = m.z * m.z + m.w * m.w + 1.f, xt = m.z * t[0] + m.w * t[1] + t[2];
    for (int r = 0; r < 2; ++r) {
      const float* R = r ? R2 : R1;
      float y[3];
      for (int k = 0; k < 3; ++k) y[k] = R[k * 3] * m.x + R[k * 3 + 1] * m.y + R[k * 3 + 2];
      const float yy = y[0] * y[0] + y[1] * y[1] + y[2] * y[2], yx = y[0] * m.z + y[1] * m.w + y[2];
      const float yt = y[0] * t[0] + y[1] * t[1] + y[2] * t[2];
      const float det = yy * xx - yx * yx;
      const float l1 = (-yt * xx + yx * xt) / det, l2 = (yy * xt - yx * yt) / det;
      acc[2 * r] += (l1 > 0.f && l2 > 0.f);
      acc[2 * r + 1] += (l1 < 0.f && l2 < 0.f);
    }
  }
}
// All threads: recoverPose of sE into sR, st; sFront the inliers in front (-1 when sE is degenerate).
__device__ void recover(const float4* p, int tot, float t2, const float* sE, float* sR, float* st, float* sR1,
                        float* sR2, float* sT, int* sFront, float* acc, float (*red)[NT]) {
  __shared__ int sOk;
  if (threadIdx.x == 0) sOk = decompose(sE, sR1, sR2, sT);
  __syncthreads();
  if (!sOk) {
    if (threadIdx.x == 0) *sFront = -1;
    __syncthreads();
    return;
  }
  cheirality_pass(p, tot, sE, t2, sR1, sR2, sT, acc);
  block_sum(acc, 4, red);
  if (threadIdx.x == 0) {
    int best = 0;
    for (int k = 1; k < 4; ++k) best = red[0][k] > red[0][best] ? k : best;
    const float* R = best < 2 ? sR1 : sR2;
    for (int k = 0; k < 9; ++k) sR[k] = R[k];
    for (int k = 0; k < 3; ++k) st[k] = (best & 1) ? -sT[k] : sT[k];
    *sFront = (int)red[0][best];
  }
  __syncthreads();
}
// e [P, 9] in: the RANSAC models, out: the polished ones; r [P, 9], t [P, 3], cnt [P] inlier counts, ok [P] int32,
// mask [P, N] uint8. fundamental: the models are fundamental matrices (rank-2 refits, no pose: r, t are the
// identity and (0, 0, 1), ok = at least `minimum` inliers). One block per pair, block 256.
extern "C" __global__ void polish(const float* __restrict__ pts, const int* __restrict__ counts,
                                  const float* __restrict__ thr2, float* __restrict__ e, float* __restrict__ r,
                                  float* __restrict__ t, int* __restrict__ cnt, int* __restrict__ ok,
                                  unsigned char* __restrict__ mask, int refits, int lm_iterations, int N,
                                  int fundamental, int minimum) {
  __shared__ float sE[9], sEn[9], sR[9], st[3], sR1[9], sR2[9], sT[3], sBest;
  __shared__ int sFront, sTry;
  __shared__ float red[8][NT];
  const int pair = blockIdx.x;
  const int tot = counts[pair];
  const float4* p = (const float4*)(pts + (long long)pair * N * 4);
  const float t2 = thr2[pair];
  if (threadIdx.x < 9) sE[threadIdx.x] = e[pair * 9 + threadIdx.x];
  __syncthreads();
  float acc[NT];
  score_pass(p, tot, N, sE, t2, acc, nullptr);
  block_sum(acc, 2, red);
  if (threadIdx.x == 0) sBest = red[0][0];
  // Weighted 8-point refits on the inliers, kept while the MSAC score rises.
  for (int it = 0; it < refits; ++it) {
    ata_pass(p, tot, sE, t2, acc);
    block_sum(acc, 46, red);
    if (threadIdx.x == 0) {
      sTry = 0;
      if (red[0][45] >= 8.f) {
        float A[81], V[81];
        int q = 0;
        for (int u = 0; u < 9; ++u)
          for (int v = u; v < 9; ++v) A[u * 9 + v] = A[v * 9 + u] = red[0][q++];
        jacobi_eig(A, V, 9);
        int lo = 0;
        for (int k = 1; k < 9; ++k) lo = A[k * 9 + k] < A[lo * 9 + lo] ? k : lo;
        float vec[9];
        for (int k = 0; k < 9; ++k) vec[k] = V[k * 9 + lo];
        sTry = project_rank2(vec, sEn, fundamental != 0);
      }
    }
    __syncthreads();
    if (!sTry) continue;
    score_pass(p, tot, N, sEn, t2, acc, nullptr);
    block_sum(acc, 2, red);
    if (threadIdx.x == 0 && red[0][0] > sBest) {
      sBest = red[0][0];
      for (int k = 0; k < 9; ++k) sE[k] = sEn[k];
    }
    __syncthreads();
  }
  // Non-linear refinement, kept when the score rises: of (R, t) from the decomposition, or of the rank-2
  // parametrisation U diag(cos p, sin p, 0) V^T of a fundamental matrix.
  if (lm_iterations > 0 && fundamental) {
    if (threadIdx.x == 0) {
      float sig[3];
      sTry = svd3(sE, sR1, sR2, sig);
      sT[0] = atan2f(sig[1], sig[0]);
    }
    __syncthreads();
    if (sTry) {
      lm_run_f(p, tot, t2, lm_iterations, sR1, sR2, sT, sEn, acc, red);
      score_pass(p, tot, N, sEn, t2, acc, nullptr);
      block_sum(acc, 2, red);
      if (threadIdx.x == 0 && red[0][0] > sBest) {
        sBest = red[0][0];
        for (int k = 0; k < 9; ++k) sE[k] = sEn[k];
      }
      __syncthreads();
    }
  } else if (lm_iterations > 0) {
    recover(p, tot, t2, sE, sR, st, sR1, sR2, sT, &sFront, acc, red);
    if (sFront > 0) {
      lm_run(p, tot, t2, lm_iterations, sR, st, acc, red);
      if (threadIdx.x == 0) {
        float tx[9];
        cross_mat(st, tx);
        mat_mul(tx, sR, sEn);
        float nrm = 0.f;
        for (int k = 0; k < 9; ++k) nrm += sEn[k] * sEn[k];
        nrm = rsqrtf(fmaxf(nrm, 1e-30f));
        for (int k = 0; k < 9; ++k) sEn[k] *= nrm;
      }
      __syncthreads();
      score_pass(p, tot, N, sEn, t2, acc, nullptr);
      block_sum(acc, 2, red);
      if (threadIdx.x == 0 && red[0][0] > sBest) {
        sBest = red[0][0];
        for (int k = 0; k < 9; ++k) sE[k] = sEn[k];
      }
      __syncthreads();
    }
  }
  // The final decomposition, inlier mask and count.
  if (fundamental) {
    if (threadIdx.x == 0) sFront = 1;
  } else {
    recover(p, tot, t2, sE, sR, st, sR1, sR2, sT, &sFront, acc, red);
  }
  score_pass(p, tot, N, sE, t2, acc, mask + (long long)pair * N);
  block_sum(acc, 2, red);
  if (threadIdx.x == 0) {
    const int count = (int)red[0][1];
    const bool good = count >= minimum && (fundamental || sFront > 0);
    cnt[pair] = count;
    ok[pair] = good;
    for (int k = 0; k < 9; ++k) {
      e[pair * 9 + k] = sE[k];
      r[pair * 9 + k] = good ? sR[k] : (k % 4 == 0 ? 1.f : 0.f);
    }
    for (int k = 0; k < 3; ++k) t[pair * 3 + k] = good ? st[k] : (k == 2 ? 1.f : 0.f);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Absolute pose for `fastforks.matching.estimate_absolute_pose`: P3P hypotheses (Lambda Twist), reprojection
// scoring (the Absolute model above) and the final Levenberg-Marquardt of [R | t]. Matches: pts [P, N, 8] fp32
// (u, v, X, Y, Z, 0, 0, 0), u, v normalised image coordinates of the 3D point (X, Y, Z); a model [R | t] (12 floats,
// R row-major) maps world points to the camera: x_cam = R X + t.

__device__ __forceinline__ double det3s(const double* s) {  // symmetric (xx, xy, xz, yy, yz, zz)
  return s[0] * (s[3] * s[5] - s[4] * s[4]) - s[1] * (s[1] * s[5] - s[4] * s[2]) + s[2] * (s[1] * s[4] - s[3] * s[2]);
}
__device__ __forceinline__ double tr_adj(const double* s, const double* t) {  // tr(adj(S) T), both symmetric
  double a00 = s[3] * s[5] - s[4] * s[4], a11 = s[0] * s[5] - s[2] * s[2], a22 = s[0] * s[3] - s[1] * s[1];
  double a01 = s[2] * s[4] - s[1] * s[5], a02 = s[1] * s[4] - s[3] * s[2], a12 = s[1] * s[2] - s[0] * s[4];
  return a00 * t[0] + a11 * t[3] + a22 * t[5] + 2.0 * (a01 * t[1] + a02 * t[2] + a12 * t[4]);
}
__device__ __forceinline__ double quad_form(const double* s, const double* u, const double* v) {  // u^T S v
  return u[0] * (s[0] * v[0] + s[1] * v[1] + s[2] * v[2]) + u[1] * (s[1] * v[0] + s[3] * v[1] + s[4] * v[2]) +
         u[2] * (s[2] * v[0] + s[4] * v[1] + s[5] * v[2]);
}
__device__ __forceinline__ void cross3d(const double* a, const double* b, double* c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}
// The real roots of c3 g^3 + c2 g^2 + c1 g + c0 (Cardano / trigonometric, then Newton polish), their count.
__device__ int cubic_roots(double c3, double c2, double c1, double c0, double* r) {
  const double scale = fmax(fmax(fabs(c3), fabs(c2)), fmax(fabs(c1), fabs(c0)));
  if (!(scale > 0.0)) return 0;
  int n = 0;
  if (fabs(c3) < 1e-12 * scale) {
    if (fabs(c2) < 1e-12 * scale) {
      if (fabs(c1) < 1e-12 * scale) return 0;
      r[0] = -c0 / c1;
      return 1;
    }
    double d = c1 * c1 - 4.0 * c2 * c0;
    if (d < 0.0) return 0;
    d = sqrt(d);
    r[0] = (-c1 + d) / (2.0 * c2);
    r[1] = (-c1 - d) / (2.0 * c2);
    return 2;
  }
  const double a = c2 / c3, b = c1 / c3, c = c0 / c3;
  const double p = b - a * a / 3.0, q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c, sh = -a / 3.0;
  const double disc = q * q / 4.0 + p * p * p / 27.0;
  if (disc >= 0.0) {
    const double sq = sqrt(disc);
    r[0] = cbrt(-q / 2.0 + sq) + cbrt(-q / 2.0 - sq) + sh;
    n = 1;
  } else {
    const double m = 2.0 * sqrt(-p / 3.0), th = acos(fmax(-1.0, fmin(1.0, 3.0 * q / (p * m)))) / 3.0;
    for (int k = 0; k < 3; ++k) r[k] = m * cos(th - 2.0943951023931953 * k) + sh;
    n = 3;
  }
  for (int k = 0; k < n; ++k)
    for (int it = 0; it < 3; ++it) {
      double x = r[k], f = ((c3 * x + c2) * x + c1) * x + c0, df = (3.0 * c3 * x + 2.0 * c2) * x + c1;
      if (fabs(df) > 1e-300) r[k] = x - f / df;
    }
  return n;
}
// Lambda Twist P3P: the up to four poses [R | t] (row-major R, then t) of the unit bearings y[3] and world points
// X[3]. The distances l_i (x_cam_i = l_i y_i) satisfy three ellipses l^T M_k l = a_k; two homogeneous differences
// D1, D2 (l^T D l = 0) are combined into a rank-2 cone D1 + g D2 (det = 0, a cubic in g) that factors into two
// planes; each plane cuts the cone D1 (or D2) in two rays, scaled by the first ellipse; two Gauss-Newton steps on
// the three ellipses polish the distances. R from the frames of the three points in both coordinate systems.
__device__ int p3p_solve(const double y[3][3], const double X[3][3], double sol[4][12]) {
  const double* X0 = X[0]; const double* X1 = X[1]; const double* X2 = X[2];
  double d01[3], d02[3], d12[3];
  for (int k = 0; k < 3; ++k) { d01[k] = X0[k] - X1[k]; d02[k] = X0[k] - X2[k]; d12[k] = X1[k] - X2[k]; }
  const double a12 = d01[0] * d01[0] + d01[1] * d01[1] + d01[2] * d01[2];
  const double a13 = d02[0] * d02[0] + d02[1] * d02[1] + d02[2] * d02[2];
  const double a23 = d12[0] * d12[0] + d12[1] * d12[1] + d12[2] * d12[2];
  if (!(a12 > 0.0) || !(a13 > 0.0) || !(a23 > 0.0)) return 0;
  const double b12 = y[0][0] * y[1][0] + y[0][1] * y[1][1] + y[0][2] * y[1][2];
  const double b13 = y[0][0] * y[2][0] + y[0][1] * y[2][1] + y[0][2] * y[2][2];
  const double b23 = y[1][0] * y[2][0] + y[1][1] * y[2][1] + y[1][2] * y[2][2];
  // Ellipses (xx, xy, xz, yy, yz, zz) and the cones.
  const double M12[6] = {1.0, -b12, 0.0, 1.0, 0.0, 0.0}, M13[6] = {1.0, 0.0, -b13, 0.0, 0.0, 1.0};
  const double M23[6] = {0.0, 0.0, 0.0, 1.0, -b23, 1.0};
  double D1[6], D2[6];
  for (int k = 0; k < 6; ++k) { D1[k] = a23 * M12[k] - a12 * M23[k]; D2[k] = a23 * M13[k] - a13 * M23[k]; }
  double g[3];
  const int ng = cubic_roots(det3s(D2), tr_adj(D2, D1), tr_adj(D1, D2), det3s(D1), g);
  // The frame of the world points (its inverse by the adjugate).
  double n01[3], Fw[9], Fwi[9];
  cross3d(d01, d02, n01);
  for (int k = 0; k < 3; ++k) { Fw[k * 3] = d01[k]; Fw[k * 3 + 1] = d02[k]; Fw[k * 3 + 2] = n01[k]; }
  {
    double det = Fw[0] * (Fw[4] * Fw[8] - Fw[5] * Fw[7]) - Fw[1] * (Fw[3] * Fw[8] - Fw[5] * Fw[6]) + Fw[2] * (Fw[3] * Fw[7] - Fw[4] * Fw[6]);
    if (!(fabs(det) > 1e-300)) return 0;
    Fwi[0] = (Fw[4] * Fw[8] - Fw[5] * Fw[7]) / det; Fwi[1] = (Fw[2] * Fw[7] - Fw[1] * Fw[8]) / det; Fwi[2] = (Fw[1] * Fw[5] - Fw[2] * Fw[4]) / det;
    Fwi[3] = (Fw[5] * Fw[6] - Fw[3] * Fw[8]) / det; Fwi[4] = (Fw[0] * Fw[8] - Fw[2] * Fw[6]) / det; Fwi[5] = (Fw[2] * Fw[3] - Fw[0] * Fw[5]) / det;
    Fwi[6] = (Fw[3] * Fw[7] - Fw[4] * Fw[6]) / det; Fwi[7] = (Fw[1] * Fw[6] - Fw[0] * Fw[7]) / det; Fwi[8] = (Fw[0] * Fw[4] - Fw[1] * Fw[3]) / det;
  }
  int count = 0;
  for (int ig = 0; ig < ng && count == 0; ++ig) {
    double D0[6];
    for (int k = 0; k < 6; ++k) D0[k] = D1[k] + g[ig] * D2[k];
    // Eigenvalues of the rank-2 cone (one is zero): e1 > 0 > e2 for a real factorisation.
    const double tr = D0[0] + D0[3] + D0[5];
    const double mn = D0[0] * D0[3] - D0[1] * D0[1] + D0[0] * D0[5] - D0[2] * D0[2] + D0[3] * D0[5] - D0[4] * D0[4];
    double disc = tr * tr - 4.0 * mn;
    if (!(disc > 0.0)) continue;
    disc = sqrt(disc);
    const double e1 = (tr + disc) / 2.0, e2 = (tr - disc) / 2.0;
    if (!(e1 > 0.0) || !(e2 < 0.0)) continue;
    double v1[3], v2[3];
    for (int w = 0; w < 2; ++w) {
      const double ev = w == 0 ? e1 : e2;
      double A[3][3] = {{D0[0] - ev, D0[1], D0[2]}, {D0[1], D0[3] - ev, D0[4]}, {D0[2], D0[4], D0[5] - ev}};
      double c[3][3], best = -1.0;
      double* v = w == 0 ? v1 : v2;
      cross3d(A[0], A[1], c[0]);
      cross3d(A[0], A[2], c[1]);
      cross3d(A[1], A[2], c[2]);
      for (int k = 0; k < 3; ++k) {
        double nn = c[k][0] * c[k][0] + c[k][1] * c[k][1] + c[k][2] * c[k][2];
        if (nn > best) { best = nn; for (int j = 0; j < 3; ++j) v[j] = c[k][j]; }
      }
      if (!(best > 0.0)) { best = -1.0; break; }
      const double inv = 1.0 / sqrt(best);
      for (int j = 0; j < 3; ++j) v[j] *= inv;
    }
    const double s1 = sqrt(e1), s2 = sqrt(-e2);
    const double* Q = fabs(g[ig]) >= 1.0 ? D1 : D2;  // the better-conditioned cone to cut with the planes
    for (int pl = 0; pl < 2 && count < 4; ++pl) {
      double w[3];
      for (int j = 0; j < 3; ++j) w[j] = s1 * v1[j] + (pl == 0 ? s2 : -s2) * v2[j];
      int kp = 0;
      for (int j = 1; j < 3; ++j) kp = fabs(w[j]) > fabs(w[kp]) ? j : kp;
      const int i1 = (kp + 1) % 3, i2 = (kp + 2) % 3;
      double r1[3] = {0.0, 0.0, 0.0}, r2[3] = {0.0, 0.0, 0.0};
      r1[i1] = 1.0; r1[kp] = -w[i1] / w[kp];
      r2[i2] = 1.0; r2[kp] = -w[i2] / w[kp];
      const double al = quad_form(Q, r1, r1), be = quad_form(Q, r1, r2), de = quad_form(Q, r2, r2);
      double dq = be * be - al * de;
      if (dq < 0.0) {
        if (dq < -1e-9 * (be * be + fabs(al * de))) continue;
        dq = 0.0;
      }
      dq = sqrt(dq);
      for (int sg = 0; sg < 2 && count < 4; ++sg) {
        double l1, l2;
        const double rt = sg == 0 ? dq : -dq;
        if (fabs(al) >= fabs(de)) { l1 = (-be + rt) / al; l2 = 1.0; }
        else { l1 = 1.0; l2 = (-be + rt) / de; }
        double l[3];
        for (int j = 0; j < 3; ++j) l[j] = l1 * r1[j] + l2 * r2[j];
        const double q12 = quad_form(M12, l, l);
        if (!(q12 > 0.0)) continue;
        double sc = sqrt(a12 / q12);
        if (l[0] + l[1] + l[2] < 0.0) sc = -sc;
        for (int j = 0; j < 3; ++j) l[j] *= sc;
        if (!(l[0] > 0.0) || !(l[1] > 0.0) || !(l[2] > 0.0)) continue;
        // Gauss-Newton on the three ellipses.
        for (int it = 0; it < 2; ++it) {
          const double f0 = quad_form(M12, l, l) - a12, f1 = quad_form(M13, l, l) - a13, f2 = quad_form(M23, l, l) - a23;
          double J[3][3] = {{2.0 * (l[0] - b12 * l[1]), 2.0 * (l[1] - b12 * l[0]), 0.0},
                            {2.0 * (l[0] - b13 * l[2]), 0.0, 2.0 * (l[2] - b13 * l[0])},
                            {0.0, 2.0 * (l[1] - b23 * l[2]), 2.0 * (l[2] - b23 * l[1])}};
          const double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
          if (!(fabs(det) > 1e-300)) break;
          const double dx = (f0 * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (f1 * J[2][2] - J[1][2] * f2) + J[0][2] * (f1 * J[2][1] - J[1][1] * f2)) / det;
          const double dy = (J[0][0] * (f1 * J[2][2] - J[1][2] * f2) - f0 * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) + J[0][2] * (J[1][0] * f2 - f1 * J[2][0])) / det;
          const double dz = (J[0][0] * (J[1][1] * f2 - f1 * J[2][1]) - J[0][1] * (J[1][0] * f2 - f1 * J[2][0]) + f0 * (J[1][0] * J[2][1] - J[1][1] * J[2][0])) / det;
          l[0] -= dx; l[1] -= dy; l[2] -= dz;
        }
        if (!(l[0] > 0.0) || !(l[1] > 0.0) || !(l[2] > 0.0)) continue;
        // Camera points, their frame, R = Fc Fw^-1, t = x_cam_0 - R X0.
        double c0[3], c1[3], c2[3], e01[3], e02[3], nc[3], Fc[9], R[9];
        for (int j = 0; j < 3; ++j) { c0[j] = l[0] * y[0][j]; c1[j] = l[1] * y[1][j]; c2[j] = l[2] * y[2][j]; e01[j] = c0[j] - c1[j]; e02[j] = c0[j] - c2[j]; }
        cross3d(e01, e02, nc);
        for (int j = 0; j < 3; ++j) { Fc[j * 3] = e01[j]; Fc[j * 3 + 1] = e02[j]; Fc[j * 3 + 2] = nc[j]; }
        bool fin = true;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) {
            R[i * 3 + j] = Fc[i * 3] * Fwi[j] + Fc[i * 3 + 1] * Fwi[3 + j] + Fc[i * 3 + 2] * Fwi[6 + j];
            fin = fin && isfinite(R[i * 3 + j]);
          }
        if (!fin) continue;
        for (int k = 0; k < 9; ++k) sol[count][k] = R[k];
        for (int i = 0; i < 3; ++i) sol[count][9 + i] = c0[i] - (R[i * 3] * X0[0] + R[i * 3 + 1] * X0[1] + R[i * 3 + 2] * X0[2]);
        ++count;
      }
    }
  }
  return count;
}
// One thread per hypothesis (a, h): the up to four poses of its three matches. pts [P, N, 8]; pairs [A]; idx
// [A, H, 3]; out [A, H, 4, 12]; valid [A, H, 4] int32.
extern "C" __global__ void p3p_hypotheses(const float* __restrict__ pts, const int* __restrict__ pairs,
                                          const int* __restrict__ idx, float* __restrict__ out,
                                          int* __restrict__ valid, int N, int H, int A) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= A * H) return;
  const int a = g / H, h = g % H, pair = pairs[a];
  double y[3][3], X[3][3], sol[4][12];
  for (int i = 0; i < 3; ++i) {
    const float* m = pts + ((long long)pair * N + idx[(long long)g * 3 + i]) * 8;
    const double inv = 1.0 / sqrt(1.0 + (double)m[0] * m[0] + (double)m[1] * m[1]);
    y[i][0] = m[0] * inv; y[i][1] = m[1] * inv; y[i][2] = inv;
    X[i][0] = m[2]; X[i][1] = m[3]; X[i][2] = m[4];
  }
  const int n = p3p_solve(y, X, sol);
  for (int s = 0; s < 4; ++s) {
    valid[(long long)g * 4 + s] = s < n;
    for (int k = 0; k < 12; ++k) out[((long long)g * 4 + s) * 12 + k] = s < n ? (float)sol[s][k] : 0.f;
  }
}

// Absolute-pose LM: per-thread J^T J (21), J^T r (6) and cost (1) of the truncated reprojection error of [R | t]
// by (omega, tau): R exp([omega]_x), t + tau. Points behind the camera cost thr2 and give no Jacobian.
#define NLMA 28
__device__ void lm_accumulate_abs(const float4* p, int tot, const float* M, float t2, float* acc) {
  for (int k = 0; k < NLMA; ++k) acc[k] = 0.f;
  for (int i = threadIdx.x; i < tot; i += blockDim.x) {
    const float4 m0 = p[2 * i], m1 = p[2 * i + 1];
    const float X[3] = {m0.z, m0.w, m1.x};
    float c[3];
    for (int r = 0; r < 3; ++r) c[r] = M[r * 3] * X[0] + M[r * 3 + 1] * X[1] + M[r * 3 + 2] * X[2] + M[9 + r];
    if (!(c[2] > 0.f)) { acc[27] += t2; continue; }
    const float iz = 1.f / c[2];
    const float res[2] = {c[0] * iz - m0.x, c[1] * iz - m0.y};
    const float e2 = res[0] * res[0] + res[1] * res[1];
    if (e2 >= t2) { acc[27] += t2; continue; }
    acc[27] += e2;
    // dc/d(omega) = -R [X]_x, dc/d(tau) = I; d(res)/dc = [[iz, 0, -c0 iz^2], [0, iz, -c1 iz^2]].
    float dc[3][6];
    for (int r = 0; r < 3; ++r) {
      const float* Rr = M + r * 3;
      dc[r][0] = -(Rr[1] * X[2] - Rr[2] * X[1]);
      dc[r][1] = -(Rr[2] * X[0] - Rr[0] * X[2]);
      dc[r][2] = -(Rr[0] * X[1] - Rr[1] * X[0]);
      for (int k = 0; k < 3; ++k) dc[r][3 + k] = r == k ? 1.f : 0.f;
    }
    float J[2][6];
    for (int k = 0; k < 6; ++k) {
      J[0][k] = (dc[0][k] - c[0] * iz * dc[2][k]) * iz;
      J[1][k] = (dc[1][k] - c[1] * iz * dc[2][k]) * iz;
    }
    int q = 0;
    for (int u = 0; u < 6; ++u)
      for (int v = u; v < 6; ++v) acc[q++] += J[0][u] * J[0][v] + J[1][u] * J[1][v];
    for (int k = 0; k < 6; ++k) acc[21 + k] += J[0][k] * res[0] + J[1][k] * res[1];
  }
}
// Per-thread MSAC score (acc[0]) and inlier count (acc[1]) of [R | t]; writes the mask when given.
__device__ void score_pass_abs(const float4* p, int tot, int N, const float* M, float t2, float* acc,
                               unsigned char* mk) {
  acc[0] = acc[1] = 0.f;
  for (int i = threadIdx.x; i < (mk ? N : tot); i += blockDim.x) {
    float g = i < tot ? Absolute::gain(p + 2 * i, M, t2) : 0.f;
    acc[0] += g;
    acc[1] += g > 0.f;
    if (mk) mk[i] = (unsigned char)(g > 0.f);
  }
}
// The final stage of one pose per pair: LM on the truncated reprojection cost (kept when the MSAC score rises),
// the inlier mask and count. m [P, 12] in / out; cnt [P]; ok [P] int32 (at least `minimum` inliers); mask [P, N]
// uint8. One block per pair, block 256.
extern "C" __global__ void polish_abs(const float* __restrict__ pts, const int* __restrict__ counts,
                                      const float* __restrict__ thr2, float* __restrict__ m, int* __restrict__ cnt,
                                      int* __restrict__ ok, unsigned char* __restrict__ mask, int lm_iterations,
                                      int N, int minimum) {
  __shared__ float sM[12], sMn[12], sStep[6], sLam, sBest, sCost;
  __shared__ float red[8][NT];
  const int pair = blockIdx.x;
  const int tot = counts[pair];
  const float4* p = (const float4*)(pts + (long long)pair * N * 8);
  const float t2 = thr2[pair];
  if (threadIdx.x < 12) sM[threadIdx.x] = m[pair * 12 + threadIdx.x];
  if (threadIdx.x == 0) sLam = 1e-3f;
  __syncthreads();
  float acc[NT];
  score_pass_abs(p, tot, N, sM, t2, acc, nullptr);
  block_sum(acc, 2, red);
  if (threadIdx.x == 0) {
    sBest = red[0][0];
    sCost = t2 * (tot - red[0][0]);
  }
  for (int it = 0; it < lm_iterations; ++it) {
    lm_accumulate_abs(p, tot, sM, t2, acc);
    block_sum(acc, NLMA, red);
    if (threadIdx.x == 0) {
      lm_step<6>(red[0], sLam, sStep);
      float Rw[9];
      exp_so3(sStep, Rw);
      mat_mul(sM, Rw, sMn);
      for (int k = 0; k < 3; ++k) sMn[9 + k] = sM[9 + k] + sStep[3 + k];
    }
    __syncthreads();
    score_pass_abs(p, tot, N, sMn, t2, acc, nullptr);
    block_sum(acc, 2, red);
    if (threadIdx.x == 0) {
      const float cost = t2 * (tot - red[0][0]);
      if (cost < sCost) {
        sCost = cost;
        sBest = red[0][0];
        for (int k = 0; k < 12; ++k) sM[k] = sMn[k];
        sLam = fmaxf(sLam * 0.1f, 1e-6f);
      } else {
        sLam = fminf(sLam * 10.f, 1e6f);
      }
    }
    __syncthreads();
  }
  score_pass_abs(p, tot, N, sM, t2, acc, mask + (long long)pair * N);
  block_sum(acc, 2, red);
  if (threadIdx.x == 0) {
    const int count = (int)red[0][1];
    cnt[pair] = count;
    ok[pair] = count >= minimum;
    for (int k = 0; k < 12; ++k) m[pair * 12 + k] = sM[k];
  }
}
