"""Pose RANSAC on Apple MLX: Metal kernels (`mx.fast.metal_kernel`), a port of `fastforks.kernels.ransac` (CUDA).

- `hypotheses`: one thread per minimal sample. Five matches: Nister's five-point solver (up to ten essential matrices);
  seven: the seven-point solver (up to three fundamental matrices); three 2D-3D matches: Lambda Twist P3P (up to four
  poses `[R | t]`).
- `score`: MSAC scores (truncated quadratic Sampson or reprojection loss) and inlier counts of candidates against the
  matches of a pair: one thread per candidate for many candidates, one threadgroup per candidate for few.
- `polish` / `polish_abs`: the final stage of one model per pair in one launch: weighted 8-point refits on the inliers,
  `recoverPose`, Levenberg-Marquardt, the inlier mask.

Apple GPUs have no float64, which the CUDA solver uses for the Gauss-Jordan elimination, the degree-10 polynomial and
its root polish. Here these run in float32 (`DF=0`) or in double-float arithmetic (`DF=1`: a float pair, ~48
mantissa bits, about 10x the cost of float32), the root search (Aberth-Ehrlich) in float32 complex as on CUDA. The 3x3
SVDs (Jacobi on `E^T E`) and the LM run in float32, as on CUDA.

Matches: `pts` `[P, N, 4]` float32 (u1, v1, u2, v2 normalised image coordinates), or `[P, N, 8]` (u, v, X, Y, Z, 0,
0, 0) for the absolute pose, the valid matches of a pair first; a launch works on a subset `pairs` `[A]` of the pairs.
"""
# ruff: noqa: ANN001, ANN002, ANN201, ANN202, PLR0913, PLR0917, E501 - Metal source lines; array-in / array-out functions named after the CUDA library's ones

from __future__ import annotations

from fastforks._core.mlx import metal_kernel, mx

THREADS = 256  # threadgroup of the scoring and polish kernels (8 SIMD groups of 32: `block_sum` assumes it)
SMALL_THREADS, SMALL = 32, 4096  # the hypotheses kernels' threadgroup for launches of at most SMALL samples
MANY = 64  # candidates per pair from which the one-thread-per-candidate scoring wins
# sample size -> (kernel, solutions per sample, floats per model)
SOLVERS = {5: ("e5pt", 10, 9), 7: ("f7pt", 3, 9), 3: ("p3p", 4, 12)}
DOUBLE_FLOAT = 1  # the five- and seven-point solvers' elimination in double-float (1) or float32 (0)

_HEADER = r"""
using namespace metal;

#define NSOL 10
#define ITER_ABERTH 48
#define NT 46
#define NLM 21
#define NLMF 36
#define NLMA 28
#define TPB 256
#define CHUNK 1024

inline void bar() { threadgroup_barrier(mem_flags::mem_threadgroup); }

// ---- double-float arithmetic (hi + lo, |lo| <= ulp(hi) / 2): float64-like precision with fp32 range -------------
struct df { float hi; float lo; };
inline df quick_two_sum(float a, float b) { float s = a + b; df r; r.hi = s; r.lo = b - (s - a); return r; }
inline df two_sum(float a, float b) {
  float s = a + b, bb = s - a;
  df r; r.hi = s; r.lo = (a - (s - bb)) + (b - bb); return r;
}
inline df operator+(df a, df b) { df s = two_sum(a.hi, b.hi); return quick_two_sum(s.hi, s.lo + a.lo + b.lo); }
inline df operator-(df a) { df r; r.hi = -a.hi; r.lo = -a.lo; return r; }
inline df operator-(df a, df b) { return a + (-b); }
inline df operator*(df a, df b) {
  float p = a.hi * b.hi, e = fma(a.hi, b.hi, -p);
  e = fma(a.hi, b.lo, e);
  e = fma(a.lo, b.hi, e);
  return quick_two_sum(p, e);
}
inline df operator/(df a, df b) {
  float q1 = a.hi / b.hi;
  df q; q.hi = q1; q.lo = 0.f;
  df r = a - b * q;
  return quick_two_sum(q1, r.hi / b.hi);
}
template <typename S> inline S from_f(float x);
template <> inline float from_f<float>(float x) { return x; }
template <> inline df from_f<df>(float x) { df r; r.hi = x; r.lo = 0.f; return r; }
inline float tof(float x) { return x; }
inline float tof(df x) { return x.hi + x.lo; }
inline float mag(float x) { return fabs(x); }
inline float mag(df x) { return fabs(x.hi); }
inline bool is_zero(float x) { return x == 0.f; }
inline bool is_zero(df x) { return x.hi == 0.f; }

// ---- five-point solver ------------------------------------------------------------------------------------------
// Monomial bases: linear (x, y, z, 1), quadratic (x^2, xy, xz, x, y^2, yz, y, z^2, z, 1), cubic in Nister's order.
constant int LL[16] = {0, 1, 2, 3, 1, 4, 5, 6, 2, 5, 7, 8, 3, 6, 8, 9};
constant int QL[40] = {0, 2, 4, 5, 2, 3, 8, 9, 4, 8, 10, 11, 5, 9, 11, 12, 3, 1, 6, 7,
                       8, 6, 13, 14, 9, 7, 14, 15, 10, 13, 16, 17, 11, 14, 17, 18, 12, 15, 18, 19};

inline void mul_ll(thread const float* a, thread const float* b, thread float* q) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) q[LL[i * 4 + j]] += a[i] * b[j];
}
inline void mul_ql(thread const float* q, thread const float* b, float s, thread float* c) {
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 4; ++j) c[QL[i * 4 + j]] += s * q[i] * b[j];
}

// Polynomials in z, coefficients highest power first.
template <typename S, int DA, int DB>
inline void pmul(thread const S* a, thread const S* b, thread S* out) {
  for (int i = 0; i < DA + DB - 1; ++i) out[i] = from_f<S>(0.f);
  for (int i = 0; i < DA; ++i)
    for (int j = 0; j < DB; ++j) out[i + j] = out[i + j] + a[i] * b[j];
}
template <typename S, int D>
inline S peval(thread const S* p, S z) {
  S v = p[0];
  for (int i = 1; i < D; ++i) v = v * z + p[i];
  return v;
}

// Null space (9 - R vectors, nul[t * 9 + k]) of the R x 9 epipolar rows of the matches idx of pair p.
template <int R>
inline void null_space(const device float4* p, thread const int* idx, thread float* nul) {
  float q[R * 9];
  for (int r = 0; r < R; ++r) {
    float4 m = p[idx[r]];
    float row[9] = {m.z * m.x, m.z * m.y, m.z, m.w * m.x, m.w * m.y, m.w, m.x, m.y, 1.f};
    for (int s = 0; s < r; ++s) {
      float d = 0.f;
      for (int k = 0; k < 9; ++k) d += row[k] * q[s * 9 + k];
      for (int k = 0; k < 9; ++k) row[k] -= d * q[s * 9 + k];
    }
    float nrm = 0.f;
    for (int k = 0; k < 9; ++k) nrm += row[k] * row[k];
    nrm = rsqrt(max(nrm, 1e-30f));
    for (int k = 0; k < 9; ++k) q[r * 9 + k] = row[k] * nrm;
  }
  float res[9];
  for (int j = 0; j < 9; ++j) {
    float s = 1.f;
    for (int r = 0; r < R; ++r) s -= q[r * 9 + j] * q[r * 9 + j];
    res[j] = s;
  }
  for (int t = 0; t < 9 - R; ++t) {
    int best = 0;
    for (int j = 1; j < 9; ++j) best = res[j] > res[best] ? j : best;
    float v[9];
    for (int k = 0; k < 9; ++k) v[k] = 0.f;
    v[best] = 1.f;
    for (int r = 0; r < R; ++r) {
      float d = q[r * 9 + best];
      for (int k = 0; k < 9; ++k) v[k] -= d * q[r * 9 + k];
    }
    for (int s = 0; s < t; ++s) {
      float d = 0.f;
      for (int k = 0; k < 9; ++k) d += v[k] * nul[s * 9 + k];
      for (int k = 0; k < 9; ++k) v[k] -= d * nul[s * 9 + k];
    }
    float nrm = 0.f;
    for (int k = 0; k < 9; ++k) nrm += v[k] * v[k];
    nrm = rsqrt(max(nrm, 1e-30f));
    for (int k = 0; k < 9; ++k) nul[t * 9 + k] = v[k] * nrm;
    for (int j = 0; j < 9; ++j) res[j] -= nul[t * 9 + j] * nul[t * 9 + j];
    res[best] = -1.f;
  }
}

// The ten cubic constraints over the 20 cubic monomials of E = x E1 + y E2 + z E3 + E4 -> m [10 * 20].
template <typename S>
inline void constraints(thread const float* nul, thread S* m) {
  float eet[90];  // E E^T, quadratic: eet[(i * 3 + j) * 10 + k]
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      thread float* q = eet + (i * 3 + j) * 10;
      for (int k = 0; k < 10; ++k) q[k] = 0.f;
      for (int k = 0; k < 3; ++k) {
        float a[4] = {nul[3 * i + k], nul[9 + 3 * i + k], nul[18 + 3 * i + k], nul[27 + 3 * i + k]};
        float b[4] = {nul[3 * j + k], nul[9 + 3 * j + k], nul[18 + 3 * j + k], nul[27 + 3 * j + k]};
        mul_ll(a, b, q);
      }
    }
  float tr[10];
  for (int k = 0; k < 10; ++k) tr[k] = eet[k] + eet[40 + k] + eet[80 + k];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float c[20];
      for (int k = 0; k < 20; ++k) c[k] = 0.f;
      for (int k = 0; k < 3; ++k) {
        float q[10];
        for (int t = 0; t < 10; ++t) q[t] = 2.f * eet[(i * 3 + k) * 10 + t] - (i == k ? tr[t] : 0.f);
        float b[4] = {nul[3 * k + j], nul[9 + 3 * k + j], nul[18 + 3 * k + j], nul[27 + 3 * k + j]};
        mul_ql(q, b, 1.f, c);
      }
      for (int k = 0; k < 20; ++k) m[(1 + 3 * i + j) * 20 + k] = from_f<S>(c[k]);
    }
  float c[20];
  for (int k = 0; k < 20; ++k) c[k] = 0.f;
  const int cof[12] = {4, 8, 5, 7, 3, 8, 5, 6, 3, 7, 4, 6};
  for (int j = 0; j < 3; ++j) {
    float q[10];
    for (int k = 0; k < 10; ++k) q[k] = 0.f;
    int ia = cof[j * 4], id = cof[j * 4 + 1], ib = cof[j * 4 + 2], ie = cof[j * 4 + 3];
    float a[4] = {nul[ia], nul[9 + ia], nul[18 + ia], nul[27 + ia]};
    float d[4] = {nul[id], nul[9 + id], nul[18 + id], nul[27 + id]};
    float nb[4] = {-nul[ib], -nul[9 + ib], -nul[18 + ib], -nul[27 + ib]};
    float e[4] = {nul[ie], nul[9 + ie], nul[18 + ie], nul[27 + ie]};
    mul_ll(a, d, q);
    mul_ll(nb, e, q);
    float e0[4] = {nul[j], nul[9 + j], nul[18 + j], nul[27 + j]};
    mul_ql(q, e0, j == 1 ? -1.f : 1.f, c);
  }
  for (int k = 0; k < 20; ++k) m[k] = from_f<S>(c[k]);
}

// Gauss-Jordan with partial pivoting: m -> [I | X]; false if singular.
template <typename S>
inline bool gauss_jordan(thread S* m) {
  for (int c = 0; c < 10; ++c) {
    int piv = c;
    for (int r = c + 1; r < 10; ++r) piv = mag(m[r * 20 + c]) > mag(m[piv * 20 + c]) ? r : piv;
    if (mag(m[piv * 20 + c]) < 1e-14f) return false;
    if (piv != c)
      for (int k = 0; k < 20; ++k) {
        S t = m[c * 20 + k];
        m[c * 20 + k] = m[piv * 20 + k];
        m[piv * 20 + k] = t;
      }
    S inv = from_f<S>(1.f) / m[c * 20 + c];
    for (int k = c; k < 20; ++k) m[c * 20 + k] = m[c * 20 + k] * inv;
    for (int r = (c >= 4 ? 4 : 0); r < 10; ++r) {
      if (r == c) continue;
      S f = m[r * 20 + c];
      if (is_zero(f)) continue;
      for (int k = c; k < 20; ++k) m[r * 20 + k] = m[r * 20 + k] - f * m[c * 20 + k];
    }
  }
  return true;
}

inline float2 cmul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
inline float2 cdiv(float2 a, float2 b) {
  float inv = 1.f / max(b.x * b.x + b.y * b.y, 1e-30f);
  return float2((a.x * b.x + a.y * b.y) * inv, (a.y * b.x - a.x * b.y) * inv);
}
inline float2 cinv(float2 b) {
  float inv = 1.f / max(b.x * b.x + b.y * b.y, 1e-30f);
  return float2(b.x * inv, -b.y * inv);
}

// All roots of the degree-10 polynomial p (highest power first) by Aberth-Ehrlich in float32 -> z[10].
inline void roots10(thread const float* pin, thread float2* z) {
  float p[11], dp[10], mx = 0.f;
  for (int i = 0; i < 11; ++i) mx = max(mx, fabs(pin[i]));
  mx = max(mx, 1e-30f);
  for (int i = 0; i < 11; ++i) p[i] = pin[i] / mx;
  for (int i = 0; i < 10; ++i) dp[i] = p[i] * float(10 - i);
  float lead = max(fabs(p[0]), 1e-30f), radius = 0.f;
  for (int k = 1; k <= 10; ++k) radius = max(radius, pow(fabs(p[k]) / lead, 1.f / float(k)));
  radius = min(max(2.f * radius, 1e-3f), 1e6f);
  for (int i = 0; i < 10; ++i) {
    float a = 6.2831853f * (float(i) + 0.37f) / 10.f;
    z[i] = float2(radius * cos(a), radius * sin(a));
  }
  for (int it = 0; it < ITER_ABERTH; ++it) {
    float2 nz[10];
    float step = 0.f;
    for (int i = 0; i < 10; ++i) {
      float2 v = float2(p[0], 0.f), d = float2(dp[0], 0.f);
      for (int k = 1; k < 11; ++k) { v = cmul(v, z[i]); v.x += p[k]; }
      for (int k = 1; k < 10; ++k) { d = cmul(d, z[i]); d.x += dp[k]; }
      float2 ratio = cdiv(v, d), s = float2(0.f, 0.f);
      for (int j = 0; j < 10; ++j) {
        if (j == i) continue;
        s += cinv(z[i] - z[j]);
      }
      float2 den = float2(1.f - (ratio.x * s.x - ratio.y * s.y), -(ratio.x * s.y + ratio.y * s.x));
      float2 upd = cdiv(ratio, den);
      nz[i] = z[i] - upd;
      step = max(step, fabs(upd.x) + fabs(upd.y));
    }
    for (int i = 0; i < 10; ++i) z[i] = nz[i];
    if (step < 1e-6f * radius) break;
  }
}

// The up to ten essential matrices of the five matches idx of pair p -> o [10 * 9], ok [10].
template <typename S>
inline void e5pt_one(const device float4* p, thread const int* idx, device float* o, device int* ok) {
  for (int s = 0; s < NSOL; ++s) {
    ok[s] = 0;
    for (int k = 0; k < 9; ++k) o[s * 9 + k] = 0.f;
  }
  for (int r = 0; r < 5; ++r)
    for (int s = r + 1; s < 5; ++s)
      if (idx[r] == idx[s]) return;
  float nul[36];
  null_space<5>(p, idx, nul);
  S m[200];
  constraints<S>(nul, m);
  if (!gauss_jordan<S>(m)) return;
  // rows 4..9 over the tail monomials (columns 10..19): k = e - z f, l = g - z h, m = i - z j.
  S px[12], py[12], pc[15];
  for (int t = 0; t < 3; ++t) {
    thread const S* e = m + (4 + 2 * t) * 20 + 10;
    thread const S* f = m + (5 + 2 * t) * 20 + 10;
    px[t * 4] = -f[0]; px[t * 4 + 1] = e[0] - f[1]; px[t * 4 + 2] = e[1] - f[2]; px[t * 4 + 3] = e[2];
    py[t * 4] = -f[3]; py[t * 4 + 1] = e[3] - f[4]; py[t * 4 + 2] = e[4] - f[5]; py[t * 4 + 3] = e[5];
    pc[t * 5] = -f[6]; pc[t * 5 + 1] = e[6] - f[7]; pc[t * 5 + 2] = e[7] - f[8]; pc[t * 5 + 3] = e[8] - f[9];
    pc[t * 5 + 4] = e[9];
  }
  S t1[8], t2[8], t3[7], u[11], poly[11];
  pmul<S, 4, 5>(py + 4, pc + 10, t1);
  pmul<S, 5, 4>(pc + 5, py + 8, t2);
  for (int i = 0; i < 8; ++i) t1[i] = t1[i] - t2[i];
  pmul<S, 4, 8>(px, t1, poly);
  pmul<S, 4, 5>(px + 4, pc + 10, t1);
  pmul<S, 5, 4>(pc + 5, px + 8, t2);
  for (int i = 0; i < 8; ++i) t1[i] = t1[i] - t2[i];
  pmul<S, 4, 8>(py, t1, u);
  for (int i = 0; i < 11; ++i) poly[i] = poly[i] - u[i];
  pmul<S, 4, 4>(px + 4, py + 8, t3);
  pmul<S, 4, 4>(py + 4, px + 8, t2);
  for (int i = 0; i < 7; ++i) t3[i] = t3[i] - t2[i];
  pmul<S, 5, 7>(pc, t3, u);
  for (int i = 0; i < 11; ++i) poly[i] = poly[i] + u[i];
  S dpoly[10];
  for (int i = 0; i < 10; ++i) dpoly[i] = poly[i] * from_f<S>(float(10 - i));
  float pf[11];
  for (int i = 0; i < 11; ++i) pf[i] = tof(poly[i]);
  float2 z[10];
  roots10(pf, z);
  for (int s = 0; s < NSOL; ++s) {
    if (fabs(z[s].y) > 1e-3f * (1.f + fabs(z[s].x))) continue;
    S zr = from_f<S>(z[s].x);
    for (int it = 0; it < 3; ++it) {
      S d = peval<S, 10>(dpoly, zr);
      if (is_zero(d)) break;
      zr = zr - peval<S, 11>(poly, zr) / d;
    }
    if (!isfinite(tof(zr))) continue;
    S rx[3], ry[3], rc[3];
    for (int t = 0; t < 3; ++t) {
      rx[t] = peval<S, 4>(px + t * 4, zr);
      ry[t] = peval<S, 4>(py + t * 4, zr);
      rc[t] = peval<S, 5>(pc + t * 5, zr);
    }
    S bx = from_f<S>(0.f), by = bx, bw = bx;
    float bq = -1.f;
    const int pr[6] = {0, 1, 0, 2, 1, 2};
    for (int t = 0; t < 3; ++t) {
      int i = pr[t * 2], j = pr[t * 2 + 1];
      S cx = ry[i] * rc[j] - rc[i] * ry[j], cy = rc[i] * rx[j] - rx[i] * rc[j], cw = rx[i] * ry[j] - ry[i] * rx[j];
      float fx = tof(cx), fy = tof(cy), fw = tof(cw);
      float sc = sqrt(fx * fx + fy * fy + fw * fw);
      float q = sc > 0.f ? fabs(fw) / sc : -1.f;
      if (q > bq) { bq = q; bx = cx; by = cy; bw = cw; }
    }
    if (bq < 1e-8f) continue;
    float x = tof(bx / bw), y = tof(by / bw), zz = tof(zr);
    float e[9], nrm = 0.f;
    for (int k = 0; k < 9; ++k) {
      e[k] = x * nul[k] + y * nul[9 + k] + zz * nul[18 + k] + nul[27 + k];
      nrm += e[k] * e[k];
    }
    if (!(nrm > 0.f) || !isfinite(nrm)) continue;
    nrm = rsqrt(nrm);
    for (int k = 0; k < 9; ++k) o[s * 9 + k] = e[k] * nrm;
    ok[s] = 1;
  }
}

// ---- seven-point solver -----------------------------------------------------------------------------------------
template <typename S>
inline S det3s9(thread const S* m) {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}
inline float cbrt_(float x) { return x < 0.f ? -pow(-x, 1.f / 3.f) : pow(x, 1.f / 3.f); }

template <typename S>
inline void f7pt_one(const device float4* p, thread const int* idx, device float* o, device int* ok) {
  for (int s = 0; s < 3; ++s) {
    ok[s] = 0;
    for (int k = 0; k < 9; ++k) o[s * 9 + k] = 0.f;
  }
  for (int r = 0; r < 7; ++r)
    for (int s = r + 1; s < 7; ++s)
      if (idx[r] == idx[s]) return;
  float nul[18];
  null_space<7>(p, idx, nul);
  S f1[9], f2[9], g[9];
  for (int k = 0; k < 9; ++k) { f1[k] = from_f<S>(nul[k]); f2[k] = from_f<S>(nul[9 + k]); }
  const S two = from_f<S>(2.f), hlf = from_f<S>(0.5f);
  const S d0 = det3s9<S>(f2), d1 = det3s9<S>(f1);
  for (int k = 0; k < 9; ++k) g[k] = two * f2[k] - f1[k];
  const S dm = det3s9<S>(g);
  for (int k = 0; k < 9; ++k) g[k] = two * f1[k] - f2[k];
  const S d2 = det3s9<S>(g);
  const S a0 = d0, a2 = hlf * (d1 + dm) - a0;
  const S a3 = (d2 - a0 - from_f<S>(4.f) * a2 - (d1 - dm)) / from_f<S>(6.f);
  const S a1 = hlf * (d1 - dm) - a3;
  float A3 = tof(a3), A2 = tof(a2), A1 = tof(a1), A0 = tof(a0);
  float roots[3];
  int nr = 0;
  const float scale = max(max(fabs(A3), fabs(A2)), max(fabs(A1), fabs(A0)));
  if (!(scale > 0.f)) return;
  if (fabs(A3) > 1e-10f * scale) {
    const float b = A2 / A3, c = A1 / A3, d = A0 / A3;
    const float q = (3.f * c - b * b) / 9.f, r = (9.f * b * c - 27.f * d - 2.f * b * b * b) / 54.f;
    const float disc = q * q * q + r * r;
    if (disc < 0.f) {
      const float th = acos(clamp(r / sqrt(-q * q * q), -1.f, 1.f)), m2 = 2.f * sqrt(-q);
      roots[0] = m2 * cos(th / 3.f) - b / 3.f;
      roots[1] = m2 * cos((th + 2.f * M_PI_F) / 3.f) - b / 3.f;
      roots[2] = m2 * cos((th + 4.f * M_PI_F) / 3.f) - b / 3.f;
      nr = 3;
    } else {
      const float sd = sqrt(disc);
      roots[0] = cbrt_(r + sd) + cbrt_(r - sd) - b / 3.f;
      nr = 1;
    }
  } else if (fabs(A2) > 1e-10f * scale) {
    const float disc = A1 * A1 - 4.f * A2 * A0;
    if (disc >= 0.f) {
      const float sd = sqrt(disc);
      roots[0] = (-A1 + sd) / (2.f * A2);
      roots[1] = (-A1 - sd) / (2.f * A2);
      nr = 2;
    }
  } else if (fabs(A1) > 1e-10f * scale) {
    roots[0] = -A0 / A1;
    nr = 1;
  }
  for (int s = 0; s < nr; ++s) {
    if (!isfinite(roots[s])) continue;
    S l = from_f<S>(roots[s]);
    for (int it = 0; it < 2; ++it) {
      const S v = ((a3 * l + a2) * l + a1) * l + a0, dv = (from_f<S>(3.f) * a3 * l + two * a2) * l + a1;
      if (!is_zero(dv)) l = l - v / dv;
    }
    if (!isfinite(tof(l))) continue;
    float f[9], nrm = 0.f;
    const S one_l = from_f<S>(1.f) - l;
    for (int k = 0; k < 9; ++k) {
      f[k] = tof(l * f1[k] + one_l * f2[k]);
      nrm += f[k] * f[k];
    }
    if (!(nrm > 0.f) || !isfinite(nrm)) continue;
    nrm = rsqrt(nrm);
    for (int k = 0; k < 9; ++k) o[s * 9 + k] = f[k] * nrm;
    ok[s] = 1;
  }
}

// ---- P3P (Lambda Twist), float32 ---------------------------------------------------------------------------------
inline float det_sym(thread const float* s) {
  return s[0] * (s[3] * s[5] - s[4] * s[4]) - s[1] * (s[1] * s[5] - s[4] * s[2]) + s[2] * (s[1] * s[4] - s[3] * s[2]);
}
inline float tr_adj(thread const float* s, thread const float* t) {
  float a00 = s[3] * s[5] - s[4] * s[4], a11 = s[0] * s[5] - s[2] * s[2], a22 = s[0] * s[3] - s[1] * s[1];
  float a01 = s[2] * s[4] - s[1] * s[5], a02 = s[1] * s[4] - s[3] * s[2], a12 = s[1] * s[2] - s[0] * s[4];
  return a00 * t[0] + a11 * t[3] + a22 * t[5] + 2.f * (a01 * t[1] + a02 * t[2] + a12 * t[4]);
}
inline float quad_form(thread const float* s, thread const float* u, thread const float* v) {
  return u[0] * (s[0] * v[0] + s[1] * v[1] + s[2] * v[2]) + u[1] * (s[1] * v[0] + s[3] * v[1] + s[4] * v[2]) +
         u[2] * (s[2] * v[0] + s[4] * v[1] + s[5] * v[2]);
}
inline void cross3(thread const float* a, thread const float* b, thread float* c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}
inline int cubic_roots(float c3, float c2, float c1, float c0, thread float* r) {
  const float scale = max(max(fabs(c3), fabs(c2)), max(fabs(c1), fabs(c0)));
  if (!(scale > 0.f)) return 0;
  int n = 0;
  if (fabs(c3) < 1e-7f * scale) {
    if (fabs(c2) < 1e-7f * scale) {
      if (fabs(c1) < 1e-7f * scale) return 0;
      r[0] = -c0 / c1;
      return 1;
    }
    float d = c1 * c1 - 4.f * c2 * c0;
    if (d < 0.f) return 0;
    d = sqrt(d);
    r[0] = (-c1 + d) / (2.f * c2);
    r[1] = (-c1 - d) / (2.f * c2);
    return 2;
  }
  const float a = c2 / c3, b = c1 / c3, c = c0 / c3;
  const float p = b - a * a / 3.f, q = 2.f * a * a * a / 27.f - a * b / 3.f + c, sh = -a / 3.f;
  const float disc = q * q / 4.f + p * p * p / 27.f;
  if (disc >= 0.f) {
    const float sq = sqrt(disc);
    r[0] = cbrt_(-q / 2.f + sq) + cbrt_(-q / 2.f - sq) + sh;
    n = 1;
  } else {
    const float m = 2.f * sqrt(-p / 3.f), th = acos(clamp(3.f * q / (p * m), -1.f, 1.f)) / 3.f;
    for (int k = 0; k < 3; ++k) r[k] = m * cos(th - 2.0943951f * float(k)) + sh;
    n = 3;
  }
  for (int k = 0; k < n; ++k)
    for (int it = 0; it < 3; ++it) {
      float x = r[k], f = ((c3 * x + c2) * x + c1) * x + c0, d = (3.f * c3 * x + 2.f * c2) * x + c1;
      if (fabs(d) > 1e-30f) r[k] = x - f / d;
    }
  return n;
}
// The up to four poses [R | t] of the unit bearings y [3 * 3] and world points X [3 * 3] -> sol [4 * 12].
inline int p3p_solve(thread const float* y, thread const float* X, thread float* sol) {
  thread const float* X0 = X; thread const float* X1 = X + 3; thread const float* X2 = X + 6;
  float d01[3], d02[3], d12[3];
  for (int k = 0; k < 3; ++k) { d01[k] = X0[k] - X1[k]; d02[k] = X0[k] - X2[k]; d12[k] = X1[k] - X2[k]; }
  const float a12 = d01[0] * d01[0] + d01[1] * d01[1] + d01[2] * d01[2];
  const float a13 = d02[0] * d02[0] + d02[1] * d02[1] + d02[2] * d02[2];
  const float a23 = d12[0] * d12[0] + d12[1] * d12[1] + d12[2] * d12[2];
  if (!(a12 > 0.f) || !(a13 > 0.f) || !(a23 > 0.f)) return 0;
  const float b12 = y[0] * y[3] + y[1] * y[4] + y[2] * y[5];
  const float b13 = y[0] * y[6] + y[1] * y[7] + y[2] * y[8];
  const float b23 = y[3] * y[6] + y[4] * y[7] + y[5] * y[8];
  const float M12[6] = {1.f, -b12, 0.f, 1.f, 0.f, 0.f}, M13[6] = {1.f, 0.f, -b13, 0.f, 0.f, 1.f};
  const float M23[6] = {0.f, 0.f, 0.f, 1.f, -b23, 1.f};
  float D1[6], D2[6];
  for (int k = 0; k < 6; ++k) { D1[k] = a23 * M12[k] - a12 * M23[k]; D2[k] = a23 * M13[k] - a13 * M23[k]; }
  float g[3];
  const int ng = cubic_roots(det_sym(D2), tr_adj(D2, D1), tr_adj(D1, D2), det_sym(D1), g);
  float n01[3], Fw[9], Fwi[9];
  cross3(d01, d02, n01);
  for (int k = 0; k < 3; ++k) { Fw[k * 3] = d01[k]; Fw[k * 3 + 1] = d02[k]; Fw[k * 3 + 2] = n01[k]; }
  {
    float det = Fw[0] * (Fw[4] * Fw[8] - Fw[5] * Fw[7]) - Fw[1] * (Fw[3] * Fw[8] - Fw[5] * Fw[6]) +
                Fw[2] * (Fw[3] * Fw[7] - Fw[4] * Fw[6]);
    if (!(fabs(det) > 1e-30f)) return 0;
    float id = 1.f / det;
    Fwi[0] = (Fw[4] * Fw[8] - Fw[5] * Fw[7]) * id; Fwi[1] = (Fw[2] * Fw[7] - Fw[1] * Fw[8]) * id;
    Fwi[2] = (Fw[1] * Fw[5] - Fw[2] * Fw[4]) * id; Fwi[3] = (Fw[5] * Fw[6] - Fw[3] * Fw[8]) * id;
    Fwi[4] = (Fw[0] * Fw[8] - Fw[2] * Fw[6]) * id; Fwi[5] = (Fw[2] * Fw[3] - Fw[0] * Fw[5]) * id;
    Fwi[6] = (Fw[3] * Fw[7] - Fw[4] * Fw[6]) * id; Fwi[7] = (Fw[1] * Fw[6] - Fw[0] * Fw[7]) * id;
    Fwi[8] = (Fw[0] * Fw[4] - Fw[1] * Fw[3]) * id;
  }
  int count = 0;
  for (int ig = 0; ig < ng && count == 0; ++ig) {
    float D0[6];
    for (int k = 0; k < 6; ++k) D0[k] = D1[k] + g[ig] * D2[k];
    const float tr = D0[0] + D0[3] + D0[5];
    const float mn = D0[0] * D0[3] - D0[1] * D0[1] + D0[0] * D0[5] - D0[2] * D0[2] + D0[3] * D0[5] - D0[4] * D0[4];
    float disc = tr * tr - 4.f * mn;
    if (!(disc > 0.f)) continue;
    disc = sqrt(disc);
    const float e1 = (tr + disc) / 2.f, e2 = (tr - disc) / 2.f;
    if (!(e1 > 0.f) || !(e2 < 0.f)) continue;
    float v1[3], v2[3];
    bool bad = false;
    for (int w = 0; w < 2; ++w) {
      const float ev = w == 0 ? e1 : e2;
      float A[9] = {D0[0] - ev, D0[1], D0[2], D0[1], D0[3] - ev, D0[4], D0[2], D0[4], D0[5] - ev};
      float c[9], best = -1.f;
      thread float* v = w == 0 ? v1 : v2;
      cross3(A, A + 3, c);
      cross3(A, A + 6, c + 3);
      cross3(A + 3, A + 6, c + 6);
      for (int k = 0; k < 3; ++k) {
        float nn = c[k * 3] * c[k * 3] + c[k * 3 + 1] * c[k * 3 + 1] + c[k * 3 + 2] * c[k * 3 + 2];
        if (nn > best) { best = nn; for (int j = 0; j < 3; ++j) v[j] = c[k * 3 + j]; }
      }
      if (!(best > 0.f)) { bad = true; break; }
      const float inv = rsqrt(best);
      for (int j = 0; j < 3; ++j) v[j] *= inv;
    }
    if (bad) continue;
    const float s1 = sqrt(e1), s2 = sqrt(-e2);
    thread const float* Q = fabs(g[ig]) >= 1.f ? D1 : D2;
    for (int pl = 0; pl < 2 && count < 4; ++pl) {
      float w[3];
      for (int j = 0; j < 3; ++j) w[j] = s1 * v1[j] + (pl == 0 ? s2 : -s2) * v2[j];
      int kp = 0;
      for (int j = 1; j < 3; ++j) kp = fabs(w[j]) > fabs(w[kp]) ? j : kp;
      const int i1 = (kp + 1) % 3, i2 = (kp + 2) % 3;
      float r1[3] = {0.f, 0.f, 0.f}, r2[3] = {0.f, 0.f, 0.f};
      r1[i1] = 1.f; r1[kp] = -w[i1] / w[kp];
      r2[i2] = 1.f; r2[kp] = -w[i2] / w[kp];
      const float al = quad_form(Q, r1, r1), be = quad_form(Q, r1, r2), de = quad_form(Q, r2, r2);
      float dq = be * be - al * de;
      if (dq < 0.f) {
        if (dq < -1e-5f * (be * be + fabs(al * de))) continue;
        dq = 0.f;
      }
      dq = sqrt(dq);
      for (int sg = 0; sg < 2 && count < 4; ++sg) {
        float l1, l2;
        const float rt = sg == 0 ? dq : -dq;
        if (fabs(al) >= fabs(de)) { l1 = (-be + rt) / al; l2 = 1.f; }
        else { l1 = 1.f; l2 = (-be + rt) / de; }
        float l[3];
        for (int j = 0; j < 3; ++j) l[j] = l1 * r1[j] + l2 * r2[j];
        const float q12 = quad_form(M12, l, l);
        if (!(q12 > 0.f)) continue;
        float sc = sqrt(a12 / q12);
        if (l[0] + l[1] + l[2] < 0.f) sc = -sc;
        for (int j = 0; j < 3; ++j) l[j] *= sc;
        if (!(l[0] > 0.f) || !(l[1] > 0.f) || !(l[2] > 0.f)) continue;
        for (int it = 0; it < 2; ++it) {
          const float f0 = quad_form(M12, l, l) - a12, f1 = quad_form(M13, l, l) - a13;
          const float f2 = quad_form(M23, l, l) - a23;
          const float J00 = 2.f * (l[0] - b12 * l[1]), J01 = 2.f * (l[1] - b12 * l[0]), J02 = 0.f;
          const float J10 = 2.f * (l[0] - b13 * l[2]), J11 = 0.f, J12 = 2.f * (l[2] - b13 * l[0]);
          const float J20 = 0.f, J21 = 2.f * (l[1] - b23 * l[2]), J22 = 2.f * (l[2] - b23 * l[1]);
          const float det = J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) + J02 * (J10 * J21 - J11 * J20);
          if (!(fabs(det) > 1e-30f)) break;
          const float dx = (f0 * (J11 * J22 - J12 * J21) - J01 * (f1 * J22 - J12 * f2) + J02 * (f1 * J21 - J11 * f2)) / det;
          const float dy = (J00 * (f1 * J22 - J12 * f2) - f0 * (J10 * J22 - J12 * J20) + J02 * (J10 * f2 - f1 * J20)) / det;
          const float dz = (J00 * (J11 * f2 - f1 * J21) - J01 * (J10 * f2 - f1 * J20) + f0 * (J10 * J21 - J11 * J20)) / det;
          l[0] -= dx; l[1] -= dy; l[2] -= dz;
        }
        if (!(l[0] > 0.f) || !(l[1] > 0.f) || !(l[2] > 0.f)) continue;
        float c0[3], c1[3], c2[3], e01[3], e02[3], nc[3], Fc[9], R[9];
        for (int j = 0; j < 3; ++j) {
          c0[j] = l[0] * y[j]; c1[j] = l[1] * y[3 + j]; c2[j] = l[2] * y[6 + j];
          e01[j] = c0[j] - c1[j]; e02[j] = c0[j] - c2[j];
        }
        cross3(e01, e02, nc);
        for (int j = 0; j < 3; ++j) { Fc[j * 3] = e01[j]; Fc[j * 3 + 1] = e02[j]; Fc[j * 3 + 2] = nc[j]; }
        bool fin = true;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) {
            R[i * 3 + j] = Fc[i * 3] * Fwi[j] + Fc[i * 3 + 1] * Fwi[3 + j] + Fc[i * 3 + 2] * Fwi[6 + j];
            fin = fin && isfinite(R[i * 3 + j]);
          }
        if (!fin) continue;
        for (int k = 0; k < 9; ++k) sol[count * 12 + k] = R[k];
        for (int i = 0; i < 3; ++i)
          sol[count * 12 + 9 + i] = c0[i] - (R[i * 3] * X0[0] + R[i * 3 + 1] * X0[1] + R[i * 3 + 2] * X0[2]);
        ++count;
      }
    }
  }
  return count;
}

// ---- scoring ------------------------------------------------------------------------------------------------------
// MSAC gain of a match: 1 - sampson / thr2 for inliers, 0 for outliers.
inline float gain(float4 m, thread const float* E, float t2) {
  float l0 = E[0] * m.x + E[1] * m.y + E[2], l1 = E[3] * m.x + E[4] * m.y + E[5], l2 = E[6] * m.x + E[7] * m.y + E[8];
  float r0 = E[0] * m.z + E[3] * m.w + E[6], r1 = E[1] * m.z + E[4] * m.w + E[7];
  float num = m.z * l0 + m.w * l1 + l2;
  float den = (l0 * l0 + l1 * l1 + r0 * r0 + r1 * r1) * t2;
  return (den > 0.f && num * num < den) ? 1.f - num * num / den : 0.f;
}
// Reprojection gain of a 2D-3D match (m0: u, v, X, Y; m1: Z, ...) under [R | t]; behind the camera: outlier.
inline float gain_abs(float4 m0, float4 m1, thread const float* M, float t2) {
  const float X = m0.z, Y = m0.w, Z = m1.x;
  float z = M[6] * X + M[7] * Y + M[8] * Z + M[11];
  if (!(z > 0.f)) return 0.f;
  float dx = (M[0] * X + M[1] * Y + M[2] * Z + M[9]) / z - m0.x;
  float dy = (M[3] * X + M[4] * Y + M[5] * Z + M[10]) / z - m0.y;
  float e = (dx * dx + dy * dy) / t2;
  return e < 1.f ? 1.f - e : 0.f;
}
template <int ABS>
inline float gain_at(const device float4* p, uint i, thread const float* E, float t2) {
  return ABS ? gain_abs(p[2 * i], p[2 * i + 1], E, t2) : gain(p[i], E, t2);
}
template <int ABS>
inline float gain_tg(const threadgroup float4* p, uint i, thread const float* E, float t2) {
  return ABS ? gain_abs(p[2 * i], p[2 * i + 1], E, t2) : gain(p[i], E, t2);
}

// Sum of acc[0..k) over the threadgroup (TPB = 8 SIMD groups) into red[0..k); every thread calls it.
inline void block_sum(thread float* acc, int k, threadgroup float* red, uint lid) {
  for (int i = 0; i < k; ++i) acc[i] = simd_sum(acc[i]);
  const uint warp = lid / 32, lane = lid % 32;
  bar();
  if (lane == 0)
    for (int i = 0; i < k; ++i) red[warp * NT + i] = acc[i];
  bar();
  if (lid < uint(k)) {
    float v = 0.f;
    for (int w = 0; w < TPB / 32; ++w) v += red[w * NT + lid];
    red[lid] = v;
  }
  bar();
}

// ---- refinement (thread-local serial helpers) -------------------------------------------------------------------
inline void cross_mat(thread const float* v, thread float* m) {
  m[0] = 0.f; m[1] = -v[2]; m[2] = v[1]; m[3] = v[2]; m[4] = 0.f; m[5] = -v[0]; m[6] = -v[1]; m[7] = v[0]; m[8] = 0.f;
}
inline void mat_mul(thread const float* a, thread const float* b, thread float* c) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}
inline float det3(thread const float* m) {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}
inline void jacobi_eig(thread float* a, thread float* v, int n) {
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
        if (fabs(apq) < 1e-30f) continue;
        float th = (a[q * n + q] - a[p * n + p]) / (2.f * apq);
        float t = (th >= 0.f ? 1.f : -1.f) / (fabs(th) + sqrt(th * th + 1.f));
        float c = rsqrt(t * t + 1.f), s = t * c;
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
// SVD of the 3 x 3 matrix e: V (columns, det +1), singular values descending, U = (E v0 / s0, E v1 / s1, u0 x u1).
inline bool svd3(thread const float* e, thread float* U, thread float* V, thread float* sig) {
  float a[9], v[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) a[i * 3 + j] = e[i] * e[j] + e[3 + i] * e[3 + j] + e[6 + i] * e[6 + j];
  jacobi_eig(a, v, 3);
  int ord[3] = {0, 1, 2};
  for (int i = 0; i < 2; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (a[ord[j] * 3 + ord[j]] > a[ord[i] * 3 + ord[i]]) { int x = ord[i]; ord[i] = ord[j]; ord[j] = x; }
  for (int c = 0; c < 3; ++c) {
    sig[c] = sqrt(max(a[ord[c] * 3 + ord[c]], 0.f));
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
inline bool decompose(thread const float* e, thread float* R1, thread float* R2, thread float* t) {
  float U[9], V[9], sig[3], tmp[9], VT[9];
  if (!svd3(e, U, V, sig)) return false;
  const float W[9] = {0.f, -1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
  const float WT[9] = {0.f, 1.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) VT[i * 3 + j] = V[j * 3 + i];
  mat_mul(U, W, tmp);
  mat_mul(tmp, VT, R1);
  mat_mul(U, WT, tmp);
  mat_mul(tmp, VT, R2);
  for (int k = 0; k < 3; ++k) t[k] = U[k * 3 + 2];
  return true;
}
inline bool project_rank2(thread const float* v, thread float* e, bool fundamental) {
  float U[9], V[9], sig[3];
  if (!svd3(v, U, V, sig)) return false;
  float s0 = rsqrt(2.f), s1 = s0;
  if (fundamental) {
    const float inv = rsqrt(sig[0] * sig[0] + sig[1] * sig[1]);
    s0 = sig[0] * inv;
    s1 = sig[1] * inv;
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) e[i * 3 + j] = s0 * U[i * 3] * V[j * 3] + s1 * U[i * 3 + 1] * V[j * 3 + 1];
  return true;
}
inline void lm_linearise(thread const float* R, thread const float* t, thread float* E, thread float* D,
                         thread float* b1, thread float* b2) {
  int k = fabs(t[0]) <= fabs(t[1]) ? (fabs(t[0]) <= fabs(t[2]) ? 0 : 2) : (fabs(t[1]) <= fabs(t[2]) ? 1 : 2);
  float ek[3] = {0.f, 0.f, 0.f};
  ek[k] = 1.f;
  float c[3] = {t[1] * ek[2] - t[2] * ek[1], t[2] * ek[0] - t[0] * ek[2], t[0] * ek[1] - t[1] * ek[0]};
  float inv = rsqrt(max(c[0] * c[0] + c[1] * c[1] + c[2] * c[2], 1e-24f));
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
template <int NP>
inline void lm_step(threadgroup const float* terms, float lam, thread float* step) {
  constexpr int NU = NP * (NP + 1) / 2;
  float M[NP * (NP + 1)];
  int q = 0;
  for (int u = 0; u < NP; ++u)
    for (int v = u; v < NP; ++v) {
      M[u * (NP + 1) + v] = M[v * (NP + 1) + u] = terms[q++];
    }
  for (int u = 0; u < NP; ++u) {
    M[u * (NP + 1) + u] += lam * max(M[u * (NP + 1) + u], 1e-12f);
    M[u * (NP + 1) + NP] = -terms[NU + u];
  }
  for (int c = 0; c < NP; ++c) {
    int piv = c;
    for (int r = c + 1; r < NP; ++r)
      if (fabs(M[r * (NP + 1) + c]) > fabs(M[piv * (NP + 1) + c])) piv = r;
    if (piv != c)
      for (int j = 0; j < NP + 1; ++j) {
        float x = M[c * (NP + 1) + j]; M[c * (NP + 1) + j] = M[piv * (NP + 1) + j]; M[piv * (NP + 1) + j] = x;
      }
    if (!(fabs(M[c * (NP + 1) + c]) > 1e-30f)) {
      for (int k = 0; k < NP; ++k) step[k] = 0.f;
      return;
    }
    for (int r = c + 1; r < NP; ++r) {
      float f = M[r * (NP + 1) + c] / M[c * (NP + 1) + c];
      for (int j = c; j < NP + 1; ++j) M[r * (NP + 1) + j] -= f * M[c * (NP + 1) + j];
    }
  }
  for (int r = NP - 1; r >= 0; --r) {
    float x = M[r * (NP + 1) + NP];
    for (int j = r + 1; j < NP; ++j) x -= M[r * (NP + 1) + j] * step[j];
    step[r] = x / M[r * (NP + 1) + r];
  }
  for (int k = 0; k < NP; ++k)
    if (!isfinite(step[k])) {
      for (int j = 0; j < NP; ++j) step[j] = 0.f;
      return;
    }
}
inline void exp_so3(thread const float* w, thread float* Rw) {
  float th = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  for (int i = 0; i < 9; ++i) Rw[i] = i % 4 == 0 ? 1.f : 0.f;
  if (th > 1e-12f) {
    float k[3] = {w[0] / th, w[1] / th, w[2] / th}, K[9], K2[9];
    cross_mat(k, K);
    mat_mul(K, K, K2);
    float s = sin(th), c = 1.f - cos(th);
    for (int i = 0; i < 9; ++i) Rw[i] += s * K[i] + c * K2[i];
  }
}
inline void lm_linearise_f(thread const float* U, thread const float* V, float phi, thread float* F, thread float* D) {
  const float s[3] = {cos(phi), sin(phi), 0.f}, ds[3] = {-sin(phi), cos(phi), 0.f};
  float US[9], gen[9], tmp[9], tmp2[9], Vt[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) { US[i * 3 + j] = U[i * 3 + j] * s[j]; Vt[i * 3 + j] = V[j * 3 + i]; }
  mat_mul(US, Vt, F);
  for (int k = 0; k < 3; ++k) {
    float e3[3] = {0.f, 0.f, 0.f};
    e3[k] = 1.f;
    cross_mat(e3, gen);
    mat_mul(U, gen, tmp);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) tmp2[i * 3 + j] = tmp[i * 3 + j] * s[j];
    mat_mul(tmp2, Vt, tmp);
    for (int i = 0; i < 9; ++i) D[i * 7 + k] = tmp[i];
    mat_mul(US, gen, tmp);
    mat_mul(tmp, Vt, tmp2);
    for (int i = 0; i < 9; ++i) D[i * 7 + 3 + k] = -tmp2[i];
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) tmp[i * 3 + j] = U[i * 3 + j] * ds[j];
  mat_mul(tmp, Vt, tmp2);
  for (int i = 0; i < 9; ++i) D[i * 7 + 6] = tmp2[i];
}

// ---- refinement (threadgroup passes over the matches of a pair) -------------------------------------------------
struct Shared {
  float E[9], En[9], R[9], t[3], R1[9], R2[9], T[3], best;
  int front, tryv, okv;
  float Rn[9], tn[3], LE[9], LEn[9], D[63], b1[3], b2[3], step[7], lam, lbest;
  float Un[9], Vn[9], phin, Fn[9];
};
template <typename A, typename B> inline void cp(A dst, B src, int n) { for (int i = 0; i < n; ++i) dst[i] = src[i]; }

template <int NP>
inline void lm_accumulate(const device float4* p, int tot, thread const float* E, thread const float* D, float t2,
                          thread float* acc, uint lid) {
  constexpr int NU = NP * (NP + 1) / 2;
  for (int k = 0; k < NU + NP + 1; ++k) acc[k] = 0.f;
  for (int i = int(lid); i < tot; i += TPB) {
    const float4 m = p[i];
    const float x1[3] = {m.x, m.y, 1.f}, x2[3] = {m.z, m.w, 1.f};
    float l0 = E[0] * m.x + E[1] * m.y + E[2], l1 = E[3] * m.x + E[4] * m.y + E[5], l2 = E[6] * m.x + E[7] * m.y + E[8];
    float r0 = E[0] * m.z + E[3] * m.w + E[6], r1 = E[1] * m.z + E[4] * m.w + E[7];
    float num = m.z * l0 + m.w * l1 + l2;
    float den = l0 * l0 + l1 * l1 + r0 * r0 + r1 * r1;
    if (!(den > 0.f)) { acc[NU + NP] += t2; continue; }
    float inv = rsqrt(den), res = num * inv;
    if (res * res >= t2) { acc[NU + NP] += t2; continue; }
    acc[NU + NP] += res * res;
    float s3 = num * inv * inv * inv, J[NP];
    for (int k = 0; k < NP; ++k) J[k] = 0.f;
    for (int i3 = 0; i3 < 3; ++i3)
      for (int j3 = 0; j3 < 3; ++j3) {
        float dd = (i3 == 0 ? l0 * x1[j3] : i3 == 1 ? l1 * x1[j3] : 0.f) +
                   (j3 == 0 ? r0 * x2[i3] : j3 == 1 ? r1 * x2[i3] : 0.f);
        float g = x2[i3] * x1[j3] * inv - s3 * dd;
        thread const float* d = D + (i3 * 3 + j3) * NP;
        for (int k = 0; k < NP; ++k) J[k] += g * d[k];
      }
    int q = 0;
    for (int u = 0; u < NP; ++u)
      for (int v = u; v < NP; ++v) acc[q++] += J[u] * J[v];
    for (int k = 0; k < NP; ++k) acc[NU + k] += J[k] * res;
  }
}
inline float thread_cost(const device float4* p, int tot, thread const float* E, float t2, uint lid) {
  float c = 0.f;
  for (int i = int(lid); i < tot; i += TPB) {
    const float4 m = p[i];
    float l0 = E[0] * m.x + E[1] * m.y + E[2], l1 = E[3] * m.x + E[4] * m.y + E[5], l2 = E[6] * m.x + E[7] * m.y + E[8];
    float r0 = E[0] * m.z + E[3] * m.w + E[6], r1 = E[1] * m.z + E[4] * m.w + E[7];
    float num = m.z * l0 + m.w * l1 + l2;
    float den = l0 * l0 + l1 * l1 + r0 * r0 + r1 * r1;
    float e2 = den > 0.f ? num * num / den : t2;
    c += min(e2, t2);
  }
  return c;
}
// The cost of the model in sh-array `Em` summed into red[0].
inline void cost_sum(const device float4* p, int tot, threadgroup const float* Em, float t2, thread float* acc,
                     threadgroup float* red, uint lid) {
  float E[9];
  cp(E, Em, 9);
  acc[0] = thread_cost(p, tot, E, t2, lid);
  block_sum(acc, 1, red, lid);
}
// LM of (R, t) in sh.R, sh.t (E = [t]_x R).
inline void lm_run(const device float4* p, int tot, float t2, int iterations, threadgroup Shared& sh, thread float* acc,
                   threadgroup float* red, uint lid) {
  if (lid == 0) {
    float R[9], t[3], E[9], D[45], b1[3], b2[3];
    cp(R, sh.R, 9); cp(t, sh.t, 3);
    lm_linearise(R, t, E, D, b1, b2);
    cp(sh.LE, E, 9); cp(sh.D, D, 45); cp(sh.b1, b1, 3); cp(sh.b2, b2, 3);
    sh.lam = 1e-3f;
  }
  bar();
  cost_sum(p, tot, sh.LE, t2, acc, red, lid);
  if (lid == 0) sh.lbest = red[0];
  for (int it = 0; it < iterations; ++it) {
    {
      float E[9], D[45];
      cp(E, sh.LE, 9); cp(D, sh.D, 45);
      lm_accumulate<5>(p, tot, E, D, t2, acc, lid);
    }
    block_sum(acc, NLM, red, lid);
    if (lid == 0) {
      float step[5], R[9], t[3], Rw[9], Rn[9], tn[3], tx[9], En[9];
      lm_step<5>(red, sh.lam, step);
      cp(R, sh.R, 9); cp(t, sh.t, 3);
      exp_so3(step, Rw);
      mat_mul(R, Rw, Rn);
      for (int i = 0; i < 3; ++i) tn[i] = t[i] + step[3] * sh.b1[i] + step[4] * sh.b2[i];
      float inv = rsqrt(max(tn[0] * tn[0] + tn[1] * tn[1] + tn[2] * tn[2], 1e-24f));
      for (int i = 0; i < 3; ++i) tn[i] *= inv;
      cross_mat(tn, tx);
      mat_mul(tx, Rn, En);
      cp(sh.Rn, Rn, 9); cp(sh.tn, tn, 3); cp(sh.LEn, En, 9);
    }
    bar();
    cost_sum(p, tot, sh.LEn, t2, acc, red, lid);
    if (lid == 0) {
      if (red[0] < sh.lbest) {
        sh.lbest = red[0];
        cp(sh.R, sh.Rn, 9); cp(sh.t, sh.tn, 3);
        sh.lam = max(sh.lam * 0.1f, 1e-6f);
        float R[9], t[3], E[9], D[45], b1[3], b2[3];
        cp(R, sh.R, 9); cp(t, sh.t, 3);
        lm_linearise(R, t, E, D, b1, b2);
        cp(sh.LE, E, 9); cp(sh.D, D, 45); cp(sh.b1, b1, 3); cp(sh.b2, b2, 3);
      } else {
        sh.lam = min(sh.lam * 10.f, 1e6f);
      }
    }
    bar();
  }
}
// LM of a fundamental matrix U diag(cos phi, sin phi, 0) V^T: U = sh.R1, V = sh.R2, phi = sh.T[0]; result in sh.En.
inline void lm_run_f(const device float4* p, int tot, float t2, int iterations, threadgroup Shared& sh,
                     thread float* acc, threadgroup float* red, uint lid) {
  if (lid == 0) {
    float U[9], V[9], F[9], D[63];
    cp(U, sh.R1, 9); cp(V, sh.R2, 9);
    lm_linearise_f(U, V, sh.T[0], F, D);
    cp(sh.En, F, 9); cp(sh.D, D, 63);
    sh.lam = 1e-3f;
  }
  bar();
  cost_sum(p, tot, sh.En, t2, acc, red, lid);
  if (lid == 0) sh.lbest = red[0];
  for (int it = 0; it < iterations; ++it) {
    {
      float F[9], D[63];
      cp(F, sh.En, 9); cp(D, sh.D, 63);
      lm_accumulate<7>(p, tot, F, D, t2, acc, lid);
    }
    block_sum(acc, NLMF, red, lid);
    if (lid == 0) {
      float step[7], U[9], V[9], Rw[9], Un[9], Vn[9], Fn[9], D[63];
      lm_step<7>(red, sh.lam, step);
      cp(U, sh.R1, 9); cp(V, sh.R2, 9);
      exp_so3(step, Rw);
      mat_mul(U, Rw, Un);
      exp_so3(step + 3, Rw);
      mat_mul(V, Rw, Vn);
      sh.phin = sh.T[0] + step[6];
      lm_linearise_f(Un, Vn, sh.phin, Fn, D);
      cp(sh.Un, Un, 9); cp(sh.Vn, Vn, 9); cp(sh.Fn, Fn, 9);
    }
    bar();
    cost_sum(p, tot, sh.Fn, t2, acc, red, lid);
    if (lid == 0) {
      if (red[0] < sh.lbest) {
        sh.lbest = red[0];
        cp(sh.R1, sh.Un, 9); cp(sh.R2, sh.Vn, 9);
        sh.T[0] = sh.phin;
        sh.lam = max(sh.lam * 0.1f, 1e-6f);
        float U[9], V[9], F[9], D[63];
        cp(U, sh.R1, 9); cp(V, sh.R2, 9);
        lm_linearise_f(U, V, sh.T[0], F, D);
        cp(sh.En, F, 9); cp(sh.D, D, 63);
      } else {
        sh.lam = min(sh.lam * 10.f, 1e6f);
      }
    }
    bar();
  }
}
// MSAC score and inlier count of sh-array `Em` into red[0], red[1]; writes the mask (N entries) when mk is not null.
inline void score_sum(const device float4* p, int tot, int N, threadgroup const float* Em, float t2, thread float* acc,
                      device uint8_t* mk, threadgroup float* red, uint lid) {
  float E[9];
  cp(E, Em, 9);
  acc[0] = acc[1] = 0.f;
  const int end = mk ? N : tot;
  for (int i = int(lid); i < end; i += TPB) {
    float g = i < tot ? gain(p[i], E, t2) : 0.f;
    acc[0] += g;
    acc[1] += g > 0.f ? 1.f : 0.f;
    if (mk) mk[i] = g > 0.f ? 1 : 0;
  }
  block_sum(acc, 2, red, lid);
}
// recoverPose of sh.E into sh.R, sh.t; sh.front the inliers in front (-1 when sh.E is degenerate).
inline void recover(const device float4* p, int tot, float t2, threadgroup Shared& sh, thread float* acc,
                    threadgroup float* red, uint lid) {
  if (lid == 0) {
    float E[9], R1[9], R2[9], T[3];
    cp(E, sh.E, 9);
    sh.okv = decompose(E, R1, R2, T) ? 1 : 0;
    cp(sh.R1, R1, 9); cp(sh.R2, R2, 9); cp(sh.T, T, 3);
  }
  bar();
  if (!sh.okv) {
    if (lid == 0) sh.front = -1;
    bar();
    return;
  }
  float E[9], R1[9], R2[9], t[3];
  cp(E, sh.E, 9); cp(R1, sh.R1, 9); cp(R2, sh.R2, 9); cp(t, sh.T, 3);
  for (int k = 0; k < 4; ++k) acc[k] = 0.f;
  for (int i = int(lid); i < tot; i += TPB) {
    const float4 m = p[i];
    if (!(gain(m, E, t2) > 0.f)) continue;
    const float xx = m.z * m.z + m.w * m.w + 1.f, xt = m.z * t[0] + m.w * t[1] + t[2];
    for (int r = 0; r < 2; ++r) {
      thread const float* R = r ? R2 : R1;
      float y[3];
      for (int k = 0; k < 3; ++k) y[k] = R[k * 3] * m.x + R[k * 3 + 1] * m.y + R[k * 3 + 2];
      const float yy = y[0] * y[0] + y[1] * y[1] + y[2] * y[2], yx = y[0] * m.z + y[1] * m.w + y[2];
      const float yt = y[0] * t[0] + y[1] * t[1] + y[2] * t[2];
      const float det = yy * xx - yx * yx;
      const float l1 = (-yt * xx + yx * xt) / det, l2 = (yy * xt - yx * yt) / det;
      acc[2 * r] += (l1 > 0.f && l2 > 0.f) ? 1.f : 0.f;
      acc[2 * r + 1] += (l1 < 0.f && l2 < 0.f) ? 1.f : 0.f;
    }
  }
  block_sum(acc, 4, red, lid);
  if (lid == 0) {
    int best = 0;
    for (int k = 1; k < 4; ++k) best = red[k] > red[best] ? k : best;
    for (int k = 0; k < 9; ++k) sh.R[k] = best < 2 ? sh.R1[k] : sh.R2[k];
    for (int k = 0; k < 3; ++k) sh.t[k] = (best & 1) ? -sh.T[k] : sh.T[k];
    sh.front = int(red[best]);
  }
  bar();
}

// ---- absolute pose refinement ------------------------------------------------------------------------------------
struct SharedA { float M[12], Mn[12], lam, best, cost; };
inline void lm_accumulate_abs(const device float4* p, int tot, thread const float* M, float t2, thread float* acc,
                              uint lid) {
  for (int k = 0; k < NLMA; ++k) acc[k] = 0.f;
  for (int i = int(lid); i < tot; i += TPB) {
    const float4 m0 = p[2 * i], m1 = p[2 * i + 1];
    const float X[3] = {m0.z, m0.w, m1.x};
    float c[3];
    for (int r = 0; r < 3; ++r) c[r] = M[r * 3] * X[0] + M[r * 3 + 1] * X[1] + M[r * 3 + 2] * X[2] + M[9 + r];
    if (!(c[2] > 0.f)) { acc[27] += t2; continue; }
    const float iz = 1.f / c[2];
    const float res0 = c[0] * iz - m0.x, res1 = c[1] * iz - m0.y;
    const float e2 = res0 * res0 + res1 * res1;
    if (e2 >= t2) { acc[27] += t2; continue; }
    acc[27] += e2;
    float dc[18];
    for (int r = 0; r < 3; ++r) {
      thread const float* Rr = M + r * 3;
      dc[r * 6] = -(Rr[1] * X[2] - Rr[2] * X[1]);
      dc[r * 6 + 1] = -(Rr[2] * X[0] - Rr[0] * X[2]);
      dc[r * 6 + 2] = -(Rr[0] * X[1] - Rr[1] * X[0]);
      for (int k = 0; k < 3; ++k) dc[r * 6 + 3 + k] = r == k ? 1.f : 0.f;
    }
    float J0[6], J1[6];
    for (int k = 0; k < 6; ++k) {
      J0[k] = (dc[k] - c[0] * iz * dc[12 + k]) * iz;
      J1[k] = (dc[6 + k] - c[1] * iz * dc[12 + k]) * iz;
    }
    int q = 0;
    for (int u = 0; u < 6; ++u)
      for (int v = u; v < 6; ++v) acc[q++] += J0[u] * J0[v] + J1[u] * J1[v];
    for (int k = 0; k < 6; ++k) acc[21 + k] += J0[k] * res0 + J1[k] * res1;
  }
}
inline void score_sum_abs(const device float4* p, int tot, int N, threadgroup const float* Mm, float t2,
                          thread float* acc, device uint8_t* mk, threadgroup float* red, uint lid) {
  float M[12];
  cp(M, Mm, 12);
  acc[0] = acc[1] = 0.f;
  const int end = mk ? N : tot;
  for (int i = int(lid); i < end; i += TPB) {
    float g = i < tot ? gain_abs(p[2 * i], p[2 * i + 1], M, t2) : 0.f;
    acc[0] += g;
    acc[1] += g > 0.f ? 1.f : 0.f;
    if (mk) mk[i] = g > 0.f ? 1 : 0;
  }
  block_sum(acc, 2, red, lid);
}
"""

# one thread per (a, h) sample: dims = (N, H, A); idx [A, H, 5 | 7]; out [A, H, S, 9]; valid [A, H, S] int32
_EPIPOLAR_SRC = """
    const int N = dims[0], H = dims[1], A = dims[2];
    const uint tid = thread_position_in_grid.x;
    if (tid >= uint(A * H)) return;
    const int a = int(tid) / H;
    const device float4* p = (const device float4*)(pts) + uint(pairs[a]) * uint(N);
    int id[SAMPLE];
    for (int r = 0; r < SAMPLE; ++r) id[r] = idx[tid * SAMPLE + r];
    device float* o = out + tid * NS * 9;
    device int* ok = valid + tid * NS;
    if (SAMPLE == 5) {
      if (DF) e5pt_one<df>(p, id, o, ok); else e5pt_one<float>(p, id, o, ok);
    } else {
      if (DF) f7pt_one<df>(p, id, o, ok); else f7pt_one<float>(p, id, o, ok);
    }
"""
_E5PT = metal_kernel(
    name="ransac_e5pt",
    input_names=["pts", "pairs", "idx", "dims"],
    output_names=["out", "valid"],
    header=_HEADER,
    source=_EPIPOLAR_SRC,
)

_P3P_SRC = """
    const int N = dims[0], H = dims[1], A = dims[2];
    const uint tid = thread_position_in_grid.x;
    if (tid >= uint(A * H)) return;
    const int a = int(tid) / H;
    const device float* base = pts + uint(pairs[a]) * uint(N) * 8;
    float y[9], X[9], sol[48];
    for (int i = 0; i < 3; ++i) {
      const device float* m = base + uint(idx[tid * 3 + i]) * 8;
      const float inv = rsqrt(1.f + m[0] * m[0] + m[1] * m[1]);
      y[i * 3] = m[0] * inv; y[i * 3 + 1] = m[1] * inv; y[i * 3 + 2] = inv;
      X[i * 3] = m[2]; X[i * 3 + 1] = m[3]; X[i * 3 + 2] = m[4];
    }
    const int n = p3p_solve(y, X, sol);
    for (int s = 0; s < 4; ++s) {
      valid[tid * 4 + s] = s < n ? 1 : 0;
      for (int k = 0; k < 12; ++k) out[(tid * 4 + s) * 12 + k] = s < n ? sol[s * 12 + k] : 0.f;
    }
"""
_P3P = metal_kernel(
    name="ransac_p3p",
    input_names=["pts", "pairs", "idx", "dims"],
    output_names=["out", "valid"],
    header=_HEADER,
    source=_P3P_SRC,
)

# one thread per candidate, the matches through threadgroup memory: dims = (N, C, n); e [A, C, NM]
_SCORE_MANY = metal_kernel(
    name="ransac_score_many",
    input_names=["pts", "counts", "thr2", "pairs", "e", "dims"],
    output_names=["out", "cnt"],
    header=_HEADER,
    source="""
    const int N = dims[0], C = dims[1], n = dims[2];
    constexpr int NF = ABS ? 2 : 1, NM = ABS ? 12 : 9, PER = CHUNK / NF;
    threadgroup float4 sm[CHUNK];
    const uint a = threadgroup_position_in_grid.y, lid = thread_position_in_threadgroup.x;
    const int c = int(thread_position_in_grid.x), pair = pairs[a];
    const int tot = min(n, counts[pair]);
    const device float4* p = (const device float4*)(pts) + uint(pair) * uint(N) * NF;
    float E[NM];
    if (c < C)
      for (int k = 0; k < NM; ++k) E[k] = e[(a * uint(C) + uint(c)) * NM + k];
    const float t2 = thr2[pair];
    float sc = 0.f;
    int in = 0;
    for (int base = 0; base < tot; base += PER) {
      const int len = min(PER, tot - base);
      bar();
      for (int i = int(lid); i < len * NF; i += TPB) sm[i] = p[base * NF + i];
      bar();
      if (c < C)
        for (int i = 0; i < len; ++i) {
          float g = gain_tg<ABS>(sm, uint(i), E, t2);
          sc += g;
          in += g > 0.f ? 1 : 0;
        }
    }
    if (c < C) {
      out[a * uint(C) + uint(c)] = sc;
      cnt[a * uint(C) + uint(c)] = in;
    }
""",
)

# one threadgroup per (c, a): dims = (N, C, n)
_SCORE_FEW = metal_kernel(
    name="ransac_score_few",
    input_names=["pts", "counts", "thr2", "pairs", "e", "dims"],
    output_names=["out", "cnt"],
    header=_HEADER,
    source="""
    const int N = dims[0], C = dims[1], n = dims[2];
    constexpr int NF = ABS ? 2 : 1, NM = ABS ? 12 : 9;
    threadgroup float red[8 * NT];
    const uint c = threadgroup_position_in_grid.x, a = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const int pair = pairs[a];
    const int tot = min(n, counts[pair]);
    const device float4* p = (const device float4*)(pts) + uint(pair) * uint(N) * NF;
    float E[NM];
    for (int k = 0; k < NM; ++k) E[k] = e[(a * uint(C) + c) * NM + k];
    const float t2 = thr2[pair];
    float acc[2] = {0.f, 0.f};
    for (int i = int(lid); i < tot; i += TPB) {
      float g = gain_at<ABS>(p, uint(i), E, t2);
      acc[0] += g;
      acc[1] += g > 0.f ? 1.f : 0.f;
    }
    block_sum(acc, 2, red, lid);
    if (lid == 0) {
      out[a * uint(C) + c] = red[0];
      cnt[a * uint(C) + c] = int(red[1]);
    }
""",
)

# one threadgroup per pair: dims = (N, refits, lm_iterations, minimum)
_POLISH = metal_kernel(
    name="ransac_polish",
    input_names=["pts", "counts", "thr2", "e", "dims"],
    output_names=["e_out", "r", "t", "cnt", "ok", "mask"],
    header=_HEADER,
    source="""
    const int N = dims[0], refits = dims[1], lm_iterations = dims[2], minimum = dims[3];
    threadgroup Shared sh;
    threadgroup float red[8 * NT];
    const uint pair = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const int tot = counts[pair];
    const device float4* p = (const device float4*)(pts) + pair * uint(N);
    const float t2 = thr2[pair];
    if (lid < 9) sh.E[lid] = e[pair * 9 + lid];
    bar();
    float acc[NT];
    score_sum(p, tot, N, sh.E, t2, acc, nullptr, red, lid);
    if (lid == 0) sh.best = red[0];
    for (int it = 0; it < refits; ++it) {
      {
        float E[9];
        cp(E, sh.E, 9);
        for (int k = 0; k < 46; ++k) acc[k] = 0.f;
        for (int i = int(lid); i < tot; i += TPB) {
          const float4 m = p[i];
          if (!(gain(m, E, t2) > 0.f)) continue;
          const float row[9] = {m.z * m.x, m.z * m.y, m.z, m.w * m.x, m.w * m.y, m.w, m.x, m.y, 1.f};
          int q = 0;
          for (int u = 0; u < 9; ++u)
            for (int v = u; v < 9; ++v) acc[q++] += row[u] * row[v];
          acc[45] += 1.f;
        }
      }
      block_sum(acc, 46, red, lid);
      if (lid == 0) {
        sh.tryv = 0;
        if (red[45] >= 8.f) {
          float A9[81], V9[81];
          int q = 0;
          for (int u = 0; u < 9; ++u)
            for (int v = u; v < 9; ++v) A9[u * 9 + v] = A9[v * 9 + u] = red[q++];
          jacobi_eig(A9, V9, 9);
          int lo = 0;
          for (int k = 1; k < 9; ++k) lo = A9[k * 9 + k] < A9[lo * 9 + lo] ? k : lo;
          float vec[9], En[9];
          for (int k = 0; k < 9; ++k) vec[k] = V9[k * 9 + lo];
          sh.tryv = project_rank2(vec, En, FUND != 0) ? 1 : 0;
          cp(sh.En, En, 9);
        }
      }
      bar();
      if (!sh.tryv) continue;
      score_sum(p, tot, N, sh.En, t2, acc, nullptr, red, lid);
      if (lid == 0 && red[0] > sh.best) {
        sh.best = red[0];
        cp(sh.E, sh.En, 9);
      }
      bar();
    }
    if (lm_iterations > 0 && FUND) {
      if (lid == 0) {
        float E[9], U[9], V[9], sig[3];
        cp(E, sh.E, 9);
        sh.tryv = svd3(E, U, V, sig) ? 1 : 0;
        cp(sh.R1, U, 9); cp(sh.R2, V, 9);
        sh.T[0] = atan2(sig[1], sig[0]);
      }
      bar();
      if (sh.tryv) {
        lm_run_f(p, tot, t2, lm_iterations, sh, acc, red, lid);
        score_sum(p, tot, N, sh.En, t2, acc, nullptr, red, lid);
        if (lid == 0 && red[0] > sh.best) {
          sh.best = red[0];
          cp(sh.E, sh.En, 9);
        }
        bar();
      }
    } else if (lm_iterations > 0) {
      recover(p, tot, t2, sh, acc, red, lid);
      if (sh.front > 0) {
        lm_run(p, tot, t2, lm_iterations, sh, acc, red, lid);
        if (lid == 0) {
          float tx[9], t3[3], R[9], En[9];
          cp(t3, sh.t, 3); cp(R, sh.R, 9);
          cross_mat(t3, tx);
          mat_mul(tx, R, En);
          float nrm = 0.f;
          for (int k = 0; k < 9; ++k) nrm += En[k] * En[k];
          nrm = rsqrt(max(nrm, 1e-30f));
          for (int k = 0; k < 9; ++k) sh.En[k] = En[k] * nrm;
        }
        bar();
        score_sum(p, tot, N, sh.En, t2, acc, nullptr, red, lid);
        if (lid == 0 && red[0] > sh.best) {
          sh.best = red[0];
          cp(sh.E, sh.En, 9);
        }
        bar();
      }
    }
    if (FUND) {
      if (lid == 0) sh.front = 1;
      bar();
    } else {
      recover(p, tot, t2, sh, acc, red, lid);
    }
    score_sum(p, tot, N, sh.E, t2, acc, mask + pair * uint(N), red, lid);
    if (lid == 0) {
      const int count = int(red[1]);
      const bool good = count >= minimum && (FUND || sh.front > 0);
      cnt[pair] = count;
      ok[pair] = good ? 1 : 0;
      for (int k = 0; k < 9; ++k) {
        e_out[pair * 9 + k] = sh.E[k];
        r[pair * 9 + k] = good ? sh.R[k] : (k % 4 == 0 ? 1.f : 0.f);
      }
      for (int k = 0; k < 3; ++k) t[pair * 3 + k] = good ? sh.t[k] : (k == 2 ? 1.f : 0.f);
    }
""",
)

# one threadgroup per pair: dims = (N, lm_iterations, minimum)
_POLISH_ABS = metal_kernel(
    name="ransac_polish_abs",
    input_names=["pts", "counts", "thr2", "m", "dims"],
    output_names=["m_out", "cnt", "ok", "mask"],
    header=_HEADER,
    source="""
    const int N = dims[0], lm_iterations = dims[1], minimum = dims[2];
    threadgroup SharedA sh;
    threadgroup float red[8 * NT];
    const uint pair = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const int tot = counts[pair];
    const device float4* p = (const device float4*)(pts) + pair * uint(N) * 2;
    const float t2 = thr2[pair];
    if (lid < 12) sh.M[lid] = m[pair * 12 + lid];
    if (lid == 0) sh.lam = 1e-3f;
    bar();
    float acc[NT];
    score_sum_abs(p, tot, N, sh.M, t2, acc, nullptr, red, lid);
    if (lid == 0) {
      sh.best = red[0];
      sh.cost = t2 * (float(tot) - red[0]);
    }
    for (int it = 0; it < lm_iterations; ++it) {
      {
        float M[12];
        cp(M, sh.M, 12);
        lm_accumulate_abs(p, tot, M, t2, acc, lid);
      }
      block_sum(acc, NLMA, red, lid);
      if (lid == 0) {
        float step[6], Rw[9], M[12], Mn[12];
        lm_step<6>(red, sh.lam, step);
        cp(M, sh.M, 12);
        exp_so3(step, Rw);
        mat_mul(M, Rw, Mn);
        for (int k = 0; k < 3; ++k) Mn[9 + k] = M[9 + k] + step[3 + k];
        cp(sh.Mn, Mn, 12);
      }
      bar();
      score_sum_abs(p, tot, N, sh.Mn, t2, acc, nullptr, red, lid);
      if (lid == 0) {
        const float cost = t2 * (float(tot) - red[0]);
        if (cost < sh.cost) {
          sh.cost = cost;
          sh.best = red[0];
          cp(sh.M, sh.Mn, 12);
          sh.lam = max(sh.lam * 0.1f, 1e-6f);
        } else {
          sh.lam = min(sh.lam * 10.f, 1e6f);
        }
      }
      bar();
    }
    score_sum_abs(p, tot, N, sh.M, t2, acc, mask + pair * uint(N), red, lid);
    if (lid == 0) {
      const int count = int(red[1]);
      cnt[pair] = count;
      ok[pair] = count >= minimum ? 1 : 0;
      for (int k = 0; k < 12; ++k) m_out[pair * 12 + k] = sh.M[k];
    }
""",
)


def _i32(*v):
    return mx.array(v, dtype=mx.int32)


def hypotheses(pts, pairs, idx, *, double_float=DOUBLE_FLOAT):
    """Models `[A, H*S, M]` and validity `[A, H*S]` (bool) of the samples `idx` `[A, H, 5 | 7 | 3]` of `pts[pairs]`.

    Five-point samples give up to `S = 10` essential matrices each (`M = 9`), seven-point samples up to `S = 3`
    fundamental matrices, three-point samples of 2D-3D matches (`pts` `[P, N, 8]`) up to `S = 4` poses (`M = 12`).
    """
    a, h, size = idx.shape
    _, s, m = SOLVERS[size]
    n = pts.shape[1]
    tg = SMALL_THREADS if a * h <= SMALL else THREADS
    dims = _i32(n, h, a)
    shapes, dtypes = [(a, h, s, m), (a, h, s)], [mx.float32, mx.int32]
    if size == 3:  # noqa: PLR2004 - P3P
        out, valid = _P3P(inputs=[pts, pairs, idx, dims], grid=(a * h, 1, 1), threadgroup=(tg, 1, 1),
                          output_shapes=shapes, output_dtypes=dtypes)  # fmt: skip
    else:
        tpl = [("SAMPLE", size), ("NS", s), ("DF", int(double_float))]
        out, valid = _E5PT(inputs=[pts, pairs, idx, dims], template=tpl, grid=(a * h, 1, 1), threadgroup=(tg, 1, 1),
                           output_shapes=shapes, output_dtypes=dtypes)  # fmt: skip
    return out.reshape(a, h * s, m), valid.reshape(a, h * s).astype(mx.bool_)


def score(pts, counts, thr2, pairs, e, n):
    """MSAC scores (float32) and inlier counts (int32), `[A, C]` each, of the candidates `e` `[A, C, 9 | 12]`.

    Scored on the first `min(n, counts)` matches of `pts[pairs]`.
    """
    a, c, m = e.shape
    dims = _i32(pts.shape[1], c, n)
    tpl = [("ABS", int(m == SOLVERS[3][2]))]
    shapes, dtypes = [(a, c), (a, c)], [mx.float32, mx.int32]
    inputs = [pts, counts, thr2, pairs, e, dims]
    if c >= MANY:
        grid = ((c + THREADS - 1) // THREADS * THREADS, a, 1)
        return _SCORE_MANY(inputs=inputs, template=tpl, grid=grid, threadgroup=(THREADS, 1, 1), output_shapes=shapes,
                           output_dtypes=dtypes)  # fmt: skip
    return _SCORE_FEW(inputs=inputs, template=tpl, grid=(THREADS * c, a, 1), threadgroup=(THREADS, 1, 1),
                      output_shapes=shapes, output_dtypes=dtypes)  # fmt: skip


def polish(pts, counts, thr2, e, refits, lm_iterations, *, fundamental=False, minimum=5):
    """The final stage of one model per pair `e` `[P, 9]`: refits, `recoverPose`, LM refinement, inlier mask.

    Returns `R` `[P, 3, 3]`, `t` `[P, 3]`, the polished model `[P, 9]`, inlier counts `[P]` int32, `ok` `[P]` bool
    (at least `minimum` inliers and a valid cheirality) and the inlier mask `[P, N]` bool; `R`, `t` are the identity
    and `(0, 0, 1)` where not `ok`. `fundamental`: rank-2 refits and LM, no pose.
    """
    p, n = pts.shape[:2]
    dims = _i32(n, refits, lm_iterations, minimum)
    shapes = [(p, 9), (p, 3, 3), (p, 3), (p,), (p,), (p, n)]
    dtypes = [mx.float32, mx.float32, mx.float32, mx.int32, mx.int32, mx.uint8]
    e_out, r, t, cnt, ok, mask = _POLISH(inputs=[pts, counts, thr2, e, dims], template=[("FUND", int(fundamental))],
                                         grid=(THREADS * p, 1, 1), threadgroup=(THREADS, 1, 1), output_shapes=shapes,
                                         output_dtypes=dtypes)  # fmt: skip
    return r, t, e_out, cnt, ok.astype(mx.bool_), mask.astype(mx.bool_)


def polish_abs(pts, counts, thr2, m, lm_iterations, minimum):
    """The final stage of one pose per pair `m` `[P, 12]` on the 2D-3D matches `pts` `[P, N, 8]`: LM, inlier mask.

    Returns the polished poses `[P, 12]`, inlier counts `[P]` int32, `ok` `[P]` bool (at least `minimum` inliers) and
    the inlier mask `[P, N]` bool.
    """
    p, n = pts.shape[:2]
    dims = _i32(n, lm_iterations, minimum)
    shapes, dtypes = [(p, 12), (p,), (p,), (p, n)], [mx.float32, mx.int32, mx.int32, mx.uint8]
    m_out, cnt, ok, mask = _POLISH_ABS(inputs=[pts, counts, thr2, m, dims], grid=(THREADS * p, 1, 1),
                                       threadgroup=(THREADS, 1, 1), output_shapes=shapes,
                                       output_dtypes=dtypes)  # fmt: skip
    return m_out, cnt, ok.astype(mx.bool_), mask.astype(mx.bool_)
