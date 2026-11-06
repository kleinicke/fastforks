// Fused (modulated) deformable convolution, fp16 NHWC in / out, fp32 sampling and accumulation.
//
// One block computes a tile of TM output pixels x TN output channels as an implicit GEMM over (tap, input channel):
// per tap it derives the bilinear corners and weights of every (pixel, deformable group) from the offset / mask
// tensor (mask = mscale * sigmoid(logit)), gathers the input channels of a KC chunk in fp32, scales them and stores
// fp16 columns in shared memory, then multiplies them with the tap's weight slice on tensor cores (WMMA, fp32
// accumulate). Epilogue: y = act(acc * s + t), s / t fp32 per output channel (optional). Nothing but the output goes
// to global memory: no column buffer.
//
// Math as mmdetection's modulated_deform_conv / torch.nn.functional.grid_sample(align_corners=True, zeros): sample
// (ho * sh - ph + i * dh + dy, wo * sw - pw + j * dw + dx); a corner outside the image contributes zero.
//
// Layouts:
//   x  [B, H, W, C] fp16, pixel stride xps (channels contiguous; xps % 8 == 0, 16-byte aligned)
//   om [B, Ho, Wo, *] fp16, pixel stride omps: channel (g * K + k) * 2 + {0: dy, 1: dx}, then (modulated) the mask
//      logits at dg * 2 * K + g * K + k
//   w  [K * C, Op] fp16 (row = tap * C + c, Op = O rounded up to 16, zero columns)
//   y  [B, Ho, Wo, O] fp16, pixel stride yps
// Requirements (checked by the Python wrapper): C % 16 == 0, (C / dg) % 8 == 0, dg <= DGMAX.
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

#define TM 128  // output pixels per block: every warp owns 16 pixels x all TN columns
#define TN 128
#define KC 32
#define NT 256
#define DGMAX 4
#define LDA (KC + 8)
#define LDB (TN + 8)

struct __align__(16) Tap {
  int i[4];    // pixel index (b * H + y) * W + x of the corners 00, 01, 10, 11 (0 if outside)
  float w[4];  // bilinear weight * mask (0 if outside)
};

__device__ __forceinline__ void fma8(float* acc, uint4 v, float wt) {
  const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float2 f = __half22float2(h[q]);
    acc[2 * q] += wt * f.x;
    acc[2 * q + 1] += wt * f.y;
  }
}

struct Tiles {
  __half a[TM * LDA];
  __half b[KC * LDB];
};

extern "C" __global__ void __launch_bounds__(NT) deform_conv_fwd(
    const __half* __restrict__ x, const __half* __restrict__ om, const __half* __restrict__ w,
    const float* __restrict__ s, const float* __restrict__ t, __half* __restrict__ y, int B, int H, int W, int C,
    int xps, int Ho, int Wo, int omps, int O, int Op, int yps, int sh, int sw, int ph, int pw, int dh, int dw, int kh,
    int kw, int dg, int modulated, float mscale, int act) {
  __shared__ __align__(32) Tiles sm;
  __shared__ Tap taps[TM * DGMAX];
  __shared__ __align__(32) float stage[8][16 * 16];

  const int K = kh * kw;
  const int cg = C / dg;
  const long long P = (long long)B * Ho * Wo;
  const long long p0 = (long long)blockIdx.x * TM;
  const int n0 = blockIdx.y * TN;
  const int nf = min(TN, Op - n0) >> 4;  // column fragments
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) wmma::fill_fragment(acc[j], 0.f);

  for (int tap = 0; tap < K; ++tap) {
    __syncthreads();
    const int ki = tap / kw, kj = tap - ki * kw;
    for (int e = threadIdx.x; e < TM * dg; e += NT) {
      const int pl = e / dg, g = e - pl * dg;
      const long long p = p0 + pl;
      Tap tp;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        tp.i[q] = 0;
        tp.w[q] = 0.f;
      }
      if (p < P) {
        const int wo = (int)(p % Wo);
        const long long r = p / Wo;
        const int ho = (int)(r % Ho);
        const int b = (int)(r / Ho);
        const __half* o = om + p * omps;
        const float dy = __half2float(o[(g * K + tap) * 2]);
        const float dx = __half2float(o[(g * K + tap) * 2 + 1]);
        float m = 1.f;
        if (modulated) m = mscale / (1.f + __expf(-__half2float(o[dg * 2 * K + g * K + tap])));
        const float py = fminf(fmaxf((float)(ho * sh - ph + ki * dh) + dy, -2.f), (float)H + 1.f);
        const float px = fminf(fmaxf((float)(wo * sw - pw + kj * dw) + dx, -2.f), (float)W + 1.f);
        const float fy = floorf(py), fx = floorf(px);
        const int y0 = (int)fy, x0 = (int)fx;
        const float ly = py - fy, lx = px - fx;
        const float wy[2] = {1.f - ly, ly}, wx[2] = {1.f - lx, lx};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int yy = y0 + (q >> 1), xx = x0 + (q & 1);
          if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
            tp.i[q] = (b * H + yy) * W + xx;
            tp.w[q] = wy[q >> 1] * wx[q & 1] * m;
          }
        }
      }
      taps[pl * DGMAX + g] = tp;
    }

    for (int c0 = 0; c0 < C; c0 += KC) {
      const int kc = min(KC, C - c0);
      __syncthreads();
      const int nv = kc >> 3;
      for (int e = threadIdx.x; e < TM * nv; e += NT) {
        const int pl = e / nv, v = e - pl * nv;
        const int c = c0 + v * 8;
        const Tap tp = taps[pl * DGMAX + c / cg];
        float a8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 4; ++q)
          if (tp.w[q] != 0.f) fma8(a8, __ldg(reinterpret_cast<const uint4*>(x + (long long)tp.i[q] * xps + c)), tp.w[q]);
        uint4 pk;
        __half2* h = reinterpret_cast<__half2*>(&pk);
#pragma unroll
        for (int q = 0; q < 4; ++q) h[q] = __floats2half2_rn(a8[2 * q], a8[2 * q + 1]);
        *reinterpret_cast<uint4*>(&sm.a[pl * LDA + v * 8]) = pk;
      }
      const int nvb = nf * 2;
      const __half* wt = w + (long long)(tap * C + c0) * Op + n0;
      for (int e = threadIdx.x; e < kc * nvb; e += NT) {
        const int r = e / nvb, v = e - r * nvb;
        *reinterpret_cast<uint4*>(&sm.b[r * LDB + v * 8]) =
            __ldg(reinterpret_cast<const uint4*>(wt + (long long)r * Op + v * 8));
      }
      __syncthreads();
      for (int kk = 0; kk < kc; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::load_matrix_sync(fa, &sm.a[warp * 16 * LDA + kk], LDA);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (j < nf) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
            wmma::load_matrix_sync(fb, &sm.b[kk * LDB + j * 16], LDB);
            wmma::mma_sync(acc[j], fa, fb, acc[j]);
          }
        }
      }
    }
  }

  // epilogue: per warp, one 16 x 16 fragment at a time; lane -> row lane / 2, columns (lane & 1) * 8 .. + 8
  const int row = lane >> 1, cb = (lane & 1) * 8;
  const long long p = p0 + warp * 16 + row;
  const bool vec = (O % 8 == 0) && (yps % 8 == 0) && ((reinterpret_cast<uintptr_t>(y) & 15) == 0);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if (j >= nf) break;
    wmma::store_matrix_sync(stage[warp], acc[j], 16, wmma::mem_row_major);
    __syncwarp();
    const int nb = n0 + j * 16 + cb;
    if (p < P && nb < O) {
      float f[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int n = min(nb + u, O - 1);
        float val = stage[warp][row * 16 + cb + u];
        if (s) val *= s[n];
        if (t) val += t[n];
        if (act == 1) val = fmaxf(val, 0.f);
        f[u] = val;
      }
      if (vec) {
        uint4 pk;
        __half2* h = reinterpret_cast<__half2*>(&pk);
#pragma unroll
        for (int q = 0; q < 4; ++q) h[q] = __floats2half2_rn(f[2 * q], f[2 * q + 1]);
        *reinterpret_cast<uint4*>(y + p * yps + nb) = pk;
      } else {
        for (int u = 0; u < 8 && nb + u < O; ++u) y[p * yps + nb + u] = __float2half_rn(f[u]);
      }
    }
    __syncwarp();
  }
}
