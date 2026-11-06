// Small stereo building blocks for the max paths (NHWC fp16 inputs, fp32 accumulation), size-independent: the
// correlation cost volume (LAS2, CoEx), depthwise line convolutions (1xK / Kx1, LAS2's attention modules), the
// soft-argmax disparity regression and the superpixel upsampling (LAS2, CoEx). Launched on torch's current stream.
#include <cuda_fp16.h>

// Correlation volume. l, r: [N, H, W, C] fp16 (NHWC); out fp16 at element strides (sn, sy, sx, sd):
//   out[n, y, x, d] = scale * sum_c l[n, y, x, c] * r[n, y, x - d, c]   (0 for x < d), d < D
// (LAS2's build_correlation_volume: scale 1 / C; CoEx's CostVolume: scale 1). Block: 32 pixels x 32 disparities of
// one row; 256 threads, 4 pixels each; channels in chunks of 32 through shared memory.
#define CT 32
extern "C" __global__ void corr_volume(const __half* __restrict__ l, const __half* __restrict__ r,
                                       __half* __restrict__ out, int H, int W, int C, int D, long long sn,
                                       long long sy, long long sx, long long sd, float scale) {
  __shared__ float ls[CT][CT + 1];
  __shared__ float rs[2 * CT - 1][CT + 1];
  const int xtiles = (W + CT - 1) / CT;
  const int x0 = (blockIdx.x % xtiles) * CT, d0 = (blockIdx.x / xtiles) * CT;
  const long long row = blockIdx.y;  // n * H + y
  const __half* lrow = l + row * W * C;
  const __half* rrow = r + row * W * C;
  const int tid = threadIdx.x, dl = tid % CT, xg = tid / CT;  // xg: 0..7, pixels xg * 4 + i
  const int base = x0 - d0 - (CT - 1);                          // rs[j] = r[base + j]
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int c0 = 0; c0 < C; c0 += CT) {
    for (int i = tid; i < CT * CT; i += blockDim.x) {
      int xi = i / CT, ci = i % CT, x = x0 + xi, c = c0 + ci;
      ls[xi][ci] = (x < W && c < C) ? __half2float(lrow[(long long)x * C + c]) : 0.f;
    }
    for (int i = tid; i < (2 * CT - 1) * CT; i += blockDim.x) {
      int j = i / CT, ci = i % CT, x = base + j, c = c0 + ci;
      rs[j][ci] = (x >= 0 && x < W && c < C) ? __half2float(rrow[(long long)x * C + c]) : 0.f;
    }
    __syncthreads();
#pragma unroll 4
    for (int ci = 0; ci < CT; ++ci) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int xl = xg * 4 + i;
        acc[i] += ls[xl][ci] * rs[xl - dl + CT - 1][ci];
      }
    }
    __syncthreads();
  }
  const int d = d0 + dl;
  if (d >= D) return;
  __half* o = out + (row / H) * sn + (row % H) * sy + d * sd;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int x = x0 + xg * 4 + i;
    if (x < W) o[x * sx] = __float2half(acc[i] * scale);
  }
}

// Depthwise line convolution, zero padding K / 2 (odd K), stride 1. x, y: [N, H, W, C] fp16 (NHWC); w: [K, C] fp32;
// b: [C] fp32 or null; res: like y or null (may alias y: each element is read before it is written).
//   y[n, h, w, c] = res[n, h, w, c] + b[c] + sum_k w[k, c] * x[n, h, w + k - K / 2, c]   (axis 0: along w, 1: along h)
// vec (C a multiple of 8, 16-byte aligned tensors): one thread per pixel and 8 channels, 16-byte loads and stores.
extern "C" __global__ void dw_line(const __half* __restrict__ x, const float* __restrict__ w,
                                   const float* __restrict__ b, const __half* res, __half* y, long long P, int H,
                                   int W, int C, int K, int axis, int vec) {
  const int half = K / 2;
  const int g = vec ? 8 : 1;  // channels per thread
  const int nc = C / g;
  const long long n_el = P * nc;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n_el;
       i += (long long)gridDim.x * blockDim.x) {
    const long long p = i / nc;
    const int c = (int)(i - p * nc) * g;
    const int px = (int)(p % W), py = (int)((p / W) % H);
    const int pos = axis == 0 ? px : py, len = axis == 0 ? W : H;
    const long long step = axis == 0 ? (long long)C : (long long)W * C;
    const long long e = p * C + c;
    if (!vec) {
      float v = b ? b[c] : 0.f;
      for (int k = 0; k < K; ++k) {
        const int q = pos + k - half;
        if (q >= 0 && q < len) v += w[(long long)k * C + c] * __half2float(x[e + (long long)(k - half) * step]);
      }
      if (res) v += __half2float(res[e]);
      y[e] = __float2half(v);
      continue;
    }
    float a[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = b ? b[c + j] : 0.f;
    const int k_lo = max(0, half - pos), k_hi = min(K, len - pos + half);
    for (int k = k_lo; k < k_hi; ++k) {
      const uint4 xv = *reinterpret_cast<const uint4*>(x + e + (long long)(k - half) * step);
      const __half2* xh = reinterpret_cast<const __half2*>(&xv);
      const float4 w0 = *reinterpret_cast<const float4*>(w + (long long)k * C + c);
      const float4 w1 = *reinterpret_cast<const float4*>(w + (long long)k * C + c + 4);
      const float2 x0 = __half22float2(xh[0]), x1 = __half22float2(xh[1]), x2 = __half22float2(xh[2]),
                   x3 = __half22float2(xh[3]);
      a[0] += w0.x * x0.x, a[1] += w0.y * x0.y, a[2] += w0.z * x1.x, a[3] += w0.w * x1.y;
      a[4] += w1.x * x2.x, a[5] += w1.y * x2.y, a[6] += w1.z * x3.x, a[7] += w1.w * x3.y;
    }
    if (res) {
      const uint4 rv = *reinterpret_cast<const uint4*>(res + e);
      const __half2* rh = reinterpret_cast<const __half2*>(&rv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 r2 = __half22float2(rh[j]);
        a[2 * j] += r2.x, a[2 * j + 1] += r2.y;
      }
    }
    __align__(16) __half2 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __floats2half2_rn(a[2 * j], a[2 * j + 1]);
    *reinterpret_cast<uint4*>(y + e) = *reinterpret_cast<const uint4*>(o);
  }
}

// Soft-argmax over the disparity axis (LAS2's softmax + disparity_regression; CoEx's top-k variant).
// cost: fp16 at element strides (sn, sd, sy, sx), [N, D, H, W]; out: fp32 [N, H, W] (dense), P = N * H * W.
//   topk <= 0: out = sum_d softmax_d(cost) * d
//   topk 1..8: the topk largest costs (lower d first on ties), out = sum_j softmax_j(cost[d_j]) * d_j
#define TOPK_MAX 8
extern "C" __global__ void soft_argmax(const __half* __restrict__ cost, float* __restrict__ out, long long P, int H,
                                       int W, int D, long long sn, long long sd, long long sy, long long sx,
                                       int topk) {
  for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x; p < P;
       p += (long long)gridDim.x * blockDim.x) {
    const int x = (int)(p % W), y = (int)((p / W) % H);
    const long long n = p / ((long long)H * W);
    const __half* c = cost + n * sn + y * sy + x * sx;
    float r = 0.f;
    if (topk <= 0) {
      float m = -INFINITY;
      for (int d = 0; d < D; ++d) m = fmaxf(m, __half2float(c[d * sd]));
      float s = 0.f, sv = 0.f;
      for (int d = 0; d < D; ++d) {
        float e = expf(__half2float(c[d * sd]) - m);
        s += e;
        sv += e * (float)d;
      }
      r = sv / s;
    } else {
      float v[TOPK_MAX];
      int ix[TOPK_MAX];
      const int k = topk < D ? topk : D;
      for (int j = 0; j < k; ++j) v[j] = -INFINITY, ix[j] = -1;
      for (int d = 0; d < D; ++d) {  // insertion into the descending list; strict > keeps the lower d on ties
        float t = __half2float(c[d * sd]);
        if (ix[k - 1] >= 0 && !(t > v[k - 1])) continue;
        int j = k - 1;
        while (j > 0 && (ix[j - 1] < 0 || t > v[j - 1])) {
          v[j] = v[j - 1], ix[j] = ix[j - 1];
          --j;
        }
        v[j] = t, ix[j] = d;
      }
      float s = 0.f, sv = 0.f;
      for (int j = 0; j < k; ++j) {
        float e = expf(v[j] - v[0]);
        s += e;
        sv += e * (float)ix[j];
      }
      r = sv / s;
    }
    out[p] = r;
  }
}

// Superpixel (context) upsampling by an integer factor F with 9 softmax weights per output pixel (LAS2, CoEx):
//   out[n, Y, X] = scale * sum_k softmax_k(s[n, k, Y, X]) * low[n, Y / F + k / 3 - 1, X / F + k % 3 - 1]
// (0 outside low). low: fp32 [N, h, w] (dense); s: fp16 logits at element strides (sn, sk, sy, sx), [N, 9, h F, w F];
// out: fp32 [N, h F, w F] (dense), P = N * h F * w F.
extern "C" __global__ void spx_upsample(const float* __restrict__ low, const __half* __restrict__ s,
                                        float* __restrict__ out, long long P, int h, int w, int F, long long sn,
                                        long long sk, long long sy, long long sx, float scale) {
  const int Hh = h * F, Ww = w * F;
  for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x; p < P;
       p += (long long)gridDim.x * blockDim.x) {
    const int X = (int)(p % Ww), Y = (int)((p / Ww) % Hh);
    const long long n = p / ((long long)Hh * Ww);
    const __half* sp = s + n * sn + Y * sy + X * sx;
    float l[9], m = -INFINITY;
#pragma unroll
    for (int k = 0; k < 9; ++k) m = fmaxf(m, l[k] = __half2float(sp[k * sk]));
    const int cy = Y / F, cx = X / F;
    const float* lo = low + n * h * w;
    float sum = 0.f, acc = 0.f;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      float e = expf(l[k] - m);
      sum += e;
      int yy = cy + k / 3 - 1, xx = cx + k % 3 - 1;
      if (yy >= 0 && yy < h && xx >= 0 && xx < w) acc += e * lo[(long long)yy * w + xx];
    }
    out[p] = scale * acc / sum;
  }
}

// Input stage of a stereo pair: replicate padding, scaling and normalisation in one pass. l, r: fp32 [N, 3, H, W]
// (dense); out: fp16 [2N, 3, Hp, Wp] channels-last (dense), left images first, the image placed at (top, left):
// out = (img[clamp(y - top), clamp(x - left)] * mul - mean[c]) / std[c]; ms: fp32 [2, 3] (mean, std).
extern "C" __global__ void norm_pad_pair(const float* __restrict__ l, const float* __restrict__ r,
                                         const float* __restrict__ ms, __half* __restrict__ out, long long N, int H,
                                         int W, int Hp, int Wp, int top, int left, float mul, int zero_fill) {
  const long long HW = (long long)H * W, HWp = (long long)Hp * Wp, P = 2 * N * HWp;
  for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x; p < P;
       p += (long long)gridDim.x * blockDim.x) {
    const long long b = p / HWp, q = p - b * HWp;
    const int yu = (int)(q / Wp) - top, xu = (int)(q % Wp) - left;
    if (zero_fill && (yu < 0 || yu >= H || xu < 0 || xu >= W)) {  // zeros after the normalisation
#pragma unroll
      for (int c = 0; c < 3; ++c) out[p * 3 + c] = __float2half(0.f);
      continue;
    }
    const int y = min(max(yu, 0), H - 1), x = min(max(xu, 0), W - 1);  // replicate
    const float* img = (b < N ? l + b * 3 * HW : r + (b - N) * 3 * HW) + (long long)y * W + x;
#pragma unroll
    for (int c = 0; c < 3; ++c) out[p * 3 + c] = __float2half((img[c * HW] * mul - ms[c]) / ms[3 + c]);
  }
}
