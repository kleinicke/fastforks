// HITNet fused kernels (max path). Same math as hitnet/model.py (TF warp_1d semantics), fp32 accumulation.
// Feature maps are NHWC (channels_last) fp16; outputs fp32 unless stated.
#include <cuda_fp16.h>

template <typename T> __device__ __forceinline__ float ld(const T* p);
template <> __device__ __forceinline__ float ld<__half>(const __half* p) { return __half2float(*p); }

// Level init: lt [N,h,w,16], rt [N,h,Wr,16] (NHWC). cost(d) = sum_c |lt(x) - rt(clip(4x+1-d))|, argmin over d
// (first minimum, like TF ArgMin). out_d / out_c: [N,h,w] fp32.
template <typename T>
__device__ void init_cost_impl(const T* __restrict__ lt, const T* __restrict__ rt, int N, int h, int w, int Wr, int D,
                               float* __restrict__ out_d, float* __restrict__ out_c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * h * w) return;
  int x = i % w, y = (i / w) % h, n = i / (w * h);
  float l[16];
  const T* lp = lt + (size_t)i * 16;
#pragma unroll
  for (int c = 0; c < 16; c++) l[c] = ld(lp + c);
  const T* rrow = rt + ((size_t)n * h + y) * Wr * 16;
  float best = 3.4e38f;
  int bd = 0;
  for (int d = 0; d < D; d++) {
    int xi = 4 * x + 1 - d;
    xi = xi < 0 ? 0 : (xi > Wr - 1 ? Wr - 1 : xi);
    const T* rp = rrow + (size_t)xi * 16;
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < 16; c++) s += fabsf(l[c] - ld(rp + c));
    if (s < best) { best = s; bd = d; }
  }
  out_d[i] = (float)bd;
  out_c[i] = best;
}

// Tile SAD features of one hypothesis: hyp [N,Hh,h,w] fp32 NCHW (channels 0..2 = d, dx, dy; hstride = channel stride),
// fl / fr [N,H,W,C] NHWC with H = 4h, W = 4w. out [N,64,h,w] fp32 NCHW, channel = by*16 + bx*4 + k,
// k: 0 sum|fl|, 1..3 SAD at offsets +1, 0, -1 of x - d_pixel.
template <typename T>
__device__ void tile_sad_impl(const float* __restrict__ hyp, long hn, long hc, const T* __restrict__ fl,
                              const T* __restrict__ fr, int N, int h, int w, int C, float* __restrict__ out, long on) {
  int H = 4 * h, W = 4 * w;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * H * W) return;
  int X = i % W, Y = (i / W) % H, n = i / (W * H);
  int tx = X >> 2, ty = Y >> 2, bx = X & 3, by = Y & 3;
  const float* hp = hyp + n * hn + (long)ty * w + tx;
  float dd = hp[0] + (bx - 1.5f) * hp[hc] + (by - 1.5f) * hp[2 * hc];
  float base = (float)X - dd;
  float fb = floorf(base);
  // positions needed: floor(base)-1 .. floor(base)+2 (offsets -1, 0, +1 and their right neighbours)
  int p[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float q = fb - 1.f + j;
    q = q < 0.f ? 0.f : (q > (float)(W - 1) ? (float)(W - 1) : q);
    p[j] = (int)q;
  }
  // weights: x_o = base + o; floor = fb + o; w_o = x_o - clip(floor_o)
  float wgt[3];
#pragma unroll
  for (int k = 0; k < 3; k++) {  // k=0: o=+1, k=1: o=0, k=2: o=-1
    int o = 1 - k;
    float f = fb + o;
    float fc = f < 0.f ? 0.f : (f > (float)(W - 1) ? (float)(W - 1) : f);
    wgt[k] = base + o - fc;
  }
  const T* lp = fl + (size_t)i * C;
  const T* rrow = fr + ((size_t)n * H + Y) * W * C;
  float sl = 0.f, s[3] = {0.f, 0.f, 0.f};
  for (int c = 0; c < C; c++) {
    float l = ld(lp + c);
    float r[4];
#pragma unroll
    for (int j = 0; j < 4; j++) r[j] = ld(rrow + (size_t)p[j] * C + c);
    sl += fabsf(l);
    // offset +1 uses p[2], p[3]; 0 uses p[1], p[2]; -1 uses p[0], p[1]
#pragma unroll
    for (int k = 0; k < 3; k++) {
      float a = r[2 - k], b = r[3 - k];
      s[k] += fabsf(l - (a + (b - a) * wgt[k]));
    }
  }
  float* op = out + n * on + (long)ty * w + tx;
  long cs = (long)h * w;
  int cb = by * 16 + bx * 4;
  op[(cb + 0) * cs] = sl;
  op[(cb + 1) * cs] = s[0];
  op[(cb + 2) * cs] = s[1];
  op[(cb + 3) * cs] = s[2];
}

// Refinement warp SAD (middlebury models): d [N,1,H,W] fp32 (stride hn per image), scaled by `scale`;
// SAD at x - (d'), x - (d'+1), x - (d'-1). out [N,3,H,W] fp32.
template <typename T>
__device__ void ref_sad_impl(const float* __restrict__ d, long hn, const T* __restrict__ fl, const T* __restrict__ fr,
                             int N, int H, int W, int C, float scale, float* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * H * W) return;
  int X = i % W, Y = (i / W) % H, n = i / (W * H);
  float ds = d[n * hn + (long)Y * W + X] * scale;
  float base = (float)X - ds;
  float fb = floorf(base);
  int p[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float q = fb - 1.f + j;
    q = q < 0.f ? 0.f : (q > (float)(W - 1) ? (float)(W - 1) : q);
    p[j] = (int)q;
  }
  const int offs[3] = {0, -1, 1};  // x - d, x - (d+1), x - (d-1)
  float wgt[3];
#pragma unroll
  for (int k = 0; k < 3; k++) {
    float f = fb + offs[k];
    float fc = f < 0.f ? 0.f : (f > (float)(W - 1) ? (float)(W - 1) : f);
    wgt[k] = base + offs[k] - fc;
  }
  const T* lp = fl + (size_t)i * C;
  const T* rrow = fr + ((size_t)n * H + Y) * W * C;
  float s[3] = {0.f, 0.f, 0.f};
  for (int c = 0; c < C; c++) {
    float l = ld(lp + c);
    float r[4];
#pragma unroll
    for (int j = 0; j < 4; j++) r[j] = ld(rrow + (size_t)p[j] * C + c);
#pragma unroll
    for (int k = 0; k < 3; k++) {
      int j0 = 1 + offs[k];  // index of floor position in p[]
      float a = r[j0], b = r[j0 + 1];
      s[k] += fabsf(l - (a + (b - a) * wgt[k]));
    }
  }
  long cs = (long)H * W;
  float* op = out + (long)n * 3 * cs + (long)Y * W + X;
  op[0] = s[0];
  op[cs] = s[1];
  op[2 * cs] = s[2];
}

extern "C" __global__ void init_cost_f16(const __half* lt, const __half* rt, int N, int h, int w, int Wr, int D, float* od, float* oc) {
  init_cost_impl<__half>(lt, rt, N, h, w, Wr, D, od, oc);
}
extern "C" __global__ void tile_sad_f16(const float* hyp, long long hn, long long hc, const __half* fl, const __half* fr, int N, int h, int w, int C,
                             float* out, long long on) {
  tile_sad_impl<__half>(hyp, hn, hc, fl, fr, N, h, w, C, out, on);
}
extern "C" __global__ void ref_sad_f16(const float* d, long long hn, const __half* fl, const __half* fr, int N, int H, int W, int C, float scale,
                            float* out) {
  ref_sad_impl<__half>(d, hn, fl, fr, N, H, W, C, scale, out);
}

// ---- Full-resolution feature head (fastforks): the image preparation, the first convolution and the finest
// transposed convolution in one pass each, without the float32 batch, cuDNN's layout transforms and the separate
// bias / LeakyReLU / crop / concatenation passes.
#define IMG_MAXC 32
#define KCHUNK 8
#define IMG_STRIDE(cin) (((cin) * 9 + 3) / 4 * 4)  // floats per filter in shared memory (float4 reads)
__device__ __forceinline__ float leaky02(float v) { return v > 0.f ? v : 0.2f * v; }

// Stores KCHUNK channels (`n` <= KCHUNK valid) at `dst` (channel stride 1): one 16-byte store when aligned and full.
__device__ __forceinline__ void store_chunk(__half* dst, const float* a, int n) {
  if (n == KCHUNK && (reinterpret_cast<unsigned long long>(dst) & 15) == 0) {
    __align__(16) __half2 v[KCHUNK / 2];
#pragma unroll
    for (int j = 0; j < KCHUNK / 2; ++j) v[j] = __floats2half2_rn(a[2 * j], a[2 * j + 1]);
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(v);
  } else {
    for (int j = 0; j < n; ++j) dst[j] = __float2half(a[j]);
  }
}

template <int CIN>
__device__ __forceinline__ void img_conv3_impl(const float* __restrict__ l, const float* __restrict__ r, int N, int h,
                                               int w, int H, int W, float padl, float padr, const float* ws,
                                               const float* bs, int K, __half* __restrict__ out,
                                               __half* __restrict__ out2, long long o2n, long long o2y,
                                               long long o2x) {
  const long long P = 2LL * N * H * W;
  for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x; p < P;
       p += (long long)gridDim.x * blockDim.x) {
    const int X = (int)(p % W), Y = (int)((p / W) % H);
    const int b = (int)(p / ((long long)H * W));
    const bool right = b >= N;
    const float* img = (right ? r : l) + (long long)(right ? b - N : b) * 3 * h * w;
    const float pad = right ? padr : padl;
    float v[IMG_STRIDE(CIN)];
#pragma unroll
    for (int i = CIN * 9; i < IMG_STRIDE(CIN); ++i) v[i] = 0.f;  // padding taps (zero weights)
#pragma unroll
    for (int t = 0; t < 9; ++t) {
      const int yy = Y + t / 3 - 1, xx = X + t % 3 - 1;
      float c0 = pad, c1 = pad, c2 = pad;
      const bool inside = yy >= 0 && yy < H && xx >= 0 && xx < W;
      if (inside && yy < h && xx < w) {
        const long long o = (long long)yy * w + xx;
        c0 = img[o], c1 = img[(long long)h * w + o], c2 = img[2LL * h * w + o];
      }
      if constexpr (CIN == 1) {
        v[t] = inside ? (0.299f * c0 + 0.587f * c1 + 0.114f * c2) * (2.f / 255.f) - 1.f : 0.f;
      } else {
        v[t] = inside ? c0 * (2.f / 255.f) - 1.f : 0.f;
        v[9 + t] = inside ? c1 * (2.f / 255.f) - 1.f : 0.f;
        v[18 + t] = inside ? c2 * (2.f / 255.f) - 1.f : 0.f;
      }
    }
    __half* o1 = out + p * K;
    __half* o2 = out2 ? out2 + (long long)b * o2n + (long long)Y * o2y + (long long)X * o2x : nullptr;
    for (int k0 = 0; k0 < K; k0 += KCHUNK) {
      const int n = min(KCHUNK, K - k0);
      float a[KCHUNK];
#pragma unroll
      for (int j = 0; j < KCHUNK; ++j) {
        const int k = k0 + j < K ? k0 + j : K - 1;
        const float4* wk = reinterpret_cast<const float4*>(ws + k * IMG_STRIDE(CIN));
        float s = bs[k];
#pragma unroll
        for (int i = 0; i < IMG_STRIDE(CIN) / 4; ++i) {
          const float4 q = wk[i];
          s += q.x * v[4 * i] + q.y * v[4 * i + 1] + q.z * v[4 * i + 2] + q.w * v[4 * i + 3];
        }
        a[j] = leaky02(s);
      }
      store_chunk(o1 + k0, a, n);
      if (o2) store_chunk(o2 + k0, a, n);
    }
  }
}

// First feature convolution (3x3, zero padding 1, stride 1) of both images, LeakyReLU(0.2). l, r: fp32 [N, 3, h, w]
// (dense, 0..255); the network input is [2N (left, then right), cin, H, W]: each image bottom / right padded to H x W
// with its constant (padl / padr, 0..255), gray = BT.601 luma (cin 1) or RGB (cin 3), scaled to [-1, 1].
// wgt fp32 [K, cin, 3, 3], bias fp32 [K], K <= IMG_MAXC. out: fp16 [2N, H, W, K] (NHWC, dense); out2 (or null): a
// second destination at element strides (o2n, o2y, o2x), channel stride 1 (a channel slice of a concatenation).
// The input taps stay in registers, the weights in shared memory (float4 reads); channels are written 8 at a time.
extern "C" __global__ void img_conv3(const float* __restrict__ l, const float* __restrict__ r, int N, int h, int w,
                                     int H, int W, float padl, float padr, int cin, const float* __restrict__ wgt,
                                     const float* __restrict__ bias, int K, __half* __restrict__ out,
                                     __half* __restrict__ out2, long long o2n, long long o2y, long long o2x) {
  __shared__ __align__(16) float ws[IMG_MAXC * IMG_STRIDE(3)];
  __shared__ float bs[IMG_MAXC];
  const int st = IMG_STRIDE(cin);
  for (int i = threadIdx.x; i < K * st; i += blockDim.x) ws[i] = i % st < cin * 9 ? wgt[i / st * cin * 9 + i % st] : 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) bs[i] = bias[i];
  __syncthreads();
  if (cin == 1) {
    img_conv3_impl<1>(l, r, N, h, w, H, W, padl, padr, ws, bs, K, out, out2, o2n, o2y, o2x);
  } else {
    img_conv3_impl<3>(l, r, N, h, w, H, W, padl, padr, ws, bs, K, out, out2, o2n, o2y, o2x);
  }
}

// 2x2 stride-2 transposed convolution, LeakyReLU(0.2), cropped to Ho x Wo (<= 2h x 2w), and the skip connection
// next to it: out[:, :K] = deconv(x), out[:, K:K + Cs] = skip. x: fp16 [B, h, w, C] (NHWC, dense); wgt fp16
// [2, 2, C, K] (ky, kx, c, k); bias fp32 [K]; C, K multiples of 8. out: fp16 [B, K + Cs, Ho, Wo]
// at element strides (osn, osy, osx), channel stride 1 (a channels-last concatenation buffer); skip (or null): fp16
// [B, Cs, Ho, Wo] at strides (ssn, ssy, ssx), channel stride 1, Cs a multiple of 8.
//   out[b, k, Y, X] = leaky(bias[k] + sum_c x[b, Y / 2, X / 2, c] * wgt[Y % 2, X % 2, c, k])
// One thread per input pixel and chunk of 8 output channels (its 2 x 2 output pixels); the weights sit in shared
// memory. Chunks K / 8 .. K / 8 + Cs / 8 - 1 copy the skip instead.
extern "C" __global__ void deconv2x2(const __half* __restrict__ x, int B, int h, int w, int C,
                                     const __half* __restrict__ wgt, const float* __restrict__ bias, int K,
                                     __half* __restrict__ out, long long osn, long long osy, long long osx, int Ho,
                                     int Wo, const __half* __restrict__ skip, int Cs, long long ssn, long long ssy,
                                     long long ssx) {
  extern __shared__ __align__(16) __half ws[];  // 4 C K halves (dynamic shared memory)
  for (int i = threadIdx.x; i < 4 * C * K; i += blockDim.x) ws[i] = wgt[i];
  __syncthreads();
  const int nk = K / KCHUNK, nc = nk + (skip ? Cs / KCHUNK : 0);
  const int h2 = (Ho + 1) / 2, w2 = (Wo + 1) / 2;
  const long long T = (long long)B * h2 * w2 * nc;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < T;
       i += (long long)gridDim.x * blockDim.x) {
    const int ch = (int)(i % nc);
    const long long p = i / nc;
    const int x0 = (int)(p % w2), y0 = (int)((p / w2) % h2);
    const int b = (int)(p / ((long long)h2 * w2));
    if (ch >= nk) {  // skip connection: 8 channels of the 2 x 2 pixels
      const int c0 = (ch - nk) * KCHUNK;
      for (int t = 0; t < 4; ++t) {
        const int Y = 2 * y0 + t / 2, X = 2 * x0 + t % 2;
        if (Y >= Ho || X >= Wo) continue;
        const uint4 v = *reinterpret_cast<const uint4*>(skip + b * ssn + Y * ssy + X * ssx + c0);
        *reinterpret_cast<uint4*>(out + b * osn + Y * osy + X * osx + K + c0) = v;
      }
      continue;
    }
    const int k0 = ch * KCHUNK;
    float a[4][KCHUNK];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
      for (int j = 0; j < KCHUNK; ++j) a[t][j] = bias[k0 + j];
    const __half* xp = x + (((long long)b * h + y0) * w + x0) * C;
    for (int c8 = 0; c8 < C; c8 += 8) {
      const uint4 xv4 = *reinterpret_cast<const uint4*>(xp + c8);
      const __half* xh = reinterpret_cast<const __half*>(&xv4);
#pragma unroll
      for (int cc = 0; cc < 8; ++cc) {
        const float xv = __half2float(xh[cc]);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const uint4 wv4 = *reinterpret_cast<const uint4*>(ws + ((long long)t * C + c8 + cc) * K + k0);
          const __half2* wh = reinterpret_cast<const __half2*>(&wv4);
#pragma unroll
          for (int j = 0; j < KCHUNK / 2; ++j) {
            const float2 wf = __half22float2(wh[j]);
            a[t][2 * j] += xv * wf.x, a[t][2 * j + 1] += xv * wf.y;
          }
        }
      }
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int Y = 2 * y0 + t / 2, X = 2 * x0 + t % 2;
      if (Y >= Ho || X >= Wo) continue;
#pragma unroll
      for (int j = 0; j < KCHUNK; ++j) a[t][j] = leaky02(a[t][j]);
      store_chunk(out + b * osn + Y * osy + X * osx + k0, a[t], KCHUNK);
    }
  }
}

// Concatenation of up to four parts along the channels into out: fp16 [N, Ct, H, W] channels-last (dense), one
// thread per pixel writing 8 channels at a time. Part j: pointer (null: absent), f32 (1: float32, 0: float16), channel
// count and element strides (n, c, y, x) - any layout.
#define CAT_PARTS 4
struct CatPart {
  const void* p;
  int f32, c;
  long long sn, sc, sy, sx;
};
extern "C" __global__ void cat_cl(__half* __restrict__ out, int N, int H, int W, int Ct, const void* p0, int f0,
                                  int c0, long long s0n, long long s0c, long long s0y, long long s0x, const void* p1,
                                  int f1, int c1, long long s1n, long long s1c, long long s1y, long long s1x,
                                  const void* p2, int f2, int c2, long long s2n, long long s2c, long long s2y,
                                  long long s2x, const void* p3, int f3, int c3, long long s3n, long long s3c,
                                  long long s3y, long long s3x) {
  const CatPart q[CAT_PARTS] = {{p0, f0, c0, s0n, s0c, s0y, s0x},
                                {p1, f1, c1, s1n, s1c, s1y, s1x},
                                {p2, f2, c2, s2n, s2c, s2y, s2x},
                                {p3, f3, c3, s3n, s3c, s3y, s3x}};
  const long long P = (long long)N * H * W;
  for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x; p < P;
       p += (long long)gridDim.x * blockDim.x) {
    const int X = (int)(p % W), Y = (int)((p / W) % H);
    const int b = (int)(p / ((long long)H * W));
    int j = 0, c0j = 0;  // current part and its first output channel
    for (int cb = 0; cb < Ct; cb += KCHUNK) {
      float a[KCHUNK];
#pragma unroll
      for (int t = 0; t < KCHUNK; ++t) {
        const int c = cb + t;
        a[t] = 0.f;
        if (c >= Ct) continue;
        while (c - c0j >= q[j].c) c0j += q[j++].c;
        const long long o = b * q[j].sn + (c - c0j) * q[j].sc + Y * q[j].sy + X * q[j].sx;
        a[t] = q[j].f32 ? static_cast<const float*>(q[j].p)[o] : __half2float(static_cast<const __half*>(q[j].p)[o]);
      }
      store_chunk(out + p * Ct + cb, a, min(KCHUNK, Ct - cb));
    }
  }
}
