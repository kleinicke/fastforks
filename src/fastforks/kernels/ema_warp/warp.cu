#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ uint2 warp4(const half* a, const half* b, const half* c, const half* d,
                                       float wa, float wb, float wc, float wd) {
    uint2 va = *reinterpret_cast<const uint2*>(a);
    uint2 vb = *reinterpret_cast<const uint2*>(b);
    uint2 vc = *reinterpret_cast<const uint2*>(c);
    uint2 vd = *reinterpret_cast<const uint2*>(d);
    uint2 result;
    const half* x0 = reinterpret_cast<const half*>(&va);
    const half* x1 = reinterpret_cast<const half*>(&vb);
    const half* x2 = reinterpret_cast<const half*>(&vc);
    const half* x3 = reinterpret_cast<const half*>(&vd);
    half* out = reinterpret_cast<half*>(&result);
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        float v = wa * float(x0[lane]);
        v = fmaf(wb, float(x1[lane]), v);
        v = fmaf(wc, float(x2[lane]), v);
        out[lane] = __float2half_rn(fmaf(wd, float(x3[lane]), v));
    }
    return result;
}

// Reproduce the rounded normalized grid before unnormalizing it. Simply
// adding the flow to the integer coordinate diverges at large image widths.
__device__ __forceinline__ float coordinate(int index, int size, float flow) {
    if (size == 1) return 0.f;
    float step = 2.f / (size - 1);
    float base = index < size / 2 ? fmaf(step, index, -1.f) : fmaf(-step, size - index - 1, 1.f);
    float grid = __fadd_rn(base, __fmul_rn(flow, step));
    return fminf(fmaxf(((grid + 1.f) * .5f) * (size - 1), 0.f), float(size - 1));
}

// Both source frames have the same strides. The output can be two consecutive
// frames or two adjacent channel groups in one convolution input.
template <typename T, int Pack>
__device__ void warp_pair(const T* first, const T* second, const float* flow, T* out,
        int batch, int channels, int height, int width, int64_t sn, int64_t sc, int64_t sh, int64_t sw,
        int64_t fn, int64_t fc, int64_t fh, int64_t fw, int out_channels, int64_t right_offset) {
    // One block spans adjacent channels/pixels of a row. The row and batch
    // indices come from the grid; only the within-row index needs division.
    unsigned int channel_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = channel_x / unsigned(channels / Pack);
    if (x >= unsigned(width)) return;
    int c = (channel_x % unsigned(channels / Pack)) * Pack;
    for (int pair_n = blockIdx.z; pair_n < 2 * batch; pair_n += gridDim.z) {
        bool right = pair_n >= batch;
        int n = right ? pair_n - batch : pair_n;
        for (int y = blockIdx.y; y < height; y += gridDim.y) {
            T* row = out + (right ? right_offset : 0) + (int64_t(n) * height + y) * width * out_channels;
            const float* f = flow + n * fn + (right ? 2 * fc : 0) + y * fh + x * fw;
            float u = coordinate(x, width, f[0]), v = coordinate(y, height, f[fc]);
            int x0 = int(u), y0 = int(v), x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
            float l = (x0 + 1) - u, r = u - x0, t = (y0 + 1) - v, b = v - y0;
            const T* s = (right ? second : first) + n * sn + c * sc;
            T* dst = row + int64_t(x) * out_channels + c;
            if constexpr (Pack == 4) {
                *reinterpret_cast<uint2*>(dst) =
                    warp4(s + y0 * sh + x0 * sw, s + y0 * sh + x1 * sw, s + y1 * sh + x0 * sw,
                          s + y1 * sh + x1 * sw, l * t, r * t, l * b, r * b);
            } else {
                float value = (l * t) * float(s[y0 * sh + x0 * sw]);
                value = fmaf(r * t, float(s[y0 * sh + x1 * sw]), value);
                value = fmaf(l * b, float(s[y1 * sh + x0 * sw]), value);
                *dst = T(fmaf(r * b, float(s[y1 * sh + x1 * sw]), value));
            }
        }
    }
}

#define WARP_ARGS(T) const T* a, const T* b, const float* f, T* o, int n, int c, int h, int w, \
    int64_t sn, int64_t sc, int64_t sh, int64_t sw, int64_t fn, int64_t fc, int64_t fh, int64_t fw, \
    int oc, int64_t right_offset
extern "C" __global__ void warp_f32(WARP_ARGS(float)) {
    warp_pair<float, 1>(a, b, f, o, n, c, h, w, sn, sc, sh, sw, fn, fc, fh, fw, oc, right_offset);
}
extern "C" __global__ void warp_f16(WARP_ARGS(half)) {
    warp_pair<half, 1>(a, b, f, o, n, c, h, w, sn, sc, sh, sw, fn, fc, fh, fw, oc, right_offset);
}
extern "C" __global__ void warp_f16x4(WARP_ARGS(half)) {
    warp_pair<half, 4>(a, b, f, o, n, c, h, w, sn, sc, sh, sw, fn, fc, fh, fw, oc, right_offset);
}

template <typename T, int Pack>
__device__ void warp_write(const T* image, const float* flow, half* out, int n, int c, int h, int w,
                           int y0, int rows, int64_t sn, int64_t sc, int64_t sh, int64_t sw,
                           int64_t fn, int64_t fc, int64_t fh, int64_t fw,
                           int64_t on, int64_t oh, int64_t ow) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    int x = index / unsigned(c / Pack), channel = (index % unsigned(c / Pack)) * Pack;
    if (x >= w) return;
    for (int b = blockIdx.z; b < n; b += gridDim.z) {
        const T* src = image + b * sn + channel * sc;
        for (int row = blockIdx.y; row < rows; row += gridDim.y) {
            int y = y0 + row;
            const float* f = flow + b * fn + y * fh + x * fw;
            float u = coordinate(x, w, f[0]), v = coordinate(y, h, f[fc]);
            int xlo = int(u), ylo = int(v), xhi = min(xlo + 1, w - 1), yhi = min(ylo + 1, h - 1);
            float l = (xlo + 1) - u, r = u - xlo, t = (ylo + 1) - v, bot = v - ylo;
            half* dst = out + b * on + row * oh + x * ow + channel;
            if constexpr (Pack == 4) {
                *reinterpret_cast<uint2*>(dst) =
                    warp4(src + ylo * sh + xlo * sw, src + ylo * sh + xhi * sw, src + yhi * sh + xlo * sw,
                          src + yhi * sh + xhi * sw, l * t, r * t, l * bot, r * bot);
            } else {
                float value = (l * t) * float(src[ylo * sh + xlo * sw]);
                value = fmaf(r * t, float(src[ylo * sh + xhi * sw]), value);
                value = fmaf(l * bot, float(src[yhi * sh + xlo * sw]), value);
                *dst = __float2half_rn(fmaf(r * bot, float(src[yhi * sh + xhi * sw]), value));
            }
        }
    }
}

#define WRITE_ARGS(T) const T* im, const float* f, half* o, int n, int c, int h, int w, int y0, int rows, \
    int64_t sn, int64_t sc, int64_t sh, int64_t sw, int64_t fn, int64_t fc, int64_t fh, int64_t fw, \
    int64_t on, int64_t oh, int64_t ow
extern "C" __global__ void warp_write_f16(WRITE_ARGS(half)) {
    warp_write<half, 1>(im, f, o, n, c, h, w, y0, rows, sn, sc, sh, sw, fn, fc, fh, fw, on, oh, ow);
}
extern "C" __global__ void warp_write_f32(WRITE_ARGS(float)) {
    warp_write<float, 1>(im, f, o, n, c, h, w, y0, rows, sn, sc, sh, sw, fn, fc, fh, fw, on, oh, ow);
}
extern "C" __global__ void warp_write_f16x4(WRITE_ARGS(half)) {
    warp_write<half, 4>(im, f, o, n, c, h, w, y0, rows, sn, sc, sh, sw, fn, fc, fh, fw, on, oh, ow);
}
