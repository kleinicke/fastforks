#include <stdint.h>
#include <cuda_fp16.h>
#include <vector_types.h>

// One permutation maps the flattened windows to raster tokens. Use its inverse
// for the return path. All copies preserve the feature bits exactly.
template <typename T>
__device__ void permute(const T* src, T* dst, int h, int w, int c, int ws, int shift, int reverse, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        int64_t token = i / c, tile = token / (ws * ws);
        int wy = token / ws % ws, wx = token % ws;
        int col = (tile % (w/ws)) * ws + wx + shift;
        int row = (tile / (w/ws) % (h/ws)) * ws + wy + shift;
        if (col >= w) col -= w;
        if (row >= h) row -= h;
        int64_t batch = token / (int64_t(w) * h);
        int64_t raster = ((batch * h + row) * w + col) * c + i % c;
        if (reverse) dst[raster] = src[i];
        else dst[i] = src[raster];
    }
}
#define ENTRY(name, T) \
extern "C" __global__ void name(const T* src, T* dst, int h, int w, int c, int ws, int shift, int reverse, int64_t count) { \
    permute(src, dst, h, w, c, ws, shift, reverse, count); \
}
ENTRY(window1, uint16_t)
ENTRY(window2, uint32_t)
ENTRY(window8, uint4)

template <int P>
__device__ void reverse_add(const __half2* src, const __half2* shortcut, const __half2* conv,
                            __half2* dst, int h, int w, int channels, int conv_channels, int ws, int shift,
                            float scale, int count) {
    const int nc = w / ws, nr = h / ws, area = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        int token = i / (channels / P), lane = (i % (channels / P)) * P;
        int tile = token / (ws * ws), col = tile % nc * ws + token % ws + shift;
        int row = (tile / nc % nr) * ws + token / ws % ws + shift;
        col = col >= w ? col - w : col;
        row = row >= h ? row - h : row;
        int raster = (token / area * h + row) * w + col;
        #pragma unroll
        for (int p = 0; p < P; ++p) {
            __half2 residual = __hadd2(shortcut[raster * channels + lane + p], src[i * P + p]);
            if (conv) {
                const float2 value = __half22float2(conv[raster * conv_channels + lane + p]);
                const __half2 scaled = __floats2half2_rn(value.x * scale, value.y * scale);
                residual = __hadd2(residual, scaled);
            }
            dst[raster * channels + lane + p] = residual;
        }
    }
}

#define RESIDUAL(name, P) \
extern "C" __global__ void name(const __half2* src, const __half2* shortcut, const __half2* conv, __half2* dst, \
                                int h, int w, int channels, int conv_channels, int ws, int shift, \
                                float scale, int count) { \
    reverse_add<P>(src, shortcut, conv, dst, h, w, channels, conv_channels, ws, shift, scale, count); \
}
RESIDUAL(reverse_add2, 1)
RESIDUAL(reverse_add4, 2)

extern "C" __global__ void stream_window2(
    const __half2* src, const __half2* skip, __half2* dst,
    int h, int w, int channels, int win, int shift, int y0, int x0, int region_cols, int first, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        int64_t token = i / channels;
        int tile = first + token / (win * win);
        int y = (y0 + tile / region_cols) * win + (token / win) % win + shift;
        int x = (x0 + tile % region_cols) * win + token % win + shift;
        y = y >= h ? y - h : y;
        x = x >= w ? x - w : x;
        int64_t raster = (int64_t(y) * w + x) * channels + i % channels;
        if (skip) dst[raster] = __hadd2(skip[raster], src[i]);
        else dst[i] = src[raster];
    }
}
