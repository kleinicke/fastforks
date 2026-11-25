#include <stdint.h>
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
