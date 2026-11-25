#include <cuda_fp16.h>
#include <stdint.h>

// Features are dense NCHW row strips. Write correlation, left/right features,
// and (for refinement only) sample coordinate directly in the 3D CNN's NCDHW layout.
template <typename Out, bool Refinement>
__device__ void volume(const half* gl, const half* gr, const half* cl, const half* cr,
        const float* samples, Out* out, int groups, int cg, int cc, int d, int h, int w, int64_t count) {
    const int channels = groups + 2 * cc + int(Refinement);
    const int64_t hw = int64_t(h) * w, dhw = d * hw;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        int64_t plane = i / dhw, pixel = i % hw;
        int ch = plane % channels, batch = plane / channels, x = pixel % w;
        float shift = Refinement ? samples[batch * dhw + i % dhw] : float(i / hw % d);
        float xr = float(x) - shift;
        bool valid = xr >= 0.f && xr <= w - 1;
        float value = 0;
        if (Refinement && ch == channels - 1) {
            value = shift;
        } else if (ch >= groups && ch < groups + cc && (Refinement || valid)) {
            value = __half2float(cl[(int64_t(batch) * cc + ch - groups) * hw + pixel]);
        } else if (valid) {
            int64_t right = pixel - x + int(xr);
            if (ch >= groups) {
                value = __half2float(cr[(int64_t(batch) * cc + ch - groups - cc) * hw + right]);
            } else {
                int width = cg / groups;
                int64_t base = (int64_t(batch) * cg + ch * width) * hw;
                for (int j = 0; j < width; ++j, base += hw)
                    value += __half2float(gl[base + pixel]) * __half2float(gr[base + right]);
                value /= float(width);
            }
        }
        out[i] = Out(value);
    }
}

#define ENTRY(name, T, STAGE) \
extern "C" __global__ void name(const half* gl, const half* gr, const half* cl, const half* cr, \
        const float* samples, T* out, int groups, int cg, int cc, int d, int h, int w, int64_t count) { \
    volume<T, STAGE>(gl, gr, cl, cr, samples, out, groups, cg, cc, d, h, w, count); \
}
ENTRY(coarse_half, half, false)
ENTRY(coarse_float, float, false)
ENTRY(refinement, half, true)

// Cas-GwcNet samples fractional disparities with a zero-padded bilinear grid.
// One warp per output pixel, with adjacent lanes handling adjacent channels.
__device__ __forceinline__ float warp_val(const half* image, int c, int width, int height, int cstride,
                                          int x, int y, float dx, float dy) {
    float a = 0.f, b = 0.f, e = 0.f, f = 0.f;
    if (y >= 0 && y < height) {
        if (x >= 0 && x < width) a = __half2float(image[(int64_t(y) * width + x) * cstride + c]);
        if (x + 1 >= 0 && x + 1 < width) b = __half2float(image[(int64_t(y) * width + x + 1) * cstride + c]);
    }
    if (y + 1 >= 0 && y + 1 < height) {
        if (x >= 0 && x < width) e = __half2float(image[(int64_t(y + 1) * width + x) * cstride + c]);
        if (x + 1 >= 0 && x + 1 < width) f = __half2float(image[(int64_t(y + 1) * width + x + 1) * cstride + c]);
    }
    return a * ((1.f - dx) * (1.f - dy)) + b * (dx * (1.f - dy)) +
           e * ((1.f - dx) * dy) + f * (dx * dy);
}

extern "C" __global__ void cascade_half(const half* left_gwc, const half* right_gwc, const half* left,
        const half* right, const float* samples, half* out, int groups, int cg, int cc, int d, int h, int w,
        int64_t sb, int64_t sd, int64_t sy, int64_t sx, int64_t count, float inv_y, float inv_x,
        float scale) {
    const int channels = groups + 2 * cc;
    const int64_t hw = int64_t(h) * w, dhw = d * hw;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    const int channels_per_group = groups ? cg / groups : 0;
    for (int64_t i = int64_t(blockIdx.x) * warps + warp; i < count; i += int64_t(gridDim.x) * warps) {
        const int64_t b = i / dhw, voxel = i % dhw, pixel = voxel % hw;
        const int x = pixel % w, y = pixel / w, z = voxel / hw;
        const float shift = samples[b * sb + z * sd + y * sy + x * sx];
        const float gx = __fsub_rn(__fmul_rn(__fsub_rn(float(x), shift), inv_x), 1.f);
        const float gy = __fsub_rn(__fmul_rn(float(y), inv_y), 1.f);
        const float xr = ((gx + 1.f) * .5f) * (w - 1), yr = ((gy + 1.f) * .5f) * (h - 1);
        const bool sample = xr > -1.f && xr < w && yr > -1.f && yr < h;
        const int xi = sample ? __float2int_rd(xr) : -2, yi = sample ? __float2int_rd(yr) : -2;
        const float dx = sample ? xr - xi : 0.f, dy = sample ? yr - yi : 0.f;
        const bool keep = !groups || float(x) >= shift;
        const half* l = left + (b * hw + pixel) * cc;
        const half* r = right + b * hw * cc;
        const half* lg = cg ? left_gwc + (b * hw + pixel) * cg : nullptr;
        const half* rg = cg ? right_gwc + b * hw * cg : nullptr;
        for (int ch = lane; ch < channels; ch += 32) {
            float val = 0.f;
            if (ch >= groups + cc) {
                if (sample) val = warp_val(r, ch - groups - cc, w, h, cc, xi, yi, dx, dy);
            } else if (keep) {
                if (ch >= groups) {
                    val = __half2float(l[ch - groups]);
                } else if (sample) {
                    const int first = ch * channels_per_group;
                    float sum = 0.f;
                    for (int c = first; c < first + channels_per_group; ++c) {
                        const float weighted = __fmul_rn(__half2float(lg[c]), warp_val(rg, c, w, h, cg, xi, yi, dx, dy));
                        sum += weighted;
                    }
                    val = sum / channels_per_group;
                }
            }
            out[i * channels + ch] = __float2half_rn(val * scale);
        }
    }
}
