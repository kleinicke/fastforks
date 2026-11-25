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
