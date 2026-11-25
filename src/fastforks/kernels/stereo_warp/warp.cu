#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float coordinate(float pixel, int size, float reciprocal) {
    const float normalized = __fadd_rn(__fmul_rn(2.f, __fmul_rn(pixel, reciprocal)), -1.f);
    return fmaf(normalized + 1.f, float(size), -1.f) * 0.5f;
}

__device__ __forceinline__ uint2 read4(
    const half* source, int y, int x, int height, int width, int64_t sy, int64_t sx) {
    if (unsigned(y) < unsigned(height) && unsigned(x) < unsigned(width)) {
        return *reinterpret_cast<const uint2*>(source + y * sy + x * sx);
    }
    return make_uint2(0, 0);
}

extern "C" __global__ void stereo_warp4(
    const half* source, const float* disp, half* out, int batch, int channels, int height, int width,
    int64_t sb, int64_t sy, int64_t sx, int64_t db, int64_t dy, int64_t dx,
    int64_t ob, int64_t oy, int64_t ox, float rh, float rw) {
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel = lane / (channels / 4);
    const int channel = (lane % (channels / 4)) * 4;
    if (pixel >= width) return;
    for (int n = blockIdx.z; n < batch; n += gridDim.z) {
        for (int row = blockIdx.y; row < height; row += gridDim.y) {
            const float d = disp[n * db + row * dy + pixel * dx];
            const float x = coordinate(__fsub_rn(float(pixel), d), width, rw);
            const float y = coordinate(float(row), height, rh);
            half* dest = out + n * ob + row * oy + pixel * ox + channel;
            if (!(x > -1.f && x < width && y > -1.f && y < height)) {
                *reinterpret_cast<uint2*>(dest) = make_uint2(0, 0);
                continue;
            }
            const int x0 = __float2int_rd(x), y0 = __float2int_rd(y);
            const float xl = (x0 + 1) - x, xr = x - x0;
            const float yt = (y0 + 1) - y, yb = y - y0;
            const half* src = source + n * sb + channel;
            const uint2 a = read4(src, y0, x0, height, width, sy, sx);
            const uint2 b = read4(src, y0, x0 + 1, height, width, sy, sx);
            const uint2 c = read4(src, y0 + 1, x0, height, width, sy, sx);
            const uint2 e = read4(src, y0 + 1, x0 + 1, height, width, sy, sx);
            const half* nw = reinterpret_cast<const half*>(&a);
            const half* ne = reinterpret_cast<const half*>(&b);
            const half* sw = reinterpret_cast<const half*>(&c);
            const half* se = reinterpret_cast<const half*>(&e);
            uint2 values;
            half* result = reinterpret_cast<half*>(&values);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float v = (xl * yt) * float(nw[i]);
                v = fmaf(xr * yt, float(ne[i]), v);
                v = fmaf(xl * yb, float(sw[i]), v);
                result[i] = __float2half_rn(fmaf(xr * yb, float(se[i]), v));
            }
            *reinterpret_cast<uint2*>(dest) = values;
        }
    }
}

