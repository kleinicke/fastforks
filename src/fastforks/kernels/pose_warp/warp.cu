#include <math.h>
#include <stdint.h>

// fixed() in shared.topdown rounds offset and step separately to OpenCV's
// ten-bit affine representation before selecting a five-bit interpolation bin.
__device__ float coordinate(float centre, float step, int extent, int output) {
    float origin = __fsub_rn(centre, __fmul_rn(step, extent * .5f));
    float q = rintf(origin * 1024.f) + rintf(__fmul_rn(step, float(output)) * 1024.f);
    return floorf((q + 16.f) / 32.f) / 32.f;
}

__device__ float sample(const float* image, int h, int w, int64_t sy, int64_t sx,
                        float y, float x, float border) {
    if (!(x > -1.f && x < w && y > -1.f && y < h)) return border;
    int iy = __float2int_rd(y), ix = __float2int_rd(x);
    float fy = y - iy, fx = x - ix, result = 0.f;
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            float v = border;
            if (unsigned(iy + dy) < unsigned(h) && unsigned(ix + dx) < unsigned(w))
                v = rintf(image[(iy + dy) * sy + (ix + dx) * sx] * 255.f);
            result += v * ((dy ? fy : 1.f - fy) * (dx ? fx : 1.f - fx));
        }
    }
    return floorf(result + .5f);
}

extern "C" __global__ void topdown(const float* image, const float* centres, const float* sizes,
        float* out, int h, int w, int oh, int ow, int64_t sc, int64_t sy, int64_t sx, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        int64_t plane = i / ow / oh, box = plane / 3;
        float step = __fdiv_rn(sizes[box * 2], float(ow));
        float x = coordinate(centres[2 * box], step, ow, i % ow);
        float y = coordinate(centres[2 * box + 1], step, oh, i / ow % oh);
        out[i] = sample(image + plane % 3 * sc, h, w, sy, sx, y, x, 0.f);
    }
}

extern "C" __global__ void letterbox(const float* image, const float* xy, float* out,
        int h, int w, int size, int64_t sc, int64_t sy, int64_t sx, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        float x = xy[i % size], y = xy[size + i / size % size];
        out[i] = sample(image + i / size / size * sc, h, w, sy, sx, y, x, 114.f);
    }
}
