#include <cuda_fp16.h>
#include <stdint.h>

// A warp computes one entry of the plane-sweep volume. Both view features
// are channel-last so the lanes read consecutive channels at each neighbor.
extern "C" __global__ void plane_sweep(const half* features, const half2* grid, float* volume,
        int views, int offset, int channels, int height, int width, int64_t samples) {
    int lane = threadIdx.x & 31;
    int64_t hw = int64_t(height) * width;
    for (int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
         i < samples; i += int64_t(gridDim.x) * (blockDim.x / 32)) {
        int view = i / (samples / views);
        float2 uv = __half22float2(grid[i]);
        float x = ((uv.x + 1.f) * .5f) * (width - 1);
        float y = ((uv.y + 1.f) * .5f) * (height - 1);
        float dot = 0;
        if (x > -1.f && x < width && y > -1.f && y < height) {
            int x0 = __float2int_rd(x), y0 = __float2int_rd(y);
            const half* left = features + (view * hw + i % hw) * channels;
            const half* right = features + ((view + offset) % views) * hw * channels;
            for (int c = lane; c < channels; c += 32) {
                float sample = 0;
                #pragma unroll
                for (int dy = 0; dy < 2; ++dy) {
                    #pragma unroll
                    for (int dx = 0; dx < 2; ++dx) {
                        int xx = x0 + dx, yy = y0 + dy;
                        if (unsigned(xx) < unsigned(width) && unsigned(yy) < unsigned(height)) {
                            float wx = dx ? x - x0 : x0 + 1 - x;
                            float wy = dy ? y - y0 : y0 + 1 - y;
                            sample += wx * wy * __half2float(right[(int64_t(yy)*width + xx)*channels + c]);
                        }
                    }
                }
                dot += __half2float(__hmul_rn(__float2half_rn(sample), left[c]));
            }
        }
        for (int stride = 16; stride; stride /= 2)
            dot += __shfl_down_sync(0xffffffffu, dot, stride);
        if (!lane) {
            if (offset == 1) volume[i] = dot;
            else volume[i] += dot;
        }
    }
}
