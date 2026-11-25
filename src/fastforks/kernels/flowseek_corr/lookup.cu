#include <cuda_fp16.h>
#include <stdint.h>

// A block shares the 10x10 integer-neighbour correlation window for one
// query and pyramid level. Each warp computes one 384-channel dot product
// at a time, loading adjacent half4 vectors from the original feature maps.
extern "C" __global__ void lookup(const half* left, const half* l0, const half* l1, const half* l2, const half* l3,
        const float* coords, half* out, int height, int width, float scale) {
    constexpr int channels = 384;
    __shared__ float dots[100];
    int p = blockIdx.x, batch = blockIdx.y, level = blockIdx.z;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int hw = height * width;
    int h = height >> level, w = width >> level;
    int64_t query = int64_t(batch) * hw + p;
    const float* xy = coords + int64_t(batch) * 2 * hw + p;
    float cx = xy[0] / (1 << level), cy = xy[hw] / (1 << level);
    int x0 = __float2int_rd(cx), y0 = __float2int_rd(cy);
    float fx = cx - x0, fy = cy - y0;
    const half* levels[] = {l0,l1,l2,l3};
    const uint2* src = reinterpret_cast<const uint2*>(left + query * channels) + lane;
    const half* other = levels[level] + int64_t(batch) * h * w * channels;
    float features[12];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        uint2 vals = src[32*g];
        const half* v = reinterpret_cast<const half*>(&vals);
        #pragma unroll
        for (int k = 0; k < 4; ++k) features[4*g+k] = __half2float(v[k]) * scale;
    }
    for (int j = warp; j < 100; j += blockDim.x / 32) {
        int x = x0 - 4 + j % 10, y = y0 - 4 + j / 10;
        float dot = 0;
        if (unsigned(x) < unsigned(w) && unsigned(y) < unsigned(h)) {
            const uint2* target = reinterpret_cast<const uint2*>(other + (int64_t(y)*w+x)*channels) + lane;
            #pragma unroll
            for (int g = 0; g < 3; ++g) {
                uint2 vals = target[32*g];
                const half* v = reinterpret_cast<const half*>(&vals);
                #pragma unroll
                for (int k = 0; k < 4; ++k) dot = fmaf(features[4*g+k],__half2float(v[k]),dot);
            }
        }
        #pragma unroll
        for (int stride=16; stride>0; stride/=2) dot += __shfl_down_sync(0xffffffff,dot,stride);
        if (!lane) dots[j] = dot;
    }
    __syncthreads();
    if (threadIdx.x < 81) {
        int j = threadIdx.x;
        // Upstream flattens (x offset, y offset) in that order.
        const float* d = dots + (j % 9) * 10 + j / 9;
        float value = (1-fy)*((1-fx)*d[0]+fx*d[1]) + fy*((1-fx)*d[10]+fx*d[11]);
        out[query*324 + level*81 + j] = __float2half_rn(value);
    }
}
