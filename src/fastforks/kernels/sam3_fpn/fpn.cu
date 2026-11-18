#include <cuda_fp16.h>
#include <stdint.h>

// The common channels-last layout lets each thread copy eight channels. The
// skip and coarse maps can then both be read with coalesced 128-bit loads.
extern "C" __global__ void nearest_add8(const uint4* coarse, const uint4* skip, uint4* out,
                                        int h, int w, int packs, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(blockDim.x) * gridDim.x) {
        int64_t p = i / packs, row = p / (2*w);
        int64_t src = ((row / (2*h) * h + row / 2 % h) * w + p / 2 % w) * packs + i % packs;
        uint4 l = coarse[src], r = skip[i], v;
        half2* dst = reinterpret_cast<half2*>(&v);
        #pragma unroll
        for (int k=0; k<4; ++k)
            dst[k] = __hadd2(reinterpret_cast<half2*>(&l)[k], reinterpret_cast<half2*>(&r)[k]);
        out[i] = v;
    }
}

// NCHW and other strided views: the output uses the coarse map's preferred
// dense format, just as interpolate followed by add does in PyTorch.
extern "C" __global__ void nearest_add(const half* coarse, const half* skip, half* out,
        int h, int w, int c, int nhwc, int64_t lb, int64_t lc, int64_t ly, int64_t lx,
        int64_t rb, int64_t rc, int64_t ry, int64_t rx, int64_t count) {
    int64_t hw = int64_t(h) * w * 4;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(blockDim.x) * gridDim.x) {
        int64_t p = (nhwc ? i/c : i) % hw, b = i / (c*hw);
        int ch = nhwc ? i%c : (i/hw)%c;
        int y = p/(2*w), x = p%(2*w);
        out[i] = __hadd(coarse[b*lb + ch*lc + (y/2)*ly + (x/2)*lx],
                         skip[b*rb + ch*rc + y*ry + x*rx]);
    }
}

// The convolution result is owned by the decoder. Apply its per-image,
// per-channel GroupNorm affine transform and ReLU directly in that storage.
extern "C" __global__ void affine_relu(half* x, const half* scale, const half* shift,
                                       int c, int hw, int nhwc, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(blockDim.x) * gridDim.x) {
        int64_t b = i / (int64_t(hw)*c);
        int ch = nhwc ? i%c : (i/hw)%c;
        int64_t j = b*c + ch;
        float v = fmaf(__half2float(x[i]), __half2float(scale[j]), __half2float(shift[j]));
        x[i] = __float2half_rn(v < 0.f ? 0.f : v);
    }
}
