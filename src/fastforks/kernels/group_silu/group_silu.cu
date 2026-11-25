#include <cuda_fp16.h>
#include <stdint.h>

constexpr int TILE = 2048, THREADS = 256;
__device__ float2 reduce_pair(float2 p, float2* partial) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int step=16; step; step/=2) {
        p.x += __shfl_down_sync(0xffffffff, p.x, step);
        p.y += __shfl_down_sync(0xffffffff, p.y, step);
    }
    if (!lane) partial[warp] = p;
    __syncthreads();
    if (!warp) {
        p = lane < THREADS/32 ? partial[lane] : make_float2(0.f, 0.f);
        for (int step=THREADS/64; step; step/=2) {
            p.x += __shfl_down_sync(0xffffffff, p.x, step);
            p.y += __shfl_down_sync(0xffffffff, p.y, step);
        }
    }
    return p; // thread zero holds the sum
}

// One spatial/channel tile within a group. Center before squaring to preserve
// variance in nearly constant patches; store the mean and centered sum of squares.
extern "C" __global__ void tile_moments(const half* x, float2* moments, int channels, int width,
                                        int64_t spatial, int tiles) {
    __shared__ float2 partial[THREADS/32];
    int64_t count = spatial * width, start = (int64_t)blockIdx.x * TILE;
    int len = min((int64_t)TILE, count - start);
    int group = blockIdx.y % (channels / width), batch = blockIdx.y / (channels / width);
    const half* base = x + (int64_t)batch * spatial * channels + group * width;
    float anchor = __half2float(base[(start / width) * channels + start % width]);
    float2 p = make_float2(0, 0);
    for (int k = threadIdx.x; k < len; k += blockDim.x) {
        int64_t i = start + k;
        float t = __half2float(base[(i / width) * channels + i % width]) - anchor;
        p.x += t;
        p.y = fmaf(t, t, p.y);
    }
    p = reduce_pair(p, partial);
    if (!threadIdx.x) {
        float mean = p.x / len;
        moments[(int64_t)blockIdx.y * tiles + blockIdx.x] =
            make_float2(anchor + mean, fmaxf(0.f, fmaf(-mean, p.x, p.y)));
    }
}

extern "C" __global__ void group_moments(const float2* moments, float2* stats, int tiles, int64_t count, float epsilon) {
    __shared__ float2 partial[THREADS/32];
    moments += (int64_t)blockIdx.x * tiles;
    float anchor = moments[0].x;
    float2 p = make_float2(0, 0);
    for (int i = threadIdx.x; i < tiles; i += blockDim.x) {
        int len = min((int64_t)TILE, count - (int64_t)i * TILE);
        float2 tile = moments[i];
        float delta = tile.x - anchor;
        p.x += delta * len;
        p.y += fmaf(delta, delta * len, tile.y);
    }
    p = reduce_pair(p, partial);
    if (!threadIdx.x) {
        float mean = p.x / count;
        stats[blockIdx.x] = make_float2(anchor + mean, rsqrtf(fmaxf(0.f, fmaf(-mean, mean, p.y / count)) + epsilon));
    }
}

extern "C" __global__ void normalize_silu(const half* x, half* out, const float2* stats, const half* weight,
                                          const half* bias, int channels, int width, int64_t spatial, int64_t count) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (int64_t)gridDim.x * blockDim.x) {
        int c = i % channels;
        int64_t batch = i / (spatial * channels);
        float2 group = stats[batch * (channels / width) + c / width];
        float y = (__half2float(x[i]) - group.x) * group.y;
        y = fmaf(y, __half2float(weight[c]), __half2float(bias[c]));
        out[i] = __float2half_rn(y / (1.f + expf(-y)));
    }
}

// Apply group statistics already folded into FP16 scale and shift. The
// normalized value is rounded exactly once before the SiLU evaluation.
extern "C" __global__ void affine_silu_h(const half* x, const half* scale, const half* shift, half* out,
                                       int channels, int64_t spatial, int64_t batch_stride, int64_t count) {
    const int packs = channels / 8;
    const int64_t sample = spatial * packs;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t batch = i / sample, local = i % sample, a = batch * packs + local % packs;
        uint4 xv = *reinterpret_cast<const uint4*>(x + batch * batch_stride + local * 8);
        uint4 sv = reinterpret_cast<const uint4*>(scale)[a];
        uint4 bv = reinterpret_cast<const uint4*>(shift)[a];
        uint4 output;
        const half* values = reinterpret_cast<const half*>(&xv);
        const half* weights = reinterpret_cast<const half*>(&sv);
        const half* biases = reinterpret_cast<const half*>(&bv);
        half* result = reinterpret_cast<half*>(&output);
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            float f = fmaf(__half2float(values[c]), __half2float(weights[c]), __half2float(biases[c]));
            float r = __half2float(__float2half_rn(f));
            result[c] = __float2half_rn(r / (1.f + expf(-r)));
        }
        reinterpret_cast<uint4*>(out)[i] = output;
    }
}
