#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int channels = 16;
constexpr int packed_channels = 2 * channels;

__device__ __forceinline__ float read_latent(const void* data, long long index, int dtype) {
    if (dtype == 0) return __half2float(static_cast<const half*>(data)[index]);
    if (dtype == 1) return static_cast<const float*>(data)[index];
    return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
}

}  // namespace

// One launch per corner keeps the upstream order between corners. Within a corner,
// atomics can still accumulate colliding sources in a different order, as in scatter_add.
extern "C" __global__ void splat_linear_slice_c16(
    const void* __restrict__ latent, const float* __restrict__ flow,
    const float* __restrict__ metric, float* __restrict__ numerator,
    float* __restrict__ denominator, int height, int width, long long pixels,
    long long ls_b, long long ls_c, long long ls_h, long long ls_w,
    long long fs_b, long long fs_c, long long fs_h, long long fs_w,
    long long ms_b, long long ms_h, long long ms_w, int dtype, int corner) {
    const int lane = threadIdx.x & (channels - 1);
    const unsigned int peers = 0xffffu << (threadIdx.x & channels);
    const int plane = height * width;
    const long long first = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / channels;
    const long long step = static_cast<long long>(gridDim.x) * (blockDim.x / channels);
    for (long long pixel = first; pixel < pixels; pixel += step) {
        long long destination = -1, latent_offset = 0;
        float weight = 0.0f, m = 0.0f;
        // One half-warp per source pixel. Lane zero finds the destination and adds the metric.
        if (lane == 0) {
            const long long batch = pixel / plane;
            const int image_pixel = static_cast<int>(pixel - batch * plane);
            const int y = image_pixel / width;
            const int x = image_pixel % width;
            const long long flow_offset = batch * fs_b + y * fs_h + x * fs_w;
            const float fx = __fadd_rn(static_cast<float>(x), flow[flow_offset]);
            const float fy = __fadd_rn(static_cast<float>(y), flow[flow_offset + fs_c]);
            // Nonfinite flows miss every target. At -1, preserve a valid zero-weight neighbor
            // because a nonfinite latent or metric times its zero weight is still nonfinite.
            if (fx >= -1.0f && fx < static_cast<float>(width) &&
                fy >= -1.0f && fy < static_cast<float>(height)) {
                const int x0 = __float2int_rd(fx), y0 = __float2int_rd(fy);
                const int tx = x0 + (corner & 1), ty = y0 + (corner >> 1);
                if (tx >= 0 && tx < width && ty >= 0 && ty < height) {
                    destination = batch * plane + ty * width + tx;
                    const float wx = corner & 1 ? __fsub_rn(fx, x0) : __fsub_rn(x0 + 1, fx);
                    const float wy = corner & 2 ? __fsub_rn(fy, y0) : __fsub_rn(y0 + 1, fy);
                    weight = __fmul_rn(wx, wy);
                    latent_offset = batch * ls_b + y * ls_h + x * ls_w;
                    m = metric[batch * ms_b + y * ms_h + x * ms_w];
                }
            }
        }
        destination = __shfl_sync(peers, destination, 0, channels);
        if (destination < 0) continue;
        latent_offset = __shfl_sync(peers, latent_offset, 0, channels);
        m = __shfl_sync(peers, m, 0, channels);
        weight = __shfl_sync(peers, weight, 0, channels);
        const float weighted = __fmul_rn(read_latent(latent, latent_offset + lane * ls_c, dtype), m);
        atomicAdd(numerator + destination * packed_channels + lane, __fmul_rn(weighted, weight));
        if (lane == 0) atomicAdd(denominator + destination, __fmul_rn(m, weight));
    }
}

extern "C" __global__ void splat_normalize_slice_c16(
    float* __restrict__ numerator, const float* __restrict__ denominator, long long elements) {
    const long long first = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    const long long step = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = first; i < elements; i += step) {
        const long long pixel = i / channels;
        const long long offset = pixel * packed_channels + i % channels;
        const float norm = denominator[pixel];
        numerator[offset] = __fdiv_rn(numerator[offset], norm == 0.0f ? 1.0f : norm);
    }
}
