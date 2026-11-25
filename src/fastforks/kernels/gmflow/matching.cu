#include <cuda_fp16.h>
#include <math_constants.h>
#include <stdint.h>

__device__ __forceinline__ float warp_sum(float x) {
    for (int d = 16; d > 0; d /= 2) {
        x += __shfl_down_sync(0xffffffff, x, d);
    }
    return x;
}

// One row of global matching. The second score pass reduces the two
// expected values directly and leaves no softmax tensor in device memory.
extern "C" __global__ void expectation2(
    const float* scores, const float2* values, float2* out, int rows, int columns, int out_stride
) {
    int row = blockIdx.x, batch = blockIdx.y;
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const float* score = scores + (int64_t(batch) * rows + row) * columns;
    const float2* value = values + int64_t(batch) * columns;
    __shared__ float scratch[3][32];
    float peak = -CUDART_INF_F;
    for (int j = tid; j < columns; j += blockDim.x) {
        peak = fmaxf(peak, score[j]);
    }
    for (int d = 16; d > 0; d /= 2) {
        peak = fmaxf(peak, __shfl_down_sync(0xffffffff, peak, d));
    }
    if (!lane) scratch[0][warp] = peak;
    __syncthreads();
    peak = -CUDART_INF_F;
    for (int w = 0; w < blockDim.x / 32; ++w) {
        peak = fmaxf(peak, scratch[0][w]);
    }
    __syncthreads();
    float sum = 0.0f, sx = 0.0f, sy = 0.0f;
    for (int j = tid; j < columns; j += blockDim.x) {
        float p = __expf(score[j] - peak);
        float2 v = value[j];
        sum += p;
        sx = fmaf(p, v.x, sx);
        sy = fmaf(p, v.y, sy);
    }
    sum = warp_sum(sum);
    sx = warp_sum(sx);
    sy = warp_sum(sy);
    if (!lane) {
        scratch[0][warp] = sum;
        scratch[1][warp] = sx;
        scratch[2][warp] = sy;
    }
    __syncthreads();
    if (!warp) {
        sum = lane < blockDim.x / 32 ? scratch[0][lane] : 0.0f;
        sx = lane < blockDim.x / 32 ? scratch[1][lane] : 0.0f;
        sy = lane < blockDim.x / 32 ? scratch[2][lane] : 0.0f;
        sum = warp_sum(sum);
        sx = warp_sum(sx);
        sy = warp_sum(sy);
        if (!lane) out[int64_t(batch) * out_stride + row] = make_float2(sx / sum, sy / sum);
    }
}
