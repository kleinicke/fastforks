#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float conv4(const float* w, int channels, int c, float x0, float x1, float x2, float x3) {
    float v = __fadd_rn(0.f, __fmul_rn(w[c], x0));
    v = __fadd_rn(v, __fmul_rn(w[channels + c], x1));
    v = __fadd_rn(v, __fmul_rn(w[2 * channels + c], x2));
    return __fadd_rn(v, __fmul_rn(w[3 * channels + c], x3));
}

// The ring preserves the half projection outputs. Widen them before the convolution,
// read the oldest frame before replacing it, and preserve the Float32 sum order.
extern "C" __global__ void history_update(
        const half* x, const float* w, const float* bias, const int64_t* order, half* history,
        float* conv, int64_t count, int channels) {
    half* oldest = history + order[0] * count;
    const half* middle = history + order[1] * count;
    const half* newest = history + order[2] * count;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        int c = i % channels;
        half current = x[i];
        float v = conv4(w, channels, c, __half2float(oldest[i]), __half2float(middle[i]),
                        __half2float(newest[i]), __half2float(current));
        conv[i] = __fadd_rn(bias[c], v);
        oldest[i] = current;
    }
}

extern "C" __global__ void history_eager(
        const half* x, const float* w, const float* bias, const half* history,
        half* next_history, float* conv, int64_t count, int channels) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        int c = i % channels;
        half zero = __float2half_rn(0.f);
        half x0 = history ? history[i] : zero;
        half x1 = history ? history[count + i] : zero;
        half x2 = history ? history[2 * count + i] : zero;
        half x3 = x[i];
        conv[i] = __fadd_rn(bias[c], conv4(w, channels, c, __half2float(x0), __half2float(x1),
                                         __half2float(x2), __half2float(x3)));
        next_history[i] = x1;
        next_history[count + i] = x2;
        next_history[2 * count + i] = x3;
    }
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

extern "C" __global__ void recurrent_update(
        const half* y, const float* conv, const float* gate, const float* decay,
        const float* a_param, const bool* first, const float* h, float* next_h, half* output,
        int64_t count, int channels) {
    bool initial = !h || (first && *first);
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        const float log_a = __fmul_rn(-8.0f * sigmoid(decay[i]), a_param[i % channels]);
        const float gain = initial ? 1.0f : sqrtf(__fsub_rn(1.0f, expf(2.0f * log_a)));
        const float input = __fmul_rn(__fmul_rn(conv[i], sigmoid(gate[i])), gain);
        float previous = h ? h[i] : 0.f;
        float next = __fadd_rn(__fmul_rn(previous, expf(log_a)), input);
        next_h[i] = next;
        output[i] = __float2half_rn(next * __half2float(y[i]));
    }
}
