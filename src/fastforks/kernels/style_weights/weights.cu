#include <cuda_fp16.h>

// One block per output channel. The input and output filters are both OHWI.
// Demodulation reduces each scaled filter in fp32 before its single fp16 write.
extern "C" __global__ void style_weights(const half* weight, const half* style, half* output,
                                         int channels, int row, int demodulate, float epsilon, float gain) {
    const int tid = threadIdx.x;
    const long long base = (long long)blockIdx.x * row;
    __shared__ float sums[8];
    float factor = 1.f;
    if (demodulate) {
        float sum = 0.f;
        for (int i = tid; i < row; i += blockDim.x) {
            const float v = __half2float(weight[base + i]) * __half2float(style[i % channels]);
            sum += v * v;
        }
        for (int delta = 16; delta > 0; delta /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, delta);
        if (tid % 32 == 0) sums[tid / 32] = sum;
        __syncthreads();
        if (tid < 32) {
            sum = tid < 8 ? sums[tid] : 0.f;
            for (int delta = 4; delta > 0; delta /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, delta);
            if (tid == 0) sums[0] = rsqrtf(sum + epsilon);
        }
        __syncthreads();
        factor = sums[0];
    }
    for (int i = tid; i < row; i += blockDim.x) {
        const float v = __half2float(weight[base + i]) * __half2float(style[i % channels]);
        output[base + i] = __float2half_rn((v * factor) * gain);
    }
}
