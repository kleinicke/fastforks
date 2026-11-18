#include <cuda_fp16.h>
#include <stdint.h>

// A contiguous NHWC tensor or a channel slice of a larger packed NHWC buffer.
extern "C" __global__ void bias_activation(half* x, const half* bias, const half* prelu,
        int channels, int pitch, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int c = i % channels;
        int64_t address = (i / channels) * pitch + c;
        half value = __hadd_rn(x[address], bias[c]);
        if (float(value) < 0)
            value = prelu ? __hmul_rn(value, prelu[c]) : __float2half_rn(float(value) * .2f);
        x[address] = value;
    }
}

extern "C" __global__ void scaled_residual(half* branch, const half* skip, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        branch[i] = __hadd_rn(__float2half_rn(float(branch[i]) * .2f), skip[i]);
}

extern "C" __global__ void span_gate(half* branch, const half* skip, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        half val = branch[i];
        half gate = __hsub_rn(__float2half_rn(1.f / (1.f + expf(-float(val)))), __float2half_rn(.5f));
        branch[i] = __hmul_rn(__hadd_rn(val, skip[i]), gate);
    }
}

// PixelShuffle plus a nearest-upsampled residual, both NHWC, one final RGB write.
extern "C" __global__ void shuffle_residual(const half* low, const half* skip, half* out,
        int h, int w, int c, int scale, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        int chan = i % c;
        int64_t pos = i / c;
        int x = pos % (w * scale), y = (pos / (w * scale)) % (h * scale);
        int64_t image = pos / (int64_t(w) * h * scale * scale);
        int64_t base = (image * h + y / scale) * w + x / scale;
        int subpixel = (y % scale) * scale + x % scale;
        out[i] = __hadd_rn(low[(base * c + chan) * scale * scale + subpixel], skip[base * c + chan]);
    }
}
