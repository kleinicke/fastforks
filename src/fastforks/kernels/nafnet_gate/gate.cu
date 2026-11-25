#include <cuda_fp16.h>
#include <stdint.h>

// NHWC input, kernel-major weights [9, 2C]. Neighboring lanes load neighboring
// channels. Each output thread accumulates one channel from each gate operand.
template<bool cropped>
__device__ __forceinline__ void gate_scalar(const half* input, const half* weight, const half* bias,
        half* output, int h, int w, int c, int64_t count, int core, int src_top, int dst_h, int dst_top) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int ch = i % c;
        const int64_t pixel = i / c;
        const int x = pixel % w;
        const int64_t row = pixel / w;
        const int rel = cropped ? row % core : row % h;
        const int64_t sample = cropped ? row / core : 0;
        const int y = rel + src_top;
        const int64_t src_pixel = cropped ? (sample * h + y) * w + x : pixel;
        float a = 0.f, b = 0.f;
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                if (unsigned(y + dy) < unsigned(h) && unsigned(x + dx) < unsigned(w)) {
                    const int64_t p = (src_pixel + int64_t(dy) * w + dx) * (2 * c) + ch;
                    const int k = ((dy + 1) * 3 + dx + 1) * (2 * c) + ch;
                    a = fmaf(__half2float(input[p]), __half2float(weight[k]), a);
                    b = fmaf(__half2float(input[p + c]), __half2float(weight[k + c]), b);
                }
            }
        }
        const half left = __float2half_rn(a + __half2float(bias[ch]));
        const half right = __float2half_rn(b + __half2float(bias[ch + c]));
        const int64_t dest = cropped ? ((sample * dst_h + dst_top + rel) * w + x) * c + ch : i;
        output[dest] = __hmul(left, right);
    }
}

extern "C" __global__ void depthwise_gate(const half* input, const half* weight, const half* bias,
        half* output, int h, int w, int c, int64_t count) {
    gate_scalar<false>(input, weight, bias, output, h, w, c, count, h, 0, h, 0);
}

extern "C" __global__ void depthwise_gate_rows(const half* input, const half* weight, const half* bias,
        half* output, int h, int w, int c, int core, int src_top, int dst_h, int dst_top, int64_t count) {
    gate_scalar<true>(input, weight, bias, output, h, w, c, count, core, src_top, dst_h, dst_top);
}

// A warp covers 64 channels of four adjacent pixels. The six overlapping taps
// of each input row are loaded once for all four outputs.
template<bool cropped>
__device__ __forceinline__ void gate_vector(const half2* input, const half2* weight,
        const half2* bias, half2* output, int h, int w, int c, int64_t rows, int core, int src_top, int dst_h, int dst_top) {
    const int tiles = (w + 15) / 16;
    const int64_t row = blockIdx.x / tiles;
    if (row >= rows) return;
    const int x0 = (blockIdx.x % tiles) * 16 + threadIdx.y * 4;
    if (x0 >= w) return;
    const int k = blockIdx.y * 32 + threadIdx.x;
    const int rel = cropped ? row % core : row % h;
    const int64_t sample = cropped ? row / core : 0;
    const int y = rel + src_top;
    const int64_t src_row = cropped ? sample * h + y : row;
    float2 a[4] = {}, b[4] = {};
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        if (unsigned(y + dy) >= unsigned(h)) continue;
        float2 wa[3], wb[3];
        #pragma unroll
        for (int t = 0; t < 3; ++t) {
            const int off = ((dy + 1) * 3 + t) * 2 * c + k;
            wa[t] = __half22float2(weight[off]);
            wb[t] = __half22float2(weight[off + c]);
        }
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            const int x = x0 + i - 1;
            if (unsigned(x) >= unsigned(w)) continue;
            const int64_t off = ((src_row + dy) * w + x) * 2 * c + k;
            const float2 va = __half22float2(input[off]), vb = __half22float2(input[off + c]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int t = i - j;
                if (unsigned(t) < 3) {
                    a[j].x = fmaf(va.x, wa[t].x, a[j].x);
                    a[j].y = fmaf(va.y, wa[t].y, a[j].y);
                    b[j].x = fmaf(vb.x, wb[t].x, b[j].x);
                    b[j].y = fmaf(vb.y, wb[t].y, b[j].y);
                }
            }
        }
    }
    const float2 ba = __half22float2(bias[k]), bb = __half22float2(bias[k + c]);
    const int64_t dest_row = cropped ? sample * dst_h + dst_top + rel : row;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (x0 + j >= w) break;
        const half2 l = __floats2half2_rn(a[j].x + ba.x, a[j].y + ba.y);
        const half2 r = __floats2half2_rn(b[j].x + bb.x, b[j].y + bb.y);
        output[(dest_row * w + x0 + j) * c + k] = __hmul2(l, r);
    }
}

extern "C" __global__ void depthwise_gate_tile(const half2* input, const half2* weight,
        const half2* bias, half2* output, int h, int w, int c, int64_t rows) {
    gate_vector<false>(input, weight, bias, output, h, w, c, rows, h, 0, h, 0);
}

extern "C" __global__ void depthwise_gate_rows_tile(const half2* input, const half2* weight,
        const half2* bias, half2* output, int h, int w, int c, int core, int src_top, int dst_h, int dst_top,
        int64_t rows) {
    gate_vector<true>(input, weight, bias, output, h, w, c, rows, core, src_top, dst_h, dst_top);
}

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, delta);
    return value;
}

// One warp per NHWC pixel. Keep the channel values in registers for both
// centered variance and normalization; no mean or variance buffer is needed.
template<int channels>
__device__ void norm_row(const half2* x, half2* out, int64_t row, float eps) {
    constexpr int pairs = channels / 64;
    const int64_t i = row * (channels / 2) + threadIdx.x;
    float2 values[pairs];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < pairs; ++k) {
        values[k] = __half22float2(x[i + 32 * k]);
        sum += values[k].x + values[k].y;
    }
    const float mean = warp_sum(sum) / channels;
    float ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < pairs; ++k) {
        values[k].x -= mean;
        values[k].y -= mean;
        ss = fmaf(values[k].x, values[k].x, ss);
        ss = fmaf(values[k].y, values[k].y, ss);
    }
    const float scale = rsqrtf(warp_sum(ss) / channels + eps);
    #pragma unroll
    for (int k = 0; k < pairs; ++k)
        out[i + 32 * k] = __floats2half2_rn(values[k].x * scale, values[k].y * scale);
}

extern "C" __global__ void channel_norm(const half2* x, half2* out, int channels, int64_t rows, float eps) {
    for (int64_t row = int64_t(blockIdx.x) * blockDim.y + threadIdx.y;
         row < rows; row += int64_t(gridDim.x) * blockDim.y) {
        switch (channels) {
            case 64: norm_row<64>(x, out, row, eps); break;
            case 128: norm_row<128>(x, out, row, eps); break;
            case 256: norm_row<256>(x, out, row, eps); break;
            case 512: norm_row<512>(x, out, row, eps); break;
            case 1024: norm_row<1024>(x, out, row, eps); break;
        }
    }
}

// Multiply by the learned local scores with the same centered edge replication as PyTorch's pad.
extern "C" __global__ void scale_replicate(
    __half2* __restrict__ x, const __half2* __restrict__ scores,
    int h, int w, int oh, int ow, int channels, int64_t count) {
    const int py = (h - oh) / 2, px = (w - ow) / 2;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t pixel = i / channels;
        const int col = pixel % w, row = pixel / w % h;
        const int sx = max(0, min(col - px, ow - 1)), sy = max(0, min(row - py, oh - 1));
        const int64_t base = (pixel / ((int64_t)h * w) * oh + sy) * ow + sx;
        x[i] = __hmul2_rn(x[i], scores[base * channels + i % channels]);
    }
}
