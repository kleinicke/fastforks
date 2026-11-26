// Instance norm (affine) + LeakyReLU of channels-last 3D fp16 activations, in place: nnU-Net's conv block epilogue.
//
// x: [N, S, C] fp16 (channels_last_3d, S = D * H * W), C % 8 == 0, 16-byte aligned. Three launches:
//   in_stats_h: per (n, c) shifted sums  sum(x - k), sum((x - k)^2)  in fp32 (k = x[n, 0, c], against cancellation),
//               each block writes its own shifted moments into [N, blocks, C, 2].
//   in_final_h: scale = w * rsqrt(var + eps), shift = b - mean * scale  (biased variance, as InstanceNorm3d) into
//               stats, in place.
//   in_apply_h: y = x * scale + shift (fp32), y = y >= 0 ? y : slope * y, stored fp16; 8 channels per thread.
#include <cuda_fp16.h>
#include <cassert>
#include <cmath>
#include <stdint.h>

union H8 {
    uint4 u;
    __half2 h[4];
};

static __device__ __forceinline__ __half2 residual_act(__half2 x, __half2 skip, float slope) {
    // The original residual add stores half before LeakyReLU promotes to float.
    const auto sum = __half22float2(__hadd2_rn(x, skip));
    return __floats2half2_rn(sum.x > 0.f ? sum.x : sum.x * slope, sum.y > 0.f ? sum.y : sum.y * slope);
}

extern "C" __global__ void vol_residual_act_h(uint4* __restrict__ out, const uint4* __restrict__ skip,
                                               int64_t vectors, float slope) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < vectors;
         i += (int64_t)gridDim.x * blockDim.x) {
        H8 a, b;
        a.u = out[i];
        b.u = skip[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) a.h[j] = residual_act(a.h[j], b.h[j], slope);
        out[i] = a.u;
    }
}

template <bool Packed, typename Index>
__device__ __forceinline__ void accumulate_h(__half* out, const __half* logits, const __half* gaussian,
                                             int channels, int d, int h, int w, int64_t oz, int64_t oy, int64_t ox) {
    const int units = Packed ? channels / 2 : channels;
    const Index count = (Index)d * h * w * units;
    for (Index i = (Index)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (Index)gridDim.x * blockDim.x) {
        const Index r = i / units, c = i % units;
        const Index row = r / w;
        const int64_t offset = (row / h) * oz + (row % h) * oy + (r % w) * ox;
        const __half g = gaussian[r];
        if constexpr (Packed) {
            auto* a = reinterpret_cast<__half2*>(out + offset);
            const __half2 x = reinterpret_cast<const __half2*>(logits)[i];
            a[c] = __hadd2_rn(a[c], __hmul2_rn(x, __halves2half2(g, g)));
        } else {
            out[offset + c] = __hadd_rn(out[offset + c], __hmul_rn(logits[i], g));
        }
    }
}

extern "C" __global__ void vol_accum_h(__half* __restrict__ out, const __half* __restrict__ logits,
                                       const __half* __restrict__ gaussian, int channels, int d,
                                       int h, int w, int64_t oz, int64_t oy, int64_t ox, int packed) {
    const bool narrow = (int64_t)d * h * w * (packed ? channels / 2 : channels) <= INT32_MAX;
    if (packed && narrow) {
        accumulate_h<true, uint32_t>(out, logits, gaussian, channels, d, h, w, oz, oy, ox);
    } else if (packed) {
        accumulate_h<true, int64_t>(out, logits, gaussian, channels, d, h, w, oz, oy, ox);
    } else if (narrow) {
        accumulate_h<false, uint32_t>(out, logits, gaussian, channels, d, h, w, oz, oy, ox);
    } else {
        accumulate_h<false, int64_t>(out, logits, gaussian, channels, d, h, w, oz, oy, ox);
    }
}

// The same half-pixel coordinates and order of operations as PyTorch's CUDA trilinear upsampling.
struct SamplingAxis {
    int base, next;
    float w0, w1;
};

static __device__ __forceinline__ SamplingAxis resample_axis(int out, int size, float scale) {
    const float pos = fmaxf(fmaf(scale, out + 0.5f, -0.5f), 0.0f);
    const int lo = (int)pos;
    const float weight = pos - lo;
    return {lo, lo + (lo < size - 1), 1.f - weight, weight};
}

static __device__ __forceinline__ float interpolate_plane(const __half* plane, int64_t ys, int64_t xs,
                                                          SamplingAxis y, SamplingAxis x, int c) {
    const __half* r0 = plane + y.base * ys + c;
    const __half* r1 = plane + y.next * ys + c;
    return y.w0 * (x.w0 * __half2float(r0[x.base * xs]) + x.w1 * __half2float(r0[x.next * xs]))
         + y.w1 * (x.w0 * __half2float(r1[x.base * xs]) + x.w1 * __half2float(r1[x.next * xs]));
}

static __device__ __forceinline__ float2 interpolate_plane2(const __half* plane, int64_t ys, int64_t xs,
                                                           SamplingAxis y, SamplingAxis x, int c) {
    const __half* r0 = plane + y.base * ys + c;
    const __half* r1 = plane + y.next * ys + c;
    const auto a = __half22float2(*reinterpret_cast<const __half2*>(r0 + x.base * xs));
    const auto b = __half22float2(*reinterpret_cast<const __half2*>(r0 + x.next * xs));
    const auto d = __half22float2(*reinterpret_cast<const __half2*>(r1 + x.base * xs));
    const auto e = __half22float2(*reinterpret_cast<const __half2*>(r1 + x.next * xs));
    return {y.w0 * (x.w0 * a.x + x.w1 * b.x) + y.w1 * (x.w0 * d.x + x.w1 * e.x),
            y.w0 * (x.w0 * a.y + x.w1 * b.y) + y.w1 * (x.w0 * d.y + x.w1 * e.y)};
}

static __device__ __forceinline__ bool larger(float v, int c, float best, int index) {
    return v > best || (v == best && c < index) || (isnan(v) && (!isnan(best) || c < index));
}

template <bool Packed, typename Index>
__device__ __forceinline__ void resize_labels_h(
    uint8_t* __restrict__ out, const __half* __restrict__ current, const __half* __restrict__ previous,
    int channels, int src_d, int src_h, int src_w, int first_src_z, int available_src,
    int out_h, int out_w, int out_depth, int first_out_z,
    int64_t zs, int64_t ys, int64_t xs, float zscale, float yscale, float xscale) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    const Index points = (Index)out_depth * out_h * out_w;
    for (Index i = (Index)blockIdx.x * warps + warp; i < points; i += (Index)gridDim.x * warps) {
        const Index row = i / out_w;
        const auto z = resample_axis(first_out_z + row / out_h, src_d, zscale);
        const auto y = resample_axis(row % out_h, src_h, yscale);
        const auto x = resample_axis(i % out_w, src_w, xscale);
        const int d0 = z.base - first_src_z, d1 = z.next - first_src_z;
        assert(d0 >= -1 && d1 < available_src && (d0 >= 0 || previous));
        const __half* a = d0 < 0 ? previous : current + zs * d0;
        const __half* b = d1 < 0 ? previous : current + zs * d1;
        const int64_t y0 = d0 < 0 ? (int64_t)src_w * channels : ys, x0 = d0 < 0 ? channels : xs;
        const int64_t y1 = d1 < 0 ? (int64_t)src_w * channels : ys, x1 = d1 < 0 ? channels : xs;
        float best = -INFINITY;
        int index = channels;
        if constexpr (Packed) {
            for (int c = lane * 2; c < channels; c += 64) {
                const auto v0 = interpolate_plane2(a, y0, x0, y, x, c);
                const auto v1 = interpolate_plane2(b, y1, x1, y, x, c);
                const auto lo = z.w0 * v0.x + z.w1 * v1.x;
                const auto hi = z.w0 * v0.y + z.w1 * v1.y;
                if (larger(lo, c, best, index)) {
                    best = lo;
                    index = c;
                }
                if (larger(hi, c + 1, best, index)) {
                    best = hi;
                    index = c + 1;
                }
            }
        } else {
            for (int c = lane; c < channels; c += 32) {
                const auto v = z.w0 * interpolate_plane(a, y0, x0, y, x, c)
                             + z.w1 * interpolate_plane(b, y1, x1, y, x, c);
                if (larger(v, c, best, index)) {
                    best = v;
                    index = c;
                }
            }
        }
        for (int delta = 16; delta > 0; delta /= 2) {
            const auto v = __shfl_down_sync(0xffffffff, best, delta);
            const auto c = __shfl_down_sync(0xffffffff, index, delta);
            if (larger(v, c, best, index)) {
                best = v;
                index = c;
            }
        }
        if (!lane) out[i] = (uint8_t)index;
    }
}

extern "C" __global__ void vol_resize_labels_h(
    uint8_t* __restrict__ out, const __half* __restrict__ current, const __half* __restrict__ previous,
    int channels, int src_d, int src_h, int src_w, int first_src_z, int available_src,
    int out_h, int out_w, int out_depth, int first_out_z,
    int64_t zs, int64_t ys, int64_t xs, float zscale, float yscale, float xscale) {
    const bool packed = channels >= 64 && !((channels | zs | ys | xs) & 1)
                        && !(((uintptr_t)current | (uintptr_t)previous) & 3);
    const bool narrow = (int64_t)out_depth * out_h * out_w <= INT32_MAX;
#define RESIZE(PACKED, INDEX) resize_labels_h<PACKED, INDEX>(out, current, previous, \
    channels, src_d, src_h, src_w, first_src_z, available_src, out_h, out_w, out_depth, first_out_z, \
    zs, ys, xs, zscale, yscale, xscale)
    if (packed && narrow) {
        RESIZE(true, uint32_t);
    } else if (packed) {
        RESIZE(true, int64_t);
    } else if (narrow) {
        RESIZE(false, uint32_t);
    } else {
        RESIZE(false, int64_t);
    }
#undef RESIZE
}

// grid (blocks, N), block (C / 8, rows); shared rows * C * 2 floats
extern "C" __global__ void in_stats_h(const __half* __restrict__ x, float* __restrict__ stats, int S, int C, const __half* conv_bias) {
    extern __shared__ float sh[];
    const int c8 = threadIdx.x, C8 = blockDim.x, n = blockIdx.y;
    const __half* xn = x + (int64_t)n * S * C;
    H8 k;
    k.u = *reinterpret_cast<const uint4*>(xn + c8 * 8);
    H8 bias;
    if (conv_bias) bias.u = __ldg(reinterpret_cast<const uint4*>(conv_bias) + c8);
    float kf[8], s1[8], s2[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 f = __half22float2(conv_bias ? __hadd2(k.h[j], bias.h[j]) : k.h[j]);
        kf[2 * j] = f.x;
        kf[2 * j + 1] = f.y;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) s1[j] = s2[j] = 0.f;
    for (int64_t r = (int64_t)blockIdx.x * blockDim.y + threadIdx.y; r < S; r += (int64_t)gridDim.x * blockDim.y) {
        H8 v;
        v.u = __ldg(reinterpret_cast<const uint4*>(xn + r * C + c8 * 8));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __half22float2(conv_bias ? __hadd2(v.h[j], bias.h[j]) : v.h[j]);
            float a = f.x - kf[2 * j], b = f.y - kf[2 * j + 1];
            s1[2 * j] += a;
            s2[2 * j] = fmaf(a, a, s2[2 * j]);
            s1[2 * j + 1] += b;
            s2[2 * j + 1] = fmaf(b, b, s2[2 * j + 1]);
        }
    }
    float* row = sh + (threadIdx.y * C8 + c8) * 16;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        row[j] = s1[j];
        row[8 + j] = s2[j];
    }
    __syncthreads();
    // threads of the block sum one (channel, moment) column each over the rows
    const int t = threadIdx.y * C8 + c8, nt = C8 * blockDim.y;
    for (int q = t; q < C * 2; q += nt) {
        int c = q >> 1, m = q & 1;  // channel c = c8' * 8 + j
        int cc8 = c >> 3, j = c & 7;
        float acc = 0.f;
        for (int y = 0; y < blockDim.y; ++y) acc += sh[(y * C8 + cc8) * 16 + m * 8 + j];
        stats[(((int64_t)n * gridDim.x + blockIdx.x) * C + c) * 2 + m] = acc;
    }
}

// Reduce the block moments in a fixed order. Adjacent lanes read adjacent
// channels and eight cooperating warps cover the spatial block axis.
extern "C" __global__ void in_final_h(const __half* __restrict__ x, const float2* __restrict__ partial,
                                      float2* __restrict__ stats, const __half* __restrict__ w,
                                      const __half* __restrict__ b, int blocks, int S, int C, float eps, const __half* conv_bias) {
    __shared__ float2 sums[8][32];
    int n = blockIdx.y, c = blockIdx.x * 32 + threadIdx.x;
    float2 sum = make_float2(0.f, 0.f);
    if (c < C) {
        for (int i = threadIdx.y; i < blocks; i += 8) {
            float2 v = partial[((int64_t)n * blocks + i) * C + c];
            sum.x += v.x;
            sum.y += v.y;
        }
    }
    sums[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y || c >= C) return;
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        sum.x += sums[i][threadIdx.x].x;
        sum.y += sums[i][threadIdx.x].y;
    }
    __half first = x[(int64_t)n * S * C + c];
    float k = __half2float(conv_bias ? __hadd(first, conv_bias[c]) : first);
    float inv = 1.f / (float)S;
    float d = sum.x * inv;
    float var = fmaxf(fmaf(-d, d, sum.y * inv), 0.f);
    float scale = __half2float(w[c]) * rsqrtf(var + eps);
    stats[(int64_t)n*C+c] = make_float2(scale, fmaf(-(k + d), scale, __half2float(b[c])));
}

// N * S * C / 8 grid-stride items
template <bool Residual>
static __device__ __forceinline__ void in_apply(__half* __restrict__ x, const float* __restrict__ stats, int64_t total,
                                                int S, int C, float slope, const __half* conv_bias,
                                                const uint4* __restrict__ skip, float residual_slope) {
    const int C8 = C >> 3;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        int c8 = (int)(i % C8);
        int n = (int)(i / ((int64_t)S * C8));
        const float4* st = reinterpret_cast<const float4*>(stats + ((int64_t)n * C + c8 * 8) * 2);
        H8 bias;
        if (conv_bias) bias.u = __ldg(reinterpret_cast<const uint4*>(conv_bias) + c8);
        H8 v, r;
        v.u = reinterpret_cast<uint4*>(x)[i];
        if constexpr (Residual) r.u = skip[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float4 p = __ldg(st + j);  // scale, shift of channels 2j, 2j+1
            float2 f = __half22float2(conv_bias ? __hadd2(v.h[j], bias.h[j]) : v.h[j]);
            float a = fmaf(f.x, p.x, p.y), bb = fmaf(f.y, p.z, p.w);
            a = a >= 0.f ? a : a * slope;
            bb = bb >= 0.f ? bb : bb * slope;
            v.h[j] = __floats2half2_rn(a, bb);
            if constexpr (Residual) v.h[j] = residual_act(v.h[j], r.h[j], residual_slope);
        }
        reinterpret_cast<uint4*>(x)[i] = v.u;
    }
}

extern "C" __global__ void in_apply_h(__half* __restrict__ x, const float* __restrict__ stats, int64_t total, int S,
                                      int C, float slope, const __half* conv_bias) {
    in_apply<false>(x, stats, total, S, C, slope, conv_bias, nullptr, 0.f);
}

extern "C" __global__ void in_apply_residual_h(__half* __restrict__ x, const float* __restrict__ stats, int64_t total,
                                               int S, int C, float slope, const __half* conv_bias,
                                               const uint4* __restrict__ skip, float residual_slope) {
    in_apply<true>(x, stats, total, S, C, slope, conv_bias, skip, residual_slope);
}

extern "C" __global__ void vol_gaussian_h(const double* __restrict__ axes, __half* __restrict__ out,
                                         int* __restrict__ smallest, int d, int h, int w, int64_t count) {
    // Match the CPU outer product and its FP64 -> FP32 -> FP16 rounding.
    const double scale = axes[d + h + w];
    int best = 0x7c00;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t row = i / w;
        const double a = axes[row / h], b = axes[d + row % h], c = axes[d + h + i % w];
        const auto val = __float2half_rn(__double2float_rn(__ddiv_rn(__dmul_rn(__dmul_rn(a, b), c), scale)));
        out[i] = val;
        const unsigned bits = __half_as_ushort(val);
        if (bits) best = min(best, (int)bits);
    }
    // The launcher uses 256 threads. Positive FP16 values sort by their unsigned bits.
    __shared__ int mins[8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int step = 16; step; step /= 2) {
        best = min(best, __shfl_down_sync(0xffffffff, best, step));
    }
    if (!lane) mins[warp] = best;
    __syncthreads();
    if (!warp) {
        best = lane < blockDim.x / 32 ? mins[lane] : 0x7c00;
        for (int step = 16; step; step /= 2) {
            best = min(best, __shfl_down_sync(0xffffffff, best, step));
        }
        if (!lane) atomicMin(smallest, best);
    }
}

extern "C" __global__ void vol_gaussian_fix_h(__half* __restrict__ out, const int* __restrict__ smallest, int64_t count) {
    const auto fallback = __ushort_as_half((unsigned short)*smallest);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (int64_t)gridDim.x * blockDim.x) {
        if (!__half_as_ushort(out[i])) out[i] = fallback;
    }
}

template <typename T>
__device__ __forceinline__ T norm_cast(float v) { return T(v); }
template <>
__device__ __forceinline__ __half norm_cast<__half>(float v) { return __float2half_rn(v); }

template <typename T>
__device__ void normalize_ct(const float* __restrict__ src, T* __restrict__ dst,
        int h, int w, int64_t count, int64_t sd, int64_t sh, int64_t sw,
        int64_t od, int64_t oh, int64_t ow, float lower, float upper, float mean, float invstd) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t row = i / w, x = i % w, y = row % h, z = row / h;
        const float v = src[z * sd + y * sh + x * sw];
        const float clipped = isnan(v) ? v : fminf(fmaxf(v, lower), upper);
        dst[z * od + y * oh + x * ow] = norm_cast<T>(__fmul_rn(__fsub_rn(clipped, mean), invstd));
    }
}

extern "C" __global__ void vol_ct_norm_f(const float* src, float* dst, int h, int w, int64_t count,
        int64_t sd, int64_t sh, int64_t sw, int64_t od, int64_t oh, int64_t ow,
        float lower, float upper, float mean, float invstd) {
    normalize_ct(src, dst, h, w, count, sd, sh, sw, od, oh, ow, lower, upper, mean, invstd);
}

extern "C" __global__ void vol_ct_norm_h(const float* src, __half* dst, int h, int w, int64_t count,
        int64_t sd, int64_t sh, int64_t sw, int64_t od, int64_t oh, int64_t ow,
        float lower, float upper, float mean, float invstd) {
    normalize_ct(src, dst, h, w, count, sd, sh, sw, od, oh, ow, lower, upper, mean, invstd);
}

extern "C" __global__ void vol_merge_u8(const uint8_t* __restrict__ src, const uint8_t* __restrict__ lut,
        uint8_t* __restrict__ out, int h, int w, int write_zero, int lut_size, int64_t count,
        int64_t sd, int64_t sh, int64_t sw, int64_t od, int64_t oh, int64_t ow) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t row = i / w, x = i % w, y = row % h, z = row / h;
        const uint8_t label = src[z * sd + y * sh + x * sw];
        const uint8_t mapped = lut ? (label < lut_size ? lut[label] : 0) : label;
        if (write_zero || mapped)
            out[z * od + y * oh + x * ow] = mapped;
    }
}

__device__ __forceinline__ __half2 vol_rotary_pair_h(__half2 value, float2 sin, float2 cos) {
    const float2 x = __half22float2(value);
    // Eager rounds both FP32 products before adding, then rounds once to FP16.
    const float even = __fadd_rn(__fmul_rn(x.x, cos.x), __fmul_rn(-x.y, sin.x));
    const float odd = __fadd_rn(__fmul_rn(x.y, cos.y), __fmul_rn(x.x, sin.y));
    return __floats2half2_rn(even, odd);
}

extern "C" __global__ void vol_rotary_padded66_h(const __half2* __restrict__ x, const float2* __restrict__ sin,
        const float2* __restrict__ cos, __half2* __restrict__ out, int tokens, int heads,
        int64_t batch_stride, int64_t head_stride, int64_t token_stride) {
    constexpr int lanes = 32;
    constexpr int pairs = 33, padded_pairs = 36;
    const int lane = threadIdx.x % lanes;
    const unsigned token = blockIdx.x * (blockDim.x / lanes) + threadIdx.x / lanes;
    if (token >= static_cast<unsigned>(tokens))
        return;
    const unsigned head = blockIdx.y, batch = blockIdx.z;
    const int64_t row = (int64_t(batch) * heads + head) * tokens + token;
    const auto* src = x + batch * batch_stride + head * head_stride + token * token_stride;
    const auto* angles_s = sin + int64_t(token) * pairs;
    const auto* angles_c = cos + int64_t(token) * pairs;
    auto* dst = out + row * padded_pairs;
    dst[lane] = vol_rotary_pair_h(src[lane], angles_s[lane], angles_c[lane]);
    if (lane < padded_pairs - lanes) {
        dst[lane + lanes] = lane == 0
            ? vol_rotary_pair_h(src[lanes], angles_s[lanes], angles_c[lanes])
            : __float2half2_rn(0.f);
    }
}
