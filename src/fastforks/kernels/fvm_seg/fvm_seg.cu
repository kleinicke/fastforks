// fvm_seg: the Mask2Former / OneFormer kernels of fast-vision-models (exp 009 seg_kernels.cu, Apache-2.0): fused
// multi-scale deformable self-attention of the pixel decoder, and the Swin layer gathers (LayerNorm + pad + roll +
// window partition; window reverse + roll back + crop + residual + layernorm_after). fp16, nothing depends on the
// input size.
#include <cuda_fp16.h>
#include <stdint.h>

// value: fp16 [B, S, NH, D] (D even); off: fp16 [B, S, NH, L, P, 2] raw offsets (pixels of the target level);
// logit: fp16 [B, S, NH, L*P] pre-softmax weights; shapes: int32 [L, 2] (h, w); start: int32 [L] level start index.
// out: fp16 [B, S, NH, D]. Softmax, locations and bilinear weights in fp32; grid_sample semantics (align_corners=False,
// zero padding): x = loc_x * w - 0.5. One thread per (b, s, h, d/2).
extern "C" __global__ void msda_self_h(const __half* __restrict__ value, const __half* __restrict__ off,
                                       const __half* __restrict__ logit, const int* __restrict__ shapes,
                                       const int* __restrict__ start, __half* __restrict__ out,
                                       int B, int S, int NH, int D, int L, int P) {
    int D2 = D / 2;
    long long n = (long long)B * S * NH * D2;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int d2 = (int)(i % D2);
        long long t = i / D2;
        int h = (int)(t % NH);
        t /= NH;
        int s = (int)(t % S);
        int b = (int)(t / S);
        // reference point of query s
        int lq = 0;
        for (int l = 1; l < L; ++l) if (s >= start[l]) lq = l;
        int qi = s - start[lq], qw = shapes[2 * lq + 1], qh = shapes[2 * lq];
        float rx = ((qi % qw) + 0.5f) / qw, ry = ((qi / qw) + 0.5f) / qh;
        long long qbase = ((long long)b * S + s) * NH + h;
        const __half* lg = logit + qbase * L * P;
        const __half* of = off + qbase * L * P * 2;
        float mx = -INFINITY;
        for (int k = 0; k < L * P; ++k) mx = fmaxf(mx, __half2float(lg[k]));
        float den = 0.f;
        for (int k = 0; k < L * P; ++k) den += __expf(__half2float(lg[k]) - mx);
        float inv = 1.f / den;
        float2 acc = make_float2(0.f, 0.f);
        for (int l = 0; l < L; ++l) {
            int hl = shapes[2 * l], wl = shapes[2 * l + 1];
            const __half2* vb = (const __half2*)(value + (((long long)b * S + start[l]) * NH + h) * D) + d2;
            long long rowp = (long long)wl * NH * D2;   // half2 stride of one row of this level
            long long colp = (long long)NH * D2;
            for (int p = 0; p < P; ++p) {
                int k = l * P + p;
                float a = __expf(__half2float(lg[k]) - mx) * inv;
                float x = (rx + __half2float(of[2 * k]) / wl) * wl - 0.5f;
                float y = (ry + __half2float(of[2 * k + 1]) / hl) * hl - 0.5f;
                int x0 = (int)floorf(x), y0 = (int)floorf(y);
                float lx = x - x0, ly = y - y0;
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    int xx = x0 + (c & 1), yy = y0 + (c >> 1);
                    if (xx < 0 || yy < 0 || xx >= wl || yy >= hl) continue;
                    float wgt = a * ((c & 1) ? lx : 1.f - lx) * ((c >> 1) ? ly : 1.f - ly);
                    float2 v = __half22float2(vb[yy * rowp + xx * colp]);
                    acc.x += wgt * v.x;
                    acc.y += wgt * v.y;
                }
            }
        }
        ((__half2*)(out + qbase * D))[d2] = __floats2half2_rn(acc.x, acc.y);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void ln_row(const __half2* __restrict__ xr, const __half2* __restrict__ w2,
                                       const __half2* __restrict__ b2, __half2* __restrict__ o, int C, float eps, int lane) {
    int C2 = C / 2;
    float s = 0.f;
    for (int c = lane; c < C2; c += 32) { float2 v = __half22float2(xr[c]); s += v.x + v.y; }
    float mean = warp_sum(s) / C;
    float q = 0.f;
    for (int c = lane; c < C2; c += 32) { float2 v = __half22float2(xr[c]); v.x -= mean; v.y -= mean; q += v.x * v.x + v.y * v.y; }
    float rstd = rsqrtf(warp_sum(q) / C + eps);
    for (int c = lane; c < C2; c += 32) {
        float2 v = __half22float2(xr[c]), ww = __half22float2(w2[c]), bb = __half22float2(b2[c]);
        o[c] = __floats2half2_rn((v.x - mean) * rstd * ww.x + bb.x, (v.y - mean) * rstd * ww.y + bb.y);
    }
}

// out[r] = idx[r] >= 0 ? LayerNorm(x[idx[r]]) : 0    (Swin: LN + pad + roll + window partition in one pass)
extern "C" __global__ void ln_gather_h(const __half* __restrict__ x, const int* __restrict__ idx, const __half* __restrict__ w,
                                       const __half* __restrict__ b, __half* __restrict__ out, int R, int C, float eps) {
    int lane = threadIdx.x & 31;
    int nw = (gridDim.x * blockDim.x) >> 5;
    for (int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; r < R; r += nw) {
        int s = idx[r];
        __half2* o = (__half2*)(out + (long long)r * C);
        if (s < 0) {
            for (int c = lane; c < C / 2; c += 32) o[c] = __float2half2_rn(0.f);
            continue;
        }
        ln_row((const __half2*)(x + (long long)s * C), (const __half2*)w, (const __half2*)b, o, C, eps, lane);
    }
}

// h[t] = res[t] + o[inv[t]] (rounded to fp16 like the eager add), y[t] = LayerNorm(h[t])
// (Swin: window reverse + roll back + crop + residual + layernorm_after in one pass)
extern "C" __global__ void gather_add_ln_h(const __half* __restrict__ o, const int* __restrict__ inv, const __half* __restrict__ res,
                                           const __half* __restrict__ w, const __half* __restrict__ b, __half* __restrict__ h,
                                           __half* __restrict__ y, int N, int C, float eps) {
    int lane = threadIdx.x & 31;
    int nw = (gridDim.x * blockDim.x) >> 5;
    int C2 = C / 2;
    for (int t = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; t < N; t += nw) {
        const __half2* src = (const __half2*)(o + (long long)inv[t] * C);
        const __half2* rr = (const __half2*)(res + (long long)t * C);
        __half2* hr = (__half2*)(h + (long long)t * C);
        for (int c = lane; c < C2; c += 32) hr[c] = __hadd2(rr[c], src[c]);
        __syncwarp();
        ln_row(hr, (const __half2*)w, (const __half2*)b, (__half2*)(y + (long long)t * C), C, eps, lane);
    }
}
