// CREStereo max-path kernels (size-independent; built into one cubin per target by tools/build_kernels.py).
// Origin: our CREStereo fast path (exp 003 pinned/CREStereo ad18c8c, nets/cre_kernels.cu: agcl_corr); flow_pack new.
#include <cuda_fp16.h>
#include <stdint.h>

// Fused AGCL correlation: warp the right features by the flow (bilinear, zero outside, == grid_sample
// align_corners=True) and correlate each of the 4 channel groups (64 ch) with 9 neighbours (3x3 or 1x9).
//   iter mode   (extra == nullptr): corr(p, o) = <L[p], warp(R)[clamp(p + o)]>/64, warp(R)[q] = R(q + flow(q))
//                                  (replicate padding of the warped map, upstream corr_iter)
//   offset mode (extra != nullptr): corr(p, o) = <L[p], R(p + flow(p) + o + extra(p, o))>/64 (corr_att_offset)
// L, R: fp16 NHWC [N, H, W, 256]; flow fp32 NCHW [N, 2, H, W]; extra fp32 NCHW [N, 18, H, W] (o-major, x/y);
// out fp16 NHWC [N, H, W, ocs] with channel g*9 + o, channels 36..ocs-1 set to 0.
// One warp per pixel, lane = 8 channels (group = lane / 8).
extern "C" __global__ void agcl_corr(const __half* __restrict__ L, const __half* __restrict__ R,
                                     const float* __restrict__ flow, const float* __restrict__ extra,
                                     __half* __restrict__ out, int N, int H, int W, int ocs, int small_patch) {
    const int lane = threadIdx.x & 31;
    const long long nwarps = (long long)gridDim.x * (blockDim.x >> 5);
    const long long P = (long long)N * H * W;
    const long long HW = (long long)H * W;
    for (long long p = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); p < P; p += nwarps) {
        const int n = (int)(p / HW);
        const int rem = (int)(p - (long long)n * HW);
        const int y = rem / W, x = rem - y * W;
        float lf[8];
        {
            uint4 v = *reinterpret_cast<const uint4*>(L + p * 256 + lane * 8);
            const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int k = 0; k < 4; k++) { float2 f = __half22float2(h2[k]); lf[2 * k] = f.x; lf[2 * k + 1] = f.y; }
        }
        const float* fx = flow + (long long)n * 2 * HW;
        const float* fy = fx + HW;
        const __half* Rn = R + (long long)n * HW * 256 + lane * 8;
        float keep[9];
#pragma unroll
        for (int o = 0; o < 9; o++) {
            int dy, dx;
            if (small_patch) { dy = o / 3 - 1; dx = o % 3 - 1; } else { dy = 0; dx = o - 4; }
            float sx, sy;
            if (extra == nullptr) {
                int qx = min(max(x + dx, 0), W - 1), qy = min(max(y + dy, 0), H - 1);
                long long q = (long long)qy * W + qx;
                sx = qx + fx[q];
                sy = qy + fy[q];
            } else {
                const float* e = extra + (long long)n * 18 * HW + rem;
                sx = x + fx[rem] + dx + e[(2 * o) * HW];
                sy = y + fy[rem] + dy + e[(2 * o + 1) * HW];
            }
            float x0f = floorf(sx), y0f = floorf(sy);
            int x0 = (int)x0f, y0 = (int)y0f;
            float ax = sx - x0f, ay = sy - y0f;
            float acc[8];
#pragma unroll
            for (int k = 0; k < 8; k++) acc[k] = 0.f;
#pragma unroll
            for (int t = 0; t < 4; t++) {
                int xx = x0 + (t & 1), yy = y0 + (t >> 1);
                float wgt = ((t & 1) ? ax : 1.f - ax) * ((t >> 1) ? ay : 1.f - ay);
                if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;
                uint4 v = *reinterpret_cast<const uint4*>(Rn + ((long long)yy * W + xx) * 256);
                const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    float2 f = __half22float2(h2[k]);
                    acc[2 * k] += wgt * f.x;
                    acc[2 * k + 1] += wgt * f.y;
                }
            }
            float d = 0.f;
#pragma unroll
            for (int k = 0; k < 8; k++) d += lf[k] * acc[k];
            d += __shfl_xor_sync(0xffffffff, d, 4);
            d += __shfl_xor_sync(0xffffffff, d, 2);
            d += __shfl_xor_sync(0xffffffff, d, 1);
            keep[o] = d * (1.f / 64.f);
        }
        // lane l in group g holds all 9 results of group g; write them out: lane (g*8 + j) writes o = j, and j = 0 also o = 8
        __half* op = out + p * ocs;
        const int g = lane >> 3, j = lane & 7;
#pragma unroll
        for (int o = 0; o < 8; o++)
            if (o == j) op[g * 9 + o] = __float2half(keep[o]);
        if (j == 0) op[g * 9 + 8] = __float2half(keep[8]);
        for (int c = 36 + lane; c < ocs; c += 32) op[c] = __float2half(0.f);
    }
}

// Flow into the update block's fp16 inputs, one pass: flow fp32 NCHW [N, 2, H, W] ->
//   f8  fp16 NHWC [N, H, W, 8]: (fx, fy, 0, ..., 0)   (the convf1 input, padded to 8 channels)
//   hx / rhx fp16 NHWC rows of `cs` channels: channels cs-2, cs-1 = (fx, fy)   (motion features' flow tail)
// Null destinations are skipped.
extern "C" __global__ void flow_pack(const float* __restrict__ flow, __half* __restrict__ f8, __half* __restrict__ hx,
                                     __half* __restrict__ rhx, int N, int HW, int cs) {
    const long long P = (long long)N * HW;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < P; p += (long long)gridDim.x * blockDim.x) {
        const long long n = p / HW, q = p - n * HW;
        const __half2 v = __floats2half2_rn(flow[(2 * n) * HW + q], flow[(2 * n + 1) * HW + q]);
        if (f8 != nullptr) {
            uint4 o = make_uint4(0, 0, 0, 0);
            o.x = *reinterpret_cast<const uint32_t*>(&v);
            *reinterpret_cast<uint4*>(f8 + p * 8) = o;
        }
        if (hx != nullptr) *reinterpret_cast<__half2*>(hx + p * cs + cs - 2) = v;
        if (rhx != nullptr) *reinterpret_cast<__half2*>(rhx + p * cs + cs - 2) = v;
    }
}
