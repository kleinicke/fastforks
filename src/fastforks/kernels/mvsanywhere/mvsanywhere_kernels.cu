// MVSAnywhere kernels (our code): fused view-agnostic MLP cost volume and the antialiased-ResNet stem pooling.
//
// mvsa_cv_fused -- one thread per (pixel, depth plane). Per source view k: project the backprojected point, bilinear-
// sample the fp16 NHWC source features (grid_sample bilinear, zeros padding, align_corners=False), build the 46-channel
// metadata vector implicitly, run the 46-32-32-2 MLP in fp32, and reduce over views with an online softmax of output 1
// weighting output 0. Layer 1 is decomposed: per-pixel terms (current features, current ray, bias) come precomputed in
// A[32][hw], per-view terms (pose distances) in Bk[K][32]; the thread multiplies only the 23 view- and depth-dependent
// inputs. The MLP weights are a by-value parameter (constant bank, broadcast operands; needs CUDA >= 12.1, sm_70+).
#include <cuda_fp16.h>

#define NH 32
#define NDYN 23   // warped 16, mask, depth_norm, dot, ray angle, src ray 3
#define KMAX 16

struct MlpWeights {
    float w1[NH * NDYN];  // [j][i], dynamic layer-1 inputs
    float w1pn[NH];       // weight of the plane-depth channel
    float w2[NH * NH];    // [j][i]
    float b2[NH];
    float w3[2 * NH];
    float b3[2];
};

__device__ __forceinline__ float lrelu(float x) { return fmaxf(x, 0.01f * x); }

extern "C" __global__ void __launch_bounds__(128) mvsa_cv_fused(
    const float* __restrict__ ray,     // [3][hw] invK @ pix (pixel centres +0.5), current camera frame
    const float* __restrict__ A,       // [32][hw] per-pixel layer-1 terms (incl. bias)
    const float* __restrict__ cur,     // [16][hw] current matching features
    const __half* __restrict__ src,    // [K][h][w][16] source matching features
    const float* __restrict__ geo,     // [K][16]: P = K_src @ src_T_cur rows 0..2 (12), cam centre c_k (3), pad
    const float* __restrict__ Bk,      // [K][32] per-view layer-1 terms
    const float* __restrict__ planes,  // [nd] depth planes
    const float* __restrict__ range,   // [2] min, max depth
    float* __restrict__ out,           // [nd][hw]
    int K, int h, int w,
    const MlpWeights wt)
{
    __shared__ float sgeo[KMAX * 16];
    __shared__ float sB[KMAX * NH];
    for (int i = threadIdx.x; i < K * 16; i += blockDim.x) sgeo[i] = geo[i];
    for (int i = threadIdx.x; i < K * NH; i += blockDim.x) sB[i] = Bk[i];
    __syncthreads();
    const int hw = h * w;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= hw) return;
    const int di = blockIdx.y;
    const float d = planes[di];
    const float mn = range[0], mx = range[1];
    const float rx = ray[p], ry = ray[hw + p], rz = ray[2 * hw + p];
    // current ray (normalize of d * ray, d > 0)
    float cn = fmaxf(sqrtf(rx * rx + ry * ry + rz * rz), 1e-12f);
    const float cx = rx / cn, cy = ry / cn, cz = rz / cn;
    float cf[16];
#pragma unroll
    for (int i = 0; i < 16; i++) cf[i] = cur[i * hw + p];
    const float pn = (d - mn) / mx;
    const float wx = d * rx, wy = d * ry, wz = d * rz;
    const float uvs_x = 1.f / (float)w, uvs_y = 1.f / (float)h;

    float a1[NH];   // k-independent layer-1 terms, loaded once per thread
#pragma unroll
    for (int j = 0; j < NH; j++) a1[j] = __ldg(A + j * hw + p) + wt.w1pn[j] * pn;
    float m = -INFINITY, s = 0.f, acc = 0.f;
    for (int k = 0; k < K; k++) {
        const float* g = sgeo + k * 16;
        float X = g[0] * wx + g[1] * wy + g[2] * wz + g[3];
        float Y = g[4] * wx + g[5] * wy + g[6] * wz + g[7];
        float Z = g[8] * wx + g[9] * wy + g[10] * wz + g[11];
        const float zd = Z + 1e-8f;
        const float sc = fabsf(Z) > 1e-8f ? 1.f / zd : 1.f;
        const float px = X * sc, py = Y * sc;
        const float maskf = zd > 0.f ? 1.f : 0.f;
        // grid_sample(bilinear, zeros, align_corners=False) on uv = 2 * pix / size - 1
        float u = 2.f * px * uvs_x - 1.f, v = 2.f * py * uvs_y - 1.f;
        float ix = ((u + 1.f) * w - 1.f) * 0.5f, iy = ((v + 1.f) * h - 1.f) * 0.5f;
        ix = fminf(fmaxf(ix, -4.f), (float)w + 4.f);   // NaN-free, far outside -> zeros anyway
        iy = fminf(fmaxf(iy, -4.f), (float)h + 4.f);
        const float fx0 = floorf(ix), fy0 = floorf(iy);
        const int x0 = (int)fx0, y0 = (int)fy0;
        const float ax = ix - fx0, ay = iy - fy0;
        float wf[16];
#pragma unroll
        for (int i = 0; i < 16; i++) wf[i] = 0.f;
        const __half* sk = src + (size_t)k * hw * 16;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int xx = x0 + (c & 1), yy = y0 + (c >> 1);
            const float bw = ((c & 1) ? ax : 1.f - ax) * ((c >> 1) ? ay : 1.f - ay);
            if (xx >= 0 && xx < w && yy >= 0 && yy < h) {
                const uint4* q = reinterpret_cast<const uint4*>(sk + ((size_t)yy * w + xx) * 16);
                uint4 q0 = __ldg(q), q1 = __ldg(q + 1);
                const __half2* h0 = reinterpret_cast<const __half2*>(&q0);
                const __half2* h1 = reinterpret_cast<const __half2*>(&q1);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    float2 a = __half22float2(h0[i]), b = __half22float2(h1[i]);
                    wf[2 * i] += bw * a.x; wf[2 * i + 1] += bw * a.y;
                    wf[8 + 2 * i] += bw * b.x; wf[8 + 2 * i + 1] += bw * b.y;
                }
            }
        }
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) dot += wf[i] * cf[i];
        dot *= maskf;
        const float dn = (zd - mn) / mx;
        float sx = wx - g[12], sy = wy - g[13], sz = wz - g[14];
        const float sn = fmaxf(sqrtf(sx * sx + sy * sy + sz * sz), 1e-12f);
        sx /= sn; sy /= sn; sz /= sn;
        const float ang = cx * sx + cy * sy + cz * sz;
        float in[NDYN];
#pragma unroll
        for (int i = 0; i < 16; i++) in[i] = wf[i];
        in[16] = maskf; in[17] = dn; in[18] = dot; in[19] = ang; in[20] = sx; in[21] = sy; in[22] = sz;
        float h1[NH];
#pragma unroll
        for (int j = 0; j < NH; j++) {
            float t = a1[j] + sB[k * NH + j];
#pragma unroll
            for (int i = 0; i < NDYN; i++) t += wt.w1[j * NDYN + i] * in[i];
            h1[j] = lrelu(t);
        }
        float o0 = wt.b3[0], o1 = wt.b3[1];
#pragma unroll
        for (int j = 0; j < NH; j++) {
            float t = wt.b2[j];
#pragma unroll
            for (int i = 0; i < NH; i++) t += wt.w2[j * NH + i] * h1[i];
            t = lrelu(t);
            o0 += wt.w3[j] * t;
            o1 += wt.w3[NH + j] * t;
        }
        const float mnew = fmaxf(m, o1);
        const float sc_old = __expf(m - mnew), e = __expf(o1 - mnew);
        s = s * sc_old + e;
        acc = acc * sc_old + e * o0;
        m = mnew;
    }
    out[(size_t)di * hw + p] = acc / s;
}

// Antialiased-ResNet stem tail, fused: ReLU -> MaxPool(2, stride 1) -> ReflectionPad(1, 2, 1, 2) -> BlurPool([1,3,3,1]^2/64,
// stride 2). Channels-last fp16 in [N][H][W][C] (conv1 output), out [N][Ho][Wo][C]; one thread per output channel pair.
__device__ __forceinline__ int refl(int i, int n) { return i < 0 ? -i : (i > n - 1 ? 2 * (n - 1) - i : i); }

extern "C" __global__ void __launch_bounds__(256) mvsa_stem_pool(
    const __half2* __restrict__ x, __half2* __restrict__ out, int N, int H, int W, int C2, int Ho, int Wo)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)N * Ho * Wo * C2;
    if (idx >= total) return;
    const int c = idx % C2;
    long long r = idx / C2;
    const int ox = r % Wo; r /= Wo;
    const int oy = r % Ho;
    const int n = r / Ho;
    const int Hm = H - 1, Wm = W - 1;          // max-pooled size
    const float f[4] = {0.125f, 0.375f, 0.375f, 0.125f};
    const __half2* xn = x + (size_t)n * H * W * C2 + c;
    float ax = 0.f, ay = 0.f;
#pragma unroll
    for (int ty = 0; ty < 4; ty++) {
        const int i = refl(2 * oy + ty - 1, Hm);
        float rx = 0.f, ry = 0.f;
#pragma unroll
        for (int tx = 0; tx < 4; tx++) {
            const int j = refl(2 * ox + tx - 1, Wm);
            const __half2* q = xn + ((size_t)i * W + j) * C2;
            __half2 m2 = __hmax2(__hmax2(q[0], q[C2]), __hmax2(q[(size_t)W * C2], q[(size_t)W * C2 + C2]));
            const float2 v = __half22float2(m2);
            rx += f[tx] * fmaxf(v.x, 0.f);
            ry += f[tx] * fmaxf(v.y, 0.f);
        }
        ax += f[ty] * rx;
        ay += f[ty] * ry;
    }
    out[idx] = __floats2half2_rn(ax, ay);
}
