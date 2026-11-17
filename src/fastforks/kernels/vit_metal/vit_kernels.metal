// Vision transformers on Apple MPS (fastforks): Perception Encoder's interleaved rotary position embedding.
#include <metal_stdlib>
using namespace metal;

// q, k of qkv [B, N, 3, H, D] (contiguous) -> out [2, B, H, N, D] (q, k; contiguous for the attention): the first npt
// (prefix) tokens copied, the others rotated pairwise with the float32 table rope [N - npt, 2 D] (sin | cos, each
// repeated per pair as timm's `apply_rot_embed_cat`): (x0, x1) -> (x0 cos - x1 sin, x1 cos + x0 sin), in float32,
// rounded once to T (timm's `.type_as`). One thread per channel pair.
template <typename T>
kernel void pe_rope(device T* out [[buffer(0)]], device const T* qkv [[buffer(1)]], device const float* rope [[buffer(2)]],
                    constant uint& B [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& H [[buffer(5)]],
                    constant uint& D [[buffer(6)]], constant uint& npt [[buffer(7)]],
                    uint gid [[thread_position_in_grid]]) {
    const uint pairs = D / 2;
    if (gid >= 2 * B * H * N * pairs) return;
    const uint d = 2 * (gid % pairs), n = (gid / pairs) % N, h = (gid / (pairs * N)) % H,
               b = (gid / (pairs * N * H)) % B, s = gid / (pairs * N * H * B);
    device const T* src = qkv + ((ulong(b) * N + n) * 3 + s) * H * D + h * D + d;
    device T* dst = out + (((ulong(s) * B + b) * H + h) * N + n) * D + d;
    if (n < npt) {
        dst[0] = src[0];
        dst[1] = src[1];
        return;
    }
    device const float* r = rope + ulong(n - npt) * 2 * D;  // sin at r[d], cos at r[D + d]
    const float x0 = float(src[0]), x1 = float(src[1]);
    dst[0] = T(x0 * r[D + d] - x1 * r[d]);
    dst[1] = T(x1 * r[D + d + 1] + x0 * r[d + 1]);
}

#define INSTANTIATE(T)                                                                                                  \
    template [[host_name("pe_rope_" #T)]] kernel void pe_rope<T>(device T*, device const T*, device const float*,      \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint);
INSTANTIATE(float)
INSTANTIATE(half)
