"""DPT heads on Apple MLX, kernels mode: Metal kernels (`mx.fast.metal_kernel`) around the convolutions.

- `resize`: `F.interpolate(mode="bilinear", align_corners=True)` of a channels-last map in one pass (four taps per
  output value, read in the storage dtype, interpolated in float32). MLX's own operations need two gathers and a
  blend per axis, which costs several times the memory traffic at the head's two finest levels.

Metal 3-level MSL that every Apple-silicon GPU runs. Threadgroup size and lane come from the tunables table
(`fastforks._core.mlx.TUNING`).
"""
# ruff: noqa: ANN001, ANN201 - array-in / array-out functions named after the portable ops they replace

from __future__ import annotations

import math

from fastforks._core.mlx import metal_kernel, mx, tuning

# one thread per (output pixel, lane of L channels)
_RESIZE = metal_kernel(
    name="dpt_resize",
    input_names=["x", "dims"],
    output_names=["out"],
    source="""
        uint H = x_shape[1], W = x_shape[2], C = x_shape[3], OH = dims[0], OW = dims[1];
        uint lanes = C / L, tid = thread_position_in_grid.x, lane = tid % lanes, pix = tid / lanes;
        uint ox = pix % OW, oy = (pix / OW) % OH, n = pix / (OW * OH);
        float sy = OH > 1 ? float(H - 1) / float(OH - 1) : 0.0f, sx = OW > 1 ? float(W - 1) / float(OW - 1) : 0.0f;
        float fy = float(oy) * sy, fx = float(ox) * sx;
        uint y0 = min(uint(fy), H - 1), x0 = min(uint(fx), W - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        float wy = fy - float(y0), wx = fx - float(x0);
        uint c0 = lane * L, base = n * H * W;
        uint a = (base + y0 * W + x0) * C + c0, b = (base + y0 * W + x1) * C + c0;
        uint c = (base + y1 * W + x0) * C + c0, d = (base + y1 * W + x1) * C + c0, o = pix * C + c0;
        for (uint k = 0; k < L; ++k) {
            float top = mix(float(x[a + k]), float(x[b + k]), wx), bottom = mix(float(x[c + k]), float(x[d + k]), wx);
            out[o + k] = T(mix(top, bottom, wy));
        }
    """,
)


def resize(x, h, w):
    """`[B, H, W, C]` -> `[B, h, w, C]`, bilinear with `align_corners=True` (as the portable `resize`)."""
    n, h0, w0, c = x.shape
    if (h0, w0) == (h, w):
        return x
    if n > 1 and max(h * w, h0 * w0) * c * n >= 2**32:  # the kernel indexes with 32-bit `uint`
        half = n // 2
        return mx.concatenate([resize(x[:half], h, w), resize(x[half:], h, w)])
    lane = math.gcd(c, tuning().warp_lane)
    return _RESIZE(inputs=[x, mx.array([h, w], dtype=mx.uint32)], template=[("T", x.dtype), ("L", lane)],
                   grid=(n * h * w * (c // lane), 1, 1), threadgroup=(tuning().threadgroup, 1, 1),
                   output_shapes=[(n, h, w, c)], output_dtypes=[x.dtype])[0]  # fmt: skip
