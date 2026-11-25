"""AnyLine on Apple MLX (`model._MlxPortable`): MTEED (`shared.teed.mlx_net`) and the line art in Metal kernels.

`model.line_art` on the CPU twin was half of the call at 1280 px (85 of 165 ms: the 17-tap blur 27, the sort behind
the median 14, the label propagation of `remove_small` 41). Here:

- the blur and `min_c(blur - x)` are two separable passes (`anyline_blur_rows`, `anyline_intensity`);
- the median of the values above the threshold is NumPy's selection (`np.partition`) on the valid values, exact;
- `anyline_blend` thresholds, drops the 4-connected specks and blends with the base map in one pass: every drawn
  pixel walks its component breadth-first until it has `MIN_PIXELS` pixels (kept) or runs out (a speck), the same
  result as `remove_small` (and `edge_components.keep_twelve` on CUDA).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fastforks._core.mlx import metal_kernel, mx
from fastforks.shared.teed import mlx_net as teed

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.shared.teed import TED

SIGMA, TAPS, THRESHOLD, MIN_PIXELS = 2.0, 17, 3, 12  # as `model` (which imports torch-only helpers this must not)


def _header() -> str:
    t = np.arange(TAPS, dtype=np.float64) - TAPS // 2
    k = np.exp(-(t**2) / (2 * SIGMA**2))
    k = (k / k.sum()).astype(np.float32)
    taps = ", ".join(f"{float(v)!r}f" for v in k)
    return f"""
constant float G[{TAPS}] = {{{taps}}};
constant int TAPS = {TAPS}, MIN_PIXELS = {MIN_PIXELS};
inline int reflect101(int i, int n) {{ return i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i); }}
"""


_ROWS = metal_kernel(
    name="anyline_blur_rows",
    input_names=["x"],
    output_names=["out"],
    header=_header(),
    source="""
        // the horizontal pass of the Gaussian over an [N, H, W, 3] float32 image, reflect-101 border
        int H = x_shape[1], W = x_shape[2];
        int px = thread_position_in_grid.x, py = thread_position_in_grid.y, n = thread_position_in_grid.z;
        if (px >= W || py >= H) return;
        size_t row = (size_t(n) * H + py) * W;
        float3 s = 0;
        for (int k = 0; k < TAPS; ++k) {
            size_t o = (row + reflect101(px + k - TAPS / 2, W)) * 3;
            s += G[k] * float3(x[o], x[o + 1], x[o + 2]);
        }
        size_t o = (row + px) * 3;
        out[o] = s.x; out[o + 1] = s.y; out[o + 2] = s.z;
    """,
)
_INTENSITY = metal_kernel(
    name="anyline_intensity",
    input_names=["rows", "x"],
    output_names=["out"],
    header=_header(),
    source="""
        // the vertical pass, then min over the channels of (blur - x), clipped to [0, 255]
        int H = x_shape[1], W = x_shape[2];
        int px = thread_position_in_grid.x, py = thread_position_in_grid.y, n = thread_position_in_grid.z;
        if (px >= W || py >= H) return;
        float3 s = 0;
        for (int k = 0; k < TAPS; ++k) {
            size_t o = ((size_t(n) * H + reflect101(py + k - TAPS / 2, H)) * W + px) * 3;
            s += G[k] * float3(rows[o], rows[o + 1], rows[o + 2]);
        }
        size_t p = (size_t(n) * H + py) * W + px;
        float3 d = s - float3(x[p * 3], x[p * 3 + 1], x[p * 3 + 2]);
        out[p] = clamp(min(d.x, min(d.y, d.z)), 0.0f, 255.0f);
    """,
)
_BLEND = metal_kernel(
    name="anyline_blend",
    input_names=["v", "med", "base"],
    output_names=["out"],
    header=_header() + "#define LEVEL(i) floor(clamp(v[i] / m * 127.0f, 0.0f, 255.0f))\n",
    source="""
        // line art level of the intensity `v` (per-image divisor `med`), kept where its 4-connected component of
        // drawn pixels has at least MIN_PIXELS pixels, screened over `base`
        int H = v_shape[1], W = v_shape[2];
        int px = thread_position_in_grid.x, py = thread_position_in_grid.y, n = thread_position_in_grid.z;
        if (px >= W || py >= H) return;
        size_t image = size_t(n) * H * W;
        float m = med[n];
        int self = py * W + px;
        float level = LEVEL(image + self), b = base[image + self];
        if (!(level > 0.0f)) { out[image + self] = b; return; }
        int seen[MIN_PIXELS];
        int count = 1;
        seen[0] = self;
        for (int i = 0; i < count && count < MIN_PIXELS; ++i) {
            int cy = seen[i] / W, cx = seen[i] % W;
            for (int d = 0; d < 4 && count < MIN_PIXELS; ++d) {
                int ny = cy + (d == 0) - (d == 1), nx = cx + (d == 2) - (d == 3);
                if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                int q = ny * W + nx;
                if (!(LEVEL(image + q) > 0.0f)) continue;
                bool known = false;
                for (int j = 0; j < count; ++j) known = known || seen[j] == q;
                if (!known) seen[count++] = q;
            }
        }
        float top = level / 255.0f;
        out[image + self] = count >= MIN_PIXELS ? 1.0f - (1.0f - top) * (1.0f - b) : b;
    """,
)


def _per_pixel(kernel: Callable[..., list[mx.array]], inputs: list[mx.array], shape: tuple[int, ...]) -> mx.array:
    n, h, w = shape[:3]
    return kernel(
        inputs=inputs, grid=(w, h, n), threadgroup=(16, 16, 1), output_shapes=[shape], output_dtypes=[mx.float32]
    )[0]


def _divisor(intensity: mx.array) -> mx.array:
    """Per image `max(16, median of the values above THRESHOLD)` (16 without any), as `model.median_above`."""
    out = []
    for v in np.asarray(intensity).reshape(intensity.shape[0], -1):
        a = v[v > THRESHOLD]
        if a.size:
            lo, hi = (a.size - 1) // 2, a.size // 2
            part = np.partition(a, (lo, hi))
            out.append(max(np.float32(16), (part[lo] + part[hi]) / np.float32(2)))
        else:
            out.append(np.float32(16))
    return mx.array(np.asarray(out, dtype=np.float32))


class MlxAnyline:
    """`model.anyline(net(x), x)` on MLX: NHWC `[B, h, w, 3]` float32 levels 0..255 -> the map `[B, h, w]` float32."""

    def __init__(self, net: TED, dtype: str) -> None:
        """Convert the weights of the float32 `net` to `dtype` (the name)."""
        self.net = teed.MlxTED(net, getattr(mx, dtype))
        self._first = mx.compile(self._maps)
        self._blend = mx.compile(lambda v, med, base: _per_pixel(_BLEND, [v, med, base], v.shape))

    def _maps(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """The base map (`safe_step(edge_map, 2)`) and the line art intensity of `x`."""
        edge = teed.edge_map(self.net(x))
        base = mx.minimum(mx.floor(edge * 3) / 2, 1)
        rows = _per_pixel(_ROWS, [x], x.shape)
        return base, _per_pixel(_INTENSITY, [rows, x], x.shape[:3])

    def __call__(self, x: mx.array) -> mx.array:
        """The AnyLine map of the detect-size image `x`."""
        base, intensity = self._first(x)
        mx.eval(base, intensity)
        return self._blend(intensity, _divisor(intensity), base)
