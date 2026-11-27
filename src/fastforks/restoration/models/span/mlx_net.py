"""SPAN on Apple MLX, hand-written (`fastforks.shared.swin_sr.mlx_cnn`'s convs): NHWC, the run dtype.

The network of `model.Portable` (every `Conv3XC` its fused 3 x 3 `eval_conv`): `(x - mean) * 255`, `conv_1`, six SPABs
(`c1` -> SiLU -> `c2` -> SiLU -> `c3`, then `(c3 + x) * (sigmoid(c3) - 0.5)`), `conv_2` of the last, the 1 x 1
`conv_cat` over `[conv_1, conv_2, block 1, block 6's SiLU(c1)]`, the pixel-shuffle upsampler. Each conv is one
compiled step (`_steps`). Kernels mode: the wide 3 x 3 convs through `conv_mlx`'s Winograd kernel, as the graph
translation's kernels mode.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared.swin_sr import mlx_cnn as cnn
from fastforks.shared.swin_sr import mlx_net as swin

if TYPE_CHECKING:
    from collections.abc import Callable

    from .net import SPAN

LAYOUT = "span-v1"
SOURCES = (Path(__file__), Path(cnn.__file__), Path(swin.__file__))

_Conv = dict[str, mx.array]


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


@functools.lru_cache(maxsize=4)
def _steps(scale: int) -> tuple[Callable[..., Any], ...]:
    """The compiled steps, one conv each (`scale`: the pixel shuffle's).

    The head, a conv + SiLU, a SPAB's last conv and gate, the tail's `conv_2`, `conv_cat` and the upsampler.
    A conv per step: with the next step queued while one runs, two steps' transients are held, and a whole SPAB per
    step held the network peak 1.6 times the graph translation's.
    """

    def head(x: mx.array, mean: mx.array, p: _Conv, rng: mx.array) -> mx.array:
        return cnn.conv(((x - mean) * rng).astype(p["b"].dtype), p)

    def conv_silu(x: mx.array, p: _Conv) -> mx.array:
        return _silu(cnn.conv(x, p))

    def gate(t: mx.array, x: mx.array, p: _Conv) -> mx.array:
        c3 = cnn.conv(t, p)
        return (c3 + x) * (mx.sigmoid(c3) - 0.5)

    def cat(f: mx.array, b6: mx.array, b1: mx.array, tap: mx.array, p: _Conv) -> mx.array:
        return cnn.conv(mx.concatenate([f, b6, b1, tap], axis=-1), p)

    def up(y: mx.array, p: _Conv) -> mx.array:
        return swin.pixel_shuffle(cnn.conv(y, p), scale).transpose(0, 3, 1, 2).astype(mx.float32)  # NCHW in the step

    return tuple(swin.stepped(f) for f in (head, conv_silu, gate, cnn.conv, cat, up))


class Network:
    """`SPAN.forward`: `[B, 3, H, W]` float32 in `[0, 1]` -> `[B, 3, sH, sW]` float32."""

    def __init__(self, net: SPAN, dtype: str, *, kernels: bool) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        self.dtype, self.kernels = dt, kernels

        def c(m: Any) -> _Conv:  # a fused Conv3XC or a plain conv
            return cnn.params(getattr(m, "eval_conv", m), dt, wino=kernels)

        self.first = c(net.conv_1)
        self.blocks = [[c(b.c1_r), c(b.c2_r), c(b.c3_r)] for b in
                       (net.block_1, net.block_2, net.block_3, net.block_4, net.block_5, net.block_6)]  # fmt: skip
        self.tail_p = {"conv_2": c(net.conv_2), "cat": c(net.conv_cat), "up": c(net.upsampler[0])}
        self.mean = swin.arr(net.mean.reshape(-1), mx.float32)
        self.rng = mx.array(float(net.img_range), mx.float32)
        self.scale = int(net.upsampler[1].upscale_factor)

    def forward(self, x: mx.array) -> mx.array:
        """The network on `[B, 3, H, W]` float32 (the steps)."""
        head, conv_silu, gate, conv, cat, up = _steps(self.scale)
        f = head(x.transpose(0, 2, 3, 1), self.mean, self.first, self.rng)
        y, b1, tap = f, None, f
        for p in self.blocks:
            tap = conv_silu(y, p[0])
            y = gate(conv_silu(tap, p[1]), y, p[2])
            b1 = y if b1 is None else b1
        t = self.tail_p
        y = cat(f, conv(y, t["conv_2"]), b1, tap, t["cat"])
        return up(y, t["up"])

    def __call__(self, x: mx.array) -> mx.array:
        """`forward` with its steps, evaluated."""
        cnn.set_mode(kernels=self.kernels)
        out = swin.steps(self.forward, x)
        mx.eval(out)
        return out


def network(net: SPAN, dtype: str, *, kernels: bool) -> Network:
    """The MLX network of a SPAN (float32 with its `Conv3XC`s fused, or the meta skeleton) in `dtype`."""
    return Network(net, dtype, kernels=kernels)
