"""The line art generator in `mlx.core` operations (Apple MLX; `model._MlxPortable`).

Written out instead of replayed from an export (`shared/mlx_graph`): the export alone took 0.7 s of the first call
(0.98 s against 0.24 s on MPS), and three things of this network run better by hand:

- the two stride-2 transposed convolutions are four small convolutions, one per output phase, interleaved
  (`mx.conv_transpose2d` convolves the zero-stuffed map: 36 taps per input pixel instead of 9);
- every instance norm takes its statistics in float32 in one pass (`graph_mlx.channel_stats`) and applies them, the
  ReLU and the cast back in one fused expression;
- the convolution biases in front of an instance norm are left out (the norm subtracts them again);
- the 7x7 head (64 -> 1 channel, 8.2 ms as one `mx.conv2d` at 1024 px) is a 1x7 convolution to seven channels, one
  per kernel row, and the sum of the seven rows each shifted by its offset (1.6 ms).

Channels-last throughout; the result is `1 - sigmoid` (the runner's line map) in float32. Each step is one compiled
function, evaluated one in flight: MLX frees a command buffer's transients only when it completes, so one compiled
forward held every step's maps to the end.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_port
from fastforks.shared.mlx_graph import _edge_pad

if TYPE_CHECKING:
    from collections.abc import Callable

    from .net.generator import Generator

LAYOUT = "lineart-v2"
"""The conversion cache layout of `MlxGenerator`'s arrays (`fastforks.shared.mlx_port.recorded`)."""

SOURCES = (Path(__file__), Path(__file__).with_name("net") / "generator.py")
"""The code the cached arrays come from: this builder and the network."""

EPS = 1e-5  # nn.InstanceNorm2d's default
_UP_TAPS = ((1,), (2, 0))  # output phase -> the kernel taps read at input offsets 0 (and 1) (stride 2, padding 1)


def _tensor(t: Any, dtype: mx.Dtype) -> mx.array:  # noqa: ANN401 - a torch tensor
    """`t` in `dtype` as it is laid out in torch, or its cached array (`mlx_port.tensor`)."""

    def convert() -> mx.array:
        a = mx.array(np.ascontiguousarray(t.detach().float().numpy())).astype(dtype)
        mx.eval(a)
        return a

    return mlx_port.tensor(t, dtype, convert)


def _ohwi(conv: Any, dtype: mx.Dtype) -> mx.array:  # noqa: ANN401 - a torch module
    return mx.contiguous(_tensor(conv.weight, dtype).transpose(0, 2, 3, 1))


def _up_kernels(conv: Any, dtype: mx.Dtype) -> list[mx.array]:  # noqa: ANN401 - a torch module
    """The four phase kernels (OHWI) of a `ConvTranspose2d(3, stride 2, padding 1, output_padding 1)`.

    Output `2m + r` reads kernel tap 1 of input `m` for `r = 0`, taps 2 and 0 of inputs `m` and `m + 1` for `r = 1`:
    the phases `(0, 0), (0, 1), (1, 0), (1, 1)` are a 1 x 1, a 1 x 2, a 2 x 1 and a 2 x 2 convolution.
    """
    w = _tensor(conv.weight, dtype)  # [Cin, Cout, 3, 3]
    out = []
    for taps_h in _UP_TAPS:
        for taps_w in _UP_TAPS:
            k = mx.take(mx.take(w, mx.array(taps_h), axis=2), mx.array(taps_w), axis=3)  # [Cin, Cout, kh, kw]
            out.append(mx.contiguous(k.transpose(1, 2, 3, 0)))
    return out


def _norm(x: mx.array, *, relu: bool) -> mx.array:
    """Instance norm (no affine) of an NHWC map with float32 statistics, the ReLU fused."""
    n, h, w, c = x.shape
    stats = [ops.channel_stats(x[i].reshape(h * w, c)) for i in range(n)]
    mean = mx.stack([s[0] for s in stats]).reshape(n, 1, 1, c)
    scale = mx.rsqrt(mx.stack([s[1] for s in stats]).reshape(n, 1, 1, c) + EPS)
    y = (x.astype(mx.float32) - mean) * scale
    return (mx.maximum(y, 0) if relu else y).astype(x.dtype)


def _up(x: mx.array, kernels: list[mx.array]) -> mx.array:
    n, h, w, _ = x.shape
    phases = []
    for k in kernels:
        pad = [(0, 0), (0, k.shape[1] - 1), (0, k.shape[2] - 1), (0, 0)]
        phases.append(mx.conv2d(mx.pad(x, pad) if k.shape[1] * k.shape[2] > 1 else x, k))
    y = mx.transpose(mx.stack(phases, axis=3).reshape(n, h, w, 2, 2, -1), (0, 1, 3, 2, 4, 5))
    return y.reshape(n, 2 * h, 2 * w, -1)


class MlxGenerator:
    """`Generator.forward` on MLX: `[B, H, W, 3]` float32 in `[0, 1]` -> the line map `[B, H, W]` float32."""

    def __init__(self, net: Generator, dtype: str) -> None:
        """Convert the weights of the float32 `net` to `dtype` (the name), each read through `_tensor`."""
        self.dtype = run = getattr(mx, dtype)
        g: Any = net
        self.stem = _ohwi(g.model0[1], run)
        self.down = [_ohwi(g.model1[i], run) for i in (0, 3)]
        self.blocks = [(_ohwi(b.conv_block[1], run), _ohwi(b.conv_block[5], run)) for b in g.model2]
        self.up = [_up_kernels(g.model3[i], run) for i in (0, 3)]
        head = _ohwi(g.model4[1], run)  # [1, 7, 7, 64]
        self.head = mx.concatenate([head[:, i : i + 1] for i in range(7)], axis=0)  # [7, 1, 7, 64]: a row each
        self.head_bias = _tensor(g.model4[1].bias, mx.float32)
        mx.eval(self.stem, self.down, self.blocks, self.up, self.head, self.head_bias)
        self._steps = [mx.compile(step) for _, step in self.stages()]

    def stages(self) -> list[tuple[str, Callable[[mx.array], mx.array]]]:
        """The forward pass as named steps (also what a profile times)."""

        def block(wa: mx.array, wb: mx.array) -> Callable[[mx.array], mx.array]:
            def run(x: mx.array) -> mx.array:
                y = _norm(mx.conv2d(_edge_pad(x, (1, 1, 1, 1), reflect=True, last=1), wa), relu=True)
                return x + _norm(mx.conv2d(_edge_pad(y, (1, 1, 1, 1), reflect=True, last=1), wb), relu=False)

            return run

        def head(x: mx.array) -> mx.array:
            h = x.shape[1]
            rows = mx.conv2d(_edge_pad(x, (3, 3, 3, 3), reflect=True, last=1), self.head).astype(mx.float32)
            y = sum((rows[:, i : i + h, :, i] for i in range(1, 7)), rows[:, :h, :, 0])
            return mx.sigmoid(-(y + self.head_bias))  # 1 - sigmoid(v)

        def stem(x: mx.array) -> mx.array:
            x = _edge_pad(x.astype(self.dtype), (3, 3, 3, 3), reflect=True, last=1)
            return _norm(mx.conv2d(x, self.stem), relu=True)

        def down(w: mx.array) -> Callable[[mx.array], mx.array]:
            return lambda x: _norm(mx.conv2d(x, w, stride=2, padding=1), relu=True)

        def up(kernels: list[mx.array]) -> Callable[[mx.array], mx.array]:
            return lambda x: _norm(_up(x, kernels), relu=True)

        return [
            ("stem", stem),
            *((f"down{i}", down(w)) for i, w in enumerate(self.down)),
            *((f"block{i}", block(wa, wb)) for i, (wa, wb) in enumerate(self.blocks)),
            *((f"up{i}", up(k)) for i, k in enumerate(self.up)),
            ("head", head),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Line map `[B, H, W]` (1 = line) of the NHWC image `x`."""
        prev = None
        for step in self._steps:
            x = step(x)
            mx.async_eval(x)
            if prev is not None:
                mx.eval(prev)
            prev = x
        return x
