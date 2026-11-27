"""Real-ESRGAN Compact (`SRVGGNetCompact`) on Apple MLX, hand-written (`fastforks.shared.swin_sr.mlx_cnn`'s convs).

The network of `model.Portable`: 3 x 3 conv + PReLU, `num_conv` more, the last conv to `3 * s * s` channels plus the
nearest-upsampled input, pixel-shuffled (the skip added before the shuffle: each input channel repeated `s * s` times
is the shuffled nearest upsampling). NHWC in the run dtype, one compiled step per conv. Kernels mode: the convs through
`conv_mlx`'s Winograd kernel with the PReLU and the last conv's skip fused, as the graph translation's kernels mode.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared.swin_sr import mlx_cnn as cnn
from fastforks.shared.swin_sr import mlx_net as swin

if TYPE_CHECKING:
    from collections.abc import Callable

    from .net import SRVGGNetCompact

LAYOUT = "compact-v1"
SOURCES = (Path(__file__), Path(cnn.__file__), Path(swin.__file__))


@functools.lru_cache(maxsize=4)
def _steps(scale: int) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """The compiled conv + PReLU step and the last conv + skip + shuffle (to NCHW float32) step."""

    def body(x: mx.array, p: dict[str, mx.array], slope: mx.array) -> mx.array:
        return cnn.conv(x, p, slope=slope)

    def last(y: mx.array, x: mx.array, p: dict[str, mx.array]) -> mx.array:
        y = swin.pixel_shuffle(cnn.conv(y, p, residual=mx.repeat(x, scale * scale, axis=-1)), scale)
        return y.transpose(0, 3, 1, 2).astype(mx.float32)  # in the step: the NHWC output is not held beside it

    return swin.stepped(body), swin.stepped(last)


class Network:
    """`SRVGGNetCompact.forward` (PReLU): `[B, 3, H, W]` float32 in `[0, 1]` -> `[B, 3, sH, sW]` float32."""

    def __init__(self, net: SRVGGNetCompact, dtype: str, *, kernels: bool) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        from torch import nn  # noqa: PLC0415 - the module types of a built network

        dt = getattr(mx, dtype)
        self.dtype, self.kernels, self.scale = dt, kernels, int(net.upscale)
        mods = list(net.body)
        if not all(isinstance(a, nn.PReLU) for a in mods[1:-1:2]):
            msg = f"Compact MLX: PReLU activations, got {sorted({type(a).__name__ for a in mods[1:-1:2]})}"
            raise NotImplementedError(msg)
        self.body = [
            (cnn.params(c, dt, wino=kernels), swin.arr(a.weight, dt))
            for c, a in zip(mods[:-1:2], mods[1::2], strict=True)
        ]
        self.last = cnn.params(mods[-1], dt, wino=kernels)

    def forward(self, x: mx.array) -> mx.array:
        """The network on `[B, 3, H, W]` float32 (the steps)."""
        body, last = _steps(self.scale)
        x = x.transpose(0, 2, 3, 1).astype(self.dtype)
        y = x
        for p, slope in self.body:
            y = body(y, p, slope)
        return last(y, x, self.last)

    def __call__(self, x: mx.array) -> mx.array:
        """`forward` with its steps, evaluated."""
        cnn.set_mode(kernels=self.kernels)
        out = swin.steps(self.forward, x)
        mx.eval(out)
        return out


def network(net: SRVGGNetCompact, dtype: str, *, kernels: bool) -> Network:
    """The MLX network of an `SRVGGNetCompact` (float32 or the meta skeleton) in `dtype`."""
    return Network(net, dtype, kernels=kernels)
