"""IS-Net on Apple MLX, hand-written: NHWC, the run dtype.

`ISNetDIS.forward` up to the first side output `d1`: the strided stem conv, the residual U-blocks (`shared.mlx_rsu`,
BatchNorms folded), the 2 x 2 ceil-mode pools, the bilinear `_upsample_like`s and `side1`, resized to the input
(bilinear, float32) and its sigmoid. The other side outputs are not computed (the runner reads `d1` only). Each
encoder stage is a step, the decoder another (`cspnext.mlx_net.stepped`: a whole-network graph held every stage's
transients at 1024 x 1024). Kernels mode: float16 3 x 3 convs through `conv_mlx`'s Winograd kernel, the stem
through its direct few-channel conv.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastforks._core.mlx import mx
from fastforks.shared import mlx_matting, mlx_rsu
from fastforks.shared.cspnext import mlx_net as csp

from .net import isnet as tnet

LAYOUT = "isnet-v1"
SOURCES = (Path(__file__), Path(tnet.__file__), Path(mlx_rsu.__file__))
STAGES = 6


class Network:
    """`sigmoid(d1)` of `ISNetDIS.forward`: normalised `[B, 3, S, S]` float32 -> `[B, 1, S, S]` float32."""

    def __init__(self, net: Any, dtype: str, *, kernels: bool = False) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        self.kernels = kernels
        stem = mlx_rsu.conv(net.conv_in, None, dt, kernels=kernels)
        enc = [mlx_rsu.rsu(getattr(net, f"stage{i}"), dt, kernels=kernels) for i in range(1, STAGES + 1)]
        dec = [mlx_rsu.rsu(getattr(net, f"stage{i}d"), dt, kernels=kernels) for i in range(1, STAGES)]
        side = mlx_rsu.conv(net.side1, None, dt, kernels=kernels)

        self.stem = csp.stepped(lambda x: enc[0](stem(x.transpose(0, 2, 3, 1).astype(dt))))
        self.enc = [csp.stepped(lambda h, s=s: s(mlx_rsu.pool2(h))) for s in enc[1:]]

        def decode(x: mx.array, *feats: mx.array) -> mx.array:
            d = feats[-1]
            for i in range(STAGES - 2, -1, -1):
                d = dec[i](mx.concatenate([mlx_rsu.up_like(d, feats[i].shape[1:3]), feats[i]], axis=-1))
            d1 = side(d).astype(mx.float32).transpose(0, 3, 1, 2)
            return mx.sigmoid(mlx_matting.resize(d1, x.shape[2:]))

        self.decode = csp.stepped(decode)

    def forward(self, x: mx.array) -> mx.array:
        """The network on `[B, 3, S, S]` float32 (the steps)."""
        feats = [self.stem(x)]
        for stage in self.enc:
            feats.append(stage(feats[-1]))
        return self.decode(x, *feats)

    def __call__(self, x: mx.array) -> mx.array:
        """`forward` with its steps, evaluated."""
        mlx_rsu.options(kernels=self.kernels)
        y = csp.steps(self.forward, x)
        mx.eval(y)
        return y


def network(net: Any, dtype: str, *, kernels: bool = False) -> Network:
    """The MLX network of an `ISNetDIS` (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype, kernels=kernels)
