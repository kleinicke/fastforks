"""U²-Net on Apple MLX, hand-written: NHWC, the run dtype.

`U2NET` / `U2NETP` `forward` up to the fused map `d0`: the residual U-blocks (`shared.mlx_rsu`, BatchNorms folded),
the 2 x 2 ceil-mode pools, the bilinear `_upsample_like`s, the six side convs and the 1 x 1 fusion; the sigmoid in
float32. The six auxiliary side sigmoids are not computed (the runner reads `d0` only).
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastforks._core.mlx import mx
from fastforks.shared import mlx_rsu

from .net import u2net as tnet

LAYOUT = "u2net-v1"
SOURCES = (Path(__file__), Path(tnet.__file__), Path(mlx_rsu.__file__))
STAGES = 6


class Network:
    """`U2NET.forward`'s `sigmoid(d0)`: normalised `[B, 3, S, S]` float32 -> `[B, S, S, 1]` float32."""

    def __init__(self, net: Any, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = self.dtype = getattr(mx, dtype)
        enc = [mlx_rsu.rsu(getattr(net, f"stage{i}"), dt) for i in range(1, STAGES + 1)]
        dec = [mlx_rsu.rsu(getattr(net, f"stage{i}d"), dt) for i in range(1, STAGES)]
        side = [mlx_rsu.conv(getattr(net, f"side{i}"), None, dt) for i in range(1, STAGES + 1)]
        out = mlx_rsu.conv(net.outconv, None, dt)

        def forward(x: mx.array) -> mx.array:
            h = x.transpose(0, 2, 3, 1).astype(dt)
            feats = []
            for i, stage in enumerate(enc):
                h = stage(h)
                feats.append(h)
                if i < STAGES - 1:
                    h = mlx_rsu.pool2(h)
            d = feats[-1]
            ds = [d]  # hx6, hx5d, ..., hx1d
            for i in range(STAGES - 2, -1, -1):
                d = dec[i](mx.concatenate([mlx_rsu.up_like(d, feats[i].shape[1:3]), feats[i]], axis=-1))
                ds.append(d)
            hw = ds[-1].shape[1:3]
            maps = [mlx_rsu.up_like(side[i](ds[STAGES - 1 - i]), hw) for i in range(STAGES)]
            d0 = out(mx.concatenate(maps, axis=-1))
            return mx.sigmoid(d0.astype(mx.float32))

        self.forward = mx.compile(forward)

    def __call__(self, x: mx.array) -> mx.array:
        """The fused map of `[B, 3, S, S]` float32 (NCHW), evaluated."""
        mlx_rsu.options()
        y = self.forward(x)
        mx.eval(y)
        return y


def network(net: Any, dtype: str) -> Network:
    """The MLX network of a `U2NET` / `U2NETP` (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype)
