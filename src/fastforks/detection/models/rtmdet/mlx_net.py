"""RTMDet on Apple MLX, written in `mlx.core` operations (`MlxRTMDet(net, dtype)`): normalised canvases -> head maps.

The maths of `net.RTMDet.forward`, channels last, on the network with its batch norms folded: the CSPNeXt backbone and
the PAFPN of `fastforks.shared.cspnext.mlx_net` (SiLU in float32), the neck's output convs and the head's towers in
the run dtype, the maps returned float32. Written out instead of replayed from an exported graph: the export and its
translation were most of the first call (1.3-2.3 s).
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastforks._core.mlx import mx
from fastforks.shared.cspnext import mlx_net as csp

__all__ = ["LAYOUT", "SOURCES", "MlxRTMDet"]

LAYOUT = "rtmdet-v1"
"""The conversion cache layout of `MlxRTMDet`'s arrays (`fastforks.shared.mlx_port.recorded`)."""

SOURCES = (Path(__file__), Path(csp.__file__), Path(__file__).with_name("net.py"))
"""The code the cached arrays come from: this builder, the shared layers, the checkpoint conversion."""


class MlxRTMDet:
    """Normalised NHWC canvases (float32) -> float32 (class logits, raw box output) NHWC maps per level, flat."""

    def __init__(self, net: Any, dtype: str) -> None:
        """The MLX layers of `net` (an `RTMDet` with its batch norms folded) in `dtype`."""
        d = mx.float16 if dtype == "float16" else mx.float32
        backbone = csp.backbone(net.backbone, d, step=True)
        neck = csp.neck(net.neck, d, step=True)
        outs = [csp.module(m, d) for m in net.neck.out_convs]
        h = net.bbox_head
        branches = [
            (
                csp.chain(*(csp.module(m, d) for m in h.cls_convs[i])),
                csp.conv(h.rtm_cls[i], d),
                csp.chain(*(csp.module(m, d) for m in h.reg_convs[i])),
                csp.conv(h.rtm_reg[i], d),
            )
            for i in range(len(outs))
        ]

        heads = [
            csp.stepped(
                lambda f, out=out, b=b: (
                    b[1](b[0](y := out(f))).astype(mx.float32),
                    b[3](b[2](y)).astype(mx.float32),
                )
            )
            for out, b in zip(outs, branches, strict=True)
        ]

        def forward(x: mx.array) -> tuple[mx.array, ...]:
            return tuple(m for f, head in zip(neck(*backbone(x.astype(d))), heads, strict=True) for m in head(f))

        self.forward = forward

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        """The six head maps of normalised canvases `[B, H, W, 3]`."""
        return csp.steps(self.forward, x)
