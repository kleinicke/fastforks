"""FoundationStereo on Apple MPS: the geometry / correlation lookup of `GeoFast` as one Metal kernel per iteration.

`prepare(net)`: puts the geometry input channels of the motion encoder's `convc1` back from `GeoFast`'s (tap, channel)
order (`prepare_inference`) to upstream's (channel, tap) order, the kernel's; in place. `MetalGeo(lib, ...)`: a drop-in for `GeoFast` (same volumes,
same values, float16).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fastforks.kernels.stereo_metal.modules import LineLookup

from ..net.inference import GeoFast

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary

    from ..net.core.foundation_stereo import FoundationStereo


@torch.no_grad()
def prepare(net: FoundationStereo) -> None:
    """Permute `convc1`'s input channels: per level `[geo (K x C, tap-major) | corr (K)]` -> geo channel-major."""
    a, conv = net.args, net.update_block.encoder.convc1
    k = 2 * a.corr_radius + 1
    c = conv.in_channels // (a.corr_levels * k) - 1
    order, base = [], 0
    for _ in range(a.corr_levels):
        order += [base + t * c + g for g in range(c) for t in range(k)] + list(range(base + c * k, base + (c + 1) * k))
        base += (c + 1) * k
    conv.weight.copy_(conv.weight[:, order])


class MetalGeo:
    """`GeoFast`'s lookup with the input channels in `prepare`'s order."""

    def __init__(
        self,
        lib: MetalLibrary,
        fl: torch.Tensor,
        fr: torch.Tensor,
        comb: torch.Tensor,
        levels: int,
        dx: torch.Tensor,
    ) -> None:
        """Build `GeoFast`'s volumes and lay them out for the kernel."""
        g = GeoFast(fl, fr, comb, levels, dx)
        b, h, w = g.B, g.H, g.W
        segments = []
        for lvl, (geo, corr) in enumerate(zip(g.geo, g.corr, strict=True)):
            segments += [
                (geo.permute(0, 2, 3, 4, 1).reshape(b * h * w, g.C, -1), lvl),
                (corr.reshape(b * h * w, -1), lvl),
            ]
        self.lookup = LineLookup(lib, (b, h, w), (dx.numel() - 1) // 2, comb.dtype, [segments], geo_dtype=comb.dtype)

    def __call__(self, disp: torch.Tensor) -> torch.Tensor:
        """`[B, 522, H, W]` (channels_last) at disparity `disp` (`[B, 1, H, W]`)."""
        return self.lookup(disp)[0]
