"""SEA-RAFT on Apple MPS with Metal kernels (`fastforks.kernels.flow_metal`): the portable network, the correlation
pyramid and its lookup replaced by `CorrWindows` (no correlation volume; same values up to rounding)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.kernels.flow_metal import CorrWindows

from ..net import FACTOR, _coords, _upsample

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary

    from ..net import SEARAFT


def forward(net: SEARAFT, lib: MetalLibrary, image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
    """`SEARAFT.forward` with `CorrWindows` over fmap2 halved `levels - 1` times (bilinear, as upstream)."""
    cfg = net.cfg
    b, _, h, w = image1.shape
    with net._amp(image1.device):
        net_, context = net.init_conv(net.cnet(torch.cat([image1, image2], dim=1))).split(cfg.dim, dim=1)
        flow = net.flow_head(net_)[:, :2].float()
        fmaps = net.fnet(torch.cat([image1, image2], dim=0))
    fmap1, fmap2 = fmaps.float().chunk(2, dim=0)
    levels = [fmap2]
    for _ in range(cfg.levels - 1):
        levels.append(F.interpolate(levels[-1], scale_factor=0.5, mode="bilinear", align_corners=False))
    corr = CorrWindows(lib, fmap1, levels, cfg.radius)
    grid = _coords(b, h // FACTOR, w // FACTOR, image1.device)
    for _ in range(iters):
        c = corr(grid + flow)
        with net._amp(image1.device):
            net_ = net.update_block(net_, context, c, flow)
            delta = net.flow_head(net_)[:, :2]
        flow = flow + delta.float()
    with net._amp(image1.device):
        mask = 0.25 * net.upsample_weight(net_)
    return _upsample(flow, mask.float())
