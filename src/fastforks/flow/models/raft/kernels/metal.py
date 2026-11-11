"""RAFT on Apple MPS with Metal kernels (`fastforks.kernels.flow_metal`): the portable network, the correlation
pyramid and its lookup replaced by `CorrWindows` (no correlation volume; same values up to rounding)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.kernels.flow_metal import CorrWindows

from ..net import FACTOR, _coords, _upsample

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary

    from ..net import RAFT


def forward(net: RAFT, lib: MetalLibrary, image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
    """`RAFT.forward` with `CorrWindows` over fmap2 average-pooled `corr_levels - 1` times."""
    cfg = net.cfg
    b, _, h, w = image1.shape
    with net._amp(image1.device):
        fmaps = net.feature_encoder(torch.cat([image1, image2], dim=0))
        ctx = net.context_encoder(image1)
    fmap1, fmap2 = fmaps.float().chunk(2, dim=0)
    levels = [fmap2]
    for _ in range(cfg.corr_levels - 1):
        levels.append(F.avg_pool2d(levels[-1], kernel_size=2, stride=2))
    corr = CorrWindows(lib, fmap1, levels, cfg.corr_radius)
    hidden, context = torch.split(ctx, [cfg.hidden, ctx.shape[1] - cfg.hidden], dim=1)
    hidden, context = torch.tanh(hidden), F.relu(context)
    coords0 = _coords(b, h // FACTOR, w // FACTOR, image1.device)
    coords1 = coords0
    for _ in range(iters):
        c = corr(coords1)
        with net._amp(image1.device):
            hidden, delta = net.update_block(hidden, context, c, coords1 - coords0)
        coords1 = coords1 + delta.float()
    mask = None
    if cfg.mask:
        with net._amp(image1.device):
            mask = net.mask_predictor(hidden)
        mask = mask.float()
    return _upsample(coords1 - coords0, mask)
