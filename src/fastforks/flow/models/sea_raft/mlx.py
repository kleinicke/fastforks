"""SEA-RAFT forward for the MLX port: the same function as `net.SEARAFT.forward`, shaped for MLX on an Apple GPU.

Built from a loaded `SEARAFT` (its modules). What differs from the portable forward:

- No correlation volume: `corr_windows` takes each window's dot products from fmap1 and the halved fmap2 levels (the
  halving is linear, so the correlation with a halved map is the halved correlation). The volume was the peak memory
  and its `grid_sample` lookups most of the update loop's time.
- The context encoder's stem (7 x 7, stride 2, on both images stacked as 6 channels) is two 3-channel convolutions
  summed, the batch norm folded into them: MLX's convolution is ~9x slower on 6 input channels than on 3 (M1 Pro,
  float16, 1080 x 1920: 46 ms against 2 x 5.3 ms).

Exact regroupings in real arithmetic; outputs differ from the portable forward by rounding only.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.mlx_cast import to_float32
from fastforks._core.volume_ops import corr_windows

from .net import FACTOR, SEARAFT, _coords, _upsample


class MlxSEARAFT(nn.Module):
    """`SEARAFT` regrouped for MLX; the same signature and result as `SEARAFT.forward` for one pair per call."""

    def __init__(self, net: SEARAFT) -> None:
        """Take over `net`'s modules."""
        super().__init__()
        self.cfg = net.cfg
        self.cnet = net.cnet
        self.init_conv = net.init_conv
        self.fnet = net.fnet
        self.flow_head = net.flow_head
        self.upsample_weight = net.upsample_weight
        self.update_block = net.update_block
        self.w_stem1: torch.Tensor
        self.w_stem2: torch.Tensor
        self.b_stem: torch.Tensor
        stem: Any = net.cnet
        conv, bn = stem.conv1, stem.bn1
        k = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
        w = conv.weight.detach() * k[:, None, None, None]
        self.register_buffer("w_stem1", w[:, :3].contiguous(), persistent=False)
        self.register_buffer("w_stem2", w[:, 3:].contiguous(), persistent=False)
        self.register_buffer("b_stem", (conv.bias.detach() - bn.running_mean) * k + bn.bias.detach(), persistent=False)

    def _context(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """`cnet` on the stacked pair, its stem split by image."""
        cnet = self.cnet
        x = F.conv2d(image1, self.w_stem1, self.b_stem, stride=2, padding=3)
        x = F.relu(x + F.conv2d(image2, self.w_stem2, stride=2, padding=3))
        return cnet.final_conv(cnet.layer3(cnet.layer2(cnet.layer1(x))))

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
        """Flow after `iters` updates, float32 `[B, 2, H, W]`."""
        cfg = self.cfg
        b, _, h, w = image1.shape
        net, context = self.init_conv(self._context(image1, image2)).split(cfg.dim, dim=1)
        flow = to_float32(self.flow_head(net)[:, :2])
        # `.float()` records nothing in the replay: the feature maps stay in its dtype (as `net.py`'s float16 note)
        fmap1, fmap2 = self.fnet(torch.cat([image1, image2], dim=0)).float().chunk(2, dim=0)
        levels = [fmap2]
        for _ in range(cfg.levels - 1):
            levels.append(F.interpolate(levels[-1], scale_factor=0.5, mode="bilinear", align_corners=False))
        scale = 1 / math.sqrt(fmap1.shape[1])
        grid = _coords(b, h // FACTOR, w // FACTOR, image1.device)
        for _ in range(iters):
            corr = corr_windows(fmap1, levels, grid + flow, cfg.radius, scale)
            net = self.update_block(net, context, corr, flow)
            flow = flow + to_float32(self.flow_head(net)[:, :2])
        mask = 0.25 * self.upsample_weight(net)
        return _upsample(flow, to_float32(mask))
