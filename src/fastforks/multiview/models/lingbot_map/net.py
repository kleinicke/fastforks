"""LingBot-Map network (upstream `GCTStream`, SDPA backend) on the VGGT family's modules.

Upstream LingBot-Map extends VGGT's code (Meta copyright headers, Apache-2.0 release): the aggregator has the same
DINOv2 patch embed, frame blocks (2D RoPE) and global blocks plus a scale token (patch tokens start at 6), the camera
head the same trunk (attention over each frame's camera token, causal in streaming) and the depth head is VGGT's DPT
head. Only the parameters are defined here (same names as upstream, so the checkpoint keys match); the streaming
forward pass is in `model.py`.
"""

from __future__ import annotations

import torch
from torch import nn

from fastforks.multiview.models.vggt.net.heads.camera_head import CameraHead
from fastforks.multiview.models.vggt.net.heads.dpt_head import DPTHead
from fastforks.multiview.models.vggt.net.models.aggregator import Aggregator

PATCH = 14  # fixed: the ViT patch size
EMBED = 1024  # fixed: the aggregator width (one checkpoint)
SPECIAL = 6  # camera, 4 registers, scale


class Net(nn.Module):
    """Upstream `GCTStream` parameters: aggregator (+ scale token), camera head, depth head (no point head)."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.aggregator = Aggregator(img_size=518, patch_size=PATCH, embed_dim=EMBED)
        self.aggregator.scale_token = nn.Parameter(torch.ones(1, 2, 1, EMBED))
        self.aggregator.patch_start_idx = SPECIAL
        self.camera_head = CameraHead(dim_in=2 * EMBED)
        self.depth_head = DPTHead(dim_in=2 * EMBED, output_dim=2, activation="exp", conf_activation="expp1")
