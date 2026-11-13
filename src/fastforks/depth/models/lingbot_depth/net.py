"""LingBot-Depth network after upstream (robbyant/lingbot-depth f3a237e, Apache-2.0, `LICENSE`), same state-dict names.

The RGB-D encoder is DINOv2 ViT-L/14 (`fastforks.shared.depth_anything_v2`) with a second patch embedding for log
depth; the neck and the depth / mask heads are MoGe's conv stacks (the same modules as upstream's `ConvStack` with the
released configs). Upstream's xFormers nested-tensor path, gradient checkpointing, MAE masking and ONNX mode are left
out; the runners drive the forward pass.
"""

from __future__ import annotations

import torch
from torch import nn

from fastforks.depth.models.moge.net import ConvStack, _stack
from fastforks.shared.depth_anything_v2 import DINOv2

WIDTH = 1024  # ViT-L embedding = encoder output width


class DepthEmbed(nn.Module):
    """The 1-channel patch embedding of the depth map (upstream `depth_patch_embed`, a stride == kernel conv)."""

    def __init__(self, dim: int, patch: int) -> None:
        """`proj`: `Conv2d(1, dim, patch, stride=patch)`."""
        super().__init__()
        self.proj = nn.Conv2d(1, dim, patch, patch)


class Encoder(nn.Module):
    """Upstream `DINOv2_RGBD_Encoder`: the ViT over image + valid depth tokens, its last image tokens projected."""

    def __init__(self) -> None:
        """ViT-L/14, its depth patch embedding, one 1x1 output projection, the ImageNet normalisation buffers."""
        super().__init__()
        self.backbone = DINOv2("vitl")
        self.backbone.depth_patch_embed = DepthEmbed(WIDTH, self.backbone.patch_size)
        self.output_projections = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 1)])
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


class LingBotDepthNet(nn.Module):
    """Encoder, the shared neck (features + view-plane uv), the depth and mask heads."""

    def __init__(self) -> None:
        """The released ViT-L configuration (both checkpoints)."""
        super().__init__()
        dims = [WIDTH, 256, 128, 64, 32]
        self.encoder = Encoder()
        self.neck: ConvStack = _stack(dims, [0, 2, 2, 2, 0], [WIDTH + 2, 2, 2, 2, 2], [None] * 5)
        self.depth_head: ConvStack = _stack(dims, [0, 1, 1, 1, 0], list(dims), [None] * 4 + [1])
        self.mask_head: ConvStack = _stack(dims, [0, 1, 1, 1, 0], list(dims), [None] * 4 + [1])
