# MIT License, Copyright (c) 2026 Computer Vision and Geometry Lab (LICENSE)
"""FlexGlue's dense descriptor: upstream cvg/FlexGlue `flexglue/descriptor.py` (`DINOv3Descriptor` without the
loader, preprocessing and the layer-0 `descriptor_head`, which the sparse matcher never reads).

The DINOv3 ConvNeXt-B backbone (its weights: DINOv3 License, LICENSE-DINOv3) and the stride-8 `LinearConvHead`.
fastforks changes: `encode` only (the caller resizes).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # fixed: upstream `DINOv3Descriptor` mean / std buffers


def to_hwc(x: torch.Tensor) -> torch.Tensor:
    """(b, c, h, w) -> (b, h, w, c)."""
    return x.permute(0, 2, 3, 1)


def to_chw(x: torch.Tensor) -> torch.Tensor:
    """(b, h, w, c) -> (b, c, h, w)."""
    return x.permute(0, 3, 1, 2)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of a [B x C x H x W] tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return to_chw(super().forward(to_hwc(x)))


class ConvNeXtBlock(nn.Module):
    """Depthwise 7 x 7, LayerNorm, MLP, layer scale, residual."""

    def __init__(self, dim: int) -> None:  # noqa: D107
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        y = self.norm(to_hwc(self.dwconv(x)))
        y = self.gamma * self.pwconv2(self.act(self.pwconv1(y)))
        return x + to_chw(y)


class ConvNeXt(nn.Module):
    """DINOv3 ConvNeXt backbone, returns the unnormalized features of all 4 stages."""

    def __init__(self, depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024)) -> None:  # noqa: D107
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(3, dims[0], 4, stride=4), LayerNorm2d(dims[0], 1e-6))]
            + [nn.Sequential(LayerNorm2d(dims[i], 1e-6), nn.Conv2d(dims[i], dims[i + 1], 2, 2)) for i in range(3)]
        )
        self.stages = nn.ModuleList(
            nn.Sequential(*[ConvNeXtBlock(d) for _ in range(n)]) for n, d in zip(depths, dims, strict=True)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:  # noqa: D102
        features = []
        for downsample, stage in zip(self.downsample_layers, self.stages, strict=True):
            x = stage(downsample(x))
            features.append(x)
        return features


class LinearConvHead(nn.Module):
    """Stride-8 features: unshuffled conv stem on stride 4 plus linear strides 8-32."""

    def __init__(self, in_dims=(128, 256, 512, 1024), dim=512, out_dim=256) -> None:  # noqa: D107
        super().__init__()
        fine_in, *coarse_in = in_dims
        c4 = dim // 4
        self.stem = nn.Sequential(LayerNorm2d(fine_in), nn.Conv2d(fine_in, c4, 3, padding=1), LayerNorm2d(c4), nn.GELU())
        self.norms = nn.ModuleList(LayerNorm2d(c) for c in coarse_in)
        self.fine_proj = nn.Conv2d(4 * c4, dim, 1)
        self.projs = nn.ModuleList(nn.Conv2d(c, dim, 1, bias=False) for c in coarse_in)
        self.mlp = nn.Sequential(LayerNorm2d(dim), nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.down = nn.Conv2d(dim, out_dim, 1)
        self.out_norm = LayerNorm2d(out_dim)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:  # noqa: D102
        fine, *coarse = features
        h, w = coarse[0].shape[-2:]
        x = self.fine_proj(F.pixel_unshuffle(self.stem(fine), 2))
        for norm, proj, c in zip(self.norms, self.projs, coarse, strict=True):
            c = proj(norm(c))  # 1x1 commutes with bilinear, so project first
            if c.shape[-2:] != (h, w):
                c = F.interpolate(c, (h, w), mode="bilinear", align_corners=False)
            x = x + c
        return self.out_norm(self.down(x + self.mlp(x)))


class Descriptor(nn.Module):
    """Dense features `[B, 256, h / 8, w / 8]` of images already at the input size (upstream `encode`)."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.backbone = ConvNeXt()
        self.head = LinearConvHead()
        self.register_buffer("mean", torch.tensor(MEAN)[:, None, None], persistent=False)
        self.register_buffer("std", torch.tensor(STD)[:, None, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(self.backbone((x - self.mean) / self.std))
