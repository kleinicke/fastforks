# Copyright 2020 The MMSegmentation Authors (Apache-2.0, see LICENSE in this directory); condensed from
# mmsegmentation v1.2.2 (mmseg/models/backbones/mscan.py, mmseg/models/decode_heads/ham_head.py) and mmcv 2.1.0
# (ConvModule): inference only, feature maps kept as `[B, C, H, W]` (upstream flattens them to tokens between the
# blocks), the NMF start bases a fixed buffer instead of a fresh random draw per call, no mmcv / mmengine.
"""SegNeXt as plain PyTorch modules, state-dict compatible with the mmsegmentation checkpoints.

- `MSCAN`: four stages of multi-scale convolutional attention blocks (a 5 x 5 depthwise conv, then depthwise strip
  convs of 7, 11 and 21 taps, summed and used as attention weights) with conv MLPs.
- `LightHamHead`: stages 2-4 resized to stride 8 and concatenated, a Hamburger block (non-negative matrix
  factorisation of the feature map into `R = 16` bases, 7 multiplicative update steps) and the classifier.

`EncoderDecoder(backbone, decode_head)` maps a normalised image (sides divisible by 32) to logits at 1/8 resolution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

GROUPS = 32  # fixed: the configs' GroupNorm of the head
NMF_RANK = 16  # fixed: the configs' `MD_R`
NMF_STEPS = 7  # fixed: the configs' `eval_steps`
NMF_EPS = 1e-6  # fixed: upstream constant
NMF_SEED = 0  # fixed: seed of the start bases (upstream draws new ones on every call)
ARCHS = {  # the configs' embed_dims, mlp_ratios, depths, head channels
    "mscan-t": ((32, 64, 160, 256), (8, 8, 4, 4), (3, 3, 5, 2), 256),
    "mscan-s": ((64, 128, 320, 512), (8, 8, 4, 4), (2, 2, 4, 2), 256),
    "mscan-b": ((64, 128, 320, 512), (8, 8, 4, 4), (3, 3, 12, 3), 512),
    "mscan-l": ((64, 128, 320, 512), (8, 8, 4, 4), (3, 5, 27, 3), 1024),
}


class Mlp(nn.Module):
    """1 x 1 conv, 3 x 3 depthwise conv, GELU, 1 x 1 conv."""

    def __init__(self, c: int, hidden: int) -> None:
        """`c` channels, `hidden` inside."""
        super().__init__()
        self.fc1 = nn.Conv2d(c, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.fc2 = nn.Conv2d(hidden, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, C, H, W]` -> `[B, C, H, W]`."""
        return self.fc2(F.gelu(self.dwconv(self.fc1(x))))


class MSCAAttention(nn.Module):
    """Multi-scale strip-conv attention: `conv3(sum of the depthwise branches) * x`."""

    # Explicit types for the strip-convolution branches registered with setattr below.
    conv0_1: nn.Conv2d
    conv0_2: nn.Conv2d
    conv1_1: nn.Conv2d
    conv1_2: nn.Conv2d
    conv2_1: nn.Conv2d
    conv2_2: nn.Conv2d

    def __init__(self, c: int) -> None:
        """`c` channels."""
        super().__init__()
        self.conv0 = nn.Conv2d(c, c, 5, padding=2, groups=c)
        for i, k in enumerate((7, 11, 21)):
            setattr(self, f"conv{i}_1", nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c))
            setattr(self, f"conv{i}_2", nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0), groups=c))
        self.conv3 = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, C, H, W]` -> `[B, C, H, W]`."""
        a = self.conv0(x)
        a0 = self.conv0_2(self.conv0_1(a))
        a1 = self.conv1_2(self.conv1_1(a))
        a2 = self.conv2_2(self.conv2_1(a))
        return self.conv3(a + a0 + a1 + a2) * x


class MSCASpatialAttention(nn.Module):
    """1 x 1 conv, GELU, `MSCAAttention`, 1 x 1 conv, plus the input."""

    def __init__(self, c: int) -> None:
        """`c` channels."""
        super().__init__()
        self.proj_1 = nn.Conv2d(c, c, 1)
        self.spatial_gating_unit = MSCAAttention(c)
        self.proj_2 = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, C, H, W]` -> `[B, C, H, W]`."""
        return self.proj_2(self.spatial_gating_unit(F.gelu(self.proj_1(x)))) + x


class MSCABlock(nn.Module):
    """BatchNorm + attention and BatchNorm + MLP, each a residual branch with a per-channel scale."""

    def __init__(self, c: int, mlp_ratio: int) -> None:
        """`c` channels, MLP width `c * mlp_ratio`."""
        super().__init__()
        self.norm1 = nn.BatchNorm2d(c)
        self.attn = MSCASpatialAttention(c)
        self.norm2 = nn.BatchNorm2d(c)
        self.mlp = Mlp(c, c * mlp_ratio)
        self.layer_scale_1 = nn.Parameter(torch.ones(c))
        self.layer_scale_2 = nn.Parameter(torch.ones(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, C, H, W]` -> `[B, C, H, W]`."""
        x = x + self.layer_scale_1[:, None, None] * self.attn(self.norm1(x))
        return x + self.layer_scale_2[:, None, None] * self.mlp(self.norm2(x))


class StemConv(nn.Module):
    """Two stride-2 3 x 3 convs (BatchNorm after each, GELU between): stride 4."""

    def __init__(self, c: int) -> None:
        """3 -> `c` channels."""
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3, c // 2, 3, 2, 1),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c, 3, 2, 1),
            nn.BatchNorm2d(c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, 3, H, W]` -> `[B, C, H / 4, W / 4]`."""
        return self.proj(x)


class OverlapPatchEmbed(nn.Module):
    """Stride-2 3 x 3 conv + BatchNorm."""

    def __init__(self, c_in: int, c: int) -> None:
        """`c_in` -> `c` channels."""
        super().__init__()
        self.proj = nn.Conv2d(c_in, c, 3, 2, 1)
        self.norm = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, C_in, H, W]` -> `[B, C, H / 2, W / 2]`."""
        return self.norm(self.proj(x))


class MSCAN(nn.Module):
    """The backbone: feature maps at strides 4, 8, 16 and 32."""

    def __init__(self, dims: tuple[int, ...], mlp_ratios: tuple[int, ...], depths: tuple[int, ...]) -> None:
        """Per stage: channels, MLP ratio, number of blocks."""
        super().__init__()
        self.stages = len(dims)
        for i, (c, r, d) in enumerate(zip(dims, mlp_ratios, depths, strict=True)):
            setattr(self, f"patch_embed{i + 1}", StemConv(c) if i == 0 else OverlapPatchEmbed(dims[i - 1], c))
            setattr(self, f"block{i + 1}", nn.ModuleList(MSCABlock(c, r) for _ in range(d)))
            setattr(self, f"norm{i + 1}", nn.LayerNorm(c))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Normalised image -> the four stage outputs (LayerNorm over the channels after each stage)."""
        outs = []
        for i in range(1, self.stages + 1):
            x = getattr(self, f"patch_embed{i}")(x)
            for blk in getattr(self, f"block{i}"):
                x = blk(x)
            x = getattr(self, f"norm{i}")(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            outs.append(x)
        return outs


class ConvModule(nn.Module):
    """Mmcv `ConvModule` as the head uses it: 1 x 1 conv, optional GroupNorm (then no conv bias), optional ReLU."""

    def __init__(self, c_in: int, c_out: int, *, norm: bool = True, act: bool = True) -> None:
        """`c_in` -> `c_out` channels."""
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1, bias=not norm)
        if norm:
            self.gn = nn.GroupNorm(GROUPS, c_out)
        self.norm, self.act = norm, act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv [-> GroupNorm] [-> ReLU]."""
        x = self.conv(x)
        if self.norm:
            x = self.gn(x)
        return F.relu(x) if self.act else x


def nmf(x: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    """Rank-`R` non-negative approximation of `x` `[B, D, N]` (>= 0) from start `bases` `[B, D, R]`: `[B, D, N]`.

    Upstream `NMF2D` (`inv_t = 1`): softmax start coefficients, `NMF_STEPS` multiplicative updates of coefficients and
    bases, one more of the coefficients.
    """
    xt = x.transpose(1, 2)
    coef = torch.softmax(xt @ bases, -1)
    for _ in range(NMF_STEPS):
        coef = coef * (xt @ bases) / (coef @ (bases.transpose(1, 2) @ bases) + NMF_EPS)
        bases = bases * (x @ coef) / (bases @ (coef.transpose(1, 2) @ coef) + NMF_EPS)
    coef = coef * (xt @ bases) / (coef @ (bases.transpose(1, 2) @ bases) + NMF_EPS)
    return bases @ coef.transpose(1, 2)


class Hamburger(nn.Module):
    """`relu(x + ham_out(nmf(relu(ham_in(x)))))`.

    `bases` holds the NMF start bases (uniform in `[0, 1)`, unit-norm columns) drawn once from `NMF_SEED`; upstream
    draws new ones on every call, so its logits change from run to run. The factorisation runs in float32 (its
    `1e-6` terms are below half-precision resolution).
    """

    bases: torch.Tensor

    def __init__(self, c: int) -> None:
        """`c` channels."""
        super().__init__()
        self.ham_in = ConvModule(c, c, norm=False, act=False)
        self.ham_out = ConvModule(c, c, act=False)
        start = torch.rand((1, c, NMF_RANK), generator=torch.Generator().manual_seed(NMF_SEED))
        self.register_buffer("bases", F.normalize(start, dim=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, C, H, W]` -> `[B, C, H, W]`."""
        b, c, h, w = x.shape
        y = F.relu(self.ham_in(x)).float().reshape(b, c, h * w)
        y = nmf(y, self.bases.float().expand(b, -1, -1)).reshape(b, c, h, w).to(x.dtype)
        if not x.is_contiguous() and x.is_contiguous(memory_format=torch.channels_last):
            y = y.contiguous(memory_format=torch.channels_last)
        return F.relu(x + self.ham_out(y))


class LightHamHead(nn.Module):
    """The decode head on stages 2-4: logits at stride 8."""

    def __init__(self, in_channels: tuple[int, ...], channels: int, classes: int) -> None:
        """Stage channels `in_channels`, head width `channels`."""
        super().__init__()
        self.squeeze = ConvModule(sum(in_channels), channels)
        self.hamburger = Hamburger(channels)
        self.align = ConvModule(channels, channels)
        self.conv_seg = nn.Conv2d(channels, classes, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Stage outputs 2-4 -> logits `[B, classes, H / 8, W / 8]`."""
        size = feats[0].shape[2:]
        x = torch.cat([feats[0]] + [F.interpolate(f, size, mode="bilinear", align_corners=False) for f in feats[1:]], 1)
        return self.conv_seg(self.align(self.hamburger(self.squeeze(x))))


class EncoderDecoder(nn.Module):
    """Mmseg `extract_feat` + `decode_head.forward`."""

    def __init__(self, backbone: MSCAN, decode_head: LightHamHead) -> None:
        """Backbone and head."""
        super().__init__()
        self.backbone, self.decode_head = backbone, decode_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised image `[B, 3, H, W]` (sides divisible by 32) -> logits `[B, classes, H / 8, W / 8]`."""
        return self.decode_head(self.backbone(x)[1:])


def build(arch: str, classes: int = 150) -> EncoderDecoder:
    """The network of `arch` (a key of `ARCHS`), randomly initialised."""
    dims, ratios, depths, channels = ARCHS[arch]
    return EncoderDecoder(MSCAN(dims, ratios, depths), LightHamHead(dims[1:], channels, classes))
