"""RTMPose network (mmpose 1.3.2 `TopdownPoseEstimator`: CSPNeXt backbone + `RTMCCHead`), inference only.

- CSPNeXt (mmdet 3.3.0, arch P5): a 3-conv stem (stride 2), then four stages, each a 3 x 3 stride-2 conv and a
  `CSPLayer`; the last stage has an SPP bottleneck (max pools 5, 9, 13) before it. A `CSPLayer` splits into a main
  branch (1 x 1 conv, `CSPNeXtBlock`s: 3 x 3 conv, then 5 x 5 depthwise + 1 x 1 pointwise, residual) and a 1 x 1
  shortcut, concatenates them, applies channel attention (global average pool, 1 x 1 conv with bias, hard sigmoid,
  multiply) and a final 1 x 1 conv. Every conv is followed by BatchNorm (eps 1e-5) and SiLU; BatchNorm is folded into
  the convs at load. Depth and width scale with `deepen` / `widen` (t, s, m, l)
- RTMCC head: a 7 x 7 conv to one map per keypoint (stride 32), flattened to a token per keypoint; ScaleNorm + a
  linear layer to 256; one gated attention unit (GAU: ScaleNorm, SiLU(linear) -> u, v, a shared base that per-dim
  scales and offsets give q and k, relu(q k^T / sqrt(s))^2 attention, u * (attn v), linear; scaled residual); then
  two linear layers to the SimCC x / y logits (2 bins per input pixel)
- RTMW (`neck`): the backbone's stride 8 / 16 / 32 maps go through a `CSPNeXtPAFPN` (top-down, then bottom-up path
  of `CSPLayer`s without attention) that gives a stride-16 and a stride-32 map, and `RTMWHead` builds each keypoint
  token from two halves: the 7 x 7 conv on the stride-32 map as above, and a 7 x 7 conv on the stride-16 map
  concatenated with the pixel-shuffled stride-32 map (the head convs: BatchNorm + ReLU)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.cspnext import STAGES, ConvAct, CSPLayer, CSPNeXt

# Shared CSPNeXt preserves the upstream module names and returns its three pyramid levels.
STRIDE = 32  # fixed: architecture (backbone output stride)
FINAL_KERNEL = 7  # fixed: architecture (head `final_layer_kernel_size`)
HIDDEN = 256  # fixed: architecture (GAU `hidden_dims`)
S = 128  # fixed: architecture (GAU shared q / k width)
EXPANSION = 2  # fixed: architecture (GAU `expansion_factor`: e = 512)
SPLIT = 2  # fixed: architecture (SimCC bins per input pixel, `simcc_split_ratio`)
NORM_EPS = 1e-5  # fixed: architecture (ScaleNorm clamp)
NECK_BLOCKS = 2  # fixed: architecture (RTMW `CSPNeXtPAFPN` `num_csp_blocks`, every size)
SHUFFLE = 2  # fixed: architecture (RTMW head pixel shuffle: stride 32 -> 16)


@dataclass(frozen=True)
class Arch:
    """Card `[params]` + variant `params`."""

    deepen: float  # CSPNeXt depth factor (blocks per stage)
    widen: float  # CSPNeXt width factor (channels)
    keypoints: int  # SimCC outputs (the variant's label set)
    height: int  # network input height
    width: int  # network input width
    neck: bool  # RTMW: PAFPN neck + the two-scale head


class PAFPN(nn.Module):
    """`CSPNeXtPAFPN` without output convs: stride 8 / 16 / 32 maps -> (stride 16 `[512 w]`, stride 32 `[1024 w]`)."""

    def __init__(self, a: Arch) -> None:
        """Layers on the backbone's stage 2 - 4 widths."""
        super().__init__()
        c3, c4, c5 = (int(s[1] * a.widen) for s in STAGES[1:])

        def csp(cin: int, cout: int) -> CSPLayer:
            return CSPLayer(cin, cout, NECK_BLOCKS, identity=False, attention=False, depthwise=False)

        self.reduce_layers = nn.ModuleList([ConvAct(c5, c4, 1), ConvAct(c4, c3, 1)])
        self.top_down_blocks = nn.ModuleList([csp(2 * c4, c4), csp(2 * c3, c3)])
        self.downsamples = nn.ModuleList([ConvAct(c3, c3, 3, 2), ConvAct(c4, c4, 3, 2)])
        self.bottom_up_blocks = nn.ModuleList([csp(2 * c3, c4), csp(2 * c4, c5)])

    def forward(self, f3: torch.Tensor, f4: torch.Tensor, f5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(stride-16 map, stride-32 map)."""
        t5 = self.reduce_layers[0](f5)
        m4 = self.top_down_blocks[0](torch.cat([F.interpolate(t5, scale_factor=2.0, mode="nearest"), f4], 1))
        t4 = self.reduce_layers[1](m4)
        m3 = self.top_down_blocks[1](torch.cat([F.interpolate(t4, scale_factor=2.0, mode="nearest"), f3], 1))
        o4 = self.bottom_up_blocks[0](torch.cat([self.downsamples[0](m3), t4], 1))
        o5 = self.bottom_up_blocks[1](torch.cat([self.downsamples[1](o4), t5], 1))
        return o4, o5


class Trunk(nn.Module):
    """Shared backbone, optionally followed by the RTMW neck, as one optimized module."""

    def __init__(self, backbone: CSPNeXt, neck: PAFPN | None) -> None:
        """The shared backbone and optional neck."""
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Last map for RTMPose, or (stride-16, stride-32) neck maps for RTMW."""
        features = self.backbone(x)
        return features[-1] if self.neck is None else self.neck(*features)


class ScaleNorm(nn.Module):
    """x / clamp(||x|| / sqrt(dim), eps) * g (one scalar gain)."""

    def __init__(self, dim: int) -> None:
        """The gain."""
        super().__init__()
        self.scale = dim**-0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised over the last axis."""
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=NORM_EPS) * self.g


class Scale(nn.Module):
    """Per-channel residual scale."""

    def __init__(self, dim: int) -> None:
        """The scales."""
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class GAU(nn.Module):
    """`RTMCCBlock` (self-attention, no relative bias, no RoPE): res_scale * x + GAU(x)."""

    def __init__(self) -> None:
        """Layers of the unit (`HIDDEN` in and out)."""
        super().__init__()
        self.e = EXPANSION * HIDDEN
        self.ln = ScaleNorm(HIDDEN)
        self.uv = nn.Linear(HIDDEN, 2 * self.e + S, bias=False)
        self.gamma = nn.Parameter(torch.ones(2, S))
        self.beta = nn.Parameter(torch.zeros(2, S))
        self.o = nn.Linear(self.e, HIDDEN, bias=False)
        self.res_scale = Scale(HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, K, HIDDEN]` -> same."""
        u, v, base = F.silu(self.uv(self.ln(x))).split([self.e, self.e, S], 2)
        q, k = (base[:, :, None] * self.gamma + self.beta).unbind(2)
        kernel = F.relu(q @ k.transpose(1, 2) / math.sqrt(S)).square()
        return x * self.res_scale.scale + self.o(u * (kernel @ v))


class Head(nn.Module):
    """`RTMCCHead`: stride-32 features -> SimCC x logits `[B, K, w * 2]`, y logits `[B, K, h * 2]`."""

    def __init__(self, a: Arch) -> None:
        """Layers for the network input size."""
        super().__init__()
        flat = (a.height // STRIDE) * (a.width // STRIDE)
        self.final_layer = nn.Conv2d(int(STAGES[-1][1] * a.widen), a.keypoints, FINAL_KERNEL, 1, FINAL_KERNEL // 2)
        self.mlp = nn.Sequential(ScaleNorm(flat), nn.Linear(flat, HIDDEN, bias=False))
        self.gau = GAU()
        self.cls_x = nn.Linear(HIDDEN, a.width * SPLIT, bias=False)
        self.cls_y = nn.Linear(HIDDEN, a.height * SPLIT, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(x logits, y logits)."""
        t = self.gau(self.mlp(self.final_layer(x).flatten(2)))
        return self.cls_x(t), self.cls_y(t)


class ConvReLU(nn.Module):
    """mmcv `ConvModule` of the RTMW head with its BatchNorm folded: 7 x 7 conv + ReLU."""

    def __init__(self, cin: int, cout: int) -> None:
        """The conv (with the folded bias)."""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, FINAL_KERNEL, 1, FINAL_KERNEL // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """relu(conv(x))."""
        return F.relu(self.conv(x))


class WholeHead(nn.Module):
    """`RTMWHead`: (stride-16, stride-32) features -> SimCC x logits `[B, K, w * 2]`, y logits `[B, K, h * 2]`."""

    def __init__(self, a: Arch) -> None:
        """Layers for the network input size."""
        super().__init__()
        flat = (a.height // STRIDE) * (a.width // STRIDE)
        c = int(STAGES[-1][1] * a.widen)
        low = c // SHUFFLE**2
        self.conv_dec = ConvReLU(low, low)
        self.final_layer = ConvReLU(c, a.keypoints)
        self.final_layer2 = ConvReLU(c // 2 + low, a.keypoints)
        self.mlp = nn.Sequential(ScaleNorm(flat), nn.Linear(flat, HIDDEN // 2, bias=False))
        fine = flat * SHUFFLE**2
        self.mlp2 = nn.Sequential(ScaleNorm(fine), nn.Linear(fine, HIDDEN // 2, bias=False))
        self.gau = GAU()
        self.cls_x = nn.Linear(HIDDEN, a.width * SPLIT, bias=False)
        self.cls_y = nn.Linear(HIDDEN, a.height * SPLIT, bias=False)

    def forward(self, feats: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """(x logits, y logits)."""
        b, t = feats
        coarse = self.mlp(self.final_layer(t).flatten(2))
        b = torch.cat([self.conv_dec(F.pixel_shuffle(t, SHUFFLE)), b], 1)
        fine = self.mlp2(self.final_layer2(b).flatten(2))
        t = self.gau(torch.cat([coarse, fine], 2))
        return self.cls_x(t), self.cls_y(t)


class RTMPose(nn.Module):
    """Normalised crops `[B, 3, h, w]` -> (SimCC x `[B, K, 2 w]`, SimCC y `[B, K, 2 h]`)."""

    def __init__(self, a: Arch) -> None:
        """Backbone and head (RTMW: with the neck between them)."""
        super().__init__()
        self.backbone = CSPNeXt(a.deepen, a.widen)
        self.neck = PAFPN(a) if a.neck else None
        self.head = WholeHead(a) if a.neck else Head(a)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SimCC logits."""
        if self.neck is None:
            return self.head(self.backbone(x)[-1])
        return self.head(self.neck(*self.backbone(x)))
