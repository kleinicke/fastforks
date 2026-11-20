"""LoFTR runners: weights -> the vendored network, and the portable forward pass.

As upstream's MegaDepth / ScanNet loaders and demo: the RGB image is converted to grayscale (0.299 R + 0.587 G +
0.114 B, OpenCV's weights), resized so its longest side is `long_side` (rounded; 0 = the input size) and both sides
cut down to multiples of 8 (bilinear, as `cv2.resize`), then upstream's LoFTR: coarse matches (dual-softmax confidence
above `threshold`, `border_rm` cells off the border, mutual maxima) refined in a 5 x 5 window at 1/2 resolution. The
matched points are scaled back to the input (`x * W / W_resized`, as upstream's loaders' `scale0` / `scale1`); every
match is its own keypoint pair, so `pairs` is `[[0, 0], [1, 1], ...]` and `scores` the coarse confidence. The network
runs in the context dtype; the coarse and fine similarities, the dual-softmax and the outputs are float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import override
from fastforks.kernels import dual_softmax
from fastforks.matching._runner import Forward, Runner
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import conv_ep

from .net.loftr import BasicBlock, LoFTR, ResNetFPN_8_2

if TYPE_CHECKING:
    from fastforks.matching._runner import Context

GRAY = (0.299, 0.587, 0.114)  # fixed: OpenCV's RGB -> gray weights (upstream reads the images with cv2 as grayscale)
DIVISOR = 8  # fixed: the coarse stride (upstream `df`)


@dataclass(frozen=True)
class Params:
    """Variant parameters and load options."""

    temp_bug_fix: bool  # the position encoding the weights were trained with (upstream issue #41)
    border_rm: int  # coarse cells removed at the border (upstream eval configs: outdoor 2, indoor 0)
    threshold: float  # coarse dual-softmax confidence (upstream 0.2)
    long_side: int  # longest side the images are matched at; 0 = the input size


def resized_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Upstream `get_resized_wh` (longest side `long_side`, rounded) + `get_divisible_wh` (multiples of 8, cut down)."""
    if long_side > 0:
        s = long_side / max(h, w)
        h, w = round(h * s), round(w * s)
    return h // DIVISOR * DIVISOR, w // DIVISOR * DIVISOR


class Matcher(nn.Module):
    """LoFTR on an RGB pair: `[1, 3, H1, W1]`, `[1, 3, H2, W2]` in `[0, 1]` -> keypoints, index pairs, scores."""

    gray: torch.Tensor

    def __init__(self, params: Params) -> None:
        """Build the network (no weights)."""
        super().__init__()
        self.params = params
        self.loftr = LoFTR(temp_bug_fix=params.temp_bug_fix, thr=params.threshold, border_rm=params.border_rm)
        self.register_buffer("gray", torch.tensor(GRAY).view(1, 3, 1, 1), persistent=False)

    def prepare(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Grayscale at the matching size and the factors (x, y) back to input pixels."""
        h, w = image.shape[-2:]
        rh, rw = resized_size(h, w, self.params.long_side)
        if min(rh, rw) < DIVISOR * 2:
            msg = f"image of {w} x {h} px is too small for LoFTR (matched at {rw} x {rh}; at least 16 px per side)"
            raise ValueError(msg)
        g = (image * self.gray).sum(1, keepdim=True)
        if (rh, rw) != (h, w):
            g = F.interpolate(g, size=(rh, rw), mode="bilinear", align_corners=False)
        return g.to(self.loftr.backbone.conv1.weight.dtype), torch.tensor([w / rw, h / rh], device=image.device)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pair's matches (batch 1)."""
        (g1, s1), (g2, s2) = self.prepare(image1), self.prepare(image2)
        out = self.loftr(g1, g2)
        n = out["confidence"].shape[0]
        idx = torch.arange(n, device=image1.device)
        return {
            "keypoints1": out["keypoints0"] * s1,
            "keypoints2": out["keypoints1"] * s2,
            "pairs": torch.stack([idx, idx], -1),
            "scores": out["confidence"],
        }


def load_matcher(ctx: Context) -> Matcher:
    """The matcher with the variant's weights (a Lightning checkpoint; the indoor one prefixes `matcher.`)."""
    net = Matcher(ctx.params)
    sd = ctx.state_dict(0, key="state_dict", strip="matcher.")
    net.loftr.load_state_dict(sd, strict=True)
    net = net.eval().requires_grad_(requires_grad=False).to(ctx.device, ctx.torch_dtype)
    if net.gray.is_cuda:  # cuDNN: the backbone channels-last (no layout conversion around every convolution)
        net.loftr.backbone.to(memory_format=torch.channels_last)
    return net


class Portable(Runner):
    """Plain PyTorch; float32 or float16."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = load_matcher(ctx)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The matched points of both images, their index pairs and confidences."""
        return self.net(image1, image2)

    def module(self) -> torch.nn.Module:
        """The matcher (data-dependent match counts: compiles with graph breaks, does not export)."""
        return Forward(self.net, lambda net, image1, image2: net(image1, image2))


def _affines(backbone: ResNetFPN_8_2) -> None:
    """The eval BatchNorms of the backbone as fp32 affines `(s, t)` next to their conv (plain attributes)."""
    pairs: list[tuple[nn.Module, nn.Module]] = [(backbone.conv1, backbone.bn1)]
    for block in backbone.modules():
        if isinstance(block, BasicBlock):
            pairs += [(block.conv1, block.bn1), (block.conv2, block.bn2)]
            if block.downsample is not None:
                pairs.append((block.downsample[0], block.downsample[1]))
    for conv, bn in pairs:
        conv.__dict__["_affine"] = cnn_kernels.bn_affine(bn, conv)


def _affine(conv: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    return conv.__dict__["_affine"]


def _block_forward(self: BasicBlock, x: torch.Tensor) -> torch.Tensor:
    """`relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut)` with the BN and ReLU in the conv epilogues."""
    relu = cnn_kernels.ACT["relu"]
    y = conv_ep(self.conv1, x, *_affine(self.conv1), relu)
    r = x if self.downsample is None else conv_ep(self.downsample[0], x, *_affine(self.downsample[0]), 0)
    pad = self.conv2.weight.shape[0] - r.shape[1]  # 196 channels: the conv pads its output to a multiple of 8
    if pad:
        r = F.pad(r, (0, 0, 0, 0, 0, pad)).contiguous(memory_format=torch.channels_last)
    return conv_ep(self.conv2, y, *_affine(self.conv2), relu, r=r)


def _fpn_forward(self: ResNetFPN_8_2, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`ResNetFPN_8_2.forward` with the stem conv + BN + ReLU fused."""
    return self._pyramid(conv_ep(self.conv1, x, *_affine(self.conv1), cnn_kernels.ACT["relu"]))


def _coarse_matches(
    self: LoFTR, feat0: torch.Tensor, feat1: torch.Tensor, hw0: tuple[int, int], hw1: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the upstream first maximum per row, testing borders only after mutual selection."""
    scale = feat0.shape[-1] ** 0.5
    logits = ((feat0[0].float() / scale) @ (feat1[0].float() / scale).mT).div_(self.temperature)
    p = dual_softmax.normalize_(logits)
    scores, cols = p.max(1)
    keep = (scores > self.thr) & (scores == p.amax(0)[cols])
    if self.border_rm:
        keep &= self._inside(hw0, p.device) & self._inside(hw1, p.device)[cols]
    rows = torch.where(keep)[0]
    return rows, cols[rows], scores[rows]


class Kernels(Portable):
    """Portable with the backbone's convs as fp16 channels-last kernels, BatchNorm and ReLU (and the residual) fused."""

    def __init__(self, ctx: Context) -> None:
        """Build the network, prepare the backbone."""
        super().__init__(ctx)
        bb = self.net.loftr.backbone
        cnn_kernels.load(ctx.device)
        _affines(bb)
        cnn_kernels.prepare_convs(bb)
        for m in bb.modules():
            if isinstance(m, BasicBlock):
                override(m, _block_forward)
        override(bb, _fpn_forward)
        override(self.net.loftr, _coarse_matches, "coarse_matches")


RUNNERS = {"portable": Portable, "kernels": Kernels}
