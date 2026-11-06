"""Kernels-mode pieces of the DPT-BEiT-L shared by MiDaS and ZoeDepth.

fp16 BEiT tokens and the DPT reassemble / fusion decoder on cuDNN fp16-accumulate convs (channels-last), returning the
decoder features both families read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.half import fp16_accumulate
from fastforks.kernels import cudnn_hacc

from . import use_sdpa_attention

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .net.dpt import DPTDepthModel

__all__ = ["conv", "decoder", "encode", "fp16_accumulate", "to_kernels"]


def conv(x: torch.Tensor, c: Any) -> torch.Tensor:  # noqa: ANN401 - a vendored Conv2d / ConvTranspose2d
    """fp16 channels-last cuDNN conv or transposed conv with fp16 accumulation."""
    if isinstance(c, torch.nn.ConvTranspose2d):
        y = cudnn_hacc.deconv_half_acc(x, c.weight, list(c.padding), list(c.stride))
        return y.add_(c.bias.view(1, -1, 1, 1))
    return cudnn_hacc.conv_half_acc(x, c.weight, c.bias, list(c.padding), list(c.stride), list(c.dilation))


def decoder(net: DPTDepthModel, layers: Sequence[torch.Tensor], h: int, w: int) -> list[torch.Tensor]:
    """Upstream reassemble + DPT fusion on fp16 tokens with cuDNN fp16-accumulate convs, channels-last.

    Returns `[out_conv, layer4_rn, path_4, path_3, path_2, path_1]` (ZoeDepth's feature hooks; `out_conv` is the
    32-channel ReLU output before the last 1x1 conv).
    """
    pre, s = net.pretrained, net.scratch
    gh, gw = h // 16, w // 16
    feats = []
    for i, t in enumerate(layers):
        post = getattr(pre, f"act_postprocess{i + 1}")
        x = post[0](t)  # project readout: Linear(cat(tokens, cls)) + GELU
        x = x.view(x.shape[0], gh, gw, -1).permute(0, 3, 1, 2)  # channels-last view
        for c in post[3:]:
            x = conv(x, c)
        feats.append(x)
    rn = [conv(f, getattr(s, f"layer{i + 1}_rn")) for i, f in enumerate(feats)]

    def rcu(r: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - vendored ResidualConvUnit_custom
        return conv(F.relu(conv(F.relu(x), r.conv1)), r.conv2).add_(x)

    def fuse(f: Any, x: torch.Tensor, skip: torch.Tensor | None, size: Sequence[int] | None) -> torch.Tensor:  # noqa: ANN401 - vendored FeatureFusionBlock_custom
        if skip is not None:
            x = x + rcu(f.resConfUnit1, skip)
        x = rcu(f.resConfUnit2, x)
        if size is None:
            size = (x.shape[2] * 2, x.shape[3] * 2)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return conv(x, f.out_conv)

    p4 = fuse(s.refinenet4, rn[3], None, rn[2].shape[2:])
    p3 = fuse(s.refinenet3, p4, rn[2], rn[1].shape[2:])
    p2 = fuse(s.refinenet2, p3, rn[1], rn[0].shape[2:])
    p1 = fuse(s.refinenet1, p2, rn[0], None)
    oc = s.output_conv
    y = conv(p1, oc[0])
    y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
    y = F.relu_(conv(y, oc[2]))
    return [y, rn[3], p4, p3, p2, p1]


def to_kernels(net: DPTDepthModel) -> None:
    """fp16 weights, channels-last conv weights, SDPA attention (in place)."""
    use_sdpa_attention(net)
    net.half()
    for m in net.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)


def encode(net: DPTDepthModel, x: torch.Tensor) -> list[torch.Tensor]:
    """fp16 BEiT tokens of the hooked blocks, fp16-accumulating GEMMs."""
    with fp16_accumulate():
        return net.pretrained.model.forward_features(x.half(), net.pretrained.hooks)
