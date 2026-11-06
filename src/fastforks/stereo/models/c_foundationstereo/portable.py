"""Portable-mode memory fixes (same math).

TAO's disparity transformer runs in chunks of its (B*H*W) disparity sequences; the cost-volume hourglass frees its
upsampled volume right after merging it and applies LeakyReLU in place.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import override

from .net.fast_foundation_stereo.fast_foundation_stereo import _FFSPostForwardHelper
from .net.foundation_stereo.convolution_helper import Conv
from .net.foundation_stereo.submodule import CostVolumeDisparityAttention, FeatureAtt

CHUNK = 1 << 15  # sequences per chunk: memory bound, and flash SDPA takes < 65536 (the result does not depend on it)


def _chunked(
    self: CostVolumeDisparityAttention, cv: torch.Tensor, window_size: tuple[int, int] = (-1, -1)
) -> torch.Tensor:
    """`CostVolumeDisparityAttention.forward`, with each sequence chunk run through all layers at once."""
    b, c, d, h, w = cv.shape
    x = cv.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, c)
    out = torch.empty_like(x)
    for i in range(0, x.shape[0], CHUNK):
        y = self.pos_embed0(x[i : i + CHUNK], resize_embed=self.resize_embed)
        for layer in self.sa:
            y = layer(y, window_size=window_size)
        out[i : i + CHUNK] = y
    return out.reshape(b, h, w, d, c).permute(0, 4, 3, 1, 2)


def _post(
    self: _FFSPostForwardHelper, conv_high: torch.Tensor, conv_low: torch.Tensor, left_feat: torch.Tensor | None = None
) -> torch.Tensor:
    """`_FFSPostForwardHelper.forward` without keeping the upsampled volume alive next to the merged one."""
    if self.op == "sum":
        x = self.upsample(conv_low) + conv_high
    else:
        x = torch.cat((self.upsample(conv_low), conv_high), dim=1)
    for layer in self.out:
        x = layer(x, left_feat) if isinstance(layer, FeatureAtt) else layer(x)
    return x


def _conv(self: Conv, x: torch.Tensor) -> torch.Tensor:
    """`Conv.forward` with the LeakyReLU in place (on the fresh conv / norm output)."""
    x = self.conv(x)
    if self.norm is not None:
        x = self.norm(x)
    return F.leaky_relu(x, inplace=True) if self.relu else x


def _channels_last(net: nn.Module) -> None:
    """Conv weights in channels-last layout: cuDNN then runs its NHWC kernels without transposing every activation."""
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
        elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)


def prepare_portable(net: nn.Module, *, channels_last: bool = False) -> None:
    """Chunk every disparity transformer of `net` and apply the hourglass memory fixes (the output is unchanged).

    With `channels_last` (the portable runner on CUDA) the conv weights go channels-last as well. The in-place LeakyReLU
    is applied to the Fast-FoundationStereo hourglass only: in FoundationStereo it leaves peak allocated unchanged and
    raised peak reserved (C-FS-S Midd-H 6590 -> 6850 MiB).
    """
    if channels_last:
        _channels_last(net)
    fast = any(isinstance(m, _FFSPostForwardHelper) for m in net.modules())
    for m in net.modules():
        if isinstance(m, CostVolumeDisparityAttention):
            override(m, _chunked)
        elif isinstance(m, _FFSPostForwardHelper):
            override(m, _post)
        elif fast and isinstance(m, Conv):
            override(m, _conv)
