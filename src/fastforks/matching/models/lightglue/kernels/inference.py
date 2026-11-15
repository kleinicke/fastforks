"""Kernels path of DISK's U-Net (the LightGlue DISK variant): fp16 NHWC convolutions through `cnn_kernels.conv_ep`.

The U-Net is 16 convolutions (5 x 5, up to 129 output channels) with InstanceNorm + PReLU in front of each, 70 % of
the DISK extractor's time in fp32 (TF32 implicit GEMM). Here the weights are fp16 NHWC, every convolution is a
fp16-accumulate cuDNN graph (CUTLASS for >= 128 output channels) with the bias in the epilogue, and InstanceNorm is one
per-channel scale / shift epilogue (fp32 statistics, in place). The first convolution (3 input channels) gets its input
padded to 8 channels. Detection (NMS, top-k) and the descriptor lookup stay float32, as in portable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import override
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import CL, conv_ep

if TYPE_CHECKING:
    from ..net.disk import DISK, Conv, Unet

_IN_PAD = 8  # input channels of the first convolution after padding


def _conv_forward(self: Conv, x: torch.Tensor) -> torch.Tensor:
    """`conv(prelu(instance_norm(x)))` (norm and gate skipped in the first block) of a channels-last fp16 `x`."""
    norm, nonl, _, conv = self
    if isinstance(norm, nn.InstanceNorm2d):
        x = cnn_kernels.instance_norm(x, norm.eps)
        x = F.prelu(x, nonl.weight)
    return conv_ep(conv, x)


def prepare_disk(disk: DISK, device: torch.device) -> None:
    """Switch the U-Net of `disk` (fp32, eval, on `device`) to the kernels path in place.

    Raises:
        KernelUnavailableError: A kernel library is unusable on `device`.
    """
    cnn_kernels.load(device)
    unet: Unet = disk.unet
    first = unet.path_down[0][1][3]
    for m in unet.modules():
        if isinstance(m, nn.Conv2d):
            m.padding = (m.kernel_size[0] // 2, m.kernel_size[1] // 2)  # padding="same" (odd kernels)
    w = first.weight.data
    first.weight.data = torch.cat([w, w.new_zeros(w.shape[0], _IN_PAD - w.shape[1], *w.shape[2:])], 1)
    first.in_channels = _IN_PAD
    cnn_kernels.prepare_convs(unet)
    for m in unet.modules():
        if type(m).__name__ == "Conv":  # the U-Net block: norm, gate, dropout, conv
            override(m, _conv_forward)
    disk.unet = KernelUnet(unet)


class KernelUnet(nn.Module):
    """The prepared U-Net on a float32 NCHW image: fp16 channels-last inside, fp16 NHWC output."""

    def __init__(self, unet: Unet) -> None:
        """Wrap the prepared `unet`."""
        super().__init__()
        self.unet = unet

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """The heatmap and descriptor channels of the U-Net (`Unet.forward` without its channel check)."""
        x = F.pad(images.half(), (0, 0, 0, 0, 0, _IN_PAD - images.shape[1])).contiguous(memory_format=CL)
        features = [x]
        for layer in self.unet.path_down:
            features.append(layer(features[-1]))
        bot = features[-1]
        for layer, hor in zip(self.unet.path_up, features[-2::-1]):
            bot = layer(bot, hor)
        return bot
