"""Depthwise convolutions on PyTorch's own CUDA kernel instead of cuDNN.

For float16 depthwise convolutions in NCHW PyTorch picks cuDNN, whose first call of each new layer configuration
(channels, kernel, stride) costs ~130 ms on cuDNN 9 (RTX 5060 Ti, Windows: a plan built at run time; bfloat16 and
float32 already take PyTorch's kernel and pay nothing). A network with a dozen such layers spends over a second in
its first call. PyTorch's depthwise kernel (`aten::_conv_depthwise2d`, the one it already runs for bfloat16) has no
such set-up and accumulates in float32.
"""

from __future__ import annotations

import torch

from fastforks._core.patch import base_type, override

__all__ = ["native_depthwise"]


def _depthwise(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    w = conv.weight
    depthwise = torch.ops.aten._conv_depthwise2d  # noqa: SLF001 - aten's op, as F.conv2d dispatches to it
    return depthwise(x, w, w.shape[2:], conv.bias, conv.stride, conv.padding, conv.dilation)


def native_depthwise(net: torch.nn.Module) -> int:
    """Run `net`'s depthwise `Conv2d`s (one input channel per group, zero padding) on PyTorch's kernel; their count."""
    n = 0
    for m in net.modules():
        if (
            isinstance(m, torch.nn.Conv2d)
            and base_type(m) is torch.nn.Conv2d  # subclasses may change forward
            and m.groups > 1
            and m.groups == m.in_channels == m.out_channels
            and m.padding_mode == "zeros"
            and not isinstance(m.padding, str)
        ):
            override(m, _depthwise)
            n += 1
    return n
