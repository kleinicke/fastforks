"""How a detection family runs its convolutions: plain PyTorch (portable) or cuDNN fused / fp16-accumulate (kernels)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import nn

    from fastforks.kernels import cudnn_hacc


class Convs:
    """How the convolutions run: portable = PyTorch conv (+ bias, + residual, + ReLU)."""

    def conv(self, c: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        """`c(x)`."""
        return F.conv2d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups)

    def conv_relu(self, c: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        """`relu(c(x))`."""
        return F.relu_(F.conv2d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups))

    def conv_add_relu(self, c: nn.Conv2d, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        """`relu(c(x) + res)`."""
        return F.relu_(F.conv2d(x, c.weight, c.bias, c.stride, c.padding, c.dilation, c.groups).add_(res))


def _padding(c: nn.Conv2d) -> tuple[int, ...]:
    """`c.padding` (always numeric here: `net.py` builds every conv with integer padding)."""
    if isinstance(c.padding, str):
        msg = f"string padding {c.padding!r}"
        raise TypeError(msg)
    return c.padding


class FusedConvs(Convs):
    """Kernels: cuDNN fused conv + bias + ReLU (+ residual); other large convs with fp16 accumulation (`cudnn_hacc`)."""

    def __init__(self, hacc: cudnn_hacc.CudnnHalfConv) -> None:
        """Use `hacc` for the convs without a ReLU."""
        self.hacc = hacc

    def conv(self, c: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        """`c(x)`, fp16 accumulation (grouped convolutions run as PyTorch's)."""
        if c.groups != 1:
            return super().conv(c, x)
        return self.hacc.conv(x, c.weight, c.bias, _padding(c), c.stride, c.dilation)

    def conv_relu(self, c: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        """`relu(c(x))`, one cuDNN call."""
        return torch.cudnn_convolution_relu(x, c.weight, c.bias, c.stride, _padding(c), c.dilation, c.groups)

    def conv_add_relu(self, c: nn.Conv2d, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        """`relu(c(x) + res)`, one cuDNN call."""
        return torch.cudnn_convolution_add_relu(
            x, c.weight, res, 1.0, c.bias, c.stride, _padding(c), c.dilation, c.groups
        )


class DepthwiseConvs(FusedConvs):
    """`FusedConvs` with the depthwise convolutions (channels % 8 == 0) through the fvm_conv kernel.

    cuDNN runs a grouped fp16 convolution with its direct kernel, well behind ours (fp32 accumulation both; D-FINE:
    -8 to -32 % per call).
    """

    def _dw(self, c: nn.Conv2d, x: torch.Tensor, act: int) -> torch.Tensor | None:
        """`act(c(x))` through the kernel (`act`: `conv_ops` code), or None if `c` is not such a depthwise conv."""
        from fastforks._optimize.conv_ops import dwconv  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

        if not (c.groups == c.in_channels == c.out_channels and c.groups % 8 == 0 and c.dilation == (1, 1)):
            return None
        w = getattr(c, "w_rsc", None)
        if w is None:  # the weight as the kernel reads it: [R, S, C]
            w = c.weight.detach()[:, 0].permute(1, 2, 0).contiguous()
            c.register_buffer("w_rsc", w, persistent=False)
        return dwconv(x, w, c.bias, c.stride, _padding(c), act)

    def conv(self, c: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        """`c(x)`."""
        y = self._dw(c, x, 0)
        return super().conv(c, x) if y is None else y

    def conv_relu(self, c: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        """`relu(c(x))`."""
        y = self._dw(c, x, 1)
        return super().conv_relu(c, x) if y is None else y
