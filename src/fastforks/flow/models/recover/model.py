"""ReCoVEr runners: weights -> network and the portable / kernels forward passes.

Both fold the encoders' batch norms into their convolutions. Portable on CUDA in float16 runs the whole network in
fp16 channels-last (the flow state, the softmax of the upsampling weights and the convex upsampling in float32);
kernels adds fp16-accumulate GEMMs (the ConvNeXt MLPs) and cuDNN fp16-accumulate convolutions.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import nn

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.flow._runner import Forward, Runner

from .net import BasicBlock, ReCoVEr, ResNetFPN

if TYPE_CHECKING:
    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` and `[options]`."""

    arch: Literal["rn", "mn", "cx"]  # context encoder: ResNet34, MobileNetV3-Large, ConvNeXt-T stages
    iters: int  # flow updates after the initial regression (upstream `iters`, 4)


@torch.no_grad()
def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    """`bn(conv(x))` -> `conv'(x)` (inference statistics)."""
    mean, var = cast("torch.Tensor", bn.running_mean), cast("torch.Tensor", bn.running_var)  # eval statistics
    s = bn.weight / torch.sqrt(var + bn.eps)
    conv.weight.data = conv.weight.data * s.view(-1, 1, 1, 1)
    b = conv.bias.data if conv.bias is not None else torch.zeros_like(mean)
    conv.bias = nn.Parameter((b - mean) * s + bn.bias)


def fold_batchnorms(net: nn.Module) -> int:
    """Fold every conv -> batch norm pair of the encoders; returns how many."""
    n = 0
    for m in list(net.modules()):
        if isinstance(m, nn.Sequential):
            for i in range(len(m) - 1):
                conv, bn = m[i], m[i + 1]
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    _fold(conv, bn)
                    m[i + 1] = nn.Identity()
                    n += 1
        if isinstance(m, (BasicBlock, ResNetFPN)):
            pairs = [("conv1", "bn1")] + ([("conv2", "bn2")] if isinstance(m, BasicBlock) else [])
            for c, b in pairs:
                _fold(getattr(m, c), getattr(m, b))
                setattr(m, b, nn.Identity())
                n += 1
            if isinstance(m, BasicBlock) and m.downsample is not None:
                m.bn3 = nn.Identity()  # ty: ignore[invalid-assignment]  # folded with downsample.0 by the Sequential pass
    return n


def _hacc_forward(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    from fastforks.kernels.cudnn_hacc import conv_half_acc  # noqa: PLC0415 - kernels only

    return conv_half_acc(x.half(), self.weight, self.bias, list(self.padding), list(self.stride))  # ty: ignore


def _hacc_convs(module: nn.Module) -> int:
    """Route every fp16 Conv2d (groups 1, dilation 1, zero padding) through cuDNN fp16-accumulate convs."""
    n = 0
    for m in module.modules():
        if (
            is_base(m, nn.Conv2d)
            and m.groups == 1
            and m.dilation == (1, 1)
            and m.padding_mode == "zeros"
            and not isinstance(m.padding, str)
            and m.weight.dtype == torch.float16
        ):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
            override(m, _hacc_forward)
            n += 1
    return n


class Portable(Runner):
    """Plain PyTorch: float32, or on CUDA the fp16 network (see the module docstring)."""

    dtypes = ("float32", "float16")
    Params = Params
    gacc = False  # fp16-accumulate GEMMs

    def __init__(self, ctx: Context) -> None:
        """Build the network, fold its batch norms, convert it."""
        super().__init__(ctx)
        self.net = ReCoVEr(ctx.params.arch)
        sd = {k: v for k, v in ctx.state_dict().items() if not k.startswith("fnet.")}  # unused by upstream
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.eval()
        fold_batchnorms(self.net)
        dt = torch.float16 if ctx.dtype == "float16" else torch.float32
        self.layout = torch.channels_last if ctx.device.type == "cuda" else torch.contiguous_format
        self.net.to(ctx.device, dt).to(memory_format=self.layout)

    def accumulate(self) -> contextlib.AbstractContextManager[None]:
        """fp16-accumulate GEMMs for kernels."""
        return fp16_accumulate() if self.gacc else contextlib.nullcontext()

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        image1, image2 = (x.contiguous(memory_format=self.layout) for x in (image1, image2))
        with torch.autocast(image1.device.type, enabled=False), self.accumulate():
            return {"flow": self.net(image1, image2, self.ctx.params.iters)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, image1, image2: net(image1, image2, iters))


class Kernels(Portable):
    """Portable's fp16 network with fp16-accumulate GEMMs and cuDNN fp16-accumulate convolutions."""

    gacc = True

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its dense convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        if _hacc_convs(self.net) == 0:
            msg = "ReCoVEr kernels: no conv routed through cudnn_hacc"
            raise RuntimeError(msg)

    def module(self) -> torch.nn.Module:
        """No exportable module: kernel launches."""
        return Runner.module(self)


RUNNERS = {"portable": Portable, "kernels": Kernels}
