"""SPAN runners: the network of hongyuanyu/SPAN and BasicSR's test call (`net(lq)`, clamped to `[0, 1]`).

Upstream's `Conv3XC` re-fuses its training branches (1 x 1 -> 3 x 3 -> 1 x 1 + 1 x 1 skip) into `eval_conv` on every
eval forward; here they are fused once at load (`update_params`, upstream's own code) and the forward is the fused 3 x 3
conv (+ leaky ReLU where upstream has one). The input mean is a plain attribute upstream (`type_as` per call), here a
buffer of the network's dtype and device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.kernels import cudnn_hacc, sr_ops
from fastforks.restoration._runner import Forward, Runner
from fastforks.shared import mlx_wrap

from .net import SPAN
from .net.span_arch import SPAB, Conv3XC

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`."""

    scale: int  # 4 or 2
    channels: int  # feature channels (48 in the release models)


def _fused(m: Conv3XC, x: torch.Tensor) -> torch.Tensor:
    """`Conv3XC.forward` in eval mode, without re-fusing the weights."""
    out = m.eval_conv(x)
    return F.leaky_relu(out, negative_slope=0.05) if m.has_relu else out


def _skeleton(p: Params) -> SPAN:
    """The network of `p` without its checkpoint (built on the CPU: upstream makes its mean with `torch.Tensor`)."""
    with torch.device("cpu"):
        return SPAN(3, 3, feature_channels=p.channels, upscale=p.scale)


def _network(ctx: Context) -> SPAN:
    """The upstream network with the checkpoint, every `Conv3XC`'s branches fused into `eval_conv` (float32, CPU)."""
    net = _skeleton(ctx.params)
    load_strict(net, ctx.state_dict(key="params_ema"), f"{ctx.variant.ref} checkpoint")
    for m in net.modules():
        if isinstance(m, Conv3XC):
            m.update_params()
    return net.eval()


class Portable(Runner):
    """Plain PyTorch in the card's dtype."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and fuse its `Conv3XC` branches once."""
        super().__init__(ctx)
        self.net = _network(ctx)
        for m in self.net.modules():
            if isinstance(m, Conv3XC):
                del m.sk, m.conv  # training branches, folded into eval_conv
                override(m, _fused)
        mean = self.net.mean
        del self.net.mean
        self.net.register_buffer("mean", mean, persistent=False)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))
        if ctx.device.type == "cuda" and ctx.dtype == "float16":
            # channels-last weights: cuDNN's fp16 kernels skip the NCHW<->NHWC conversion around each conv
            # (512x512 x2 23.6 -> 17.2 ms, peak 606 -> 498 MB, RTX 5060 Ti, Windows); float32 stays NCHW
            self.net.to(memory_format=torch.channels_last)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        return {"image": self.net(image.to(getattr(torch, self.ctx.dtype)))}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype."""
        dtype = getattr(torch, self.ctx.dtype)
        return Forward(self.net, lambda net, image: net(image.to(dtype)))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


def _span_block(m: SPAB, x: torch.Tensor, *, keep_tap: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    first = m.act1(m.c1_r(x))
    second = m.act1(m.c2_r(first))
    tap = first if keep_tap else None
    del first
    return sr_ops.residual(m.c3_r(second), x, attention=True), tap


def _span(m: SPAN, x: torch.Tensor) -> torch.Tensor:
    """Consume the six block outputs as they run; only the first output and the final block's tap feed conv_cat."""
    base = m.conv_1((x - m.mean) * m.img_range)
    first, _ = _span_block(m.block_1, base)
    y = first
    for block in (m.block_2, m.block_3, m.block_4, m.block_5):
        y, _ = _span_block(block, y)
    y, tap = _span_block(m.block_6, y, keep_tap=True)
    assert tap is not None  # noqa: S101 - keep_tap=True returns the intermediate activation
    return m.upsampler(m.conv_cat(torch.cat((base, m.conv_2(y), first, tap), 1)))


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convolutions (`cudnn_hacc`)."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        sr_ops.load(ctx.device)
        override(self.net, _span)
        for m in self.net.modules():
            if is_base(m, torch.nn.Conv2d):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        x = image.to(torch.float16).contiguous(memory_format=torch.channels_last)
        with fp16_accumulate():
            return {"image": self.net(x)}


class _MlxPortable(Portable):
    """Apple MLX: the network written in MLX operations (`mlx_net`)."""

    module = Runner.module  # the MLX network is not exportable
    KERNELS = False
    CACHE_MB: ClassVar[dict[str, int]] = {}  # the buffer cache this mode asks for, by variant ('*': all)

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts and fuses the checkpoint's weights)."""
        Runner.__init__(self, ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        kernels = self.KERNELS and ctx.dtype == "float16"  # float32 runs as portable (as the graph port's kernels)
        self.mlx = mlx_port.recorded(
            ctx,
            f"{mlx_net.LAYOUT}-{'kernels' if kernels else 'portable'}",
            lambda net: mlx_net.network(net, ctx.dtype, kernels=kernels),
            real=lambda: _network(ctx),
            skeleton=lambda: _skeleton(ctx.params),
            sources=mlx_net.SOURCES,
        )
        self.net = None
        if mb := self.CACHE_MB.get(ctx.variant.id, self.CACHE_MB.get("*", 0)):  # held while this runner lives
            from fastforks._core.mlx import request_cache  # noqa: PLC0415 - MLX only

            request_cache(self, mb)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        return {"image": to_torch(self.mlx(to_mx(image.float())))}


class _MlxKernels(_MlxPortable):
    """Apple MLX, float16: the wide 3 x 3 convs through `conv_mlx`'s Winograd kernel."""

    KERNELS = True
    # 512 x 512 float16: 72 / 65 / 61 ms at a 128 / 512 / 1024 MB buffer cache (the graph port's 512)
    CACHE_MB: ClassVar[dict[str, int]] = {"*": 512}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
    "kernels_mlx": _MlxKernels,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph ports the hand-written network replaced
GRAPH_RUNNERS = {
    "portable_mlx": mlx_wrap.port(Portable),
    "kernels_mlx": mlx_wrap.port(Portable, kernels_fp16_only=True, cache_mb=512),
}


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load(path)`: a SPAN state dict as a variant and its params (feature channels, scale); keys unchanged."""
    if "block_1.c1_r.sk.weight" not in sd or "conv_1.sk.weight" not in sd:
        return None
    first, up = sd["conv_1.sk.weight"], sd.get("upsampler.0.weight")
    scale = round((up.shape[0] / 3) ** 0.5) if up is not None else 0
    problems = []
    if "no_norm" in sd:
        problems.append("no input normalisation (fastforks: the DIV2K mean, x 255)")
    if first.shape[1] != 3 or up is None or up.shape[0] != 3 * scale**2:  # noqa: PLR2004 - RGB
        problems.append("not 3 input and output channels")
    if problems:
        msg = f"SPAN checkpoint with {', '.join(problems)}: not supported"
        raise WeightsError(msg)
    variant = f"span-x{scale}" if scale in {2, 4} else "span-x4"
    return variant, {"scale": scale, "channels": first.shape[0]}, dict(sd)
