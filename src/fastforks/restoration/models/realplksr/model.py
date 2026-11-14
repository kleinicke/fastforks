"""RealPLKSR runners: neosr's `realplksr` network and its test call (`net(lq)`, clamped to `[0, 1]`).

The network is upstream's (eval path: the large-kernel conv writes into the first channels in place); on CUDA it runs
channels-last. GroupNorm's group count is not stored in a checkpoint: 4, neosr's and spandrel's default.

GroupNorm runs through `fastforks.shared.group_norm.group_norm_stats` at every size (it also wins at 128-192 px):
PyTorch's channels-last GroupNorm took 4.2 ms per block at 512 x 512 (40 % of the network), this 0.6 ms; the result
is within fp16 rounding of float32 GroupNorm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import override
from fastforks.kernels import cudnn_hacc
from fastforks.restoration._runner import Forward, Runner
from fastforks.shared.group_norm import fast_group_norms

from .net import realplksr

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`."""

    scale: int  # pixel-shuffle factor
    dim: int  # feature channels
    n_blocks: int  # PLK blocks
    kernel_size: int  # the partial large-kernel conv
    split_ratio: float  # share of the channels the large kernel convolves
    use_ea: bool  # element-wise attention in each block


class Portable(Runner):
    """Plain PyTorch in the card's dtype, channels-last on CUDA."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        p = ctx.params
        self.net = realplksr(
            dim=p.dim,
            n_blocks=p.n_blocks,
            upscaling_factor=p.scale,
            kernel_size=p.kernel_size,
            split_ratio=p.split_ratio,
            use_ea=p.use_ea,
        )
        load_strict(self.net, ctx.state_dict(), f"{ctx.variant.ref} checkpoint")
        self.fmt = torch.channels_last if torch.device(ctx.device).type == "cuda" else torch.contiguous_format
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype)).to(memory_format=self.fmt)
        fast_group_norms(self.net, always=True)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        return {"image": self.net(image.to(getattr(torch, self.ctx.dtype)).contiguous(memory_format=self.fmt))}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype."""
        dtype, fmt = getattr(torch, self.ctx.dtype), self.fmt
        return Forward(self.net, lambda net, image: net(image.to(dtype).contiguous(memory_format=fmt)))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convolutions (`cudnn_hacc`)."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if type(m) is torch.nn.Conv2d:
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        x = image.to(torch.float16).contiguous(memory_format=self.fmt)
        with fp16_accumulate():
            return {"image": self.net(x)}


RUNNERS = {"portable": Portable, "kernels": Kernels}

_BLOCK = re.compile(r"feats\.(\d+)\.lk\.conv\.weight")


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load_file`: a neosr `realplksr` state dict as a variant and its params; keys unchanged."""
    if "feats.0.weight" not in sd or "feats.1.lk.conv.weight" not in sd or "feats.1.channel_mixer.0.weight" not in sd:
        return None
    first, lk = sd["feats.0.weight"], sd["feats.1.lk.conv.weight"]
    blocks = sum(1 for k in sd if _BLOCK.fullmatch(k))
    last = sd.get(f"feats.{blocks + 2}.weight")  # after the blocks and the dropout
    scale = round((last.shape[0] / 3) ** 0.5) if last is not None else 0
    problems = []
    if "to_img.init_pos" in sd:
        problems.append("a DySample upsampler (fastforks: pixel shuffle)")
    if "feats.1.layer_norm.bias" in sd or "feats.1.norm.weight" not in sd:
        problems.append("no GroupNorm (fastforks: neosr's GroupNorm blocks)")
    if first.shape[1] != 3 or last is None or last.shape[0] != 3 * scale**2:  # noqa: PLR2004 - RGB
        problems.append("not 3 input and output channels")
    if problems:
        msg = f"RealPLKSR checkpoint with {', '.join(problems)}: not supported"
        raise WeightsError(msg)
    params = {
        "scale": scale,
        "dim": first.shape[0],
        "n_blocks": blocks,
        "kernel_size": lk.shape[-1],
        "split_ratio": lk.shape[0] / first.shape[0],
        "use_ea": "feats.1.attn.f.0.weight" in sd,
    }
    return "realplksr-nomoswebphoto-x4", params, dict(sd)
