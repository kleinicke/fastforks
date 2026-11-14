"""Real-ESRGAN Compact runners: `SRVGGNetCompact` and the call of upstream `RealESRGANer.enhance`.

As upstream (`inference_realesrgan.py -n realesr-general-x4v3 -dn 1`, no tiling, `pre_pad` 0): the RGB image in
`[0, 1]` goes through the network and the output is clamped to `[0, 1]`. Upstream runs float16 by default (`--fp32`
for float32). The network is upstream's; on CUDA it runs channels-last.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.kernels import cudnn_hacc
from fastforks.restoration._runner import Forward, Runner

from .net import SRVGGNetCompact

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`."""

    scale: int  # pixel-shuffle factor of the last conv
    num_feat: int  # feature channels
    num_conv: int  # 3 x 3 convs between the first and the last


class Portable(Runner):
    """Plain PyTorch in the card's dtype, channels-last on CUDA."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        p = ctx.params
        self.net = SRVGGNetCompact(3, 3, num_feat=p.num_feat, num_conv=p.num_conv, upscale=p.scale, act_type="prelu")
        load_strict(self.net, ctx.state_dict(key="params"), f"{ctx.variant.ref} checkpoint")
        self.fmt = torch.channels_last if torch.device(ctx.device).type == "cuda" else torch.contiguous_format
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype)).to(memory_format=self.fmt)

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
            if is_base(m, torch.nn.Conv2d):
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        x = image.to(torch.float16).contiguous(memory_format=self.fmt)
        with fp16_accumulate():
            return {"image": self.net(x)}


RUNNERS = {"portable": Portable, "kernels": Kernels}

_BODY = re.compile(r"body\.(\d+)\.(weight|bias)")


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load_file`: an `SRVGGNetCompact` state dict (Real-ESRGAN Compact) as a variant and params; keys unchanged."""
    first = sd.get("body.0.weight")
    if first is None or first.ndim != 4 or not all(_BODY.fullmatch(k) for k in sd):  # noqa: PLR2004 - a conv
        return None
    convs = sorted(int(m.group(1)) for k, v in sd.items() if (m := _BODY.fullmatch(k)) and v.ndim == 4)  # noqa: PLR2004
    last = sd[f"body.{convs[-1]}.weight"]
    scale = round((last.shape[0] / 3) ** 0.5)
    problems = []
    if convs != list(range(0, 2 * len(convs) - 1, 2)):
        problems.append("not alternating convs and activations")
    if "body.1.weight" not in sd:
        problems.append("activations without parameters (fastforks: PReLU)")
    if first.shape[1] != 3 or last.shape[0] != 3 * scale**2:  # noqa: PLR2004 - RGB
        problems.append("not 3 input and output channels")
    if problems:
        msg = f"SRVGGNetCompact checkpoint with {', '.join(problems)}: not supported"
        raise WeightsError(msg)
    params = {"scale": scale, "num_feat": first.shape[0], "num_conv": len(convs) - 2}
    return "realesr-general-x4v3", params, dict(sd)
