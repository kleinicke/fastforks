"""HED runners: weights -> network and the portable forward pass.

Inference as controlnet_aux's `HEDdetector` (ComfyUI's "HED Soft-Edge Lines"): the image resized to the detect size
(`edges._runner.to_detect`), as 0..255 floats (the network subtracts its own learned mean), the five side outputs
resized to the detect size (bilinear, OpenCV's `INTER_LINEAR`), averaged, sigmoid; `safe` quantizes as upstream's
`safe_step` (3 levels); then resized to the input size (bilinear). Upstream's `scribble` mode is not ported.

`portable`: upstream's modules in plain PyTorch, fp32 or fp16 (channels-last on CUDA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.edges._runner import Forward, Runner, safe_step, to_detect, to_input

from .net import ControlNetHED_Apache2

if TYPE_CHECKING:
    from fastforks.edges._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`.

    Attributes:
        resolution: Detect resolution: the short side of the network input (rounded to a multiple of 64).
        safe: Quantize the map to 3 levels as controlnet_aux's `safe=True` (`safe_step`).
    """

    resolution: int
    safe: bool


class Portable(Runner):
    """Upstream's network in plain PyTorch, float32 or float16 (channels-last on CUDA)."""

    Params = Params
    dtypes = ("float32", "float16")
    compiled_methods = ("net",)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        net = ControlNetHED_Apache2()
        net.load_state_dict(ctx.state_dict())
        self.cl = ctx.device.type == "cuda"
        fmt = torch.channels_last if self.cl else torch.contiguous_format
        self.net = net.eval().to(ctx.device, self.dtype).to(memory_format=fmt)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Soft edge map `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        x = to_detect(image, self.ctx.params.resolution)
        size = x.shape[-2:]
        if self.cl:
            x = x.contiguous(memory_format=torch.channels_last)
        sides = self.net(x.to(self.dtype))
        acc = sides[0][:, 0].float()
        for e in sides[1:]:
            acc = acc + F.interpolate(e.float(), size, mode="bilinear", align_corners=False)[:, 0]
        edge = torch.sigmoid(acc / len(sides))
        if self.ctx.params.safe:
            edge = safe_step(edge)
        return {"edges": to_input(edge, image)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> edges `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["edges"])


RUNNERS = {"portable": Portable}
