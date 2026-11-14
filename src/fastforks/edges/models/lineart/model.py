"""Line art runners: weights -> network and the portable forward pass.

Inference as controlnet_aux's `LineartDetector` (ComfyUI's "Realistic Lineart"): the image resized to the detect size
(`edges._runner.to_detect`), divided by 255, the generator's sigmoid map, inverted (upstream: `255 - map` of the
8-bit map; here float, 1 = line) and resized to the input size (bilinear). `lineart-coarse` is the coarse model.

`portable`: upstream's modules in plain PyTorch, fp32 or fp16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks.edges._runner import Forward, Runner, to_detect, to_input

from .net import Generator

if TYPE_CHECKING:
    from fastforks.edges._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`.

    Attributes:
        resolution: Detect resolution: the short side of the network input (rounded to a multiple of 64).
    """

    resolution: int


class Portable(Runner):
    """Upstream's generator in plain PyTorch, float32 or float16."""

    Params = Params
    dtypes = ("float32", "float16")
    compiled_methods = ("net",)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        net = Generator(3, 1, 3)
        net.load_state_dict(ctx.state_dict())
        self.net = net.eval().to(ctx.device, self.dtype)  # NCHW: channels-last is 25-40 % slower (instance norm)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Line map `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        x = to_detect(image, self.ctx.params.resolution) / 255
        y = self.net(x.to(self.dtype))[:, 0].float()
        return {"edges": to_input(1 - y, image)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> edges `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["edges"])


RUNNERS = {"portable": Portable}
