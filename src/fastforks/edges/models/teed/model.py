"""TEED runners: weights -> network and the portable forward pass.

Inference as controlnet_aux's `TEEDdetector` (ComfyUI's "TEED Soft-Edge Lines"): the image resized to the detect size
(`edges._runner.to_detect`), as 0..255 floats, the four outputs resized to the detect size (bilinear, OpenCV's
`INTER_LINEAR`), the sigmoid of their mean (`fastforks.shared.teed.edge_map`), `safe_step(edge, safe_steps)`, then
resized to the input size (bilinear).

`portable`: upstream's modules in plain PyTorch, fp32 or fp16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks.edges._runner import Forward, Runner, safe_step, to_detect, to_input
from fastforks.shared.teed import TED, edge_map

if TYPE_CHECKING:
    from fastforks.edges._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`.

    Attributes:
        resolution: Detect resolution: the short side of the network input (rounded to a multiple of 64).
        safe_steps: Quantization steps of the map (controlnet_aux `safe_step`); 0 = the soft map.
    """

    resolution: int
    safe_steps: int


class Portable(Runner):
    """Upstream's network in plain PyTorch, float32 or float16."""

    Params = Params
    dtypes = ("float32", "float16")
    compiled_methods = ("net",)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        net = TED()
        net.load_state_dict(ctx.state_dict())
        # NCHW: channels-last measured 20-30 % slower and +20 % memory at 1024 px (RTX 5060 Ti)
        self.net = net.eval().to(ctx.device, self.dtype)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Soft edge map `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        x = to_detect(image, self.ctx.params.resolution)
        edge = edge_map(self.net(x.to(self.dtype)))
        if self.ctx.params.safe_steps:
            edge = safe_step(edge, self.ctx.params.safe_steps)
        return {"edges": to_input(edge, image)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> edges `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["edges"])


RUNNERS = {"portable": Portable}
