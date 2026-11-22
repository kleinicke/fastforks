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
from fastforks.kernels import smish
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.shared.teed import TED, edge_map
from fastforks.shared.teed.net.Xsmish import Smish

if TYPE_CHECKING:
    from fastforks.edges._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`.

    Attributes:
        short_side: The short side of the network input (controlnet_aux detect resolution; rounded to a multiple of 64).
        safe_steps: Quantization steps of the map (controlnet_aux `safe_step`); 0 = the soft map.
        cuda_graphs: Kernels replays a CUDA graph for small repeated shapes.
    """

    short_side: int
    safe_steps: int
    cuda_graphs: bool = True


class Portable(Runner):
    """Upstream's network in plain PyTorch, float32 or float16."""

    # images per network call: launch bound (RTX 5060 Ti, Windows, fp16, 8 images, ms per image: 512 px 3.5 -> 2.0,
    # 1024 px 4.8 -> 2.7); the network runs at the detect size, so memory per image does not grow with the input
    batch = 16

    Params = Params
    dtypes = ("float32", "float16")

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
        return {"edges": to_input(self._edges(to_detect(image, self.ctx.params.short_side)), image)}

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        """The edge map `[B, h, w]` at the detect size of `x` `[B, 3, h, w]` in 0..255."""
        edge = edge_map(self.net(x.to(self.dtype)))
        if self.ctx.params.safe_steps:
            edge = safe_step(edge, self.ctx.params.safe_steps)
        return edge

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> edges `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["edges"])


# kernels replays a CUDA graph up to this many images x H x W at the network size: launch bound (RTX 5060 Ti,
# Windows, fp16: one 512 px image 2.97 -> 1.71 ms, 1024 x 768 3.5 -> 2.6);
# batches beyond it gain nothing (the batch amortizes the launches)
GRAPH_PIXELS = 4 * 512 * 512  # fixed: a speed threshold, not the architecture


class Kernels(Portable):
    """Run TED's Smish activations as fused CUDA operations."""

    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Replace module Smish calls while preserving the original graph's tensor lifetimes."""
        super().__init__(ctx)
        for name, module in list(self.net.named_modules(remove_duplicate=False)):
            if isinstance(module, Smish):
                self.net.set_submodule(name, smish.Activation(ctx.device))
        forward = super()._edges
        self.graphs = SmallShapeGraphs(forward, pixels=GRAPH_PIXELS, enabled=ctx.params.cuda_graphs)

    def release_graphs(self) -> None:
        """Release the CUDA graphs and their pool."""
        self.graphs.release()

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        """`Portable._edges`, from a CUDA graph for a small repeated shape."""
        return self.graphs(x)


class _OneImage(Portable):
    """Portable for the MLX port: one image per call, as before batching (not measured on Apple GPUs)."""

    batch = 1


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_OneImage),
    "kernels_mlx": mlx_wrap.port(_OneImage),
}
