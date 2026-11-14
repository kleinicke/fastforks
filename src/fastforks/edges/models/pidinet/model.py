"""PiDiNet runners: weights -> network and the portable forward pass.

Inference as controlnet_aux's `PidiNetDetector` (ComfyUI's "PiDiNet Soft-Edge Lines"): the image resized to the
detect size (`edges._runner.to_detect`), BGR, divided by 255; the fused (last) sigmoid map; `apply_filter` thresholds at
0.5 and `safe` quantizes as upstream's `safe_step` (3 levels); then resized to the input size (bilinear). Upstream's
`scribble` mode is not ported.

`portable`: the pixel-difference convolutions converted to plain 3x3 / 5x5 depthwise convolutions at load
(`PDCBlock_converted`, the authors' re-parameterization; the same function), fp32 or fp16 (channels-last on CUDA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks.edges._runner import Forward, Runner, safe_step, to_detect, to_input

from .net import PiDiNet, nets

if TYPE_CHECKING:
    from fastforks.edges._runner import Context

CONFIG = "carv4"  # fixed: controlnet_aux `pidinet()`: 60 channels, dilation 24, spatial attention
_AD = [3, 0, 1, 6, 4, 2, 7, 8, 5]  # the angular difference's clockwise neighbour of each tap


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`.

    Attributes:
        resolution: Detect resolution: the short side of the network input (rounded to a multiple of 64).
        safe: Quantize the map to 3 levels as controlnet_aux's `safe=True` (`safe_step`).
        apply_filter: Threshold the map at 0.5 as controlnet_aux's `apply_filter=True`.
    """

    resolution: int
    safe: bool
    apply_filter: bool


def _vanilla(op: str, w: torch.Tensor) -> torch.Tensor:
    """A pixel-difference kernel as the plain kernel of the same function (`createConvFunc`, dilation 1)."""
    if op == "cv":
        return w
    if op == "cd":  # conv(x, w) - conv(x, sum(w)) at the centre tap
        w = w.clone()
        w[:, :, 1, 1] -= w.sum((2, 3))
        return w
    flat = w.flatten(2)
    if op == "ad":
        return (flat - flat[:, :, _AD]).view_as(w)
    out = w.new_zeros(*w.shape[:2], 25)  # rd: the 5x5 radial difference
    out[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = flat[:, :, 1:]
    out[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -flat[:, :, 1:]
    return out.view(*w.shape[:2], 5, 5)


def convert(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """controlnet_aux's checkpoint (PDC kernels, `module.` stripped) -> the `convert=True` network's state dict."""
    sd = dict(sd)
    ops = [nets[CONFIG][f"layer{i}"] for i in range(16)]
    blocks = [f"block{s}_{j}" for s, n in ((1, 3), (2, 4), (3, 4), (4, 4)) for j in range(1, n + 1)]
    names = ["init_block.weight"] + [f"{b}.conv1.weight" for b in blocks]
    for op, name in zip(ops, names, strict=True):
        sd[name] = _vanilla(op, sd[name])
    return sd


class Portable(Runner):
    """The converted network in plain PyTorch, float32 or float16 (channels-last on CUDA)."""

    Params = Params
    dtypes = ("float32", "float16")
    compiled_methods = ("net",)

    def __init__(self, ctx: Context) -> None:
        """Build the network (`convert=True`) from the PDC checkpoint."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        net = PiDiNet(60, [nets[CONFIG][f"layer{i}"] for i in range(16)], dil=24, sa=True, convert=True)
        net.load_state_dict(convert(ctx.state_dict(key="state_dict")))
        self.cl = ctx.device.type == "cuda"
        fmt = torch.channels_last if self.cl else torch.contiguous_format
        self.net = net.eval().to(ctx.device, self.dtype).to(memory_format=fmt)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Soft edge map `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        x = to_detect(image, self.ctx.params.resolution).flip(1) / 255  # upstream feeds BGR
        if self.cl:
            x = x.contiguous(memory_format=torch.channels_last)
        edge = self.net(x.to(self.dtype))[-1][:, 0].float()
        if self.ctx.params.apply_filter:
            edge = (edge > 0.5).float()  # noqa: PLR2004 - upstream's threshold
        if self.ctx.params.safe:
            edge = safe_step(edge)
        return {"edges": to_input(edge, image)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> edges `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["edges"])


RUNNERS = {"portable": Portable}
