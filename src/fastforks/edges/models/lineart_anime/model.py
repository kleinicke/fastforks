"""Anime line art runners: weights -> network and the portable forward pass.

Inference as controlnet_aux's `LineartAnimeDetector` (ComfyUI's "Anime Lineart"): the image resized to the detect size
(`edges._runner.to_detect`), then with OpenCV's `INTER_CUBIC` (uint8) to the next multiple of 256, scaled to [-1, 1],
the U-Net's tanh map back to 0..255, `INTER_CUBIC` (float) to the detect size, clipped and inverted (upstream: `255 -
map` of the 8-bit map; here float, 1 = line), then resized to the input size (bilinear).

`portable`: upstream's modules in plain PyTorch, fp32 or fp16.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from fastforks.edges._runner import Forward, Runner, to_detect, to_detect_cuda, to_input
from fastforks.kernels import cv_resample
from fastforks.shared import mlx_wrap
from fastforks.shared.cv2_resize import cv2_resize

from .net import UnetGenerator

if TYPE_CHECKING:
    from fastforks.edges._runner import Context

PAD = 256  # fixed: the U-Net's 8 stride-2 levels; upstream resizes the detect image to a multiple of 256


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`.

    Attributes:
        short_side: The short side of the network input (controlnet_aux detect resolution; rounded to a multiple of 64).
    """

    short_side: int


class Portable(Runner):
    """Upstream's U-Net in plain PyTorch, float32 or float16."""

    _detect = staticmethod(to_detect)
    _resize = staticmethod(cv2_resize)
    Params = Params
    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        norm = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        net = UnetGenerator(3, 1, 8, 64, norm_layer=norm, use_dropout=False)
        net.load_state_dict(ctx.state_dict())  # `module.` (DataParallel) is stripped
        self.net = net.eval().to(ctx.device, self.dtype)  # NCHW: channels-last is 25-40 % slower (instance norm)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Line map `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        x = self._detect(image, self.ctx.params.short_side)
        h, w = x.shape[-2:]
        hn, wn = PAD * math.ceil(h / PAD), PAD * math.ceil(w / PAD)
        if (hn, wn) != (h, w):
            x = self._resize(x, hn, wn, "cubic")
        x = x / 127.5 - 1
        y = self.net(x.to(self.dtype))[:, :1].float() * 127.5 + 127.5
        if (hn, wn) != (h, w):
            y = self._resize(y, h, w, "cubic", levels=False)
        return {"edges": to_input(1 - y[:, 0].clamp(0, 255) / 255, image)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> edges `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["edges"])


class Kernels(Portable):
    """Use fused OpenCV-compatible edge preprocessing on CUDA."""

    module = Runner.module
    _resize = staticmethod(cv_resample.resize)

    _detect = staticmethod(to_detect_cuda)


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable),
}
