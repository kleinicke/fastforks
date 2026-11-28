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
from fastforks.shared.cuda_graphs import SmallShapeGraphs
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
        cuda_graphs: Kernels replays a CUDA graph for small repeated shapes.
    """

    short_side: int
    cuda_graphs: bool = True


def _unet() -> UnetGenerator:
    """The (untrained) U-Net."""
    norm = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    return UnetGenerator(3, 1, 8, 64, norm_layer=norm, use_dropout=False)


class Portable(Runner):
    """Upstream's U-Net in plain PyTorch, float32 or float16."""

    # images per network call: launch bound (RTX 5060 Ti, Windows, fp16, 8 images, ms per image: 512 px 4.4 -> 2.6,
    # 1024 px 4.7 -> 3.3); the network runs at the detect size, so memory per image does not grow with the input
    batch = 16

    _detect = staticmethod(to_detect)
    _resize = staticmethod(cv2_resize)
    Params = Params
    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        net = _unet()
        net.load_state_dict(ctx.state_dict())  # `module.` (DataParallel) is stripped
        self.net = net.eval().to(ctx.device, self.dtype)  # NCHW: channels-last is 25-40 % slower (instance norm)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Line map `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        x = self._detect(image, self.ctx.params.short_side)
        h, w = x.shape[-2:]
        hn, wn = PAD * math.ceil(h / PAD), PAD * math.ceil(w / PAD)
        if (hn, wn) != (h, w):
            x = self._resize(x, hn, wn, "cubic")
        y = self._lines(x)
        if (hn, wn) != (h, w):
            y = self._resize(y, h, w, "cubic", levels=False)
        return {"edges": to_input(1 - y[:, 0].clamp(0, 255) / 255, image)}

    def _lines(self, x: torch.Tensor) -> torch.Tensor:
        """The network's line levels `[B, 1, H, W]` (0..255) of the padded `x` `[B, 3, H, W]` in 0..255."""
        return self.net((x / 127.5 - 1).to(self.dtype))[:, :1].float() * 127.5 + 127.5

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> edges `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["edges"])


# kernels replays a CUDA graph up to this many images x H x W at the network size: launch bound (RTX 5060 Ti,
# Windows, fp16: one 512 px image 5.0 -> 3.8 ms, 1024 x 768 6.3 -> 6.0);
# batches beyond it gain nothing (the batch amortizes the launches)
GRAPH_PIXELS = 4 * 512 * 512  # fixed: a speed threshold, not the architecture


class Kernels(Portable):
    """Use fused OpenCV-compatible edge preprocessing on CUDA."""

    module = Runner.module
    _resize = staticmethod(cv_resample.resize)

    _detect = staticmethod(to_detect_cuda)

    def __init__(self, ctx: Context) -> None:
        """The portable network, replayed from CUDA graphs for small repeated shapes."""
        super().__init__(ctx)
        forward = super()._lines
        self.graphs = SmallShapeGraphs(forward, pixels=GRAPH_PIXELS, enabled=ctx.params.cuda_graphs)

    def release_graphs(self) -> None:
        """Release the CUDA graphs and their pool."""
        self.graphs.release()

    def _lines(self, x: torch.Tensor) -> torch.Tensor:
        """`Portable._lines`, from a CUDA graph for a small repeated shape."""
        return self.graphs(x)


class _OneImage(Portable):
    """Portable for the MLX port: one image per call, as before batching (not measured on Apple GPUs)."""

    batch = 1


class _MlxPortable(_OneImage):
    """Apple MLX: the U-Net written in MLX operations (`mlx_net`); one image per call."""

    module = Runner.module  # the MLX network is not exportable

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts the checkpoint's weights)."""
        Runner.__init__(self, ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        def real() -> UnetGenerator:
            net = _unet()
            net.load_state_dict(ctx.state_dict())
            return net.eval()

        self.mlx = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            lambda net: mlx_net.network(net, ctx.dtype),
            real=real,
            skeleton=_unet,
            sources=mlx_net.SOURCES,
        )
        from fastforks._core.mlx import request_cache  # noqa: PLC0415 - imports mlx

        request_cache(self, 512)  # the buffer cache the network's temporaries reuse (held while this runner lives)

    def _lines(self, x: torch.Tensor) -> torch.Tensor:
        """The line levels `[B, 1, H, W]` (0..255) of the padded `x` `[B, 3, H, W]` in 0..255."""
        from fastforks._core.mlx import mx, to_mx, to_torch  # noqa: PLC0415 - imports mlx

        y = self.mlx(to_mx((x / 127.5 - 1).permute(0, 2, 3, 1).contiguous()).astype(getattr(mx, self.ctx.dtype)))
        return to_torch(y.astype(mx.float32)).permute(0, 3, 1, 2) * 127.5 + 127.5


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaces
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(_OneImage)}
