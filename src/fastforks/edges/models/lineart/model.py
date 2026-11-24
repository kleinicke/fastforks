"""Line art runners: weights -> network and the portable forward pass.

Inference as controlnet_aux's `LineartDetector` (ComfyUI's "Realistic Lineart"): the image resized to the detect size
(`edges._runner.to_detect`), divided by 255, the generator's sigmoid map, inverted (upstream: `255 - map` of the
8-bit map; here float, 1 = line) and resized to the input size (bilinear). `lineart-coarse` is the coarse model.

`portable`: upstream's modules in plain PyTorch, fp32 or fp16. On Apple MLX both modes are `mlx_net`: the generator
written out in `mlx.core` operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fastforks.edges._runner import Forward, Runner, to_detect, to_input
from fastforks.kernels.vol_norm import prepare_image_norms
from fastforks.shared import cnn_kernels, mlx_wrap

from .net import Generator

if TYPE_CHECKING:
    from fastforks.edges._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`.

    Attributes:
        short_side: The short side of the network input (controlnet_aux detect resolution; rounded to a multiple of 64).
    """

    short_side: int


class Portable(Runner):
    """Upstream's generator in plain PyTorch, float32 or float16."""

    Params = Params
    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        net = Generator(3, 1, 3)
        net.load_state_dict(ctx.state_dict())
        self.net = net.eval().to(ctx.device, self.dtype)  # NCHW: channels-last is 25-40 % slower (instance norm)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Line map `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        x = to_detect(image, self.ctx.params.short_side) / 255
        y = self.net(x.to(self.dtype))[:, 0].float()
        return {"edges": to_input(1 - y, image)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> edges `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["edges"])


class Kernels(Portable):
    """Channels-last convolutions with fused instance normalization and channels-last reflection padding."""

    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Fuse the image instance norms and their adjacent ReLUs."""
        super().__init__(ctx)
        prepare_image_norms(self.net, ctx.device)
        cnn_kernels.load(ctx.device)
        cnn_kernels.prepare_convs(self.net)
        cnn_kernels.patch_reflection_pads(self.net)


class _MlxPortable(Portable):
    """Apple MLX: the generator written in `mlx.core` operations (`mlx_net`), the resizes as `Portable`."""

    net: Any  # mlx_net.MlxGenerator

    def __init__(self, ctx: Context) -> None:
        Runner.__init__(self, ctx)
        from . import mlx_net  # noqa: PLC0415 - imports mlx

        net = Generator(3, 1, 3)
        net.load_state_dict(ctx.state_dict())
        self.net = mlx_net.MlxGenerator(net, mlx_wrap.run_dtype(ctx))  # the PyTorch network is not kept

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        x = to_detect(image, self.ctx.params.short_side).float().permute(0, 2, 3, 1) / 255
        return {"edges": to_input(to_torch(self.net(to_mx(x.contiguous()))), image)}

    module = Runner.module


_Mlx = mlx_wrap.port(_MlxPortable, attrs=())
RUNNERS = {"portable": Portable, "kernels": Kernels, "portable_mlx": _Mlx, "kernels_mlx": _Mlx}
