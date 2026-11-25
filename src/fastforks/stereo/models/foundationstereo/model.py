"""FoundationStereo runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from fastforks._core.checkpoint import Inert, latin1_encode, load_strict, numeric_scalar
from fastforks.shared import mlx_wrap
from fastforks.stereo._runner import Forward, Runner

from .net.core.foundation_stereo import FoundationStereo
from .net.core.submodule import PositionalEmbedding
from .net.inference import Opts, forward_fast, prepare_inference

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context

    from .net.core.submodule import CostVolumeDisparityAttention


def _extra() -> dict[tuple[str, str], object]:
    """What the released checkpoints pickle besides tensors: training state and the OmegaConf config (inert)."""
    scalar = numeric_scalar
    return {
        ("collections", "defaultdict"): defaultdict,
        ("numpy", "dtype"): np.dtype,
        ("numpy._core.multiarray", "scalar"): scalar,
        ("numpy.core.multiarray", "scalar"): scalar,
        ("_codecs", "encode"): latin1_encode,
        ("typing", "Any"): Any,
        ("omegaconf.dictconfig", "DictConfig"): Inert,
        ("omegaconf.listconfig", "ListConfig"): Inert,
        ("omegaconf.base", "ContainerMetadata"): Inert,
        ("omegaconf.base", "Metadata"): Inert,
        ("omegaconf.nodes", "AnyNode"): Inert,
    }


class _Args(dict):
    """The config the network reads: attribute access and `.get`, like upstream's OmegaConf."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass(frozen=True)
class Params:
    """Architecture (card `[params]` + variant `params`) and load options (card `[options]`)."""

    max_disp: int
    encoder: str  # "vitl" or "vits"
    iters: int


def _net(ctx: Context, *, assign: bool = False) -> FoundationStereo:
    args = _Args(  # the released cfg.yaml of both checkpoints
        vit_size=ctx.params.encoder,
        max_disp=ctx.params.max_disp,
        corr_levels=2,
        corr_radius=4,
        hidden_dims=[128] * 3,
        n_downsample=2,
        n_gru_layers=3,
        mixed_precision=False,  # autocast is handled by `Opts`, not by the network
    )
    if assign:
        with torch.device("meta"):
            net = FoundationStereo(args)
        # These two upstream constants are ordinary tensor attributes outside the state dict.
        with torch.device("cpu"):
            r = args["corr_radius"]
            net.dx = torch.linspace(-r, r, 2 * r + 1, requires_grad=False).reshape(net.dx.shape)
            attention = cast("CostVolumeDisparityAttention", net.cost_agg.atts["4"])
            position = attention.pos_embed0
            _, length, dim = position.pe.shape
            position.pe = PositionalEmbedding(dim, max_len=length).pe
    else:
        net = FoundationStereo(args)
    state = ctx.state_dict(key="model", strip="", extra=_extra())
    if assign:
        load_strict(net, state, ctx.variant.ref, assign=True)
    else:
        net.load_state_dict(state)
    del state
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch: folded BatchNorm, pure fp16 (CUDA, MPS) or fp32 weights, NHWC on CUDA, strip-wise cost volume."""

    deterministic_cudnn = True

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        cuda = ctx.device.type == "cuda"
        prepare_inference(self.net, Opts(half=ctx.dtype == "float16", cl=cuda))

    def _run(self, net: FoundationStereo, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward_fast(net, left, right, iters=self.ctx.params.iters)[:, 0]

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


class MetalKernels(Portable):
    """Apple MPS: the float16 portable network with the geometry / correlation lookup as a Metal kernel."""

    module = Runner.module  # the Metal kernels are not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network and load the kernel."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the kernel library

        from .kernels.metal import MetalGeo, prepare  # noqa: PLC0415

        mps.adapt(self.net)
        prepare(self.net)
        self.geo = partial(MetalGeo, stereo_metal.load(ctx.device))

    def _run(self, net: FoundationStereo, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward_fast(net, left, right, iters=self.ctx.params.iters, geo_type=self.geo)[:, 0]


class Kernels(Runner):
    """Our kernels: fused geometry lookup / GRU / epilogues, CUTLASS 3-D convolutions, fp16-accumulate convolutions."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        # Assigning the published ViT-L checkpoint avoids a second large CPU allocation.
        assign = (
            ctx.device.type == "cuda"
            and ctx.params.encoder == "vitl"
            and ctx.user_weights is None
            and ctx.user_state is None
        )
        self.net = _net(ctx, assign=assign)
        prepare_kernels(self.net, ctx.device)
        self._forward = forward_kernels

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._forward(self.net, left, right, iters=self.ctx.params.iters)[:, 0]}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "kernels_mps": MetalKernels,
    "portable_mlx": mlx_wrap.port(Portable, whole="disparity"),
    "kernels_mlx": mlx_wrap.port(Portable, whole="disparity"),
}
