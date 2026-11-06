"""NMRF runner: weights -> network and the portable forward pass."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

from fastforks._core import mps
from fastforks.stereo._runner import Forward, Runner

from .net.inference import forward, prepare
from .net.nmrf import NMRF, build

if TYPE_CHECKING:
    import torch

    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture (card `[params]`)."""

    max_disp: int


def _cfg(p: Params) -> SimpleNamespace:
    """Upstream's config (nmrf/config/defaults.py + configs/zero_shot_evaluation.yaml) for the card's params."""
    return SimpleNamespace(
        BACKBONE=SimpleNamespace(NORM_FN="instance", OUT_CHANNELS=256),
        DPN=SimpleNamespace(MAX_DISP=p.max_disp, COST_GROUP=4, NUM_PROPOSALS=4, CONTEXT_DIM=64),
        NMP=SimpleNamespace(
            PROP_EMBED_DIM=128,
            INFER_EMBED_DIM=128,
            MLP_RATIO=4,
            SPLIT_SIZE=1,
            WINDOW_SIZE=6,
            REFINE_WINDOW_SIZE=4,
            PROP_N_HEADS=4,
            INFER_N_HEADS=4,
            NUM_PROP_LAYERS=5,
            NUM_INFER_LAYERS=5,
            NUM_REFINE_LAYERS=5,
            NORMALIZE_BEFORE=True,
        ),
    )


def _net(ctx: Context) -> NMRF:
    net = build(_cfg(ctx.params))
    sd = ctx.state_dict(key="model", strip="")
    sd.pop("device_indicator_tensor", None)  # upstream's device tracker (an empty buffer)
    net.load_state_dict(sd)
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch: float16 autocast with float16 LayerNorms (CUDA) or float32, SDPA stripe attention."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = prepare(_net(ctx), half=ctx.dtype == "float16")

    def _run(self, net: NMRF, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        with mps.quiet_nd_padding(left.device):  # the window attention pads 6-D tensors
            return forward(net, left, right, half=self.ctx.dtype == "float16")

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


class Kernels(Runner):
    """Our kernels: the portable float16 path with the window attention as one fused kernel (`fastbb`)."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and load the kernels."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = prepare_kernels(_net(ctx), ctx.device)
        self._forward = forward_kernels

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._forward(self.net, left, right)}


class MetalKernels(Portable):
    """Apple MPS: the portable float16 path with every window attention as one Metal kernel (`win_rpe_attn`)."""

    module = Runner.module  # the Metal kernel is not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network and swap in the kernel."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only

        mps.adapt(self.net)  # the faster / fp16-capable transposed convolutions, as for the portable runner
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the Metal library

        from .kernels.inference import use_metal_attention  # noqa: PLC0415

        use_metal_attention(self.net, stereo_metal.load(ctx.device))


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
