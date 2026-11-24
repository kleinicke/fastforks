"""WAFT runners: weights -> network and the portable forward pass."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.shared.depth_anything_v2 import fold_layer_scale, sdpa_attention

from .net import ViTWarpV8

if TYPE_CHECKING:
    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[options]`."""

    iters: int  # warping iterations (upstream config/eval: 5)
    cuda_graphs: bool


def _net() -> ViTWarpV8:
    # upstream config/a1/*.json: DAv2-S features, ViT-S refinement; var_max / var_min only shape the training loss
    args = SimpleNamespace(dav2_backbone="vits", network_backbone="vits", iters=5, var_max=10, var_min=0)
    return ViTWarpV8(args)


class Portable(Runner):
    """Plain PyTorch; on CUDA in fp16 the network runs under autocast (warping coordinates and the flow in fp32).

    The card says `pad = "none"`: the network normalises and pads itself (zeros after the normalisation, to multiples
    of 112, as upstream's Padder 'sintel' mode). Depth Anything's attention runs through SDPA with LayerScale folded.
    """

    dtypes = ("float32", "float16")
    Params = Params
    batch = None
    # pixels (pairs x H x W) per network call: 8 pairs 480 x 640 101.6 -> 80.7 ms per pair at 1.4 GB (RTX 5060 Ti,
    # Windows, fp16; mean |flow diff| to one pair per call 0.004 px); 720p runs one pair per call, as before
    batch_pixels = 1_300_000

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net()
        sd = ctx.state_dict()
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        sdpa_attention(self.net.da_feature)
        fold_layer_scale(self.net.da_feature)
        self.net.eval().to(ctx.device)
        self.fp16 = ctx.dtype == "float16"

    def _autocast(self, x: torch.Tensor) -> contextlib.AbstractContextManager:
        return torch.autocast(x.device.type, torch.float16) if self.fp16 else contextlib.nullcontext()

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a batch (the last iteration's)."""
        final_only = image1.is_cuda and not (
            torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.is_grad_enabled()
        )
        with self._autocast(image1):
            flow = self.net(image1, image2, self.ctx.params.iters, final_flow_only=final_only)["flow"][-1]
        return {"flow": flow.float()}

    def module(self) -> torch.nn.Module:
        """The network; returns the last iteration's flow."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, image1, image2: net(image1, image2, iters)["flow"][-1])


class Kernels(Runner):
    """Our kernels: fp16 NHWC convs with fused epilogues, Depth Anything's fast ViT; the flow state in fp32."""

    Params = Params
    _GRAPH_PIXELS = 336 * 448  # fixed: cap padded work where replay pays off

    def __init__(self, ctx: Context) -> None:
        """Build the network and prepare it for the kernels."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, graph_pixels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net()
        sd = ctx.state_dict()
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.eval().to(ctx.device)
        prepare_kernels(self.net, ctx.device)
        self._forward = forward_kernels
        self._graphs = SmallShapeGraphs(
            self._flow,
            pixels=self._GRAPH_PIXELS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
            size=graph_pixels,
            one_shape=True,
        )

    def _flow(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return self._forward(self.net, image1, image2, self.ctx.params.iters)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a batch (the last iteration's)."""
        return {"flow": self._graphs(image1, image2)}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        self._graphs.release()


class _OnePair(Portable):
    """Portable for the MLX port: one pair per call, as before batching (not measured on Apple GPUs)."""

    batch = 1


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_OnePair),
}
