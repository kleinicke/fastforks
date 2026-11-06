"""WAFT-Stereo runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks.stereo._runner import Forward, Runner

from .net.waft import WAFT

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]` + variant `params`)."""

    max_disp: int
    arch: str  # Depth Anything V2 feature encoder: "vits" or "vitb"


def _cfg(arch: str, max_disp: int) -> types.SimpleNamespace:
    """The `configs/SynLarge/DAv2<S|B>-4.yaml` values the network reads."""
    ns = types.SimpleNamespace
    iterative = ns(TYPE="vit", ARCH="vits", PATCH_SIZE=8, LORA_RANK=None, LORA_ALPHA=None)
    return ns(
        WAFT=ns(
            MAX_DISP=max_disp,
            LOSS=["mixlap_40"],
            FEATURE_ENCODER=ns(TYPE="dav2", ARCH=arch, LORA_RANK=8, LORA_ALPHA=16),
            ITERATIVE_MODULE=ns(TASK=["delta"] * 3, DELTA_ITER=iterative, PROP_ITER=iterative),
        )
    )


class _CfgNode(dict):
    """Stands for upstream's yacs `CfgNode` (the checkpoint's `model_config`, unused here)."""


_EXTRA: dict[tuple[str, str], object] = {  # what the checkpoints' pickles reference besides tensors
    ("bridgedepth.config.config", "CfgNode"): _CfgNode,
}


def _net(ctx: Context) -> WAFT:
    net = WAFT(_cfg(ctx.params.arch, ctx.params.max_disp))
    net.load_state_dict(ctx.state_dict(key="model", strip="", extra=_EXTRA))
    return net.eval().to(ctx.device)


def _disparity(net: WAFT, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return net({"img1": left, "img2": right})["disp_pred"]


class Portable(Runner):
    """Plain PyTorch: float32, or fp16 autocast on CUDA (upstream forward).

    NCHW like upstream: channels-last makes the convolutions faster but costs more in the layout copies between the
    convolutions and the ViT / attention parts (6 % slower in total at Middlebury-H).
    """

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.half = ctx.dtype == "float16"

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if not self.half:
            return _disparity(self.net, left, right)
        with torch.autocast(left.device.type, dtype=torch.float16):
            return _disparity(self.net, left, right)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a batch (0..255 images; the network pads and crops itself)."""
        return {"disparity": self._run(left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, lambda _net, left, right: self._run(left, right))


class Kernels(Runner):
    """Our kernels: merged LoRA, fp16 NHWC convs with fused InstanceNorm / SiLU / GELU / residual epilogues."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and prepare it for the kernels."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        prepare_kernels(self.net, ctx.device)
        self._forward = forward_kernels

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a batch (0..255 images)."""
        return {"disparity": self._forward(self.net, left, right)}


RUNNERS = {"portable": Portable, "kernels": Kernels}
