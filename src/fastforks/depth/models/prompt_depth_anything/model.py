"""Prompt Depth Anything runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `PromptDA.forward`: the image is shrunk to at most `max_size` on its longer side with both sides a
multiple of 14 (upstream `load_image`: area interpolation), normalised with the ImageNet mean / std; the depth prompt is
scaled to `[0, 1]` by its own min / max, the sigmoid output is mapped back with them (metres). The result is resampled
to the input size (bilinear).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.half import pure_half
from fastforks.depth._runner import Forward, Runner, scale
from fastforks.kernels.dpt_fused import load as load_dpt_fused
from fastforks.shared.depth_anything_v2 import fold_layer_scale, use_sdpa_attention

from .fused import FusedViT, PromptFusedHead
from .net import PromptDepthAnything

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks.depth._runner import Context

PATCH = 14  # fixed: the DINOv2 ViT-*/14 patch; upstream hard-codes it in `load_image`
_CONFIGS = {  # upstream model_configs
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}
_EPS = 1e-6  # fixed: a constant prompt would divide by zero in upstream's min / max scaling


@dataclass(frozen=True)
class Params:
    """Variant `params` + load options."""

    arch: str  # "vits" or "vitl"
    max_size: int  # longest network side, rounded down to a multiple of 14


def network_size(h: int, w: int, max_size: int) -> tuple[int, int]:
    """Upstream `load_image`: shrink so that the longer side is `max_size // 14 * 14`; each side rounds down to 14."""
    m = max(max_size // PATCH, 1) * PATCH
    s = min(1.0, m / max(h, w))
    return max(int(h * s) // PATCH, 1) * PATCH, max(int(w * s) // PATCH, 1) * PATCH


def _net(ctx: Context) -> PromptDepthAnything:
    cfg = _CONFIGS[ctx.params.arch]
    net = PromptDepthAnything(ctx.params.arch, int(cfg["features"]), list(cfg["out_channels"]))  # ty: ignore[invalid-argument-type]
    sd = ctx.state_dict(key="state_dict", strip="pipeline.")
    sd.pop("_mean", None)  # the runners normalise
    sd.pop("_std", None)
    net.load_state_dict(sd)
    return net.eval()


def _resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Area interpolation when shrinking (upstream's `cv2.INTER_AREA`), bicubic for the odd up-rounding."""
    if tuple(image.shape[-2:]) == size:
        return image
    if size[0] <= image.shape[-2] and size[1] <= image.shape[-1]:
        return F.interpolate(image, size, mode="area")
    return F.interpolate(image, size, mode="bicubic", align_corners=False)


def _range(prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample min and the (non-zero) max - min of a prompt `[B, 1, h, w]`, both `[B, 1, 1, 1]`."""
    lo = prompt.amin((1, 2, 3), keepdim=True)
    return lo, (prompt.amax((1, 2, 3), keepdim=True) - lo).clamp_min(_EPS)


def _restore(d: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Network output `[B, h, w]` -> float32 `[B, H, W]` at the input size."""
    return F.interpolate(d[:, None].float(), size, mode="bilinear", align_corners=False)[:, 0]


class Portable(Runner):
    """Plain PyTorch with fused SDPA attention: float32, or on CUDA the network in fp16 (channels-last convs).

    The fp16 network has no autocast: weights are cast once, the 1-channel output conv runs in fp32 (depth precision).
    """

    dtypes = ("float32", "float16")
    batch = None
    Params = Params
    prompt_depth = "required"

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx).to(ctx.device)
        use_sdpa_attention(self.net)
        fold_layer_scale(self.net)
        if ctx.dtype == "float16":
            pure_half(self.net, fp32=[self.net.depth_head.scratch.output_conv2[2]])

    def _run(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        x = scale(_resize(image, network_size(*image.shape[-2:], self.ctx.params.max_size)), self.ctx.io)
        lo, span = _range(prompt)
        d = self.net(x, (prompt - lo) / span).float() * span[:, 0] + lo[:, 0]
        return _restore(d, image.shape[-2:])

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        image: torch.Tensor,
        _intrinsics: torch.Tensor | None,
        prompt_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Metric depth `[B, H, W]` of a batch and its depth prompts `[B, 1, h, w]`."""
        return {"depth": self._run(image, prompt_depth)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `([B, 3, H, W] in [0, 1], prompt [B, 1, h, w])` -> depth `[B, H, W]`."""
        return Forward(self.net, lambda _net, image, prompt: self._run(image, prompt))


class Kernels(Runner):
    """dpt_fused kernels: fused-epilogue CUTLASS GEMMs / convs (fp16 accumulation), flash attention, NHWC fp16 head.

    The ViT is the Depth Anything V2 one; the head adds the fused prompt branches (`fused.py`). Weights are converted
    on the CPU, so the GPU holds only the fp16 network.
    """

    batch = None
    Params = Params
    prompt_depth = "required"

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, convert it, move it to the device."""
        super().__init__(ctx)
        net = _net(ctx)
        k = load_dpt_fused(ctx.device)
        self.vit = FusedViT(net.pretrained, ctx.device, k, "hh")
        self.head = PromptFusedHead(net.depth_head, ctx.device, k)
        self.layers = net.layers

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        image: torch.Tensor,
        _intrinsics: torch.Tensor | None,
        prompt_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Metric depth `[B, H, W]` of a batch and its depth prompts `[B, 1, h, w]`."""
        size = network_size(*image.shape[-2:], self.ctx.params.max_size)
        lo, span = _range(prompt_depth)
        with torch.autocast("cuda", enabled=False):
            feats = self.vit.features(_resize(image, size), size, self.layers)  # the resize is the identity here
            d = self.head(feats, size[0] // PATCH, size[1] // PATCH, (prompt_depth - lo) / span)
        return {"depth": _restore(d * span[:, 0] + lo[:, 0], image.shape[-2:])}


RUNNERS = {"portable": Portable, "kernels": Kernels}
