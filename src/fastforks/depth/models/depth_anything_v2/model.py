"""Depth Anything V2 runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `infer_image`: resize so that both sides are at least `short_side` and multiples of 14 (keeping
the aspect ratio, bicubic), normalise with the ImageNet mean / std, run, ReLU, and upsample the relative inverse depth
back to the input size (bilinear, `align_corners=True`). Upstream resizes with OpenCV's `INTER_CUBIC` on the CPU;
PyTorch's bicubic uses the same kernel (a = -0.75, replicated borders) on the device.

Metric checkpoints (`max_depth > 0`, upstream `metric_depth/`) end the head in a sigmoid instead of the ReLU and scale
it by `max_depth`: metric depth in metres. Distill-Any-Depth checkpoints are the same network; its Large ships the
authors' `DepthAnything` layout (`layout = "dam"`: `backbone.*` keys with chunked blocks), renamed on load.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import pure_half
from fastforks.depth._runner import Forward, Runner, scale
from fastforks.kernels.dpt_fused import load as load_dpt_fused
from fastforks.shared.depth_anything_v2 import DepthAnythingV2, fold_layer_scale, use_sdpa_attention

from .fused import FusedHead, FusedViT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks.depth._runner import Context

PATCH = 14  # fixed: the DINOv2 ViT-*/14 patch; upstream hard-codes ensure_multiple_of=14
_CONFIGS = {  # upstream run.py model_configs
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}


@dataclass(frozen=True)
class Params:
    """Variant `params` + load options."""

    arch: str  # "vits", "vitb" or "vitl"
    short_side: int  # lower bound of both resized sides
    max_depth: float  # metric head: sigmoid * max_depth (metres); 0: relative inverse depth (ReLU head)
    layout: str  # checkpoint keys: "dav2" (upstream DepthAnythingV2) or "dam" (Distill-Any-Depth's DepthAnything)


def network_size(h: int, w: int, short_side: int) -> tuple[int, int]:
    """Upstream `Resize(keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound")` for `h x w`."""
    s = max(short_side / h, short_side / w)

    def multiple(x: float) -> int:
        y = round(x / PATCH) * PATCH  # numpy's round: half to even, like Python's
        return y if y >= short_side else math.ceil(x / PATCH) * PATCH

    return multiple(s * h), multiple(s * w)


def _state_dict(ctx: Context) -> dict[str, torch.Tensor]:
    """The checkpoint with upstream DepthAnythingV2 keys.

    `dam` (one chunk of DINOv2 blocks): `backbone.blocks.0.<i>.` -> `pretrained.blocks.<i>.`, `backbone.` ->
    `pretrained.`.
    """
    sd = ctx.state_dict()
    if ctx.params.layout == "dav2":
        return sd
    if ctx.params.layout != "dam":
        msg = f"depth_anything_v2: unknown checkpoint layout {ctx.params.layout!r}"
        raise ValueError(msg)
    return {re.sub(r"^backbone\.(blocks\.)(?:0\.)?|^backbone\.", r"pretrained.\1", k): v for k, v in sd.items()}


def _cpu_net(ctx: Context) -> DepthAnythingV2:
    """The network on the CPU in float32; a metric head ends in a sigmoid (upstream `metric_depth` DPTHead)."""
    net = DepthAnythingV2(encoder=ctx.params.arch, **_CONFIGS[ctx.params.arch])
    if ctx.params.max_depth > 0:
        net.depth_head.scratch.output_conv2[3] = nn.Sigmoid()  # the shared net's final ReLU is then a no-op
    net.load_state_dict(_state_dict(ctx))
    return net.eval()


def _net(ctx: Context) -> DepthAnythingV2:
    return _cpu_net(ctx).to(ctx.device)


def _key(ctx: Context) -> str:
    return "depth" if ctx.params.max_depth > 0 else "inverse_depth"


def _scaled(ctx: Context, d: torch.Tensor) -> torch.Tensor:
    """Restored network output -> metric depth (`sigmoid * max_depth`), or the relative inverse depth as it is."""
    return d * ctx.params.max_depth if ctx.params.max_depth > 0 else d


def _prepare(ctx: Context, image: torch.Tensor) -> torch.Tensor:
    """`[B, 3, H, W]` in `[0, 1]` -> resized, normalised network input (float32)."""
    h, w = network_size(*image.shape[-2:], ctx.params.short_side)
    return scale(F.interpolate(image, (h, w), mode="bicubic", align_corners=False), ctx.io)


def _restore(d: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Network output `[B, h, w]` -> float32 `[B, H, W]` at the input size."""
    return F.interpolate(d[:, None].float(), size, mode="bilinear", align_corners=True)[:, 0]


class Portable(Runner):
    """Plain PyTorch with fused SDPA attention: float32, or on CUDA the network in fp16 (channels-last convs).

    The fp16 network has no autocast: weights are cast once, the 1-channel output conv runs in fp32 (depth precision).
    """

    dtypes = ("float32", "float16")
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        use_sdpa_attention(self.net)
        fold_layer_scale(self.net)
        self.half = ctx.dtype == "float16"
        if self.half:
            pure_half(self.net, fp32=[self.net.depth_head.scratch.output_conv2[2]])

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return _scaled(self.ctx, _restore(self.net(_prepare(self.ctx, image)), image.shape[-2:]))

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Relative inverse depth (metric depth for metric checkpoints) of a batch."""
        return {_key(self.ctx): self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> inverse (metric) depth `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self._run(image))


# flash attention variant per encoder (fastbb `flash64_*`): fp16 QK^T accumulation ("hh") keeps ViT-S / L within
# 1.3e-3 / 2.4e-3 rel p99 of the fp32 reference, ViT-B needs fp32 QK^T ("fh": 5.6e-3 -> 2.6e-3, +0.9 ms at 518x770)
_KERNELS_ATTN = {"vitb": "fh"}


class Kernels(Runner):
    """dpt_fused kernels: fused-epilogue CUTLASS GEMMs / convs (fp16 accumulation), flash attention, NHWC fp16 head.

    The input resize + normalisation, the patch embedding and the position embedding are fused too; the 1-channel
    output conv runs in fp32. Weights are converted on the CPU, so the GPU holds only the fp16 network.
    """

    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, convert it, move it to the device."""
        super().__init__(ctx)
        net = _cpu_net(ctx)
        k = load_dpt_fused(ctx.device)
        self.vit = FusedViT(net.pretrained, ctx.device, k, _KERNELS_ATTN.get(ctx.params.arch, "hh"))
        self.head = FusedHead(net.depth_head, ctx.device, k)
        self.layers = net.intermediate_layer_idx[net.encoder]

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Relative inverse depth (metric depth for metric checkpoints) of a batch."""
        h, w = network_size(*image.shape[-2:], self.ctx.params.short_side)
        with torch.autocast("cuda", enabled=False):
            feats = self.vit.features(image, (h, w), self.layers)
            d = self.head(feats, h // PATCH, w // PATCH)
        return {_key(self.ctx): _scaled(self.ctx, _restore(d, image.shape[-2:]))}


RUNNERS = {"portable": Portable, "kernels": Kernels}
