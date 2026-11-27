"""Depth Anything V2 building blocks shared by stereo families (DEFOM-Stereo, WAFT-Stereo, ...).

Vendored unchanged from upstream (see `net/` and `meta.toml`): the DINOv2 ViT, the DPT feature-fusion blocks and the
DPT head. Families that modify the head (extra outputs, other fusion blocks) keep their own modified copy in their
`net/` and import only the unmodified ViT from here.

The weights are not shared: DAv2-Small is Apache-2.0, DAv2-Base/Large are CC-BY-NC-4.0 (see `meta.toml`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._torch import Attention as Attention
    from ._torch import Block as Block
    from ._torch import DepthAnythingV2 as DepthAnythingV2
    from ._torch import DINOv2 as DINOv2
    from ._torch import DPTHead as DPTHead
    from ._torch import FastViT as FastViT
    from ._torch import FeatureFusionBlock as FeatureFusionBlock
    from ._torch import LayerScale as LayerScale
    from ._torch import Mlp as Mlp
    from ._torch import SdpaAttention as SdpaAttention
    from ._torch import SwiGLUFFN as SwiGLUFFN
    from ._torch import _make_scratch as _make_scratch
    from ._torch import flash_attention as flash_attention
    from ._torch import fold_layer_scale as fold_layer_scale
    from ._torch import fp16_accumulate as fp16_accumulate
    from ._torch import fused_bilinear as fused_bilinear
    from ._torch import sdpa_attention as sdpa_attention
    from ._torch import use_fast_vit as use_fast_vit
    from ._torch import use_fused_bilinear as use_fused_bilinear
    from ._torch import use_sdpa_attention as use_sdpa_attention
    from ._torch import vit_base as vit_base
    from ._torch import vit_giant2 as vit_giant2
    from ._torch import vit_large as vit_large
    from ._torch import vit_small as vit_small

__all__ = [
    "Attention",
    "DINOv2",
    "DPTHead",
    "DepthAnythingV2",
    "FastViT",
    "FeatureFusionBlock",
    "SdpaAttention",
    "_make_scratch",
    "flash_attention",
    "fold_layer_scale",
    "fp16_accumulate",
    "sdpa_attention",
    "use_fast_vit",
    "use_fused_bilinear",
    "use_sdpa_attention",
    "vit_base",
    "vit_giant2",
    "vit_large",
    "vit_small",
]

_TORCH = frozenset(__all__) | {"Block", "LayerScale", "Mlp", "SwiGLUFFN", "fused_bilinear"}


def __getattr__(name: str) -> object:
    """The torch building blocks (`_torch`), imported at first use: `mlx_net` imports this package without torch."""
    if name in _TORCH:
        from . import _torch  # noqa: PLC0415 - imports torch

        return getattr(_torch, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
