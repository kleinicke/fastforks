"""Depth Anything V2 building blocks shared by stereo families (DEFOM-Stereo, WAFT-Stereo, ...).

Vendored unchanged from upstream (see `net/` and `meta.toml`): the DINOv2 ViT, the DPT feature-fusion blocks and the
DPT head. Families that modify the head (extra outputs, other fusion blocks) keep their own modified copy in their
`net/` and import only the unmodified ViT from here.

The weights are not shared: DAv2-Small is Apache-2.0, DAv2-Base/Large are CC-BY-NC-4.0 (see `meta.toml`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import base_type, override

from .fast_vit import FastViT, flash_attention, fp16_accumulate, use_fast_vit
from .net.dinov2 import DINOv2, vit_base, vit_giant2, vit_large, vit_small
from .net.dinov2_layers.attention import Attention
from .net.dinov2_layers.block import Block
from .net.dinov2_layers.layer_scale import LayerScale
from .net.dinov2_layers.mlp import Mlp
from .net.dinov2_layers.swiglu_ffn import SwiGLUFFN
from .net.dpt import DepthAnythingV2, DPTHead
from .net.util.blocks import FeatureFusionBlock, _make_scratch

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
    "use_sdpa_attention",
    "vit_base",
    "vit_giant2",
    "vit_large",
    "vit_small",
]


def _sdpa_forward(self: Attention, x: torch.Tensor) -> torch.Tensor:
    b, n, c = x.shape
    qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
    out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])  # default scale = head_dim**-0.5, as upstream
    return self.proj(out.transpose(1, 2).reshape(b, n, c))


class SdpaAttention(Attention):
    """DINOv2 `Attention` with fused `scaled_dot_product_attention` (the maths of `use_sdpa_attention`)."""

    forward = _sdpa_forward


def sdpa_attention(module: nn.Module) -> int:
    """`use_sdpa_attention` by switching every DINOv2 `Attention` under `module` to `SdpaAttention`.

    Changes each layer's class instead of binding a per-instance forward, so it creates no reference cycle (the
    module is freed without the garbage collector). Returns the number of switched layers.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, Attention):
            m.__class__ = SdpaAttention
            n += 1
    return n


def _scale_rows(lin: nn.Linear, g: torch.Tensor) -> None:
    lin.weight.copy_((lin.weight.float() * g[:, None]).to(lin.weight.dtype))
    if lin.bias is not None:
        lin.bias.copy_((lin.bias.float() * g).to(lin.bias.dtype))


def fold_layer_scale(module: nn.Module) -> int:
    """Fold each DINOv2 block's LayerScale into the linear before it (`attn.proj`, the MLP's last); in place.

    `ls1(proj(o)) = (gamma * W) o + gamma * b`: the products are computed in float32 and stored in the layer's dtype,
    and `ls1` / `ls2` become `nn.Identity` (one elementwise pass less per residual branch). Only blocks whose
    branches end in that linear (`Attention`, `Mlp`, `SwiGLUFFN`); returns the number of folded LayerScales.
    """
    n = 0
    with torch.no_grad():
        for blk in module.modules():
            if not isinstance(blk, Block):
                continue
            last = {"ls1": blk.attn.proj if isinstance(blk.attn, Attention) else None}
            mlp = blk.mlp
            last["ls2"] = (
                mlp.fc2 if base_type(mlp) is Mlp else mlp.w3 if issubclass(base_type(mlp), SwiGLUFFN) else None
            )
            for name, lin in last.items():
                ls = getattr(blk, name)
                if not isinstance(lin, nn.Linear) or base_type(ls) is not LayerScale:
                    continue
                _scale_rows(lin, ls.gamma.detach().float())
                setattr(blk, name, nn.Identity().train(ls.training))  # a new Module starts in training mode
                n += 1
    return n


def use_sdpa_attention(module: nn.Module) -> int:
    """Replace every DINOv2 `Attention` forward under `module` by fused `scaled_dot_product_attention`.

    Same maths as upstream (softmax(q k^T * head_dim**-0.5) v; attention dropout is 0 at inference), different
    accumulation order, so results differ at rounding level. Returns the number of patched layers.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, Attention):
            override(m, _sdpa_forward)
            n += 1
    return n
