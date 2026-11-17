"""MobileSAM's TinyViT image encoder in a traceable form.

Same maths as upstream `TinyViT.forward_features` on the upstream modules, with the shapes fixed when the network is
built (each stage's `input_resolution`; upstream asserts and branches on tensor shapes, which `torch.fx` cannot
trace). Tokens stay `[B, H, W, C]` between the blocks, so the depthwise convolutions see a channels-last tensor
without a copy; the window attention runs through SDPA with the relative-position bias as its additive mask.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .net.tiny_vit_sam import Attention, BasicLayer, PatchMerging, TinyViT, TinyViTBlock


class Norm2d(torch.nn.Module):
    """An upstream `LayerNorm2d` in float32 (its squared deviations overflow float16), back in the input's dtype."""

    def __init__(self, norm: torch.nn.Module) -> None:
        """Wrap `norm`."""
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, C, H, W]` normalised over the channels."""
        return self.norm(x.float()).to(x.dtype)


def attention(attn: Attention, x: torch.Tensor) -> torch.Tensor:
    """Upstream `Attention.forward` on windows `[W, N, C]` (SDPA scales by `key_dim ** -0.5` like `attn.scale`)."""
    qkv = attn.qkv(attn.norm(x)).unflatten(-1, (attn.num_heads, -1)).transpose(1, 2)
    q, k, v = qkv.split([attn.key_dim, attn.key_dim, attn.d], dim=3)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=cast("torch.Tensor", attn.ab))
    return attn.proj(x.transpose(1, 2).flatten(2))


def block(blk: TinyViTBlock, x: torch.Tensor) -> torch.Tensor:
    """Upstream `TinyViTBlock.forward` on `[B, H, W, C]` tokens."""
    (h, w), ws, c = blk.input_resolution, blk.window_size, blk.dim
    ph, pw = -h % ws, -w % ws
    y = F.pad(x, (0, 0, 0, pw, 0, ph)) if ph or pw else x
    nh, nw = (h + ph) // ws, (w + pw) // ws
    y = y.reshape(-1, nh, ws, nw, ws, c).transpose(2, 3).reshape(-1, ws * ws, c)
    y = attention(blk.attn, y)
    y = y.reshape(-1, nh, nw, ws, ws, c).transpose(2, 3).reshape(-1, h + ph, w + pw, c)
    x = x + y[:, :h, :w]
    x = blk.local_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return x + blk.mlp(x)


def merge(pm: PatchMerging, x: torch.Tensor) -> torch.Tensor:
    """Upstream `PatchMerging.forward`: `[B, C, H, W]` -> `[B, H', W', C']` tokens."""
    x = pm.conv3(pm.act(pm.conv2(pm.act(pm.conv1(x)))))
    return x.permute(0, 2, 3, 1)


class Trunk(torch.nn.Module):
    """Upstream `TinyViT` up to its neck's last convolution: `[B, 3, 1024, 1024]` -> `[B, 256, 64, 64]`.

    Without the final `LayerNorm2d`, which the runners apply in float32.
    """

    def __init__(self, enc: TinyViT) -> None:
        """Take `enc`'s modules."""
        super().__init__()
        self.patch_embed = enc.patch_embed
        self.layers = enc.layers
        self.conv1, norm, self.conv2 = enc.neck[0], enc.neck[1], enc.neck[2]
        self.norm = Norm2d(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The neck's output before its last norm."""
        first, *rest = cast("list[BasicLayer]", list(self.layers))
        x = self.patch_embed(x)
        for mb in first.blocks:
            x = mb(x)
        x = merge(cast("PatchMerging", first.downsample), x)
        for layer in rest:
            for blk in layer.blocks:
                x = block(cast("TinyViTBlock", blk), x)
            if layer.downsample is not None:
                x = merge(layer.downsample, x.permute(0, 3, 1, 2))
        return self.conv2(self.norm(self.conv1(x.permute(0, 3, 1, 2))))
