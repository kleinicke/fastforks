"""Sapiens segmentation network (ViT + `VitHead`) with the state-dict names of the Sapiens-Lite TorchScript archives.

Ported from facebookresearch/sapiens (`seg/mmseg/models/decode_heads/vit_head.py`, the mmpretrain `VisionTransformer`
it uses as backbone; CC-BY-NC-4.0, portions Apache-2.0, see LICENSE) as the archives trace it:

- backbone: 16 x 16 patch conv, a learned position table for the 64 x 48 grid of a 1024 x 768 input (no class token),
  pre-norm blocks (LayerNorm eps 1e-6, multi-head attention with a packed qkv projection, GELU MLP), a final LayerNorm
- decode head: 3 x (4 x 4 stride-2 transposed conv, no bias, + InstanceNorm + SiLU), 3 x (1 x 1 conv + InstanceNorm +
  SiLU), a 1 x 1 classifier; InstanceNorm without affine parameters (eps 1e-5)

Logits come out at 1/2 of the input size (8x upsampling of the 1/16 tokens). The archives' `data_preprocessor`
buffers are not part of the network (the runner normalises with the card's mean / std).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

LN_EPS = 1e-6  # fixed: mmpretrain VisionTransformer norm_cfg eps of the Sapiens configs
IN_EPS = 1e-5  # fixed: torch InstanceNorm2d default (VitHead norm_cfg IN)
PATCH = 16  # fixed: patch size of every Sapiens ViT
KERNEL = 4  # fixed: VitHead deconv kernel (stride 2, padding 1)
STAGES = 3  # fixed: deconv and 1 x 1 conv layers of the seg configs (each Sequential holds conv, norm, act)


@dataclass(frozen=True)
class Arch:
    """Network shape (card `[params]`)."""

    embed_dim: int
    depth: int
    num_heads: int
    grid: tuple[int, int]  # position-table grid (input size / 16)
    channels: int  # decode head width
    num_classes: int


class Attention(nn.Module):
    """mmpretrain `MultiheadAttention` (packed qkv, SDPA)."""

    def __init__(self, dim: int, heads: int) -> None:
        """The qkv and output projections."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, N, C]` -> `[B, N, C]`."""
        b, n, c = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))


class Ffn(nn.Module):
    """mmpretrain `FFN`: `layers.0 = (fc1, GELU)`, `layers.1 = fc2`."""

    def __init__(self, dim: int) -> None:
        """Hidden width 4 x `dim`."""
        super().__init__()
        fc1, fc2 = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)
        self.layers = nn.Sequential(nn.Sequential(fc1, nn.GELU()), fc2)
        self.fc: tuple[nn.Linear, nn.Linear] = (fc1, fc2)  # typed views of `layers` (not registered twice)


class Layer(nn.Module):
    """mmpretrain `TransformerEncoderLayer` (pre-norm)."""

    def __init__(self, dim: int, heads: int) -> None:
        """Norms, attention, MLP."""
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=LN_EPS)
        self.attn = Attention(dim, heads)
        self.ln2 = nn.LayerNorm(dim, eps=LN_EPS)
        self.ffn = Ffn(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One block."""
        x = x + self.attn(self.ln1(x))
        return x + self.ffn.layers(self.ln2(x))


class PatchEmbed(nn.Module):
    """mmpretrain `PatchEmbed` (`projection`)."""

    def __init__(self, dim: int) -> None:
        """The patch conv."""
        super().__init__()
        self.projection = nn.Conv2d(3, dim, PATCH, PATCH)


class Backbone(nn.Module):
    """mmpretrain `VisionTransformer` as the Sapiens configs build it (no class token, final norm `ln1`)."""

    def __init__(self, a: Arch) -> None:
        """Patch embedding, position table, blocks, final norm."""
        super().__init__()
        self.patch_embed = PatchEmbed(a.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, a.grid[0] * a.grid[1], a.embed_dim))
        self.blocks: list[Layer] = [Layer(a.embed_dim, a.num_heads) for _ in range(a.depth)]
        self.layers = nn.ModuleList(self.blocks)
        self.ln1 = nn.LayerNorm(a.embed_dim, eps=LN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised image `[B, 3, 16 gh, 16 gw]` (the table's grid) -> feature map `[B, C, gh, gw]`."""
        t = self.patch_embed.projection(x)
        b, c, gh, gw = t.shape
        t = t.flatten(2).transpose(1, 2) + self.pos_embed
        for layer in self.blocks:
            t = layer(t)
        return self.ln1(t).view(b, gh, gw, c).permute(0, 3, 1, 2)


def _norm_act(ch: int) -> list[nn.Module]:
    return [nn.InstanceNorm2d(ch, eps=IN_EPS), nn.SiLU()]  # no affine, no running stats


class DecodeHead(nn.Module):
    """Sapiens `VitHead`: deconvolutions, 1 x 1 convs, classifier."""

    def __init__(self, a: Arch) -> None:
        """Layers at the state-dict indices 0, 3, 6 of each Sequential."""
        super().__init__()
        ch = a.channels
        self.deconvs: list[nn.ConvTranspose2d] = [  # typed views of the Sequentials (not registered twice)
            nn.ConvTranspose2d(a.embed_dim if i == 0 else ch, ch, KERNEL, 2, 1, bias=False) for i in range(STAGES)
        ]
        self.convs: list[nn.Conv2d] = [nn.Conv2d(ch, ch, 1) for _ in range(STAGES)]
        self.deconv_layers = nn.Sequential(*(m for d in self.deconvs for m in (d, *_norm_act(ch))))
        self.conv_layers = nn.Sequential(*(m for c in self.convs for m in (c, *_norm_act(ch))))
        self.conv_seg = nn.Conv2d(ch, a.num_classes, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Feature map -> logits at 8 x its size."""
        return self.conv_seg(self.conv_layers(self.deconv_layers(f)))


class Sapiens(nn.Module):
    """Sapiens `EncoderDecoder` (same state-dict names, without `data_preprocessor`)."""

    def __init__(self, a: Arch) -> None:
        """Untrained network of shape `a`."""
        super().__init__()
        self.backbone: Backbone = Backbone(a)
        self.decode_head: DecodeHead = DecodeHead(a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised `[B, 3, H, W]` at the table's input size -> logits `[B, classes, H / 2, W / 2]`."""
        return self.decode_head(self.backbone(x))
