"""The timm ViT pieces WAFT-Stereo and WAFT use, written out so `timm` is not a dependency.

fastforks addition (not upstream code): `Mlp` and the pre-norm ViT `Block` of timm 1.0 (Apache-2.0, Ross Wightman) with
the same module tree and parameter names, restricted to what WAFT builds: `Mlp` (Linear or 1x1-conv, GELU),
`vit_{tiny,small,base,large}_patch16_224` blocks (qkv bias, LayerNorm eps 1e-6, MLP ratio 4, no layer scale / dropout,
fused scaled-dot-product attention).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, use_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear = (lambda i, o: nn.Conv2d(i, o, kernel_size=1)) if use_conv else nn.Linear
        self.fc1 = linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x):
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v)
        return self.proj_drop(self.proj(self.norm(x.transpose(1, 2).reshape(b, n, c))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


_VIT = {  # timm model name -> (embed dim, depth, heads)
    "vit_tiny_patch16_224": (192, 12, 3),
    "vit_small_patch16_224": (384, 12, 6),
    "vit_base_patch16_224": (768, 12, 12),
    "vit_large_patch16_224": (1024, 24, 16),
}


class ViTBlocks(nn.Module):
    """Holder with `.blocks` = the transformer blocks of a timm ViT (randomly initialised; weights come from the checkpoint)."""

    def __init__(self, name: str):
        super().__init__()
        dim, depth, heads = _VIT[name]
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(depth)])


def create_vit_blocks(name: str) -> ViTBlocks:
    return ViTBlocks(name)
