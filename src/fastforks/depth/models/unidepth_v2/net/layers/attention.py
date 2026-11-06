"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_scale import LayerScale
from .mlp import MLP


# fastforks: SimpleAttention and AttentionDecoderBlock (unused by UniDepth V2) are not vendored


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        context_dim = context_dim or dim
        self.mlp = MLP(dim, expansion=expansion, dropout=dropout, gated=gated)
        self.kv = nn.Linear(context_dim, dim * 2, bias=use_bias)
        self.q = nn.Linear(dim, dim, bias=use_bias)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out = nn.Linear(dim, dim, bias=use_bias)
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        # fastforks: einops.rearrange '(kv h d) -> h n d kv' / '(h d) -> h n d' as view / permute
        kv = self.kv(context)
        k, v = kv.view(*kv.shape[:2], 2, self.num_heads, -1).permute(0, 3, 1, 4, 2).unbind(dim=-1)
        q = self.q(x)
        q = q.view(*q.shape[:2], self.num_heads, -1).transpose(1, 2)

        if pos_embed is not None:
            pos_embed = pos_embed.view(*pos_embed.shape[:2], self.num_heads, -1).transpose(1, 2)  # fastforks: einops
            q = q + pos_embed
        if pos_embed_context is not None:
            pos_embed_context = pos_embed_context.view(  # fastforks: einops.rearrange
                *pos_embed_context.shape[:2], self.num_heads, -1
            ).transpose(1, 2)
            k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, attn_mask=attn_bias
        )
        x = x.transpose(1, 2).flatten(2)  # fastforks: einops.rearrange 'b h n d -> b n (h d)'
        x = self.out(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        x = (
            self.ls1(
                self.attn(
                    x,
                    attn_bias=attn_bias,
                    context=context,
                    pos_embed=pos_embed,
                    pos_embed_context=pos_embed_context,
                )
            )
            + x
        )
        x = self.ls2(self.mlp(x)) + x
        return x


class AttentionLayer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    expansion=expansion,
                    dropout=dropout,
                    cosine=cosine,
                    gated=gated,
                    layer_scale=layer_scale,
                    context_dim=context_dim,
                    use_bias=use_bias,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                context=context,
                pos_embed=pos_embed,
                pos_embed_context=pos_embed_context,
                attn_bias=attn_bias,
            )
        return x
