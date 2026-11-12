# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# fastforks: the DINOv3 ViT-L/16 descriptor of RoMa v2 (`torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16")`
# at adc254450203739c8149213a7a69d8d905b4fcfa, `models/vision_transformer.py` `DinoVisionTransformer`), cut to what
# `get_intermediate_layers(n=[11, 17])` needs in eval: the blocks after the last requested layer are not built (their
# output is never used), the masked / chunked / training paths are dropped. Parameter names are those of the RoMa v2
# checkpoint (`f.*`).

from functools import partial

import torch
from torch import Tensor, nn

from .amp import autocast
from .vit.block import SelfAttentionBlock
from .vit.ffn_layers import Mlp
from .vit.patch_embed import PatchEmbed
from .vit.rope import RopePositionEmbedding


class DinoVisionTransformer(nn.Module):
    amp: torch.dtype | None = None  # fastforks: upstream's descriptor runs in bf16 (weights and autocast; amp.py)

    def __init__(self, *, layers: tuple[int, ...] = (11, 17)) -> None:
        super().__init__()
        embed_dim, num_heads, self.n_storage_tokens = 1024, 16, 4
        self.layers = layers
        self.patch_size = 16
        self.patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=embed_dim, flatten_embedding=False)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.empty(1, self.n_storage_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.rope_embed = RopePositionEmbedding(
            embed_dim, num_heads=num_heads, base=100, normalize_coords="separate", rescale_coords=2, dtype=torch.float32
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-5)  # "layernormbf16"
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=1.0e-05,
                mask_k_bias=True,
            )
            for _ in range(max(layers) + 1)
        )
        self.norm = norm_layer(embed_dim)

    def prepare_tokens_with_masks(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)
        cls_token = self.cls_token + 0 * self.mask_token
        x = torch.cat([cls_token.expand(B, -1, -1), self.storage_tokens.expand(B, -1, -1), x], dim=1)
        return x, (H, W)

    def forward(self, x: Tensor) -> list[Tensor]:
        """`get_intermediate_layers(x, n=layers)` (normed, patch tokens only) as `[B, H, W, D]` maps."""
        with autocast(x, self.amp):
            x, (H, W) = self.prepare_tokens_with_masks(x)
            output = []
            for i, blk in enumerate(self.blocks):
                rope_sincos = self.rope_embed(H=H, W=W)
                x = blk(x, rope_sincos)
                if i in self.layers:
                    output.append(x)
            outputs = [self.norm(out)[:, self.n_storage_tokens + 1 :] for out in output]
        return [o.reshape(o.shape[0], H, W, o.shape[-1]) for o in outputs]
