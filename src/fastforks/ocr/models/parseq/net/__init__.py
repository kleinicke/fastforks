"""PARSeq: a ViT encoder over the `32 x 128` crop and a one-layer two-stream decoder (upstream `strhub.models.parseq`).

Same parameter names as the released checkpoints (`encoder.*` timm `VisionTransformer` without class token,
`decoder.layers.0.*` with `nn.MultiheadAttention`, `head`, `text_embed`, `pos_queries`). Inference only: greedy
autoregressive decoding, then `refine_iters` cloze refinement passes, as `PARSeq.forward` with `decode_ar=True`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

PATCH = (4, 8)  # fixed: the released models' patch size
IMAGE = (32, 128)  # fixed: their input size
MAX_LENGTH = 25  # fixed: max_label_length


class Attention(nn.Module):
    """timm `Attention`: fused qkv projection, scaled dot-product attention, output projection."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    """timm `Mlp`: fc1, exact GELU, fc2."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """timm pre-norm ViT block (LayerNorm eps 1e-6)."""

    def __init__(self, dim: int, heads: int, ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    """timm `PatchEmbed`: a strided convolution, flattened to tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, PATCH, PATCH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Encoder(nn.Module):
    """ViT without class token; the output is every token after the final LayerNorm."""

    def __init__(self, dim: int, depth: int, heads: int, ratio: int) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (IMAGE[0] // PATCH[0]) * (IMAGE[1] // PATCH[1]), dim))
        self.blocks = nn.ModuleList(Block(dim, heads, ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Upstream `DecoderLayer`: pre-LN two-stream (query / content) attention; only the query stream is needed here.

    The content stream is updated only by layers before the last; the released models have one layer.
    """

    def __init__(self, dim: int, heads: int, hidden: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.linear1 = nn.Linear(dim, hidden)
        self.linear2 = nn.Linear(hidden, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)

    def forward(
        self, query: torch.Tensor, content: torch.Tensor, memory: torch.Tensor, query_mask: torch.Tensor,
        padding: torch.Tensor | None,
    ) -> torch.Tensor:  # fmt: skip
        kv = self.norm_c(content)
        x = (
            query
            + self.self_attn(
                self.norm_q(query), kv, kv, attn_mask=query_mask, key_padding_mask=padding, need_weights=False
            )[0]
        )
        x = x + self.cross_attn(self.norm1(x), memory, memory, need_weights=False)[0]
        return x + self.linear2(F.gelu(self.linear1(self.norm2(x))))


class Decoder(nn.Module):
    """One `DecoderLayer` and the final LayerNorm."""

    def __init__(self, dim: int, heads: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim, heads, hidden)])
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, query: torch.Tensor, content: torch.Tensor, memory: torch.Tensor, query_mask: torch.Tensor,
        padding: torch.Tensor | None = None,
    ) -> torch.Tensor:  # fmt: skip
        return self.norm(self.layers[0](query, content, memory, query_mask, padding))


class TokenEmbedding(nn.Module):
    """Token embedding scaled by `sqrt(dim)`."""

    def __init__(self, tokens: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(tokens, dim)
        self.scale = math.sqrt(dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.scale * self.embedding(ids)


class PARSeq(nn.Module):
    """PARSeq for inference: `[B, 3, 32, 128]` in `[-1, 1]` -> logits `[B, 26, classes]`.

    Token ids: 0 = end of sequence, 1..94 the charset, 95 = begin, 96 = padding (`strhub.data.utils.Tokenizer`).
    """

    def __init__(self, dim: int, enc_heads: int, dec_heads: int, *, depth: int = 12, ratio: int = 4) -> None:
        super().__init__()
        classes = 95  # end of sequence + 94 characters (begin and padding are never predicted)
        self.encoder = Encoder(dim, depth, enc_heads, ratio)
        self.decoder = Decoder(dim, dec_heads, dim * ratio)
        self.head = nn.Linear(dim, classes)
        self.text_embed = TokenEmbedding(classes + 2, dim)
        self.pos_queries = nn.Parameter(torch.zeros(1, MAX_LENGTH + 1, dim))
        self.eos, self.bos, self.pad = 0, classes, classes + 1

    def _decode(
        self, tgt: torch.Tensor, memory: torch.Tensor, query: torch.Tensor, query_mask: torch.Tensor,
        padding: torch.Tensor | None = None,
    ) -> torch.Tensor:  # fmt: skip
        null = self.text_embed(tgt[:, :1])
        content = torch.cat((null, self.pos_queries[:, : tgt.shape[1] - 1] + self.text_embed(tgt[:, 1:])), 1)
        return self.decoder(query, content, memory, query_mask, padding)

    def forward(self, images: torch.Tensor, refine_iters: int = 1, *, stop: bool = True) -> torch.Tensor:
        """Greedy autoregressive decoding (stops once every sequence has an end token), then cloze refinement.

        The refinement queries all 26 positions over the decoded context, so the output has 26 steps whenever
        `refine_iters > 0` (upstream's shapes). `stop=False` always runs all 26 steps (no data-dependent exit, for
        export); the refined output is the same, as the context after the first end token is masked either way.
        """
        b, steps = images.shape[0], MAX_LENGTH + 1
        memory = self.encoder(images)
        queries = self.pos_queries[:, :steps].expand(b, -1, -1)
        mask = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=images.device), 1)
        tgt = torch.full((b, steps), self.pad, dtype=torch.long, device=images.device)
        tgt[:, 0] = self.bos
        logits = []
        for i in range(steps):
            j = i + 1  # one query at position i over the context up to it (the forward mask makes past outputs fixed)
            p = self.head(self._decode(tgt[:, :j], memory, queries[:, i:j], mask[i:j, :j]))
            logits.append(p)
            if j < steps:
                tgt[:, j] = p[:, 0].argmax(-1)
                if stop and bool((tgt == self.eos).any(-1).all()):
                    break
        out = torch.cat(logits, 1)
        cloze = mask & ~torch.triu(torch.ones_like(mask), 2)  # the forward mask with the right context unmasked
        bos = tgt[:, :1]
        for _ in range(refine_iters):
            tgt_in = torch.cat((bos, out[:, :-1].argmax(-1)), 1)
            padding = (tgt_in == self.eos).int().cumsum(-1) > 0  # tokens from the first end token on
            out = self.head(self._decode(tgt_in, memory, queries, cloze[:, : tgt_in.shape[1]], padding))
        return out
