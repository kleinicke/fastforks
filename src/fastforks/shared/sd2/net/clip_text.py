# fastforks: the CLIP text transformer of `transformers.CLIPTextModel` (huggingface/transformers, Apache-2.0, see
# ../LICENSE), re-implemented without transformers for the SD2 text encoder (OpenCLIP ViT-H/14 text tower cut
# to 23 layers: width 1024, 16 heads, exact GELU, causal mask). Module and parameter names are transformers' below
# `text_model.`. Marigold only ever encodes the empty prompt, once; the output is `last_hidden_state`.
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = (p(x).view(b, n, self.heads, -1).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(o.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Layer(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, heads)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, 4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class Embeddings(nn.Module):
    def __init__(self, vocab: int, dim: int, positions: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, dim)
        self.position_embedding = nn.Embedding(positions, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(ids) + self.position_embedding.weight[: ids.shape[1]][None]


class Encoder(nn.Module):
    def __init__(self, dim: int, heads: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer(dim, heads) for _ in range(depth)])


class CLIPTextTransformer(nn.Module):
    """Token ids `[B, n]` -> `last_hidden_state` `[B, n, 1024]` (after the final LayerNorm)."""

    def __init__(
        self, vocab: int = 49408, dim: int = 1024, heads: int = 16, depth: int = 23, positions: int = 77
    ) -> None:
        super().__init__()
        self.embeddings = Embeddings(vocab, dim, positions)
        self.encoder = Encoder(dim, heads, depth)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(ids)
        for layer in self.encoder.layers:
            x = layer(x)
        return self.final_layer_norm(x)
