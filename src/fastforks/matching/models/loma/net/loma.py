"""LoMa's matcher: upstream davnords/loma `loma/loma.py` (MIT) `LoMa.forward` + `filter_matches`, inference only.
fastforks changes: the transformer blocks and the positional encoding are fastforks.shared.lightglue's (LoMa copies
LightGlue's; the same state-dict keys), attention through torch's SDPA in the compute dtype as upstream (no float16
cast); the training-time `log_double_softmax` and the unused `matchability` head's forward removed (the weights stay
so the checkpoint loads strictly); batch 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.lightglue.net.lightglue import (
    CrossBlock,
    LearnableFourierPositionalEncoding,
    SelfBlock,
    apply_cached_rotary_emb,
)


class _SelfBlock(SelfBlock):
    """Upstream LoMa `SelfBlock`: LightGlue's with plain SDPA (no float16 cast, no mask)."""

    def forward(self, x: torch.Tensor, encoding: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002
        qkv = self.Wqkv(x).unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2].contiguous()
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = F.scaled_dot_product_attention(q, k, v)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class _CrossBlock(CrossBlock):
    """Upstream LoMa `CrossBlock`: LightGlue's with plain SDPA both ways."""

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, mask: torch.Tensor | None = None) -> list[torch.Tensor]:  # noqa: ARG002
        qk0, qk1 = self.to_qk(x0), self.to_qk(x1)
        v0, v1 = self.to_v(x0), self.to_v(x1)
        qk0, qk1, v0, v1 = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (qk0, qk1, v0, v1))
        m0 = F.scaled_dot_product_attention(qk0, qk1, v1)
        m1 = F.scaled_dot_product_attention(qk1, qk0, v0)
        m0, m1 = (self.to_out(t.transpose(1, 2).flatten(start_dim=-2)) for t in (m0, m1))
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return [x0, x1]


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = _SelfBlock(embed_dim, num_heads, flash=True)  # flash: keeps torch's flash SDP enabled
        self.cross_attn = _CrossBlock(embed_dim, num_heads, flash=True)

    def forward(
        self, desc0: torch.Tensor, desc1: torch.Tensor, enc0: torch.Tensor, enc1: torch.Tensor
    ) -> list[torch.Tensor]:
        desc0 = self.self_attn(desc0, enc0)
        desc1 = self.self_attn(desc1, enc1)
        return self.cross_attn(desc0, desc1)


class MatchAssignment(nn.Module):
    """Upstream inference head: the product of the row and column softmaxes of the projected similarity."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.matchability = nn.Linear(dim, 1)  # training only (upstream `log_double_softmax`)
        self.final_proj = nn.Linear(dim, dim)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        d = mdesc0.shape[-1]
        sim = torch.einsum("bmd,bnd->bmn", mdesc0 / d**0.25, mdesc1 / d**0.25)
        return F.softmax(sim, dim=2) * F.softmax(sim, dim=1)


def filter_matches(scores: torch.Tensor, th: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `filter_matches` for batch 1: mutual maxima above `th`. `[1, M, N]` -> match index per row (-1:
    none) `[M]` and its score `[M]`."""
    max0, max1 = scores[0].max(1), scores[0].max(0)
    m0, m1 = max0.indices, max1.indices
    mutual0 = torch.arange(m0.shape[0], device=m0.device) == m1[m0]
    mscores0 = torch.where(mutual0, max0.values, max0.values.new_tensor(0))
    valid0 = mutual0 & (mscores0 > th)
    return torch.where(valid0, m0, -1), mscores0


class LoMaNet(nn.Module):
    """Upstream `LoMa` without its detector and descriptor: keypoints (normalised) and descriptions of both images
    `[1, M, 2]`, `[1, N, 2]`, `[1, M, D]`, `[1, N, D]` -> match scores `[1, M, N]`."""

    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, n_layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        head_dim = embed_dim // num_heads
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim, gamma=1.0)
        self.transformers = nn.ModuleList([TransformerLayer(embed_dim, num_heads) for _ in range(n_layers)])
        self.log_assignment = nn.ModuleList([MatchAssignment(embed_dim) for _ in range(n_layers)])

    def forward(
        self, kpts0: torch.Tensor, kpts1: torch.Tensor, desc0: torch.Tensor, desc1: torch.Tensor
    ) -> torch.Tensor:
        desc0, desc1 = self.input_proj(desc0), self.input_proj(desc1)
        enc0, enc1 = self.posenc(kpts0), self.posenc(kpts1)
        for layer in self.transformers:
            desc0, desc1 = layer(desc0, desc1, enc0, enc1)
        return self.log_assignment[-1](desc0, desc1)
