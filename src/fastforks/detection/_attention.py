"""Attention shared by the DETR-style detection families: multi-head and multi-scale deformable."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    """`nn.MultiheadAttention` (batch first, no masks) with its parameter names, as scaled-dot-product attention."""

    def __init__(self, d: int, heads: int) -> None:
        """Create the parameters."""
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d))
        self.out_proj = nn.Linear(d, d)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """`[B, S, d]` queries, keys, values -> `[B, Sq, d]`."""
        d = q.shape[-1]
        w, b = self.in_proj_weight, self.in_proj_bias
        q, k, v = (F.linear(t, w[i * d : (i + 1) * d], b[i * d : (i + 1) * d]) for i, t in enumerate((q, k, v)))
        q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).flatten(2))


def deformable_sample(
    value: torch.Tensor,
    loc: torch.Tensor,
    weights: torch.Tensor,
    shapes: list[tuple[int, int]],
    points: int | list[int],
) -> torch.Tensor:
    """Multi-scale deformable attention core: bilinear samples of the per-level maps, weighted and summed.

    `value` `[B, S, heads, dh]` (levels flattened one after the other), `loc` `[B, Q, heads, sum(points), 2]` sampling
    locations in 0..1, `weights` `[B, Q, heads, sum(points)]` (softmaxed); `points` per level (one number: all levels).
    Returns `[B, Q, heads * dh]`.
    """
    b, heads, dh = value.shape[0], value.shape[2], value.shape[3]
    values = value.permute(0, 2, 3, 1).flatten(0, 1).split([h * w for h, w in shapes], dim=-1)
    maps = [v.reshape(b * heads, dh, h, w) for v, (h, w) in zip(values, shapes, strict=True)]
    return sample_maps(maps, loc, weights, points)


def sample_maps(
    maps: list[torch.Tensor], loc: torch.Tensor, weights: torch.Tensor, points: int | list[int]
) -> torch.Tensor:
    """`deformable_sample` on the per-level value maps `[B * heads, dh, h, w]`."""
    b, q, heads, n = weights.shape
    dh = maps[0].shape[1]
    grids = (2 * loc - 1).permute(0, 2, 1, 3, 4).flatten(0, 1).split(points, dim=-2)  # per level [B*heads, Q, p, 2]
    sampled = [
        F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        for v, g in zip(maps, grids, strict=True)
    ]
    weights = weights.permute(0, 2, 1, 3).reshape(b * heads, 1, q, n)
    return (torch.cat(sampled, dim=-1) * weights).sum(-1).reshape(b, heads * dh, q).transpose(1, 2)
