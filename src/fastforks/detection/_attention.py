"""Attention shared by the DETR-style detection families: multi-head and multi-scale deformable."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.tracing import in_onnx_export


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


def _export_bilinear(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Zero-padded, align_corners=False sampling expressed as four gathers."""
    batch, channels, height, width = value.shape
    gh, gw = grid.shape[1:3]
    x = (grid[..., 0] + 1) * (width / 2) - 0.5
    y = (grid[..., 1] + 1) * (height / 2) - 0.5
    x0, y0 = x.floor(), y.floor()
    dx, dy = x - x0, y - y0
    flat = value.flatten(2)

    def corner(xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
        mask = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        xi = torch.where(xi < 0, torch.zeros_like(xi), xi)
        yi = torch.where(yi < 0, torch.zeros_like(yi), yi)
        xi = torch.where(xi >= width, torch.full_like(xi, width - 1), xi)
        yi = torch.where(yi >= height, torch.full_like(yi, height - 1), yi)
        idx = yi.long() * width + xi.long()
        idx = idx.reshape(batch, 1, gh * gw).expand(batch, channels, gh * gw)
        return flat.gather(2, idx).reshape(batch, channels, gh, gw) * mask.unsqueeze(1)

    return (
        corner(x0, y0) * ((1 - dx) * (1 - dy)).unsqueeze(1)
        + corner(x0 + 1, y0) * (dx * (1 - dy)).unsqueeze(1)
        + corner(x0, y0 + 1) * ((1 - dx) * dy).unsqueeze(1)
        + corner(x0 + 1, y0 + 1) * (dx * dy).unsqueeze(1)
    )


def sample_maps(
    maps: list[torch.Tensor],
    loc: torch.Tensor,
    weights: torch.Tensor,
    points: int | list[int],
    *,
    export_with_gather: bool = False,
) -> torch.Tensor:
    """`deformable_sample` on the per-level value maps `[B * heads, dh, h, w]`."""
    b, q, heads, n = weights.shape
    dh = maps[0].shape[1]
    grids = (2 * loc - 1).permute(0, 2, 1, 3, 4).flatten(0, 1).split(points, dim=-2)  # per level [B*heads, Q, p, 2]
    dtype = maps[0].dtype
    exporting = in_onnx_export()
    if exporting and dtype != torch.float32:
        # fp32 island of the exported graph: TensorRT's fp16 GridSample returns NaN / huge values (D-FINE-X at 800x1344)
        maps, grids = [v.float() for v in maps], [g.float() for g in grids]
    weights = weights.permute(0, 2, 1, 3).reshape(b * heads, 1, q, n).split(points, dim=-1)
    out = None
    for v, g, w in zip(maps, grids, weights, strict=True):  # fastforks: per level, no cat of the samples (a copy)
        s = (
            _export_bilinear(v, g)
            if exporting and export_with_gather
            else F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        ).to(dtype)
        s = (s * w).sum(-1)
        out = s if out is None else out + s
    return out.reshape(b, heads * dh, q).transpose(1, 2)  # ty: ignore[unresolved-attribute]
