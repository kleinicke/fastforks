"""Group-wise correlation cost volume as one op, so the MLX replay can run it as one Metal kernel.

`gwc_volume(left, right, depth, groups)[b, g, d, y, x]` is the channel mean, per group, of `left[x] * right[x - d]`
(zero where `x < d`) -- upstream's `build_gwc_volume` family. Eagerly it is the slab loop; an exported graph keeps
the op and `fastforks.shared.mlx_graph` lowers it to `fastforks.kernels.graph_mlx.ops.gwc_volume`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastforks._core.sampling import grid_sample
from fastforks._core.tracing import EXPORTING_FOR_MLX


@torch.library.custom_op("fastforks::gwc_volume", mutates_args=())
def gwc_volume(left: torch.Tensor, right: torch.Tensor, depth: int, groups: int) -> torch.Tensor:
    """`[B, groups, depth, H, W]` group-wise correlation of `[B, C, H, W]` features (`C % groups == 0`)."""
    b, c, h, w = left.shape
    slabs = []
    for i in range(depth):
        if i < w:
            prod = (left[..., i:] * right[..., : w - i]).view(b, groups, c // groups, h, w - i).mean(dim=2)
            slabs.append(F.pad(prod, (i, 0)))
        else:
            slabs.append(left.new_zeros((b, groups, h, w)))
    return torch.stack(slabs, dim=2)


@gwc_volume.register_fake
def _(left: torch.Tensor, right: torch.Tensor, depth: int, groups: int) -> torch.Tensor:  # noqa: ARG001 - same schema
    b, _, h, w = left.shape
    return left.new_empty((b, groups, depth, h, w))


def normalized_groups(x: torch.Tensor, groups: int, eps: float = 1e-5) -> torch.Tensor:
    """`x / (||x|| + eps)` over each group's channels (upstream's `groupwise_correlation_norm`), `[B, C, H, W]`."""
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    return (x / (torch.norm(x, p=2, dim=2, keepdim=True) + eps)).view(b, c, h, w)


@torch.library.custom_op("fastforks::warp_dot", mutates_args=())
def warp_dot(feat: torch.Tensor, other: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Plane-sweep correlation: `out[n, d, y, x] = sum_c feat[n, c, y, x] * bilinear(other[n], grid[n, d, y*W+x])[c]`.

    `feat`, `other` `[N, C, H, W]`, `grid` `[N, D, H * W, 2]` (`grid_sample` coordinates, align_corners, zeros
    padding). Returns `[N, D, H, W]` float32. Eagerly it is `grid_sample` plus a channel sum (32 depths at a time);
    `fastforks.shared.mlx_graph` lowers it to one Metal kernel (no `[N, C, D, H, W]` warped volume).
    """
    n, c, h, w = feat.shape
    d = grid.shape[1]
    out = []
    for j in range(0, d, 32):
        g = grid[:, j : j + 32].reshape(n, -1, w, 2).to(other.dtype)
        warped = grid_sample(other, g, mode="bilinear", padding_mode="zeros", align_corners=True)
        out.append((warped.view(n, c, -1, h, w) * feat[:, :, None]).sum(1, dtype=torch.float32))
    return torch.cat(out, 1)


@warp_dot.register_fake
def _(feat: torch.Tensor, other: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:  # noqa: ARG001 - same schema
    n, _, h, w = feat.shape
    return feat.new_empty((n, grid.shape[1], h, w), dtype=torch.float32)


def _linear_sample(vol: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    d = vol.shape[-1]
    x0 = torch.floor(x)
    w1 = x - x0
    i0 = x0.long()
    i1 = i0 + 1
    v0 = ((i0 >= 0) & (i0 < d)).float() * (1 - w1)
    v1 = ((i1 >= 0) & (i1 < d)).float() * w1
    idx = torch.cat([i0, i1], 1).clamp_(0, d - 1)
    wgt = torch.cat([v0, v1], 1)
    k = x.shape[1]
    if vol.dim() == 3:  # noqa: PLR2004 - [P, C, D] against [P, D]
        p, c, _ = vol.shape
        s = vol.gather(2, idx.unsqueeze(1).expand(p, c, 2 * k)).float() * wgt.unsqueeze(1)
        return (s[..., :k] + s[..., k:]).reshape(p, c * k)
    s = vol.gather(1, idx).float() * wgt
    return s[:, :k] + s[:, k:]


@torch.library.custom_op("fastforks::linear_sample", mutates_args=())
def _linear_sample_op(vol: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _linear_sample(vol, x)


@_linear_sample_op.register_fake
def _(vol: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    c = vol.shape[1] if vol.dim() == 3 else 1  # noqa: PLR2004 - [P, C, D] against [P, D]
    return x.new_empty((x.shape[0], c * x.shape[1]), dtype=torch.float32)


def linear_sample(vol: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation of the IGEV-family geometry / correlation lookups (zeros outside, fp32 math).

    `vol` `[P, C, D]` or `[P, D]`, `x` `[P, K]` fp32 positions along `D`; returns `[P, C * K]` (`[P, K]`) float32,
    `out[p, c * K + k] = lerp(vol[p, c], x[p, k])`. Eagerly two gathers; while exporting for the MLX replay it is the
    `fastforks::linear_sample` op, one Metal kernel there (a compiled gather chain costs ~8x more per lookup).
    """
    if EXPORTING_FOR_MLX.get():
        return _linear_sample_op(vol, x)
    return _linear_sample(vol, x)


def _gwc_warp(left: torch.Tensor, right: torch.Tensor, disp: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = left.shape
    d = disp.shape[1]
    ys = torch.arange(h, dtype=left.dtype, device=left.device).view(1, 1, h, 1).expand(b, d, h, w)
    xs = torch.arange(w, dtype=left.dtype, device=left.device).view(1, 1, 1, w)
    gx = (xs - disp) / ((w - 1.0) / 2.0) - 1.0
    gy = ys / ((h - 1.0) / 2.0) - 1.0
    grid = torch.stack([gx, gy], dim=4).view(b, d * h, w, 2)
    warped = F.grid_sample(right, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    masked = torch.where((xs < disp)[:, None], 0, left[:, :, None])
    return (masked * warped.view(b, c, d, h, w)).view(b, groups, c // groups, d, h, w).mean(dim=2)


@torch.library.custom_op("fastforks::gwc_warp", mutates_args=())
def _gwc_warp_op(left: torch.Tensor, right: torch.Tensor, disp: torch.Tensor, groups: int) -> torch.Tensor:
    return _gwc_warp(left, right, disp, groups)


@_gwc_warp_op.register_fake
def _(left: torch.Tensor, right: torch.Tensor, disp: torch.Tensor, groups: int) -> torch.Tensor:  # noqa: ARG001
    b, _, h, w = left.shape
    return left.new_empty((b, groups, disp.shape[1], h, w))


def gwc_warp(left: torch.Tensor, right: torch.Tensor, disp: torch.Tensor, groups: int) -> torch.Tensor:
    """Group-wise correlation at per-pixel disparity samples (RT-MonSter++'s `build_gwc_volume_selective`).

    `left`, `right` `[B, C, H, W]`, `disp` `[B, D, H, W]`; returns `[B, groups, D, H, W]`: the per-group channel mean
    of `left[y, x]` times `right` bilinearly sampled at `(x - disp, y)` (upstream's `grid_sample` with coordinates
    normalized by `W - 1`, `align_corners=False`, zeros padding), zero where `x < disp`. Eagerly the warped volume;
    while exporting for the MLX replay the `fastforks::gwc_warp` op, one Metal kernel there (no `[B, C, D, H, W]`).
    """
    if EXPORTING_FOR_MLX.get():
        return _gwc_warp_op(left, right, disp, groups)
    return _gwc_warp(left, right, disp, groups)


@torch.library.custom_op("fastforks::upsample_soft_argmax", mutates_args=())
def upsample_soft_argmax(cost: torch.Tensor, max_disp: int, height: int, width: int) -> torch.Tensor:
    """Trilinear upsample of `[B, 1, d, h, w]` to `[max_disp, height, width]`, softmax over disparity, soft-argmin.

    Returns `[B, height, width]` float32 (upstream PSMNet / GwcNet test tail; `align_corners=False`).
    """
    up = F.interpolate(cost.float(), size=(max_disp, height, width), mode="trilinear", align_corners=False)
    values = torch.arange(max_disp, device=cost.device, dtype=torch.float32).view(1, max_disp, 1, 1)
    return torch.sum(F.softmax(up.squeeze(1), dim=1) * values, 1)


@upsample_soft_argmax.register_fake
def _(cost: torch.Tensor, max_disp: int, height: int, width: int) -> torch.Tensor:  # noqa: ARG001 - same schema
    return cost.new_empty((cost.shape[0], height, width), dtype=torch.float32)


def _corr_windows(
    f1: torch.Tensor, levels: list[torch.Tensor], coords: torch.Tensor, radius: int, scale: float
) -> torch.Tensor:
    b, c, h, w = f1.shape
    side, k = 2 * radius + 2, 2 * radius + 1
    a = f1.flatten(2).transpose(1, 2)  # [B, P, C]
    steps = torch.arange(side, device=f1.device)
    out = []
    for lvl, f2 in enumerate(levels):
        hl, wl = f2.shape[-2:]
        flat = f2.flatten(2).transpose(1, 2)  # [B, hl * wl, C]
        xy = coords.float().flatten(2) / 2**lvl
        rows = []
        for i in range(b):
            cx, cy = xy[i, 0], xy[i, 1]
            fx, fy = cx.floor(), cy.floor()
            wx, wy = (cx - fx)[:, None, None], (cy - fy)[:, None, None]
            gx, gy = fx.long()[:, None] - radius + steps, fy.long()[:, None] - radius + steps
            inside = ((gy >= 0) & (gy < hl))[:, :, None] & ((gx >= 0) & (gx < wl))[:, None, :]
            idx = gy.clamp(0, hl - 1)[:, :, None] * wl + gx.clamp(0, wl - 1)[:, None, :]
            near = flat[i].index_select(0, idx.view(-1)).view(h * w, side * side, c)
            d = torch.bmm(near.float(), a[i, :, :, None].float()).view(h * w, side, side) * inside * scale  # [P, y, x]
            v = (1 - wy) * ((1 - wx) * d[:, :-1, :-1] + wx * d[:, :-1, 1:]) + wy * (
                (1 - wx) * d[:, 1:, :-1] + wx * d[:, 1:, 1:]
            )
            rows.append(v.transpose(1, 2).reshape(h * w, k * k).t())  # x offset major
        out.append(torch.stack(rows))
    return torch.cat(out, 1).view(b, len(levels) * k * k, h, w)


@torch.library.custom_op("fastforks::corr_windows", mutates_args=())
def _corr_windows_op(
    f1: torch.Tensor, levels: list[torch.Tensor], coords: torch.Tensor, radius: int, scale: float
) -> torch.Tensor:
    return _corr_windows(f1, levels, coords, radius, scale)


@_corr_windows_op.register_fake
def _(f1: torch.Tensor, levels: list[torch.Tensor], coords: torch.Tensor, radius: int, scale: float) -> torch.Tensor:  # noqa: ARG001
    b, _, h, w = f1.shape
    return f1.new_empty((b, len(levels) * (2 * radius + 1) ** 2, h, w), dtype=torch.float32)


def corr_windows(
    f1: torch.Tensor, levels: list[torch.Tensor], coords: torch.Tensor, radius: int, scale: float
) -> torch.Tensor:
    """RAFT-style correlation windows without the all-pairs volume (`[B, L * (2r + 1)^2, H, W]` float32).

    `f1` `[B, C, H, W]`; `levels`: the second feature map and its pooled levels `[B, C, h_l, w_l]` (level `l` of the
    volume is `scale * f1 . levels[l]`); `coords` `[B, 2, H, W]` (x, y at level 0). Channel `l K^2 + i K + j` is level
    `l` bilinearly sampled (align_corners, zeros outside) at `(x / 2^l + i - r, y / 2^l + j - r)`: x offset major, as
    RAFT's `CorrBlock`. Eagerly a gather of each pixel's (2r + 2)^2 neighbours; while exporting for the MLX replay the
    `fastforks::corr_windows` op, one Metal kernel there (memory O(HW), not O((HW)^2)).
    """
    if EXPORTING_FOR_MLX.get():
        return _corr_windows_op(f1, levels, coords, radius, scale)
    return _corr_windows(f1, levels, coords, radius, scale)
