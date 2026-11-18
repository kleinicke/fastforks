"""Portable sampling operators used by ONNX export graphs."""

from __future__ import annotations

import torch


def export_bilinear(value: torch.Tensor, grid: torch.Tensor, *, align_corners: bool = False) -> torch.Tensor:
    """Zero-padded bilinear sampling expressed as four gathers for ONNX."""
    batch, channels, height, width = value.shape
    gh, gw = grid.shape[1:3]
    if align_corners:
        x = (grid[..., 0] + 1) / 2 * (width - 1)
        y = (grid[..., 1] + 1) / 2 * (height - 1)
    else:
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
