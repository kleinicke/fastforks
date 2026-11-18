"""Fused fp32 per-head RMS normalization and rotary embedding on PointDiT's packed fp16 projections."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library

SIGNATURES = {"pointdit_qk": "pppiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_HEAD = 64
_QKV = 3


def normalize_rope(qkv: torch.Tensor, weight: torch.Tensor, table: torch.Tensor) -> None:
    """Replace Q and K in place, preserving V and rounding normalized values to fp16 before rotation."""
    b, n, parts, heads, width = qkv.shape
    if (
        min(b, n, heads) < 1
        or (parts, width) != (_QKV, _HEAD)
        or weight.shape != (2, 1, width)
        or table.shape != (n, 2, width // 2)
        or qkv.dtype != torch.float16
        or weight.dtype != table.dtype
        or table.dtype != torch.float32
        or any(t.device != qkv.device or not t.is_contiguous() for t in (weight, table, qkv))
    ):
        msg = "PointDiT expects contiguous fp16 QKV [B,N,3,H,64], fp32 scales [2,1,64] and angles [N,2,32] on one GPU"
        raise ValueError(msg)
    rows = b * n * 2 * heads
    grid = (min((rows + 7) // 8, 65535), 1, 1)
    load(qkv.device).launch("pointdit_qk", grid, (256, 1, 1), qkv, weight, table, n, heads, rows)
