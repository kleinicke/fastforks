"""Indexed k-nearest-neighbour attention used by the ReSplat point transformer."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"knn8": "pppl", "knn16": "pppl"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_WIDTH = 64  # single-head point transformer attention projection


def attention(qkv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Attend to the caller's precomputed neighbour indices without gathering their K/V feature tensors."""
    n = qkv.shape[0]
    if (
        qkv.shape != (n, 3 * _WIDTH)
        or qkv.dtype != torch.float16
        or not qkv.is_contiguous()
        or qkv.data_ptr() % 16
        or indices.ndim != 2  # noqa: PLR2004 - matrix
        or indices.shape[0] != n
        or indices.shape[1] not in (8, 16)
        or indices.dtype != torch.int64
        or indices.device != qkv.device
        or not indices.is_contiguous()
    ):
        msg = "Expected aligned packed fp16 [N, 192] QKV and contiguous int64 [N, 8 or 16] neighbour indices"
        raise ValueError(msg)
    out = qkv.new_empty(n, _WIDTH)
    load(qkv.device).launch(f"knn{indices.shape[1]}", *linear(n * 32), qkv, indices, out, n)
    return out
