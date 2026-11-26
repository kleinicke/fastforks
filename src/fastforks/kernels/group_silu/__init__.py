"""GroupNorm and SiLU with tile statistics and one activation write."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"tile_moments": "ppiili", "group_moments": "ppilf", "normalize_silu": "pppppiill"}
SIGNATURES["affine_silu_h"] = "ppppilll"
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_TILE = 2048  # elements per statistics tile (TILE in group_silu.cu)


def group_silu(norm: torch.nn.GroupNorm, x: torch.Tensor, *, inplace: bool) -> torch.Tensor:
    """Apply a channels-last fp16 GroupNorm and SiLU, optionally overwriting the input after both reductions."""
    n, c, h, w = x.shape
    g = norm.num_groups
    if (
        norm.weight is None
        or norm.bias is None
        or norm.num_channels != c
        or c % g
        or not x.is_contiguous(memory_format=torch.channels_last)
        or x.dtype != torch.float16
        or any(v.device != x.device or v.dtype != x.dtype or not v.is_contiguous() for v in (norm.weight, norm.bias))
    ):
        msg = "Expected channels-last fp16 activations and matching contiguous fp16 GroupNorm weights and bias"
        raise ValueError(msg)
    width, spatial = c // g, h * w
    tiles = -(-(width * spatial) // _TILE)
    moments = x.new_empty((n * g, tiles, 2), dtype=torch.float32)
    stats = x.new_empty((n * g, 2), dtype=torch.float32)
    lib = load(x.device)
    lib.launch("tile_moments", (tiles, n * g, 1), (256, 1, 1), x, moments, c, width, spatial, tiles)
    lib.launch("group_moments", (n * g, 1, 1), (256, 1, 1), moments, stats, tiles, spatial * width, norm.eps)
    out = x if inplace else torch.empty_like(x)
    lib.launch(
        "normalize_silu", *linear(x.numel()), x, out, stats, norm.weight, norm.bias, c, width, spatial, x.numel()
    )
    return out


def affine_silu(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, *, inplace: bool = False
) -> torch.Tensor | None:
    """Fuse the half-rounded affine and SiLU, optionally reusing an owned contiguous NHWC map."""
    b, c, h, w = x.shape
    if (
        not x.is_cuda
        or x.dtype != torch.float16
        or c % 8
        or x.stride()[1:] != (1, c * w, c)
        or (inplace and not x.is_contiguous(memory_format=torch.channels_last))
        or any(t.data_ptr() % 16 for t in (x, scale, shift))
        or x.stride(0) % 8
        or any(
            t.shape != (b, c, 1, 1) or t.device != x.device or t.dtype != x.dtype or not t.is_contiguous()
            for t in (scale, shift)
        )
    ):
        return None
    out = x if inplace else torch.empty_like(x, memory_format=torch.channels_last)
    count = x.numel() // 8
    load(x.device).launch("affine_silu_h", *linear(count), x, scale, shift, out, c, h * w, x.stride(0), count)
    return out
