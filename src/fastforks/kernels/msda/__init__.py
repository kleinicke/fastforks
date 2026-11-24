"""Multi-scale bilinear sampling and attention-weighted reduction in fp16."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"deform_sample": "ppppppiiiiiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def sample(value, shapes, shapes_list, start, loc, weights, step):  # noqa: ANN001, ANN201, ARG001, PLR0913, PLR0917 - transformers attention callback
    """Sample packed value levels using the original zero-padded, align_corners=False grid."""
    batch, length, heads, channels = value.shape
    loc_batch, queries, loc_heads, levels, points, coords = loc.shape
    if value.dtype != torch.float16 or loc.dtype != value.dtype or weights.dtype != value.dtype:
        msg = "The fused sampler requires fp16 values, locations and weights"
        raise ValueError(msg)
    if (loc_batch, loc_heads, coords) != (batch, heads, 2) or weights.shape != loc.shape[:-1]:
        msg = "The fused sampler requires matching batch, head, query, level and point dimensions"
        raise ValueError(msg)
    if (
        shapes.dtype != torch.int64
        or start.dtype != torch.int64
        or shapes.shape != (levels, 2)
        or start.shape != (levels,)
    ):
        msg = "The fused sampler requires int64 spatial shapes [levels, 2] and level starts [levels]"
        raise ValueError(msg)
    if any(t.device != value.device for t in (shapes, start, loc, weights)):
        msg = "The fused sampler requires all tensors on the value device"
        raise ValueError(msg)
    out = value.new_empty((batch, queries, heads * channels))
    if not out.numel():
        return out
    load(value.device).launch(
        "deform_sample",
        *linear(out.numel()),
        value.contiguous(),
        shapes.contiguous(),
        start.contiguous(),
        loc.contiguous(),
        weights.contiguous(),
        out,
        length,
        queries,
        heads,
        channels,
        levels,
        points,
        out.numel(),
    )
    return out
