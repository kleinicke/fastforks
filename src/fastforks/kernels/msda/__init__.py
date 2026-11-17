"""Multi-scale bilinear sampling and attention-weighted reduction in fp16."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"deform_sample": "ppppppiiiiiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def sample(value, shapes, shapes_list, start, loc, weights, step):  # noqa: ANN001, ANN201, ARG001, PLR0913, PLR0917 - transformers attention callback
    """Sample packed value levels using the original zero-padded, align_corners=False grid."""
    batch, length, heads, channels = value.shape
    _, queries, _, levels, points, _ = loc.shape
    if value.dtype != torch.float16 or loc.dtype != value.dtype or weights.dtype != value.dtype:
        msg = "The fused sampler requires fp16 values, locations and weights"
        raise ValueError(msg)
    out = value.new_empty((batch, queries, heads * channels))
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
