"""MapAnything dense-head projection and model-space normal/depth discontinuities."""

import math
from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {
    "dense_project": "ppppl",
    "normals": "pppiilllll",
    "angle_edge": "ppiilffi",
    "depth_edges": "ppppiillllf",
}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def project(features: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply the learned FP32 128-to-6 head without materializing a float32 copy of the full half feature map."""
    views, channels, height, width = features.shape
    if (
        min(views, height, width) < 1
        or channels != 128  # noqa: PLR2004 - trained head width
        or features.dtype != torch.float16
        or not features.is_contiguous(memory_format=torch.channels_last)
        or weight.dtype != torch.float32
        or weight.shape != (6, 128)
        or not weight.is_contiguous()
        or bias.dtype != torch.float32
        or bias.shape != (6,)
        or not bias.is_contiguous()
        or features.device != weight.device
        or features.device != bias.device
    ):
        msg = "Expected channels-last FP16 features and the matching FP32 128-to-6 dense head on one device"
        raise ValueError(msg)
    pixels = views * height * width
    out = torch.empty((views, height, width, 6), device=features.device, dtype=torch.float32)
    grid = (min((pixels + 31) // 32, 65535), 1, 1)
    load(features.device).launch("dense_project", grid, (192, 1, 1), features, weight, bias, out, pixels)
    return out


def edge_mask(
    points: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor, angle: float, relative: float
) -> torch.Tensor:
    """Keep pixels without a simultaneous normal and relative-depth discontinuity in the neighboring window."""
    if (
        depth.ndim != 3  # noqa: PLR2004 - a stack of depth maps
        or depth.numel() == 0
        or not 0 <= angle <= math.pi
        or points.shape != (*depth.shape, 3)
        or points.dtype != torch.float32
        or depth.dtype != torch.float32
        or mask.shape != depth.shape
        or mask.dtype != torch.bool
        or points.device != depth.device
        or mask.device != depth.device
    ):
        msg = "Expected FP32 world points and depths and a matching validity mask on one device"
        raise ValueError(msg)
    _, height, width = depth.shape
    pixels = depth.numel()
    mask = mask.contiguous()
    normals = torch.empty((*depth.shape, 4), device=depth.device, dtype=torch.float32)
    edges = torch.empty_like(mask)
    result = torch.empty_like(mask)
    lib = load(depth.device)
    grid = linear(pixels)
    lib.launch("normals", *grid, points, mask, normals, height, width, pixels, *points.stride())
    lib.launch(
        "angle_edge", *grid, normals, edges, height, width, pixels, angle, math.cos(angle), int(points.stride(-1) == 1)
    )
    lib.launch("depth_edges", *grid, depth, mask, edges, result, height, width, pixels, *depth.stride(), relative)
    return result
