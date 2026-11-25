"""Omni-DC's finite-difference normal operator and full-resolution depth propagation."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"pool_edges": "ppiil", "normal": "ppppiil", "spn_step": "pppppiiillllffl"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_SCALE = 4
_STEPS = 6  # fixed: Omni-DC's full-resolution DySPN propagation
_PROPAGATION_CHANNELS = 3 * 5 * _STEPS


def apply(image: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    """Apply AᵀCA + diag(d) to the single-channel search direction for each image."""
    batch, channels, height, width = image.shape
    if (
        channels != 1
        or height % _SCALE
        or width % _SCALE
        or image.dtype != torch.float32
        or diagonal.shape != image.shape
        or diagonal.dtype != image.dtype
        or diagonal.device != image.device
    ):
        msg = "Expected matching float32 [B, 1, H, W] grids with spatial dimensions divisible by four"
        raise ValueError(msg)
    image = image.contiguous()
    out = torch.empty_like(image)
    cells = batch * ((height // 2) * (width // 2) + (height // 4) * (width // 4))
    edges = image.new_empty((cells, 2))
    kernel = load(image.device)
    kernel.launch("pool_edges", *linear(cells), image, edges, height, width, cells)
    kernel.launch("normal", *linear(out.numel()), image, diagonal.contiguous(), edges, out, height, width, out.numel())
    return out


def propagate(
    depth: torch.Tensor, offsets: torch.Tensor, missing_weight: torch.Tensor, observed_depth: torch.Tensor
) -> torch.Tensor:
    """Apply the six DySPN updates on two depth buffers, sampling from strided offset and affinity fields."""
    batch, channels, h, w = depth.shape
    if (
        channels != 1
        or min(batch, h, w) <= 0
        or h * w >= 2**31
        or not depth.is_cuda
        or offsets.shape != (batch, _PROPAGATION_CHANNELS, h, w)
        or missing_weight.shape != depth.shape
        or observed_depth.shape != depth.shape
        or any(
            x.device != depth.device or x.dtype != torch.float32
            for x in (depth, offsets, missing_weight, observed_depth)
        )
    ):
        msg = "Depth propagation needs CUDA FP32 depth and weight maps with six groups of five offsets and affinities"
        raise ValueError(msg)
    current = depth.contiguous()
    missing_weight = missing_weight.contiguous()
    observed_depth = observed_depth.contiguous()
    buffers = torch.empty_like(current), torch.empty_like(current)
    launch = linear(depth.numel())
    kernel = load(depth.device)
    for step in range(_STEPS):
        out = buffers[step % len(buffers)]
        kernel.launch(
            "spn_step", *launch, current, offsets, missing_weight, observed_depth, out, h, w, step,
            *offsets.stride(), 1.0 / h, 1.0 / w, depth.numel(),
        )  # fmt: skip
        current = out
    return current
