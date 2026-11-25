"""Decoder steps as single ops, so the MLX replay can run each as one Metal kernel.

A decoder stage that doubles a map, adds a skip and applies a ReLU is three full-size passes and three full-size
buffers in a replayed graph; at 1080p the 64-channel maps (0.27 GB each in float16) are larger than MLX's buffer
cache, so each is a fresh allocation. Here the stage is one op: eagerly the plain PyTorch ops, in an exported graph
the `fastforks::` op that `fastforks.shared.mlx_graph` lowers to `fastforks.kernels.graph_mlx.ops`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastforks._core.tracing import exporting_for_mlx


def _up2(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


def _up2_skip(x: torch.Tensor, skip: torch.Tensor, *, relu: bool) -> torch.Tensor:
    y = _up2(x) + skip
    return F.relu(y) if relu else y


@torch.library.custom_op("fastforks::up2_skip", mutates_args=())
def _up2_skip_op(x: torch.Tensor, skip: torch.Tensor, relu: bool) -> torch.Tensor:  # noqa: FBT001 - an op's schema
    return _up2_skip(x, skip, relu=relu)


@_up2_skip_op.register_fake
def _(x: torch.Tensor, skip: torch.Tensor, relu: bool) -> torch.Tensor:  # noqa: ARG001, FBT001 - same schema
    return x.new_empty(skip.shape)


def up2_skip(x: torch.Tensor, skip: torch.Tensor, *, relu: bool = False) -> torch.Tensor:
    """`bilinear2x(x) + skip` (`align_corners=False`), then a ReLU if `relu`: `[B, C, H, W]`, `[B, C, 2H, 2W]`.

    While exporting for the MLX replay the `fastforks::up2_skip` op: one Metal kernel there (the taps and the sum in
    float32, one rounding to the run dtype).
    """
    if exporting_for_mlx():
        return _up2_skip_op(x, skip, relu)
    return _up2_skip(x, skip, relu=relu)


def _up2_proj(
    x: torch.Tensor, src: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, relu: bool
) -> torch.Tensor:
    y = _up2(x) + F.conv2d(src, weight[:, :, None, None].to(src.dtype), bias.to(src.dtype)).to(x.dtype)
    return F.relu(y) if relu else y


@torch.library.custom_op("fastforks::up2_proj", mutates_args=())
def _up2_proj_op(
    x: torch.Tensor,
    src: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    relu: bool,  # noqa: FBT001 - an op's schema
) -> torch.Tensor:
    return _up2_proj(x, src, weight, bias, relu=relu)


@_up2_proj_op.register_fake
def _(x: torch.Tensor, src: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, relu: bool) -> torch.Tensor:  # noqa: ARG001, FBT001
    return x.new_empty((x.shape[0], x.shape[1], *src.shape[2:]))


def up2_proj(
    x: torch.Tensor, src: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, relu: bool = False
) -> torch.Tensor:
    """`bilinear2x(x) + conv1x1(src, weight, bias)`, then a ReLU if `relu`.

    `x` `[B, C, H, W]`, `src` `[B, K, 2H, 2W]` with few channels (an image), `weight` `[C, K]`, `bias` `[C]`. While
    exporting for the MLX replay the `fastforks::up2_proj` op: one Metal kernel there, so the projected skip (`C`
    channels at the output size) is never a buffer.
    """
    if exporting_for_mlx():
        return _up2_proj_op(x, src, weight, bias, relu)
    return _up2_proj(x, src, weight, bias, relu=relu)


def _area_pool(x: torch.Tensor, factor: int) -> torch.Tensor:
    return F.avg_pool2d(x.float(), factor).to(x.dtype)


@torch.library.custom_op("fastforks::area_pool", mutates_args=())
def _area_pool_op(x: torch.Tensor, factor: int) -> torch.Tensor:
    return _area_pool(x, factor)


@_area_pool_op.register_fake
def _(x: torch.Tensor, factor: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:2], x.shape[2] // factor, x.shape[3] // factor))


def area_pool(x: torch.Tensor, factor: int) -> torch.Tensor:
    """The mean of each `factor` x `factor` block of `[B, C, H, W]` (`H`, `W` multiples of `factor`), float32 sums.

    `F.interpolate(x, scale_factor=1 / factor, mode="area")` for such sizes. While exporting for the MLX replay the
    `fastforks::area_pool` op: one Metal kernel there (no float32 copy of the input).
    """
    if exporting_for_mlx():
        return _area_pool_op(x, factor)
    return _area_pool(x, factor)
