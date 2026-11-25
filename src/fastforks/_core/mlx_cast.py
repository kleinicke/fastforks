"""Casts to the network's own dtype that the MLX replay maps to its run dtype.

An MLX runner exports a float32 PyTorch twin, so `x.to(ctx_dtype)` and `.to(y.dtype)` are recorded as literal
float32 casts: a float16 replay would run everything downstream of them in float32 (twice the memory, slower).
`to_run_dtype` records `fastforks::run_dtype` instead while exporting for MLX; `fastforks.shared.mlx_graph` lowers it
to a cast to the replay's dtype.

The other way round, `.float()` on a float32 twin value records nothing, so a float32 island (a disparity state
summed over the iterations, softmax weights of an upsampling) runs in the replay's dtype. `to_float32` records
`fastforks::float32`, which the replay keeps as a cast to float32.
"""

from __future__ import annotations

from typing import Any

import torch

from fastforks._core.tracing import exporting_for_mlx


@torch.library.custom_op("fastforks::run_dtype", mutates_args=())
def run_dtype(x: torch.Tensor) -> torch.Tensor:
    """Identity in PyTorch (a copy: a custom op's output may not alias its input)."""
    return x.clone()


@run_dtype.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def to_run_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """`x.to(dtype)`, where `dtype` is the network's activation dtype; the MLX replay's run dtype on MLX."""
    if exporting_for_mlx():
        return run_dtype(x if x.dtype == dtype else x.to(dtype))
    return x.to(dtype)


@torch.library.custom_op("fastforks::float32", mutates_args=())
def float32(x: torch.Tensor) -> torch.Tensor:
    """`x.float()` as a copy (a custom op's output may not alias its input)."""
    return x.to(torch.float32, copy=True)


@float32.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.float32)


def to_float32(x: torch.Tensor) -> torch.Tensor:
    """`x.float()` that stays a float32 cast in a float16 MLX replay."""
    if exporting_for_mlx():
        return float32(x)
    return x.float()


def _float32_arg(a: Any) -> Any:  # noqa: ANN401 - a tensor or nested lists of tensors
    return [_float32_arg(v) for v in a] if isinstance(a, (list, tuple)) else to_float32(a)


def float32_call(module: torch.nn.Module, *args: Any) -> Any:  # noqa: ANN401 - the module's own arguments and result
    """`module(*args)` with float32 inputs (`to_float32`); a float16 MLX replay also keeps its weights float32.

    The arguments are tensors or (nested) lists of tensors. Outside an MLX export this is
    `module(*(a.float() for a in args))`: the module's own weights are used as they are (a float32 stage of a
    half-precision PyTorch network is `.float()`-ed by its runner).
    """
    args = tuple(_float32_arg(a) for a in args)
    if not exporting_for_mlx():
        return module(*args)
    state = {k: to_float32(v) if v.is_floating_point() else v for k, v in module.state_dict(keep_vars=True).items()}
    return torch.func.functional_call(module, state, args)
