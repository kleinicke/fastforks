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

import contextvars
from typing import Any

import torch

from fastforks._core.patch import base_type, override
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


_KEPT: contextvars.ContextVar[bool] = contextvars.ContextVar("mlx_keep_float32", default=False)


def _kept_forward(self: torch.nn.Module, *args: Any) -> Any:  # noqa: ANN401 - the module's own arguments and result
    """The module's own `forward`; a `float32_call` of it while exporting for MLX (once: not inside itself)."""
    if not exporting_for_mlx() or _KEPT.get():
        return base_type(self).forward(self, *args)
    token = _KEPT.set(True)
    try:
        return float32_call(self, *args)
    finally:
        _KEPT.reset(token)


def keep_float32(*modules: torch.nn.Module) -> None:
    """Make every call of each module a `float32_call` in an MLX replay; for the float32 twin of an MLX runner.

    A float16 PyTorch runner keeps a submodule float32 by converting it (`module.float()`, `pure_half(fp32=...)`, a
    cast to the module's weight dtype in the forward). The MLX runner's twin is float32 throughout, so none of that
    leaves a trace in the export and the submodule would run in the replay's float16. Call this on the twin's
    submodules (`mlx_wrap.twin_dtype() == "float16"`), on the modules the forward calls: a `ModuleList` stands for its
    entries. Outside an MLX export the modules are unchanged. The result is float32: where the float16 runner casts
    it back (`.to(x.dtype)`, a no-op on the twin), the forward needs `to_run_dtype`.
    """
    for module in modules:
        if isinstance(module, torch.nn.ModuleList):
            keep_float32(*module)
            continue
        if "forward" in dict(getattr(type(module), "_fastforks_methods", ())):
            msg = f"{type(module).__name__}: its forward is already overridden"
            raise ValueError(msg)
        override(module, _kept_forward)
