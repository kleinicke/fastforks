"""Export detection that costs nothing at run time.

`torch.onnx.is_in_onnx_export()` imports `torch.onnx` on first use (~1.5 s: onnxscript, the dynamo exporter), so a
forward that asks it pays that on its first call. No ONNX export can be running before `torch.onnx` is imported.
"""

from __future__ import annotations

import contextvars
import sys
from typing import TYPE_CHECKING

from fastforks._core.lazy import lazy

if TYPE_CHECKING:
    import torch
else:
    torch = lazy("torch")


def in_onnx_export() -> bool:
    """`torch.onnx.is_in_onnx_export()`, without importing `torch.onnx` when nothing has."""
    onnx = sys.modules.get("torch.onnx")
    return onnx is not None and onnx.is_in_onnx_export()


EXPORTING_FOR_MLX: contextvars.ContextVar[bool] = contextvars.ContextVar("mlx_exporting", default=False)
"""True while a network is exported for the MLX replay: models pick a functional, cond-free form."""


def exporting_for_mlx() -> bool:
    """Whether this is the MLX replay's non-strict export."""
    # MLX uses non-strict export. Dynamo handles compile and strict export, and on torch 2.8 it cannot trace
    # ContextVar.get(). Skip the context lookup in a Dynamo trace before the call becomes part of its graph.
    return not torch.compiler.is_dynamo_compiling() and EXPORTING_FOR_MLX.get()
