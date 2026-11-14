"""Export detection that costs nothing at run time.

`torch.onnx.is_in_onnx_export()` imports `torch.onnx` on first use (~1.5 s: onnxscript, the dynamo exporter), so a
forward that asks it pays that on its first call. No ONNX export can be running before `torch.onnx` is imported.
"""

from __future__ import annotations

import sys


def in_onnx_export() -> bool:
    """`torch.onnx.is_in_onnx_export()`, without importing `torch.onnx` when nothing has."""
    onnx = sys.modules.get("torch.onnx")
    return onnx is not None and onnx.is_in_onnx_export()
