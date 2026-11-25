"""CUDA bilinear interpolation of a row range at the full output scale."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"bilinear_rows": "pp" + "i" * 6 + "ffl"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_VECTOR = 8
_ALIGNMENT = _VECTOR * torch.float16.itemsize


def rows(x: torch.Tensor, size: tuple[int, int], row_range: tuple[int, int]) -> torch.Tensor:
    """Resize aligned-corner NHWC FP16 data, returning the requested output rows."""
    b, hs, ws, channels = x.shape
    height, width = size
    start, stop = row_range
    if x.dtype != torch.float16 or not x.is_contiguous() or channels % _VECTOR or x.data_ptr() % _ALIGNMENT:
        msg = "DPT resize needs 16-byte aligned contiguous FP16 NHWC rows with channels divisible by 8"
        raise ValueError(msg)
    if not 0 <= start < stop <= height:
        msg = "DPT resize row range must be inside the output"
        raise ValueError(msg)
    y = x.new_empty((b, stop - start, width, channels))
    sh = (hs - 1) / (height - 1) if height > 1 else 0.0
    sw = (ws - 1) / (width - 1) if width > 1 else 0.0
    count = y.numel() // _VECTOR
    load(x.device).launch(
        "bilinear_rows", *linear(count), x, y, hs, ws, stop - start, width, channels, start, sh, sw, count
    )
    return y
