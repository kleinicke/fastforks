"""`F.grid_sample` without cuDNN's batch limit.

PyTorch sends 4-D bilinear / zeros / `align_corners=True` samples to cuDNN's grid sampler, which raises
CUDNN_STATUS_NOT_SUPPORTED for more than `65535 * max(1, 1024 // (roundup4(C) * roundup4(W_out)))` batch rows when the
output has one row (`H_out == 1`, the `[N, C, 1, W]` correlation / cost lookups of the stereo nets, `N` = pixels; fit
exact at N_max and N_max + 1 for six shapes, fp32 and fp16, cuDNN 9.24; 65535 is CUDA's grid y limit). A 4K RAFT
lookup has ~500k rows. `grid_sample` runs those calls on PyTorch's native kernel (as fast as cuDNN at these sizes)
and every other call through `F.grid_sample` unchanged, so the result at sizes cuDNN takes is bit-identical.
`torch.compile` needs the switch too (its fake-tensor trace of `F.grid_sample` fails on the same batches; the size
check is a shape guard); exported graphs keep `F.grid_sample` (ONNX / TensorRT GridSample has no such limit, and a
check on a dynamic dimension would constrain it).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_GRID_Y_MAX = 65535
_NCHW = 4


def cudnn_max_rows(channels: int, h_out: int, w_out: int) -> int:
    """Largest batch cuDNN's grid sampler takes (measured fit for `h_out == 1`, else the minimum over all shapes)."""
    if h_out != 1:
        return _GRID_Y_MAX
    return _GRID_Y_MAX * max(1, 1024 // (-(-channels // 4) * 4 * (-(-w_out // 4) * 4)))


def grid_sample(
    input: torch.Tensor,  # noqa: A002 (the name of F.grid_sample's argument)
    grid: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool | None = None,
) -> torch.Tensor:
    """`F.grid_sample` (options by keyword), on the native kernel where cuDNN would refuse the batch size."""
    if (
        align_corners
        and mode == "bilinear"
        and padding_mode == "zeros"
        and input.is_cuda
        and input.dim() == _NCHW
        and not (torch.compiler.is_exporting() or torch.jit.is_tracing() or torch.onnx.is_in_onnx_export())
        and input.shape[0] > cudnn_max_rows(input.shape[1], grid.shape[1], grid.shape[2])
    ):
        # the direct call skips autocast's fp32 cast of grid_sample; do it here
        if torch.is_autocast_enabled(input.device.type):
            input, grid = input.float(), grid.float()  # noqa: A001
        return torch.grid_sampler_2d(
            input, grid.to(input.dtype), interpolation_mode=0, padding_mode=0, align_corners=True
        )  # bilinear, zeros
    return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
