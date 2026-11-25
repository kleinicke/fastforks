"""Exact greedy box suppression in score order with bounded device scratch space."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library

SIGNATURES = {"small": "ppiff", "masks": "ppiiiiff", "select_rows": "pppiii"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_ROWS = 256


def indices(boxes: torch.Tensor, iou: float, eps: float = 0.0) -> torch.Tensor:
    """Keep the first box from each greedy overlap comparison, returning the surviving row indices."""
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.dtype != torch.float32 or not boxes.is_cuda:  # noqa: PLR2004
        msg = "Box suppression expects sorted float32 CUDA boxes in XYXY format"
        raise ValueError(msg)
    n = boxes.shape[0]
    if not n:
        return boxes.new_empty(0, dtype=torch.long)
    x = boxes.contiguous()
    # A contiguous slice can still start at a 4-byte-aligned offset, short of a CUDA float4.
    if x.data_ptr() % 16:
        x = x.clone()
    keep = boxes.new_empty(n, dtype=torch.bool)
    lib = load(x.device)
    if n <= _ROWS:
        lib.launch("small", (1, 1, 1), (_ROWS, 1, 1), x, keep, n, iou, eps)
    else:
        words = (n + 31) // 32
        gone = boxes.new_zeros(words, dtype=torch.int32)
        mask = boxes.new_empty((_ROWS, words), dtype=torch.int32)
        for first in range(0, n, _ROWS):
            rows = min(n - first, _ROWS)
            lib.launch("masks", (words, (rows + 7) // 8, 1), (32, 8, 1), x, mask, n, first, rows, words, iou, eps)
            lib.launch("select_rows", (1, 1, 1), (_ROWS, 1, 1), mask, gone, keep, first, rows, words)
    return torch.where(keep)[0]
