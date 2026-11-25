"""Bound the temporary memory of CUDA softmax on a contiguous attention matrix."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_ALIGNMENT_BYTES = 16


def softmax_rows_(scores: torch.Tensor, group_bytes: int) -> torch.Tensor:
    """Normalize owned CUDA scores in place in aligned groups of complete rows."""
    row_bytes = scores.shape[-1] * scores.element_size()
    multiple = _ALIGNMENT_BYTES // math.gcd(_ALIGNMENT_BYTES, row_bytes)
    rows = max(multiple, group_bytes // (row_bytes * multiple) * multiple)
    for group in scores.flatten(0, -2).split(rows):
        group.copy_(group.softmax(dim=-1))
    return scores
