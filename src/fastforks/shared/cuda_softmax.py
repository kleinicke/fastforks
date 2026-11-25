"""Bound the temporary memory of CUDA softmax on a contiguous attention matrix."""

from __future__ import annotations

import torch


def softmax_rows_(scores: torch.Tensor) -> torch.Tensor:
    """Normalize the rows of an owned, contiguous CUDA attention matrix in its existing storage."""
    return torch.softmax(scores, dim=-1, out=scores)
