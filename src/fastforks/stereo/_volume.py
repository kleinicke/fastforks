"""Functional cost-volume builders for the MLX export.

Upstream fills a preallocated volume slice by slice (`volume[:, :, i, :, i:] = ...`); exported, that is a chain of
`slice_scatter` ops, each copying the whole volume. Here every disparity is one padded slab and the volume a single
`stack`, which an exported graph replays without the copies. Eager runs keep the in-place builders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable


def stacked_volume(
    left: torch.Tensor,
    right: torch.Tensor,
    disparities: int,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """`[B, C', D, H, W]` with slab `i` = `fn(left[..., i:], right[..., :w - i])` shifted right by `i` (zeros left of it).

    `fn` maps two `[B, C, H, W - i]` tensors to `[B, C', H, W - i]`; disparities >= width stay zero.
    """
    w = left.shape[-1]
    slabs = []
    zero = None
    for i in range(disparities):
        if i < w:
            slabs.append(F.pad(fn(left[..., i:], right[..., : w - i]), (i, 0)))
        else:
            if zero is None:
                zero = torch.zeros_like(slabs[0])
            slabs.append(zero)
    return torch.stack(slabs, dim=2)


def concat_fn(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Channel concatenation of the overlapping parts."""
    return torch.cat((left, right), dim=1)
