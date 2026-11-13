"""The 2D RoPE of VGGT and Depth Anything 3 (same upstream module) with a table sized by the token count."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def rope_forward(self: nn.Module, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Upstream `RotaryPositionEmbedding2D.forward` with a table sized by the token count, not `positions.max()`.

    Positions are at most the token count, and a longer table has the same rows, so the result is unchanged; the
    data-dependent size made torch.export fail and cost a device sync. Both halves rotate in one pass:
    `tokens cos + swap(tokens) sin` with the y | x tables side by side, `swap` exchanging the quarters of each half
    (the sign of upstream's `(-x2, x1)` sits in the sin table). Use with `override(rope, rope_forward)`.
    """
    rope: Any = self
    d = tokens.size(-1) // 2
    cc, sc = rope._compute_frequency_components(d, tokens.size(-2) + 1, tokens.device, tokens.dtype)  # noqa: SLF001
    sc = torch.cat((-sc[:, : d // 2], sc[:, d // 2 :]), -1)
    cos, sin = (torch.cat((F.embedding(positions[..., 0], t), F.embedding(positions[..., 1], t)), -1) for t in (cc, sc))
    swapped = tokens.unflatten(-1, (2, 2, d // 2)).flip(-2).flatten(-3)
    return torch.addcmul(tokens * cos[:, None], swapped, sin[:, None])
