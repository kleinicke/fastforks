"""The 2D RoPE of VGGT and Depth Anything 3 (same upstream module) with a table sized by the token count."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def rope_forward(self: nn.Module, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Upstream `RotaryPositionEmbedding2D.forward` with a table sized by the token count, not `positions.max()`.

    Positions are at most the token count, and a longer table has the same rows, so the result is unchanged; the
    data-dependent size made torch.export fail and cost a device sync. Use with `override(rope, rope_forward)`.
    """
    rope: Any = self
    d = tokens.size(-1) // 2
    cc, sc = rope._compute_frequency_components(d, tokens.size(-2) + 1, tokens.device, tokens.dtype)  # noqa: SLF001
    v, h = tokens.chunk(2, dim=-1)
    return torch.cat(
        (rope._apply_1d_rope(v, positions[..., 0], cc, sc), rope._apply_1d_rope(h, positions[..., 1], cc, sc)),  # noqa: SLF001
        dim=-1,
    )
