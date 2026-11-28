"""TEED on Apple MLX: `fastforks.shared.teed.mlx_net` (hand-written, shared with AnyLine) with its weights cached.

The network's weights are converted once in the run dtype and kept in the conversion cache (`LAYOUT`, `SOURCES`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastforks._core.mlx import mx
from fastforks.shared.teed import mlx_net as shared

if TYPE_CHECKING:
    from fastforks.shared.teed.net.ted import TED

LAYOUT = "teed-v1"
SOURCES = (Path(__file__), Path(shared.__file__))


def network(net: TED, dtype: str) -> shared.MlxTED:
    """The MLX network of a TED (float32, or the meta skeleton) in `dtype`."""
    return shared.MlxTED(net, getattr(mx, dtype))
