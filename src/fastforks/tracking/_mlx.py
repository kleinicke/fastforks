"""The MLX runner of a tracking model: the whole offline pass (`module()`) as one exported graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fastforks.shared import mlx_wrap

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.tracking._runner import Context, Runner


def port(portable: type[Runner], *, cache_mb: int = 0) -> type[Runner]:
    """`portable` with its `module()` (video `[T, 3, H, W]` and queries in, tracks and visibility out) run on MLX.

    `cache_mb`: as for `mlx_wrap.port`.
    """

    class Graph(portable):  # ty: ignore[unsupported-base]
        """The runner whose pass is one exported module."""

        def __init__(self, ctx: Context) -> None:
            """Build the PyTorch runner, then its `module()` the graph is exported from."""
            super().__init__(ctx)
            self._fwd = portable.module(self)

        def _run(self, video: torch.Tensor, queries: torch.Tensor) -> dict[str, torch.Tensor]:
            return self._fwd(video, queries)

        def __call__(
            self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            """Tracks and visibility of the queries."""
            del size
            return self._run(torch.cat([frames(t) for t in range(n)]), queries)

    Graph.__name__ = Graph.__qualname__ = f"Graph{portable.__name__}"
    return mlx_wrap.port(Graph, attrs=(), methods=("_run",), net_attr="_fwd", cache_mb=cache_mb)
