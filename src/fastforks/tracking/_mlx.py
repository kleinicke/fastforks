"""The MLX runner of a tracking model: the whole offline pass (`module()`) as one exported graph.

The queries enter a float16 replay as float32 (the point coordinates stay float32: `scale_xy` keeps their pixel
factors exact, `fastforks._core.mlx_cast.to_float32` marks the coordinate / logit state the networks carry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from fastforks.shared import mlx_wrap

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fastforks.tracking._runner import Context, Runner


def port(
    portable: type[Runner], *, cache_mb: int | Mapping[str, int] = 0, chunk_mb: int | Mapping[str, int] = 0
) -> type[Runner]:
    """`portable` with its `module()` (video `[T, 3, H, W]` and queries in, tracks and visibility out) run on MLX.

    `cache_mb`, `chunk_mb`: as for `mlx_wrap.port`.
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
    return cast(
        "type[Runner]",
        mlx_wrap.port(
            Graph,
            attrs=(),
            methods=("_run",),
            net_attr="_fwd",
            cache_mb=cache_mb,
            chunk_mb=chunk_mb,
            fp32_inputs=("queries",),
        ),
    )


def scale_xy(xy: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    """`xy * (sx, sy)` over the last axis with Python-float factors: a float16 MLX replay keeps them exact.

    A factor tensor would be a float16 constant there: `256 / 150` is 4e-4 off, 0.1 px at 256 px.
    """
    return torch.cat((xy[..., :1] * sx, xy[..., 1:] * sy), -1)
