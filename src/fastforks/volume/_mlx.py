"""The MLX runner of an nnU-Net family: every fold network of every part is an exported graph.

The sliding-window loop, the Gaussian merging and the resampling stay PyTorch (CPU); `Predictor` calls `net(window)`,
which for an `MlxModule` runs the window on the GPU through `fastforks.shared.mlx_graph`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastforks.shared import mlx_wrap

if TYPE_CHECKING:
    from fastforks.volume._runner import Context, Runner


def port(portable: type[Runner]) -> type[Runner]:
    """`portable` with its fold networks on MLX."""

    class Graph(portable):  # ty: ignore[unsupported-base]
        """The runner whose fold networks are `MlxModule`s."""

        def __init__(self, ctx: Context) -> None:
            """Build the PyTorch runner, then replace each network by its MLX graph."""
            from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - MLX only (imports mlx)

            super().__init__(ctx)
            dtype = mlx_wrap.run_dtype(ctx)
            for part in self.parts:
                part.nets = [MlxModule(net, dtype, release=True) for net in part.nets]  # ty: ignore[invalid-assignment]

    Graph.__name__ = Graph.__qualname__ = f"Graph{portable.__name__}"
    return mlx_wrap.port(Graph, attrs=())
