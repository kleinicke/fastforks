"""Apple MLX ports of a family's PyTorch runner: its network runs through `fastforks.shared.mlx_graph`.

`port(Portable)` is the family's `portable_mlx` / `kernels_mlx` runner: the portable runner is built on the CPU in float32
(its pre- and post-processing stays PyTorch on the CPU), then each network attribute (`attrs`, default `net`) is replaced
by an `MlxModule`, which exports that module once per input signature, converts its parameters to the run dtype and
replays the graph with `mx.compile` on the GPU. Kernels mode (`ctx.mode == "kernels"`) swaps in our Metal kernels
(`fastforks.kernels.graph_mlx`) for depthwise convolution and global pooling.
"""

from __future__ import annotations

import contextvars
import dataclasses
from typing import TYPE_CHECKING

from fastforks.shared.mlx_graph import MlxModule

if TYPE_CHECKING:
    from fastforks._core.runner import Context, Runner

__all__ = ["port", "run_dtype"]

_RUN_DTYPE: contextvars.ContextVar[str | None] = contextvars.ContextVar("mlx_run_dtype", default=None)


def run_dtype(ctx: Context) -> str:
    """The dtype the network runs in: while an MLX runner builds its float32 PyTorch twin, the MLX run dtype.

    A runner that rescales weights for float16 (to keep activations finite) asks this instead of `ctx.dtype`.
    """
    return _RUN_DTYPE.get() or ctx.dtype


def port(portable: type[Runner], *, attrs: tuple[str, ...] = ("net",)) -> type[Runner]:
    """The MLX runner of `portable`: same call, the module(s) named by `attrs` run on MLX in `ctx.dtype`."""

    class Mlx(portable):  # ty: ignore[unsupported-base]
        __doc__ = f"Apple MLX: `{portable.__name__}` with its network run by the exported-graph translation."

        def __init__(self, ctx: Context) -> None:
            """Build the PyTorch runner in float32 on the CPU, then hand its network(s) to MLX."""
            token = _RUN_DTYPE.set(ctx.dtype)
            try:
                super().__init__(dataclasses.replace(ctx, dtype="float32"))
            finally:
                _RUN_DTYPE.reset(token)
            for name in attrs:
                setattr(self, name, MlxModule(getattr(self, name), ctx.dtype, kernels=ctx.mode == "kernels"))

    Mlx.__name__ = Mlx.__qualname__ = f"Mlx{portable.__name__}"
    return Mlx
