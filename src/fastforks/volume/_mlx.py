"""The MLX runner of an nnU-Net family: every fold network of every part is an exported graph.

The sliding-window loop, the Gaussian merging and the resampling stay PyTorch (CPU); `Predictor` calls `net(window)`,
which for an `MlxModule` runs the window on the GPU through `fastforks.shared.mlx_graph`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks.shared import mlx_wrap
from fastforks.volume._nnunet import Predictor, _labels_dtype

if TYPE_CHECKING:
    from fastforks.volume._runner import Context, Runner


class MlxPredictor(Predictor):
    """`Predictor` whose argmax over the heads runs on the GPU (the CPU argmax of 100+ heads is a second per part)."""

    def decide(self, logits: torch.Tensor, c0: int = 0, seg: torch.Tensor | None = None) -> torch.Tensor:
        """`Predictor.decide`; without regions the argmax runs on MLX."""
        if self.plan.regions is not None:
            return super().decide(logits, c0, seg)
        from fastforks._core.mlx import mx, to_mx  # noqa: PLC0415 - MLX only

        dtype = _labels_dtype(self.plan.n_labels)
        idx = mx.argmax(to_mx(logits), axis=0).astype(mx.uint8 if dtype == torch.uint8 else mx.int16)
        mx.eval(idx)
        return torch.from_numpy(np.array(idx))


def port(portable: type[Runner]) -> type[Runner]:
    """`portable` with its fold networks on MLX."""

    class Graph(portable):  # ty: ignore[unsupported-base]
        """The runner whose fold networks are `MlxModule`s."""

        def __init__(self, ctx: Context) -> None:
            """Build the PyTorch runner, then replace each network by its MLX graph."""
            from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - MLX only (imports mlx)

            super().__init__(ctx)
            dtype = mlx_wrap.run_dtype(ctx)
            kernels = ctx.mode == "kernels"
            self.parts = [
                MlxPredictor(**{f.name: getattr(part, f.name) for f in dataclasses.fields(part) if f.init})
                for part in self.parts
            ]
            for part in self.parts:
                part.nets = [  # ty: ignore[invalid-assignment]
                    MlxModule(net, dtype, kernels=kernels, release=True) for net in part.nets
                ]

    Graph.__name__ = Graph.__qualname__ = f"Graph{portable.__name__}"
    return mlx_wrap.port(Graph, attrs=())
