"""MLX lowering of the pixel decoder's deformable attention op (`fastforks.shared.maskformer.msda_op`).

Imported while a network is exported for MLX: registers the op with `fastforks.shared.mlx_graph` as one Metal kernel
of `fastforks.kernels.graph_mlx`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fastforks.kernels.graph_mlx import ops
from fastforks.shared.mlx_graph import OPS

from . import msda_op  # noqa: F401 - registers the op

if TYPE_CHECKING:
    from fastforks._core.mlx import mx


def _msda(value: mx.array, loc: mx.array, w: mx.array, shapes: list[int]) -> mx.array:
    return ops.msda(value, loc, w, list(zip(shapes[::2], shapes[1::2], strict=True)))


OPS[torch.ops.fastforks.msda.default] = _msda
