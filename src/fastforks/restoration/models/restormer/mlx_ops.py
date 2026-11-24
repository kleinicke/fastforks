"""MLX lowerings of Restormer's export ops (`model.token_gram_op`, `model.dw_gate_op`).

Imported by the MLX port's runner: registers the ops with `fastforks.shared.mlx_graph`, each as one Metal kernel of
`fastforks.kernels.graph_mlx` on the channels-last activation (the convolutions' own layout).
"""

from __future__ import annotations

import torch

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared.mlx_graph import CL_OPS, OPS

from . import model  # noqa: F401 - registers the ops


def _token_gram(t: mx.array, c: int) -> tuple[mx.array, mx.array]:
    return ops.token_gram(t, c)


def _dw_gate(x: mx.array, w: mx.array) -> mx.array:
    """NCHW views in and out; the data is channels-last."""
    y = ops.dw_gate(mx.transpose(x, (0, 2, 3, 1)), mx.transpose(w[:, 0], (1, 2, 0)))
    return mx.transpose(y, (0, 3, 1, 2))


OPS[torch.ops.fastforks.restormer_token_gram.default] = _token_gram
OPS[torch.ops.fastforks.restormer_dw_gate.default] = CL_OPS[torch.ops.fastforks.restormer_dw_gate.default] = _dw_gate
