"""MLX lowerings of HITNet's cost ops (`net.hitnet.init_cost_op`, `tile_sad_op`, `refine_sad_op`).

Imported by the MLX port's runner: registers the ops with `fastforks.shared.mlx_graph`, each as one Metal kernel of
`fastforks.kernels.graph_mlx` on the channels-last features (the convolutions' own layout).
"""

from __future__ import annotations

import torch

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared.mlx_graph import OPS

from .net import hitnet  # noqa: F401 - registers the ops


def _nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def _init_cost(lt: mx.array, rt: mx.array, max_disp: int) -> tuple[mx.array, mx.array]:
    return ops.hitnet_init_cost(_nhwc(lt), _nhwc(rt), max_disp)


def _tile_sad(hyp: mx.array, fl: mx.array, fr: mx.array) -> mx.array:
    return mx.transpose(ops.hitnet_tile_sad(hyp, _nhwc(fl), _nhwc(fr)), (0, 3, 1, 2))


def _refine_sad(d: mx.array, fl: mx.array, fr: mx.array) -> mx.array:
    return mx.transpose(ops.hitnet_refine_sad(d, _nhwc(fl), _nhwc(fr)), (0, 3, 1, 2))


OPS[torch.ops.fastforks.hitnet_init_cost.default] = _init_cost
OPS[torch.ops.fastforks.hitnet_tile_sad.default] = _tile_sad
OPS[torch.ops.fastforks.hitnet_refine_sad.default] = _refine_sad
