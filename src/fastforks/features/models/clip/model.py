"""CLIP / OpenCLIP: timm's network walked by `fastforks.shared.libnet.features` (MLX: `features.MlxPortable`)."""

from fastforks.shared import mlx_wrap
from fastforks.shared.libnet import features

RUNNERS = {
    "portable": features.Portable,
    "kernels": features.Kernels,
    "portable_mlx": features.MlxPortable,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(features.Portable, attrs=(), methods=("_forward",))}
