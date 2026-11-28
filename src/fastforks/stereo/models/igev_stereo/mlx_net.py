"""IGEV-Stereo on Apple MLX, written in `mlx.core` operations (`MlxIGEV(net, dtype, iters)`): pair -> disparity.

The maths of `IGEVStereo.forward_fast`, channels last, in place of the export and graph translation the MLX runner
replayed before. The shared frame (`igev_blocks.mlx_stereo`) holds the features, the volume, the lookups, the update
loop and the upsampling; its `MlxIgevUpdate` adds the context encoder with its `context_zqr_convs` and the three
`ConvGRU`s.
"""

from __future__ import annotations

from fastforks.shared.igev_blocks.mlx_stereo import MlxIgevUpdate

__all__ = ["MlxIGEV"]


class MlxIGEV(MlxIgevUpdate):
    """`IGEVStereo.forward_fast` of `net` (a loaded `IGEVStereo`, float32 weights) in `dtype` (the name)."""
