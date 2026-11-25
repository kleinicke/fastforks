"""IGEV-Stereo on Apple MLX, written in `mlx.core` operations (`MlxIGEV(net, dtype, iters)`): pair -> disparity.

The maths of `IGEVStereo.forward_fast`, channels last, in place of the export and graph translation the MLX runner
replayed before. The shared frame (`igev_blocks.mlx_stereo`) holds the features, the volume, the lookups, the update
loop and the upsampling; this module adds the context encoder with its `context_zqr_convs` and the three `ConvGRU`s.
"""

from __future__ import annotations

from typing import Any

from fastforks._core.mlx import mx
from fastforks.shared.igev_blocks import mlx_blocks as bl
from fastforks.shared.igev_blocks.mlx_stereo import MlxIgevBase

__all__ = ["MlxIGEV"]


class MlxIGEV(MlxIgevBase):
    """`IGEVStereo.forward_fast` of `net` (a loaded `IGEVStereo`, float32 weights) in `dtype` (the name)."""

    def _family(self, net: Any, w: bl.Weights) -> None:  # noqa: ANN401 - `IGEVStereo`
        self.cnet = w.context(net.cnet, net.context_zqr_convs)
        ub = net.update_block
        self.grus = [w.gru(ub.gru04), w.gru(ub.gru08), w.gru(ub.gru16)]

    def _context(self, im: mx.array) -> tuple[list[mx.array], list[mx.array]]:
        context = bl.context(im, self.cnet)
        return [n for n, _ in context], [c for _, cs in context for c in cs]

    def _iteration(self, *state: mx.array) -> tuple[mx.array, ...]:
        """One update (`BasicMultiUpdateBlock`, three GRU levels) and the disparity step."""
        net0, net1, net2, disp = state[:4]
        ctx, lookups = state[4:10], list(state[10:])
        net2 = bl.gru(net2, list(ctx[4:6]), bl.pool2x(net1), self.grus[2])
        x = mx.concatenate([bl.pool2x(net0), bl.interp(net2, net1.shape[1:3])], axis=-1)
        net1 = bl.gru(net1, list(ctx[2:4]), x, self.grus[1])
        x = mx.concatenate([self._motion(disp, lookups), bl.interp(net1, net0.shape[1:3])], axis=-1)
        net0 = bl.gru(net0, list(ctx[0:2]), x, self.grus[0])
        return net0, net1, net2, disp + self._delta(net0)
