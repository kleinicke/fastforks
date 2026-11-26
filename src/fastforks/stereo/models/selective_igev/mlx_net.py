"""Selective-IGEV on Apple MLX, written in `mlx.core` operations (`MlxSelectiveIGEV(net, dtype, iters)`).

The maths of `portable.forward_fast`, channels last, in place of the export and graph translation the MLX runner
replayed before. The shared frame (`igev_blocks.mlx_stereo`) holds the features, the volume, the lookups, the update
loop and the upsampling; this module adds upstream's selective part: the context reweighted by channel attention
(`cam`), the spatial attention map (`sam`) and `SelectiveConvGRU` (a 1 x 1 and a 3 x 3 ConvGRU blended by the map).
Each GRU's input starts with the constant context, so its share of the z / r / q convolutions is computed once.
"""

from __future__ import annotations

from typing import Any

from fastforks._core.mlx import mx
from fastforks.shared.igev_blocks import mlx_blocks as bl
from fastforks.shared.igev_blocks.mlx_stereo import MlxIgevBase

__all__ = ["MlxSelectiveIGEV"]

PER_LEVEL = 5  # constants per GRU level: small [z | r], small q, large [z | r], large q, attention


class MlxSelectiveIGEV(MlxIgevBase):
    """`forward_fast` of `net` (a loaded Selective-IGEV `IGEVStereo`, float32 weights) in `dtype` (the name)."""

    mask_scale = 0.25  # upstream: "scale mask to balance gradients"

    def __init__(self, net: Any, dtype: str, iters: int) -> None:  # noqa: ANN401 - `IGEVStereo`
        """The base's weights; the update runs as four compiled parts (`_iteration`)."""
        super().__init__(net, dtype, iters)
        self._parts = [mx.compile(f) for f in (self._coarse, self._gates, self._large, self._fine)]
        self._step = self._iteration

    def _family(self, net: Any, w: bl.Weights) -> None:  # noqa: ANN401 - `IGEVStereo`
        self.cnet = w.context(net.cnet, ())
        self.cam = (w.conv(net.cam.fc[0]), w.conv(net.cam.fc[2]))
        self.sam = w.conv(net.sam.samconv)
        ub = net.update_block
        self.grus = []
        for g in (ub.gru04, ub.gru08, ub.gru16):
            hidden = g.small_gru.convz.out_channels
            small, small_ctx = w.gru_split(g.small_gru, hidden, hidden)
            large, large_ctx = w.gru_split(g.large_gru, hidden, hidden)
            self.grus.append((small, large, small_ctx, large_ctx))

    def _attend(self, inp: mx.array) -> tuple[mx.array, mx.array]:
        """`cam(inp) * inp` and `sam` of it."""
        fc0, fc1 = self.cam

        def fc(v: mx.array) -> mx.array:
            return bl.conv(bl.relu(bl.conv(v, fc0)), fc1)

        pooled = inp.astype(mx.float32)
        avg = pooled.mean(axis=(1, 2), keepdims=True).astype(inp.dtype)
        inp = mx.sigmoid(fc(avg) + fc(inp.max(axis=(1, 2), keepdims=True))) * inp
        maps = mx.concatenate(
            [inp.astype(mx.float32).mean(axis=-1, keepdims=True).astype(inp.dtype), inp.max(axis=-1, keepdims=True)],
            axis=-1,
        )
        return inp, mx.sigmoid(bl.conv(maps, self.sam))

    def _context(self, im: mx.array) -> tuple[list[mx.array], list[mx.array]]:
        nets, constants = [], []
        for (net, (raw,)), (_, _, small_ctx, large_ctx) in zip(bl.context(im, self.cnet), self.grus, strict=True):
            inp, att = self._attend(raw)
            nets.append(net)
            for ctx in (small_ctx, large_ctx):
                c = bl.conv(inp, ctx)
                k = c.shape[-1] // 3
                constants += [c[..., : 2 * k], c[..., 2 * k :]]
            constants.append(att)
        return nets, constants

    def _gru(self, level: int, h: mx.array, x: mx.array, constants: tuple[mx.array, ...]) -> mx.array:
        """`SelectiveConvGRU(att, h, inp, *x)`: `small * att + large * (1 - att)`."""
        small, large = self.grus[level][:2]
        sz, sq, lz, lq, att = constants[PER_LEVEL * level : PER_LEVEL * (level + 1)]
        hs = bl.gru(h, [sz, sq], x, small)
        hl = bl.gru(h, [lz, lq], x, large)
        return hl + att * (hs - hl)

    def _coarse(self, *state: mx.array) -> tuple[mx.array, ...]:
        """The 1/16 and 1/8 GRUs and the 1/4 GRUs' input: `net1, net2, x0`."""
        net0, net1, net2, disp = state[:4]
        constants, lookups = state[4 : 4 + 3 * PER_LEVEL], list(state[4 + 3 * PER_LEVEL :])
        net2 = self._gru(2, net2, bl.pool2x(net1), constants)
        x = mx.concatenate([bl.pool2x(net0), bl.interp(net2, net1.shape[1:3])], axis=-1)
        net1 = self._gru(1, net1, x, constants)
        return net1, net2, mx.concatenate([self._motion(disp, lookups), bl.interp(net1, net0.shape[1:3])], axis=-1)

    def _gates(self, h: mx.array, x: mx.array, zr_ctx: mx.array) -> mx.array:
        """The 1/4 large GRU's `[z | r]`."""
        return mx.sigmoid(bl.conv(mx.concatenate([h, x], axis=-1), self.grus[0][1][0]) + zr_ctx)

    def _large(self, h: mx.array, x: mx.array, zr: mx.array, lq: mx.array) -> mx.array:
        """The 1/4 large GRU's new state from its `[z | r]`."""
        large = self.grus[0][1]
        k = h.shape[-1]
        z, r = zr[..., :k], zr[..., k:]
        return h + z * (mx.tanh(bl.conv(mx.concatenate([r * h, x], axis=-1), large[1]) + lq) - h)

    def _fine(self, *a: mx.array) -> tuple[mx.array, mx.array]:
        """The rest of the 1/4 update (`net0`, `disp`) after the large GRU: the small GRU, the blend, the head."""
        h, x, hl, disp, sz, sq, att = a
        hs = bl.gru(h, [sz, sq], x, self.grus[0][0])
        net0 = hl + att * (hs - hl)
        return net0, disp + self._delta(net0)

    def _iteration(self, *state: mx.array) -> tuple[mx.array, ...]:
        """One update (`BasicSelectiveMultiUpdateBlock`, three GRU levels) and the disparity step, in four parts.

        Each part is compiled and evaluated before the next is queued, the last one too (so the base's loop, which
        keeps one step in flight, does not overlap two): as one compiled update, the 1/4 large GRU's 3 x 3
        convolutions were alive with the rest of the step's transients until it completed (994 x 1482 float16:
        network peak 1.7x the graph translation's).
        """
        net0, disp = state[0], state[3]
        sz, sq, lz, lq, att = state[4 : 4 + PER_LEVEL]
        net1, net2, x = self._parts[0](*state)
        mx.eval(net1, net2, x)
        zr = self._parts[1](net0, x, lz)
        mx.eval(zr)
        hl = self._parts[2](net0, x, zr, lq)
        del zr
        mx.eval(hl)
        net0, disp = self._parts[3](net0, x, hl, disp, sz, sq, att)
        mx.eval(net0, disp)
        return net0, net1, net2, disp
