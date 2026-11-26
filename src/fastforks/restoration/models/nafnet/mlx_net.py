"""NAFNet on Apple MLX, written in `mlx.core` operations (`MlxNAFNet(net, dtype)`): image -> restored image.

`NAFNet.forward` of the vendored upstream network as `Portable` prepares it (each block's norm affine and `beta` /
`gamma` folded into its 1 x 1 convs, the residual stream scaled by 1 / 1024 and the channel attention by 1 / 64, the
TLC windows of `NAFNetLocal`), channels last. Written out instead of replayed from an exported graph: the export,
translation and segment tuning were ~2-3 s of a 2.2-3.3 s first call at 512 x 512.

- each block is one `mx.compile`d function per level (its weights arguments, so the 36 blocks of a level share a
  trace): the channel LayerNorm (no affine; written out below 256 channels), the 1 x 1 convs as matrix products,
  the depthwise 3 x 3 the shared depthwise kernel, SimpleGate and the channel attention fused around them;
- the TLC channel attention (`nafnet-gopro`): box means of the gated map over the level's window, from float32
  running sums along each axis (upstream's integral image, one axis at a time), projected by the attention's 1 x 1
  conv before the replicate padding (the same 1 x 1 at every pixel: `Kernels._scale_features`);
- the 2 x 2 stride-2 downsampling convs are a space-to-depth reshape and one GEMM, the upsamplers a GEMM and a pixel
  shuffle; the intro / ending 3 x 3 convs `mx.conv2d`.

Each block is evaluated as it is scheduled, one in flight (`_step`): MLX frees a command buffer's transients only
when it completes, and one lazy graph of the whole U-Net holds far more than its live set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops

if TYPE_CHECKING:
    from collections.abc import Callable

    Block = tuple[Callable[..., mx.array], Callable[..., mx.array], Callable[..., mx.array]]  # whole, mix, ffn

__all__ = ["MlxNAFNet"]

ONE = (1, 1)


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().float().numpy()


def _shuffle(x: mx.array) -> mx.array:
    """`PixelShuffle(2)` of an NHWC map (input channel `c * 4 + i * 2 + j`)."""
    n, h, w, c = x.shape
    x = x.reshape(n, h, w, c // 4, 2, 2).transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(n, h * 2, w * 2, c // 4)


def _gate(x: mx.array) -> mx.array:
    c = x.shape[-1] // 2
    return x[..., :c] * x[..., c:]


def _layer_norm(x: mx.array, eps: float) -> mx.array:
    """The channel LayerNorm without affine, float32 statistics.

    Up to `ops.ROW_LN_MAX` channels `ops.row_layer_norm` (one simdgroup per row, as the graph translation): written
    out in MLX ops, the float32 copy and the centred map were full-size buffers even when compiled (512 x 512 x 64
    float16: 194 MB above the live set for a 32 MB output); `mx.fast.layer_norm` is 4-6x slower at these widths.
    """
    if x.shape[-1] > ops.ROW_LN_MAX:
        return mx.fast.layer_norm(x, None, None, eps)
    return ops.row_layer_norm(x, None, None, eps)


def _box_mean(x: mx.array, k1: int, k2: int) -> mx.array:
    """Means of every `k1` x `k2` window of an NHWC map (valid positions), from float32 running sums."""
    s = mx.cumsum(x.astype(mx.float32), axis=1)
    s = s[:, k1 - 1 :] - mx.pad(s[:, : s.shape[1] - k1], [(0, 0), (1, 0), (0, 0), (0, 0)])
    s = mx.cumsum(s, axis=2)
    s = s[:, :, k2 - 1 :] - mx.pad(s[:, :, : s.shape[2] - k2], [(0, 0), (0, 0), (1, 0), (0, 0)])
    return s * (1 / (k1 * k2))


def _attention_scale(g: mx.array, ws: mx.array, bs: mx.array, window: int) -> mx.array:
    """The simplified channel attention's scores of the gated map `g` (global, or the TLC window's local means)."""
    h, w = g.shape[1], g.shape[2]
    if window >= h and window >= w:
        mean = mx.mean(g.astype(mx.float32), axis=(1, 2), keepdims=True).astype(g.dtype)
        return mx.addmm(bs, mean.reshape(g.shape[0], -1), ws)[:, None, None]
    k1, k2 = min(h, window), min(w, window)
    scores = mx.addmm(bs, _box_mean(g, k1, k2).astype(g.dtype), ws)
    pad = [(0, 0), ((k1 - 1) // 2, k1 // 2), ((k2 - 1) // 2, k2 // 2), (0, 0)]
    return mx.pad(scores, pad, mode="edge")


def _block_fn(window: int, eps: float, depthwise: bool) -> Block:  # noqa: FBT001
    """`NAFBlock.forward` (folded) on NHWC maps, compiled: the whole block, its mixing half and its FFN half.

    `window`: the TLC window (0: global attention).
    """

    def mix(inp: mx.array, *w: mx.array) -> mx.array:
        w1, b1, wd, bd, ws, bs, w3, b3 = w[:8]
        x = mx.addmm(b1, _layer_norm(inp, eps), w1)
        x = ops.depthwise(x, wd, bd, ONE, ONE, ONE) if depthwise else mx.conv2d(x, wd, padding=1, groups=wd.shape[0])
        x = x if depthwise else x + bd
        g = _gate(x)
        g = g * _attention_scale(g, ws, bs, window or max(g.shape[1], g.shape[2]))
        return inp + mx.addmm(b3, g, w3)

    def ffn(y: mx.array, *w: mx.array) -> mx.array:
        w4, b4, w5, b5 = w[8:]
        return y + mx.addmm(b5, _gate(mx.addmm(b4, _layer_norm(y, eps), w4)), w5)

    def block(inp: mx.array, *w: mx.array) -> mx.array:
        return ffn(mix(inp, *w), *w)

    return mx.compile(block), mx.compile(mix), mx.compile(ffn)


_PENDING: list[mx.array] = []  # the step in flight (`_step`)


SYNC_BYTES = 8 << 20
"""A step whose output is at least this large also waits for itself (`_step`)."""


def _step(x: mx.array) -> mx.array:
    """Schedule `x` and wait for the step before it (pipeline depth 1: the GPU stays busy, the transients bounded).

    A step of at least `SYNC_BYTES` (the full- and half-resolution levels) waits for itself
    too: its transients are many times its output, and two such steps in flight held both.
    """
    mx.async_eval(x)
    if _PENDING:
        mx.eval(_PENDING.pop())
    if x.nbytes >= SYNC_BYTES:
        mx.eval(x)
    else:
        _PENDING.append(x)
    return x


class MlxNAFNet:
    """The network: `[B, 3, H, W]` float32 -> the restored `[B, 3, H, W]` float32, unclamped (`NAFNet.forward`)."""

    def __init__(self, net: Any, dtype: str) -> None:  # noqa: ANN401 - `Portable.net` (folded, scaled)
        """The weights of `net` (float32, upstream layout, prepared by `Portable`) in `dtype`."""
        self.dtype = getattr(mx, dtype)
        a = lambda v: mx.array(np.ascontiguousarray(v, np.float32)).astype(self.dtype)  # noqa: E731
        conv3 = lambda m: (a(_np(m.weight).transpose(0, 2, 3, 1)), a(_np(m.bias)))  # noqa: E731 - OHWI
        self.intro, self.ending = conv3(net.intro), conv3(net.ending)
        self.downs = []
        for m in net.downs:  # [O, C, 2, 2] -> [(i, j, c), O] of the space-to-depth rows
            self.downs.append((a(_np(m.weight).transpose(2, 3, 1, 0).reshape(-1, m.out_channels)), a(_np(m.bias))))
        self.ups = [a(_np(m[0].weight)[:, :, 0, 0].T) for m in net.ups]
        self.padder = net.padder_size
        n = len(net.encoders)
        levels = [*enumerate(net.encoders), (n, net.middle_blks)]
        levels += [(n - 1 - i, s) for i, s in enumerate(net.decoders)]
        self.eps = next(iter(net.middle_blks)).norm1.eps
        self._fns: dict[int, Block] = {}
        self.stages = [[self._block(m) for m in stage] for _, stage in levels]

    def _block(self, m: Any) -> tuple[Block, list[mx.array]]:  # noqa: ANN401 - a folded `NAFBlock`
        a = lambda v: mx.array(np.ascontiguousarray(v, np.float32)).astype(self.dtype)  # noqa: E731
        pw = lambda conv: [a(_np(conv.weight)[:, :, 0, 0].T), a(_np(conv.bias))]  # noqa: E731 - [C in, C out]
        if m.norm1.eps != self.eps or m.norm2.eps != self.eps:
            msg = "NAFNet MLX: the blocks' norm epsilons differ"
            raise NotImplementedError(msg)
        pool = m.sca[0]
        window = getattr(pool, "window", 0)  # `_LocalMean` (TLC) or the global `AdaptiveAvgPool2d(1)`
        dw = m.conv2
        depthwise = dw.out_channels % 4 == 0
        wd = _np(dw.weight)[:, 0].transpose(1, 2, 0) if depthwise else _np(dw.weight).transpose(0, 2, 3, 1)
        weights = [*pw(m.conv1), a(wd), a(_np(dw.bias)), *pw(m.sca[1]), *pw(m.conv3), *pw(m.conv4), *pw(m.conv5)]
        key = window * 2 + depthwise
        if key not in self._fns:
            self._fns[key] = _block_fn(window, self.eps, depthwise)
        mx.eval(weights)
        return self._fns[key], weights

    def __call__(self, image: mx.array) -> mx.array:
        """`NAFNet.forward`: pad to a multiple of the levels' stride, the U-Net, the residual, crop."""
        try:
            return self._forward(image)
        finally:
            _PENDING.clear()

    def _forward(self, image: mx.array) -> mx.array:
        _, _, h, w = image.shape
        p = self.padder
        inp = mx.transpose(image, (0, 2, 3, 1)).astype(self.dtype)
        inp = mx.pad(inp, [(0, 0), (0, -h % p), (0, -w % p), (0, 0)])
        x = mx.conv2d(inp, self.intro[0], padding=1) + self.intro[1]
        stages = iter(self.stages)
        skips = []
        for wd, bd in self.downs:
            x = self._stage(next(stages), x)
            skips.append(x)
            b, hh, ww, c = x.shape
            x = x.reshape(b, hh // 2, 2, ww // 2, 2, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, hh // 2, ww // 2, 4 * c)
            x = mx.addmm(bd, x, wd)
        x = self._stage(next(stages), x)
        for up in self.ups:
            x = self._stage(next(stages), _shuffle(x @ up) + skips.pop())
        out = mx.conv2d(x, self.ending[0], padding=1) + self.ending[1] + inp
        return mx.transpose(out[:, :h, :w], (0, 3, 1, 2)).astype(mx.float32)

    @staticmethod
    def _stage(blocks: list[tuple[Block, list[mx.array]]], x: mx.array) -> mx.array:
        """The blocks of a level, each one step, or two (its halves) on a map of at least `SYNC_BYTES`.

        As one compiled block at full resolution its transients (about 30x its input) were alive at once.
        """
        for (whole, mix, ffn), weights in blocks:
            large = x.nbytes >= SYNC_BYTES
            x = _step(ffn(_step(mix(x, *weights)), *weights)) if large else _step(whole(x, *weights))
        return x
