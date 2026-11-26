"""RAFT-Stereo on Apple MLX, written in `mlx.core` operations (`MlxRAFTStereo`): a padded pair -> disparity.

The maths of `net/raft_stereo.py` (`forward` in test mode, the released architecture), channels last. Written out
instead of replayed from an exported graph: the export was most of a 7-10 s first call.

- the context encoder's batch norms are folded (`fold_batchnorm`); the feature encoder runs one image at a time, its
  instance norms from float32 statistics (one shared Metal kernel pass); a convolution before one drops its bias.
- correlation: the float32 all-pairs volume is one matrix product per row; the pyramid lookup is the shared
  `linear_sample` kernel (zeros outside, as the gather lookup).
- the update step is one compiled function: each GRU's z and r gates are one convolution whose bias is added into
  the context once, the motion encoder's last conv has 128 outputs (two zero filters, the flow added into their
  place) and the flow head only its x output. The upsampling mask is computed once, after the last step.
- kernels mode: the stride-1 3x3 convolutions are the `conv_mlx` Winograd kernel (bias and ReLU fused).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["MlxRAFTStereo"]

RADIUS, LEVELS = 4, 4  # correlation lookup (the released architecture)
MOTION = 126  # the motion encoder's own outputs; the flow takes the channels after them
ONE = (1, 1)
WINO_SMALL_HW = 16 * 16  # at most this many pixels: the F(2,3) tile (as `shared.mlx_graph`)


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().float().numpy()


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _instance_norm(x: mx.array, add: mx.array | None = None) -> mx.array:
    """`relu(instance_norm(x) [+ add])` of one NHWC image: float32 statistics, one multiply-add in the map's dtype."""
    mean, var = ops.channel_stats(x.reshape(-1, x.shape[-1]))
    scale = mx.rsqrt(var + 1e-5)
    y = x * scale.astype(x.dtype) - (mean * scale).astype(x.dtype)
    return _relu(y if add is None else y + add)


def _interp(x: mx.array, size: Sequence[int]) -> mx.array:
    """Bilinear, align_corners=True, NHWC."""
    h, w = x.shape[1:3]
    ratios = ((h - 1) / (size[0] - 1), (w - 1) / (size[1] - 1))
    return ops.bilinear_nhwc(x, (size[0], size[1]), ratios, align=True)


def _pool(x: mx.array) -> mx.array:
    """`avg_pool2d(x, 3, stride=2, padding=1)` (the padding counts)."""
    c = x.shape[-1]
    return ops.depthwise(x, mx.full((3, 3, c), 1 / 9, x.dtype), mx.zeros((c,), x.dtype), (2, 2), ONE, ONE)


class _Conv:
    """A `Conv2d` on NHWC maps: `mx.conv2d`, or the Winograd kernel for a stride-1 3x3 in kernels mode."""

    def __init__(self, w: np.ndarray, b: np.ndarray | None, conv: Any, dtype: mx.Dtype, *, kernels: bool) -> None:  # noqa: ANN401
        self.w = mx.array(np.ascontiguousarray(w)).astype(dtype)  # [O, kh, kw, I]
        self.b = None if b is None else mx.array(b).astype(dtype)
        self.stride, self.padding = conv.stride[0], conv.padding[0]
        self.wino = kernels and w.shape[1:3] == (3, 3) and self.stride == 1 and self.padding == 1
        self.u: dict[int, mx.array] = {}  # Winograd-domain weights per tile

    def __call__(self, x: mx.array, *, relu: bool = False, bias: bool = True) -> mx.array:
        b = self.b if bias else None
        if self.wino:
            from fastforks.kernels.conv_mlx import ops as conv_ops  # noqa: PLC0415 - kernels mode only

            if conv_ops.supported(tuple(x.shape), self.w.shape[0]):
                m = 2 if x.shape[1] * x.shape[2] <= WINO_SMALL_HW else 4
                if m not in self.u:
                    self.u[m] = conv_ops.weights(self.w, m, x.dtype)
                return conv_ops.conv3x3(x, self.u[m], b, m=m, act=int(relu))
        y = mx.conv2d(x, self.w, stride=self.stride, padding=self.padding)
        y = y if b is None else y + b
        return _relu(y) if relu else y


class MlxRAFTStereo:
    """The network `net` (float32 weights, batch norms folded) in `dtype` (the name)."""

    def __init__(self, net: Any, dtype: str, iters: int, *, kernels: bool) -> None:  # noqa: ANN401 - `RAFTStereo`
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        self.iters = iters
        self.kernels = kernels
        c, f, ub = net.cnet, net.fnet, net.update_block
        self.cnet = {n: self._layer(getattr(c, n)) for n in ("layer1", "layer2", "layer3", "layer4", "layer5")}
        self.cnet_stem = self._conv(c.conv1)
        self.heads = [
            [(self._block(s[0]), self._conv(s[1])) for s in c.outputs08],
            [(self._block(s[0]), self._conv(s[1])) for s in c.outputs16],
            [(None, self._conv(s)) for s in c.outputs32],
        ]
        self.context = [self._conv(m) for m in net.context_zqr_convs]
        self.fnet = {n: self._layer(getattr(f, n)) for n in ("layer1", "layer2", "layer3")}
        self.fnet_stem = self._conv(f.conv1)
        self.fnet_out = (self._a(_np(f.conv2.weight)[:, :, 0, 0].T), self._a(_np(f.conv2.bias)))
        self.grus = []
        for g in (ub.gru08, ub.gru16, ub.gru32):
            zr = np.concatenate([_np(g.convz.weight), _np(g.convr.weight)]).transpose(0, 2, 3, 1)
            bias = np.concatenate([_np(g.convz.bias), _np(g.convr.bias), _np(g.convq.bias)])
            self.grus.append(
                (_Conv(zr, None, g.convz, self.dtype, kernels=kernels), self._conv(g.convq), self._a(bias))
            )
        e = ub.encoder
        self.convc1 = (self._a(_np(e.convc1.weight)[:, :, 0, 0].T), self._a(_np(e.convc1.bias)))
        self.convc2, self.convf2 = self._conv(e.convc2), self._conv(e.convf2)
        self.convf1 = (self._a(_np(e.convf1.weight).transpose(0, 2, 3, 1)[..., :1]), self._a(_np(e.convf1.bias)))
        w, b = _np(e.conv.weight).transpose(0, 2, 3, 1), _np(e.conv.bias)
        pad = w.shape[3] - w.shape[0]
        self.motion = _Conv(
            np.pad(w, [(0, pad), (0, 0), (0, 0), (0, 0)]), np.pad(b, (0, pad)), e.conv, self.dtype, kernels=kernels
        )
        self.flow_slot = self._a(np.eye(w.shape[3], dtype=np.float32)[MOTION])
        self.flow1 = self._conv(ub.flow_head.conv1)
        w2 = ub.flow_head.conv2
        self.flow2 = (self._a(_np(w2.weight).transpose(0, 2, 3, 1)[:1]), self._a(_np(w2.bias)[:1]))  # the x flow
        self.mask1 = self._conv(ub.mask[0])
        self.mask2 = (self._a(_np(ub.mask[2].weight)[:, :, 0, 0].T), self._a(_np(ub.mask[2].bias)))
        self.taps = mx.arange(-RADIUS, RADIUS + 1, dtype=mx.float32)[None]
        self._parts = [mx.compile(f) for f in (self._coarse, self._motion, self._gates0, self._hidden0, self._head)]
        self._step = self._iteration
        self._compiled: dict[tuple[int, str, int], Any] = {}  # `_trunk`'s blocks, two parts each

    def _a(self, a: np.ndarray) -> mx.array:
        return mx.array(np.ascontiguousarray(a)).astype(self.dtype)

    def _conv(self, conv: Any) -> _Conv:  # noqa: ANN401 - a `Conv2d`
        bias = None if conv.bias is None else _np(conv.bias)
        return _Conv(_np(conv.weight).transpose(0, 2, 3, 1), bias, conv, self.dtype, kernels=self.kernels)

    # encoders ---------------------------------------------------------------------------------------------------------

    def _block(self, m: Any) -> tuple[_Conv, _Conv, _Conv | None]:  # noqa: ANN401 - `ResidualBlock`
        return self._conv(m.conv1), self._conv(m.conv2), None if m.downsample is None else self._conv(m.downsample[0])

    def _layer(self, seq: Any) -> list[tuple[_Conv, _Conv, _Conv | None]]:  # noqa: ANN401 - two `ResidualBlock`s
        return [self._block(m) for m in seq]

    @staticmethod
    def _residual(block: tuple[_Conv, _Conv, _Conv | None], x: mx.array, *, norm: bool) -> mx.array:
        """`ResidualBlock.forward`; `norm`: instance norms (their convolutions' biases vanish), else folded."""
        return MlxRAFTStereo._residual_out(block, x, MlxRAFTStereo._residual_in(block, x, norm=norm), norm=norm)

    @staticmethod
    def _residual_in(block: tuple[_Conv, _Conv, _Conv | None], x: mx.array, *, norm: bool) -> mx.array:
        """`ResidualBlock.forward`'s first convolution (with its norm and ReLU)."""
        return _instance_norm(block[0](x, bias=False)) if norm else block[0](x, relu=True)

    @staticmethod
    def _residual_out(block: tuple[_Conv, _Conv, _Conv | None], x: mx.array, y: mx.array, *, norm: bool) -> mx.array:
        """`ResidualBlock.forward` after its first convolution `y`."""
        _, conv2, down = block
        if norm:
            y = _instance_norm(conv2(y, bias=False))
            return _relu(x + y) if down is None else _instance_norm(down(x, bias=False), y)
        y = conv2(y, relu=True)
        return _relu((x if down is None else down(x)) + y)

    def _trunk(self, layers: dict[str, Any], names: Sequence[str], x: mx.array, *, norm: bool) -> mx.array:
        """The residual blocks of `layers[names]`, each convolution compiled with its elementwise steps and evaluated.

        Uncompiled, a block's every elementwise step (the norms' scale, shift and ReLU, the residual add) was its own
        full-size buffer, all alive until the block's evaluation: at H / 2 the transients outweighed the graph
        translation's fused steps. A convolution's transients are several times its output, so each is its own
        evaluation (994 x 1482 float16: a whole block per evaluation peaked at 1.43 GB, two blocks in flight 1.8 GB).
        """
        for name in names:
            for i, block in enumerate(layers[name]):
                key = (id(layers), name, i)
                fns = self._compiled.get(key)
                if fns is None:
                    fns = self._compiled[key] = (
                        mx.compile(functools.partial(self._residual_in, block, norm=norm)),
                        mx.compile(functools.partial(self._residual_out, block, norm=norm)),
                    )
                y = fns[0](x)
                mx.eval(y)
                x = fns[1](x, y)
                del y
                mx.eval(x)
        return x

    def _features(self, image: mx.array) -> mx.array:
        """`BasicEncoder.forward` of one image: `[1, H / 4, W / 4, 256]` float32."""
        x = _instance_norm(self.fnet_stem(image, bias=False))
        x = self._trunk(self.fnet, ("layer1", "layer2", "layer3"), x, norm=True)
        x = (x @ self.fnet_out[0] + self.fnet_out[1]).astype(mx.float32)
        mx.eval(x)
        return x

    def _context(self, image: mx.array) -> tuple[list[mx.array], list[mx.array]]:
        """`MultiBasicEncoder.forward` and the context convolutions.

        The hidden states and, per level, the GRU context `[zr | q]` with the gate convolutions' biases added.
        """
        x = self._trunk(self.cnet, ("layer1", "layer2", "layer3"), self.cnet_stem(image, relu=True), norm=False)
        y = self._trunk(self.cnet, ("layer4",), x, norm=False)
        z = self._trunk(self.cnet, ("layer5",), y, norm=False)
        net, ctx = [], []
        for level, (heads, gru) in zip((x, y, z), zip(self.heads, self.grus, strict=True), strict=True):
            hidden, inp = (
                conv(level if block is None else self._residual(block, level, norm=False)) for block, conv in heads
            )
            net.append(mx.tanh(hidden))
            ctx.append(self.context[len(ctx)](_relu(inp)) + gru[2])
        mx.eval(net, ctx)
        return net, ctx

    # update block -----------------------------------------------------------------------------------------------------

    def _gru(self, i: int, h: mx.array, ctx: mx.array, *x: mx.array) -> mx.array:
        """`ConvGRU.forward`, the z and r gates from one convolution."""
        convzr, convq, _ = self.grus[i]
        c = h.shape[-1]
        zr = mx.sigmoid(convzr(mx.concatenate([h, *x], axis=-1)) + ctx[..., : 2 * c])
        z, r = zr[..., :c], zr[..., c:]
        q = mx.tanh(convq(mx.concatenate([r * h, *x], axis=-1), bias=False) + ctx[..., 2 * c :])
        return (1 - z) * h + z * q

    def _coarse(self, net: list[mx.array], ctx: list[mx.array]) -> tuple[mx.array, mx.array]:
        """The 1/16 and 1/8 GRUs of an update step: their new hidden states."""
        n2 = self._gru(2, net[2], ctx[2], _pool(net[1]))
        return n2, self._gru(1, net[1], ctx[1], _pool(net[0]), _interp(n2, net[1].shape[1:3]))

    def _motion(self, flow: mx.array, pyramid: list[mx.array]) -> mx.array:
        """The correlation lookup at `flow` and the motion encoder."""
        _, h, w, _ = flow.shape
        x = (mx.arange(w, dtype=mx.float32)[None, :, None] + flow[0]).reshape(h * w, 1)
        corr = [ops.linear_sample(vol, x / 2**i + self.taps) for i, vol in enumerate(pyramid)]
        corr = mx.concatenate(corr, axis=-1).reshape(1, h, w, -1).astype(self.dtype)
        cor = self.convc2(_relu(corr @ self.convc1[0] + self.convc1[1]), relu=True)
        half = flow.astype(self.dtype)
        flo = self.convf2(_relu(mx.conv2d(half, self.convf1[0], padding=3) + self.convf1[1]), relu=True)
        return self.motion(mx.concatenate([cor, flo], axis=-1), relu=True) + half * self.flow_slot

    def _gates0(self, h0: mx.array, ctx0: mx.array, motion: mx.array, n1: mx.array) -> tuple[mx.array, mx.array]:
        """The 1/4 GRU's z and r gates (`_gru`'s first convolution) and the upsampled 1/8 hidden state."""
        c = h0.shape[-1]
        up = _interp(n1, h0.shape[1:3])
        convzr = self.grus[0][0]
        return mx.sigmoid(convzr(mx.concatenate([h0, motion, up], axis=-1)) + ctx0[..., : 2 * c]), up

    def _hidden0(self, h0: mx.array, ctx0: mx.array, motion: mx.array, up: mx.array, zr: mx.array) -> mx.array:
        """The 1/4 GRU's new hidden state from its gates (`_gru`'s second convolution)."""
        c = h0.shape[-1]
        z, r = zr[..., :c], zr[..., c:]
        q = mx.tanh(self.grus[0][1](mx.concatenate([r * h0, motion, up], axis=-1), bias=False) + ctx0[..., 2 * c :])
        return (1 - z) * h0 + z * q

    def _head(self, n0: mx.array, flow: mx.array) -> mx.array:
        """The flow head: the updated flow."""
        delta = mx.conv2d(self.flow1(n0, relu=True), self.flow2[0], padding=1) + self.flow2[1]
        return flow + delta.astype(mx.float32)

    def _iteration(self, net: list[mx.array], flow: mx.array, ctx: list[mx.array], pyramid: list[mx.array]) -> Any:  # noqa: ANN401
        """One update step as five compiled parts, each evaluated before the next is queued.

        A command buffer's transients are freed when it completes, and the full-resolution (1/4) convolutions'
        transients are several times their outputs: one compiled step held all of them (994 x 1482 float16: 1.1 GB
        above the live set, the graph translation's whole network peak 1.04 GB), so each 1/4 convolution is its own
        evaluation.
        """
        n2, n1 = self._parts[0](net, ctx)
        mx.eval(n2, n1)
        motion = self._parts[1](flow, pyramid)
        mx.eval(motion)
        zr, up = self._parts[2](net[0], ctx[0], motion, n1)
        mx.eval(zr, up)
        n0 = self._parts[3](net[0], ctx[0], motion, up, zr)
        mx.eval(n0)
        del zr, up, motion
        return [n0, n1, n2], self._parts[4](n0, flow)

    def _upsample(self, flow: mx.array, hidden: mx.array) -> mx.array:
        """The mask head and upstream's convex upsampling (factor 4) of the x flow: `[1, 4 H, 4 W]` float32."""
        _, h, w, _ = flow.shape
        mask = self.mask1(hidden, relu=True) @ self.mask2[0] + self.mask2[1]
        mask = mx.softmax(0.25 * mask.astype(mx.float32).reshape(h, w, 9, 4, 4), axis=2)
        fp = mx.pad(4 * flow[0, :, :, 0], 1)
        near = mx.stack([fp[i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=-1)  # [H, W, 9]
        up = (mask * near[..., None, None]).sum(axis=2)  # [H, W, 4, 4]
        return up.transpose(0, 2, 1, 3).reshape(1, 4 * h, 4 * w)

    # the network ------------------------------------------------------------------------------------------------------

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32) of a padded float32 pair `[1, 3, H, W]` in 0-255."""
        left = (2 * (left.transpose(0, 2, 3, 1) / 255.0) - 1.0).astype(self.dtype)
        right = (2 * (right.transpose(0, 2, 3, 1) / 255.0) - 1.0).astype(self.dtype)
        net, ctx = self._context(left)
        fmap1, fmap2 = self._features(left), self._features(right)
        _, h, w, d = fmap1.shape
        corr = (fmap1[0] @ mx.swapaxes(fmap2[0], 1, 2)).reshape(h * w, w) * d**-0.5  # row by row: [H W, W]
        pyramid = [corr]
        for _ in range(1, LEVELS):
            n = corr.shape[1] // 2
            corr = corr[:, : 2 * n].reshape(-1, n, 2).mean(axis=-1)
            pyramid.append(corr)
        flow = mx.zeros((1, h, w, 1), mx.float32)
        mx.eval(pyramid)
        del fmap1, fmap2
        for _ in range(self.iters):
            net, flow = self._step(net, flow, ctx, pyramid)
            mx.eval(net, flow)
        return -self._upsample(flow, net[0])
