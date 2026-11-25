"""RAFT on Apple MLX, written in `mlx.core` operations (`MlxRAFT(net, dtype, iters)`): a padded pair -> flow.

The maths of `net.RAFT.forward`, channels last. Written out instead of replayed from an exported graph: the export
and its translation were most of the first call (raft-small 1.9 s, raft-large 2.6-3.9 s).

- the feature encoder runs one image at a time, its instance norms from float32 statistics (`ops.channel_stats`); a
  convolution before one drops its bias. The context encoder's batch norms (raft-large) are folded.
- no correlation volume: `graph_mlx.ops.corr_windows` takes each window's dot products from fmap1 and the 2 x 2
  average-pooled fmap2 levels (the mean of the correlations is the correlation with the mean).
- the update step is one compiled function. The GRU convolutions over `[hidden, context, motion]` split by input:
  the context part does not change over the updates, so it is convolved once (with the biases); each update
  convolves `[hidden, motion]`, z and r in one convolution. The motion features are zero-padded to a multiple of 32
  channels with the flow last (small: 82 -> 96), and the motion encoder's last conv has zero outputs up to what MLX's
  Winograd 3 x 3 takes (large 126 -> 128, small 80 -> 128).
- float32: the flow state and its first conv, the correlation windows, the upsampling.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops

if TYPE_CHECKING:
    from collections.abc import Callable

    _Layer = Callable[[mx.array], mx.array]

__all__ = ["MlxRAFT"]

FACTOR = 8
ALIGN = 32  # MLX's fast 3 x 3 convolution: input and output channels in multiples of 32
WINOGRAD = 256  # ... and at least this many input + output channels


def _up(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().float().numpy().astype(np.float64)


def _ohwi(w: np.ndarray) -> np.ndarray:
    return w.transpose(0, 2, 3, 1)


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _instance_norm(x: mx.array) -> mx.array:
    """`InstanceNorm2d` (no affine, eps 1e-5) of one NHWC image: float32 statistics, a multiply-add in its dtype."""
    mean, var = ops.channel_stats(x.reshape(-1, x.shape[-1]))
    scale = mx.rsqrt(var + 1e-5)
    return x * scale.astype(x.dtype) - (mean * scale).astype(x.dtype)


def _halve(x: mx.array) -> mx.array:
    """`F.avg_pool2d(x, 2, 2)` of an NHWC map (an odd last row / column dropped), in float32."""
    n, h, w, c = x.shape
    h, w = h // 2, w // 2
    return 0.25 * x[:, : 2 * h, : 2 * w].astype(mx.float32).reshape(n, h, 2, w, 2, c).sum(axis=(2, 4))


class MlxRAFT:
    """The network `net` (a loaded `RAFT`, float32 weights) in `dtype` (the name), `iters` updates."""

    def __init__(self, net: Any, dtype: str, iters: int) -> None:  # noqa: ANN401 - `RAFT`, read by attribute
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        self.iters = iters
        cfg = self.cfg = net.cfg
        self.fnet = self._encoder(net.feature_encoder, "instance")
        self.cnet = self._encoder(net.context_encoder, "batch" if cfg.context_norm else None)
        ub = net.update_block
        context = cfg.context_layers[-1] - cfg.hidden
        self.motion_out = cfg.motion_out
        self.motion_pad = _up(cfg.motion_out) - cfg.motion_out  # zero channels before the flow
        self._motion_encoder(ub.motion_encoder)
        rb = ub.recurrent_block
        grus = [rb.convgru1] + ([rb.convgru2] if hasattr(rb, "convgru2") else [])
        self.grus = [self._gru(g, cfg.hidden, context) for g in grus]
        fh = ub.flow_head
        self.flow1 = self._conv(fh.conv1)
        self.flow2 = self._conv(fh.conv2)
        if cfg.mask:
            mp = net.mask_predictor
            self.mask1 = self._conv(mp.convrelu[0])
            self.mask2 = (
                self._a(0.25 * _np(mp.conv.weight)[:, :, 0, 0].T),
                self._a(0.25 * _np(mp.conv.bias)),
            )
        self._step = mx.compile(self._iteration)

    def _a(self, a: np.ndarray, dtype: mx.Dtype | None = None) -> mx.array:
        return mx.array(np.ascontiguousarray(a, dtype=np.float32)).astype(dtype or self.dtype)

    def _conv(self, conv: Any, dtype: mx.Dtype | None = None) -> tuple[mx.array, mx.array]:  # noqa: ANN401 - a `Conv2d`
        return self._a(_ohwi(_np(conv.weight)), dtype), self._a(_np(conv.bias), dtype)

    # encoders ---------------------------------------------------------------------------------------------------------

    def _unit(self, seq: Any, norm: str | None, *, act: bool = True) -> _Layer:  # noqa: ANN401 - `_cna`'s Sequential
        """`Conv2dNormActivation`: conv, instance norm (the bias dropped) or folded batch norm, ReLU."""
        conv = seq[0]
        w, b = _ohwi(_np(conv.weight)), _np(conv.bias)
        if norm == "batch":
            bn = seq[1]
            k = _np(bn.weight) / np.sqrt(_np(bn.running_var) + bn.eps)
            w, b = w * k[:, None, None, None], (b - _np(bn.running_mean)) * k + _np(bn.bias)
        stride, pad = conv.stride[0], conv.padding[0]
        inorm = norm == "instance"
        if w.shape[1] == 1:
            wm, bm = self._a(w[:, 0, 0].T), self._a(b)

            def conv_fn(x: mx.array) -> mx.array:
                y = (x[:, ::stride, ::stride] if stride > 1 else x) @ wm
                return y if inorm else y + bm
        else:
            wc, bc = self._a(w), self._a(b)

            def conv_fn(x: mx.array) -> mx.array:
                y = mx.conv2d(x, wc, stride, pad)
                return y if inorm else y + bc

        def run(x: mx.array) -> mx.array:
            y = conv_fn(x)
            y = _instance_norm(y) if inorm else y
            return _relu(y) if act else y

        return run

    def _encoder(self, enc: Any, norm: str | None) -> _Layer:  # noqa: ANN401 - `FeatureEncoder`
        """`FeatureEncoder.forward` of one image."""
        stem = self._unit(enc.convnormrelu, norm)
        blocks = []
        for layer in (enc.layer1, enc.layer2, enc.layer3):
            for blk in layer:
                units = [self._unit(blk.convnormrelu1, norm), self._unit(blk.convnormrelu2, norm)]
                if hasattr(blk, "convnormrelu3"):
                    units.append(self._unit(blk.convnormrelu3, norm))
                ds = blk.downsample
                down = self._unit(ds, norm, act=False) if list(ds.children()) else None
                blocks.append((units, down))
        out = (self._a(_np(enc.conv.weight)[:, :, 0, 0].T), self._a(_np(enc.conv.bias)))

        def run(x: mx.array) -> mx.array:
            x = stem(x)
            for units, down in blocks:
                y = x
                for u in units:
                    y = u(y)
                x = _relu((down(x) if down else x) + y)
                mx.eval(x)
            return x @ out[0] + out[1]

        return run

    # update block -----------------------------------------------------------------------------------------------------

    def _motion_encoder(self, e: Any) -> None:  # noqa: ANN401 - `MotionEncoder`
        c1 = e.convcorr1[0]
        self.convc1 = (self._a(_np(c1.weight)[:, :, 0, 0].T), self._a(_np(c1.bias)))
        self.convc2 = self._conv(e.convcorr2[0]) if list(e.convcorr2.children()) else None
        self.convf1 = self._conv(e.convflow1[0], mx.float32)
        self.convf2 = self._conv(e.convflow2[0])
        conv = e.conv[0]
        w, b = _ohwi(_np(conv.weight)), _np(conv.bias)
        extra = max(_up(w.shape[0]), WINOGRAD - w.shape[3]) - w.shape[0]
        self.motion = (
            self._a(np.pad(w, [(0, extra), (0, 0), (0, 0), (0, 0)])),
            self._a(np.pad(b, (0, extra))),
        )

    def _gru(self, gru: Any, hidden: int, context: int) -> dict[str, Any]:  # noqa: ANN401 - `ConvGRU`
        """A `ConvGRU`'s weights regrouped: `[hidden | motion, zeros, flow]` per update, `[context]` once."""
        m, pad = self.motion_out, self.motion_pad

        def loop(w: np.ndarray) -> np.ndarray:  # [O, hidden + context + motion, kh, kw] -> OHWI without the context
            h, mo = w[:, :hidden], w[:, hidden + context :]
            z = np.zeros((w.shape[0], pad, *w.shape[2:]))
            return _ohwi(np.concatenate([h, mo[:, : m - 2], z, mo[:, m - 2 :]], axis=1))

        wz, wr, wq = (_np(c.weight) for c in (gru.convz, gru.convr, gru.convq))
        ctx = np.concatenate([w[:, hidden : hidden + context] for w in (wz, wr, wq)])
        bias = np.concatenate([_np(c.bias) for c in (gru.convz, gru.convr, gru.convq)])
        return {
            "pad": tuple(gru.convz.padding),
            "zr": self._a(np.concatenate([loop(wz), loop(wr)])),
            "q": self._a(loop(wq)),
            "ctx": (self._a(_ohwi(ctx)), self._a(bias)),
        }

    def _iteration(
        self, hidden: mx.array, flow: mx.array, shares: list[mx.array], fmap1: mx.array, *levels: mx.array
    ) -> tuple[mx.array, mx.array]:
        _, h, w, _ = flow.shape
        cfg = self.cfg
        grid = mx.stack(
            [mx.broadcast_to(mx.arange(w, dtype=mx.float32)[None], (h, w)),
             mx.broadcast_to(mx.arange(h, dtype=mx.float32)[:, None], (h, w))],
        )  # fmt: skip
        coords = grid[None] + flow[0].transpose(2, 0, 1)[None]
        corr = ops.corr_windows(fmap1, list(levels), coords, cfg.corr_radius, 1 / math.sqrt(fmap1.shape[-1]))
        corr = corr[0].reshape(corr.shape[1], h * w).T.astype(self.dtype)  # [h w, L (2r + 1)^2]
        cor = _relu(corr @ self.convc1[0] + self.convc1[1]).reshape(1, h, w, -1)
        if self.convc2 is not None:
            cor = _relu(mx.conv2d(cor, self.convc2[0], padding=1) + self.convc2[1])
        flo = _relu(mx.conv2d(flow, self.convf1[0], padding=3) + self.convf1[1]).astype(self.dtype)
        flo = _relu(mx.conv2d(flo, self.convf2[0], padding=1) + self.convf2[1])
        m = _relu(mx.conv2d(mx.concatenate([cor, flo], axis=-1), self.motion[0], padding=1) + self.motion[1])
        motion = mx.concatenate([m[..., : self.motion_out - 2 + self.motion_pad], flow.astype(self.dtype)], axis=-1)
        c = hidden.shape[-1]
        for g, share in zip(self.grus, shares, strict=True):
            zr = mx.sigmoid(mx.conv2d(mx.concatenate([hidden, motion], axis=-1), g["zr"], padding=g["pad"])
                            + share[..., : 2 * c])  # fmt: skip
            z, r = zr[..., :c], zr[..., c:]
            q = mx.tanh(mx.conv2d(mx.concatenate([r * hidden, motion], axis=-1), g["q"], padding=g["pad"])
                        + share[..., 2 * c :])  # fmt: skip
            hidden = (1 - z) * hidden + z * q
        y = _relu(mx.conv2d(hidden, self.flow1[0], padding=1) + self.flow1[1])
        delta = mx.conv2d(y, self.flow2[0], padding=1) + self.flow2[1]
        return hidden, flow + delta.astype(mx.float32)

    def _upsample(self, flow: mx.array, hidden: mx.array) -> mx.array:
        """Flow at 1/8 -> `[1, 8 h, 8 w, 2]` float32: convex upsampling (large) or bilinear (small)."""
        _, h, w, _ = flow.shape
        if not self.cfg.mask:
            size = (FACTOR * h, FACTOR * w)
            ratios = ((h - 1) / (size[0] - 1), (w - 1) / (size[1] - 1))
            return FACTOR * ops.bilinear_nhwc(flow, size, ratios, align=True)
        mask = _relu(mx.conv2d(hidden, self.mask1[0], padding=1) + self.mask1[1]) @ self.mask2[0] + self.mask2[1]
        mask = mx.softmax(mask.astype(mx.float32).reshape(h, w, 9, FACTOR, FACTOR), axis=2)
        fp = mx.pad(FACTOR * flow[0], ((1, 1), (1, 1), (0, 0)))
        near = mx.stack([fp[i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=2)  # [h, w, 9, 2]
        up = (mask[..., None] * near[:, :, :, None, None]).sum(axis=2)  # [h, w, 8, 8, 2]
        return up.transpose(0, 2, 1, 3, 4).reshape(1, FACTOR * h, FACTOR * w, 2)

    # the network ------------------------------------------------------------------------------------------------------

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[1, 2, H, W]` (float32) of a padded float32 pair `[1, 3, H, W]` in [-1, 1]."""
        cfg = self.cfg
        x1 = image1.transpose(0, 2, 3, 1).astype(self.dtype)
        x2 = image2.transpose(0, 2, 3, 1).astype(self.dtype)
        fmap1 = self.fnet(x1)
        level = self.fnet(x2)
        mx.eval(fmap1, level)
        levels = [level]
        for _ in range(cfg.corr_levels - 1):
            levels.append(_halve(levels[-1]).astype(level.dtype))
        ctx = self.cnet(x1)
        hidden, context = mx.tanh(ctx[..., : cfg.hidden]), _relu(ctx[..., cfg.hidden :])
        shares = [mx.conv2d(context, g["ctx"][0], padding=g["pad"]) + g["ctx"][1] for g in self.grus]
        _, h, w, _ = hidden.shape
        flow = mx.zeros((1, h, w, 2), mx.float32)
        mx.eval(levels, hidden, shares)
        del ctx, context
        for _ in range(self.iters):
            hidden, flow = self._step(hidden, flow, shares, fmap1, *levels)
            mx.eval(hidden, flow)
        up = self._upsample(flow, hidden).transpose(0, 3, 1, 2)
        mx.eval(up)
        return up
