"""SEA-RAFT on Apple MLX, written in `mlx.core` operations (`MlxSEARAFT(net, dtype, iters)`): a padded pair -> flow.

The maths of `net.SEARAFT.forward`, channels last, batch norms folded. Written out instead of replayed from an
exported graph: the export and its translation were most of the first call (sea-raft-s-spring 1.6 s).

- No correlation volume: `graph_mlx.ops.corr_windows` takes each window's dot products from fmap1 and the halved
  fmap2 levels (the halving is linear, so the correlation with a halved map is the halved correlation).
- The context encoder's stem (7 x 7, stride 2, on both images stacked as 6 channels) is two 3-channel convolutions
  summed: MLX's convolution is ~9x slower on 6 input channels than on 3.
- The update step is one compiled function: the motion encoder's last conv has 128 outputs (two zero filters, the
  flow added into their place; 128 outputs run in MLX's Winograd), the flow head only its two flow outputs, the
  ConvNeXt blocks' depthwise 7 x 7 and layer norm the shared kernels, their 1 x 1 convs matrix products.
- float32 in a float16 run: the encoders' 128 -> 128 3 x 3 convs (MLX's half Winograd put sea-raft-s-spring's golden
  over its tolerance), the flow state and its first conv, the correlation windows, the upsampling mask's softmax
  and the convex upsampling. The feature maps stay float16 for the correlation (`net.py`'s note).
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

__all__ = ["MlxSEARAFT"]

FACTOR = 8
ONE = (1, 1)
F32_CONV = 128  # float16 3 x 3 convs of this width in and out compute in float32


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().float().numpy().astype(np.float64)


def _ohwi(conv: Any) -> np.ndarray:  # noqa: ANN401 - a `Conv2d`
    return _np(conv.weight).transpose(0, 2, 3, 1)


def _folded(conv: Any, bn: Any) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - torch modules
    """A convolution's `[O, kh, kw, I]` weight and bias with the batch norm after it folded in."""
    k = _np(bn.weight) / np.sqrt(_np(bn.running_var) + bn.eps)
    bias = _np(conv.bias) if conv.bias is not None else 0.0
    return _ohwi(conv) * k[:, None, None, None], (bias - _np(bn.running_mean)) * k + _np(bn.bias)


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _gelu(x: mx.array) -> mx.array:
    """`nn.GELU()` (the erf form)."""
    return 0.5 * x * (1 + mx.erf(x * (1 / math.sqrt(2))))


def _halve(x: mx.array) -> mx.array:
    """`F.interpolate(x, scale_factor=0.5, mode="bilinear")` of an NHWC map: the mean of each 2 x 2 block."""
    n, h, w, c = x.shape
    h, w = h // 2, w // 2
    y = x[:, : 2 * h, : 2 * w].astype(mx.float32).reshape(n, h, 2, w, 2, c)
    return (0.25 * y.sum(axis=(2, 4))).astype(x.dtype)


class MlxSEARAFT:
    """The network `net` (a loaded `SEARAFT`, float32 weights) in `dtype` (the name), `iters` updates."""

    def __init__(self, net: Any, dtype: str, iters: int) -> None:  # noqa: ANN401 - `SEARAFT`, read by attribute
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        self.iters = iters
        self.cfg = net.cfg
        cnet, fnet = net.cnet, net.fnet
        w, b = _folded(cnet.conv1, cnet.bn1)
        self.cstem = (self._a(w[..., :3]), self._a(w[..., 3:]), self._a(b))
        self.cnet = self._encoder(cnet)
        self.fstem = tuple(self._a(a) for a in _folded(fnet.conv1, fnet.bn1))
        self.fnet = self._encoder(fnet)
        self.init_conv = self._conv(net.init_conv)
        fh = net.flow_head
        self.flow1 = self._conv(fh[0])
        self.flow2 = (self._a(_ohwi(fh[2])[:2]), self._a(_np(fh[2].bias)[:2]))  # the flow; not the uncertainty
        uw = net.upsample_weight
        self.mask1 = self._conv(uw[0])
        self.mask2 = (self._a(0.25 * _np(uw[2].weight)[:, :, 0, 0].T), self._a(0.25 * _np(uw[2].bias)))
        self._update(net.update_block)
        self._step = mx.compile(self._iteration)

    def _a(self, a: np.ndarray, dtype: mx.Dtype | None = None) -> mx.array:
        return mx.array(np.ascontiguousarray(a, dtype=np.float32)).astype(dtype or self.dtype)

    def _conv(self, conv: Any) -> tuple[mx.array, mx.array]:  # noqa: ANN401 - a `Conv2d`
        return self._a(_ohwi(conv)), self._a(_np(conv.bias))

    # encoders ---------------------------------------------------------------------------------------------------------

    def _layer(self, w: np.ndarray, b: np.ndarray, stride: int) -> _Layer:
        """A folded conv (padding k // 2); a float16 128 -> 128 3 x 3 conv computes in float32."""
        pad = w.shape[1] // 2
        if w.shape[1] == 1:
            wm, bm = self._a(w[:, 0, 0].T), self._a(b)
            return lambda x: (x[:, ::stride, ::stride] if stride > 1 else x) @ wm + bm
        f32 = self.dtype == mx.float16 and stride == 1 and w.shape[0] == w.shape[3] == F32_CONV
        wm, bm = self._a(w, mx.float32 if f32 else None), self._a(b, mx.float32 if f32 else None)
        if f32:
            return lambda x: (mx.conv2d(x.astype(mx.float32), wm, stride, pad) + bm).astype(self.dtype)
        return lambda x: mx.conv2d(x, wm, stride, pad) + bm

    def _encoder(self, enc: Any) -> _Layer:  # noqa: ANN401 - `ResNetFPN`
        """`ResNetFPN.forward` after the stem (its relu included)."""
        blocks = []
        for layer in (enc.layer1, enc.layer2, enc.layer3):
            for blk in layer:
                stride = blk.conv1.stride[0]
                c1 = self._layer(*_folded(blk.conv1, blk.bn1), stride)
                c2 = self._layer(*_folded(blk.conv2, blk.bn2), 1)
                down = self._layer(*_folded(blk.downsample[0], blk.bn3), stride) if blk.downsample else None
                blocks.append((c1, c2, down))
        final = self._layer(_ohwi(enc.final_conv), _np(enc.final_conv.bias), 1)

        def run(x: mx.array) -> mx.array:
            for c1, c2, down in blocks:
                y = _relu(c2(_relu(c1(x))))
                x = _relu((down(x) if down else x) + y)
                mx.eval(x)  # block by block: the large convs' transients do not pile up
            return final(x)

        return run

    # update block -----------------------------------------------------------------------------------------------------

    def _update(self, ub: Any) -> None:  # noqa: ANN401 - `BasicUpdateBlock`
        e = ub.encoder
        self.convc1 = (self._a(_np(e.convc1.weight)[:, :, 0, 0].T), self._a(_np(e.convc1.bias)))
        self.convc2 = self._conv(e.convc2)
        self.convf1 = (self._a(_ohwi(e.convf1), mx.float32), self._a(_np(e.convf1.bias), mx.float32))
        self.convf2 = self._conv(e.convf2)
        w, b = _ohwi(e.conv), _np(e.conv.bias)
        self.motion_out = w.shape[0]  # dim - 2: the flow takes the last two channels
        pad = w.shape[3] // 2 - w.shape[0]
        self.motion = (self._a(np.pad(w, [(0, pad), (0, 0), (0, 0), (0, 0)])), self._a(np.pad(b, (0, pad))))
        self.refine = []
        for blk in ub.refine:
            gamma = _np(blk.gamma)
            self.refine.append(
                {
                    "dw": self._a(_np(blk.dwconv.weight)[:, 0].transpose(1, 2, 0)),
                    "dwb": self._a(_np(blk.dwconv.bias)),
                    "ln": (self._a(_np(blk.norm.weight)), self._a(_np(blk.norm.bias)), blk.norm.eps),
                    "pw1": (self._a(_np(blk.pwconv1.weight).T), self._a(_np(blk.pwconv1.bias))),
                    "pw2": (self._a(_np(blk.pwconv2.weight).T * gamma), self._a(_np(blk.pwconv2.bias) * gamma)),
                    "final": (self._a(_np(blk.final.weight)[:, :, 0, 0].T), self._a(_np(blk.final.bias))),
                }
            )

    def _convnext(self, p: dict[str, Any], x: mx.array) -> mx.array:
        y = ops.depthwise(x, p["dw"], p["dwb"], ONE, (3, 3), ONE)
        y = ops.row_layer_norm(y, *p["ln"])
        y = _gelu(y @ p["pw1"][0] + p["pw1"][1]) @ p["pw2"][0] + p["pw2"][1]
        return (x + y) @ p["final"][0] + p["final"][1]

    def _flow_head(self, net: mx.array) -> mx.array:
        """The flow head's flow channels, float32 `[1, h, w, 2]`."""
        y = _relu(mx.conv2d(net, self.flow1[0], padding=1) + self.flow1[1])
        return (mx.conv2d(y, self.flow2[0], padding=1) + self.flow2[1]).astype(mx.float32)

    def _iteration(
        self, net: mx.array, context: mx.array, flow: mx.array, fmap1: mx.array, *levels: mx.array
    ) -> tuple[mx.array, mx.array]:
        _, h, w, _ = flow.shape
        cfg = self.cfg
        grid = mx.stack(
            [mx.broadcast_to(mx.arange(w, dtype=mx.float32)[None], (h, w)),
             mx.broadcast_to(mx.arange(h, dtype=mx.float32)[:, None], (h, w))],
        )  # fmt: skip
        coords = grid[None] + flow[0].transpose(2, 0, 1)[None]
        corr = ops.corr_windows(fmap1, list(levels), coords, cfg.radius, 1 / math.sqrt(fmap1.shape[-1]))
        corr = corr[0].reshape(corr.shape[1], h * w).T.astype(self.dtype)  # [h w, L (2r + 1)^2]
        cor = _relu(corr @ self.convc1[0] + self.convc1[1]).reshape(1, h, w, -1)
        cor = _relu(mx.conv2d(cor, self.convc2[0], padding=1) + self.convc2[1])
        flo = _relu(mx.conv2d(flow, self.convf1[0], padding=3) + self.convf1[1]).astype(self.dtype)
        flo = _relu(mx.conv2d(flo, self.convf2[0], padding=1) + self.convf2[1])
        motion = _relu(mx.conv2d(mx.concatenate([cor, flo], axis=-1), self.motion[0], padding=1) + self.motion[1])
        motion = mx.concatenate([motion[..., : self.motion_out], flow.astype(self.dtype)], axis=-1)
        inp = mx.concatenate([context, motion], axis=-1)
        for p in self.refine:
            net = self._convnext(p, mx.concatenate([net, inp], axis=-1))
        return net, flow + self._flow_head(net)

    def _upsample(self, flow: mx.array, net: mx.array) -> mx.array:
        """The mask head and the convex upsampling by 8: float32 `[1, 8 h, 8 w, 2]`."""
        _, h, w, _ = flow.shape
        mask = _relu(mx.conv2d(net, self.mask1[0], padding=1) + self.mask1[1]) @ self.mask2[0] + self.mask2[1]
        mask = mx.softmax(mask.astype(mx.float32).reshape(h, w, 9, FACTOR, FACTOR), axis=2)
        fp = mx.pad(FACTOR * flow[0], ((1, 1), (1, 1), (0, 0)))
        near = mx.stack([fp[i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=2)  # [h, w, 9, 2]
        up = (mask[..., None] * near[:, :, :, None, None]).sum(axis=2)  # [h, w, 8, 8, 2]
        return up.transpose(0, 2, 1, 3, 4).reshape(1, FACTOR * h, FACTOR * w, 2)

    # the network ------------------------------------------------------------------------------------------------------

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[1, 2, H, W]` (float32) of a padded float32 pair `[1, 3, H, W]` in [-1, 1]."""
        x1 = image1.transpose(0, 2, 3, 1).astype(self.dtype)
        x2 = image2.transpose(0, 2, 3, 1).astype(self.dtype)
        w1, w2, b = self.cstem
        x = _relu(mx.conv2d(x1, w1, 2, 3) + mx.conv2d(x2, w2, 2, 3) + b)
        cmap = self.cnet(x)
        mx.eval(cmap)
        cmap = mx.conv2d(cmap, self.init_conv[0], padding=1) + self.init_conv[1]
        dim = self.cfg.dim
        net, context = cmap[..., :dim], cmap[..., dim:]
        flow = self._flow_head(net)
        mx.eval(net, context, flow)
        w, b = self.fstem
        fmap1 = self.fnet(_relu(mx.conv2d(x1, w, 2, 3) + b))  # one image at a time: half the peak
        mx.eval(fmap1)
        levels = [self.fnet(_relu(mx.conv2d(x2, w, 2, 3) + b))]
        for _ in range(self.cfg.levels - 1):
            levels.append(_halve(levels[-1]))
        mx.eval(fmap1, levels)
        del cmap
        for _ in range(self.iters):
            net, flow = self._step(net, context, flow, fmap1, *levels)
            mx.eval(net, flow)
        up = self._upsample(flow, net).transpose(0, 3, 1, 2)
        mx.eval(up)
        return up
