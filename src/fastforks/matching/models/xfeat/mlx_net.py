"""XFeat's dense network on Apple MLX, written in `mlx.core` operations (`MlxDense(net)`): image -> descriptors, scores.

`XFeatModel.forward` (`net/model.py`) with the steps of `detectAndCompute` that have a fixed shape (the resize to a
multiple of 32, the descriptor normalisation, the keypoint heatmap and its score map, `mlx_post.score_map`) as one
compiled function. The network is 26 small convolutions, so it is written out instead of replayed from an exported
graph: the export alone is ~1 s of a first call (the whole first call here is ~0.2 s). The weights are read from the
PyTorch module once: batch norm folded into each convolution, channels last, the 24-channel layers zero-padded to 32
(MLX's convolution is slower on output channels that are not a multiple of 16).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import bilinear_nhwc

from . import mlx_post

if TYPE_CHECKING:
    from .net.model import XFeatModel

__all__ = ["MlxDense"]

_Conv = tuple[mx.array, mx.array, int, int, bool]  # weight [O, kh, kw, I], bias, stride, padding, ReLU after


_ALIGN = 16


def _aligned(n: int) -> int:
    """`n` channels as the layer is run: from `_ALIGN` up, its next multiple."""
    return n if n < _ALIGN else -(-n // _ALIGN) * _ALIGN


def _conv(conv: Any, norm: Any, *, out: int | None = None) -> _Conv:  # noqa: ANN401 - torch modules of the vendored net
    """A `Conv2d` with its `BatchNorm2d` (or None) folded in, zero-padded to the aligned widths (`out`: outputs)."""
    w = conv.weight.detach().float().numpy()
    b = np.zeros(len(w), np.float32) if conv.bias is None else conv.bias.detach().float().numpy()
    if norm is not None:  # `affine=False`: the running statistics only
        scale = 1.0 / np.sqrt(norm.running_var.numpy().astype(np.float64) + norm.eps)
        w = (w * scale[:, None, None, None]).astype(np.float32)
        b = ((b - norm.running_mean.numpy()) * scale).astype(np.float32)
    o, i = w.shape[:2]
    full = np.zeros((out or _aligned(o), w.shape[2], w.shape[3], _aligned(i)), np.float32)
    full[:o, :, :, :i] = w.transpose(0, 2, 3, 1)
    bias = np.zeros(len(full), np.float32)
    bias[:o] = b
    return mx.array(full), mx.array(bias), int(conv.stride[0]), int(conv.padding[0]), norm is not None


def _layers(block: Any, *, out: int | None = None) -> list[_Conv]:  # noqa: ANN401 - torch modules of the vendored net
    """The convolutions of a block of `BasicLayer`s and plain convolutions (`out`: outputs of the last one)."""
    convs = []
    for k, m in enumerate(block):
        last = out if k == len(block) - 1 else None
        if hasattr(m, "layer"):
            convs.append(_conv(m.layer[0], m.layer[1], out=last))
        elif hasattr(m, "weight"):
            convs.append(_conv(m, None, out=last))
    return convs


def _run(x: mx.array, convs: list[_Conv]) -> mx.array:
    for w, b, stride, padding, relu in convs:
        x = mx.conv2d(x, w, stride=stride, padding=padding) + b
        if relu:
            x = mx.maximum(x, 0)
    return x


def _resize(x: mx.array, h: int, w: int) -> mx.array:
    """`F.interpolate(x, (h, w), mode="bilinear")` of `[1, H, W, C]`."""
    return bilinear_nhwc(x, (h, w), (x.shape[1] / h, x.shape[2] / w), align=False)


class MlxDense:
    """The extractor network `net` (eval, weights loaded) with the score map: one compiled function per image size."""

    def __init__(self, net: XFeatModel, threshold: float) -> None:
        """Copy the weights of `net` to MLX; `threshold` is the detection threshold of the score map."""
        self.threshold = threshold
        self.eps = float(net.norm.eps)
        self.blocks = [_layers(b) for b in (net.block1, net.block2, net.block3, net.block4, net.block5)]
        self.skip = _layers(net.skip1)[0]
        self.fusion = _layers(net.block_fusion)
        self.reliability = _layers(net.heatmap_head, out=1)
        self.keypoints = _layers(net.keypoint_head, out=65)
        self._compiled = mx.compile(self._forward)

    def _forward(self, image: mx.array) -> tuple[mx.array, mx.array]:
        x = image.transpose(0, 2, 3, 1)
        h, w = x.shape[1] // 32 * 32, x.shape[2] // 32 * 32
        if x.shape[1:3] != (h, w):  # upstream resizes always: at the same size that is the identity
            x = _resize(x, h, w)
        x = x.mean(axis=-1, keepdims=True)
        x = (x - x.mean()) * mx.rsqrt(x.var() + self.eps)  # `InstanceNorm2d(1)`
        skip = x.reshape(1, h // 4, 4, w // 4, 4).mean(axis=(2, 4))[..., None]
        x1 = _run(x, self.blocks[0]) + (skip * self.skip[0].reshape(-1) + self.skip[1])
        x3 = _run(_run(x1, self.blocks[1]), self.blocks[2])
        x4 = _run(x3, self.blocks[3])
        x5 = _run(x4, self.blocks[4])
        fused = x3 + _resize(x4, h // 8, w // 8) + _resize(x5, h // 8, w // 8)
        feats = _run(fused, self.fusion)
        reliability = mx.sigmoid(_run(feats, self.reliability))[0, :, :, 0]
        # `_unfold2d(x, 8)`: an 8x8 cell's pixels as its 64 channels; `get_kpts_heatmap` is the way back
        cells = x.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(1, h // 8, w // 8, 64)
        logits = _run(cells, self.keypoints)
        heat = mx.softmax(logits, axis=-1)[0, :, :, :64]
        heat = heat.reshape(h // 8, w // 8, 8, 8).transpose(0, 2, 1, 3).reshape(h, w)
        feats = feats[0] * mx.rsqrt(mx.maximum((feats[0] * feats[0]).sum(axis=-1, keepdims=True), 1e-24))
        return feats, mlx_post.score_map(heat, reliability, self.threshold)

    def __call__(self, image: mx.array) -> tuple[mx.array, mx.array]:
        """Of a float32 RGB image `[1, 3, H, W]`: unit descriptors `[H/8, W/8, 64]` and the score map `[H, W]`.

        Both of the image resized to multiples of 32 (`H`, `W` here); lazy, one evaluation computes both.
        """
        return self._compiled(image)
