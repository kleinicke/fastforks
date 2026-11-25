"""DeDoDe's networks on Apple MLX, written in `mlx.core` operations.

The VGG encoders, the `ConvRefiner` decoders, the detector's multi-scale logits and the descriptor sampled at
keypoints (`sparse.sample`'s maths).

Channels last. The eval BatchNorms are folded into the convolutions (float64 on the host, stored once in the run
dtype). In float16 the dense 3 x 3 convolutions that MLX would run as its own Winograd transform (17x the error of the
direct convolution) run as `conv_mlx`'s F(2,3) Winograd instead (about 2x the direct error); the depthwise 5 x 5
convolutions are `graph_mlx`'s kernel. The detector's logits are float32 throughout (their output channel too) and
resized bicubic between scales as upstream; the description values are sampled and blended in float32.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fastforks._core.mlx import mx, to_mx
from fastforks.kernels.conv_mlx import ops as conv_mlx
from fastforks.kernels.graph_mlx.ops import depthwise
from fastforks.shared.mlx_graph import _mlx_winograd
from fastforks.shared.mlx_vit import dinov2_layers, dinov2_params, dinov2_table

from .net.dedode import VGG, Decoder, Descriptor, Detector, VGGDINOv2

__all__ = ["MlxDescriptor", "MlxDetector", "resize"]

WINO_M = 2  # F(2,3): the float16 detector logits pick the keypoints; F(4,3) costs recall
PATCH, HEADS, DEPTH = 14, 16, 24  # DINOv2 ViT-L/14


def _folded(conv: Any, bn: Any) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - torch modules
    """`conv` followed by the eval `bn`: weight `[O, I/g, kh, kw]` and bias `[O]` (float64)."""
    w = conv.weight.detach().double().numpy()
    b = conv.bias.detach().double().numpy() if conv.bias is not None else np.zeros(w.shape[0])
    if bn is None:
        return w, b
    s = bn.weight.detach().double().numpy() / np.sqrt(bn.running_var.detach().double().numpy() + bn.eps)
    mean, shift = bn.running_mean.detach().double().numpy(), bn.bias.detach().double().numpy()
    return w * s[:, None, None, None], (b - mean) * s + shift


def _arr(a: np.ndarray, dtype: mx.Dtype) -> mx.array:
    out = mx.array(np.ascontiguousarray(a, dtype=np.float32)).astype(dtype)
    mx.eval(out)
    return out


class _Conv3:
    """A dense 3 x 3 conv (padding 1, stride 1) with a folded BatchNorm and ReLU."""

    def __init__(self, conv: Any, bn: Any, dtype: mx.Dtype) -> None:  # noqa: ANN401 - torch modules
        w, b = _folded(conv, bn)
        ohwi = w.transpose(0, 2, 3, 1)
        self.cout = ohwi.shape[0]
        self.w, self.b = _arr(ohwi, dtype), _arr(b, dtype)
        self.u = conv_mlx.weights(mx.array(ohwi.astype(np.float32)), WINO_M, dtype) if dtype == mx.float16 else None
        self.w32 = None  # float32 weights for a float16 Winograd-shaped conv `conv_mlx` declines (built on demand)

    def __call__(self, x: mx.array) -> mx.array:
        if self.u is not None and conv_mlx.supported(x.shape, self.cout):
            return conv_mlx.conv3x3(x, self.u, self.b, m=WINO_M, act=1)
        positions = x.shape[1] * x.shape[2]
        if x.dtype == mx.float16 and _mlx_winograd(self.w.shape, self.cout, (1, 1), (1, 1), 1, positions):
            if self.w32 is None:
                self.w32 = self.w.astype(mx.float32), self.b.astype(mx.float32)
            w, b = self.w32
            return mx.maximum(mx.conv2d(x.astype(mx.float32), w, padding=1) + b, 0).astype(x.dtype)
        return mx.maximum(mx.conv2d(x, self.w, padding=1) + self.b, 0)


def _pool(x: mx.array) -> mx.array:
    """`nn.MaxPool2d(2, 2)` (floor)."""
    _, h, w, c = x.shape
    x = x[:, : h // 2 * 2, : w // 2 * 2]
    return x.reshape(1, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


class _Vgg:
    """`VGG`: the features before each max-pool (channels last) and their sizes."""

    def __init__(self, vgg: VGG, dtype: mx.Dtype) -> None:
        layers, self.ops = list(vgg.layers), []
        for i, layer in enumerate(layers):
            if isinstance(layer, torch.nn.MaxPool2d):
                self.ops.append(None)
            elif isinstance(layer, torch.nn.Conv2d):
                self.ops.append(_Conv3(layer, layers[i + 1], dtype))

    def __call__(self, x: mx.array) -> list[mx.array]:
        feats = []
        pools = self.ops.count(None)
        for op in self.ops:
            if op is None:
                mx.eval(x)  # one stage's temporaries at a time
                feats.append(x)
                if len(feats) == pools:
                    break  # the last max-pool's output is not a feature
                x = _pool(x)
            else:
                x = op(x)
        return feats


class _Refiner:
    """`ConvRefiner` (BatchNorms folded): `[1, H, W, in]` -> its 1 x 1 output, split at `split` channels."""

    def __init__(self, m: Any, split: int, dtype: mx.Dtype, *, fp32_head: bool) -> None:  # noqa: ANN401 - a `ConvRefiner`
        self.dtype = dtype
        b1 = m.block1
        w, b = _folded(b1[0], b1[1])  # 1 x 1
        self.b1 = (_arr(w[:, :, 0, 0].T, dtype), _arr(b, dtype), *self._lin(b1[3], dtype))
        self.hidden = []
        for blk in m.hidden_blocks:
            w, b = _folded(blk[0], blk[1])  # depthwise 5 x 5: [C, 1, 5, 5]
            self.hidden.append((_arr(w[:, 0].transpose(1, 2, 0), dtype), _arr(b, dtype), *self._lin(blk[3], dtype)))
        w, b = self._lin(m.out_conv, mx.float32)
        self.split = split
        head = mx.float32 if fp32_head else dtype
        self.head = (w[:, :split].astype(head), b[:split].astype(head))  # the prototypes (logits / descriptions)
        self.rest = (w[:, split:].astype(dtype), b[split:].astype(dtype))  # the context

    @staticmethod
    def _lin(conv: Any, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:  # noqa: ANN401 - a 1 x 1 `nn.Conv2d`
        w = conv.weight.detach().double().numpy()[:, :, 0, 0]
        return _arr(w.T, dtype), _arr(conv.bias.detach().double().numpy(), dtype)

    def hidden_map(self, x: mx.array) -> mx.array:
        """`(hidden_blocks(block1(x)) + block1(x)) / 1.4`, the input of `out_conv`."""
        w1, b1, w2, b2 = self.b1
        x0 = mx.maximum(x @ w1 + b1, 0) @ w2 + b2
        h = x0
        for wd, bd, wp, bp in self.hidden:
            h = mx.maximum(depthwise(h, wd, bd, (1, 1), (2, 2), (1, 1)), 0) @ wp + bp
        return (h + x0) / 1.4

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        h = self.hidden_map(x)
        w, b = self.head
        return h.astype(w.dtype) @ w + b, h @ self.rest[0] + self.rest[1]


def _taps(n_in: int, n_out: int, *, cubic: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """`F.interpolate(align_corners=False)` along one axis: (input index, weight) per tap, as torch computes them."""
    scale = np.float32(n_in / n_out)
    real = scale * (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) - np.float32(0.5)
    if not cubic:
        real = np.maximum(real, np.float32(0))
        i0 = np.floor(real).astype(np.int32)
        t = (real - i0).astype(np.float32)
        return [(np.minimum(i0, n_in - 1), 1 - t), (np.minimum(i0 + 1, n_in - 1), t)]
    i0 = np.floor(real).astype(np.int32)
    t = (real - i0).astype(np.float32)
    a = np.float32(-0.75)

    def c1(x: np.ndarray) -> np.ndarray:
        return ((a + 2) * x - (a + 3)) * x * x + 1

    def c2(x: np.ndarray) -> np.ndarray:
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

    weights = [c2(t + 1), c1(t), c1(1 - t), c2(2 - t)]
    return [(np.clip(i0 + k - 1, 0, n_in - 1), wk.astype(np.float32)) for k, wk in enumerate(weights)]


def resize(x: mx.array, size: tuple[int, int], *, cubic: bool = False) -> mx.array:
    """`F.interpolate(mode="bilinear" | "bicubic", align_corners=False)` of `[1, H, W, C]` to `size`, in x's dtype."""
    for axis, n in ((1, size[0]), (2, size[1])):
        if x.shape[axis] == n:
            continue
        shape = [1, 1, 1, 1]
        shape[axis] = n
        terms = [
            mx.take(x, mx.array(idx), axis=axis) * mx.array(wt.reshape(shape)).astype(x.dtype)
            for idx, wt in _taps(x.shape[axis], n, cubic=cubic)
        ]
        x = sum(terms[1:], terms[0])
    return x


class MlxDetector:
    """`Detector` (DeDoDe-L or DaD's S): a normalised `[1, H, W, 3]` float32 image -> float32 logits `[H, W]`."""

    def __init__(self, net: Detector, dtype: str) -> None:
        """Copy the weights of `net` (eval, float32) to MLX in `dtype` (the name)."""
        self.dtype = getattr(mx, dtype)
        self.encoder = _Vgg(net.encoder, self.dtype)
        dec: Decoder = net.decoder
        self.layers = [_Refiner(m, dec.num_prototypes, self.dtype, fp32_head=True) for m in dec.layers.values()]

    def __call__(self, x: mx.array) -> mx.array:
        """The logits of `x`, each scale evaluated in turn."""
        feats = self.encoder(x.astype(self.dtype))
        logits = context = None
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            f = feats.pop()
            delta, context = layer(f if context is None else mx.concatenate([f, context], axis=-1))
            logits = delta if logits is None else logits + delta
            if i < last:
                size = feats[-1].shape[1:3]
                logits = resize(logits, size, cubic=True)
                context = resize(context.astype(mx.float32), size).astype(self.dtype)
            mx.eval(logits, context)
        assert logits is not None  # noqa: S101 - every detector has scales
        return logits[0, ..., 0]


class MlxDescriptor:
    """`Descriptor` (DeDoDe-B / -G, LoMa's small ones) sampled at keypoints.

    A `[1, S, S, 3]` float32 image and normalised keypoints `[N, 2]` -> their float32 descriptions
    `[N, num_prototypes]` (`sparse.sample`: the finest scale's projection only at the four pixels around each
    keypoint).
    """

    def __init__(self, net: Descriptor, dtype: str) -> None:
        """Copy the weights of `net` (eval, float32; its DINOv2 too) to MLX in `dtype` (the name)."""
        self.dtype = getattr(mx, dtype)
        enc = net.encoder
        self.dino = None
        if isinstance(enc, VGGDINOv2):
            state = {k: v.detach().float() for k, v in enc.dinov2[0].state_dict().items()}
            p = dinov2_params(state, PATCH)
            host = {k: torch.from_numpy(np.array(p.pop(k))) for k in ("pos.cls", "pos.grid", "patch.bias")}
            self.dino = {}
            for k in list(p):  # cast one by one: one float32 tensor beside the run-dtype copy
                a = p.pop(k).astype(self.dtype)
                mx.eval(a)
                self.dino[k] = a
            self._host, self._tables = host, {}
            enc = enc.vgg
        self.encoder = _Vgg(enc, self.dtype)
        dec: Decoder = net.decoder
        self.protos = dec.num_prototypes
        self.layers = [_Refiner(m, self.protos, self.dtype, fp32_head=False) for m in dec.layers.values()]

    def _tokens(self, x: mx.array) -> mx.array:
        """The DINOv2's normalised patch tokens as a `[1, h/14, w/14, 1024]` map."""
        assert self.dino is not None  # noqa: S101 - only with a DINOv2
        h0, w0 = x.shape[1] // PATCH, x.shape[2] // PATCH
        if (h0, w0) not in self._tables:
            h = self._host
            self._tables[h0, w0] = to_mx(dinov2_table(h["pos.cls"], h["pos.grid"], h["patch.bias"], h0, w0))
        table = self._tables[h0, w0].astype(self.dtype)
        tokens = dinov2_layers(self.dino, HEADS, x.transpose(0, 3, 1, 2), table, (DEPTH,))[0][:, 1:]
        return tokens.reshape(1, h0, w0, -1)

    def __call__(self, x: mx.array, keypoints: mx.array) -> mx.array:
        """The descriptions of `keypoints` in the image `x`."""
        x = x.astype(self.dtype)
        feats = self.encoder(x)
        if self.dino is not None:
            feats.append(self._tokens(x))
            mx.eval(feats[-1])
        prior = context = None
        n = len(self.layers)
        for i, layer in enumerate(self.layers[:-1]):
            f = feats.pop()
            delta, context = layer(f if context is None else mx.concatenate([f, context], axis=-1))
            prior = delta if prior is None else delta + prior
            size = feats[-1].shape[1:3]
            context = resize(context, size)
            if i < n - 2:
                prior = resize(prior, size)
            mx.eval(prior, context)
        assert prior is not None  # noqa: S101 - multi-scale decoders only
        assert context is not None  # noqa: S101
        f = feats.pop()
        hidden = self.layers[-1].hidden_map(mx.concatenate([f, context], axis=-1))
        return self._sample(hidden, prior, keypoints)

    def _sample(self, hidden: mx.array, prior: mx.array, keypoints: mx.array) -> mx.array:
        """`sparse.sample` from the finest map's `out_conv` input: the four neighbours, blended with the prior."""
        _, h, w, c = hidden.shape
        kp = keypoints.astype(mx.float32)
        x = ((kp[:, 0] + 1) * w - 1) / 2
        y = ((kp[:, 1] + 1) * h - 1) / 2
        x0, y0 = mx.floor(x), mx.floor(y)
        wx, wy = x - x0, y - y0
        ix = mx.stack([x0, x0 + 1, x0, x0 + 1])
        iy = mx.stack([y0, y0, y0 + 1, y0 + 1])
        weights = mx.stack([(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy])
        weights = mx.where((ix < 0) | (ix >= w) | (iy < 0) | (iy >= h), 0, weights)
        ix, iy = mx.clip(ix, 0, w - 1), mx.clip(iy, 0, h - 1)
        points = hidden.reshape(h * w, c)[(iy * w + ix).astype(mx.int32)]  # [4, N, C]
        wo, bo = self.layers[-1].head
        delta = points @ wo + bo
        # the prior resized bilinear (border replicated) at the same pixel centres
        _, ch, cw, cp = prior.shape
        cx = mx.maximum((ix + 0.5) * (cw / w) - 0.5, 0)
        cy = mx.maximum((iy + 0.5) * (ch / h) - 0.5, 0)
        left, top = mx.floor(cx), mx.floor(cy)
        fx, fy = (cx - left)[..., None], (cy - top)[..., None]
        right, bottom = mx.minimum(left + 1, cw - 1), mx.minimum(top + 1, ch - 1)
        flat = prior.reshape(ch * cw, cp)

        def at(r: mx.array, col: mx.array) -> mx.array:
            return flat[(r * cw + col).astype(mx.int32)].astype(mx.float32)

        upper = (1 - fx) * at(top, left) + fx * at(top, right)
        lower = (1 - fx) * at(bottom, left) + fx * at(bottom, right)
        resized = ((1 - fy) * upper + fy * lower).astype(prior.dtype)
        values = (delta + resized).astype(mx.float32)
        return (values * weights[..., None]).sum(axis=0)
