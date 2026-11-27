"""DETR (ResNet-50 / 101 + encoder-decoder transformer) on Apple MLX, hand-written: NHWC, the run dtype.

The network of `model.Portable._logits`: the normalised image `[1, 3, h, w]` float32 -> (class logits `[1, Q, C + 1]`,
sigmoid boxes `[1, Q, 4]`), both float32. The ResNet convs (FrozenBatchNorm folded) dispatch as the graph translator's
(`fastforks.shared.cspnext.mlx_net.conv`, `graph="portable"`); bias, ReLU and residual fuse into one compiled pass per
block. The stem, each ResNet bottleneck, the projection and each encoder layer are a step, the decoder another. The
transformer runs in the run dtype as the PyTorch network: post-norm layers (`mx.fast.layer_norm`), the queries and
keys of a self-attention as one GEMM over `x + pos`, MLX's fused attention; the class and box heads in float32 (as
the float16 PyTorch runner keeps them). The sine position table is made once per feature size.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.shared.cspnext import mlx_net as csp
from fastforks.shared.swin_sr import mlx_net as swin

from . import net as tnet

if TYPE_CHECKING:
    from collections.abc import Callable

    Layer = Callable[[mx.array], mx.array]

LAYOUT = "detr-v1"
SOURCES = (Path(__file__), Path(tnet.__file__), Path(csp.__file__), Path(swin.__file__))
GRAPH = "portable"  # the translator's conv dispatch (`csp.conv`)


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def max_pool(x: mx.array) -> mx.array:
    """`F.max_pool2d(x, 3, 2, 1)` on NHWC: maxima of the strided slices of the -inf-padded map."""
    n, h, w, c = x.shape
    oh, ow = (h - 1) // 2 + 1, (w - 1) // 2 + 1
    y = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], constant_values=-math.inf)
    rows = y[:, 0 : 2 * oh - 1 : 2]
    for i in (1, 2):
        rows = mx.maximum(rows, y[:, i : i + 2 * oh - 1 : 2])
    out = rows[:, :, 0 : 2 * ow - 1 : 2]
    for j in (1, 2):
        out = mx.maximum(out, rows[:, :, j : j + 2 * ow - 1 : 2])
    return out.reshape(n, oh, ow, c)


def sine_position(h: int, w: int, d: int) -> np.ndarray:
    """`net.sine_position` in numpy (float64): `[h * w, d]`, rows first."""
    half, scale = d // 2, 2 * math.pi
    y = np.arange(1, h + 1, dtype=np.float32) / np.float32(h + 1e-6) * np.float32(scale)
    x = np.arange(1, w + 1, dtype=np.float32) / np.float32(w + 1e-6) * np.float32(scale)
    dim_t = np.arange(half, dtype=np.float32)
    dim_t = np.float32(tnet.POS_TEMPERATURE) ** (2 * (dim_t // 2) / np.float32(half))
    py, px = y[:, None] / dim_t, x[:, None] / dim_t
    py = np.stack((np.sin(py[:, 0::2]), np.cos(py[:, 1::2])), axis=2).reshape(h, half)
    px = np.stack((np.sin(px[:, 0::2]), np.cos(px[:, 1::2])), axis=2).reshape(w, half)
    pos = np.concatenate((np.broadcast_to(py[:, None], (h, w, half)), np.broadcast_to(px[None], (h, w, half))), 2)
    return pos.reshape(h * w, d)


def _block(blk: Any, dtype: mx.Dtype) -> Layer:
    """The torchvision `Bottleneck`: `relu(conv3(relu(conv2(relu(conv1 x)))) + shortcut x)`."""
    c1, c2, c3 = (csp.conv(getattr(blk, n), dtype, GRAPH) for n in ("conv1", "conv2", "conv3"))
    down = None if blk.downsample is None else csp.conv(blk.downsample[0], dtype, GRAPH)

    def apply(x: mx.array) -> mx.array:
        return _relu(c3(_relu(c2(_relu(c1(x))))) + (x if down is None else down(x)))

    return apply


class _Attention:
    """`detection._attention.Attention` (`nn.MultiheadAttention` names), the in-projection split as `[in, out]`."""

    def __init__(self, att: Any, dtype: mx.Dtype, heads: int, *, fused_qk: bool) -> None:
        w, b = att.in_proj_weight, att.in_proj_bias
        d = w.shape[1]
        self.heads, self.scale = heads, (d // heads) ** -0.5
        if fused_qk:  # self-attention: q and k read the same input
            self.qk = (swin.arr(w[: 2 * d].t(), dtype), swin.arr(b[: 2 * d], dtype))
        else:
            self.q = (swin.arr(w[:d].t(), dtype), swin.arr(b[:d], dtype))
            self.k = (swin.arr(w[d : 2 * d].t(), dtype), swin.arr(b[d : 2 * d], dtype))
        self.v = (swin.arr(w[2 * d :].t(), dtype), swin.arr(b[2 * d :], dtype))
        self.out = swin.linear(att.out_proj, dtype)

    def _split(self, t: mx.array) -> mx.array:
        b, s, _ = t.shape
        return t.reshape(b, s, self.heads, -1).transpose(0, 2, 1, 3)

    def attend(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        b, s, _ = q.shape
        o = mx.fast.scaled_dot_product_attention(self._split(q), self._split(k), self._split(v), scale=self.scale)
        return swin.dense(o.transpose(0, 2, 1, 3).reshape(b, s, -1), *self.out)

    def self_attn(self, qk_in: mx.array, x: mx.array) -> mx.array:
        q, k = mx.split(swin.dense(qk_in, *self.qk), 2, axis=-1)
        return self.attend(q, k, swin.dense(x, *self.v))

    def cross(self, q_in: mx.array, k_in: mx.array, v_in: mx.array) -> mx.array:
        return self.attend(swin.dense(q_in, *self.q), swin.dense(k_in, *self.k), swin.dense(v_in, *self.v))


def _norm(layer: Any, dtype: mx.Dtype) -> Layer:
    w, b = swin.arr(layer.weight, dtype), swin.arr(layer.bias, dtype)
    return lambda x: mx.fast.layer_norm(x, w, b, tnet.LN_EPS)


def _ffn(layer: Any, dtype: mx.Dtype) -> Layer:
    l1, l2 = swin.linear(layer.linear1, dtype), swin.linear(layer.linear2, dtype)
    return lambda x: swin.dense(_relu(swin.dense(x, *l1)), *l2)


class Network:
    """`Detr.forward`: normalised `[1, 3, h, w]` float32 -> (logits, boxes) float32."""

    def __init__(self, net: Any, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        self.dtype = dt
        bb = net.backbone
        c1 = csp.conv(bb.conv1, dt, GRAPH)
        self.stem = csp.stepped(lambda x: max_pool(_relu(c1(x.transpose(0, 2, 3, 1).astype(dt)))))
        self.stages = [  # a step per bottleneck (one per stage held float32 R101's 23-block stage 3 in one buffer)
            csp.stepped(_block(b, dt)) for layer in (bb.layer1, bb.layer2, bb.layer3, bb.layer4) for b in layer
        ]
        self.proj = csp.conv(net.input_proj, dt, GRAPH)
        tr = net.transformer
        self.d = int(net.query_embed.weight.shape[1])
        heads = int(tr.decoder.layers[0].self_attn.heads)
        self.enc = [
            (_Attention(e.self_attn, dt, heads, fused_qk=True), _norm(e.norm1, dt), _ffn(e, dt), _norm(e.norm2, dt))
            for e in tr.encoder.layers
        ]
        self.dec = [
            (
                _Attention(d.self_attn, dt, heads, fused_qk=True),
                _Attention(d.multihead_attn, dt, heads, fused_qk=False),
                _norm(d.norm1, dt),
                _norm(d.norm2, dt),
                _ffn(d, dt),
                _norm(d.norm3, dt),
            )
            for d in tr.decoder.layers
        ]
        self.qpos = swin.arr(net.query_embed.weight, dt)
        self.final = _norm(tr.decoder.norm, dt)
        self.cls = swin.linear(net.class_embed, mx.float32)
        self.box = [swin.linear(layer, mx.float32) for layer in net.bbox_embed.layers]
        self.pos: dict[tuple[int, int], mx.array] = {}
        self.project = csp.stepped(lambda f: (src := self.proj(f)).reshape(src.shape[0], -1, self.d))
        self.encode = [csp.stepped(self._encoder(*e)) for e in self.enc]
        self.head = csp.stepped(self._decoder)

    @staticmethod
    def _encoder(att: _Attention, n1: Layer, ffn: Layer, n2: Layer) -> Callable[[mx.array, mx.array], mx.array]:
        def apply(mem: mx.array, pos: mx.array) -> mx.array:
            mem = n1(mem + att.self_attn(mem + pos, mem))
            return n2(mem + ffn(mem))

        return apply

    def _decoder(self, mem: mx.array, pos: mx.array) -> tuple[mx.array, mx.array]:
        """The encoder's memory `[1, h w, d]` -> (logits, boxes)."""
        b = mem.shape[0]
        qpos = mx.broadcast_to(self.qpos[None], (b, *self.qpos.shape))
        out = mx.zeros_like(qpos)
        kpos = mem + pos
        for sa, ca, n1, n2, ffn, n3 in self.dec:
            out = n1(out + sa.self_attn(out + qpos, out))
            out = n2(out + ca.cross(out + qpos, kpos, mem))
            out = n3(out + ffn(out))
        out = self.final(out).astype(mx.float32)
        logits = swin.dense(out, *self.cls)
        y = out
        for i, (w, bias) in enumerate(self.box):
            y = swin.dense(y, w, bias)
            if i < len(self.box) - 1:
                y = _relu(y)
        return logits, mx.sigmoid(y)

    def _pos(self, h: int, w: int) -> mx.array:
        if (h, w) not in self.pos:
            p = mx.array(sine_position(h, w, self.d)).astype(self.dtype)[None]
            mx.eval(p)
            self.pos[h, w] = p
        return self.pos[h, w]

    def forward(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """The network on `[1, 3, h, w]` float32 (the steps)."""
        y = self.stem(x)
        for s in self.stages:
            y = s(y)
        pos = self._pos(y.shape[1], y.shape[2])
        mem = self.project(y)
        for e in self.encode:
            mem = e(mem, pos)
        return self.head(mem, pos)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """`forward` with its steps, evaluated."""
        csp.options(GRAPH)
        out = csp.steps(self.forward, x)
        mx.eval(out)
        return out


def network(net: Any, dtype: str) -> Network:
    """The MLX network of a `Detr` (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype)
