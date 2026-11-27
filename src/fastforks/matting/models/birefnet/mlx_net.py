"""BiRefNet on Apple MLX, hand-written: NHWC, the run dtype.

`BiRefNet.forward` up to its last prediction, through a sigmoid in float32:

- the Swin v1 encoder (`shared.mlx_swin_net`: LayerNorm + window shuffles as gathers, the shared window attention
  kernel) on the image and on the image at half size (bilinear, `align_corners=True`), each layer one step;
- the encoder merge (concatenations of the `align_corners` resizes), the squeeze block and the four decoder blocks:
  convolutions with their BatchNorms folded (`shared.mlx_rsu`); the deformable ASPP as the offset and modulator convs
  of each branch as one conv, the bilinear sampling of the taps (`graph_mlx.ops.grid_sample`, float32 coordinates),
  the modulated columns times the deformable conv's weight (BatchNorm folded, ReLU), each branch's 256 channels
  straight into its share of the 1 x 1 fusion conv (no 1280-channel concatenation), in row chunks of at most
  `CHUNK_BYTES` of columns, each a step; the global pooling branch as a bias of the fusion;
- the image patches of each decoder level (a reshape of the NHWC image), the gradient attention, the laterals;
- the last projection commuted with the final resize (as the CUDA fp16 path of `net/birefnet.py`): `_p1`'s part
  projected to one channel at a quarter of the input, then resized; the image branch's two linear convs end in a
  single-channel 3 x 3 conv (the projection folded into the second), run in row strips.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_rsu, mlx_swin_net
from fastforks.shared.cspnext import mlx_net as csp

from .net import birefnet as tnet

if TYPE_CHECKING:
    from collections.abc import Callable

    Layer = Callable[[mx.array], mx.array]

LAYOUT = "birefnet-v1"
SOURCES = (Path(__file__), Path(tnet.__file__), Path(mlx_rsu.__file__), *mlx_swin_net.SOURCES)
CHUNK_BYTES = 64 << 20  # the sampled columns of one deformable ASPP step
STRIP = 256  # rows of the image branch per step


_np, _arr = mlx_swin_net.np64, mlx_swin_net.arr


def _bn(bn: Any) -> tuple[np.ndarray, np.ndarray]:
    """An eval `BatchNorm2d` as `y = x * s + t` (float64)."""
    s = _np(bn.weight) / np.sqrt(_np(bn.running_var) + bn.eps)
    return s, _np(bn.bias) - _np(bn.running_mean) * s


def _conv(layer: Any, bn: Any | None, dtype: mx.Dtype) -> Layer:
    return mlx_rsu.conv(layer, bn, dtype)


def _up(x: mx.array, hw: tuple[int, int]) -> mx.array:
    """`F.interpolate(mode="bilinear", align_corners=True)` on NHWC."""
    return mlx_rsu.up_like(x, hw, align=True)


class _Deform:
    """`_ASPPModuleDeformable`: ReLU(BatchNorm(modulated deformable conv)), its offsets and modulator one conv.

    `cols(x, om, ys)` -> the branch's `[rows * W, 256]` for the output rows `ys` (`om`: their offset / modulator map).
    """

    def __init__(self, m: Any, dtype: mx.Dtype) -> None:
        dc = m.atrous_conv
        oc, mc, rc = dc.offset_conv, dc.modulator_conv, dc.regular_conv
        o, c, kh, kw = (int(d) for d in rc.weight.shape)
        self.k, self.pad, self.dtype = kh * kw, kh // 2, dtype
        k = self.k

        def om() -> tuple[np.ndarray, np.ndarray]:
            return np.concatenate([_np(oc.weight), _np(mc.weight)]), np.concatenate([_np(oc.bias), _np(mc.bias)])

        self.om = mlx_rsu.conv_of((3 * k, c, kh, kw), om, dtype, padding=(self.pad, self.pad))

        def weight() -> np.ndarray:  # rows (tap, channel): the sampled columns' order
            s, _ = _bn(m.bn)
            return (_np(rc.weight) * s[:, None, None, None]).transpose(2, 3, 1, 0).reshape(k * c, o)

        self.w = _arr((k * c, o), dtype, weight)
        self.b = _arr((o,), dtype, lambda: _bn(m.bn)[1])
        tap = np.arange(k)
        self.ty = mx.array((tap // kw - self.pad).astype(np.float32))
        self.tx = mx.array((tap % kw - self.pad).astype(np.float32))

    def cols(self, x: mx.array, om: mx.array, ys: mx.array) -> mx.array:
        _, h, w, c = x.shape
        k, rows = self.k, om.shape[1]
        om = om.astype(mx.float32)
        off = om[..., : 2 * k].reshape(rows, w, k, 2)  # (dy, dx) per tap
        mask = (2.0 * mx.sigmoid(om[..., 2 * k :])).reshape(rows * w, k, 1).astype(x.dtype)
        px = mx.arange(w, dtype=mx.float32)[None, :, None] + self.tx  # [1, W, k]
        py = ys[:, None, None] + self.ty  # [rows, 1, k]
        gx = (px + off[..., 1]) * (2.0 / max(w - 1, 1)) - 1.0
        gy = (py + off[..., 0]) * (2.0 / max(h - 1, 1)) - 1.0
        grid = mx.stack([gx, gy], axis=-1).reshape(1, rows * w, k, 2)
        s = ops.grid_sample(x, grid, align_corners=True, border=False)[0]  # [rows * W, k, C]
        return mx.maximum(mx.addmm(self.b, (s * mask).reshape(rows * w, k * c), self.w), 0)


class _Aspp:
    """`ASPPDeformable` (dropout: inference identity) on NHWC `[1, H, W, 64]`."""

    def __init__(self, m: Any, dtype: mx.Dtype) -> None:
        self.dtype = dtype
        self.branches = [_Deform(b, dtype) for b in (m.aspp1, *m.aspp_deforms)]
        pool_conv, pool_bn = m.global_avg_pool[1], m.global_avg_pool[2]
        mid, c = (int(d) for d in pool_conv.weight.shape[:2])
        o = int(m.conv1.weight.shape[0])
        n = len(self.branches)

        def pool() -> np.ndarray:
            s, _ = _bn(pool_bn)
            return (_np(pool_conv.weight)[:, :, 0, 0] * s[:, None]).T

        self.pool = (_arr((c, mid), mx.float32, pool), _arr((mid,), mx.float32, lambda: _bn(pool_bn)[1]))

        def fuse() -> np.ndarray:  # [n + 1, mid, out]: conv1 per concatenated part, bn1 folded
            s, _ = _bn(m.bn1)
            return (_np(m.conv1.weight)[:, :, 0, 0] * s[:, None]).T.reshape(n + 1, mid, o)

        fused = _arr((n + 1, mid, o), dtype, fuse)
        self.fuse = [fused[i] for i in range(n)]  # the pooling branch's part: `fuse_pool`, float32
        self.fuse_pool = _arr((mid, o), mx.float32, lambda: fuse()[n])
        self.bias = _arr((o,), mx.float32, lambda: _bn(m.bn1)[1])
        self.head = csp.stepped(self._head)
        self.chunk = csp.stepped(self._chunk)

    def _head(self, x: mx.array) -> tuple[mx.array, ...]:
        """The offset / modulator maps of the branches and the fusion's bias (with the pooling branch)."""
        v = x.astype(mx.float32).mean(axis=(0, 1, 2))[None]
        v = mx.maximum(mx.addmm(self.pool[1], v, self.pool[0]), 0)
        bias = mx.addmm(self.bias, v, self.fuse_pool)[0]
        return (bias, *(b.om(x) for b in self.branches))

    def _chunk(self, x: mx.array, bias: mx.array, ys: mx.array, *oms: mx.array) -> mx.array:
        y = bias
        for b, om, wf in zip(self.branches, oms, self.fuse, strict=True):
            y = y + b.cols(x, om, ys) @ wf
        return mx.maximum(y, 0).astype(self.dtype).reshape(oms[0].shape[1], x.shape[2], -1)

    def rows(self, h: int, w: int, c: int) -> int:
        """Output rows per chunk: the widest branch's columns within `CHUNK_BYTES`, equal chunks."""
        k = max(b.k for b in self.branches)
        fit = max(1, CHUNK_BYTES // (w * k * c * self.dtype.size))
        n = -(-h // fit)
        while h % n:
            n += 1
        return h // n

    def __call__(self, x: mx.array) -> mx.array:
        _, h, w, c = x.shape
        bias, *oms = self.head(x)
        r = self.rows(h, w, c)
        ys = mx.arange(h, dtype=mx.float32)
        parts = [self.chunk(x, bias, ys[r0 : r0 + r], *(om[:, r0 : r0 + r] for om in oms)) for r0 in range(0, h, r)]
        return (parts[0] if len(parts) == 1 else mx.concatenate(parts))[None]


class _DecBlk:
    """`BasicDecBlk`: conv_in + BN + ReLU, the deformable ASPP, conv_out + BN."""

    def __init__(self, m: Any, dtype: mx.Dtype) -> None:
        cin = _conv(m.conv_in, m.bn_in, dtype)
        cout = _conv(m.conv_out, m.bn_out, dtype)
        self.aspp = _Aspp(m.dec_att, dtype)
        self.cin = csp.stepped(lambda x: mx.maximum(cin(x), 0))
        self.cout = csp.stepped(cout)

    def __call__(self, x: mx.array) -> mx.array:
        return self.cout(self.aspp(self.cin(x)))


def _grid(x: mx.array, ref: mx.array) -> mx.array:
    """`image2patches(x, patch_ref=ref)` (its resize to `ref`'s size is the identity)."""
    return _patches(x, x.shape[1] // ref.shape[1])


def _patches(x: mx.array, g: int) -> mx.array:
    """`image2patches` of NHWC `x` on a `g x g` grid: `[1, H / g, W / g, 3 g g]`, channels (c, hg, wg)."""
    _, h, w, c = x.shape
    t = x.reshape(1, g, h // g, g, w // g, c).transpose(0, 2, 4, 5, 1, 3)
    return t.reshape(1, h // g, w // g, c * g * g)


class Network:
    """`sigmoid(BiRefNet.forward(x)[-1])`: normalised `[1, 3, S, S]` float32 -> `[1, 1, S, S]` float32."""

    def __init__(self, net: Any, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = self.dtype = getattr(mx, dtype)
        self.bb = mlx_swin_net.Encoder(net.bb, dt)
        self.squeeze = [_DecBlk(m, dt) for m in net.squeeze_module]
        d = net.decoder
        self.dec = [_DecBlk(getattr(d, f"decoder_block{i}"), dt) for i in (4, 3, 2, 1)]
        self.lat = [_conv(getattr(d, f"lateral_block{i}").conv, None, dt) for i in (4, 3, 2)]
        self.ipt = [self._simple(getattr(d, f"ipt_blk{i}"), dt) for i in (5, 4, 3, 2)]
        gdt = []
        for i in (4, 3, 2):
            seq = getattr(d, f"gdt_convs_{i}")
            conv, attn = _conv(seq[0], seq[1], dt), _conv(getattr(d, f"gdt_convs_attn_{i}")[0], None, dt)
            gdt.append(lambda p, conv=conv, attn=attn: p * mx.sigmoid(attn(mx.maximum(conv(p), 0))))
        self.attend = [csp.stepped(f) for f in gdt]
        self._final(d, dt)
        self.enc = csp.stepped(self._encode)
        self.inputs = csp.stepped(self._inputs)
        self.top = csp.stepped(lambda x4, xf: mx.concatenate([x4, self.ipt[0](_grid(xf, x4))], axis=-1))
        self.lateral = [csp.stepped(lambda p, lx, xf, j=j: self._lateral(p, lx, xf, j)) for j in range(3)]
        self.out = csp.stepped(self._out)

    @staticmethod
    def _simple(m: Any, dtype: mx.Dtype) -> Layer:
        a, b = _conv(m.conv1, None, dtype), _conv(m.conv_out, None, dtype)
        return lambda x: b(a(x))

    def _final(self, d: Any, dtype: mx.Dtype) -> None:
        """`conv_out1` split: `_p1`'s columns, and the image branch's second conv with the rest folded in."""
        head, conv_out = d.conv_out1[0], d.ipt_blk1.conv_out
        c = int(head.weight.shape[1]) - int(conv_out.weight.shape[0])
        mid = int(conv_out.weight.shape[1])
        self.head = mlx_rsu.conv_of((1, c, 1, 1), lambda: (_np(head.weight)[:, :c], np.zeros(1)), dtype)

        def tail() -> tuple[np.ndarray, np.ndarray]:
            w2 = _np(head.weight)[0, c:, 0, 0]  # [48]
            w = np.einsum("o,oihw->ihw", w2, _np(conv_out.weight))[None]
            return w, np.array([w2 @ _np(conv_out.bias) + _np(head.bias)[0]])

        self.tail = mlx_rsu.conv_of((1, mid, 3, 3), tail, dtype, padding=(1, 1))
        conv1 = _conv(d.ipt_blk1.conv1, None, dtype)
        self.whole = csp.stepped(lambda x: self.tail(conv1(x)))
        self.strip = csp.stepped(lambda x: self.tail(conv1(x))[:, 2:-2])
        self.strip_top = csp.stepped(lambda x: self.tail(conv1(x))[:, :-2])
        self.strip_bottom = csp.stepped(lambda x: self.tail(conv1(x))[:, 2:])

    def _encode(self, x: mx.array) -> mx.array:
        """The image NHWC in the run dtype and its half-size resize."""
        x = x.transpose(0, 2, 3, 1).astype(self.dtype)
        return x, _up(x, (x.shape[1] // 2, x.shape[2] // 2))

    def _inputs(self, f: list[mx.array], g: list[mx.array]) -> tuple[mx.array, ...]:
        """`forward_enc`'s merge: each level with the half-size one resized, `x4` with the other levels."""
        xs = [mx.concatenate([a, _up(b, a.shape[1:3])], axis=-1) for a, b in zip(f, g, strict=True)]
        hw = xs[3].shape[1:3]
        x4 = mx.concatenate([*(_up(a, hw) for a in xs[:3]), xs[3]], axis=-1)
        return (*xs[:3], x4)

    def _lateral(self, p: mx.array, lx: mx.array, xf: mx.array, j: int) -> mx.array:
        """`_p = up(p) + lateral(x)`, with the image patches of its level concatenated."""
        q = _up(p, lx.shape[1:3]) + self.lat[j](lx)
        return mx.concatenate([q, self.ipt[j + 1](_grid(xf, q))], axis=-1)

    def _out(self, p: mx.array, detail: mx.array) -> mx.array:
        """`sigmoid(conv_out1(cat(up(_p1), ipt_blk1(x))))` with the projection of `_p1` before its resize."""
        _, h, w, _ = detail.shape
        y = _up(self.head(p).astype(mx.float32), (h, w)) + detail.astype(mx.float32)
        return mx.sigmoid(y).reshape(1, 1, h, w)

    def _strips(self, x: mx.array) -> mx.array:
        """The image branch (two 3 x 3 convs, the projection folded): `[1, S, S, 1]`, row strips with a halo."""
        h = x.shape[1]
        if h <= STRIP:
            return self.whole(x)
        parts = []
        for r0 in range(0, h, STRIP):
            r1 = min(h, r0 + STRIP)
            top, bottom = max(0, r0 - 2), min(h, r1 + 2)
            step = self.strip if top < r0 and bottom > r1 else self.strip_top if top == r0 else self.strip_bottom
            parts.append(step(x[:, top:bottom]))
        return mx.concatenate(parts, axis=1)

    def forward(self, x: mx.array) -> mx.array:
        """The network on `[1, 3, S, S]` float32 (the steps)."""
        xf, xh = self.enc(x)
        x1, x2, x3, x4 = self.inputs(self.bb(xf), self.bb(xh))
        for blk in self.squeeze:
            x4 = blk(x4)
        p = self.attend[0](self.dec[0](self.top(x4, xf)))
        for j, lx in enumerate((x3, x2, x1)):
            p = self.dec[j + 1](self.lateral[j](p, lx, xf))
            if j < len(self.attend) - 1:
                p = self.attend[j + 1](p)
        return self.out(p, self._strips(xf))

    def __call__(self, x: mx.array) -> mx.array:
        """`forward` with its steps, evaluated."""
        mlx_rsu.options()
        y = csp.steps(self.forward, x)
        mx.eval(y)
        return y


def network(net: Any, dtype: str) -> Network:
    """The MLX network of a `BiRefNet` (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype)
