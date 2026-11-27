"""ViTPose / ViTPose++ on Apple MLX, hand-written (`fastforks.shared.mlx_vit`'s blocks): NHWC, the run dtype.

The network of `model.Portable.heatmaps`: crops `[P, 3, h, w]` in [0, 1] -> float32 heatmaps `[P, K, h / 4, w / 4]`.
ImageNet normalisation, the 16 x 16 patch embedding as one GEMM over the patches (the conv's 2 px zero padding: the
top and left 2 rows / columns padded, the last 2 dropped, as the stride-16 conv reads them), the position table with
the patch bias summed at load, the pre-norm blocks (`mlx_vit.block`'s maths, LayerNorm eps 1e-12, evaluated in half
blocks: `_blocks`), the final LayerNorm and the decoder: "classic" (two 4 x 4 stride-2 transposed convs, BatchNorm
folded, the first as a GEMM with the overlap-add, ReLU, then the 1 x 1 conv as a GEMM) or "simple" (ReLU, the 4x
bilinear resize of `mlx_seg`, the 3 x 3 conv).
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastforks._core.mlx import mx
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD
from fastforks.shared import mlx_seg, mlx_vit
from fastforks.shared.swin_sr import mlx_net as swin

from . import net as tnet

LAYOUT = "vitpose-v2"
SOURCES = (Path(__file__), Path(tnet.__file__), Path(mlx_vit.__file__), Path(swin.__file__))


def _np(t: Any) -> Any:
    return t.detach().double().numpy()


def _interleave(parts: list[mx.array]) -> mx.array:
    """The 4 phase maps `[N, h, w, C]` ((0, 0), (0, 1), (1, 0), (1, 1)) as one `[N, 2h, 2w, C]` map."""
    n, h, w, c = parts[0].shape
    y = mx.stack(parts).reshape(2, 2, n, h, w, c)
    return y.transpose(2, 3, 0, 4, 1, 5).reshape(n, 2 * h, 2 * w, c)


@mx.compile
def _relu_bias(x: mx.array, b: mx.array) -> mx.array:
    return mx.maximum(x + b, 0)


def _deconv_gemm(x: mx.array, wm: mx.array) -> mx.array:
    """A 4 x 4 stride-2 padding-1 transposed conv as one GEMM to the 16 taps and the overlap-add.

    Output phase r (row 2a + r): r = 0 sums input a - 1 (tap 3) and a (tap 1), r = 1 input a (tap 2) and a + 1
    (tap 0). For the decoder's first one (Cin = the width, Cout 256): `mx.conv_transpose2d` runs it as a dilated conv
    at 4x the FLOPs (ViTPose-B float16, 15 crops: 10.6 -> 4.9 ms). The second (Cin = Cout = 256, twice the size)
    stays `mx.conv_transpose2d`: per-phase 2 x 2 convs were 1.5 ms faster but twice its peak.
    """
    n, h, w, cin = x.shape
    c = wm.shape[1] // 16
    y = (x.reshape(-1, cin) @ wm).reshape(n, h, w, 4, 4, c)
    y = mx.pad(y, [(0, 0), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0)])
    tap = (((1, 1), (0, 3)), ((1, 2), (2, 0)))  # per phase: (padded input offset, tap)
    parts = []
    for ry in range(2):
        for rx in range(2):
            t = None
            for oy, ky in tap[ry]:
                for ox, kx in tap[rx]:
                    s = y[:, oy : oy + h, ox : ox + w, ky, kx]
                    t = s if t is None else t + s
            parts.append(t)
    return _interleave(parts)


def _attention(p: dict[str, mx.array], b: str, x: mx.array, heads: int) -> mx.array:
    """The attention half of `mlx_vit.block` (plain ViT: no qk norm, no RoPE)."""
    h = mx.fast.layer_norm(x, p[b + "norm1.weight"], p[b + "norm1.bias"], tnet.EPS).astype(p[b + "qkv.weight"].dtype)
    h = mlx_vit.attention(mx.addmm(p[b + "qkv.bias"], h, p[b + "qkv.weight"]), heads)
    return x + mx.addmm(p[b + "proj.bias"], h, p[b + "proj.weight"])


def _mlp(p: dict[str, mx.array], b: str, x: mx.array) -> mx.array:
    """The GELU MLP half of `mlx_vit.block`."""
    h = mx.fast.layer_norm(x, p[b + "norm2.weight"], p[b + "norm2.bias"], tnet.EPS).astype(p[b + "fc1.weight"].dtype)
    h = mlx_vit.gelu(mx.addmm(p[b + "fc1.bias"], h, p[b + "fc1.weight"]))
    return x + mx.addmm(p[b + "fc2.bias"], h, p[b + "fc2.weight"])


def _blocks(p: dict[str, mx.array], depth: int, heads: int, t: mx.array) -> mx.array:
    """The blocks, each half queued (`mx.async_eval`) and the one before it awaited.

    `mlx_vit.blocks`' `bounded` steps whole blocks, so two blocks' transients are alive; at 15 crops (2880 tokens)
    that is the whole network's peak. Half blocks keep its speed at about half the transients (ViTPose-B, M1 Pro:
    float32 141.0 ms +262 MB -> 141.0 ms +152 MB, float16 127.4 ms +131 MB -> 127.6 ms +84 MB; one block per
    synchronous eval: 143.4 / 129.9 ms at +143 / +89 MB).
    """
    prev = None
    for i in range(depth):
        b = f"blocks.{i}."
        for half in (lambda x, b=b: _attention(p, b, x, heads), lambda x, b=b: _mlp(p, b, x)):
            t = half(t)
            mx.async_eval(t)
            if prev is not None:
                mx.eval(prev)
            prev = t
    return t


class Network:
    """`ViTPose.forward` with the runner's normalisation: crops `[P, 3, h, w]` float32 in [0, 1] -> heatmaps."""

    def __init__(self, net: Any, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        a = net.arch
        self.dtype, self.heads, self.depth = dt, a.heads, a.depth
        self.gh, self.gw = net.gh, net.gw
        self.simple = a.decoder == "simple"
        w, pos, bias = net.patch.weight, net.pos, net.patch.bias  # [C, 3, 16, 16]: the patches' (row, column, channel)
        self.patch = swin.derived(
            (w.shape[1] * w.shape[2] * w.shape[3], a.dim),
            dt,
            lambda: w.detach().double().numpy().transpose(2, 3, 1, 0).reshape(-1, a.dim),
        )
        self.table = swin.derived(
            (self.gh * self.gw, a.dim), dt, lambda: pos.detach().double().numpy()[0] + bias.detach().double().numpy()
        )
        p: dict[str, mx.array] = {}
        for i, blk in enumerate(net.blocks):
            b = f"blocks.{i}."
            for n in ("norm1", "norm2"):
                layer = getattr(blk, n)
                p[b + n + ".weight"], p[b + n + ".bias"] = swin.arr(layer.weight, dt), swin.arr(layer.bias, dt)
            for n in ("qkv", "proj", "fc1", "fc2"):
                p[b + n + ".weight"], p[b + n + ".bias"] = swin.linear(getattr(blk, n), dt)
        self.p = p
        self.norm = (swin.arr(net.norm.weight, dt), swin.arr(net.norm.bias, dt))
        head = net.head
        if self.simple:
            conv = head[2]
            self.out_w = swin.arr(conv.weight.permute(0, 2, 3, 1), dt)  # [K, 3, 3, C]
            self.out_b = swin.arr(conv.bias, dt)
        else:
            d0, d1 = head[0], head[2]  # weights [Cin, Cout, 4, 4]
            ci0, co0 = d0.weight.shape[:2]
            self.deconv0 = (  # [Cin, (ky, kx, Cout)]
                swin.derived((ci0, 16 * co0), dt, lambda: _np(d0.weight).transpose(0, 2, 3, 1).reshape(ci0, -1)),
                swin.arr(d0.bias, dt),
            )
            self.deconv1 = (  # [Cout, 4, 4, Cin]: `mx.conv_transpose2d` (Cin = Cout: as fast as the phases, leaner)
                swin.arr(d1.weight.permute(1, 2, 3, 0), dt),
                swin.arr(d1.bias, dt),
            )
            conv = head[4]
            self.out_w = swin.arr(conv.weight[:, :, 0, 0].t(), dt)  # [256, K]
            self.out_b = swin.arr(conv.bias, dt)
        self.mean = mx.array(IMAGENET_MEAN, mx.float32)
        self.std = mx.array(IMAGENET_STD, mx.float32)

    def tokens(self, x: mx.array) -> mx.array:
        """Normalised patch tokens `[P, N, C]` with the position table of crops `[P, 3, h, w]` float32."""
        s, pad = tnet.PATCH, tnet.PATCH_PAD
        b = x.shape[0]
        y = ((x.transpose(0, 2, 3, 1) - self.mean) / self.std).astype(self.dtype)  # NHWC
        h, w = self.gh * s, self.gw * s
        y = mx.pad(y, [(0, 0), (pad, 0), (pad, 0), (0, 0)])[:, :h, :w]  # what the stride-16 conv reads
        y = y.reshape(b, self.gh, s, self.gw, s, 3).transpose(0, 1, 3, 2, 4, 5).reshape(b, self.gh * self.gw, -1)
        return y @ self.patch + self.table

    def head(self, t: mx.array) -> mx.array:
        """The final LayerNorm and the decoder: tokens `[P, N, C]` -> heatmaps `[P, K, 4 gh, 4 gw]` float32."""
        t = mx.fast.layer_norm(t, *self.norm, tnet.EPS)
        y = t.reshape(t.shape[0], self.gh, self.gw, -1)
        if self.simple:
            y = mlx_seg.resize(mx.maximum(y, 0), (self.gh * tnet.UPSAMPLE, self.gw * tnet.UPSAMPLE))
            y = mx.conv2d(y, self.out_w, padding=1) + self.out_b
        else:
            y = _relu_bias(_deconv_gemm(y, self.deconv0[0]), self.deconv0[1])
            y = _relu_bias(mx.conv_transpose2d(y, self.deconv1[0], stride=2, padding=1), self.deconv1[1])
            y = swin.dense(y, self.out_w, self.out_b)
        return y.transpose(0, 3, 1, 2).astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        """Heatmaps of crops `[P, 3, h, w]` float32 in [0, 1], evaluated."""
        out = self.head(_blocks(self.p, self.depth, self.heads, self.tokens(x)))
        mx.eval(out)
        return out


def network(net: Any, dtype: str) -> Network:
    """The MLX network of a `ViTPose` (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype)
