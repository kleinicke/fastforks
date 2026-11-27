"""BEN2 on Apple MLX, hand-written: NHWC, the run dtype.

`BEN_Base.forward` (the mask logits) on one normalised `[1, S, S, 3]` image:

- the four quarter patches and the image at half size (bilinear) through the Swin v1 encoder
  (`shared.mlx_swin_net.Encoder`, the patch embedding as the first map), one image after the other, each layer one step;
- the decoder's conv + InstanceNorm + GELU blocks (`make_cbr`: statistics in float32), the multi-field cross attention
  (`MCLM`) and the four refiners (`MCRM`): `nn.MultiheadAttention` with one head as MLX's fused SDPA, the sine position
  encodings and the adaptive average pools (exact factors) as reshapes; each module one step;
- the shallow branch only where it is read: the 3 x 3 conv followed by the exact 4x / 2x bilinear downsamplings is one
  4 x 4 conv of stride 4 / 2 (the four averaged taps folded into the kernel in numpy), never the full-size map;
- the two nearest 2x upsamplings and the 3 x 3 convs after them as four 2 x 2 phase convs on the small map,
  interleaved (the kernel rows / columns that read the same source pixel summed); the last block's InstanceNorm, GELU
  and the output conv in row strips of at most `STRIP_BYTES`, each a step.

The full-size decoder steps (the mask head, both upsamplings, the last block's statistics) run one at a time
(`_serial`): pipelined, two of them were resident at once (float16 1024 x 1024 call peak 2.3 GB, now 1.6 GB).
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.shared import mlx_matting, mlx_rsu, mlx_swin_net
from fastforks.shared.cspnext import mlx_net as csp

from .net import ben2 as tnet

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    Layer = Callable[[mx.array], mx.array]

LAYOUT = "ben2-v1"
SOURCES = (Path(__file__), Path(tnet.__file__), Path(mlx_rsu.__file__), *mlx_swin_net.SOURCES)
STRIP_BYTES = 64 << 20  # the full-size map of one output strip (activation dtype)
IN_EPS = 1e-5  # nn.InstanceNorm2d's default
LN_EPS = 1e-5  # nn.LayerNorm's default

_np, _arr, _lin = mlx_swin_net.np64, mlx_swin_net.arr, mlx_swin_net.lin


def _gelu(x: mx.array) -> mx.array:
    """Exact (erf) GELU computed in float32."""
    y = x.astype(mx.float32)
    return (y * (0.5 + 0.5 * mx.erf(y * 0.7071067811865476))).astype(x.dtype)


def _stats(x: mx.array) -> tuple[mx.array, mx.array]:
    """Per-image, per-channel mean and `1 / sqrt(var + eps)` of NHWC `x`, float32 `[N, 1, 1, C]`."""
    y = x.astype(mx.float32)
    mean = y.mean(axis=(1, 2), keepdims=True)
    var = mx.square(y - mean).mean(axis=(1, 2), keepdims=True)
    return mean, mx.rsqrt(var + IN_EPS)


def _norm_gelu(x: mx.array, stats: tuple[mx.array, mx.array]) -> mx.array:
    mean, inv = stats
    y = (x.astype(mx.float32) - mean) * inv
    return (y * (0.5 + 0.5 * mx.erf(y * 0.7071067811865476))).astype(x.dtype)


def _cbr(seq: Any, dtype: mx.Dtype) -> Layer:
    """`make_cbr`: GELU(InstanceNorm(conv(x)))."""
    conv = mlx_rsu.conv(seq[0], None, dtype)

    def apply(x: mx.array) -> mx.array:
        y = conv(x)
        return _norm_gelu(y, _stats(y))

    return apply


def _ln(m: Any, dtype: mx.Dtype) -> Layer:
    w, b = mlx_swin_net.norm(m, dtype)
    return lambda x: mx.fast.layer_norm(x, w, b, LN_EPS)


def _ffn(a: Any, b: Any, dtype: mx.Dtype) -> Layer:
    """`b(GELU(a(x)))`."""
    la, lb = _lin(a, dtype), _lin(b, dtype)
    return lambda x: mx.addmm(lb[1], _gelu(mx.addmm(la[1], x, la[0])), lb[0])


def _mha(m: Any, dtype: mx.Dtype) -> Callable[[mx.array, mx.array, mx.array], mx.array]:
    """`nn.MultiheadAttention(d, 1)(q, k, v)[0]` on token rows `[L, d]`."""
    d = int(m.embed_dim)
    w, b = m.in_proj_weight, m.in_proj_bias
    proj = [
        (
            _arr((d, d), dtype, lambda i=i: _np(w)[i * d : (i + 1) * d].T),
            _arr((d,), dtype, lambda i=i: _np(b)[i * d : (i + 1) * d]),
        )
        for i in range(3)
    ]
    out = _lin(m.out_proj, dtype)

    def apply(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        q, k, v = (mx.addmm(p[1], t, p[0])[None, None] for p, t in zip(proj, (q, k, v), strict=True))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)[0, 0]
        return mx.addmm(out[1], o, out[0])

    return apply


def _to_image(x: mx.array) -> mx.array:
    """`patches2image`: the four patches `[4, h, w, C]` (row-major quarters) -> `[1, 2h, 2w, C]`."""
    _, h, w, c = x.shape
    return x.reshape(2, 2, h, w, c).transpose(0, 2, 1, 3, 4).reshape(1, 2 * h, 2 * w, c)


def _to_patches(x: mx.array) -> mx.array:
    """`image2patches`: `[1, 2h, 2w, C]` -> `[4, h, w, C]`."""
    _, hh, ww, c = x.shape
    h, w = hh // 2, ww // 2
    return x.reshape(2, h, 2, w, c).transpose(0, 2, 1, 3, 4).reshape(4, h, w, c)


def _pool(x: mx.array, out: tuple[int, int]) -> mx.array:
    """`F.adaptive_avg_pool2d(x, out)` of NHWC `x` whose sides `out` divide, in float32."""
    n, h, w, c = x.shape
    oh, ow = out
    if h % oh or w % ow:
        msg = f"BEN2's pools need exact factors ({h} x {w} -> {oh} x {ow})"
        raise ValueError(msg)
    if (h, w) == out:
        return x
    return x.astype(mx.float32).reshape(n, oh, h // oh, ow, w // ow, c).mean(axis=(2, 4)).astype(x.dtype)


def sine(h: int, w: int, feats: int) -> np.ndarray:
    """`PositionEmbeddingSine(feats, normalize=True)` of an `h x w` map as `[h * w, 2 feats]` rows (float64)."""
    eps, scale = 1e-6, 2 * math.pi
    y = (np.arange(1, h + 1, dtype=np.float64) - 0.5) / (h + eps) * scale
    x = (np.arange(1, w + 1, dtype=np.float64) - 0.5) / (w + eps) * scale
    dim_t = 10000.0 ** (2 * (np.arange(feats) // 2) / feats)

    def enc(p: np.ndarray) -> np.ndarray:
        q = p[:, None] / dim_t
        return np.stack([np.sin(q[:, 0::2]), np.cos(q[:, 1::2])], axis=2).reshape(len(p), feats)

    py, px = enc(y), enc(x)
    return np.concatenate([np.repeat(py[:, None], w, 1), np.repeat(px[None], h, 0)], axis=2).reshape(h * w, -1)


class _Mclm:
    """`MCLM(l, g)`: the four local maps `[4, h, w, C]` and the global one `[1, h, w, C]` -> `[5, h, w, C]`."""

    def __init__(self, m: Any, dtype: mx.Dtype) -> None:
        self.dtype, self.ratios = dtype, tuple(int(r) for r in m.pool_ratios)
        self.feats = int(m.positional_encoding.num_pos_feats)
        self.att = [_mha(a, dtype) for a in m.attention]
        self.ffn_g, self.ffn_l = _ffn(m.linear1, m.linear2, dtype), _ffn(m.linear3, m.linear4, dtype)
        self.norm1, self.norm2 = _ln(m.norm1, dtype), _ln(m.norm2, dtype)
        self.pos: dict[tuple[int, int], mx.array] = {}
        self.step = csp.stepped(self._forward)

    def _sine(self, h: int, w: int) -> mx.array:
        if (h, w) not in self.pos:
            self.pos[h, w] = mx.array(sine(h, w, self.feats).astype(np.float32)).astype(self.dtype)
        return self.pos[h, w]

    def positions(self, h: int, w: int) -> tuple[mx.array, mx.array]:
        """The pools' and the global map's encodings (constants of the map size)."""
        sizes = [(round(h / r), round(w / r)) for r in self.ratios]
        return mx.concatenate([self._sine(*s) for s in sizes]), self._sine(h, w)

    def _forward(self, x: mx.array, p_pos: mx.array, g_pos: mx.array) -> mx.array:
        _, h, w, c = x.shape
        loc, g = x[:4], x[4:].reshape(h * w, c)
        image = _to_image(loc)
        pools = mx.concatenate([_pool(image, (round(h / r), round(w / r))).reshape(-1, c) for r in self.ratios])
        g = self.norm1(g + self.att[0](g + g_pos, pools + p_pos, pools))
        g = self.norm2(g + self.ffn_g(g))
        quarters = g.reshape(2, h // 2, 2, w // 2, c).transpose(0, 2, 1, 3, 4).reshape(4, -1, c)
        tokens = loc.reshape(4, h * w, c)
        out = mx.stack([self.att[i + 1](tokens[i], quarters[i], quarters[i]) for i in range(4)])
        lt = self.norm1(tokens + out)
        lt = self.norm2(lt + self.ffn_l(lt))
        return mx.concatenate([lt.reshape(4, h, w, c), g.reshape(1, h, w, c)])

    def __call__(self, x: mx.array) -> mx.array:
        return self.step(x, *self.positions(x.shape[1], x.shape[2]))


class _Mcrm:
    """`MCRM(x)[0]`: `[5, h, w, C]` (four local maps, the global one last) -> `[5, h, w, C]`."""

    def __init__(self, m: Any, dtype: mx.Dtype) -> None:
        self.ratios = tuple(int(r) for r in m.pool_ratios)
        self.att = [_mha(a, dtype) for a in m.attention]
        self.ffn = _ffn(m.linear3, m.linear4, dtype)
        self.norm1, self.norm2 = _ln(m.norm1, dtype), _ln(m.norm2, dtype)
        self.sal = mlx_rsu.conv(m.sal_conv, None, dtype)

    def __call__(self, x: mx.array) -> mx.array:
        _, h, w, c = x.shape
        loc, glb = x[:4], x[4:]
        patched = _to_patches(glb)
        tam = mx.sigmoid(self.sal(glb))  # [1, h, w, 1]
        tam = mx.repeat(mx.repeat(tam, 2, axis=1), 2, axis=2)  # nearest 2x
        loc = loc * _to_patches(tam)
        pools = mx.concatenate(
            [_pool(patched, (round(h / r), round(w / r))).reshape(4, -1, c) for r in self.ratios], axis=1
        )
        tokens = loc.reshape(4, h * w, c)
        out = mx.stack([self.att[i](tokens[i], pools[i], pools[i]) for i in range(4)])
        src = self.norm1(tokens + out)
        src = self.norm2(src + self.ffn(src)).reshape(4, h, w, c)
        glb = glb + _to_image(src)[:, ::2, ::2]  # nearest 1/2
        return mx.concatenate([src, glb])


def _taps(w: np.ndarray) -> np.ndarray:
    """A 3 x 3 kernel followed by the mean of two adjacent outputs per axis: the 4 x 4 kernel of that mean."""
    o, i = w.shape[:2]
    k = np.zeros((o, i, 4, 4))
    for a in (0, 1):
        for b in (0, 1):
            k[:, :, a : a + 3, b : b + 3] += 0.25 * w
    return k


def _shallow(conv: Any, factor: int, dtype: mx.Dtype) -> Layer:
    """`F.interpolate(conv(x), scale 1 / factor, bilinear)` for `factor` 2 or 4: one 4 x 4 conv of that stride."""
    o, i = (int(d) for d in conv.weight.shape[:2])
    pad = 1 if factor == 2 else 0  # noqa: PLR2004 - the 2x mean reads one row / column before the first output's

    def folded() -> tuple[np.ndarray, np.ndarray]:
        return _taps(_np(conv.weight)), _np(conv.bias)

    return mlx_rsu.conv_of((o, i, 4, 4), folded, dtype, stride=(factor, factor), padding=(pad, pad))


class _UpConv:
    """`conv(nearest 2x (x))` for a 3 x 3 conv: four 2 x 2 phase convs on `x`, as `[N, h, 2, w, 2, O]` phases."""

    def __init__(self, conv: Any, dtype: mx.Dtype) -> None:
        o, i = (int(d) for d in conv.weight.shape[:2])
        rows = ([[0], [1, 2]], [[0, 1], [2]])  # phase 0 reads (i - 1, i), phase 1 (i, i + 1): the taps summed

        def phase(a: int, b: int) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
            def folded() -> tuple[np.ndarray, np.ndarray]:
                w = _np(conv.weight)
                k = np.zeros((o, i, 2, 2))
                for r, rs in enumerate(rows[a]):
                    for s, cs in enumerate(rows[b]):
                        k[:, :, r, s] = w[:, :, rs][:, :, :, cs].sum(axis=(2, 3))
                return k, _np(conv.bias)

            return folded

        self.phases = [[mlx_rsu.conv_of((o, i, 2, 2), phase(a, b), dtype) for b in (0, 1)] for a in (0, 1)]

    def __call__(self, x: mx.array) -> mx.array:
        _, h, w, _ = x.shape
        xp = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
        ph = [
            mx.stack([self.phases[a][b](xp[:, a : a + h + 1, b : b + w + 1]) for b in (0, 1)], axis=3) for a in (0, 1)
        ]
        return mx.stack(ph, axis=2)  # [N, h, 2, w, 2, O]


def _serial(f: Callable[..., Any]) -> Callable[..., Any]:
    """`f` compiled as a step that starts once the step in flight is done (`csp.stepped` overlaps two steps).

    The decoder's last steps work on the full-size map; pipelined, two of them were resident at once.
    """
    g = mx.compile(f)

    def run(*args: Any) -> Any:
        if csp._PENDING:  # noqa: SLF001 - the shared pipeline
            mx.eval(csp._PENDING.pop())  # noqa: SLF001
        y = g(*args)
        mx.async_eval(y)
        csp._PENDING.append(y)  # noqa: SLF001
        return y

    return run


def _interleave(p: mx.array) -> mx.array:
    n, h, _, w, _, c = p.shape
    return p.reshape(n, 2 * h, 2 * w, c)


class Network:
    """`BEN_Base.forward`: a normalised `[1, S, S, 3]` float32 image -> the mask logits `[1, S, S, 1]` float32."""

    def __init__(self, net: Any, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = self.dtype = getattr(mx, dtype)
        self.bb = mlx_swin_net.Encoder(net.backbone, dt, embedding=True)
        self.outputs = [_cbr(getattr(net, f"output{i}"), dt) for i in range(1, 6)]
        self.mclm = _Mclm(net.multifieldcrossatt, dt)
        self.refine = [_Mcrm(getattr(net, f"dec_blk{i}"), dt) for i in range(1, 5)]
        self.convs = [_cbr(getattr(net, f"conv{i}"), dt) for i in range(1, 5)]
        head = net.insmask_head
        self.head = (_cbr(head[0:3], dt), _cbr(head[3:6], dt), mlx_rsu.conv(head[6], None, dt))
        self.shallow_low = _shallow(net.shallow[0], 4, dt)
        self.shallow_mid = _shallow(net.shallow[0], 2, dt)
        self.up1, self.up2 = _UpConv(net.upsample1[0], dt), _UpConv(net.upsample2[0], dt)
        self.out = mlx_rsu.conv(net.output[0], None, dt)
        self.side_steps = [csp.stepped(f) for f in self.outputs]
        self.level_steps = [csp.stepped(self._level(i)) for i in range(4)]
        self.head_step = _serial(self._head)
        self.up1_step = _serial(self._up1)
        self.up2_step = _serial(lambda y, m: self.up2(y + m))
        self.stats_step = _serial(lambda p: _stats(p.reshape(1, -1, 1, p.shape[-1])))
        self.out_step = csp.stepped(self._output)  # small strips: pipelined

    def _level(self, i: int) -> Callable[[mx.array, mx.array], mx.array]:
        """Refiner `4 - i` and its conv on the level's map plus the coarser result (bilinear)."""
        blk, conv = self.refine[3 - i], self.convs[3 - i]

        def apply(e: mx.array, coarser: mx.array) -> mx.array:
            return conv(blk(e + mlx_rsu.up_like(coarser, e.shape[1:3])))

        return apply

    def _head(self, e1: mx.array, low: mx.array) -> mx.array:
        y = _to_image(e1[:4])
        y = y + mlx_rsu.up_like(e1[4:], y.shape[1:3])
        y = self.head[2](self.head[1](self.head[0](y)))
        return y + low

    def _up1(self, y: mx.array) -> mx.array:
        p = self.up1(y)
        return _norm_gelu(p, _stats(p.reshape(1, -1, 1, p.shape[-1])))

    def _output(self, p: mx.array, stats: Sequence[mx.array], halo: tuple[int, int]) -> mx.array:
        """The output conv of an interleaved strip of the normed, activated last block (`halo`: rows before, after)."""
        y = _norm_gelu(_interleave(p), (stats[0], stats[1]))
        y = self.out(y).astype(mx.float32)
        return y[:, halo[0] : y.shape[1] - halo[1]]

    def encode(self, x: mx.array) -> list[mx.array]:
        """The five maps of each of the four quarters and the half-size image, `[5, h, w, C]` per level."""
        s, x = x.shape[1], x.astype(self.dtype)
        half = mlx_matting.resize(x.transpose(0, 3, 1, 2), (s // 2, s // 2)).transpose(0, 2, 3, 1)
        images = [*(q[None] for q in _to_patches(x)), half]
        feats = [self.bb(im) for im in images]
        return [mx.concatenate([f[i] for f in feats]) for i in range(len(feats[0]))]

    def forward(self, x: mx.array) -> mx.array:
        """The logits of the image, step by step."""
        feats = self.encode(x)
        e = [step(f) for step, f in zip(self.side_steps, feats, strict=True)]  # e1 .. e5
        del feats  # each map is freed once the step that reads it is done
        y = self.mclm(e.pop())
        for step in self.level_steps:
            y = step(e.pop(), y)  # e4 .. e1
        xs = x.astype(self.dtype)
        y = self.head_step(y, self.shallow_low(xs))
        y = self.up1_step(y)  # [1, h, 2, w, 2, C] phases
        p = self.up2_step(_interleave(y), self.shallow_mid(xs))  # [1, h, 2, w, 2, C] phases of the last block
        stats = self.stats_step(p)
        h, w, c = p.shape[1], p.shape[3], p.shape[-1]
        rows = max(1, min(h, STRIP_BYTES // (4 * w * c * p.itemsize)))
        out = []
        for start in range(0, h, rows):
            stop = min(start + rows, h)
            lo, hi = max(0, start - 1), min(h, stop + 1)
            out.append(self.out_step(p[:, lo:hi], stats, (2 * (start - lo), 2 * (hi - stop))))
        return mx.concatenate(out, axis=1)

    def __call__(self, x: mx.array) -> mx.array:
        """`forward` with its steps, evaluated."""
        mlx_rsu.options()
        y = csp.steps(self.forward, x)
        mx.eval(y)
        return y


def network(net: Any, dtype: str) -> Network:
    """The MLX network of a `BEN_Base` (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype)
