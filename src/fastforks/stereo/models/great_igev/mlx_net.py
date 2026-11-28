"""GREAT-IGEV on Apple MLX, written in `mlx.core` operations (`MlxGREAT(net, dtype, iters)`): pair -> disparity.

The maths of `portable.forward_fast` (`net/great_igev.py`, `net/transformers.py`), channels last, in place of the export
and graph translation the MLX runner replayed before. The IGEV frame (`igev_blocks.mlx_stereo.MlxIgevUpdate`) holds the
MobileNetV2 features, the context encoder, the update loop, the geometry lookups and the upsampling; this module adds
the excitive-attention volume:

- `_match`: the `AdaptiveGlobalRefiner` (channel / spatial attention, small / large receptive-field blend);
- `_geometry`: the positional embedding and the row cross-attention refiner of the matching features, the four
  outlook / window attention context encoders, the concat cost volume through `cc_stem` x `feat_stem` -> `fuse_stem`
  in disparity chunks, the hourglass with its volume transformer blocks and the softmax over disparity.

Attention keeps upstream's sink competition, `(softmax(s) @ v + eps * sum(v)) / (1 + n * eps)` (`_core.mlx.attention`
for the softmax part), and the outlook attention upstream's flat reshape of the NCHW logits. The first `cc_stem`
convolution is split into its left and right halves: the left half is one 2-D convolution of the left features for
every disparity, the right half one convolution of the right features, shifted per disparity (with the one column the
volume's zero padding changes corrected). Float32: layer- and instance-norm statistics, the softmaxes, the sink
corrections, the volume attention (logits, update), the initial disparity and everything the frame keeps in float32.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import numpy as np

from fastforks._core.mlx import attention, mx
from fastforks.kernels.graph_mlx.ops import depthwise
from fastforks.shared import mlx_port
from fastforks.shared.igev_blocks import mlx_blocks as bl
from fastforks.shared.igev_blocks import mlx_stereo
from fastforks.shared.igev_blocks.mlx_stereo import MlxIgevUpdate

__all__ = ["LAYOUT", "SOURCES", "MlxGREAT"]

LAYOUT = "great_igev-v1"  # bump on any change of the converted arrays (names, shapes, folds)
SOURCES = (Path(__file__), Path(bl.__file__), Path(mlx_stereo.__file__))

EPS = 1e-6  # the attention layers' sink eps
OUTLOOK_CHUNK = 2**24  # outlook logits per chunk (elements), as `transformers.OUTLOOK_CHUNK`
CC_PLANES = 2**28  # concat-volume elements (2 C x H x W per plane) per disparity chunk, as `great_igev.CC_CHUNK`
F32 = mx.float32

Lin = tuple[mx.array, mx.array | None]  # [I, O] weight, bias
Norm = tuple[mx.array, mx.array, float]  # LayerNorm weight, bias, eps


# weights --------------------------------------------------------------------------------------------------------------


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().float().numpy()


def _lin(w: bl.Weights, m: Any, dtype: mx.Dtype | None = None) -> Lin:  # noqa: ANN401 - nn.Linear / 1x1 nn.Conv2d
    """A linear layer (or 1x1 convolution) as `x @ W + b`, `W` `[I, O]`."""
    o, i = int(m.weight.shape[0]), int(m.weight.shape[1])
    weight = w.t((i, o), lambda: _np(m.weight).reshape(o, i).T, dtype)
    return weight, None if m.bias is None else w.t((o,), lambda: _np(m.bias), dtype)


def _norm(w: bl.Weights, m: Any) -> Norm:  # noqa: ANN401 - nn.LayerNorm
    c = int(m.weight.shape[0])
    return w.t((c,), lambda: _np(m.weight)), w.t((c,), lambda: _np(m.bias)), float(m.eps)


def _ffn(w: bl.Weights, m: Any) -> dict[str, Any]:  # noqa: ANN401 - transformers.FeedForwardLayer
    return {
        "norm": _norm(w, m.norm),
        "in": _lin(w, m.in_conv),
        "dw": [w.dw(getattr(m, f"dw_conv_{k}")) for k in (3, 5, 7)],
        "pw": _lin(w, m.pointwise_conv[0]),
        "out": _lin(w, m.out_conv),
    }


def _attn(w: bl.Weights, m: Any) -> dict[str, Any]:  # noqa: ANN401 - AttentionLayer / WindowAttentionLayer
    return {
        "norm": _norm(w, m.norm),
        "ctx": _norm(w, m.norm_context),
        "q": _lin(w, m.to_q),
        "kv": _lin(w, m.to_kv),
        "out": _lin(w, m.to_out),
        "scale": float(m.scale),
    }


def _outlook(w: bl.Weights, m: Any) -> dict[str, Any]:  # noqa: ANN401 - transformers.OutlookAttentionLayer
    k = int(m.kernel_size)
    n, c = k**4, int(m.to_attn.weight.shape[1])
    return {
        "norm": _norm(w, m.norm),
        "v": _lin(w, m.to_v),
        "attn": w.t((n, c), lambda: _np(m.to_attn.weight)[:n].reshape(n, c)),  # [k^4, C]: the flat logit order
        "attn_b": w.t((n, 1), lambda: _np(m.to_attn.bias)[:n, None]),
        "out": _lin(w, m.to_out),
        "k": k,
        "scale": float(m.scale),
    }


def _encoder(w: bl.Weights, e: Any) -> dict[str, Any]:  # noqa: ANN401 - GlobalAttentionFeatureEncoder
    return {
        "agg": (w.basic(e.channels_agg[0]), w.conv(e.channels_agg[1])),
        "outlook": _outlook(w, e.outlook_self_attn),
        "outlook_ffn": _ffn(w, e.outlook_self_ffn),
        "win": _attn(w, e.win_self_attn),
        "win_ffn": _ffn(w, e.win_self_ffn),
        "swin": _attn(w, e.swin_self_attn),
        "swin_ffn": _ffn(w, e.swin_self_ffn),
        "splits": int(e.num_splits),
    }


def _vtb(w: bl.Weights, m: Any) -> dict[str, Any]:  # noqa: ANN401 - transformers.VolumeTransformerBlock
    """The reassociated volume attention: float32 `W_q` `[hidden, in]`, `scale * gamma`, `W_o^T`, `b_o`."""
    a, f = m.volume_attn, m.feedforward
    hid, cin = (int(n) for n in a.to_q.weight.shape)
    out = int(a.to_out.weight.shape[0])
    return {
        "ctx": _norm(w, a.norm_context),
        "kv": _lin(w, a.to_kv),
        "q": w.t((hid, cin), lambda: _np(a.to_q.weight), F32),
        "gamma": w.t((cin,), lambda: float(a.scale) * _np(a.norm.weight), F32),
        "wo": w.t((hid, out), lambda: _np(a.to_out.weight).T, F32),
        "bo": w.t((out,), lambda: _np(a.to_out.bias), F32),
        "eps": float(a.norm.eps),
        "ffn": (_norm(w, f[0]), _lin(w, f[1]), _lin(w, f[3])),
    }


# layers ---------------------------------------------------------------------------------------------------------------


def _mm(x: mx.array, m: Lin) -> mx.array:
    y = x @ m[0]
    return y if m[1] is None else y + m[1]


def _ln(x: mx.array, n: Norm) -> mx.array:
    return mx.fast.layer_norm(x, n[0], n[1], n[2])


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _sink(o: mx.array, v: mx.array) -> mx.array:
    """Upstream's `softmax + eps, renormalised` of the attention output `o` over the keys `v` `[..., L, C]`."""
    vs = v.astype(F32).sum(axis=-2, keepdims=True)
    return ((o.astype(F32) + EPS * vs) / (1 + v.shape[-2] * EPS)).astype(o.dtype)


def _ffn_fwd(x: mx.array, f: dict[str, Any]) -> mx.array:
    """`FeedForwardLayer`: LayerNorm -> 1x1 -> SiLU -> y + SiLU(1x1(cat(dw3, dw5, dw7))) -> 1x1."""
    y = _silu(_mm(_ln(x, f["norm"]), f["in"]))
    z = mx.concatenate([depthwise(y, wd, b, (1, 1), (wd.shape[0] // 2,) * 2, (1, 1)) for wd, b, _ in f["dw"]], axis=-1)
    return _mm(y + _silu(_mm(z, f["pw"])), f["out"])


@functools.cache
def _position(h: int, w: int, npf: int) -> np.ndarray:
    """`PositionalEmbeddingCosine2D(npf, normalize=True)` `[H, W, 2 npf]` float32 (`[y | x]`)."""

    def axis(n: int) -> np.ndarray:
        p = np.arange(1, n + 1, dtype=np.float32) / np.float32(n + 1e-6) * np.float32(2 * math.pi)
        dim_t = np.float32(10000) ** (2 * (np.arange(npf) // 2).astype(np.float32) / np.float32(npf))
        e = p[:, None] / dim_t
        return np.stack([np.sin(e[:, 0::2]), np.cos(e[:, 1::2])], axis=2).reshape(n, npf)

    y, x = axis(h), axis(w)
    return np.concatenate(
        [np.broadcast_to(y[:, None], (h, w, npf)), np.broadcast_to(x[None], (h, w, npf))], axis=2
    ).astype(np.float32)


def _pos(x: mx.array) -> mx.array:
    _, h, w, c = x.shape
    return mx.array(_position(h, w, c // 2)).astype(x.dtype)[None]


def _row_attention(a: mx.array, ctx: mx.array, m: dict[str, Any]) -> mx.array:
    """`AttentionLayer`: every row of `a` `[B, H, W, C]` attends to the same row of `ctx`."""
    c = a.shape[-1]
    q = _mm(_ln(a, m["norm"]), m["q"])
    kv = _mm(_ln(ctx, m["ctx"]), m["kv"])
    k, v = kv[..., :c], kv[..., c:]
    o = _sink(attention(q, k, v, scale=m["scale"]), v)  # [B, H, W, C]: the rows are the heads
    return _mm(o, m["out"])


def _windows(t: mx.array, s: int) -> mx.array:
    """`[1, H, W, C]` -> `[s, s, L, C]` (window `(i, j)`, as upstream's `split_feature`)."""
    _, h, w, c = t.shape
    return t.reshape(s, h // s, s, w // s, c).transpose(0, 2, 1, 3, 4).reshape(s, s, (h // s) * (w // s), c)


@functools.cache
def _shift_masks(h: int, w: int, s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The boolean masks of the last-column, last-row and corner windows (`shift_regions`): same region attends."""
    wh, ww = h // s, w // s

    def axis(n: int, win: int, shift: int) -> np.ndarray:
        i = np.arange(n)
        return (i >= n - win).astype(np.int64) + (i >= n - shift)

    ids = axis(h, wh, wh // 2)[:, None] * 3 + axis(w, ww, ww // 2)[None, :]
    win = ids.reshape(s, wh, s, ww).transpose(0, 2, 1, 3).reshape(s, s, wh * ww)

    def mask(i: int, j: int) -> np.ndarray:
        r = win[i, j]
        return r[:, None] == r[None, :]

    return mask(0, s - 1), mask(s - 1, 0), mask(s - 1, s - 1)


def _window_attention(x: mx.array, m: dict[str, Any], s: int, *, shift: bool) -> mx.array:
    """`WindowAttentionLayer(x, x, s, with_shift=shift)`: self attention in `s x s` windows."""
    _, h, w, c = x.shape
    sh, sw = (h // s) // 2, (w // s) // 2
    q = _mm(_ln(x, m["norm"]), m["q"])
    kv = _mm(_ln(x, m["ctx"]), m["kv"])
    k, v = kv[..., :c], kv[..., c:]
    if shift:
        q, k, v = (mx.roll(t, (-sh, -sw), axis=(1, 2)) for t in (q, k, v))
    qw, kw, vw = (_windows(t, s) for t in (q, k, v))
    scale = m["scale"]

    def part(i: slice, j: slice, mask: np.ndarray | None) -> mx.array:
        qq = qw[i, j]
        a, b, n = qq.shape[:3]
        heads = [t.reshape(1, a * b, n, c) for t in (qq, kw[i, j], vw[i, j])]
        o = attention(*heads, scale=scale, mask=None if mask is None else mx.array(mask))
        return o.reshape(a, b, n, c)

    if not shift:
        o = part(slice(None), slice(None), None)
    else:
        col, row, corner = _shift_masks(h, w, s)
        e = s - 1
        bottom = part(slice(e, None), slice(e, None), corner)
        if s > 1:
            top = mx.concatenate(
                [part(slice(None, e), slice(None, e), None), part(slice(None, e), slice(e, None), col)], axis=1
            )
            bottom = mx.concatenate([part(slice(e, None), slice(None, e), row), bottom], axis=1)
            o = mx.concatenate([top, bottom], axis=0)
        else:
            o = bottom
    o = _sink(o, vw)
    o = o.reshape(s, s, h // s, w // s, c).transpose(0, 2, 1, 3, 4).reshape(1, h, w, c)
    if shift:
        o = mx.roll(o, (sh, sw), axis=(1, 2))
    return _mm(o, m["out"])


def _outlook_attention(x: mx.array, m: dict[str, Any]) -> mx.array:
    """`OutlookAttentionLayer` (stride 1, one head) of `[1, H, W, C]`; logit chunks evaluated one at a time."""
    _, h, w, c = x.shape
    k, dt = m["k"], x.dtype
    p, kk, n = k // 2, k * k, h * w
    xn = _ln(x, m["norm"])
    vp = mx.pad(_mm(xn, m["v"])[0], [(p, p), (p, p), (0, 0)])
    v = mx.stack([vp[i : i + h, j : j + w] for i in range(k) for j in range(k)], axis=2).reshape(n, kk, c)
    xt = xn.reshape(n, c).T  # [C, N]
    wt, bias = m["attn"], m["attn_b"]
    step = max(1, OUTLOOK_CHUNK // (kk * kk))
    chunks = []
    for m0 in range(0, n, step):
        m1 = min(n, m0 + step)
        f0, f1 = m0 * kk * kk, m1 * kk * kk
        c0, c1 = f0 // n, (f1 - 1) // n + 1
        lg = (wt[c0:c1] @ xt + bias[c0:c1]).reshape(-1)[f0 - c0 * n : f1 - c0 * n]
        lg = lg.reshape(m1 - m0, kk, kk).astype(F32)
        pr = mx.softmax(lg * m["scale"], axis=-1) + EPS
        pr = pr / pr.sum(axis=-1, keepdims=True)
        chunks.append(pr.astype(dt) @ v[m0:m1])
        if m1 < n:
            mx.eval(chunks[-1])
    out = mx.pad(
        (chunks[0] if len(chunks) == 1 else mx.concatenate(chunks)).reshape(h, w, kk, c),
        [(p, p), (p, p), (0, 0), (0, 0)],
    )
    # fold: window position t = i k + j of pixel (y, x) adds to pixel (y + i - p, x + j - p)
    acc = None
    for i in range(k):
        for j in range(k):
            t = out[2 * p - i : 2 * p - i + h, 2 * p - j : 2 * p - j + w, i * k + j].astype(F32)
            acc = t if acc is None else acc + t
    assert acc is not None  # noqa: S101 - k >= 1
    return _mm(acc.astype(dt)[None], m["out"])


def _call_ffn(x: mx.array, f: dict[str, Any]) -> mx.array:
    """The compiled `_ffn_fwd` of `f` (when `MlxGREAT` compiled it)."""
    fn = f.get("fn")
    return _ffn_fwd(x, f) if fn is None else fn(x)


def _encode(x: mx.array, e: dict[str, Any]) -> mx.array:
    """`GlobalAttentionFeatureEncoder`."""
    x = bl.conv(bl.basic(x, e["agg"][0]), e["agg"][1])
    x = x + _outlook_attention(x, e["outlook"])
    x = x + _call_ffn(x, e["outlook_ffn"])
    x = x + _pos(x)
    s = e["splits"]
    x = x + _window_attention(x, e["win"], s, shift=False)
    x = x + _call_ffn(x, e["win_ffn"])
    x = x + _window_attention(x, e["swin"], s, shift=True)
    return x + _call_ffn(x, e["swin_ffn"])


def _softmax_d(s: mx.array) -> mx.array:
    """`volume_softmax`'s softmax over D (axis 1) + eps, renormalised."""
    p = mx.softmax(s, axis=1) + EPS
    return p / p.sum(axis=1, keepdims=True)


def _vtb_fwd(t: mx.array, feats: mx.array, m: dict[str, Any]) -> mx.array:
    """`VolumeTransformerBlock(t [1, D, H, W, C], feats [1, H, W, C])`, reassociated as upstream."""
    c, dt = t.shape[-1], t.dtype
    kv = _mm(_ln(feats, m["ctx"]), m["kv"]).astype(F32)
    k, v = kv[..., :c], kv[..., c:]
    g = (k @ m["q"]) * m["gamma"]  # [1, H, W, C]
    vo = v @ m["wo"]
    s = (mx.fast.layer_norm(t.astype(F32), None, None, m["eps"]) * g[:, None]).sum(axis=-1)
    p = _softmax_d(s)
    t = (t.astype(F32) + p[..., None] * vo[:, None] + m["bo"]).astype(dt)
    n, a, b = m["ffn"]
    return t + _mm(_silu(_mm(_ln(t, n), a)), b)


class MlxGREAT(MlxIgevUpdate):
    """`forward_fast` of `net` (a loaded `GREATStereo`, float32 weights) in `dtype` (the name)."""

    def _family(self, net: Any, w: bl.Weights) -> None:  # noqa: ANN401 - `GREATStereo`
        super()._family(net, w)
        eav = net.excitive_attention_volume
        g = eav.global_spatial_refiner
        self.agr: dict[str, Any] = {
            "pre": w.conv(g.pre_channel_agg[0]),
            "cam": (_lin(w, g.cam_conv[0]), _lin(w, g.cam_conv[2])),
            "sam": w.conv(g.sam_conv),
            "small": w.basic(g.small_reception_conv),
            "large": w.basic(g.large_repcetion_conv),
            "fuse": (w.basic(g.final_fusion[0]), _lin(w, g.final_fusion[1])),
        }
        r = eav.global_matching_refiner
        self.refiner = [
            (_attn(w, b.cross_attn), _ffn(w, b.feedforward))
            for b in (getattr(r, f"refinement_{i + 1}") for i in range(r.depth))
        ]
        gc = eav.global_context_refiner
        self.encoders = [_encoder(w, getattr(gc, f"feature_encoder_{s}")) for s in (4, 8, 16, 32)]
        cc0 = eav.cc_stem[0]
        c = int(cc0.conv.weight.shape[0])
        full, b0 = w.pair(cc0.conv, cc0.bn, (0, 2, 3, 4, 1))  # [O, 1, 3, 3, 2C]
        wr = full[:, 0, :, :, c:]
        self.cc0 = (full[:, 0, :, :, :c], wr, wr[:, :, 2:3], b0)  # left, right, its dx = +1 column, bias
        self.cc = [w.basic(eav.cc_stem[1]), w.basic(eav.cc_stem[2])]
        self.feat_stem = (w.basic(eav.feat_stem[0]), w.conv(eav.feat_stem[1]))
        fw, fb, _, _ = w.conv(eav.fuse_stem.conv, eav.fuse_stem.bn)  # (1, 5, 5): a 2-D convolution per plane
        self.fuse = (fw[:, 0], fb)
        agg = eav.eav_agg
        self.agg: dict[str, Any] = {  # BasicConvs, and lists of them for the `nn.Sequential`s
            n: [w.basic(x) for x in sub] if hasattr(sub, "__getitem__") else w.basic(sub)
            for n, sub in agg.named_children()
            if not n.startswith("feature_attn")
        }
        self.vtb = {n: _vtb(w, sub) for n, sub in agg.named_children() if n.startswith("feature_attn")}
        self._compile()

    def _compile(self) -> None:
        """One compiled function per volume transformer block, feed-forward layer and volume chunk filter."""
        for m in self.vtb.values():
            m["fn"] = mx.compile(functools.partial(_vtb_fwd, m=m))
        for f in [f for _, f in self.refiner] + [e[k] for e in self.encoders for k in e if k.endswith("_ffn")]:
            f["fn"] = mx.compile(functools.partial(_ffn_fwd, f=f))
        self._cc_fn = mx.compile(self._cc_chunk)
        self._fuse_fn = mx.compile(self._fuse_chunk)
        self._hg_fn = mx.compile(lambda x, *feats: self._hourglass(x, list(feats)))

    def _cc_chunk(self, vol: mx.array) -> mx.array:
        """`cc_stem[1:]` of a chunk of concat planes `[1, d, H, W, C]`."""
        for m in self.cc:
            vol = bl.basic(vol, m)
        return vol

    def _fuse_chunk(self, vol: mx.array, feat: mx.array) -> mx.array:
        """`feat_stem` excitation and `fuse_stem` of the filtered planes `[d, H, W, 8]`: `[d, H, W, 1]`."""
        d, h, w, _ = vol.shape
        x = (vol * feat).reshape(d, h, w, -1)
        return bl.leaky(mx.conv2d(x, self.fuse[0], padding=2) + self.fuse[1])

    # matching features ------------------------------------------------------------------------------------------------

    def _match(self, f4: mx.array) -> mx.array:
        """`AdaptiveGlobalRefiner` of the 1/4 features `[1, H, W, 96]`."""
        a = self.agr
        x = f4
        f = bl.relu(bl.inorm(bl.conv(x, a["pre"])))
        ff = f.astype(F32)
        pooled = mx.concatenate([ff.mean(axis=(1, 2), keepdims=True), ff.max(axis=(1, 2), keepdims=True)], axis=0)
        ca = _mm(bl.relu(_mm(pooled.astype(f.dtype), a["cam"][0])), a["cam"][1])
        f = f * mx.sigmoid(ca[:1] + ca[1:])
        ff = f.astype(F32)
        sm = mx.concatenate([ff.mean(axis=-1, keepdims=True), ff.max(axis=-1, keepdims=True)], axis=-1)
        m = mx.sigmoid(bl.conv(sm.astype(f.dtype), a["sam"]))
        small, large = bl.basic(f, a["small"]), bl.basic(f, a["large"])
        out = _mm(bl.basic(small * m + large * (1 - m) + x, a["fuse"][0]), a["fuse"][1])
        return mlx_port.tap("excitive_attention_volume.global_spatial_refiner", out)

    def _refine(self, ml: mx.array, mr: mx.array) -> tuple[mx.array, mx.array]:
        """Positional embedding + `TransformerFeatureRefiner`: both views as one batch, each attending to the other."""
        pos = _pos(ml)
        a = mx.concatenate([ml + pos, mr + pos], axis=0)
        for attn, ffn in self.refiner:
            a = a + _row_attention(a, mx.concatenate([a[1:], a[:1]], axis=0), attn)
            a = a + ffn["fn"](a)
            mx.eval(a)
        return mlx_port.tap("global_matching_refiner[0]", a[:1]), mlx_port.tap("global_matching_refiner[1]", a[1:])

    # cost volume ------------------------------------------------------------------------------------------------------

    def _cost_volume(self, left: mx.array, right: mx.array, ctx0: mx.array) -> mx.array:
        """`ExcitiveAttentionVolume.cost_volume`: `[1, D, H, W, 8]`, built and filtered in disparity chunks."""
        _, h, w, c = left.shape
        d = self.depth
        wl, wr, we, b0 = self.cc0
        lc = mx.conv2d(left, wl, padding=1) + b0  # the left half of every plane
        # the right half: plane `d` column `x` is `y[x - d + 1]` (zero left of it), but its column `w - 1` reads the
        # right view's column `w - d` where the volume has its zero padding: `y - e` there
        y = mx.conv2d(mx.pad(right, [(0, 0), (1, 1), (2, 1), (0, 0)]), wr)  # [1, H, W + 1, C]
        z = y[:, :, :w] - mx.conv2d(mx.pad(right, [(0, 0), (1, 1), (0, 0), (0, 0)]), we)
        feat = bl.conv(bl.basic(ctx0, self.feat_stem[0]), self.feat_stem[1])
        mx.eval(lc, y, z, feat)

        def plane(i: int) -> mx.array:
            if i == 0:
                r = y[:, :, 1:]
            elif i > w:
                r = mx.zeros_like(lc)
            else:
                parts = [y[:, :, : w - i], z[:, :, w - i : w - i + 1]]
                if i > 1:
                    parts.insert(0, mx.zeros((1, h, i - 1, c), lc.dtype))
                r = mx.concatenate(parts, axis=2)
            return bl.leaky(lc + r)

        step = max(1, CC_PLANES // (2 * c * h * w))
        outs = []
        for d0 in range(0, d, step):
            d1 = min(d, d0 + step)
            a, e = max(0, d0 - 2), min(d, d1 + 2)
            vol = self._cc_fn(mx.stack([plane(i)[0] for i in range(a, e)])[None])
            outs.append(self._fuse_fn(vol[0, d0 - a : d1 - a], feat)[None])
            mx.eval(outs[-1])
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)

    def _hourglass(self, x: mx.array, feats: list[mx.array]) -> mx.array:
        """`ExcitiveAttentionHourglassEncoder.hourglass`: the filtered volume `[1, D, H, W, 8]`."""
        m, vt = self.agg, self.vtb

        def seq(x: mx.array, name: str) -> mx.array:
            for c in m[name]:
                x = bl.basic(x, c)
            return x

        def up(x: mx.array, name: str) -> mx.array:
            return bl.basic(x, m[name], transposed=True)

        conv0 = vt["feature_attn_4"]["fn"](bl.basic(x, m["conv0"]), feats[0])
        conv1 = vt["feature_attn_8"]["fn"](seq(conv0, "conv1"), feats[1])
        conv2 = vt["feature_attn_16"]["fn"](seq(conv1, "conv2"), feats[2])
        conv3 = vt["feature_attn_32"]["fn"](seq(conv2, "conv3"), feats[3])
        y = mx.concatenate([up(conv3, "conv3_up"), conv2], axis=-1)
        conv2 = vt["feature_attn_up_16"]["fn"](seq(y, "agg_0"), feats[2])
        y = mx.concatenate([up(conv2, "conv2_up"), conv1], axis=-1)
        conv1 = vt["feature_attn_up_8"]["fn"](seq(y, "agg_1"), feats[1])
        return up(conv1, "conv1_up")

    def _geometry(
        self, match_l: mx.array, match_r: mx.array, feats: list[mx.array]
    ) -> tuple[list[mx.array], mx.array, list[mx.array]]:
        ml, mr = self._refine(match_l, match_r)
        context = []
        for f, e, s in zip(feats, self.encoders, (4, 8, 16, 32), strict=True):
            context.append(mlx_port.tap(f"global_context_refiner.feature_encoder_{s}", _encode(f, e)))
            mx.eval(context[-1])
        vol = self._cost_volume(ml, mr, context[0])
        geo = mlx_port.tap("eav_agg.conv1_up", self._hg_fn(vol, *context))
        del vol
        mx.eval(geo)
        t = geo.astype(F32)
        g = context[0].astype(F32) * 8**-0.5
        p = _softmax_d((t * g[:, None]).sum(axis=-1))
        disp = (p * mx.arange(self.depth, dtype=F32)[:, None, None]).sum(axis=1)[..., None]
        lookups = bl.GeoLookup(ml, mr, geo, self.levels, self.radius, self.dtype).state()
        mx.eval(disp, lookups)
        return lookups, disp, []
