"""ViTMatte on Apple MLX, hand-written: NHWC, the run dtype.

`VitMatteForImageMatting.forward` on the padded `[1, H, W, 4]` input (normalised RGB, trimap):

- the ViTDet backbone: the patch embedding as one matrix product, the position embedding resized bicubically to the
  token grid (float32, cached per grid); each layer one step. Attention is MLX's fused SDPA at the head dim (64) with
  upstream's decomposed relative positions `rel_h[q, kh] + rel_w[q, kw]` (from the unscaled queries) as an additive
  mask; a global layer builds it in query-row chunks of at most `MASK_BYTES` (`mx.depends`), so its N x N matrix is
  never resident at once. The relative tables are interpolated linearly to the grid in numpy (cached per grid).
  Windowed layers partition the normed map, zero-padded to whole 14 x 14 windows (the padded tokens are keys, as
  upstream's). LayerNorms are `mx.fast.layer_norm` (float32 statistics), GELU exact in float32;
- the residual bottlenecks (1 x 1, LN, GELU, 3 x 3, LN, GELU, 1 x 1, LN) channels-last, the LayerNorm2d being a
  LayerNorm over channels;
- the detail capture: conv + BatchNorm (folded, `shared.mlx_rsu`) + ReLU stages, the fusions' 2x bilinear upsampling
  (`align_corners=False`), the head and its sigmoid; from 1024 x 1024 pixels the last fusion and the head run in row
  strips (upstream `matte_in_strips`), each one step. The other fusions run alone (not pipelined with the step
  before), in row strips of at most `FUSION_VALUES` input values: vitmatte-s float32 at 1088 x 1920, the call's
  peak 2.2 GB pipelined and whole, 1.2 GB so, at the same speed.
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

from .net import vitmatte as tnet

if TYPE_CHECKING:
    from collections.abc import Callable

    Layer = Callable[[mx.array], mx.array]

LAYOUT = "vitmatte-v1"
SOURCES = (Path(__file__), Path(tnet.__file__), Path(mlx_rsu.__file__), *mlx_swin_net.SOURCES)
MASK_BYTES = 64 << 20  # bound of one global layer's mask chunk; the result does not depend on it
STRIP_PIXELS = 1024 * 1024  # upstream `matte_in_strips`: from this many pixels the matting tail runs in strips
STRIP_ROW_PIXELS = 128 * 1024  # upstream's strip height: this many pixels (at least 32 rows of the half map)
FUSION_VALUES = 32 << 20  # bound of a fusion strip's input values (`d`, the upsampled map); the result is the same

_np, _arr, _lin = mlx_swin_net.np64, mlx_swin_net.arr, mlx_swin_net.lin


def _gelu(x: mx.array) -> mx.array:
    """Exact (erf) GELU computed in float32."""
    y = x.astype(mx.float32)
    return (y * (0.5 + 0.5 * mx.erf(y * 0.7071067811865476))).astype(x.dtype)


def _ln(m: Any, dtype: mx.Dtype, eps: float) -> Layer:
    w, b = mlx_swin_net.norm(m, dtype)
    return lambda x: mx.fast.layer_norm(x, w, b, eps)


def _mat(conv: Any, dtype: mx.Dtype) -> mx.array:
    """A bias-free 1 x 1 `Conv2d` as `[in, out]`."""
    o, i = (int(d) for d in conv.weight.shape[:2])
    return _arr((i, o), dtype, lambda: _np(conv.weight).reshape(o, i).T)


def _lerp(n: int, out: int) -> np.ndarray:
    """`[out, n]` weights of `F.interpolate(mode="linear")` (`align_corners=False`)."""
    src = np.maximum((np.arange(out) + 0.5) * (n / out) - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(int), n - 1)
    i1, lam, rows = np.minimum(i0 + 1, n - 1), src - i0, np.arange(out)
    m = np.zeros((out, n))
    np.add.at(m, (rows, i0), 1.0 - lam)
    np.add.at(m, (rows, i1), lam)
    return m


def rel_table(table: np.ndarray, size: int) -> np.ndarray:
    """Upstream `get_rel_pos(size, size, table)`: `[size, size, d]` (the table resized linearly to `2 size - 1`)."""
    n = 2 * size - 1
    if table.shape[0] != n:
        table = _lerp(table.shape[0], n) @ table
    idx = np.arange(size)[:, None] - np.arange(size)[None, :] + size - 1
    return table[idx]


def _attend(x: mx.array, qkv: Any, proj: Any, heads: int, rel: tuple[mx.array, mx.array]) -> mx.array:
    """Upstream `VitDetAttention.forward`: `[B, h, w, C]` -> `[B, h, w, C]` (`rel`: tables `[h, h, d]`, `[w, w, d]`)."""
    rh, rw = rel
    b, h, w, c = x.shape
    d, n = c // heads, h * w
    t = mx.addmm(qkv[1], x.reshape(-1, c), qkv[0]).reshape(b, h, w, 3, heads, d)
    q, k, v = t[..., 0, :, :], t[..., 1, :, :], t[..., 2, :, :]
    qh = q.transpose(1, 0, 2, 3, 4).reshape(h, b * w * heads, d)
    rel_h = (qh @ rh.transpose(0, 2, 1)).reshape(h, b, w, heads, h).transpose(1, 3, 0, 2, 4)
    qw = q.transpose(2, 0, 1, 3, 4).reshape(w, b * h * heads, d)
    rel_w = (qw @ rw.transpose(0, 2, 1)).reshape(w, b, h, heads, w).transpose(1, 3, 2, 0, 4)
    q, k, v = (a.reshape(b, n, heads, d).transpose(0, 2, 1, 3) for a in (q, k, v))
    rows = max(1, min(h, MASK_BYTES // (b * heads * w * n * x.itemsize)))
    outs: list[mx.array] = []
    for r in range(0, h, rows):
        mask = (rel_h[:, :, r : r + rows, :, :, None] + rel_w[:, :, r : r + rows, :, None, :]).reshape(b, heads, -1, n)
        qi = q[:, :, r * w : (r + rows) * w]
        if outs:
            qi = mx.depends(qi, outs[-1])
        outs.append(mx.fast.scaled_dot_product_attention(qi, k, v, scale=d**-0.5, mask=mask))
    o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
    return mx.addmm(proj[1], o.transpose(0, 2, 1, 3).reshape(-1, c), proj[0]).reshape(b, h, w, c)


class _Layer:
    """A `VitDetLayer` on `[h, w, C]` tokens (with its residual bottleneck, if it has one)."""

    def __init__(self, layer: Any, dtype: mx.Dtype, eps: float) -> None:
        a = layer.attention
        self.ws, self.heads = int(layer.window_size), int(a.num_heads)
        self.norm1, self.norm2 = _ln(layer.norm1, dtype, eps), _ln(layer.norm2, dtype, eps)
        self.qkv, self.proj = _lin(a.qkv, dtype), _lin(a.proj, dtype)
        self.fc1, self.fc2 = _lin(layer.mlp.fc1, dtype), _lin(layer.mlp.fc2, dtype)
        # the relative tables in float32 (numpy reads them per token grid)
        self.rel = tuple(_arr(tuple(p.shape), mx.float32, lambda p=p: _np(p)) for p in (a.rel_pos_h, a.rel_pos_w))
        self.res = None
        if layer.use_residual_block:
            r = layer.residual
            self.res = (
                _mat(r.conv1, dtype), _ln(r.norm1, dtype, r.norm1.eps), mlx_rsu.conv(r.conv2, None, dtype),
                _ln(r.norm2, dtype, r.norm2.eps), _mat(r.conv3, dtype), _ln(r.norm3, dtype, r.norm3.eps),
            )  # fmt: skip
        self.step = csp.stepped(self._forward)

    def tables(self, h: int, w: int, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        """The relative tables `[h, h, d]`, `[w, w, d]` of a window or the token grid, in `dtype`."""
        sh, sw = (self.ws, self.ws) if self.ws else (h, w)
        th, tw = (np.asarray(t.astype(mx.float32), dtype=np.float64) for t in self.rel)
        out = tuple(mx.array(rel_table(t, s).astype(np.float32)).astype(dtype) for t, s in ((th, sh), (tw, sw)))
        mx.eval(out)
        return out[0], out[1]

    def _attention(self, y: mx.array, rh: mx.array, rw: mx.array) -> mx.array:
        h, w, c = y.shape
        ws = self.ws
        if not ws:
            return _attend(y[None], self.qkv, self.proj, self.heads, (rh, rw))[0]
        ph, pw = -h % ws, -w % ws
        if ph or pw:
            y = mx.pad(y, [(0, ph), (0, pw), (0, 0)])
        nh, nw = (h + ph) // ws, (w + pw) // ws
        y = y.reshape(nh, ws, nw, ws, c).transpose(0, 2, 1, 3, 4).reshape(-1, ws, ws, c)
        y = _attend(y, self.qkv, self.proj, self.heads, (rh, rw))
        y = y.reshape(nh, nw, ws, ws, c).transpose(0, 2, 1, 3, 4).reshape(h + ph, w + pw, c)
        return y[:h, :w] if ph or pw else y

    def _forward(self, x: mx.array, rh: mx.array, rw: mx.array) -> mx.array:
        h, w, c = x.shape
        x = x + self._attention(self.norm1(x), rh, rw)
        rows = x.reshape(-1, c)
        rows = rows + mx.addmm(self.fc2[1], _gelu(mx.addmm(self.fc1[1], self.norm2(rows), self.fc1[0])), self.fc2[0])
        if self.res is not None:
            w1, n1, conv2, n2, w3, n3 = self.res
            y = _gelu(n1(rows @ w1)).reshape(1, h, w, -1)
            y = _gelu(n2(conv2(y))).reshape(h * w, -1)
            rows = rows + n3(y @ w3)
        return rows.reshape(h, w, c)


def _conv_relu(m: Any, dtype: mx.Dtype) -> Layer:
    """`VitMatteBasicConv3x3`: ReLU(BatchNorm(conv(x))), the BatchNorm folded."""
    f = mlx_rsu.conv(m.conv, m.batch_norm, dtype)
    return lambda x: mx.maximum(f(x), 0)


def _alone(f: Callable[..., mx.array]) -> Callable[..., mx.array]:
    """`f` compiled as a step that runs alone: it starts once the step in flight is done (`csp.stepped` overlaps them).

    The fusions at the high resolutions are the call's largest steps (vitmatte-s float32 at 1088 x 1920, whole: 0.8-1.3
    GB each); pipelined, two of them were resident at once.
    """
    g = mx.compile(f)

    def run(*args: mx.array) -> mx.array:
        if csp._PENDING:  # noqa: SLF001 - the shared pipeline
            mx.eval(csp._PENDING.pop())  # noqa: SLF001
        y = g(*args)
        mx.async_eval(y)
        csp._PENDING.append(y)  # noqa: SLF001
        return y

    return run


def _up2(x: mx.array) -> mx.array:
    return mlx_rsu.up_like(x, (2 * x.shape[1], 2 * x.shape[2]))


def _in_strips(step: Callable[[mx.array, mx.array], mx.array], y: mx.array, d: mx.array, rows: int) -> mx.array:
    """`step(y, d)` (2x upsampled `y` with `d` at twice its size, 3 x 3 convs) in strips of `rows` rows of `y`.

    Each strip reads two more rows of `y` on either side (the bilinear 2x and up to two 3 x 3 convs reach 1.5
    upsampled rows) and keeps its own rows: the result is the whole map's.
    """
    half = y.shape[1]
    if rows >= half:
        return step(y, d)
    out = []
    for start in range(0, half, rows):
        stop = min(start + rows, half)
        lo, hi = max(0, start - 2), min(half, stop + 2)
        z = step(y[:, lo:hi], d[:, 2 * lo : 2 * hi])
        out.append(z[:, 2 * (start - lo) : 2 * (stop - lo)])
    return mx.concatenate(out, axis=1)


class Network:
    """`VitMatteForImageMatting`: the padded `[1, H, W, 4]` float32 input -> alpha `[1, H, W, 1]` float32."""

    def __init__(self, net: Any, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = self.dtype = getattr(mx, dtype)
        cfg, bb = net.config, net.backbone
        emb = bb.embeddings
        proj = emb.projection
        c, cin, p, _ = (int(d) for d in proj.weight.shape)
        self.patch = p
        self.embed = (
            _arr((p * p * cin, c), dt, lambda: _np(proj.weight).transpose(2, 3, 1, 0).reshape(-1, c)),
            _arr((c,), dt, lambda: _np(proj.bias)),
        )
        pos = emb.position_embeddings
        self.pos = _arr(tuple(pos.shape), mx.float32, lambda: _np(pos))  # [1, 1 + S * S, C], the class token first
        self.layers = [_Layer(layer, dt, float(cfg.layer_norm_eps)) for layer in bb.encoder.layer]
        self.geometry: dict[tuple[int, int], tuple[mx.array, list[tuple[mx.array, mx.array]]]] = {}
        dec = net.decoder
        self.stream = [_conv_relu(m, dt) for m in dec.convstream.convs]
        self.fusions = [_conv_relu(f.conv, dt) for f in dec.fusion_blocks]
        head = dec.matting_head.matting_convs
        self.head = (mlx_rsu.conv(head[0], head[1], dt), mlx_rsu.conv(head[3], None, dt))
        self.embed_step = csp.stepped(self._embed)
        self.stream_steps = [csp.stepped(f) for f in self.stream]
        self.fusion_steps = [
            _alone(lambda y, d, f=f: f(mx.concatenate([d, _up2(y)], axis=-1))) for f in self.fusions[:-1]
        ]
        self.matte_step = csp.stepped(self._matte)

    def _grid(self, h: int, w: int) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """The position embedding `[h, w, C]` (float32) and every layer's relative tables for a token grid."""
        found = self.geometry.get((h, w))
        if found is None:
            pos = self.pos[0, 1:]
            s = math.isqrt(pos.shape[0])
            pos = pos.reshape(1, s, s, -1).transpose(0, 3, 1, 2)
            pos = mlx_matting.resize_cubic(pos, (h, w))[0].transpose(1, 2, 0)
            mx.eval(pos)
            found = pos, [layer.tables(h, w, self.dtype) for layer in self.layers]
            self.geometry[h, w] = found
        return found

    def _embed(self, x: mx.array, pos: mx.array) -> mx.array:
        _, hh, ww, cin = x.shape
        p = self.patch
        h, w = hh // p, ww // p
        t = x.astype(self.dtype).reshape(h, p, w, p, cin).transpose(0, 2, 1, 3, 4).reshape(h * w, -1)
        return (mx.addmm(self.embed[1], t, self.embed[0]).reshape(h, w, -1) + pos).astype(self.dtype)

    def _matte(self, y: mx.array, pixels: mx.array) -> mx.array:
        y = self.fusions[-1](mx.concatenate([pixels, _up2(y)], axis=-1))
        y = self.head[1](mx.maximum(self.head[0](y), 0))
        return mx.sigmoid(y.astype(mx.float32))

    def forward(self, x: mx.array) -> mx.array:
        """The alpha of the padded input, step by step."""
        _, hh, ww, _ = x.shape
        pos, rel = self._grid(hh // self.patch, ww // self.patch)
        t = self.embed_step(x, pos)
        for layer, (rh, rw) in zip(self.layers, rel, strict=True):
            t = layer.step(t, rh, rw)
        pixels = x.astype(self.dtype)
        details = [pixels]
        for step in self.stream_steps:
            details.append(step(details[-1]))
        y = t[None]
        for step in self.fusion_steps:
            d = details.pop()
            rows = FUSION_VALUES // (2 * d.shape[2] * (y.shape[3] + d.shape[3]))
            y = _in_strips(step, y, d, max(8, rows))
        if hh * ww < STRIP_PIXELS:
            return self.matte_step(y, pixels)
        return _in_strips(self.matte_step, y, pixels, max(32, STRIP_ROW_PIXELS // ww))

    def __call__(self, x: mx.array) -> mx.array:
        """`forward` with its steps, evaluated."""
        mlx_rsu.options()
        y = csp.steps(self.forward, x)
        mx.eval(y)
        return y


def network(net: Any, dtype: str) -> Network:
    """The MLX network of a `VitMatteForImageMatting` (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype)
