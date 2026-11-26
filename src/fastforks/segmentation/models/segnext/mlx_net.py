"""SegNeXt on Apple MLX, written in `mlx.core` operations (`MlxSegNeXt(net, dtype)`): a normalised image -> logits.

`EncoderDecoder.forward` of the folded network (`net.fold`), channels last: one compiled function per encoder block
and three for the decode head, evaluated one in flight. Written out instead of replayed from an exported graph: the
export of MSCAN-B's 21 blocks (MSCAN-L: 38) was 1.5-2.3 s of a first call whose network runs in 0.1-0.2 s, and the
shared depthwise kernel is one Metal shader per layer shape (32 here).

- a block's seven strip convolutions are two kernels: the 7, 11 and 21 tap rows in one pass over the 5 x 5 conv's
  output, then their three columns, the sum and the skip in one pass (`a + conv0_2(..) + conv1_2(..) + conv2_2(..)`);
- the MLP's 3 x 3 depthwise conv has its GELU fused;
- the kernels read the map size and the channel count at run time, so the four of them are compiled once per dtype;
- the 1 x 1 convs are GEMMs with the bias fused (`mlx_graph.conv1x1`);
- the head's GroupNorms and the matrix factorisation run in float32, as the PyTorch runner's factorisation does.

Depthwise sums accumulate in float32 and are rounded once to the activation dtype.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np

from fastforks._core.mlx import metal_kernel, mx
from fastforks.kernels.graph_mlx.ops import _ERF4
from fastforks.shared.mlx_graph import conv1x1, conv1x1_ok
from fastforks.shared.mlx_seg import resize

from . import net as sn

__all__ = ["MlxSegNeXt"]

_STRIPS = (7, 11, 21)  # fixed: MSCA's strip lengths, as the kernels below hard-code their tap offsets
_PREAMBLE = """
        const int R = 4;  // outputs per thread
        uint c4 = thread_position_in_grid.x, tile = thread_position_in_grid.y, n = thread_position_in_grid.z;
        int H = x_shape[1], W = x_shape[2], C4 = x_shape[3] / 4;
        if (c4 >= C4) return;
        const device vec<T, 4>* x4 = (const device vec<T, 4>*)x;
        const device vec<T, 4>* w4 = (const device vec<T, 4>*)w;
        const device vec<T, 4>* b4 = (const device vec<T, 4>*)b;
"""

_SQUARE = metal_kernel(
    name="segnext_dw",
    input_names=["x", "w", "b"],
    output_names=["out"],
    header=_ERF4,
    source=_PREAMBLE
    + """
        // one thread: 4 adjacent channels of R neighbouring outputs of one row of the K x K depthwise conv (stride 1,
        // same size), with the exact GELU when GELU is set
        int tiles = (W + R - 1) / R;
        int oh = tile / tiles, ow0 = (tile % tiles) * R;
        float4 acc[R];
        for (int r = 0; r < R; ++r) acc[r] = float4(b4[c4]);
        for (int i = 0; i < K; ++i) {
            int ih = oh - K / 2 + i;
            if (ih < 0 || ih >= H) continue;
            size_t row = (size_t(n) * H + ih) * W;
            float4 wv[K];
            for (int j = 0; j < K; ++j) wv[j] = float4(w4[(i * K + j) * C4 + c4]);
            for (int t = 0; t < K + R - 1; ++t) {
                int iw = ow0 - K / 2 + t;
                if (iw < 0 || iw >= W) continue;
                float4 xv = float4(x4[(row + iw) * C4 + c4]);
                for (int r = 0; r < R; ++r) {
                    int j = t - r;
                    if (j >= 0 && j < K) acc[r] += xv * wv[j];
                }
            }
        }
        for (int r = 0; r < R; ++r) {
            if (ow0 + r >= W) break;
            float4 v = acc[r];
            if (GELU) v = 0.5f * v * (1.0f + erf4(v * 0.70710678f));
            ((device vec<T, 4>*)out)[((size_t(n) * H + oh) * W + ow0 + r) * C4 + c4] = vec<T, 4>(v);
        }
    """,
)

_ROWS = metal_kernel(
    name="segnext_strip_rows",
    input_names=["x", "w", "b"],
    output_names=["h7", "h11", "h21"],
    source=_PREAMBLE
    + """
        // one thread: 4 adjacent channels of R neighbouring outputs of one row of the three 1 x k depthwise convs
        // (`w` holds their 7 + 11 + 21 taps, `b` their biases); every input pixel is loaded once for all of them
        int tiles = (W + R - 1) / R;
        int oh = tile / tiles, ow0 = (tile % tiles) * R;
        float4 wv[39];
        for (int j = 0; j < 39; ++j) wv[j] = float4(w4[j * C4 + c4]);
        float4 a7[R], a11[R], a21[R];
        for (int r = 0; r < R; ++r) {
            a7[r] = float4(b4[c4]);
            a11[r] = float4(b4[C4 + c4]);
            a21[r] = float4(b4[2 * C4 + c4]);
        }
        size_t row = (size_t(n) * H + oh) * W;
        for (int t = 0; t < 21 + R - 1; ++t) {
            int iw = ow0 - 10 + t;
            if (iw < 0 || iw >= W) continue;
            float4 xv = float4(x4[(row + iw) * C4 + c4]);
            for (int r = 0; r < R; ++r) {
                int j = t - r;  // the tap of the 21 strip; the shorter strips are centred on it
                if (j >= 0 && j < 21) a21[r] += xv * wv[18 + j];
                if (j >= 5 && j < 16) a11[r] += xv * wv[2 + j];
                if (j >= 7 && j < 14) a7[r] += xv * wv[j - 7];
            }
        }
        for (int r = 0; r < R; ++r) {
            if (ow0 + r >= W) break;
            size_t o = (row + ow0 + r) * C4 + c4;
            ((device vec<T, 4>*)h7)[o] = vec<T, 4>(a7[r]);
            ((device vec<T, 4>*)h11)[o] = vec<T, 4>(a11[r]);
            ((device vec<T, 4>*)h21)[o] = vec<T, 4>(a21[r]);
        }
    """,
)

_COLUMNS = metal_kernel(
    name="segnext_strip_columns",
    input_names=["x", "h7", "h11", "h21", "w", "b"],
    output_names=["out"],
    source=_PREAMBLE
    + """
        // one thread: 4 adjacent channels of R neighbouring outputs of one column of
        // x + conv7x1(h7) + conv11x1(h11) + conv21x1(h21) (`w` holds the 7 + 11 + 21 taps, `b` the summed biases)
        int ow = tile % W, oh0 = (tile / W) * R;
        const device vec<T, 4>* p7 = (const device vec<T, 4>*)h7;
        const device vec<T, 4>* p11 = (const device vec<T, 4>*)h11;
        const device vec<T, 4>* p21 = (const device vec<T, 4>*)h21;
        float4 wv[39];
        for (int j = 0; j < 39; ++j) wv[j] = float4(w4[j * C4 + c4]);
        float4 acc[R];
        for (int r = 0; r < R; ++r) {
            acc[r] = float4(b4[c4]);
            if (oh0 + r < H) acc[r] += float4(x4[((size_t(n) * H + oh0 + r) * W + ow) * C4 + c4]);
        }
        for (int t = 0; t < 21 + R - 1; ++t) {
            int ih = oh0 - 10 + t;
            if (ih < 0 || ih >= H) continue;
            size_t p = ((size_t(n) * H + ih) * W + ow) * C4 + c4;
            float4 v21 = float4(p21[p]);
            float4 v11 = (t >= 5 && t < 16 + R - 1) ? float4(p11[p]) : float4(0.0f);
            float4 v7 = (t >= 7 && t < 14 + R - 1) ? float4(p7[p]) : float4(0.0f);
            for (int r = 0; r < R; ++r) {
                int j = t - r;
                if (j >= 0 && j < 21) acc[r] += v21 * wv[18 + j];
                if (j >= 5 && j < 16) acc[r] += v11 * wv[2 + j];
                if (j >= 7 && j < 14) acc[r] += v7 * wv[j - 7];
            }
        }
        for (int r = 0; r < R; ++r) {
            if (oh0 + r >= H) break;
            ((device vec<T, 4>*)out)[((size_t(n) * H + oh0 + r) * W + ow) * C4 + c4] = vec<T, 4>(acc[r]);
        }
    """,
)


def _grid(c: int, tiles: int, n: int) -> dict[str, tuple[int, int, int]]:
    """Grid and threadgroup of a kernel with one thread per 4 channels of a tile."""
    c4 = c // 4
    tg = min(c4, 32)
    return {"grid": (-(-c4 // tg) * tg, tiles, n), "threadgroup": (tg, 1, 1)}


def _square(x: mx.array, w: mx.array, b: mx.array, *, gelu: bool = False) -> mx.array:
    """The K x K depthwise conv of NHWC `x` (stride 1, same size; `w` `[K * K, C]`), then the exact GELU if `gelu`."""
    n, h, wd, c = x.shape
    return _SQUARE(
        inputs=[x, w, b],
        template=[("T", x.dtype), ("K", math.isqrt(w.shape[0])), ("GELU", int(gelu))],
        **_grid(c, h * -(-wd // 4), n),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def _strips(x: mx.array, wh: mx.array, bh: mx.array, wv: mx.array, bv: mx.array) -> mx.array:
    """`x` plus its three strip-conv branches (`1 x k` then `k x 1` for k = 7, 11, 21): two kernels."""
    n, h, wd, c = x.shape
    rows = _ROWS(
        inputs=[x, wh, bh],
        template=[("T", x.dtype)],
        **_grid(c, h * -(-wd // 4), n),
        output_shapes=[x.shape] * 3,
        output_dtypes=[x.dtype] * 3,
    )
    return _COLUMNS(
        inputs=[x, *rows, wv, bv],
        template=[("T", x.dtype)],
        **_grid(c, wd * -(-h // 4), n),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def _gelu(x: mx.array) -> mx.array:
    y = x.astype(mx.float32)
    return (0.5 * y * (1 + mx.erf(y * (1 / math.sqrt(2))))).astype(x.dtype)


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _nmf(x: mx.array, bases: mx.array) -> mx.array:
    """`net.nmf` on the channels-last view: `x` `[B, N, D]` (>= 0), start `bases` `[1, D, R]` -> `[B, N, D]`."""
    xt = x.swapaxes(1, 2)

    def coefficients(coef: mx.array, bases: mx.array) -> mx.array:
        return coef * (x @ bases) / (coef @ (bases.swapaxes(1, 2) @ bases) + sn.NMF_EPS)

    coef = mx.softmax(x @ bases, axis=-1)
    for _ in range(sn.NMF_STEPS):
        coef = coefficients(coef, bases)
        bases = bases * (xt @ coef) / (bases @ (coef.swapaxes(1, 2) @ coef) + sn.NMF_EPS)
    return coefficients(coef, bases) @ bases.swapaxes(1, 2)


_Pointwise = tuple[bool, mx.array, mx.array | None]  # GEMM or conv, weight ([in, out] or [out, 1, 1, in]), bias


SYNC_BYTES = 8 << 20
"""An encoder step whose output is at least this large also waits for itself (`_forward`)."""


class MlxSegNeXt:
    """The folded network `net` (eval, float32 weights) in `dtype` (the name): one compiled function per input shape."""

    def __init__(self, net: sn.EncoderDecoder, dtype: str) -> None:
        """Convert the weights of `net`."""
        self.dtype = getattr(mx, dtype)
        b: Any = net.backbone
        stem = b.patch_embed1.proj
        self.embeds = [[self._conv(stem[0]), self._conv(stem[3])]]
        self.embeds += [[self._conv(getattr(b, f"patch_embed{i}").proj)] for i in range(2, b.stages + 1)]
        self.stages = [[self._block(blk) for blk in getattr(b, f"block{i}")] for i in range(1, b.stages + 1)]
        self.norms = [self._norm(getattr(b, f"norm{i}")) for i in range(1, b.stages + 1)]
        head = net.decode_head
        self.squeeze = self._module(head.squeeze)
        self.ham_in = self._module(head.hamburger.ham_in)
        self.ham_out = self._module(head.hamburger.ham_out)
        self.bases = mx.array(head.hamburger.bases.float().numpy())
        self.align = self._module(head.align)
        self.classifier = self._pointwise(head.conv_seg)
        # the casts at once: a lazy `astype` keeps its float32 source resident and is redone inside every call
        mx.eval(self.embeds, self.stages, self.norms, self.squeeze, self.ham_in, self.ham_out, self.bases)
        mx.eval(self.align, self.classifier)
        # one compiled function per block (its weights as constants) and three for the head, evaluated one in flight
        self._blocks = [[mx.compile(functools.partial(self._run_block, p=p)) for p in st] for st in self.stages]
        self._squeeze_c, self._ham_c, self._out_c = (
            mx.compile(f) for f in (self._squeeze_step, self._ham_step, self._out_step)
        )
        # the patch embeddings and stage norms compiled too: uncompiled, the float32 GELU between the stem's two
        # convolutions was a chain of full-resolution float32 buffers
        self._embeds = [mx.compile(functools.partial(self._embed, convs=e)) for e in self.embeds]
        self._norms = [mx.compile(functools.partial(self._stage_norm, wbe=n)) for n in self.norms]

    def _a(self, t: Any) -> mx.array:  # noqa: ANN401 - a torch tensor
        return mx.array(np.ascontiguousarray(t.detach().float().numpy())).astype(self.dtype)

    def _conv(self, conv: Any) -> tuple[mx.array, mx.array]:  # noqa: ANN401 - torch modules
        return self._a(conv.weight.permute(0, 2, 3, 1)), self._a(conv.bias)

    def _pointwise(self, conv: Any) -> _Pointwise:  # noqa: ANN401
        cout, cin = conv.weight.shape[:2]
        bias = None if conv.bias is None else self._a(conv.bias)
        if conv1x1_ok(cin, cout, 0, (1, 1), (1, 1), (0, 0), 1):
            return True, self._a(conv.weight.reshape(cout, cin).T), bias
        return False, self._a(conv.weight.permute(0, 2, 3, 1)), bias

    def _depthwise(self, conv: Any) -> mx.array:  # noqa: ANN401
        """`[C, 1, kh, kw]` as `[kh * kw, C]`."""
        return self._a(conv.weight.flatten(1).T)

    def _block(self, blk: Any) -> dict[str, Any]:  # noqa: ANN401
        sgu, mlp = blk.attn.spatial_gating_unit, blk.mlp
        rows = [getattr(sgu, f"conv{i}_1") for i in range(len(_STRIPS))]
        columns = [getattr(sgu, f"conv{i}_2") for i in range(len(_STRIPS))]
        assert tuple(m.kernel_size[1] for m in rows) == _STRIPS  # noqa: S101 - the kernels' tap layout
        return {
            "proj_1": self._pointwise(blk.attn.proj_1),
            "conv0": (self._depthwise(sgu.conv0), self._a(sgu.conv0.bias)),
            "rows": (
                mx.concatenate([self._depthwise(m) for m in rows]),
                mx.concatenate([self._a(m.bias) for m in rows]),
            ),
            "columns": (
                mx.concatenate([self._depthwise(m) for m in columns]),
                self._a(sum(m.bias.detach().double() for m in columns)),
            ),
            "conv3": self._pointwise(sgu.conv3),
            "proj_2": self._pointwise(blk.attn.proj_2),
            "res_scale": self._a(blk.attn.res_scale.flatten()),
            "fc1": self._pointwise(mlp.fc1),
            "dwconv": (self._depthwise(mlp.dwconv), self._a(mlp.dwconv.bias)),
            "fc2": self._pointwise(mlp.fc2),
        }

    @staticmethod
    def _norm(norm: Any) -> tuple[mx.array, mx.array, float]:  # noqa: ANN401
        return mx.array(norm.weight.detach().float().numpy()), mx.array(norm.bias.detach().float().numpy()), norm.eps

    def _module(self, m: Any) -> tuple[_Pointwise, tuple[mx.array, mx.array, float] | None]:  # noqa: ANN401
        """A head `ConvModule`: its conv and its GroupNorm (None without)."""
        return self._pointwise(m.conv), (self._norm(m.gn) if m.norm else None)

    # ------------------------------------------------------------------------------------------------ forward
    @staticmethod
    def _pw(x: mx.array, p: _Pointwise) -> mx.array:
        gemm, w, b = p
        if gemm:
            return conv1x1(x, w, b)
        y = mx.conv2d(x, w)
        return y if b is None else y + b

    def _run_block(self, x: mx.array, p: dict[str, Any]) -> mx.array:
        y = _gelu(self._pw(x, p["proj_1"]))
        a = _strips(_square(y, *p["conv0"]), *p["rows"], *p["columns"])
        x = self._pw(self._pw(a, p["conv3"]) * y, p["proj_2"]) + x * p["res_scale"]
        return x + self._pw(_square(self._pw(x, p["fc1"]), *p["dwconv"], gelu=True), p["fc2"])

    def _group_norm(self, x: mx.array, norm: tuple[mx.array, mx.array, float]) -> mx.array:
        """GroupNorm of NHWC `x` with float32 statistics."""
        w, b, eps = norm
        n, h, wd, c = x.shape
        g = x.astype(mx.float32).reshape(n, h * wd, sn.GROUPS, c // sn.GROUPS)
        mean = mx.mean(g, axis=(1, 3), keepdims=True)
        var = mx.mean(mx.square(g - mean), axis=(1, 3), keepdims=True)
        return (((g - mean) * mx.rsqrt(var + eps)).reshape(x.shape) * w + b).astype(self.dtype)

    def _conv_module(self, x: mx.array, m: tuple[_Pointwise, tuple[mx.array, mx.array, float] | None]) -> mx.array:
        x = self._pw(x, m[0])
        return x if m[1] is None else self._group_norm(x, m[1])

    def _forward(self, image: mx.array) -> mx.array:
        """The encoder block by block, one evaluation in flight (`step`), then the head (`_decode`).

        One compiled function over the whole network holds every block's transients until its command buffer
        completes (MSCAN-T float16 at 1024 x 2048: 249 MB against the graph translation's 176). A step whose output is
        at least `SYNC_BYTES` also waits for itself: two full-size blocks in flight held both's transients (MSCAN-S
        float32 at 1024 x 2048: 467 MB, one at a time 267, the graph translation 313).
        """
        x = image.transpose(0, 2, 3, 1).astype(self.dtype)
        feats, pending = [], []

        def step(y: mx.array) -> mx.array:
            mx.async_eval(y)
            if pending:
                mx.eval(pending.pop())
            if y.nbytes >= SYNC_BYTES:
                mx.eval(y)
            else:
                pending.append(y)
            return y

        for embed, blocks, norm in zip(self._embeds, self._blocks, self._norms, strict=True):
            x = step(embed(x))
            for block in blocks:
                x = step(block(x))
            x = step(norm(x))
            feats.append(x)
        return self._decode(feats)

    @staticmethod
    def _embed(x: mx.array, convs: list[tuple[mx.array, mx.array]]) -> mx.array:
        """A stage's patch embedding: its stride-2 3 x 3 convolutions (a GELU between two)."""
        for i, (cw, cb) in enumerate(convs):
            x = mx.conv2d(_gelu(x) if i else x, cw, stride=2, padding=1) + cb
        return x

    def _stage_norm(self, x: mx.array, wbe: tuple[mx.array, mx.array, float]) -> mx.array:
        """A stage's closing layer norm (float32)."""
        w, b, eps = wbe
        return mx.fast.layer_norm(x.astype(mx.float32), w, b, eps).astype(self.dtype)

    def _decode(self, feats: list[mx.array]) -> mx.array:
        """The decode head on the four stages' maps, in three compiled parts, each evaluated before the next.

        As one compiled head its transients were alive at once until it completed: MSCAN-T float16 at
        1024 x 2048, 188 MB against the encoder blocks' 66 at most.
        """
        x = self._squeeze_c(feats)
        mx.eval(x)
        y = self._ham_c(x)
        mx.eval(y)
        return self._out_c(x, y)

    def _squeeze_step(self, feats: list[mx.array]) -> mx.array:
        size = feats[1].shape[1:3]
        x = mx.concatenate([feats[1]] + [resize(f, size).astype(self.dtype) for f in feats[2:]], axis=-1)
        return _relu(self._conv_module(x, self.squeeze))

    def _ham_step(self, x: mx.array) -> mx.array:
        n, h, wd, c = x.shape
        y = _relu(self._conv_module(x, self.ham_in)).astype(mx.float32).reshape(n, h * wd, c)
        return _nmf(y, self.bases).reshape(x.shape).astype(self.dtype)

    def _out_step(self, x: mx.array, y: mx.array) -> mx.array:
        x = _relu(x + self._conv_module(y, self.ham_out))
        x = _relu(self._conv_module(x, self.align))
        return self._pw(x, self.classifier).astype(mx.float32).transpose(0, 3, 1, 2)

    def __call__(self, image: mx.array) -> mx.array:
        """Normalised image `[B, 3, H, W]` (sides divisible by 32) -> float32 logits `[B, classes, H / 8, W / 8]`."""
        return self._forward(image)
