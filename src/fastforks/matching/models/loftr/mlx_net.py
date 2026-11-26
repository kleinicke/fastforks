"""LoFTR on Apple MLX, written in `mlx.core` operations (`MlxLoFTR`): an RGB pair -> matched points and confidences.

`model.Matcher.forward` (grayscale, resize, `net.loftr.LoFTR.forward`), channels last, in two parts:

- `_dense`, three compiled stages evaluated one after the other: each image's ResNet-FPN, then the coarse
  transformer and the coarse matching. It returns each coarse cell's best partner, its confidence and whether the two
  are mutual maxima above the threshold and off the border; the fine maps and the coarse features stay on the GPU.
- `_refine`, after the one host read of a call (which cells matched), compiled per `_BUCKET` matches (the ids padded
  with cell 0, the padding's offsets dropped): the 5 x 5 fine windows of the matched cells, the fine transformer and
  the expected offset.

Nothing is exported, so a first call is the Metal shader builds only. Differences to the PyTorch forward, all exact
in real arithmetic:

- batch norm folded into the convolutions; the backbone's 196-channel maps run `_WIDE` wide (zero filters): MLX's
  3x3 convolution is slow on channel counts that are not multiples of 32 / 64 (a zero channel stays zero through the
  ReLUs, the residuals, the 1x1 convolutions and the upsampling).
- the coarse linear attention with the heads as one block-diagonal `[C, C]` matrix (`k^T v` masked to the heads'
  blocks): two GEMMs on the `[L, C]` tokens instead of per-head products of transposed `[H, L, D]` views. The fine
  stage's attention over 25-token windows is one Metal kernel (`window_attention`, a threadgroup per window and head):
  there the masked GEMMs were 64 ms of zeros, the kernel 12 ms.
- the layer norms as one Metal kernel (`layer_norm`, a simdgroup per row, the residual added in it): float16 in and
  out, float32 statistics; `mx.fast.layer_norm` with float32 weights returns float32 and took 10x as long.
- the dual-softmax as `exp(2 s - logsumexp_rows - logsumexp_columns)`, and "mutual maximum" as the two argmaxes
  agreeing instead of the two maxima being equal; a large score matrix in row blocks, two passes (`_dual_softmax`).

Float16: the backbone's convolutions and both transformers' GEMMs; float32: the grayscale resize, the attention's
`k^T v` sums and normalisers, the layer norms' statistics and weights, the position encoding's sum, the coarse and fine
similarities, the dual-softmax and the outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fastforks._core.mlx import metal_kernel, mx, to_mx
from fastforks.kernels.graph_mlx.ops import bilinear_nhwc

from .net.loftr import sine_encoding

if TYPE_CHECKING:
    import torch

    from .model import Params

__all__ = ["MlxLoFTR", "prepare"]


GRAY = (0.299, 0.587, 0.114)  # fixed: as `model.GRAY`
DIVISOR = 8  # fixed: the coarse stride
_NARROW, _WIDE = 196, 224  # the backbone's 196-channel maps are run this wide
_COARSE = ("self", "cross") * 4
_FINE = ("self", "cross")
_HEADS = 8
_WINDOW = 5  # fixed: the fine window (upstream `fine_window_size`)
_FINE_STRIDE = 4  # fixed: fine pixels per coarse cell (1/2 and 1/8 resolution)
_BUCKET = 512  # the refinement runs on a multiple of this many matches
_SLOPE = 0.01  # `nn.LeakyReLU()`
_ATT_EPS = 1e-6
_NORM_EPS = 1e-5
_TEMPERATURE = 0.1  # fixed: upstream `dsmax_temperature`
_SCORE_BYTES = 64 << 20  # float32 scores per block of the coarse matching (`_dual_softmax`)


def _folded(sd: dict[str, torch.Tensor], conv: str, bn: str | None) -> tuple[np.ndarray, np.ndarray]:
    """The convolution `conv` as `[O, kh, kw, I]` with the eval batch norm `bn` folded in, and its bias."""
    w = sd[f"{conv}.weight"].double().numpy()
    b = np.zeros(len(w))
    if bn is not None:
        scale = sd[f"{bn}.weight"].double().numpy() / np.sqrt(sd[f"{bn}.running_var"].double().numpy() + 1e-5)
        w = w * scale[:, None, None, None]
        b = sd[f"{bn}.bias"].double().numpy() - sd[f"{bn}.running_mean"].double().numpy() * scale
    return w.transpose(0, 2, 3, 1), b


def _widened(w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`_NARROW` inputs / outputs of a convolution zero-padded to `_WIDE`."""
    grow_o = _WIDE - w.shape[0] if w.shape[0] == _NARROW else 0
    grow_i = _WIDE - w.shape[3] if w.shape[3] == _NARROW else 0
    return np.pad(w, [(0, grow_o), (0, 0), (0, 0), (0, grow_i)]), np.pad(b, (0, grow_o))


def _backbone_convs(sd: dict[str, torch.Tensor]) -> list[tuple[str, str | None]]:
    """The backbone's convolutions and the batch norm after each (names under `backbone.`)."""
    convs: list[tuple[str, str | None]] = [("conv1", "bn1")]
    convs += [(f"layer{i}_outconv", None) for i in (3, 2, 1)]
    for layer in (1, 2, 3):
        for block in (0, 1):
            name = f"layer{layer}.{block}"
            convs += [(f"{name}.conv1", f"{name}.bn1"), (f"{name}.conv2", f"{name}.bn2")]
            if f"backbone.{name}.downsample.0.weight" in sd:
                convs.append((f"{name}.downsample.0", f"{name}.downsample.1"))
    for head in ("layer2_outconv2", "layer1_outconv2"):
        convs += [(f"{head}.0", f"{head}.1"), (f"{head}.3", None)]
    return convs


def prepare(sd: dict[str, torch.Tensor]) -> dict[str, mx.array]:
    """The float32 parameters of the MLX network from upstream's state dict (`matcher.` stripped).

    `<conv>.w` `[O, kh, kw, I]` / `<conv>.b`: the backbone's convolutions, batch norm folded, 1x1 ones as the GEMM
    `<conv>.m` `[I, O]`; `<linear>.m` `[in, out]` (+ `.b`); the layer norms' `.weight` / `.bias` as they are.
    """
    p: dict[str, np.ndarray] = {}
    for conv, bn in _backbone_convs(sd):
        w, b = _widened(*_folded(sd, f"backbone.{conv}", None if bn is None else f"backbone.{bn}"))
        if w.shape[1] == 1:
            p[f"{conv}.m"] = w[:, 0, 0, :].T
        else:
            p[f"{conv}.w"] = w
        if bn is not None:
            p[f"{conv}.b"] = b
    for key, value in sd.items():
        if key.startswith("backbone."):
            continue
        v = value.float().numpy()
        if key.endswith(".weight") and v.ndim == 2:  # noqa: PLR2004 - a linear layer
            p[key.removesuffix(".weight") + ".m"] = v.T
        elif key.endswith(".bias") and ".norm" not in key:
            p[key.removesuffix(".bias") + ".b"] = v
        else:
            p[key] = v
    return {k: mx.array(np.ascontiguousarray(v, dtype=np.float32)) for k, v in p.items()}


_WINDOW_ATTENTION = metal_kernel(
    name="loftr_window_attention",
    input_names=["q", "k", "v"],
    output_names=["out"],
    source="""
        // `linear_attention` of one match's window and one head: a threadgroup of D x D threads, float32 inside
        constexpr int D = C / H;
        uint t = thread_position_in_threadgroup.x;
        uint h = threadgroup_position_in_grid.y, m = threadgroup_position_in_grid.z;
        threadgroup float ks[S * D], vs[S * D], qs[L * D], kv[D * D], km[D], zs[L];
        size_t bq = size_t(m) * L * C + h * D, bk = size_t(m) * S * C + h * D;
        for (uint i = t; i < S * D; i += D * D) {
            float x = float(k[bk + (i / D) * C + i % D]);
            ks[i] = x > 0.0f ? x + 1.0f : metal::exp(x);
            vs[i] = float(v[bk + (i / D) * C + i % D]);
        }
        for (uint i = t; i < L * D; i += D * D) {
            float x = float(q[bq + (i / D) * C + i % D]);
            qs[i] = x > 0.0f ? x + 1.0f : metal::exp(x);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc = 0.0f;
        for (int s = 0; s < S; ++s) acc += ks[s * D + t / D] * vs[s * D + t % D];
        kv[t] = acc / S;
        if (t < D) {
            float sum = 0.0f;
            for (int s = 0; s < S; ++s) sum += ks[s * D + t];
            km[t] = sum / S;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < L) {
            float z = 0.0f;
            for (int d = 0; d < D; ++d) z += qs[t * D + d] * km[d];
            zs[t] = z + 1e-6f / S;  // `_ATT_EPS`
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = t; i < L * D; i += D * D) {
            uint l = i / D, e = i % D;
            float o = 0.0f;
            for (int d = 0; d < D; ++d) o += qs[l * D + d] * kv[d * D + e];
            out[bq + l * C + e] = T(o / zs[l]);
        }
    """,
)


def window_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """`net.loftr.linear_attention` (`_HEADS` heads) of `[M, L, C]` queries and `[M, S, C]` keys / values.

    One threadgroup per window and head (the fine stage's 5 x 5 windows; a GEMM there is mostly the head mask's zeros).
    """
    m, length, c = q.shape
    d = c // _HEADS
    return _WINDOW_ATTENTION(
        inputs=[q, k, v],
        template=[("T", q.dtype), ("L", length), ("S", k.shape[1]), ("C", c), ("H", _HEADS)],
        grid=(d * d, _HEADS, m),
        threadgroup=(d * d, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


_LAYER_NORM = metal_kernel(
    name="loftr_layer_norm",
    input_names=["x", "w", "b", "r"],
    output_names=["out"],
    source="""
        // `r + layer_norm(x) * w + b` (RES; else without `r`) of the rows of `x` `[R, C]`: one simdgroup per row, fp32
        uint lane = thread_position_in_grid.x, row = thread_position_in_grid.y;
        if (row >= x_shape[0]) return;  // uniform per simdgroup
        constexpr int N = C / 32;
        const device T* xr = x + size_t(row) * C;
        float v[N], sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            v[i] = float(xr[lane + 32 * i]);
            sum += v[i];
        }
        float mean = simd_sum(sum) / C, sq = 0.0f;
        for (int i = 0; i < N; ++i) {
            v[i] -= mean;
            sq += v[i] * v[i];
        }
        float rstd = metal::rsqrt(simd_sum(sq) / C + 1e-5f);  // `_NORM_EPS`
        for (int i = 0; i < N; ++i) {
            uint c = lane + 32 * i;
            float y = v[i] * rstd * w[c] + b[c];
            if (RES) y += float(r[size_t(row) * C + c]);
            out[size_t(row) * C + c] = T(y);
        }
    """,
)


def layer_norm(x: mx.array, w: mx.array, b: mx.array, residual: mx.array | None = None) -> mx.array:
    """`residual + nn.LayerNorm(x)` (float32 statistics and float32 `w`, `b`) over the last axis, in `x`'s dtype.

    MLX's `fast.layer_norm` takes ~10 ms on the fine stage's `[185000, 128]` tokens; one simdgroup per row is ~1 ms.
    """
    c = x.shape[-1]
    rows = x.reshape(-1, c)
    return _LAYER_NORM(
        inputs=[rows, w, b, rows if residual is None else residual.reshape(-1, c)],
        template=[("T", x.dtype), ("C", c), ("RES", residual is not None)],
        grid=(32, rows.shape[0], 1),
        threadgroup=(32, 8, 1),
        output_shapes=[rows.shape],
        output_dtypes=[x.dtype],
    )[0].reshape(x.shape)


def _dual_softmax(a: mx.array, b: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Per row of `s = a b^T` (`[L, S]`, float32): the best column, its dual-softmax confidence, the best row of it.

    The confidence is `exp(2 s - logsumexp_rows - logsumexp_columns)`; the best row of column `j` maximises
    `2 s[:, j] - logsumexp_rows`. Small products are one matrix; a large one runs in blocks of `_SCORE_BYTES` rows,
    one after the other (`mx.depends`), in two passes (the statistics, then the maxima), so at most one block of
    scores is alive instead of the whole matrix and its transpose (loftr at 768 x 1024: 12288^2 scores, 604 MB each).
    """
    n_l, n_s = a.shape[0], b.shape[0]
    rows = max(1, _SCORE_BYTES // (4 * n_s))
    if n_l <= 2 * rows:
        s, st = a @ b.T, b @ a.T  # [L, S] and its transpose: every reduction along the last axis
        lse_l, lse_s = mx.logsumexp(s, axis=-1), mx.logsumexp(st, axis=-1)
        r = 2 * s - lse_s  # log confidence + lse_l, per row
        j = mx.argmax(r, axis=-1)
        return j, mx.exp(mx.max(r, axis=-1) - lse_l), mx.argmax(2 * st - lse_l, axis=-1)[j]
    starts = range(0, n_l, rows)
    lse_rows: list[mx.array] = []
    lse_s = None
    for i in starts:  # pass 1: each block's row statistics, the columns' accumulated
        ai = a[i : i + rows] if lse_s is None else mx.depends(a[i : i + rows], lse_s)
        s = ai @ b.T
        lse_rows.append(mx.logsumexp(s, axis=-1))
        col = mx.logsumexp(s, axis=0)
        lse_s = col if lse_s is None else mx.logaddexp(lse_s, col)
    assert lse_s is not None  # noqa: S101 - at least one block
    js, confs = [], []
    best = arg = None
    for i, lse_l in zip(starts, lse_rows, strict=True):  # pass 2: the rows' maxima, the columns' running maxima
        ai = a[i : i + rows] if best is None else mx.depends(a[i : i + rows], best)
        s = ai @ b.T
        r = 2 * s - lse_s
        js.append(mx.argmax(r, axis=-1))
        confs.append(mx.exp(mx.max(r, axis=-1) - lse_l))
        v = 2 * s - lse_l[:, None]
        top, at = mx.max(v, axis=0), mx.argmax(v, axis=0) + i
        if best is None or arg is None:
            best, arg = top, at
        else:  # the first maximum wins, as one argmax over the whole column does
            arg = mx.where(top > best, at, arg)
            best = mx.maximum(best, top)
    assert arg is not None  # noqa: S101 - at least one block
    j = mx.concatenate(js)
    return j, mx.concatenate(confs), arg[j]


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _head_mask(c: int, dtype: mx.Dtype) -> mx.array:
    """`[C, C]`: 1 where both channels are of one attention head."""
    head = np.arange(c) // (c // _HEADS)
    return mx.array(head[:, None] == head[None]).astype(dtype)


class MlxLoFTR:
    """The MLX matcher: the converted weights and the compiled stages of the dense pass and the refinement."""

    def __init__(self, p32: dict[str, mx.array], params: Params, dtype: str, **stages: str) -> None:
        """`p32` is `prepare`'s output, `params` the variant's parameters, `dtype` the run dtype.

        `stages` (`backbone`, `coarse`, `fine`) override the run dtype per stage.
        """
        self.params = params
        dts = {s: getattr(mx, stages.get(s, dtype)) for s in ("backbone", "coarse", "fine")}
        self.dt_backbone, self.dt_coarse, self.dt_fine = dts["backbone"], dts["coarse"], dts["fine"]

        def cast(k: str, v: mx.array) -> mx.array:
            if ".norm" in k:
                return v
            if k.startswith(("loftr_fine.", "fine_preprocess.")):
                return v.astype(self.dt_fine)
            return v.astype(self.dt_coarse if k.startswith("loftr_coarse.") else self.dt_backbone)

        self.p = {k: cast(k, v) for k, v in p32.items()}
        mx.eval(self.p)
        self._features = mx.compile(self._image)
        self._coarse = mx.compile(self._match)
        self._refine_compiled = mx.compile(self._refine)

    # ----------------------------------------------------------------------------------------------- backbone
    def _conv(self, x: mx.array, name: str, stride: int = 1) -> mx.array:
        """The convolution `name` of NHWC `x` (with the folded batch norm's bias, if it has one)."""
        m = self.p.get(f"{name}.m")
        if m is not None:
            x = x[:, ::stride, ::stride] if stride > 1 else x
            n, h, w, c = x.shape
            y = (x.reshape(-1, c) @ m).reshape(n, h, w, -1)
        else:
            wt = self.p[f"{name}.w"]
            y = mx.conv2d(x, wt, stride=stride, padding=wt.shape[1] // 2)
        b = self.p.get(f"{name}.b")
        return y if b is None else y + b

    def _block(self, x: mx.array, name: str, stride: int) -> mx.array:
        y = self._conv(_relu(self._conv(x, f"{name}.conv1", stride)), f"{name}.conv2")
        return _relu(y + (self._conv(x, f"{name}.downsample.0", stride) if stride > 1 else x))

    def _head(self, x: mx.array, name: str) -> mx.array:
        y = self._conv(x, f"{name}.0")
        return self._conv(mx.maximum(y, _SLOPE * y), f"{name}.3")

    @staticmethod
    def _up2(x: mx.array) -> mx.array:
        """`F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)`."""
        h, w = x.shape[1:3]
        return bilinear_nhwc(x, (2 * h, 2 * w), ((h - 1) / (2 * h - 1), (w - 1) / (2 * w - 1)), align=True)

    def _backbone(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """`ResNetFPN_8_2`: gray `[N, H, W, 1]` -> the coarse `[N, H/8, W/8, 256]` and fine `[N, H/2, W/2, 128]` map."""
        x = _relu(self._conv(x, "conv1", 2))
        x1 = self._block(self._block(x, "layer1.0", 1), "layer1.1", 1)
        x2 = self._block(self._block(x1, "layer2.0", 2), "layer2.1", 1)
        x3 = self._block(self._block(x2, "layer3.0", 2), "layer3.1", 1)
        coarse = self._conv(x3, "layer3_outconv")
        x2 = self._head(self._conv(x2, "layer2_outconv") + self._up2(coarse), "layer2_outconv2")
        fine = self._head(self._conv(x1, "layer1_outconv") + self._up2(x2), "layer1_outconv2")
        return coarse, fine

    # -------------------------------------------------------------------------------------------- transformer
    def _attention(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """`net.loftr.linear_attention` on `[..., L, C]` queries and `[..., S, C]` keys / values."""
        c, s = q.shape[-1], k.shape[-2]
        dt = q.dtype
        q = mx.where(q > 0, q + 1, mx.exp(q))  # `elu(q) + 1`
        k = mx.where(k > 0, k + 1, mx.exp(k)).astype(mx.float32)
        kv = (mx.swapaxes(k, -1, -2) @ (v.astype(mx.float32) / s)) * _head_mask(c, mx.float32)  # [..., C, C]
        k_mean = k.mean(axis=-2, keepdims=True)  # [..., 1, C]
        z = (q.astype(mx.float32) * k_mean).reshape(*q.shape[:-1], _HEADS, c // _HEADS).sum(axis=-1, keepdims=True)
        z = mx.broadcast_to(z + _ATT_EPS / s, (*q.shape[:-1], _HEADS, c // _HEADS)).reshape(q.shape)
        return ((q @ kv.astype(dt)).astype(mx.float32) / z).astype(dt)

    def _layer(self, x: mx.array, source: mx.array, name: str) -> mx.array:
        """`LoFTREncoderLayer.forward`."""
        p = self.p
        c = x.shape[-1]
        attention = window_attention if x.ndim == 3 else self._attention  # noqa: PLR2004 - `[M, 25, C]` windows
        message = attention(x @ p[f"{name}.q_proj.m"], source @ p[f"{name}.k_proj.m"], source @ p[f"{name}.v_proj.m"])
        message = layer_norm(message @ p[f"{name}.merge.m"], p[f"{name}.norm1.weight"], p[f"{name}.norm1.bias"])
        m1 = p[f"{name}.mlp.0.m"]  # on `cat([x, message])`
        hidden = _relu(x @ m1[:c] + message @ m1[c:])
        return layer_norm(hidden @ p[f"{name}.mlp.2.m"], p[f"{name}.norm2.weight"], p[f"{name}.norm2.bias"], x)

    def _transformer(self, f0: mx.array, f1: mx.array, name: str, layers: tuple[str, ...]) -> tuple[mx.array, mx.array]:
        """`LocalFeatureTransformer.forward`."""
        for i, kind in enumerate(layers):
            layer = f"{name}.layers.{i}"
            if kind == "self" and f0.ndim == 3:  # noqa: PLR2004 - the fine windows: both images as one batch
                f0, f1 = mx.split(self._layer(both := mx.concatenate([f0, f1]), both, layer), 2)
            elif kind == "self":
                f0, f1 = self._layer(f0, f0, layer), self._layer(f1, f1, layer)
            else:  # cross: `f1` attends to the updated `f0`, as upstream
                f0 = self._layer(f0, f1, layer)
                f1 = self._layer(f1, f0, layer)
        return f0, f1

    # ------------------------------------------------------------------------------------------------- dense
    def _gray(self, image: mx.array) -> mx.array:
        """`Matcher.prepare`: RGB `[1, 3, H, W]` -> gray `[1, h, w, 1]` at the matching size (float32)."""
        h, w = image.shape[-2:]
        rh, rw = self.sizes(h, w)
        g = (image.transpose(0, 2, 3, 1) * mx.array(GRAY, dtype=mx.float32)).sum(axis=-1, keepdims=True)
        return g if (rh, rw) == (h, w) else bilinear_nhwc(g, (rh, rw), (h / rh, w / rw), align=False)

    def sizes(self, h: int, w: int) -> tuple[int, int]:
        """`model.resized_size`: the matching size of an `h` x `w` image."""
        long_side = self.params.long_side
        if long_side > 0:
            s = long_side / max(h, w)
            h, w = round(h * s), round(w * s)
        return h // DIVISOR * DIVISOR, w // DIVISOR * DIVISOR

    def _tokens(self, coarse: mx.array) -> mx.array:
        """`LoFTR.coarse`'s input: the coarse map `[h, w, C]` plus the position encoding, `[h * w, C]`."""
        h, w, c = coarse.shape
        pe = sine_encoding(c, h, w, temp_bug_fix=self.params.temp_bug_fix, device="cpu")  # ty: ignore[invalid-argument-type]
        tokens = coarse.astype(mx.float32) + to_mx(pe[0].permute(1, 2, 0))
        return tokens.astype(self.dt_coarse).reshape(h * w, c)

    def _inside(self, h: int, w: int) -> np.ndarray:
        """Cells at least `border_rm` cells off the border, flattened `[h * w]`."""
        b = self.params.border_rm
        y, x = np.arange(h), np.arange(w)
        return (((y >= b) & (y < h - b))[:, None] & ((x >= b) & (x < w - b))[None]).reshape(-1)

    def _image(self, image: mx.array) -> tuple[mx.array, mx.array]:
        """One image's coarse map `[h, w, C]` and padded fine map `[H + 4, W + 4, C]`."""
        coarse, fine = self._backbone(self._gray(image).astype(self.dt_backbone))
        r = _WINDOW // 2
        return coarse[0], mx.pad(fine[0], [(r, r), (r, r), (0, 0)])

    def _match(self, c0: mx.array, c1: mx.array) -> tuple[mx.array, ...]:
        """The coarse transformer's features and, per cell of image 0, its partner, confidence and match."""
        hw0, hw1 = c0.shape[:2], c1.shape[:2]
        t0, t1 = self._transformer(self._tokens(c0), self._tokens(c1), "loftr_coarse", _COARSE)
        # `LoFTR.coarse_matches`, float32
        scale = 1.0 / (t0.shape[-1] * _TEMPERATURE) ** 0.5
        a, b = t0.astype(mx.float32) * scale, t1.astype(mx.float32) * scale
        j, conf, i_back = _dual_softmax(a, b)
        keep = (conf > self.params.threshold) & (i_back == mx.arange(a.shape[0]))
        if self.params.border_rm > 0:
            keep = keep & mx.array(self._inside(*hw0)) & mx.array(self._inside(*hw1))[j]
        return t0, t1, j, conf, keep

    def _dense(self, image0: mx.array, image1: mx.array) -> tuple[mx.array, ...]:
        """Both images' padded fine maps and coarse features; per cell of image 0 its partner, confidence, match.

        Three compiled stages, each evaluated before the next is queued: one image's backbone, the other's, then the
        coarse transformer and matching. MLX returns a command buffer's transients only when it completes, so one
        compiled function over all of it held both backbones' maps and the score blocks at once (1.5x the peak).
        One image at a time: a batch of two is no faster and doubles the Winograd convolutions' workspace.
        """
        c0, f0 = self._features(image0)
        mx.eval(c0, f0)
        c1, f1 = self._features(image1)
        mx.eval(c1, f1)
        return f0, f1, *self._coarse(c0, c1)

    # ------------------------------------------------------------------------------------------------ refine
    @staticmethod
    def _windows(fine: mx.array, ids: mx.array) -> mx.array:
        """`FinePreprocess._windows` on the padded fine map `[H + 4, W + 4, C]`: `[n, 25, C]`."""
        hp, wp, c = fine.shape
        cw = (wp - 2 * (_WINDOW // 2)) // _FINE_STRIDE  # the coarse width
        d = mx.arange(_WINDOW)
        rows = (ids // cw * _FINE_STRIDE)[:, None, None] + d[None, :, None]
        cols = (ids % cw * _FINE_STRIDE)[:, None, None] + d[None, None, :]
        return fine.reshape(hp * wp, c)[(rows * wp + cols).reshape(-1)].reshape(len(ids), _WINDOW * _WINDOW, c)

    def _refine(
        self, fine: tuple[mx.array, mx.array], coarse: tuple[mx.array, mx.array], i: mx.array, j: mx.array
    ) -> mx.array:
        """`FinePreprocess`, the fine transformer and `LoFTR.fine` of the matches `i` -> `j`.

        Their offsets in the window, `[M, 2]` in [-1, 1].
        """
        p = self.p
        dt = self.dt_fine
        (f0, f1), (t0, t1) = fine, coarse
        c = (
            mx.concatenate([t0[i], t1[j]]).astype(dt) @ p["fine_preprocess.down_proj.m"]
            + p["fine_preprocess.down_proj.b"]
        )
        merge = p["fine_preprocess.merge_feat.m"]  # on `cat([window, coarse])`
        half = merge.shape[0] // 2
        c = (c @ merge[half:] + p["fine_preprocess.merge_feat.b"])[:, None]
        m = len(i)
        w0 = self._windows(f0.astype(dt), i) @ merge[:half] + c[:m]
        w1 = self._windows(f1.astype(dt), j) @ merge[:half] + c[m:]
        w0, w1 = self._transformer(w0, w1, "loftr_fine", _FINE)
        ww, ch = w0.shape[1:]
        centre = w0[:, ww // 2].astype(mx.float32)[:, :, None]  # [M, C, 1]
        heat = mx.softmax((w1.astype(mx.float32) @ centre)[..., 0] / ch**0.5, axis=-1)  # [M, WW]
        g = np.linspace(-1, 1, _WINDOW, dtype=np.float32)
        grid = np.stack([np.tile(g, _WINDOW), np.repeat(g, _WINDOW)], -1)  # [WW, 2] (x, y)
        return heat @ mx.array(grid)

    def __call__(self, image0: mx.array, image1: mx.array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Float32 RGB `[1, 3, H, W]` images in [0, 1] -> `Matcher.forward`'s keypoints of both images and scores."""
        (h0, w0), (h1, w1) = image0.shape[-2:], image1.shape[-2:]
        (rh0, rw0), (rh1, rw1) = self.sizes(h0, w0), self.sizes(h1, w1)
        f0, f1, t0, t1, j, conf, keep = self._dense(image0, image1)
        mx.async_eval(f0, f1, t0, t1, j, conf, keep)
        i_ids = np.flatnonzero(np.array(keep))
        j_ids = np.array(j)[i_ids].astype(np.int64)
        scores = np.array(conf)[i_ids]
        cw0, cw1 = rw0 // DIVISOR, rw1 // DIVISOR
        mk0 = np.stack([i_ids % cw0, i_ids // cw0], 1).astype(np.float32) * DIVISOR
        mk1 = np.stack([j_ids % cw1, j_ids // cw1], 1).astype(np.float32) * DIVISOR
        if m := len(i_ids):  # padded to a multiple of `_BUCKET` matches: a few compiled sizes
            n = -(-m // _BUCKET) * _BUCKET
            pad = [np.pad(ids, (0, n - m)).astype(np.int32) for ids in (i_ids, j_ids)]
            offsets = np.array(self._refine_compiled((f0, f1), (t0, t1), *map(mx.array, pad)))[:m]
            mk1 = mk1 + offsets * (_WINDOW // 2) * (DIVISOR // _FINE_STRIDE)
        scale0 = np.array([w0 / rw0, h0 / rh0], np.float32)
        scale1 = np.array([w1 / rw1, h1 / rh1], np.float32)
        return mk0 * scale0, mk1 * scale1, scores
