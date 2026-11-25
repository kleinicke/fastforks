"""RT-DETRv2 and D-FINE on Apple MLX, written in `mlx.core` operations (`build(net, dtype)`): images -> raw outputs.

The maths of `rtdetrv2.net.RtDetr` / `dfine.net.DFine` (BatchNorm folded, RepVGG merged), channels last, weights
copied from the float32 PyTorch twin. Written out instead of replayed from an exported graph: the export, its
translation and the compile of the whole graph were 1.5-2.7 s of each first call.

- Backbone (PResNet-vd / HGNetV2), the hybrid encoder (AIFI layer, FPN / PAN) and the decoder layers in the run dtype;
  the convs are `cspnext.mlx_net.conv` with the graph translator's dispatch (1 x 1 convs a GEMM, depthwise the shared
  kernel, the rest `mx.conv2d` with the translator's channel padding).
- As the PyTorch runner's float heads: the query selection, the query position head, the box refinement and the class
  heads (D-FINE: also the distribution refinement and the quality estimator) are float32.
- Deformable cross-attention: the shared `msda` kernel; D-FINE's per-level point counts (3 / 6 / 3) are padded to the
  largest with zero weights (the softmax is taken over the real points first).
- The anchors, their validity mask and the AIFI position table are fixed by the feature sizes and made once per size.
- Each backbone stage, the encoder's AIFI / FPN / PAN halves and the decoder are compiled steps of depth 1
  (`cspnext.mlx_net.stepped`): bounded transients.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import math
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx, to_mx, to_torch
from fastforks.detection._rtdetr import ANCHOR_SIZE, HEADS, LN_EPS, POS_TEMPERATURE, VALID_EPS
from fastforks.kernels.graph_mlx import ops
from fastforks.shared.cspnext import mlx_net as csp

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import torch

    Layer = Callable[[mx.array], mx.array]

__all__ = ["build"]

_GRAPH = "portable"  # the translator's conv dispatch (`cspnext.mlx_net.conv`)
_Wb = tuple[mx.array, mx.array]


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _gelu(x: mx.array) -> mx.array:
    """Exact GELU (`F.gelu`)."""
    return 0.5 * x * (1 + mx.erf(x * (1 / math.sqrt(2))))


_ACT: dict[str | None, Layer | None] = {"relu": _relu, "silu": _silu, None: None}


def _lin(layer: Any, dtype: mx.Dtype) -> _Wb:
    """An `nn.Linear` as `(weight [in, out], bias)`."""
    return csp.arr(layer.weight.t(), dtype), csp.arr(layer.bias, dtype)


def _linear(x: mx.array, wb: _Wb) -> mx.array:
    return mx.addmm(wb[1], x, wb[0]) if x.ndim == 2 else x @ wb[0] + wb[1]  # noqa: PLR2004 - a matrix


def _norm(layer: Any, dtype: mx.Dtype) -> _Wb:
    return csp.arr(layer.weight, dtype), csp.arr(layer.bias, dtype)


def _ln(x: mx.array, wb: _Wb) -> mx.array:
    return mx.fast.layer_norm(x, wb[0], wb[1], LN_EPS)


def _mlp(m: Any, dtype: mx.Dtype) -> Layer:
    """`_rtdetr.Mlp`: the linears with ReLU / SiLU between."""
    layers = [_lin(layer, dtype) for layer in m.layers]
    act = _silu if m.act.__name__ == "silu" else _relu

    def apply(x: mx.array) -> mx.array:
        for i, wb in enumerate(layers):
            x = _linear(x, wb)
            if i < len(layers) - 1:
                x = act(x)
        return x

    return apply


def _conv(m: Any, dtype: mx.Dtype) -> Callable[..., mx.array]:
    """`ConvBn` (and D-FINE's `HgConv` with its affine block): `act(conv(x))`, or `relu(conv(x) + res)`."""
    conv, act = csp.conv(m.conv, dtype, _GRAPH), _ACT[m.act]
    lab = getattr(m, "lab", None)
    affine = None if lab is None else (float(lab.scale.detach()), float(lab.bias.detach()))

    def apply(x: mx.array, res: mx.array | None = None) -> mx.array:
        y = conv(x)
        if res is not None:
            y = _relu(y + res)
        elif act is not None:
            y = act(y)
        return y if affine is None else y * affine[0] + affine[1]

    return apply


def _repvgg(m: Any, dtype: mx.Dtype) -> Layer:
    conv = csp.conv(m.conv, dtype, _GRAPH)
    return lambda x: _silu(conv(x))


def _csp_rep(m: Any, dtype: mx.Dtype) -> Layer:
    """`CspRep`: `conv3(bottlenecks(conv1(x)) + conv2(x))`."""
    conv1, conv2 = _conv(m.conv1, dtype), _conv(m.conv2, dtype)
    blocks = csp.chain(*(_repvgg(b, dtype) for b in m.bottlenecks))
    conv3 = None if m.conv3 is None else _conv(m.conv3, dtype)

    def apply(x: mx.array) -> mx.array:
        y = blocks(conv1(x)) + conv2(x)
        return y if conv3 is None else conv3(y)

    return apply


def _cat(*xs: mx.array) -> mx.array:
    return mx.concatenate(xs, axis=-1)


class _Attention:
    """`_attention.Attention` (`nn.MultiheadAttention`'s weights): q / k one GEMM when they are the same input."""

    def __init__(self, m: Any, dtype: mx.Dtype) -> None:
        w, b = m.in_proj_weight.detach(), m.in_proj_bias.detach()
        d = w.shape[1]
        self.heads = m.heads
        self.qk = csp.arr(w[: 2 * d].t(), dtype), csp.arr(b[: 2 * d], dtype)
        self.v = csp.arr(w[2 * d :].t(), dtype), csp.arr(b[2 * d :], dtype)
        self.out = _lin(m.out_proj, dtype)

    def __call__(self, qk: mx.array, v: mx.array) -> mx.array:
        """Self-attention: `[B, S, d]` queries = keys, values -> `[B, S, d]`."""
        b, s, d = v.shape
        h = self.heads
        q, k = mx.split(qk @ self.qk[0] + self.qk[1], 2, axis=-1)
        q, k, v = (t.reshape(b, s, h, d // h).transpose(0, 2, 1, 3) for t in (q, k, _linear(v, self.v)))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=(d // h) ** -0.5)
        return _linear(out.transpose(0, 2, 1, 3).reshape(b, s, d), self.out)


def _aifi(m: Any, dtype: mx.Dtype) -> Callable[[mx.array, mx.array], mx.array]:
    """`AifiLayer`: post-norm encoder layer, position added to queries / keys, exact GELU."""
    att, fc1, fc2 = _Attention(m.self_attn, dtype), _lin(m.linear1, dtype), _lin(m.linear2, dtype)
    n1, n2 = _norm(m.norm1, dtype), _norm(m.norm2, dtype)

    def apply(x: mx.array, pos: mx.array) -> mx.array:
        x = _ln(x + att(x + pos, x), n1)
        return _ln(x + _linear(_gelu(_linear(x, fc1)), fc2), n2)

    return apply


@cache
def _position(h: int, w: int, dim: int) -> np.ndarray:
    """`sincos_position` (float32): `[1, h * w, dim]`."""
    gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    omega = 1.0 / (np.float32(POS_TEMPERATURE) ** (np.arange(dim // 4, dtype=np.float32) / np.float32(dim // 4)))
    oy, ox = gy.reshape(-1, 1) * omega[None], gx.reshape(-1, 1) * omega[None]
    return np.concatenate([np.sin(oy), np.cos(oy), np.sin(ox), np.cos(ox)], axis=1)[None].astype(np.float32)


@cache
def _anchors(shapes: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, np.ndarray]:
    """`anchors`: log-odds anchors `[1, S, 4]` (inf where invalid) and the validity mask `[1, S, 1]` (float32)."""
    out = []
    for lvl, (h, w) in enumerate(shapes):
        gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        xy = np.stack([(gx + 0.5) / np.float32(w), (gy + 0.5) / np.float32(h)], axis=-1).reshape(-1, 2)
        wh = np.full_like(xy, np.float32(ANCHOR_SIZE) * np.float32(2.0**lvl))
        out.append(np.concatenate([xy, wh], axis=-1))
    a = np.concatenate(out)[None]
    valid = ((a > VALID_EPS) & (a < 1 - VALID_EPS)).all(-1, keepdims=True)
    with np.errstate(divide="ignore"):
        anc = np.where(valid, np.log(a / (1 - a)), np.inf).astype(np.float32)
    return anc, valid.astype(np.float32)


def _inverse_sigmoid(x: mx.array, eps: float = 1e-5) -> mx.array:
    x = mx.clip(x, 0.0, 1.0)
    return mx.log(mx.maximum(x, eps) / mx.maximum(1 - x, eps))


def _select(
    mem: mx.array, shapes: tuple[tuple[int, int], ...], heads: tuple[Any, ...], queries: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array]:
    """The float32 query selection: memory `[B, S, d]` -> queries `[B, Q, d]` (run dtype) and reference boxes."""
    enc_proj, enc_norm, score_head, bbox_head = heads
    anc, valid = (mx.array(a) for a in _anchors(shapes))
    out = _ln(_linear(valid * mem.astype(mx.float32), enc_proj), enc_norm)
    coords = bbox_head(out) + anc
    scores = mx.max(_linear(out, score_head), axis=-1)
    idx = mx.argsort(-scores, axis=-1)[:, : min(queries, mem.shape[1])][..., None]
    ref = mx.sigmoid(mx.take_along_axis(coords, idx, axis=1))
    return mx.take_along_axis(out, idx, axis=1).astype(dtype), ref


class _Net:
    """Shared by both: the image steps (`self.image`: NHWC image -> memory maps) and the decoder step."""

    image: list[Callable[..., Any]]
    decoder: Callable[..., Any]
    dtype: mx.Dtype

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`[B, 3, H, W]` images in 0..1 (float, CPU) -> float32 class logits `[B, Q, classes]`, boxes `[B, Q, 4]`."""
        csp.options(_GRAPH)
        logits, boxes = csp.steps(self.forward, to_mx(x.float().permute(0, 2, 3, 1)))
        mx.eval(logits, boxes)
        return to_torch(logits), to_torch(boxes)

    def forward(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """NHWC images -> class logits, boxes."""
        raise NotImplementedError


# ---- RT-DETRv2 ------------------------------------------------------------------------------------------------------


def _short(m: Any, dtype: mx.Dtype) -> Layer | None:
    """The block shortcut: identity (None), a 1 x 1 `ConvBn`, or variant d's 2 x 2 average pool then the 1 x 1."""
    if m is None:
        return None
    if hasattr(m, "act"):
        return _conv(m, dtype)
    conv = _conv(m.conv, dtype)

    def apply(x: mx.array) -> mx.array:
        n, h, w, c = x.shape
        if h % 2 or w % 2:  # ceil mode: a partial window averages fewer values (never at multiples of 32)
            msg = f"odd map {h}x{w}: the input is not a multiple of 32"
            raise ValueError(msg)
        pooled = x.reshape(n, h // 2, 2, w // 2, 2, c).astype(mx.float32).mean(axis=(2, 4)).astype(x.dtype)
        return conv(pooled)

    return apply


def _resnet_block(m: Any, dtype: mx.Dtype) -> Layer:
    """`Block`: `relu(branches(x) + shortcut(x))`."""
    a, b = _conv(m.branch2a, dtype), _conv(m.branch2b, dtype)
    c = _conv(m.branch2c, dtype) if m.bottleneck else None
    short = _short(m.short, dtype)

    def apply(x: mx.array) -> mx.array:
        res = x if short is None else short(x)
        return c(b(a(x)), res) if c is not None else b(a(x), res)

    return apply


def _max_pool3(x: mx.array) -> mx.array:
    """`F.max_pool2d(x, 3, 2, 1)` on NHWC (even sizes): the nine taps of the -inf-padded map."""
    h, w = x.shape[1], x.shape[2]
    y = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], constant_values=-math.inf)
    oh, ow = (h - 1) // 2 + 1, (w - 1) // 2 + 1
    taps = [y[:, dy : dy + 2 * oh : 2, dx : dx + 2 * ow : 2] for dy in range(3) for dx in range(3)]
    out = taps[0]
    for t in taps[1:]:
        out = mx.maximum(out, t)
    return out


class _MlxRtDetr(_Net):
    def __init__(self, net: Any, dtype: mx.Dtype) -> None:
        self.dtype = d = dtype
        bb, enc, dec = net.backbone, net.encoder, net.decoder
        stem = csp.chain(*(_conv(getattr(bb.conv1, f"conv1_{i}"), d) for i in (1, 2, 3)), _max_pool3)
        stages = [csp.chain(*(_resnet_block(b, d) for b in s.blocks)) for s in bb.res_layers]
        proj = [_conv(p, d) for p in enc.input_proj]
        aifi = [_aifi(layer, d) for layer in enc.encoder[0].layers]
        lateral = [_conv(m, d) for m in enc.lateral_convs]
        fpn = [_csp_rep(m, d) for m in enc.fpn_blocks]
        down = [_conv(m, d) for m in enc.downsample_convs]
        pan = [_csp_rep(m, d) for m in enc.pan_blocks]

        def first(x: mx.array) -> mx.array:
            return stages[0](stem(x.astype(d)))

        def top(c5: mx.array) -> mx.array:
            p5 = proj[2](c5)
            _, h, w, c = p5.shape
            pos = mx.array(_position(h, w, c)).astype(d)
            mem = p5.reshape(p5.shape[0], h * w, c)
            for layer in aifi:
                mem = layer(mem, pos)
            return mem.reshape(p5.shape)

        def fpn_step(f5: mx.array, c4: mx.array, c3: mx.array) -> tuple[mx.array, ...]:
            high5 = lateral[0](f5)
            inner4 = fpn[0](_cat(csp.up2(high5), proj[1](c4)))
            high4 = lateral[1](inner4)
            inner3 = fpn[1](_cat(csp.up2(high4), proj[0](c3)))
            return inner3, high4, high5

        def pan_step(o3: mx.array, high4: mx.array, high5: mx.array) -> tuple[mx.array, ...]:
            o4 = pan[0](_cat(down[0](o3), high4))
            return o3, o4, pan[1](_cat(down[1](o4), high5))

        self.image = [csp.stepped(f) for f in (first, *stages[1:], top, fpn_step, pan_step)]
        self.decoder = csp.stepped(self._decoder_fn(dec, d))

    @staticmethod
    def _decoder_fn(dec: Any, d: mx.Dtype) -> Callable[..., tuple[mx.array, mx.array]]:
        f = mx.float32
        proj = [_conv(p, d) for p in dec.input_proj]
        heads = (_lin(dec.enc_output.proj, f), _norm(dec.enc_output.norm, f), _lin(dec.enc_score_head, f))
        heads = (*heads, _mlp(dec.enc_bbox_head, f))
        qpos_head = _mlp(dec.query_pos_head, f)
        n = dec.eval_layers
        layers = [_rt_layer(layer, d) for layer in list(dec.decoder.layers)[:n]]
        bbox_heads = [_mlp(m, f) for m in list(dec.dec_bbox_head)[:n]]
        score_head = _lin(dec.dec_score_head[n - 1], f)
        queries = dec.queries

        def apply(*feats: mx.array) -> tuple[mx.array, mx.array]:
            maps = [p(x) for p, x in zip(proj, feats, strict=True)]
            shapes = tuple((int(m.shape[1]), int(m.shape[2])) for m in maps)
            mem = mx.concatenate([m.reshape(m.shape[0], -1, m.shape[-1]) for m in maps], axis=1)
            tgt, ref = _select(mem, shapes, heads, queries, d)
            boxes = ref
            for layer, bbox_head in zip(layers, bbox_heads, strict=True):
                tgt = layer(tgt, ref, mem, shapes, qpos_head(ref).astype(d))
                boxes = mx.sigmoid(bbox_head(tgt.astype(f)) + _inverse_sigmoid(ref))
                ref = boxes
            return _linear(tgt.astype(f), score_head), boxes

        return apply

    def forward(self, x: mx.array) -> tuple[mx.array, mx.array]:
        first, s3, s4, s5, top, fpn_step, pan_step = self.image
        c2 = first(x)
        c3 = s3(c2)
        c4 = s4(c3)
        c5 = s5(c4)
        inner3, high4, high5 = fpn_step(top(c5), c4, c3)
        return self.decoder(*pan_step(inner3, high4, high5))


def _rt_layer(m: Any, d: mx.Dtype) -> Callable[..., mx.array]:
    """RT-DETRv2's `DecoderLayer`: self-attention, deformable cross-attention (value / output projections), FFN."""
    att = _Attention(m.self_attn, d)
    ca = m.cross_attn
    off, aw = _lin(ca.sampling_offsets, d), _lin(ca.attention_weights, d)
    offw = mx.concatenate([off[0], aw[0]], axis=1), mx.concatenate([off[1], aw[1]])
    no = off[0].shape[1]
    value, out = _lin(ca.value_proj, d), _lin(ca.output_proj, d)
    scale = ca.num_points_scale.detach().float()
    half_scale = csp.arr(scale * 0.5, d)
    fc1, fc2 = _lin(m.linear1, d), _lin(m.linear2, d)
    n1, n2, n3 = (_norm(n, d) for n in (m.norm1, m.norm2, m.norm3))

    def apply(x: mx.array, ref: mx.array, mem: mx.array, shapes: Sequence[tuple[int, int]], qpos: mx.array) -> mx.array:
        b, q, dim = x.shape
        nl = len(shapes)
        x = _ln(x + att(x + qpos, x), n1)
        ow = _linear(x + qpos, offw)
        pts = no // (HEADS * nl * 2)
        o = ow[..., :no].reshape(b, q, HEADS, nl, pts, 2)
        w = mx.softmax(ow[..., no:].reshape(b, q, HEADS, nl * pts), axis=-1).reshape(b, q, HEADS, nl, pts)
        r = ref.astype(d)[:, :, None, None, None]
        loc = r[..., :2] + o * half_scale.reshape(nl, pts, 1) * r[..., 2:]
        v = _linear(mem, value).reshape(b, mem.shape[1], HEADS, dim // HEADS)
        x = _ln(x + _linear(ops.msda(v, loc, w, shapes), out), n2)
        return _ln(x + _linear(_relu(_linear(x, fc1)), fc2), n3)

    return apply


# ---- D-FINE ---------------------------------------------------------------------------------------------------------


def _hg_stem(m: Any, dtype: mx.Dtype) -> Layer:
    """`Embedder`: the 2 x 2 branch and a 2 x 2 / stride-1 max pool of the zero-padded map, concatenated."""
    first, left, right, third, fourth = (
        _conv(getattr(m, n), dtype) for n in ("stem1", "stem2a", "stem2b", "stem3", "stem4")
    )
    pad = [(0, 0), (0, 1), (0, 1), (0, 0)]

    def apply(x: mx.array) -> mx.array:
        x = mx.pad(first(x.astype(dtype)), pad)
        x2 = right(mx.pad(left(x), pad))
        x1 = mx.maximum(mx.maximum(x[:, :-1, :-1], x[:, 1:, :-1]), mx.maximum(x[:, :-1, 1:], x[:, 1:, 1:]))
        return fourth(third(_cat(x1, x2)))

    return apply


def _hg_block(m: Any, dtype: mx.Dtype) -> Layer:
    """`HgBlock`: chained layers, all outputs concatenated, two 1 x 1 aggregation convs, optional residual."""
    layers = [
        csp.chain(_conv(layer.conv1, dtype), _conv(layer.conv2, dtype))
        if hasattr(layer, "conv1")
        else _conv(layer, dtype)
        for layer in m.layers
    ]
    aggregate = csp.chain(*(_conv(c, dtype) for c in m.aggregation))
    residual = m.residual

    def apply(x: mx.array) -> mx.array:
        outs = [x]
        for layer in layers:
            outs.append(layer(outs[-1]))
        y = aggregate(mx.concatenate(outs, axis=-1))
        return y + x if residual else y

    return apply


def _hg_stage(m: Any, dtype: mx.Dtype) -> Layer:
    ops_ = [] if m.downsample is None else [_conv(m.downsample, dtype)]
    return csp.chain(*ops_, *(_hg_block(b, dtype) for b in m.blocks))


def _elan(m: Any, dtype: mx.Dtype) -> Layer:
    """`RepNcspElan4`: split, two CSP-RepVGG + 3 x 3 steps, everything concatenated, 1 x 1."""
    conv1, conv2, conv3, conv4 = (_conv(c, dtype) for c in (m.conv1, m.conv2, m.conv3, m.conv4))
    rep1, rep2 = _csp_rep(m.csp_rep1, dtype), _csp_rep(m.csp_rep2, dtype)

    def apply(x: mx.array) -> mx.array:
        a, b = mx.split(conv1(x), 2, axis=-1)
        c = conv2(rep1(b))
        return conv4(_cat(a, b, c, conv3(rep2(c))))

    return apply


class _MlxDFine(_Net):
    def __init__(self, net: Any, dtype: mx.Dtype) -> None:
        self.dtype = d = dtype
        stem = _hg_stem(net.backbone.embedder, d)
        stages = [_hg_stage(s, d) for s in net.backbone.encoder.stages]
        levels = net.backbone.levels
        enc = net.encoder
        proj = [_conv(p, d) for p in net.encoder_input_proj]
        aifi = _aifi(enc.aifi, d)
        lateral = [_conv(m, d) for m in enc.lateral_convs]
        fpn = [_elan(m, d) for m in enc.fpn_blocks]
        down = [csp.chain(_conv(m.conv1, d), _conv(m.conv2, d)) for m in enc.downsample_convs]
        pan = [_elan(m, d) for m in enc.pan_blocks]
        dproj = [_conv(p, d) for p in net.decoder_input_proj]

        def first(x: mx.array) -> mx.array:
            return stages[0](stem(x))

        def top(c: mx.array) -> mx.array:
            p = proj[-1](c)
            _, h, w, ch = p.shape
            mem = aifi(p.reshape(p.shape[0], h * w, ch), mx.array(_position(h, w, ch)).astype(d))
            return mem.reshape(p.shape)

        def fpn_step(last: mx.array, *feats: mx.array) -> tuple[mx.array, ...]:
            # feats: the backbone maps below the last level, finest first
            inner = [last]
            for i, (lat, blk) in enumerate(zip(lateral, fpn, strict=True)):
                inner[0] = lat(inner[0])
                inner.insert(0, blk(_cat(csp.up2(inner[0]), proj[len(feats) - 1 - i](feats[len(feats) - 1 - i]))))
            return tuple(inner)

        def pan_step(*inner: mx.array) -> tuple[mx.array, ...]:
            outs = [inner[0]]
            for i, (dn, blk) in enumerate(zip(down, pan, strict=True)):
                outs.append(blk(_cat(dn(outs[-1]), inner[i + 1])))
            return tuple(p(o) for p, o in zip(dproj, outs, strict=True)) if dproj else tuple(outs)

        self.levels = levels
        self.image = [csp.stepped(f) for f in (first, *stages[1:], top, fpn_step, pan_step)]
        self.decoder = csp.stepped(self._decoder_fn(net, d))

    @staticmethod
    def _decoder_fn(net: Any, d: mx.Dtype) -> Callable[..., tuple[mx.array, mx.array]]:
        f = mx.float32
        dec = net.decoder
        heads = (_lin(net.enc_output[0], f), _norm(net.enc_output[1], f), _lin(net.enc_score_head, f))
        heads = (*heads, _mlp(net.enc_bbox_head, f))
        qpos_head, pre_bbox = _mlp(dec.query_pos_head, f), _mlp(dec.pre_bbox_head, f)
        layers = [_dfine_layer(layer, d) for layer in dec.layers]
        bbox_heads = [_mlp(m, f) for m in dec.bbox_embed]
        class_head, lqe = _lin(dec.class_embed, f), _mlp(dec.lqe, f)
        project = csp.arr(dec.project, f)
        reg = float(dec.reg_scale)
        queries, top_k, bins = net.queries, 4, project.shape[0]

        def distance2bbox(ref: mx.array, dist: mx.array) -> mx.array:
            wh = ref[..., 2:] / reg
            tl = ref[..., :2] - (0.5 * reg + dist[..., :2]) * wh
            br = ref[..., :2] + (0.5 * reg + dist[..., 2:]) * wh
            return mx.concatenate([(tl + br) / 2, br - tl], axis=-1)

        def apply(*maps: mx.array) -> tuple[mx.array, mx.array]:
            shapes = tuple((int(m.shape[1]), int(m.shape[2])) for m in maps)
            mem = mx.concatenate([m.reshape(m.shape[0], -1, m.shape[-1]) for m in maps], axis=1)
            tgt, ref = _select(mem, shapes, heads, queries, d)
            ref_init = ref
            prev: mx.array | float = 0.0
            corners: mx.array | float = 0.0
            h = prob = ref
            for i, (layer, bbox_head) in enumerate(zip(layers, bbox_heads, strict=True)):
                qpos = mx.clip(qpos_head(ref), -10, 10).astype(d)
                tgt = layer(tgt, ref, mem, shapes, qpos)
                h = tgt.astype(f)
                if i == 0:
                    ref_init = mx.sigmoid(pre_bbox(h) + _inverse_sigmoid(ref))
                corners = bbox_head(h + prev) + corners
                prob = mx.softmax(corners.reshape(*corners.shape[:-1], 4, bins), axis=-1)
                ref = distance2bbox(ref_init, prob @ project)
                prev = h
            top = -mx.sort(-prob, axis=-1)[..., :top_k]
            stat = mx.concatenate([top, top.mean(axis=-1, keepdims=True)], axis=-1)
            return _linear(h, class_head) + lqe(stat.reshape(*stat.shape[:-2], -1)), ref

        return apply

    def forward(self, x: mx.array) -> tuple[mx.array, mx.array]:
        first, *rest = self.image
        stages, (top, fpn_step, pan_step) = rest[:-3], rest[-3:]
        outs = [first(x)]
        for s in stages:
            outs.append(s(outs[-1]))
        feats = outs[-self.levels :]
        inner = fpn_step(top(feats[-1]), *feats[:-1])
        return self.decoder(*pan_step(*inner))


def _dfine_layer(m: Any, d: mx.Dtype) -> Callable[..., mx.array]:
    """D-FINE's `DecoderLayer`: self-attention, gated deformable cross-attention on the maps, FFN (clamped)."""
    att = _Attention(m.self_attn, d)
    ea = m.encoder_attn
    points = list(ea.points)
    n, pmax = sum(points), max(points)
    off, aw = _lin(ea.sampling_offsets, d), _lin(ea.attention_weights, d)
    offw = mx.concatenate([off[0], aw[0]], axis=1), mx.concatenate([off[1], aw[1]])
    no = off[0].shape[1]
    scale = csp.arr(ea.num_points_scale[:, None] * 0.5, d)  # OFFSET_SCALE
    gate, gnorm = _lin(m.gateway.gate, d), _norm(m.gateway.norm, d)
    n1, n3 = _norm(m.self_attn_layer_norm, d), _norm(m.final_layer_norm, d)
    fc1, fc2 = _lin(m.fc1, d), _lin(m.fc2, d)
    act = _silu if m.act.__name__ == "silu" else _relu
    starts = np.cumsum([0, *points]).tolist()

    def ragged(t: mx.array) -> mx.array:
        """`[B, Q, H, n, ...]` -> `[B, Q, H, L, max points, ...]`, each level zero-padded."""
        if len(set(points)) == 1:
            return t.reshape(*t.shape[:3], len(points), pmax, *t.shape[4:])
        parts = []
        for s, p in zip(starts, points, strict=False):
            part = t[:, :, :, s : s + p]
            if p < pmax:
                pad = [(0, 0)] * t.ndim
                pad[3] = (0, pmax - p)
                part = mx.pad(part, pad)
            parts.append(part)
        return mx.stack(parts, axis=3)

    def apply(x: mx.array, ref: mx.array, mem: mx.array, shapes: Sequence[tuple[int, int]], qpos: mx.array) -> mx.array:
        b, q, dim = x.shape
        x = _ln(x + att(x + qpos, x), n1)
        ow = _linear(x + qpos, offw)
        o = ow[..., :no].reshape(b, q, HEADS, n, 2)
        w = mx.softmax(ow[..., no:].reshape(b, q, HEADS, n), axis=-1)
        r = ref.astype(d)[:, :, None, None]
        loc = r[..., :2] + o * scale * r[..., 2:]
        v = mem.reshape(b, mem.shape[1], HEADS, dim // HEADS)
        ca = ops.msda(v, ragged(loc), ragged(w), shapes)
        g1, g2 = mx.split(mx.sigmoid(_linear(_cat(x, ca), gate)), 2, axis=-1)
        x = _ln(g1 * x + g2 * ca, gnorm)
        y = x + _linear(act(_linear(x, fc1)), fc2)
        return _ln(mx.clip(y, -65504.0, 65504.0), n3)

    return apply


def build(net: torch.nn.Module, dtype: str) -> _Net:
    """The MLX network of an `RtDetr` or a `DFine` (float32, CPU, BatchNorm folded) in `dtype` (the name)."""
    d = getattr(mx, dtype)
    if type(net).__name__ == "DFine":
        return _MlxDFine(net, d)
    return _MlxRtDetr(net, d)
