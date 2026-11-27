"""Mask2Former / OneFormer (Swin-L) on Apple MLX, written in `mlx.core` operations (`MlxMaskFormer(model, dtype)`).

The maths of the transformers networks as `fastforks.shared.maskformer` runs them, channels last, one image:

- Swin: `shared.mlx_swin` (LayerNorm + pad + cyclic shift + window partition as one gather kernel, the inverse gather
  with the residual and the second LayerNorm another; float16 window attention the shared `window_attention` kernel,
  float32 `mx.fast.scaled_dot_product_attention`).
- Pixel decoder: the deformable attention after its projections is the shared `msda` kernel (offsets and attention
  weights one GEMM, their softmax and the sampling locations float32), GroupNorm statistics the `channel_stats` kernel.
- Transformer decoder: the masked cross-attention's mask needs the mask logits only at the attended level's size;
  bilinear resizing is linear, so they are the query embeddings times the mask features resized once per level
  (`[Q, HW_l]` instead of the `[Q, H/4, W/4]` logits and their resize per layer). Only the last layer's mask logits
  are made at 1/4 resolution, and only its class logits.
- OneFormer: the task token of the fixed task is computed once in float32 (transformers casts the token ids to the
  model dtype first); the query transformer (over the 1/4-resolution features) runs in float32, as the PyTorch path
  (its values over the sine table of the map are cached per size).

Weights are converted once, layer by layer (no second float32 copy of the model is kept).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastforks._core.mlx import attention, mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_swin, mlx_vit

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["MlxMaskFormer", "cache_mb"]


def cache_mb(dtype: str) -> int:
    """MLX buffer cache the runners request (`request_cache`).

    The cache is resident: 1024 MB was 0.9 GB of footprint for 1-2% of warm time. float32 slows sooner as it shrinks
    (OneFormer 1024 / 512 / 256 MB: 952 / 962 / 989 ms; float16 833 / 837 / 845).
    """
    return 256 if dtype == "float16" else 512


_Hf = Any  # transformers modules
_EPS = 1e-5  # every LayerNorm / GroupNorm of both networks (the configs' layer_norm_eps and torch's default)
_GROUPS = 32  # the pixel decoder's GroupNorms
_HEADS = 8  # pixel decoder and transformer decoder attention heads (the configs' num_attention_heads)

_ln = mlx_swin.ln
_linear = mlx_swin.linear


def _affine(x: mx.array, a: mx.array, b: mx.array) -> mx.array:
    return (x.astype(mx.float32) * a + b).astype(x.dtype)


def _affine_relu(x: mx.array, a: mx.array, b: mx.array) -> mx.array:
    return mx.maximum(x.astype(mx.float32) * a + b, 0).astype(x.dtype)


_AFFINE = mx.compile(_affine, shapeless=True)
_AFFINE_RELU = mx.compile(_affine_relu, shapeless=True)


def _group_norm(x: mx.array, w: mx.array, b: mx.array, *, relu: bool = False) -> mx.array:
    """GroupNorm (32 groups) of `[P, C]`, optionally + ReLU: float32 statistics and arithmetic, rounded once."""
    c = x.shape[1]
    mean, var = ops.channel_stats(x)
    gm = mean.reshape(_GROUPS, -1).mean(axis=1)
    gv = mx.maximum((var + mx.square(mean)).reshape(_GROUPS, -1).mean(axis=1) - mx.square(gm), 0)
    rstd = mx.repeat(mx.rsqrt(gv + _EPS), c // _GROUPS)
    a = rstd * w
    return (_AFFINE_RELU if relu else _AFFINE)(x, a, b - mx.repeat(gm, c // _GROUPS) * a)


def _arrays(obj: object) -> list[mx.array]:
    """Every MLX array held by `obj` (through lists, tuples, dicts and this module's weight holders)."""
    if isinstance(obj, mx.array):
        return [obj]
    if isinstance(obj, dict):
        obj = list(obj.values())
    elif type(obj).__module__ == __name__ and hasattr(obj, "__dict__"):
        obj = list(vars(obj).values())
    if isinstance(obj, (list, tuple)):
        return [a for o in obj for a in _arrays(o)]
    return []


def _arr(t: torch.Tensor, dtype: mx.Dtype) -> mx.array:
    return mx.array(t.detach().float().numpy()).astype(dtype)


def _lin(weight: torch.Tensor, bias: torch.Tensor | None, dtype: mx.Dtype) -> tuple[mx.array, mx.array | None]:
    """`[in, out]` weight (contiguous: MLX's GEMM copies a transposed view on every call) and bias."""
    w = mx.contiguous(_arr(weight, dtype).T)
    return w, None if bias is None else _arr(bias, dtype)


def _attend(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None = None) -> mx.array:
    """Multi-head attention of projected `[Lq, E]` / `[Lk, E]` rows (8 heads) -> `[Lq, E]`."""
    e = q.shape[-1]
    d = e // _HEADS

    def heads(t: mx.array) -> mx.array:
        return t.reshape(1, -1, _HEADS, d).transpose(0, 2, 1, 3)

    o = attention(heads(q), heads(k), heads(v), scale=d**-0.5, mask=mask)
    return o.transpose(0, 2, 1, 3).reshape(-1, e)


class _Swin(mlx_swin.Block):
    """One Swin layer's weights in the run dtype."""

    def __init__(self, m: _Hf, dtype: mx.Dtype) -> None:
        a = m.attention
        self.window, self.shift = int(m.window_size), int(m.shift_size)
        self.heads, self.head_dim = int(a.num_attention_heads), int(a.head_dim)
        self.scale = float(a.scaling)
        qkv_w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
        qkv_b = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias])
        self.qkv = _lin(qkv_w, qkv_b, dtype)
        self.o = _lin(a.o_proj.weight, a.o_proj.bias, dtype)
        self.fc1 = _lin(m.mlp.fc1.weight, m.mlp.fc1.bias, dtype)
        self.fc2 = _lin(m.mlp.fc2.weight, m.mlp.fc2.bias, dtype)
        with torch.no_grad():
            self.rpb = _arr(a.relative_position_bias(), dtype)  # [1, heads, T, T]
        self.ln1 = (_arr(m.layernorm_before.weight, dtype), _arr(m.layernorm_before.bias, dtype))
        self.ln2 = (_arr(m.layernorm_after.weight, dtype), _arr(m.layernorm_after.bias, dtype))


class _MultiheadAttention:
    """An `nn.MultiheadAttention` (packed in-projection) or a DETR-style attention (q / k / v / out projections)."""

    def __init__(self, m: _Hf, dtype: mx.Dtype) -> None:
        if hasattr(m, "in_proj_weight"):
            w, b = m.in_proj_weight, m.in_proj_bias
            e = w.shape[1]
            ws, bs = (w[:e], w[e : 2 * e], w[2 * e :]), (b[:e], b[e : 2 * e], b[2 * e :])
        else:
            ws = (m.q_proj.weight, m.k_proj.weight, m.v_proj.weight)
            bs = (m.q_proj.bias, m.k_proj.bias, m.v_proj.bias)
        self.q, self.k, self.v = (_lin(wi, bi, dtype) for wi, bi in zip(ws, bs, strict=True))
        self.qk = _lin(torch.cat(ws[:2]), torch.cat(bs[:2]), dtype)
        self.out = _lin(m.out_proj.weight, m.out_proj.bias, dtype)

    def self_attention(self, x: mx.array, pos: mx.array) -> mx.array:
        q, k = mx.split(_linear(x + pos, self.qk), 2, axis=-1)
        return _linear(_attend(q, k, _linear(x, self.v)), self.out)

    def cross(self, x: mx.array, pos: mx.array, k: mx.array, v: mx.array, mask: mx.array | None) -> mx.array:
        """`k` / `v`: the projected keys and values."""
        return _linear(_attend(_linear(x + pos, self.q), k, v, mask), self.out)


def _norm(m: _Hf, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    return _arr(m.weight, dtype), _arr(m.bias, dtype)


class _DecoderLayer:
    """A masked-attention decoder layer (post-norm): cross-attention, self-attention, FFN."""

    def __init__(self, layer: _Hf, dtype: mx.Dtype) -> None:
        if hasattr(layer, "ffn"):  # OneFormer
            self.cross = _MultiheadAttention(layer.cross_attn.multihead_attn, dtype)
            self.self = _MultiheadAttention(layer.self_attn.self_attn, dtype)
            norms = (layer.cross_attn.norm, layer.self_attn.norm, layer.ffn.norm)
            fc1, fc2 = layer.ffn.linear1, layer.ffn.linear2
        else:  # Mask2Former
            self.cross = _MultiheadAttention(layer.cross_attn, dtype)
            self.self = _MultiheadAttention(layer.self_attn, dtype)
            norms = (layer.cross_attn_layer_norm, layer.self_attn_layer_norm, layer.final_layer_norm)
            fc1, fc2 = layer.fc1, layer.fc2
        self.norms = [_norm(n, dtype) for n in norms]
        self.fc1, self.fc2 = _lin(fc1.weight, fc1.bias, dtype), _lin(fc2.weight, fc2.bias, dtype)

    def __call__(self, x: mx.array, qpos: mx.array, feat: mx.array, pos: mx.array, mask: mx.array) -> mx.array:
        c = self.cross
        k, v = _linear(feat + pos, c.k), _linear(feat, c.v)
        x = _ln(x + c.cross(x, qpos, k, v, mask), *self.norms[0])
        x = _ln(x + self.self.self_attention(x, qpos), *self.norms[1])
        h = mx.maximum(_linear(x, self.fc1), 0)
        return _ln(x + _linear(h, self.fc2), *self.norms[2])


class _QueryTransformer:
    """OneFormer's query transformer (float32): DETR decoder layers over the 1/4-resolution map, then its norm."""

    def __init__(self, qt: _Hf) -> None:
        f32 = mx.float32
        dec = qt.decoder
        self.layers = []
        for layer in dec.layers:
            self.layers.append((
                _MultiheadAttention(layer.self_attn, f32),
                _MultiheadAttention(layer.multihead_attn, f32),
                [_norm(n, f32) for n in (layer.norm1, layer.norm2, layer.norm3)],
                _lin(layer.linear1.weight, layer.linear1.bias, f32),
                _lin(layer.linear2.weight, layer.linear2.bias, f32),
            ))  # fmt: skip
        self.norm = _norm(dec.norm, f32)
        self._values: dict[tuple[int, int], tuple[mx.array, list[mx.array]]] = {}

    def __call__(self, x: mx.array, qpos: mx.array, memory: mx.array, mpos: mx.array) -> mx.array:
        """Queries `[N, E]`, memory (the sine table of the map) and its position (the projected mask features)."""
        for sa, ca, norms, fc1, fc2 in self.layers:
            x = _ln(x + sa.self_attention(x, qpos), *norms[0])
            k, v = _linear(memory + mpos, ca.k), _linear(memory, ca.v)
            x = _ln(x + ca.cross(x, qpos, k, v, None), *norms[1])
            x = _ln(x + _linear(mx.maximum(_linear(x, fc1), 0), fc2), *norms[2])
        return _ln(x, *self.norm)


class MlxMaskFormer:
    """The network of one loaded model: weights in the run dtype, `raw(x)` -> mask logits and class logits on MLX.

    Args:
        model: `Mask2FormerForUniversalSegmentation` or `OneFormerForUniversalSegmentation` (float32, eval; not kept).
        dtype: `"float16"` or `"float32"`.
        task_inputs: OneFormer: the task's token ids (CLIP tokenizer, zero-padded to the config's `task_seq_len`).
    """

    def __init__(self, model: _Hf, dtype: str, task_inputs: Sequence[int] | None = None) -> None:
        """Convert the weights (layer by layer)."""
        self.dtype = getattr(mx, dtype)
        self._load_backbone(model.model.pixel_level_module.encoder)
        self._load_pixel_decoder(model.model.pixel_level_module.decoder)
        self._load_decoder(model, task_inputs)
        mx.eval(_arrays(self))
        self._tables: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
        self._memory: dict[tuple[int, int], mx.array] = {}

    def _load_backbone(self, bb: _Hf) -> None:
        dt = self.dtype

        sw = bb.swin
        emb = sw.embeddings
        proj = emb.patch_embeddings.projection
        self.patch = int(proj.kernel_size[0])
        pw = proj.weight.detach().float().numpy().transpose(2, 3, 1, 0).reshape(-1, proj.weight.shape[0])
        self.embed = (mx.contiguous(mx.array(pw).astype(dt)), _arr(proj.bias, dt))
        self.embed_norm = _norm(emb.norm, dt)
        self.stages: list[tuple[list[_Swin], Any]] = []
        for st in sw.encoder.layers:
            blocks = [_Swin(b, dt) for b in st.blocks]
            ds = st.downsample
            merge = None if ds is None else (_norm(ds.norm, dt), _lin(ds.reduction.weight, None, dt)[0])
            self.stages.append((blocks, merge))
            mx.eval([b.__dict__ for b in blocks], merge)
        self.stage_norms = [_norm(bb.hidden_states_norms[f"stage{i + 1}"], dt) for i in range(len(self.stages))]

    def _load_pixel_decoder(self, pd: _Hf) -> None:
        dt = self.dtype
        self.pos_embed = pd.position_embedding  # a parameter-free module: the sine tables, float32 on the CPU
        self.level_embed = mx.array(pd.level_embed.detach().float().numpy())
        self.in_proj = [
            (_lin(p[0].weight[:, :, 0, 0], p[0].bias, dt), _norm(p[1], mx.float32)) for p in pd.input_projections
        ]
        self.enc = []
        for layer in pd.encoder.layers:
            a = layer.self_attn
            self.levels, self.points = int(a.n_levels), int(a.n_points)
            ow = torch.cat([a.sampling_offsets.weight, a.attention_weights.weight])
            ob = torch.cat([a.sampling_offsets.bias, a.attention_weights.bias])
            self.enc.append({
                "value": _lin(a.value_proj.weight, a.value_proj.bias, dt),
                "offw": _lin(ow, ob, dt),
                "out": _lin(a.output_proj.weight, a.output_proj.bias, dt),
                "fc1": _lin(layer.fc1.weight, layer.fc1.bias, dt),
                "fc2": _lin(layer.fc2.weight, layer.fc2.bias, dt),
                "ln1": _norm(layer.self_attn_layer_norm, dt),
                "ln2": _norm(layer.final_layer_norm, dt),
            })  # fmt: skip
        self.n_offsets = int(pd.encoder.layers[0].self_attn.sampling_offsets.weight.shape[0])
        lat, out = pd.adapter_1, pd.layer_1
        self.lateral = (_lin(lat[0].weight[:, :, 0, 0], None, dt)[0], _norm(lat[1], mx.float32))
        self.fpn_conv = mx.contiguous(_arr(out[0].weight, dt).transpose(0, 2, 3, 1))  # [out, 3, 3, in]
        self.fpn_norm = _norm(out[1], mx.float32)
        self.mask_proj = _lin(pd.mask_projection.weight[:, :, 0, 0], pd.mask_projection.bias, dt)

    def _load_decoder(self, model: _Hf, task_inputs: Sequence[int] | None) -> None:
        dt = self.dtype
        m = model.model
        tm = m.transformer_module
        self.levels_embed = _arr(tm.level_embed.weight, dt)
        self.qpos = _arr(tm.queries_embedder.weight, dt)
        dec = tm.decoder
        self.layers = [_DecoderLayer(layer, dt) for layer in dec.layers]
        self.oneformer = hasattr(m, "task_encoder")
        if self.oneformer:
            if not task_inputs:
                msg = "OneFormer needs task_inputs"
                raise ValueError(msg)
            task = torch.zeros((1, model.config.task_seq_len), dtype=torch.long)
            task[0, : len(task_inputs)] = torch.tensor(list(task_inputs))
            with torch.no_grad():
                token = dec.decoder_norm(m.task_encoder(task.float()))  # [1, E], float32
            self.task = mx.array(token.numpy())
            self.query_transformer = _QueryTransformer(dec.query_transformer)
            self.qt_pos = mx.array(tm.queries_embedder.weight.detach().float().numpy()[:-1])
            qp = dec.query_input_projection
            self.qt_proj = _lin(qp.weight[:, :, 0, 0], qp.bias, mx.float32)
            self.head_norm = _norm(dec.decoder_norm, dt)
            head, cls = dec.mask_embed, dec.class_embed
            self.init = None
        else:
            self.init = _arr(tm.queries_features.weight, dt)
            self.head_norm = _norm(dec.layernorm, dt)
            head, cls = dec.mask_predictor.mask_embedder, model.class_predictor
        self.mask_mlp = [_lin(lin.weight, lin.bias, dt) for lin in head.modules() if isinstance(lin, torch.nn.Linear)]
        self.cls = _lin(cls.weight, cls.bias, dt)

    def _sine(self, h: int, w: int) -> np.ndarray:
        """The sine position table `[h * w, E]` (float32) of an `h x w` map (normalised, as every table here)."""
        with torch.no_grad():
            t = self.pos_embed(torch.Size((1, 1, h, w)), "cpu", torch.float32)
        return t[0].permute(1, 2, 0).reshape(h * w, -1).numpy()

    def _table(self, shapes: tuple[tuple[int, int], ...]) -> dict[str, Any]:
        """Per input size: position tables, reference points, offset normalisers (pixel decoder levels `shapes`)."""
        if shapes not in self._tables:
            sines = [self._sine(h, w) for h, w in shapes]
            le = np.asarray(self.level_embed)
            pos = np.concatenate([s + le[i] for i, s in enumerate(sines)])
            ref = np.concatenate([
                np.stack(np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h), -1).reshape(-1, 2)
                for h, w in shapes
            ]).astype(np.float32)  # fmt: skip
            norm = np.asarray([[w, h] for h, w in shapes], np.float32)
            tab = {
                "pos": mx.array(pos).astype(self.dtype),
                "ref": mx.array(ref).reshape(1, -1, 1, 1, 1, 2),
                "norm": mx.array(norm).reshape(1, 1, 1, len(shapes), 1, 2),
                "dec_pos": [mx.array(s).astype(self.dtype) for s in sines],
            }
            mx.eval(tab)
            self._tables = {shapes: tab}  # the last input size only
        return self._tables[shapes]

    def _backbone(self, x: mx.array) -> list[tuple[mx.array, tuple[int, int]]]:
        """Normalised `[1, 3, H, W]` -> the four normed stage maps `([h * w, C], (h, w))`."""
        h, w = x.shape[2] // self.patch, x.shape[3] // self.patch
        t = mlx_vit.patches(x.astype(self.dtype), self.patch)[0]
        t = _ln(mx.addmm(self.embed[1], t, self.embed[0]), *self.embed_norm)
        feats = []
        for (blocks, merge), norm in zip(self.stages, self.stage_norms, strict=True):
            for blk in blocks:
                t = blk(t, (h, w))
                mx.eval(t)
            feats.append((_ln(t, *norm), (h, w)))
            if merge is not None:
                c = t.shape[-1]
                t = t.reshape(h // 2, 2, w // 2, 2, c).transpose(0, 2, 3, 1, 4).reshape(-1, 4 * c)
                h, w = h // 2, w // 2
                t = _ln(t, *merge[0]) @ merge[1]
        mx.eval([f for f, _ in feats])
        return feats

    def _pixel_decoder(self, feats: list[tuple[mx.array, tuple[int, int]]]) -> tuple[list[mx.array], mx.array, Any]:
        """-> the three encoder levels `[HW_l, E]` (1/32, 1/16, 1/8), the mask features `[H/4 * W/4, E]`, the table."""
        srcs = feats[::-1][: len(self.in_proj)]
        shapes = tuple(hw for _, hw in srcs)
        tab = self._table(shapes)
        x = mx.concatenate([
            _group_norm(_linear(f, wb), *gn) for (f, _), (wb, gn) in zip(srcs, self.in_proj, strict=True)
        ])  # fmt: skip
        s = x.shape[0]
        no = self.n_offsets
        for p in self.enc:
            q = x + tab["pos"]
            v = _linear(x, p["value"]).reshape(1, s, _HEADS, -1)
            ow = _linear(q, p["offw"]).astype(mx.float32)
            off = ow[:, :no].reshape(1, s, _HEADS, self.levels, self.points, 2)
            aw = mx.softmax(ow[:, no:].reshape(1, s, _HEADS, -1), axis=-1)
            loc = tab["ref"] + off / tab["norm"]
            a = ops.msda(v, loc, aw.reshape(1, s, _HEADS, self.levels, self.points), shapes)[0]
            x = _ln(x + _linear(a, p["out"]), *p["ln1"])
            x = _ln(x + _linear(mx.maximum(_linear(x, p["fc1"]), 0), p["fc2"]), *p["ln2"])
            mx.eval(x)
        starts = np.cumsum([0] + [h * w for h, w in shapes])
        levels = [x[starts[i] : starts[i + 1]] for i in range(len(shapes))]
        (f1, (h, w)), (h8, w8) = feats[0], shapes[-1]
        e = x.shape[-1]
        y = _group_norm(f1 @ self.lateral[0], *self.lateral[1])
        up = ops.bilinear_nhwc(levels[-1].reshape(1, h8, w8, e), (h, w), (h8 / h, w8 / w), align=False)
        y = mx.conv2d((y + up.reshape(-1, e)).reshape(1, h, w, e), self.fpn_conv, padding=1).reshape(-1, e)
        y = _group_norm(y, *self.fpn_norm, relu=True)
        mf = _linear(y, self.mask_proj)
        mx.eval(mf, levels)
        return levels, mf, tab

    def _embed(self, x: mx.array) -> mx.array:
        """Normed queries -> mask embeddings (the mask MLP)."""
        for i, wb in enumerate(self.mask_mlp):
            x = _linear(x, wb)
            if i < len(self.mask_mlp) - 1:
                x = mx.maximum(x, 0)
        return x

    def raw(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Normalised `[1, 3, H, W]` (sides multiples of 32) -> mask logits `[H/4, W/4, Q]`, class logits `[Q, 151]`.

        Mask logits in the run dtype, class logits float32.
        """
        feats = self._backbone(x)
        levels, mf, tab = self._pixel_decoder(feats)
        h4, w4 = feats[0][1]
        shapes = [hw for _, hw in feats[::-1][: len(levels)]]
        e = mf.shape[-1]
        mf_map = mf.reshape(1, h4, w4, e)
        mfl = [
            mx.contiguous(ops.bilinear_nhwc(mf_map, hw, (h4 / hw[0], w4 / hw[1]), align=False).reshape(-1, e).T)
            for hw in shapes
        ]  # fmt: skip
        feats_l = [lv + self.levels_embed[i] for i, lv in enumerate(levels)]
        if self.oneformer:
            memory, mpos = self._qt_memory(h4, w4), _linear(mf.astype(mx.float32), self.qt_proj)
            n = self.qt_pos.shape[0]
            obj = self.query_transformer(mx.repeat(self.task, n, axis=0), self.qt_pos, memory, mpos)
            out = mx.concatenate([obj, self.task]).astype(self.dtype)
        else:
            out = self.init
        mx.eval(out)

        def mask(out: mx.array, level: int) -> mx.array:
            keep = self._embed(_ln(out, *self.head_norm)) @ mfl[level] >= 0  # sigmoid(logit) >= 0.5
            keep = keep | ~mx.any(keep, axis=-1, keepdims=True)  # a fully masked query attends everywhere
            return keep[None, None]

        m = mask(out, 0)
        for i, layer in enumerate(self.layers):
            lv = i % len(levels)
            out = layer(out, self.qpos, feats_l[lv], tab["dec_pos"][lv], m)
            if i < len(self.layers) - 1:
                m = mask(out, (i + 1) % len(levels))
            mx.eval(out, m)
        h = _ln(out, *self.head_norm)
        masks = (mf @ self._embed(h).T).reshape(h4, w4, -1)
        classes = _linear(h, self.cls).astype(mx.float32)
        return masks, classes

    def _qt_memory(self, h: int, w: int) -> mx.array:
        """The query transformer's memory: the float32 sine table of the 1/4-resolution map (cached per size)."""
        if (h, w) not in self._memory:
            t = mx.array(self._sine(h, w))
            mx.eval(t)
            self._memory = {(h, w): t}  # the last input size only
        return self._memory[(h, w)]
