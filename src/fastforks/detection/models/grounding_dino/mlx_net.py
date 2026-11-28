"""Grounding DINO on Apple MLX, hand-written from the checkpoint's tensors (no transformers network): one image.

The maths of transformers' `GroundingDinoForObjectDetection` as `model.Portable.image` runs it (two-stage, every pixel
and token valid), channels last, the run dtype unless noted:

- Pixels (`pixels`): uint8 values, the antialiased bilinear resize as two matrix products (PyTorch's separable
  weights), rounded; ImageNet mean / std; float32 NHWC.
- Swin backbone: the input padded to the patch, the patch convolution as one GEMM, the layers on `shared.mlx_swin`
  (transformers' Swin: pad after the LayerNorm, -100 shift mask; a map no wider than the window gets no shift), each
  odd map zero-padded before `PatchMerging`; stages 2-4 normed.
- Input projections: 1 x 1 convolutions as GEMMs, the fourth level a 3 x 3 stride-2 convolution of the stage-4 map,
  GroupNorm (32) with float32 statistics.
- Text tower (`text`, once per query list): BERT with the phrase mask and phrase position ids, weights stored in the
  run dtype and computed in float32 (upcast layer by layer), the projection; the text position table (transformers
  casts the sine of the integer ids back to int64: only the exact +-1 survive) gathered by the position ids.
- Encoder layers: the bi-directional fusion (one score matrix, softmax over text for the vision side and over pixels
  for the text side, float32 softmax), the text enhancer (phrase mask), the deformable layer (offsets and weights one
  GEMM in float32, the shared `msda` kernel).
- Query selection: the proposals' logits, the encoder output's best token score (float32), the top 900 in descending
  order, the box MLP on those rows only; reference points float32.
- Decoder: query position from the sine of the reference boxes, self-attention, text cross-attention, deformable
  cross-attention, FFN; the refinement of the reference points float32 (layers 0-4: the last one's is not used).
- Heads float32: logits of the normed last hidden state against the text features (MM-Grounding-DINO: / sqrt(d) plus
  its bias), boxes of the last head plus the logit of the last input reference points.

The arrays come through `shared.mlx_port.tensor` (the conversion cache: on a hit `sd` holds shapes only).
"""

# ruff: noqa: ANN401 - the checkpoint's torch tensors

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.errors import InputError
from fastforks._core.mlx import attention, mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_port, mlx_swin, mlx_vit
from fastforks.shared.swin_sr import mlx_net as swin

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LAYOUT = "grounding-dino-v1"
SOURCES = (Path(__file__), Path(mlx_swin.__file__), Path(swin.__file__))
MEAN = (0.485, 0.456, 0.406)  # fixed: ImageNet normalisation, as `model.MEAN`
STD = (0.229, 0.224, 0.225)  # fixed: as `model.STD`
EPS = 1e-5  # fixed: the transformer's LayerNorm / GroupNorm eps (config `layer_norm_eps`, GroupNorm's default)
BERT_EPS = 1e-12  # fixed: BERT's LayerNorm eps
GROUPS = 32  # fixed: the input projections' GroupNorm groups
HEADS = 8  # fixed: deformable / decoder attention heads (config `encoder_attention_heads`, `decoder_attention_heads`)
LEVELS, POINTS = 4, 4  # fixed: feature levels and sampling points per level
REF_EPS = 1e-5  # fixed: the decoder's `torch.special.logit(eps=1e-5)`
VALID = (0.01, 0.99)  # fixed: proposals with a coordinate outside are invalid (`generate_encoder_output_proposals`)
_SW = "model.backbone.conv_encoder.model."
_BERT = "model.text_backbone."


def _affine(x: mx.array, a: mx.array, b: mx.array) -> mx.array:
    return (x.astype(mx.float32) * a + b).astype(x.dtype)


_AFFINE = mx.compile(_affine, shapeless=True)


def _group_norm(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """GroupNorm (32 groups) of `[P, C]`: float32 statistics and arithmetic, rounded once."""
    c = x.shape[1]
    mean, var = ops.channel_stats(x)
    gm = mean.reshape(GROUPS, -1).mean(axis=1)
    gv = mx.maximum((var + mx.square(mean)).reshape(GROUPS, -1).mean(axis=1) - mx.square(gm), 0)
    rstd = mx.repeat(mx.rsqrt(gv + EPS), c // GROUPS)
    a = rstd * w.astype(mx.float32)
    return _AFFINE(x, a, b.astype(mx.float32) - mx.repeat(gm, c // GROUPS) * a)


def _np(t: Any) -> np.ndarray:
    return t.detach().double().numpy()


def _lin(sd: Mapping[str, Any], name: str, dt: mx.Dtype) -> tuple[mx.array, mx.array | None]:
    """`name` (an `nn.Linear`'s tensors) as (`[in, out]`, bias)."""
    return swin.linear(SimpleNamespace(weight=sd[name + ".weight"], bias=sd.get(name + ".bias")), dt)


def _fused(sd: Mapping[str, Any], names: Sequence[str], dt: mx.Dtype) -> tuple[mx.array, mx.array]:
    """Linears of one input as one: `[in, sum(out)]` and the biases."""
    ws = [sd[n + ".weight"] for n in names]
    bs = [sd[n + ".bias"] for n in names]
    i, o = int(ws[0].shape[1]), sum(int(w.shape[0]) for w in ws)
    w = swin.derived((i, o), dt, lambda: np.concatenate([_np(t) for t in ws]).T)
    return w, swin.derived((o,), dt, lambda: np.concatenate([_np(t) for t in bs]))


def _norm(sd: Mapping[str, Any], name: str, dt: mx.Dtype) -> tuple[mx.array, mx.array]:
    return swin.arr(sd[name + ".weight"], dt), swin.arr(sd[name + ".bias"], dt)


def _linear(x: mx.array, wb: tuple[mx.array, mx.array | None]) -> mx.array:
    return mlx_swin.linear(x, wb)


def _relu_linear(x: mx.array, a: tuple[mx.array, mx.array | None], b: tuple[mx.array, mx.array | None]) -> mx.array:
    return _linear(mx.maximum(_linear(x, a), 0), b)


def _mlp(x: mx.array, layers: Sequence[tuple[mx.array, mx.array | None]]) -> mx.array:
    """`GroundingDinoMLPPredictionHead`: ReLU between the layers."""
    for i, wb in enumerate(layers):
        x = _linear(x, wb)
        if i < len(layers) - 1:
            x = mx.maximum(x, 0)
    return x


def _ln(x: mx.array, wb: tuple[mx.array, mx.array]) -> mx.array:
    return mlx_swin.ln(x, *wb)


def _heads(x: mx.array, heads: int) -> mx.array:
    """`[L, heads * D]` -> `[1, heads, L, D]`."""
    return x.reshape(1, x.shape[0], heads, -1).transpose(0, 2, 1, 3)


def _merge(x: mx.array) -> mx.array:
    """`[1, heads, L, D]` -> `[L, heads * D]`."""
    return x.transpose(0, 2, 1, 3).reshape(x.shape[2], -1)


def _attend(q: mx.array, k: mx.array, v: mx.array, heads: int, mask: mx.array | None = None) -> mx.array:
    """Multi-head attention of projected rows `[Lq, E]`, `[Lk, E]` -> `[Lq, E]`."""
    d = q.shape[-1] // heads
    return _merge(attention(_heads(q, heads), _heads(k, heads), _heads(v, heads), scale=d**-0.5, mask=mask))


def logit(x: mx.array, eps: float) -> mx.array:
    """`torch.special.logit(x, eps)`."""
    x = mx.clip(x, eps, 1 - eps)
    return mx.log(x / (1 - x))


def aa_matrix(inp: int, out: int) -> np.ndarray:
    """`[out, inp]` weights of PyTorch's antialiased bilinear resize along one axis (`mlx_graph._aa_matrix`)."""
    s = inp / out
    support, inv = max(s, 1.0), (1.0 / s if s >= 1.0 else 1.0)
    m = np.zeros((out, inp), dtype=np.float64)
    for i in range(out):
        c = s * (i + 0.5)
        lo, hi = max(int(c - support + 0.5), 0), min(int(c + support + 0.5), inp)
        w = np.maximum(0.0, 1.0 - np.abs((np.arange(lo, hi) - c + 0.5) * inv))
        m[i, lo:hi] = w / w.sum()
    return m.astype(np.float32)


def image_sine(h: int, w: int) -> np.ndarray:
    """`GroundingDinoSinePositionEmbedding` of an all-valid `h x w` map (temperature 20): `[h * w, 256]` float32."""
    eps, scale = np.float32(1e-6), np.float32(2 * math.pi)
    y = np.arange(1, h + 1, dtype=np.float32) / (np.float32(h) + eps) * scale
    x = np.arange(1, w + 1, dtype=np.float32) / (np.float32(w) + eps) * scale
    k = np.arange(128, dtype=np.float32)
    dim_t = np.float32(20) ** (2 * np.floor(k / 2) / np.float32(128))

    def emb(v: np.ndarray) -> np.ndarray:
        p = v[:, None] / dim_t
        return np.where(np.arange(128) % 2 == 0, np.sin(p), np.cos(p)).astype(np.float32)

    py, px = emb(y), emb(x)
    return np.concatenate(
        [np.broadcast_to(py[:, None], (h, w, 128)), np.broadcast_to(px[None], (h, w, 128))], -1
    ).reshape(h * w, 256)


def text_sine_table(n: int) -> np.ndarray:
    """`get_sine_pos_embed` of the integer position ids `0..n-1` (256 features), float32 `[n, 256]`.

    transformers casts the result back to int64 (`.to(pos_tensor.dtype)`): the sines truncated.
    """
    k = np.arange(256, dtype=np.float32)
    dim_t = np.float32(10000) ** (2 * np.floor(k / 2) / np.float32(256))
    p = (np.arange(n, dtype=np.float32) * np.float32(2 * math.pi))[:, None] / dim_t
    return np.trunc(np.where(np.arange(256) % 2 == 0, np.sin(p), np.cos(p))).astype(np.float32)


def _ref_sine(ref: mx.array) -> mx.array:
    """`get_sine_pos_embed` of the reference boxes `[Q, 4]` (128 features each, `[y, x, w, h]`): `[Q, 512]` float32."""
    k = np.arange(128, dtype=np.float32)
    dim_t = mx.array(np.float32(10000) ** (2 * np.floor(k / 2) / np.float32(128)))
    e = (ref * np.float32(2 * math.pi))[..., None] / dim_t  # [Q, 4, 128]
    even = mx.array(np.arange(128) % 2 == 0)
    e = mx.where(even, mx.sin(e), mx.cos(e))
    return mx.concatenate([e[:, 1], e[:, 0], e[:, 2], e[:, 3]], axis=-1)


def _count(sd: Mapping[str, Any], prefix: str) -> int:
    """The number of `prefix{i}.` modules in the checkpoint."""
    pat = re.compile(re.escape(prefix) + r"(\d+)\.")
    return 1 + max(int(m.group(1)) for k in sd if (m := pat.match(k)))


def _first(sd: Mapping[str, Any], names: Sequence[str], suffix: str) -> str:
    """The first of `names` whose `suffix` tensor the checkpoint holds."""
    for n in names:
        if n + suffix in sd:
            return n
    msg = f"none of {list(names)} in the checkpoint"
    raise KeyError(msg)


def _bert_layer(sd: Mapping[str, Any], p: str, dt: mx.Dtype) -> dict[str, Any]:
    """A BERT layer's arrays (`p`: its prefix)."""
    return {
        "qkv": _fused(sd, [p + f"attention.self.{n}" for n in ("query", "key", "value")], dt),
        "o": _lin(sd, p + "attention.output.dense", dt),
        "ln1": _norm(sd, p + "attention.output.LayerNorm", dt),
        "fc1": _lin(sd, p + "intermediate.dense", dt),
        "fc2": _lin(sd, p + "output.dense", dt),
        "ln2": _norm(sd, p + "output.LayerNorm", dt),
    }  # fmt: skip


def _enc_layer(sd: Mapping[str, Any], p: str, dt: mx.Dtype) -> dict[str, Any]:
    """An encoder layer's arrays: fusion, text enhancer, deformable layer."""
    f, t, dfm = p + "fusion_layer.", p + "text_enhancer_layer.", p + "deformable_layer."
    sa = dfm + "self_attn."
    return {
        "ln_v": _norm(sd, f + "layer_norm_vision", dt),
        "ln_t": _norm(sd, f + "layer_norm_text", dt),
        "v_proj": _fused(sd, [f + "attn.vision_proj", f + "attn.values_vision_proj"], dt),
        "t_proj": _fused(sd, [f + "attn.text_proj", f + "attn.values_text_proj"], dt),
        "out_v": _lin(sd, f + "attn.out_vision_proj", dt),
        "out_t": _lin(sd, f + "attn.out_text_proj", dt),
        "gamma_v": swin.arr(sd[f + "vision_param"], dt),
        "gamma_t": swin.arr(sd[f + "text_param"], dt),
        "t_qk": _fused(sd, [t + "self_attn.query", t + "self_attn.key"], dt),
        "t_v": _lin(sd, t + "self_attn.value", dt),
        "t_o": _lin(sd, t + "self_attn.out_proj", dt),
        "t_ln1": _norm(sd, t + "layer_norm_before", dt),
        "t_fc1": _lin(sd, t + "fc1", dt),
        "t_fc2": _lin(sd, t + "fc2", dt),
        "t_ln2": _norm(sd, t + "layer_norm_after", dt),
        "value": _lin(sd, sa + "value_proj", dt),
        "offw": _fused(sd, [sa + "sampling_offsets", sa + "attention_weights"], dt),
        "out": _lin(sd, sa + "output_proj", dt),
        "ln1": _norm(sd, dfm + "self_attn_layer_norm", dt),
        "fc1": _lin(sd, dfm + "fc1", dt),
        "fc2": _lin(sd, dfm + "fc2", dt),
        "ln2": _norm(sd, dfm + "final_layer_norm", dt),
    }  # fmt: skip


def _dec_layer(sd: Mapping[str, Any], p: str, dt: mx.Dtype) -> dict[str, Any]:
    """A decoder layer's arrays."""
    ca = p + "encoder_attn."
    return {
        "qk": _fused(sd, [p + "self_attn.query", p + "self_attn.key"], dt),
        "v": _lin(sd, p + "self_attn.value", dt),
        "o": _lin(sd, p + "self_attn.out_proj", dt),
        "ln1": _norm(sd, p + "self_attn_layer_norm", dt),
        "tq": _lin(sd, p + "encoder_attn_text.query", dt),
        "tkv": _fused(sd, [p + "encoder_attn_text.key", p + "encoder_attn_text.value"], dt),
        "to": _lin(sd, p + "encoder_attn_text.out_proj", dt),
        "ln2": _norm(sd, p + "encoder_attn_text_layer_norm", dt),
        "value": _lin(sd, ca + "value_proj", dt),
        "offw": _fused(sd, [ca + "sampling_offsets", ca + "attention_weights"], dt),
        "out": _lin(sd, ca + "output_proj", dt),
        "ln3": _norm(sd, p + "encoder_attn_layer_norm", dt),
        "fc1": _lin(sd, p + "fc1", dt),
        "fc2": _lin(sd, p + "fc2", dt),
        "ln4": _norm(sd, p + "final_layer_norm", dt),
    }  # fmt: skip


@dataclass(frozen=True)
class Text:
    """Text features of one caption (MLX arrays, the run dtype)."""

    features: mx.array  # [T, 256]: the projected BERT output
    positions: mx.array  # [T, 256]: the text position embedding
    mask: mx.array  # [1, 1, T, T] bool: tokens attend within their phrase


class _SwinBlock(mlx_swin.Block):
    def __init__(self, sd: Mapping[str, Any], p: str, shift: bool, dt: mx.Dtype) -> None:  # noqa: FBT001
        a = p + "attention.self."
        table = sd[a + "relative_position_bias_table"]
        rows, heads = (int(d) for d in table.shape)
        c = int(sd[a + "query.weight"].shape[0])
        self.window = (math.isqrt(rows) + 1) // 2
        self.shift = self.window // 2 if shift else 0
        self.heads, self.head_dim = heads, c // heads
        self.scale = self.head_dim**-0.5
        self.qkv = _fused(sd, [a + n for n in ("query", "key", "value")], dt)
        self.o = _lin(sd, p + "attention.output.dense", dt)
        self.fc1, self.fc2 = _lin(sd, p + "intermediate.dense", dt), _lin(sd, p + "output.dense", dt)
        t, idx = self.window * self.window, sd[a + "relative_position_index"]

        def rpb() -> np.ndarray:
            return _np(table)[idx.numpy().reshape(-1)].reshape(t, t, heads).transpose(2, 0, 1)[None]

        self.rpb = swin.derived((1, heads, t, t), dt, rpb)
        self.ln1, self.ln2 = _norm(sd, p + "layernorm_before", dt), _norm(sd, p + "layernorm_after", dt)
        self.unshifted = self
        if self.shift:
            self.unshifted = copy.copy(self)
            self.unshifted.shift = 0

    def run(self, x: mx.array, hw: tuple[int, int]) -> mx.array:
        """The layer on `hw`: a map no wider than the window is one unshifted window (`set_shift_and_window_size`)."""
        if min(hw) > self.window:
            return self(x, hw)
        if min(hw) < self.window:
            msg = f"the image is too narrow for the backbone (a {hw} map, {self.window} x {self.window} windows)"
            raise InputError(msg)
        return self.unshifted(x, hw)


class Network:
    """The Grounding DINO network of a checkpoint's tensors (`config`: the variant's `config.json`)."""

    def __init__(self, sd: Mapping[str, Any], config: Mapping[str, Any], dtype: str) -> None:  # noqa: PLR0915 - one builder
        """The arrays of the state dict `sd` (float32 tensors, or meta ones on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        f32 = mx.float32
        self.dtype = dt
        self.queries = int(config.get("num_queries", 900))
        self.d = int(config.get("d_model", 256))
        # Swin
        proj = sd[_SW + "embeddings.patch_embeddings.projection.weight"]  # [C, 3, p, p]
        c0, _, self.patch, _ = (int(v) for v in proj.shape)
        self.embed = (
            swin.derived((3 * self.patch**2, c0), dt, lambda: _np(proj).transpose(2, 3, 1, 0).reshape(-1, c0)),
            swin.arr(sd[_SW + "embeddings.patch_embeddings.projection.bias"], dt),
        )
        self.embed_norm = _norm(sd, _SW + "embeddings.norm", dt)
        n_layers = _count(sd, _SW + "encoder.layers.")
        self.stages: list[tuple[list[_SwinBlock], Any]] = []
        for i in range(n_layers):
            p = f"{_SW}encoder.layers.{i}."
            depth = _count(sd, p + "blocks.")
            blocks = [_SwinBlock(sd, f"{p}blocks.{j}.", j % 2 == 1, dt) for j in range(depth)]
            ds = None
            if p + "downsample.reduction.weight" in sd:
                ds = (_norm(sd, p + "downsample.norm", dt), _lin(sd, p + "downsample.reduction", dt)[0])
            self.stages.append((blocks, ds))
        self.stage_norms = {i: _norm(sd, f"{_SW}hidden_states_norms.stage{i + 1}", dt) for i in range(1, n_layers)}
        # input projections
        self.in_proj = []
        for lvl in range(LEVELS):
            w = sd[f"model.input_proj_vision.{lvl}.0.weight"]  # [256, C, k, k]
            o, ci, kh, kw = (int(v) for v in w.shape)
            if kh == 1:
                wt = swin.derived((ci, o), dt, lambda w=w: _np(w)[:, :, 0, 0].T)
            else:
                wt = swin.derived((o, kh, kw, ci), dt, lambda w=w: _np(w).transpose(0, 2, 3, 1))
            b = swin.arr(sd[f"model.input_proj_vision.{lvl}.0.bias"], dt)
            self.in_proj.append(((wt, b), _norm(sd, f"model.input_proj_vision.{lvl}.1", dt)))
        self.level_embed = swin.arr(sd["model.level_embed"], f32)
        # text tower: weights in the run dtype, computed in float32
        self.word = swin.arr(sd[_BERT + "embeddings.word_embeddings.weight"], dt)
        self.word_pos = swin.arr(sd[_BERT + "embeddings.position_embeddings.weight"], dt)
        self.word_type = swin.arr(sd[_BERT + "embeddings.token_type_embeddings.weight"][0], dt)
        self.word_norm = _norm(sd, _BERT + "embeddings.LayerNorm", dt)
        n_bert = _count(sd, _BERT + "encoder.layer.")
        self.bert = [_bert_layer(sd, f"{_BERT}encoder.layer.{i}.", dt) for i in range(n_bert)]
        self.bert_heads = int(config.get("text_config", {}).get("num_attention_heads", 12))
        self.text_proj = _lin(sd, "model.text_projection", dt)
        self.text_table = text_sine_table(int(config.get("max_text_len", 256)))
        # encoder
        n_enc = int(config.get("encoder_layers", 6))
        self.enc = [_enc_layer(sd, f"model.encoder.layers.{i}.", dt) for i in range(n_enc)]
        self.fusion_heads = int(config.get("encoder_attention_heads", 8)) // 2
        # query selection
        self.enc_output = _lin(sd, "model.enc_output", dt)
        self.enc_norm = _norm(sd, "model.enc_output_norm", dt)
        self.enc_box = [_lin(sd, f"model.encoder_output_bbox_embed.layers.{i}", dt) for i in range(3)]
        self.target = swin.arr(sd["model.query_position_embeddings.weight"], dt)
        # decoder
        self.ref_head = [_lin(sd, f"model.decoder.reference_points_head.layers.{i}", dt) for i in range(2)]
        n_dec = int(config.get("decoder_layers", 6))
        self.dec = [_dec_layer(sd, f"model.decoder.layers.{i}.", dt) for i in range(n_dec)]
        self.dec_norm = _norm(sd, "model.decoder.layer_norm", dt)

        def head(i: int) -> str:
            return _first(sd, [f"bbox_embed.{i}", f"model.decoder.bbox_embed.{i}", "bbox_embed.0",
                               "model.decoder.bbox_embed.0"], ".layers.0.weight")  # fmt: skip

        refine: dict[str, list[tuple[mx.array, mx.array | None]]] = {}  # the refinement heads, shared ones once
        self.refine = []
        for i in range(n_dec - 1):  # the last layer's refined points are not used
            name = head(i)
            if name not in refine:
                refine[name] = [_lin(sd, f"{name}.layers.{j}", dt) for j in range(3)]
            self.refine.append(refine[name])
        self.box_head = [_lin(sd, f"{head(n_dec - 1)}.layers.{j}", f32) for j in range(3)]
        cls = [f"class_embed.{n_dec - 1}", f"model.decoder.class_embed.{n_dec - 1}", "class_embed.0",
               "model.decoder.class_embed.0"]  # fmt: skip
        self.logit_bias = None  # MM-Grounding-DINO: logits / sqrt(d) + a learned bias
        if any(n + ".bias" in sd for n in cls):
            self.logit_bias = swin.arr(sd[_first(sd, cls, ".bias") + ".bias"], f32)
        self.mean = mx.array(MEAN, f32)
        self.std = mx.array(STD, f32)
        self._tables: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
        self._resize: dict[tuple[int, int, int, int], tuple[mx.array, mx.array]] = {}

    # pre-processing

    def pixels(self, image: mx.array, size: tuple[int, int]) -> mx.array:
        """`[1, 3, H, W]` float32 in [0, 1] -> the normalised network input `[1, h, w, 3]` float32 at `size`."""
        x = mx.clip(mx.round(image[0] * 255), 0, 255)
        h, w = x.shape[1:]
        if (h, w) != tuple(size):
            key = (h, w, *size)
            if key not in self._resize:
                self._resize = {key: (mx.array(aa_matrix(h, size[0])), mx.array(aa_matrix(w, size[1]).T))}
            mh, mw = self._resize[key]
            x = mx.clip(mx.round(mh @ x @ mw), 0, 255)
        return ((x.transpose(1, 2, 0) / 255 - self.mean) / self.std)[None]

    # text

    def text(self, ids: np.ndarray, att: np.ndarray, pos: np.ndarray) -> Text:
        """Token ids `[T]`, the phrase mask `[T, T]` (True: attend) and position ids `[T]` -> text features."""
        f32 = mx.float32
        n = len(ids)
        i, p = mx.array(ids.astype(np.int32)), mx.array(pos.astype(np.int32))
        x = (mx.take(self.word, i, axis=0).astype(f32) + mx.take(self.word_pos, p, axis=0).astype(f32)
             + self.word_type.astype(f32))  # fmt: skip
        x = mx.fast.layer_norm(x, *(a.astype(f32) for a in self.word_norm), BERT_EPS)
        mask = mx.array(att)[None, None]
        heads = self.bert_heads
        for layer in self.bert:
            w = {k: tuple(None if a is None else a.astype(f32) for a in v) for k, v in layer.items()}
            qkv = _linear(x, w["qkv"]).reshape(1, n, 3, heads, -1).transpose(2, 0, 3, 1, 4)
            d = qkv.shape[-1]
            o = _merge(attention(qkv[0], qkv[1], qkv[2], scale=d**-0.5, mask=mask))
            x = mx.fast.layer_norm(x + _linear(o, w["o"]), *w["ln1"], BERT_EPS)
            h = _linear(mlx_vit.gelu(_linear(x, w["fc1"])), w["fc2"])
            x = mx.fast.layer_norm(x + h, *w["ln2"], BERT_EPS)
            mx.eval(x)
        proj = tuple(None if a is None else a.astype(f32) for a in self.text_proj)
        feat = mlx_port.tap("text_projection", _linear(x, proj)[None])[0].astype(self.dtype)
        tpos = mx.array(self.text_table[pos]).astype(self.dtype)
        out = Text(feat, tpos, mx.array(att)[None, None])
        mx.eval(out.features, out.positions, out.mask)
        return out

    # image

    def _swin(self, x: mx.array) -> list[tuple[mx.array, tuple[int, int]]]:
        """Normalised `[1, H, W, 3]` -> the normed stage 2-4 maps `([h * w, C], (h, w))`."""
        ps = self.patch
        hp, wp = -(-x.shape[1] // ps) * ps, -(-x.shape[2] // ps) * ps
        x = mx.pad(x.astype(self.dtype), [(0, 0), (0, hp - x.shape[1]), (0, wp - x.shape[2]), (0, 0)])
        h, w = hp // ps, wp // ps
        t = x.reshape(h, ps, w, ps, 3).transpose(0, 2, 1, 3, 4).reshape(h * w, -1)
        t = mlx_swin.ln(mx.addmm(self.embed[1], t, self.embed[0]), *self.embed_norm)
        feats = []
        for i, (blocks, ds) in enumerate(self.stages):
            for blk in blocks:
                t = blk.run(t, (h, w))
                mx.eval(t)
            if i in self.stage_norms:
                feats.append((mx.contiguous(_ln(t, self.stage_norms[i])), (h, w)))
            if ds is not None:
                c = t.shape[-1]
                m = t.reshape(h, w, c)
                if h % 2 or w % 2:
                    m = mx.pad(m, [(0, h % 2), (0, w % 2), (0, 0)])
                h, w = (h + 1) // 2, (w + 1) // 2
                m = m.reshape(h, 2, w, 2, c).transpose(0, 2, 3, 1, 4).reshape(h * w, 4 * c)
                t = _ln(m, ds[0]) @ ds[1]
                mx.eval(t, feats)
        return feats

    def _table(self, shapes: tuple[tuple[int, int], ...]) -> dict[str, Any]:
        """Per input size: level positions, encoder reference points, offset normalisers, proposals."""
        if shapes not in self._tables:
            le = np.array(self.level_embed)
            pos = np.concatenate([image_sine(h, w) + le[i] for i, (h, w) in enumerate(shapes)])
            grids = [
                np.stack(np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h), -1).reshape(-1, 2)
                for h, w in shapes
            ]
            ref = np.concatenate(grids).astype(np.float32)
            prop = np.concatenate([
                np.concatenate([g, np.full_like(g, 0.05 * 2.0**lvl)], -1) for lvl, g in enumerate(grids)
            ]).astype(np.float32)  # fmt: skip
            valid = ((prop > VALID[0]) & (prop < VALID[1])).all(-1, keepdims=True)
            with np.errstate(divide="ignore"):
                prop_logit = np.where(valid, np.log(prop / (1 - prop)), np.inf).astype(np.float32)
            tab = {
                "pos": mx.array(pos.astype(np.float32)).astype(self.dtype),
                "ref": mx.array(ref).reshape(1, -1, 1, 1, 1, 2),
                "norm": mx.array(np.asarray([[w, h] for h, w in shapes], np.float32)).reshape(1, 1, 1, -1, 1, 2),
                "proposals": mx.array(prop_logit),
                "valid": mx.array(valid),
            }
            mx.eval(tab)
            self._tables = {shapes: tab}  # the last input size only
        return self._tables[shapes]

    def _levels(self, feats: list[tuple[mx.array, tuple[int, int]]]) -> tuple[mx.array, tuple[tuple[int, int], ...]]:
        """The projected, normed levels `[S, 256]` (flattened one after the other) and their shapes."""
        maps, shapes = [], []
        for lvl, ((wt, b), gn) in enumerate(self.in_proj):
            if lvl < len(feats):
                f, hw = feats[lvl]
                y = mx.addmm(b, f, wt)
            else:
                f, (h, w) = feats[-1]
                y = mx.conv2d(f.reshape(1, h, w, -1), wt, stride=2, padding=1) + b
                hw = (y.shape[1], y.shape[2])
                y = y.reshape(-1, y.shape[-1])
            maps.append(mlx_port.tap(f"input_proj_vision.{lvl}", _group_norm(y, *gn).reshape(1, *hw, -1)))
            shapes.append(hw)
        src = mx.concatenate([m.reshape(-1, m.shape[-1]) for m in maps])
        mx.eval(src)
        return src, tuple(shapes)

    def _deformable(
        self,
        p: dict[str, Any],
        q: mx.array,
        value: mx.array,
        shapes: tuple[tuple[int, int], ...],
        loc: Any,
    ) -> mx.array:
        """Multi-scale deformable attention: `loc(offsets [1, Q, heads, L, P, 2])` -> sampling locations."""
        nq, s, no = q.shape[0], value.shape[0], HEADS * LEVELS * POINTS * 2
        v = _linear(value, p["value"]).reshape(1, s, HEADS, -1)
        ow = mx.addmm(p["offw"][1], q, p["offw"][0]).astype(mx.float32)
        off = ow[:, :no].reshape(1, nq, HEADS, LEVELS, POINTS, 2)
        aw = mx.softmax(ow[:, no:].reshape(1, nq, HEADS, -1), axis=-1).reshape(1, nq, HEADS, LEVELS, POINTS)
        return _linear(ops.msda(v, loc(off), aw, shapes)[0], p["out"])

    def _fusion(self, p: dict[str, Any], v: mx.array, t: mx.array) -> tuple[mx.array, mx.array]:
        """`GroundingDinoFusionLayer`: the bi-directional attention of the normed vision and text features."""
        heads, e = self.fusion_heads, p["out_v"][0].shape[0]
        vn, tn = _ln(v, p["ln_v"]), _ln(t, p["ln_t"])
        qv = _linear(vn, p["v_proj"])
        kt = _linear(tn, p["t_proj"])
        d = e // heads
        qh, vv = _heads(qv[:, :e], heads), _heads(qv[:, e:], heads)  # [1, heads, S, d]
        kh, vt = _heads(kt[:, :e], heads), _heads(kt[:, e:], heads)  # [1, heads, T, d]
        scores = (qh * d**-0.5) @ kh.transpose(0, 1, 3, 2)  # [1, heads, S, T]
        ov = mx.softmax(scores, axis=-1, precise=True) @ vt
        ot = mx.softmax(scores.transpose(0, 1, 3, 2), axis=-1, precise=True) @ vv
        v = vn + p["gamma_v"] * _linear(_merge(ov), p["out_v"])
        t = tn + p["gamma_t"] * _linear(_merge(ot), p["out_t"])
        return v, t

    def _encoder(
        self, v: mx.array, txt: Text, shapes: tuple[tuple[int, int], ...], tab: dict[str, Any]
    ) -> tuple[mx.array, mx.array]:
        t = txt.features
        for i, p in enumerate(self.enc):
            v, t = self._fusion(p, v, t)
            qk = _linear(t + txt.positions, p["t_qk"])
            e = t.shape[-1]
            o = _attend(qk[:, :e], qk[:, e:], _linear(t, p["t_v"]), self.fusion_heads, txt.mask)
            t = _ln(t + _linear(o, p["t_o"]), p["t_ln1"])
            t = _ln(t + _relu_linear(t, p["t_fc1"], p["t_fc2"]), p["t_ln2"])
            a = self._deformable(p, v + tab["pos"], v, shapes, lambda off: tab["ref"] + off / tab["norm"])
            v = _ln(v + a, p["ln1"])
            v = _ln(v + _relu_linear(v, p["fc1"], p["fc2"]), p["ln2"])
            mlx_port.tap(f"encoder.layers.{i}[0]", v[None])
            mlx_port.tap(f"encoder.layers.{i}[1]", t[None])
            mx.eval(v, t)
        return v, t

    def _select(self, v: mx.array, t: mx.array, tab: dict[str, Any]) -> mx.array:
        """Language-guided query selection: the top `queries` reference boxes `[Q, 4]` float32 (best first)."""
        f32 = mx.float32
        obj = _ln(_linear(mx.where(tab["valid"], v, 0), self.enc_output), self.enc_norm)
        mlx_port.tap("enc_output_norm", obj[None])
        score = mx.max(obj.astype(f32) @ t.astype(f32).T, axis=-1)
        top = mx.argsort(-score)[: self.queries]
        coords = _mlp(obj[top], self.enc_box).astype(f32) + tab["proposals"][top]
        ref = mx.sigmoid(coords)
        mx.eval(ref)
        return ref

    def _decoder(
        self, vis: mx.array, t: mx.array, ref: mx.array, shapes: tuple[tuple[int, int], ...]
    ) -> tuple[mx.array, mx.array]:
        """-> the last hidden state `[Q, 256]` and its input reference points `[Q, 4]` float32."""
        f32 = mx.float32
        h = self.target
        e = h.shape[-1]
        prev = ref
        for i, p in enumerate(self.dec):
            qpos = _mlp(_ref_sine(ref).astype(self.dtype), self.ref_head)
            q = h + qpos
            qk = _linear(q, p["qk"])
            h = _ln(h + _linear(_attend(qk[:, :e], qk[:, e:], _linear(h, p["v"]), HEADS), p["o"]), p["ln1"])
            q = h + qpos
            kv = _linear(t, p["tkv"])
            h = _ln(h + _linear(_attend(_linear(q, p["tq"]), kv[:, :e], kv[:, e:], HEADS), p["to"]), p["ln2"])
            r = ref.reshape(1, -1, 1, 1, 1, 4)
            a = self._deformable(p, h + qpos, vis, shapes, lambda off, r=r: r[..., :2] + off / POINTS * r[..., 2:] / 2)
            h = _ln(h + a, p["ln3"])
            h = _ln(h + _relu_linear(h, p["fc1"], p["fc2"]), p["ln4"])
            mlx_port.tap(f"decoder.layers.{i}[0]", h[None])
            prev = ref
            if i < len(self.refine):
                ref = mx.sigmoid(_mlp(h, self.refine[i]).astype(f32) + logit(ref, REF_EPS))
            mx.eval(h, ref)
        return h, prev

    def image(self, x: mx.array, txt: Text) -> tuple[mx.array, mx.array]:
        """Normalised `[1, H, W, 3]` float32 + text features -> token logits `[Q, T]`, boxes `[Q, 4]`, float32.

        Boxes as centre / size, 0..1 of the image; both evaluated.
        """
        f32 = mx.float32
        src, shapes = self._levels(self._swin(x))
        tab = self._table(shapes)
        vis, t = self._encoder(src, txt, shapes, tab)
        ref = self._select(vis, t, tab)
        h, prev = self._decoder(vis, t, ref, shapes)
        hid = _ln(h, self.dec_norm).astype(f32)
        logits = hid @ t.astype(f32).T
        if self.logit_bias is not None:
            logits = logits / math.sqrt(self.d) + self.logit_bias
        boxes = mx.sigmoid(_mlp(hid, self.box_head) + logit(prev, REF_EPS))
        mx.eval(logits, boxes)
        return logits, boxes


def network(sd: Any, config: Mapping[str, Any], dtype: str) -> Network:
    """The MLX Grounding DINO of a state dict (float32 or meta tensors) in `dtype`."""
    return Network(sd, config, dtype)
