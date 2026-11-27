"""LW-DETR on Apple MLX, written in `mlx.core` operations (`MlxLWDETR(state, config, dtype, size)`): image -> outputs.

The maths of `model.Network` (transformers' `LwDetrForObjectDetection`, one query group, every pixel valid), channels
last, weights read from the PyTorch twin's state (BatchNorms folded). Written out instead of replayed from an exported
graph: the export, its translation and the compile of the whole graph were 1.9-2.7 s of each first call.

- Input: `square`'s uint8 values; the `/ 255` and the ImageNet normalisation are folded into the patch embedding
  (exact in real arithmetic: the patch convolution has no padding), its bias into the position table. The pretraining
  position table (14 x 14) is resized bicubically to the patch grid once, at load, as transformers does per call.
- Backbone: the ViT keeps the window layout throughout (`[B * windows, tokens, C]`); its global blocks attend over the
  same rows viewed as `[B, windows * tokens, C]` (attention ignores the row order). LayerScale is folded into the
  projections (`mlx_vit.block`; the key projection has no bias: a zero one).
- Projector: per level, the sampling layers (identity at scale 1, a 2x2 stride-2 transposed conv as one matrix
  product at scale 2, a stride-2 3x3 conv at scale 0.5), the C2f stage (1x1 convs as matrix products, 3x3 convs
  `mx.conv2d`, folded BatchNorms, SiLU) and the closing channel LayerNorm.
- From the projector on the network runs in float32, as RF-DETR's (`shared.rfdetr.mlx_net`): the two-stage selection
  (its top-k decides which queries exist) and the decoder are a small share of the time. The two-stage selection and
  the decoder are RF-DETR's (the same maths: unrefined reference points, deformable cross-attention as the shared
  `msda` kernel over one or two levels); only the weights' names and the invalid proposals' `-inf` scores differ.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx, to_mx, to_torch
from fastforks.shared import mlx_vit
from fastforks.shared.rfdetr.mlx_net import MlxRFDETR

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

__all__ = ["MlxLWDETR"]

_Wb = tuple[mx.array, mx.array]
_VIT = "model.backbone.backbone."
_PROJ = "model.backbone.projector.scale_layers."
_DEC = "model.decoder."
_EPS_VIT = 1e-6  # the ViT LayerNorms and the projector's channel LayerNorm (`LwDetrLayerNorm`)
_EPS = 1e-5  # torch's default: the decoder's and the two-stage LayerNorms
SYNC_BYTES = 8 << 20
"""A C2f bottleneck output from which the projector evaluates bottleneck by bottleneck and level by level: lazy, the
float32 3 x 3 conv workspaces of every bottleneck and level were alive at once (lw-detr-large float16: network peak
548 -> 324 MB, the graph translation 319). Large / xlarge's stride-8 level (80 x 80 x 384 float32, 9.8 MB) is above
it; the single-level tiny / small / medium maps (40 x 40 x 256, 1.6 MB) stay below: there the evaluations only cost
time (float32 0.96 -> 1.04x the graph's)."""
_VALID = (0.01, 0.99)  # a proposal inside these bounds is valid (`gen_encoder_output_proposals`)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class _State:
    """The twin's state as MLX arrays, each tensor dropped once read (the PyTorch copy shrinks as the MLX one grows)."""

    def __init__(self, state: dict[str, torch.Tensor], dtype: mx.Dtype) -> None:
        self.sd, self.dtype = state, dtype

    def get(self, k: str) -> np.ndarray:
        return self.sd.pop(k).detach().double().numpy()

    def arr(self, v: np.ndarray, dtype: mx.Dtype | None = None) -> mx.array:
        """`v` in `dtype` (default: the run dtype), cast by NumPy: no float32 copy left in MLX's buffer cache."""
        half = (dtype or self.dtype) == mx.float16
        return mx.array(np.ascontiguousarray(v, dtype=np.float16 if half else np.float32))

    def lin(self, k: str) -> _Wb:
        return self.arr(self.get(k + ".weight").T), self.arr(self.get(k + ".bias"))

    def norm(self, k: str) -> _Wb:
        return self.arr(self.get(k + ".weight")), self.arr(self.get(k + ".bias"))

    def mlp(self, k: str, n: int) -> list[_Wb]:
        return [self.lin(f"{k}.layers.{j}") for j in range(n)]

    def conv(self, k: str) -> _Wb:
        """A folded conv: the weight `[in, out]` (1x1) or OHWI, the bias."""
        w = self.get(k + ".weight")
        return self.arr(w[:, :, 0, 0].T if w.shape[-1] == 1 else w.transpose(0, 2, 3, 1)), self.arr(
            self.get(k + ".bias")
        )


def _conv(x: mx.array, wb: _Wb, stride: int = 1) -> mx.array:
    w, b = wb
    if w.ndim == 2:  # noqa: PLR2004 - a 1x1 conv as `[in, out]`
        return mx.addmm(b, x, w)
    return mx.conv2d(x, w, stride=stride, padding=w.shape[1] // 2) + b


class MlxLWDETR(MlxRFDETR):
    """`model.Network` on MLX in `dtype` at a `size` x `size` input (`MlxRFDETR`'s two-stage selection and decoder)."""

    def __init__(self, state: dict[str, torch.Tensor], config: Any, dtype: str, size: int) -> None:  # noqa: ANN401 - transformers config
        """Read the folded twin's `state` (emptied) and make the resolution's fixed tables."""
        self.dtype = getattr(mx, dtype)
        bb = config.backbone_config
        if config.decoder_activation_function != "relu" or config.activation_function != "silu":
            msg = "MlxLWDETR: ReLU decoder FFN and SiLU projector only"
            raise NotImplementedError(msg)
        if bb.hidden_act != "gelu" or not bb.use_absolute_position_embeddings:
            msg = "MlxLWDETR: exact GELU and absolute position embeddings only"
            raise NotImplementedError(msg)
        self.patch, self.vit_heads, self.depth = bb.patch_size, bb.num_attention_heads, bb.num_hidden_layers
        self.side, self.windows = size // self.patch, bb.num_windows_side
        if size % self.patch or self.side % self.windows:
            msg = (
                f"MlxLWDETR: a {size} input does not split into {self.windows}^2 windows of {self.patch}-pixel patches"
            )
            raise ValueError(msg)
        self.global_blocks = frozenset(i for i in range(self.depth) if i not in bb.window_block_indices)
        self.out_blocks = tuple(i - 1 for i in bb.out_indices)  # hidden state i is the output of block i - 1
        self.scales = tuple(float(s) for s in config.projector_scale_factors)
        self.shapes = [(int(self.side * s), int(self.side * s)) for s in self.scales]
        self.num_queries, self.points = int(config.num_queries), int(config.decoder_n_points)
        self.ca_heads, self.sa_heads = (
            int(config.decoder_cross_attention_heads),
            int(config.decoder_self_attention_heads),
        )
        self.kp = self.seg = False
        wt, wq = _State(state, self.dtype), _State(state, mx.float32)
        self._load_vit(wt)
        self.proj: list[dict[str, Any]] = [
            self._load_level(wq, f"{_PROJ}{i}.", s, len(bb.out_indices)) for i, s in enumerate(self.scales)
        ]
        self._load_queries_lw(wq, int(config.d_model))
        self.layers = [self._layer_lw(wq, f"{_DEC}layers.{i}.") for i in range(int(config.decoder_layers))]
        self.dec_norm = wq.norm(_DEC + "layernorm")
        self.bbox, self.cls = wq.mlp("bbox_embed", 3), wq.lin("class_embed")
        state.clear()  # the other query groups (training only) and the BatchNorms' leftovers
        mx.eval(_arrays(self.__dict__))
        mx.clear_cache()  # the build's transients

    def _load_vit(self, wt: _State) -> None:
        """The ViT in `mlx_vit`'s layout, `/ 255` and the normalisation folded into the patch embedding."""
        import torch  # noqa: PLC0415 - loaded with the PyTorch twin
        import torch.nn.functional as F  # noqa: PLC0415

        from fastforks.detection.models.lwdetr.model import MEAN, STD  # noqa: PLC0415 - the runner imports this module

        p, n, vit = self.patch, self.side, {}
        w, b = wt.get(_VIT + "embeddings.projection.weight"), wt.get(_VIT + "embeddings.projection.bias")
        m, s = np.asarray(MEAN).reshape(1, 3, 1, 1), np.asarray(STD).reshape(1, 3, 1, 1)
        b, w = b - (w * m / s).sum((1, 2, 3)), w / (255 * s)
        vit["patch.weight"] = wt.arr(w.transpose(2, 3, 1, 0).reshape(p * p * 3, -1))
        pos = wt.sd.pop(_VIT + "embeddings.position_embeddings")[0, 1:].float()  # the cls slot dropped
        g = math.isqrt(pos.shape[0])
        if g != n:  # `get_absolute_positions`: bicubic, no corner alignment
            grid = pos.reshape(1, g, g, -1).permute(0, 3, 1, 2)
            with torch.no_grad():
                pos = F.interpolate(grid, size=(n, n), mode="bicubic", align_corners=False)[0].permute(1, 2, 0)
        vit["pos.grid"] = wt.arr(pos.reshape(n * n, -1).double().numpy() + b)
        for i in range(self.depth):
            k, o = f"{_VIT}encoder.layer.{i}.", f"blocks.{i}."
            g1, g2 = wt.get(k + "gamma_1"), wt.get(k + "gamma_2")
            att = k + "attention."
            qw, kw, vw = (wt.get(f"{att}{c}_proj.weight") for c in "qkv")
            qb, vb = wt.get(att + "q_proj.bias"), wt.get(att + "v_proj.bias")
            vit[o + "qkv.weight"] = wt.arr(np.concatenate([qw, kw, vw]).T)
            vit[o + "qkv.bias"] = wt.arr(np.concatenate([qb, np.zeros_like(qb), vb]))  # the key: no bias
            vit[o + "proj.weight"] = wt.arr(wt.get(att + "o_proj.weight").T * g1)
            vit[o + "proj.bias"] = wt.arr(wt.get(att + "o_proj.bias") * g1)
            vit[o + "fc1.weight"], vit[o + "fc1.bias"] = wt.lin(k + "intermediate.fc1")
            vit[o + "fc2.weight"] = wt.arr(wt.get(k + "intermediate.fc2.weight").T * g2)
            vit[o + "fc2.bias"] = wt.arr(wt.get(k + "intermediate.fc2.bias") * g2)
            vit[o + "norm1.weight"], vit[o + "norm1.bias"] = wt.norm(k + "layernorm_before")
            vit[o + "norm2.weight"], vit[o + "norm2.bias"] = wt.norm(k + "layernorm_after")
        self.vit = vit

    @staticmethod
    def _load_level(wq: _State, k: str, scale: float, maps: int) -> dict[str, Any]:
        """One projector level (`LwDetrScaleProjector`): sampling layers per map, the C2f stage, the LayerNorm."""
        sampling = []
        for j in range(maps):
            s = f"{k}sampling_layers.{j}.layers."
            if scale == 2.0:  # noqa: PLR2004
                pre = wq.conv(s + "0.conv") if s + "0.conv.weight" in wq.sd else None  # wide maps: 1x1 ReLU first
                t = s + ("1" if pre is not None else "0")
                w = wq.get(t + ".weight")  # ConvTranspose2d `[in, out, 2, 2]` -> `[in, (2, 2, out)]`
                up = (wq.arr(w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1)), wq.arr(np.tile(wq.get(t + ".bias"), 4)))
                sampling.append(("up", pre, up))
            elif scale == 0.5:  # noqa: PLR2004
                sampling.append(("down", wq.conv(s + "0.conv"), None))
            elif scale == 1.0:
                sampling.append(("same", None, None))
            else:
                msg = f"MlxLWDETR: projector scale {scale}"
                raise NotImplementedError(msg)
        c2f = k + "projector_layer."
        nb = sum(1 for n in wq.sd if n.startswith(c2f + "bottlenecks.") and n.endswith(".conv1.conv.weight"))
        return {
            "sampling": sampling,
            "cv1": wq.conv(c2f + "conv1.conv"),
            "cv2": wq.conv(c2f + "conv2.conv"),
            "m": [
                (wq.conv(f"{c2f}bottlenecks.{j}.conv1.conv"), wq.conv(f"{c2f}bottlenecks.{j}.conv2.conv"))
                for j in range(nb)
            ],
            "norm": wq.norm(k + "layer_norm"),
        }

    def _load_queries_lw(self, wq: _State, d: int) -> None:
        """Two-stage selection (group 0): its heads, the fixed proposals per level, the learned queries, the sines."""
        self.enc_output, self.enc_norm = wq.lin("model.enc_output.0"), wq.norm("model.enc_output_norm.0")
        self.enc_class = wq.lin("model.enc_out_class_embed.0")
        self.enc_bbox = wq.mlp("model.enc_out_bbox_embed.0", 3)
        props = []
        for level, (h, w) in enumerate(self.shapes):
            gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
            size = np.full_like(gx, 0.05 * 2.0**level)
            props.append(np.stack([(gx + 0.5) / w, (gy + 0.5) / h, size, size], -1).reshape(h * w, 4))
        prop = np.concatenate(props)
        valid = ((prop > _VALID[0]) & (prop < _VALID[1])).all(-1, keepdims=True)
        self.valid = wq.arr(valid.astype(np.float32))
        self.score_mask = wq.arr(np.where(valid[:, 0], 0.0, -np.inf))
        self.proposals = wq.arr(prop * valid)  # invalid ones are zero (`gen_encoder_output_proposals`)
        self.refpoint = wq.arr(wq.get("model.reference_point_embed.weight")[: self.num_queries])
        self.query_feat = wq.arr(wq.get("model.query_feat.weight")[: self.num_queries])
        nf = d // 2  # `encode_sinusoidal_position_embedding(num_pos_feats=d_model // 2)`
        dim_t = 10000 ** (2 * (np.arange(nf) // 2) / nf)
        self.sine_scale = wq.arr(2 * math.pi / dim_t)
        self.sine_odd = mx.array(np.arange(nf) % 2 == 1)
        self.ref_head = wq.mlp(_DEC + "ref_point_head", 2)

    @staticmethod
    def _layer_lw(wq: _State, k: str) -> dict[str, _Wb]:
        """Decoder layer `k` in `MlxRFDETR._layer`'s layout: q-k one GEMM, offsets and weights one GEMM."""
        sa, ca = k + "self_attn.", k + "cross_attn."
        qk = [wq.get(f"{sa}{c}_proj.{t}") for c in "qk" for t in ("weight", "bias")]
        ow = np.concatenate([wq.get(ca + "sampling_offsets.weight"), wq.get(ca + "attention_weights.weight")])
        ob = np.concatenate([wq.get(ca + "sampling_offsets.bias"), wq.get(ca + "attention_weights.bias")])
        return {
            "qk": (wq.arr(np.concatenate([qk[0], qk[2]]).T), wq.arr(np.concatenate([qk[1], qk[3]]))),
            "v": wq.lin(sa + "v_proj"),
            "sa_out": wq.lin(sa + "o_proj"),
            "value": wq.lin(ca + "value_proj"),
            "offw": (wq.arr(ow.T), wq.arr(ob)),
            "ca_out": wq.lin(ca + "output_proj"),
            "fc1": wq.lin(k + "mlp.fc1"),
            "fc2": wq.lin(k + "mlp.fc2"),
            "norm1": wq.norm(k + "self_attn_layer_norm"),
            "norm2": wq.norm(k + "cross_attn_layer_norm"),
            "norm3": wq.norm(k + "layer_norm"),
        }

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-method-override]
        """`square`'s `[B, 3, size, size]` uint8 values (float, CPU) -> logits `[B, Q, 91]`, boxes `[B, Q, 4]`."""
        logits, boxes = self.raw(to_mx(x.float()))
        mx.eval(logits, boxes)
        return to_torch(logits), to_torch(boxes)

    def raw(self, x: mx.array) -> tuple[mx.array, mx.array]:  # ty: ignore[invalid-method-override]
        """`__call__` on MLX arrays (lazy)."""
        feats = [f.astype(mx.float32) for f in self._vit(x)]  # projector on: float32 (module docstring)
        levels = []
        for p in self.proj:  # level by level from `SYNC_BYTES` on
            levels.append(self._level(feats, p).reshape(x.shape[0], -1, self.query_feat.shape[-1]))
            if levels[-1].nbytes >= SYNC_BYTES:
                mx.eval(levels[-1])
        memory = mx.concatenate(levels, 1)
        if memory.nbytes >= SYNC_BYTES:
            mx.eval(memory)
        ref = self._two_stage(memory)
        hs, _ = self._decoder(memory, ref, None)
        delta = _mlp_heads(hs, self.bbox)
        boxes = mx.concatenate(
            [delta[..., :2] * ref[..., 2:] + ref[..., :2], mx.exp(delta[..., 2:]) * ref[..., 2:]], -1
        )
        return mx.addmm(self.cls[1], hs, self.cls[0]), boxes

    def _vit(self, x: mx.array) -> list[mx.array]:
        """`[B, 3, R, R]` -> the hidden states of `out_indices`, `[B, R/p, R/p, C]` each (no final norm)."""
        v, b, n, nw = self.vit, x.shape[0], self.side, self.windows
        s = n // nw
        t = mx.addmm(v["pos.grid"], mlx_vit.patches(x.astype(self.dtype), self.patch), v["patch.weight"])
        c = t.shape[-1]
        t = t.reshape(b, nw, s, nw, s, c).transpose(0, 1, 3, 2, 4, 5).reshape(b * nw * nw, s * s, c)
        feats, prev = [], None
        for i in range(self.depth):
            if i in self.global_blocks:
                t = mlx_vit.block(v, i, t.reshape(b, -1, c), self.vit_heads, _EPS_VIT).reshape(b * nw * nw, -1, c)
            else:
                t = mlx_vit.block(v, i, t, self.vit_heads, _EPS_VIT)
            if i in self.out_blocks:
                feats.append(t.reshape(b, nw, nw, s, s, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, n, n, c))
            # One block in flight (as `MlxRFDETR._backbone`): start this block, wait for the one before.
            mx.async_eval(t, *feats)
            if prev is not None:
                mx.eval(prev)
            prev = t
        return feats

    @staticmethod
    def _level(feats: Sequence[mx.array], p: dict[str, Any]) -> mx.array:
        """The backbone maps -> one projector level `[B, h, w, D]` (`LwDetrScaleProjector`)."""
        xs = []
        for f, (kind, conv, up) in zip(feats, p["sampling"], strict=True):
            y = f
            if kind == "up":
                if conv is not None:
                    y = mx.maximum(_conv(y, conv), 0)
                b, h, w, _ = y.shape
                y = mx.addmm(up[1], y, up[0]).reshape(b, h, w, 2, 2, -1)
                y = y.transpose(0, 1, 3, 2, 4, 5).reshape(b, 2 * h, 2 * w, -1)
            elif kind == "down":
                y = mx.maximum(_conv(y, conv, 2), 0)
            xs.append(y)
        y = _silu(_conv(mx.concatenate(xs, axis=-1), p["cv1"]))
        half = y.shape[-1] // 2
        ys = [y[..., :half], y[..., half:]]
        for c1, c2 in p["m"]:
            ys.append(_silu(_conv(_silu(_conv(ys[-1], c1)), c2)))
            if ys[-1].nbytes >= SYNC_BYTES // 2:  # bottleneck by bottleneck (`SYNC_BYTES`; a half-width map)
                mx.eval(ys[-1])
        y = _silu(_conv(mx.concatenate(ys, axis=-1), p["cv2"]))
        return mx.fast.layer_norm(y, p["norm"][0], p["norm"][1], _EPS_VIT)


def _mlp_heads(x: mx.array, layers: Sequence[_Wb]) -> mx.array:
    """`LwDetrMLPPredictionHead`: linear layers with ReLU between them."""
    for i, (w, b) in enumerate(layers):
        x = mx.addmm(b, x, w)
        if i < len(layers) - 1:
            x = mx.maximum(x, 0)
    return x


def _arrays(tree: object) -> list[mx.array]:
    """The MLX arrays in nested dicts, lists and tuples (other leaves skipped)."""
    if isinstance(tree, mx.array):
        return [tree]
    if isinstance(tree, dict):
        return [a for v in tree.values() for a in _arrays(v)]
    if isinstance(tree, (list, tuple)):
        return [a for v in tree for a in _arrays(v)]
    return []
