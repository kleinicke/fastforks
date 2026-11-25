"""RF-DETR on Apple MLX, written in `mlx.core` operations (`MlxRFDETR(net, dtype, mean, std)`): image -> raw outputs.

The maths of the exported network (`LWDETR.forward_export`), channels last, weights copied from the PyTorch twin.
Written out instead of replayed from an exported graph: the export, its translation and the compile of the whole graph
were 1.8-2.6 s of each first call.

- Input: the square bilinear resize is the shared `bilinear` kernel; the ImageNet normalization is folded into the
  patch embedding (exact in real arithmetic: the patch convolution has no padding), its bias into the position table.
- Backbone: the windowed DINOv2 keeps the window layout throughout (`[B * windows^2, 1 + tokens, C]`); its global
  blocks attend over the same rows viewed as `[B, windows^2 * (1 + tokens), C]` (attention ignores the row order).
  LayerScale is folded into the projections (`mlx_vit.block`).
- Projector: the C2f stage's 1x1 convs are matrix products, its 3x3 convs `mx.conv2d`, the channel LayerNorms
  `mx.fast.layer_norm` over the last axis.
- From the projector on the network runs in float32: projector, two-stage selection, decoder, box and class heads
  (a small share of the time; in float16 near-tied top-k picks flipped whole detections against the goldens, seg-s /
  seg-m scores 0.033-0.037 against 0.03, seg-m boxes 0.41 with only the projector in float16). The proposals, their
  validity mask and the learned reference points are fixed by the resolution and made once.
- Decoder: the deformable cross-attention after its projections is the shared `msda` kernel (offsets and attention
  weights one GEMM). The reference points are not refined between layers (`lite_refpoint_refine`), so the query
  position embedding and the locations' scale are made once per call.
- Segmentation head (in the run dtype): the depthwise 3x3 convs are the shared `depthwise` kernel; the mask logits one
  matrix product.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx, to_mx, to_torch
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_vit

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import torch

__all__ = ["MlxRFDETR"]

_Wb = tuple[mx.array, mx.array]
_ENC = "backbone.0.encoder.encoder."
_PROJ = "backbone.0.projector.stages.0."
_DEC = "transformer.decoder."
_EPS_VIT = 1e-6  # the DINOv2 LayerNorms, the projector's and the segmentation blocks' channel LayerNorms
_VALID = (0.01, 0.99)  # a proposal inside these bounds is valid (`gen_encoder_output_proposals`)
_EPS = 1e-5  # torch's default: the decoder's and the two-stage LayerNorms, the query block's


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().double().numpy()


def _arrays(tree: object) -> Iterator[mx.array]:
    """The MLX arrays in nested dicts, lists and tuples (other leaves skipped)."""
    if isinstance(tree, mx.array):
        yield tree
    elif isinstance(tree, dict):
        for v in tree.values():
            yield from _arrays(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _arrays(v)


def _linear(x: mx.array, wb: _Wb) -> mx.array:
    return mx.addmm(wb[1], x, wb[0])


def _mlp(x: mx.array, layers: Sequence[_Wb]) -> mx.array:
    """`math.MLP`: linear layers with ReLU between them."""
    for i, wb in enumerate(layers):
        x = _linear(x, wb)
        if i < len(layers) - 1:
            x = mx.maximum(x, 0)
    return x


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _ln(x: mx.array, wb: _Wb, eps: float) -> mx.array:
    return mx.fast.layer_norm(x, wb[0], wb[1], eps)


def _heads(t: mx.array, heads: int) -> mx.array:
    b, n, c = t.shape
    return t.reshape(b, n, heads, c // heads).transpose(0, 2, 1, 3)


class _Weights:
    """The exported network's state, read as MLX arrays in the run dtype (Linear weights as `[in, out]`)."""

    def __init__(self, net: torch.nn.Module, dtype: mx.Dtype) -> None:
        self.sd, self.dtype = net.state_dict(), dtype

    def get(self, k: str) -> np.ndarray:
        return _np(self.sd[k])

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

    def convx(self, k: str) -> tuple[mx.array, _Wb]:
        """A projector ConvX (conv, LayerNorm, SiLU): the weight `[in, out]` (1x1) or OHWI (3x3), the norm."""
        w = self.get(k + ".conv.weight")
        return self.arr(w[:, :, 0, 0].T if w.shape[-1] == 1 else w.transpose(0, 2, 3, 1)), self.norm(k + ".bn")


class MlxRFDETR:
    """The exported RF-DETR network (detection or Seg, one projector level) on MLX in `dtype`."""

    def __init__(self, net: torch.nn.Module, dtype: str, mean: Sequence[float], std: Sequence[float]) -> None:
        """Copy the weights of the exported PyTorch `net` (float32, CPU) and make the resolution's fixed tables."""
        self.dtype = getattr(mx, dtype)
        a: Any = net  # the vendored LWDETR
        cfg, dec = a.backbone[0].encoder.encoder.config, a.transformer.decoder
        if len(a.backbone[0].projector.stages) != 1 or not dec.lite_refpoint_refine or not a.bbox_reparam:
            msg = "MlxRFDETR: one projector level, bbox reparametrization and unrefined reference points only"
            raise NotImplementedError(msg)
        self.resolution = int(a.backbone[0].encoder.shape[0])
        self.patch, self.windows, self.vit_heads = cfg.patch_size, cfg.num_windows, cfg.num_attention_heads
        self.side = self.resolution // self.patch
        self.depth = cfg.num_hidden_layers
        self.global_blocks = frozenset(i for i in range(self.depth) if i not in cfg.window_block_indexes)
        self.out_blocks = tuple(i - 1 for i in cfg.out_indices)  # hidden state i is the output of block i - 1
        self.num_queries = int(a.num_queries)
        self.points, self.ca_heads = dec.layers[0].cross_attn.n_points, dec.layers[0].cross_attn.n_heads
        self.sa_heads = dec.layers[0].self_attn.num_heads
        wt, wq = _Weights(net, self.dtype), _Weights(net, mx.float32)
        self._load_backbone(wt, mean, std)
        self.cv1, self.cv2 = wq.convx(_PROJ + "0.cv1"), wq.convx(_PROJ + "0.cv2")
        nb = len(a.backbone[0].projector.stages[0][0].m)
        self.bottlenecks = [(wq.convx(f"{_PROJ}0.m.{j}.cv1"), wq.convx(f"{_PROJ}0.m.{j}.cv2")) for j in range(nb)]
        self.proj_norm = wq.norm(_PROJ + "1")
        self._load_queries(wq)
        self.layers = [self._layer(wq, f"{_DEC}layers.{i}.") for i in range(len(dec.layers))]
        self.dec_norm = wq.norm(_DEC + "norm")
        self.bbox, self.cls = wq.mlp("bbox_embed", 3), wq.lin("class_embed")
        self.seg = a.segmentation_head is not None
        if self.seg:
            self._load_seg(wt, len(a.segmentation_head.blocks), a.segmentation_head.downsample_ratio)
        mx.eval(list(_arrays(self.__dict__)))
        mx.clear_cache()  # the build's transients

    def _load_backbone(self, wt: _Weights, mean: Sequence[float], std: Sequence[float]) -> None:
        """The windowed DINOv2 in `mlx_vit`'s layout, the input normalization folded into the patch embedding."""
        p, vit = self.patch, {}
        w, b = (
            wt.get(_ENC + "embeddings.patch_embeddings.projection.weight"),
            wt.get(_ENC + "embeddings.patch_embeddings.projection.bias"),
        )
        m, s = np.asarray(mean).reshape(1, 3, 1, 1), np.asarray(std).reshape(1, 3, 1, 1)
        b, w = b - (w * m / s).sum((1, 2, 3)), w / s
        vit["patch.weight"] = wt.arr(w.transpose(2, 3, 1, 0).reshape(p * p * 3, -1))
        pos = wt.get(_ENC + "embeddings.position_embeddings")[0]  # export: resampled to the resolution
        if pos.shape[0] != self.side**2 + 1:
            msg = f"MlxRFDETR: {pos.shape[0] - 1} position embeddings for a {self.side} x {self.side} patch grid"
            raise ValueError(msg)
        vit["pos.cls"] = wt.arr(pos[:1] + wt.get(_ENC + "embeddings.cls_token")[0])
        vit["pos.grid"] = wt.arr(pos[1:] + b)
        qkv = ("query", "key", "value")
        for i in range(self.depth):
            k, o = f"{_ENC}encoder.layer.{i}.", f"blocks.{i}."
            g1, g2 = wt.get(k + "layer_scale1.lambda1"), wt.get(k + "layer_scale2.lambda1")
            att = k + "attention.attention."
            vit[o + "qkv.weight"] = wt.arr(np.concatenate([wt.get(f"{att}{n}.weight") for n in qkv]).T)
            vit[o + "qkv.bias"] = wt.arr(np.concatenate([wt.get(f"{att}{n}.bias") for n in qkv]))
            vit[o + "proj.weight"] = wt.arr(wt.get(k + "attention.output.dense.weight").T * g1)
            vit[o + "proj.bias"] = wt.arr(wt.get(k + "attention.output.dense.bias") * g1)
            vit[o + "fc1.weight"], vit[o + "fc1.bias"] = wt.lin(k + "mlp.fc1")
            vit[o + "fc2.weight"] = wt.arr(wt.get(k + "mlp.fc2.weight").T * g2)
            vit[o + "fc2.bias"] = wt.arr(wt.get(k + "mlp.fc2.bias") * g2)
            for n in ("norm1", "norm2"):
                vit[o + n + ".weight"], vit[o + n + ".bias"] = wt.norm(k + n)
        self.vit, self.vit_norm = vit, wt.norm(_ENC + "layernorm")

    def _load_queries(self, wt: _Weights) -> None:
        """Two-stage selection (group 0): its heads, the fixed proposals, the learned queries, the sine constants."""
        self.enc_output, self.enc_norm = wt.lin("transformer.enc_output.0"), wt.norm("transformer.enc_output_norm.0")
        self.enc_class = wt.lin("transformer.enc_out_class_embed.0")
        self.enc_bbox = wt.mlp("transformer.enc_out_bbox_embed.0", 3)
        n = self.side
        gy, gx = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32), indexing="ij")
        prop = np.stack([(gx + 0.5) / n, (gy + 0.5) / n, *([np.full_like(gx, 0.05)] * 2)], -1).reshape(n * n, 4)
        valid = ((prop > _VALID[0]) & (prop < _VALID[1])).all(-1, keepdims=True)
        self.valid = wt.arr(valid.astype(np.float32))
        self.proposals = wt.arr(prop * valid, mx.float32)  # invalid ones are zero (`gen_encoder_output_proposals`)
        self.refpoint = wt.arr(wt.get("refpoint_embed.weight")[: self.num_queries], mx.float32)
        self.query_feat = wt.arr(wt.get("query_feat.weight")[: self.num_queries])
        dim_t = 10000 ** (2 * (np.arange(128) // 2) / 128)
        self.sine_scale = wt.arr(2 * math.pi / dim_t, mx.float32)
        self.sine_odd = mx.array(np.arange(128) % 2 == 1)
        self.ref_head = wt.mlp(_DEC + "ref_point_head", 2)

    @staticmethod
    def _layer(wt: _Weights, k: str) -> dict[str, _Wb]:
        """Decoder layer `k`: self-attention's in-projection split into q-k and v, offsets and weights one GEMM."""
        iw, ib = wt.get(k + "self_attn.in_proj_weight"), wt.get(k + "self_attn.in_proj_bias")
        e = iw.shape[1]
        ca = k + "cross_attn."
        ow = np.concatenate([wt.get(ca + "sampling_offsets.weight"), wt.get(ca + "attention_weights.weight")])
        ob = np.concatenate([wt.get(ca + "sampling_offsets.bias"), wt.get(ca + "attention_weights.bias")])
        return {
            "qk": (wt.arr(iw[: 2 * e].T), wt.arr(ib[: 2 * e])),
            "v": (wt.arr(iw[2 * e :].T), wt.arr(ib[2 * e :])),
            "sa_out": wt.lin(k + "self_attn.out_proj"),
            "value": wt.lin(ca + "value_proj"),
            "offw": (wt.arr(ow.T), wt.arr(ob)),
            "ca_out": wt.lin(ca + "output_proj"),
            "fc1": wt.lin(k + "linear1"),
            "fc2": wt.lin(k + "linear2"),
            **{n: wt.norm(k + n) for n in ("norm1", "norm2", "norm3")},
        }

    def _load_seg(self, wt: _Weights, blocks: int, ratio: int) -> None:
        """The segmentation head: depthwise blocks, the spatial and query projections."""
        sh = "segmentation_head."
        self.mask_side = self.resolution // ratio
        self.seg_blocks = [
            (
                wt.arr(wt.get(f"{sh}blocks.{j}.dwconv.weight")[:, 0].transpose(1, 2, 0)),
                wt.arr(wt.get(f"{sh}blocks.{j}.dwconv.bias")),
                wt.norm(f"{sh}blocks.{j}.norm"),
                wt.lin(f"{sh}blocks.{j}.pwconv1"),
            )
            for j in range(blocks)
        ]
        sw = wt.get(sh + "spatial_features_proj.weight")[:, :, 0, 0]
        self.spatial_proj = (wt.arr(sw.T), wt.arr(wt.get(sh + "spatial_features_proj.bias")))
        self.qf_norm = wt.norm(sh + "query_features_block.norm_in")
        self.qf_fc1, self.qf_fc2 = (wt.lin(f"{sh}query_features_block.layers.{j}") for j in (0, 2))
        self.qf_proj = wt.lin(sh + "query_features_proj")
        self.mask_bias = wt.arr(wt.get(sh + "bias"))

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """RGB `[B, 3, H, W]` in [0, 1] (CPU) -> the exported network's outputs, float32 CPU tensors.

        `pred_boxes` `[B, Q, 4]`, `pred_logits` `[B, Q, 91]` and, for Seg, `pred_masks` `[B, Q, R / 4, R / 4]`.
        """
        out = self.raw(to_mx(image.float()))
        mx.eval(out)
        return {k: to_torch(v) for k, v in out.items()}

    def raw(self, image: mx.array) -> dict[str, mx.array]:
        """`__call__` on MLX arrays (lazy)."""
        r = self.resolution
        h, w = image.shape[-2:]
        x = ops.bilinear(image, (r, r), (h / r, w / r), align=False)
        feat = self._projector(self._backbone(x).astype(mx.float32))  # projector on: float32 (module docstring)
        b, c = feat.shape[0], feat.shape[-1]
        memory = feat.reshape(b, -1, c)
        ref = self._two_stage(memory)
        hs = self._decoder(memory, ref)
        delta = _mlp(hs, self.bbox)
        boxes = mx.concatenate(
            [delta[..., :2] * ref[..., 2:] + ref[..., :2], mx.exp(delta[..., 2:]) * ref[..., 2:]], -1
        )
        out = {"pred_boxes": boxes, "pred_logits": _linear(hs, self.cls)}
        if self.seg:
            out["pred_masks"] = self._masks(feat.astype(self.dtype), hs).astype(mx.float32)
        return out

    def _backbone(self, x: mx.array) -> mx.array:
        """Resized RGB `[B, 3, R, R]` -> the four normed hidden states, concatenated `[B, R/p, R/p, 4 * C]`."""
        v, b, n, nw = self.vit, x.shape[0], self.side, self.windows
        s = n // nw
        t = mx.addmm(v["pos.grid"], mlx_vit.patches(x.astype(self.dtype), self.patch), v["patch.weight"])
        c = t.shape[-1]
        t = t.reshape(b, nw, s, nw, s, c).transpose(0, 1, 3, 2, 4, 5).reshape(b * nw * nw, s * s, c)
        t = mx.concatenate([mx.broadcast_to(v["pos.cls"], (b * nw * nw, 1, c)), t], axis=1)
        feats = []
        for i in range(self.depth):
            if i in self.global_blocks:
                t = mlx_vit.block(v, i, t.reshape(b, -1, c), self.vit_heads, _EPS_VIT).reshape(b * nw * nw, -1, c)
            else:
                t = mlx_vit.block(v, i, t, self.vit_heads, _EPS_VIT)
            if i in self.out_blocks:
                f = _ln(t[:, 1:], self.vit_norm, _EPS_VIT)
                feats.append(f.reshape(b, nw, nw, s, s, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, n, n, c))
        return mx.concatenate(feats, axis=-1)

    def _projector(self, x: mx.array) -> mx.array:
        """`[B, n, n, 4 C]` -> the projector's `[B, n, n, 256]` (C2f with 1x1 / 3x3 ConvX: conv, LayerNorm, SiLU)."""

        def convx(x: mx.array, wn: tuple[mx.array, _Wb]) -> mx.array:
            w, ln = wn
            y = mx.conv2d(x, w, padding=1) if w.ndim == 4 else x @ w  # noqa: PLR2004 - OHWI
            return _silu(_ln(y, ln, _EPS_VIT))

        y = convx(x, self.cv1)
        half = y.shape[-1] // 2
        ys = [y[..., :half], y[..., half:]]
        for c1, c2 in self.bottlenecks:
            ys.append(convx(convx(ys[-1], c1), c2))
        return _ln(convx(mx.concatenate(ys, axis=-1), self.cv2), self.proj_norm, _EPS_VIT)

    def _two_stage(self, memory: mx.array) -> mx.array:
        """Memory `[B, HW, D]` -> the decoder's reference boxes `[B, Q, 4]` (centre, size; float32)."""
        pre = _linear(memory * self.valid, self.enc_output)
        scores = mx.max(_linear(_ln(pre, self.enc_norm, _EPS), self.enc_class), axis=-1)
        k = min(self.num_queries, pre.shape[1])
        idx = mx.argsort(-scores, axis=-1)[:, :k]
        tgt = _ln(mx.take_along_axis(pre, idx[..., None], axis=1), self.enc_norm, _EPS)
        delta = _mlp(tgt, self.enc_bbox)
        prop = mx.take(self.proposals, idx, axis=0)
        ts = mx.concatenate(
            [delta[..., :2] * prop[..., 2:] + prop[..., :2], mx.exp(delta[..., 2:]) * prop[..., 2:]], -1
        )
        rp = self.refpoint[:k]
        ref = mx.concatenate([rp[:, :2] * ts[..., 2:] + ts[..., :2], mx.exp(rp[:, 2:]) * ts[..., 2:]], -1)
        if k < self.num_queries:
            ref = mx.concatenate([ref, mx.broadcast_to(self.refpoint[k:], (ref.shape[0], self.num_queries - k, 4))], 1)
        return ref

    def _decoder(self, memory: mx.array, ref: mx.array) -> mx.array:
        """Memory `[B, HW, D]`, reference boxes `[B, Q, 4]` -> the normed last decoder output `[B, Q, D]`."""
        b, q = ref.shape[:2]
        e = mx.concatenate([ref[..., 1::-1], ref[..., 2:]], -1)[..., None] * self.sine_scale  # y, x, w, h
        sine = mx.where(self.sine_odd, mx.cos(e), mx.sin(e)).reshape(b, q, -1)
        qpos = _mlp(sine, self.ref_head)
        nh, pts = self.ca_heads, self.points
        xy = ref[:, :, None, None, None, :2]
        scale = ref[:, :, None, None, None, 2:] * (0.5 / pts)
        shapes = [(self.side, self.side)]
        x = mx.broadcast_to(self.query_feat, (b, q, self.query_feat.shape[-1]))
        d = x.shape[-1]
        for p in self.layers:
            qk = _linear(x + qpos, p["qk"])
            v = _linear(x, p["v"])
            hd = d // self.sa_heads
            att = mx.fast.scaled_dot_product_attention(
                _heads(qk[..., :d], self.sa_heads), _heads(qk[..., d:], self.sa_heads), _heads(v, self.sa_heads),
                scale=hd**-0.5,
            ).transpose(0, 2, 1, 3).reshape(b, q, d)  # fmt: skip
            x = _ln(x + _linear(att, p["sa_out"]), p["norm1"], _EPS)
            ow = _linear(x + qpos, p["offw"])
            no = nh * pts * 2
            off = ow[..., :no].reshape(b, q, nh, 1, pts, 2)
            aw = mx.softmax(ow[..., no:].reshape(b, q, nh, pts), axis=-1).reshape(b, q, nh, 1, pts)
            value = _linear(memory, p["value"]).reshape(b, memory.shape[1], nh, -1)
            ca = ops.msda(value, xy + off * scale, aw, shapes)
            x = _ln(x + _linear(ca, p["ca_out"]), p["norm2"], _EPS)
            x = _ln(x + _linear(mx.maximum(_linear(x, p["fc1"]), 0), p["fc2"]), p["norm3"], _EPS)
        return _ln(x, self.dec_norm, _EPS)

    def _masks(self, feat: mx.array, hs: mx.array) -> mx.array:
        """Projector map `[B, n, n, D]`, decoder output `[B, Q, D]` -> mask logits `[B, Q, R/4, R/4]`."""
        b, n, _, d = feat.shape
        m = self.mask_side
        x = ops.bilinear_nhwc(feat, (m, m), (n / m, n / m), align=False)
        for dw, db, ln, pw in self.seg_blocks:
            y = ops.depthwise(x, dw, db, (1, 1), (1, 1), (1, 1))
            x = x + mlx_vit.gelu(_linear(_ln(y, ln, _EPS_VIT), pw))
        spatial = _linear(x.reshape(b, m * m, d), self.spatial_proj)
        hs = hs.astype(self.dtype)
        qf = hs + _linear(mlx_vit.gelu(_linear(_ln(hs, self.qf_norm, _EPS), self.qf_fc1)), self.qf_fc2)
        qf = _linear(qf, self.qf_proj)
        return (qf @ spatial.transpose(0, 2, 1) + self.mask_bias).reshape(b, -1, m, m)
