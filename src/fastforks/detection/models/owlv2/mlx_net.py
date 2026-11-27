"""OWLv2 on Apple MLX, hand-written from the checkpoint's tensors (no transformers network): the run dtype.

- Pixels (`pixels`): the image padded to a square (bottom / right, zeros), the processor's Gaussian blur as the
  weighted sum of shifted slices of the reflect-padded map (rows, then columns), the bilinear resize of
  `mlx_seg.resize`, CLIP mean / std; float32, NHWC.
- Image tower (`image`): the patch convolution as one GEMM over the patches, the class token and the position table,
  the pre-norm, the CLIP blocks (`mlx_vit.blocks`, QuickGELU, LayerNorm eps 1e-5) in the run dtype; the post-norm,
  the merged class-token LayerNorm and the heads in float32 (as the PyTorch runner keeps them): class embeddings
  normalised, logit shift, `elu(scale) + 1`, `sigmoid(box_head + box_bias)`.
- Text tower (`text`): token and position embeddings, causal pre-norm blocks in the run dtype, the final norm, the
  end-of-text token's features (the highest id) through the projection, normalised in float32.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.shared import mlx_seg, mlx_vit
from fastforks.shared.swin_sr import mlx_net as swin

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LAYOUT = "owlv2-v1"
SOURCES = (Path(__file__), Path(mlx_vit.__file__), Path(swin.__file__))
MEAN = (0.48145466, 0.4578275, 0.40821073)  # fixed: OpenAI CLIP normalisation, as `model.MEAN`
STD = (0.26862954, 0.26130258, 0.27577711)  # fixed: as `model.STD`
EPS = 1e-5  # fixed: the CLIP towers' LayerNorm eps (transformers' default)
QUICK_GELU = 1.702  # fixed: QuickGELU(x) = x * sigmoid(1.702 x)


def _quick_gelu(x: mx.array) -> mx.array:
    y = x.astype(mx.float32)
    return (y * mx.sigmoid(QUICK_GELU * y)).astype(x.dtype)


quick_gelu = mx.compile(_quick_gelu, shapeless=True)  # one kernel, float32 inside


def box_bias(gh: int, gw: int) -> np.ndarray:
    """`Owlv2ForObjectDetection.compute_box_bias` in numpy (float32): `[gh * gw, 4]`, rows first."""
    xx, yy = np.meshgrid(np.arange(1, gw + 1, dtype=np.float32) / gw, np.arange(1, gh + 1, dtype=np.float32) / gh)
    xy = np.clip(np.stack((xx, yy), -1).reshape(-1, 2), 0.0, 1.0)
    coord = np.log(xy + np.float32(1e-4)) - np.log1p(-xy + np.float32(1e-4))
    size = np.broadcast_to(np.array([1 / gw, 1 / gh], np.float32), xy.shape)
    size_bias = np.log(size + np.float32(1e-4)) - np.log1p(-size + np.float32(1e-4))
    return np.concatenate((coord, size_bias), -1).astype(np.float32)


def gaussian_taps(factor: float) -> np.ndarray:
    """The processor's blur kernel for a downscale by `factor` (torchvision `gaussian_blur`), float32; empty if none."""
    sigma = max((factor - 1) / 2, 0.0)
    if sigma <= 0:
        return np.zeros(0, np.float32)
    k = 2 * math.ceil(3 * sigma) + 1
    t = np.linspace(-(k // 2), k // 2, k, dtype=np.float32)
    g = np.exp(np.float32(-0.5) * (t / np.float32(sigma)) ** 2)
    return (g / g.sum()).astype(np.float32)


def _reflect(x: mx.array, r: int, axis: int) -> mx.array:
    """Reflect padding by `r` (the edge not repeated) along `axis` of an NHWC map."""
    n = x.shape[axis]
    lo = mx.take(x, mx.array(list(range(r, 0, -1))), axis=axis)
    hi = mx.take(x, mx.array(list(range(n - 2, n - 2 - r, -1))), axis=axis)
    return mx.concatenate([lo, x, hi], axis=axis)


def blur(x: mx.array, g: Sequence[float]) -> mx.array:
    """Separable blur of NHWC `x` with the taps `g` (reflect border): rows, then columns, float32."""
    k = len(g)
    r = k // 2
    for axis in (1, 2):
        n = x.shape[axis]
        y = _reflect(x, r, axis)
        out = None
        for i in range(k):
            s = [slice(None)] * 4
            s[axis] = slice(i, i + n)
            t = y[tuple(s)] * g[i]
            out = t if out is None else out + t
        x = out if out is not None else x
    return x


class _Linear(SimpleNamespace):
    weight: Any
    bias: Any


def _lin(sd: Mapping[str, Any], name: str) -> _Linear:
    return _Linear(weight=sd[name + ".weight"], bias=sd.get(name + ".bias"))


def _tower(sd: Mapping[str, Any], prefix: str, depth: int, dt: mx.Dtype) -> dict[str, mx.array]:
    """The encoder layers under `prefix` as `mlx_vit.block`'s parameters (q / k / v merged, weights `[in, out]`)."""
    p: dict[str, mx.array] = {}
    for i in range(depth):
        s, b = f"{prefix}.encoder.layers.{i}.", f"blocks.{i}."
        for src, dst in (("layer_norm1", "norm1"), ("layer_norm2", "norm2")):
            p[b + dst + ".weight"], p[b + dst + ".bias"] = (swin.arr(sd[s + src + n], dt) for n in (".weight", ".bias"))
        a = s + "self_attn."
        ws = [sd[f"{a}{n}_proj.weight"] for n in "qkv"]
        bs = [sd[f"{a}{n}_proj.bias"] for n in "qkv"]
        c = ws[0].shape[1]
        p[b + "qkv.weight"] = swin.derived(
            (c, 3 * c), dt, lambda ws=ws: np.concatenate([w.detach().double().numpy() for w in ws]).T
        )
        p[b + "qkv.bias"] = swin.derived((3 * c,), dt, lambda bs=bs: np.concatenate([v.double().numpy() for v in bs]))
        for src, dst in (("self_attn.out_proj", "proj"), ("mlp.fc1", "fc1"), ("mlp.fc2", "fc2")):
            p[b + dst + ".weight"], p[b + dst + ".bias"] = swin.linear(_lin(sd, s + src), dt)
    return p


def _text_block(p: dict[str, mx.array], i: int, x: mx.array, heads: int) -> mx.array:
    """`mlx_vit.block` with the causal mask (CLIP's text encoder layer)."""
    b = f"blocks.{i}."
    bsz, n, c = x.shape
    h = mx.fast.layer_norm(x, p[b + "norm1.weight"], p[b + "norm1.bias"], EPS)
    qkv = mx.addmm(p[b + "qkv.bias"], h, p[b + "qkv.weight"]).reshape(bsz, n, 3, heads, -1).transpose(2, 0, 3, 1, 4)
    d = c // heads
    o = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=d**-0.5, mask="causal")
    x = x + mx.addmm(p[b + "proj.bias"], o.transpose(0, 2, 1, 3).reshape(bsz, n, c), p[b + "proj.weight"])
    h = mx.fast.layer_norm(x, p[b + "norm2.weight"], p[b + "norm2.bias"], EPS)
    h = quick_gelu(mx.addmm(p[b + "fc1.bias"], h, p[b + "fc1.weight"]))
    return x + mx.addmm(p[b + "fc2.bias"], h, p[b + "fc2.weight"])


class Network:
    """The OWLv2 towers and heads of a checkpoint's tensors (`config`: the variant's `config.json`)."""

    def __init__(self, sd: Mapping[str, Any], config: Mapping[str, Any], dtype: str) -> None:
        """The arrays of the state dict `sd` (float32 tensors, or meta ones on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        f32 = mx.float32
        self.dtype = dt
        vc, tc = config.get("vision_config", {}), config.get("text_config", {})
        self.size, self.patch = int(vc.get("image_size", 768)), int(vc.get("patch_size", 16))
        self.depth, self.heads = int(vc.get("num_hidden_layers", 12)), int(vc.get("num_attention_heads", 12))
        self.t_depth, self.t_heads = int(tc.get("num_hidden_layers", 12)), int(tc.get("num_attention_heads", 8))
        self.grid = self.size // self.patch
        v = "owlv2.vision_model"
        w = sd[f"{v}.embeddings.patch_embedding.weight"]  # [C, 3, p, p]
        c = int(w.shape[0])
        self.patch_w = swin.derived(
            (3 * self.patch * self.patch, c),
            dt,
            lambda: w.detach().double().numpy().transpose(2, 3, 1, 0).reshape(-1, c),
        )
        self.cls = swin.arr(sd[f"{v}.embeddings.class_embedding"], dt)
        self.pos = swin.arr(sd[f"{v}.embeddings.position_embedding.weight"], dt)
        self.pre = tuple(swin.arr(sd[f"{v}.pre_layernorm.{n}"], dt) for n in ("weight", "bias"))
        self.p = _tower(sd, v, self.depth, dt)
        self.post = tuple(swin.arr(sd[f"{v}.post_layernorm.{n}"], f32) for n in ("weight", "bias"))
        self.merge = tuple(swin.arr(sd[f"layer_norm.{n}"], f32) for n in ("weight", "bias"))
        self.cls_dense = swin.linear(_lin(sd, "class_head.dense0"), f32)
        self.shift = swin.linear(_lin(sd, "class_head.logit_shift"), f32)
        self.scale = swin.linear(_lin(sd, "class_head.logit_scale"), f32)
        self.box = [swin.linear(_lin(sd, f"box_head.dense{i}"), f32) for i in range(3)]
        self.box_bias = mx.array(box_bias(self.grid, self.grid))
        t = "owlv2.text_model"
        self.tok = swin.arr(sd[f"{t}.embeddings.token_embedding.weight"], dt)
        self.t_pos = swin.arr(sd[f"{t}.embeddings.position_embedding.weight"], dt)
        self.tp = _tower(sd, t, self.t_depth, dt)
        self.t_final = tuple(swin.arr(sd[f"{t}.final_layer_norm.{n}"], dt) for n in ("weight", "bias"))
        self.t_proj = swin.arr(sd["owlv2.text_projection.weight"].t(), dt)
        self.mean = mx.array(MEAN, f32)
        self.std = mx.array(STD, f32)
        self._taps: dict[int, list[float]] = {}

    def pixels(self, image: mx.array) -> mx.array:
        """`[1, 3, H, W]` float32 in [0, 1] -> the normalised network input `[1, S, S, 3]` float32 (NHWC)."""
        _, _, h, w = image.shape
        s = max(h, w)
        x = mx.pad(image.transpose(0, 2, 3, 1), [(0, 0), (0, s - h), (0, s - w), (0, 0)])
        if s != self.size:
            if s not in self._taps:
                self._taps[s] = [float(v) for v in gaussian_taps(s / self.size)]
            if self._taps[s]:
                x = blur(x, self._taps[s])
            x = mlx_seg.resize(x, (self.size, self.size))
        return (x - self.mean) / self.std

    def image(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Normalised NHWC input `[B, S, S, 3]` float32 -> (class embeddings, shift, scale, boxes), float32, evaluated.

        Class embeddings `[B, P, D]` normalised, shift and scale `[B, P, 1]`, boxes `[B, P, 4]` (centre / size, 0..1).
        """
        b, g, ps = x.shape[0], self.grid, self.patch
        y = x.astype(self.dtype).reshape(b, g, ps, g, ps, 3).transpose(0, 1, 3, 2, 4, 5).reshape(b, g * g, -1)
        t = mx.concatenate([mx.broadcast_to(self.cls, (b, 1, self.cls.shape[0])), y @ self.patch_w], axis=1)
        t = mx.fast.layer_norm(t + self.pos, *self.pre, EPS)
        t, _ = mlx_vit.blocks(self.p, self.depth, self.heads, EPS, t, (), act=quick_gelu, bounded=True)
        t = mx.fast.layer_norm(t.astype(mx.float32), *self.post, EPS)
        f = mx.fast.layer_norm(t[:, 1:] * t[:, :1], *self.merge, EPS)
        emb = swin.dense(f, *self.cls_dense)
        emb = emb / (mx.linalg.norm(emb, axis=-1, keepdims=True) + 1e-6)
        z = f
        for i, (wt, bias) in enumerate(self.box):
            z = swin.dense(z, wt, bias)
            if i < len(self.box) - 1:
                z = mlx_vit.gelu(z)
        scale = swin.dense(f, *self.scale)
        out = (
            emb,
            swin.dense(f, *self.shift),
            mx.where(scale > 0, scale, mx.expm1(scale)) + 1,
            mx.sigmoid(z + self.box_bias),
        )
        mx.eval(out)
        return out

    def text(self, ids: np.ndarray) -> mx.array:
        """Token ids `[Q, L]` (int, zero-padded) -> normalised text embeddings `[Q, D]` float32, evaluated."""
        q, n = ids.shape
        t = mx.take(self.tok, mx.array(ids.reshape(-1).astype(np.int32)), axis=0).reshape(q, n, -1) + self.t_pos[:n]
        for i in range(self.t_depth):
            t = _text_block(self.tp, i, t, self.t_heads)
        t = mx.fast.layer_norm(t, *self.t_final, EPS)
        eot = mx.array(ids.argmax(-1).astype(np.int32))
        pooled = t[mx.arange(q), eot]
        e = (pooled @ self.t_proj).astype(mx.float32)
        e = e / (mx.linalg.norm(e, axis=-1, keepdims=True) + 1e-6)
        mx.eval(e)
        return e


def network(sd: Any, config: Mapping[str, Any], dtype: str) -> Network:  # noqa: ANN401 - torch tensors
    """The MLX OWLv2 of a state dict (float32 or meta tensors) in `dtype`."""
    return Network(sd, config, dtype)
