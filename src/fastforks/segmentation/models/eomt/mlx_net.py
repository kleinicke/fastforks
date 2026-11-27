"""EoMT on Apple MLX, hand-written from the checkpoint's tensors: the ViT blocks of `fastforks.shared.mlx_vit`.

The maths of `model._Net` (folded the same way, exact in real arithmetic): q / k / v as one GEMM, LayerScale / `s`
folded into the output projection and the second MLP layer, the residual stream kept divided by `res_scale` (`s`;
the embeddings, registers and queries scaled once, each LayerNorm's eps divided by `s^2`). DINOv2: the position
table (plus the patch bias) added to the patch tokens; DINOv3: RoPE on the patch tokens' q / k (float32 tables, as
transformers, cast to the run dtype), identity rows for the prefix tokens (cls, registers, then also the queries).

Pre / post-processing in MLX, channels last: the window batch, the 2 x 2 stride-2 transposed convs of the upscale
blocks as one GEMM each, the mask logits `[B, h / 4 * w / 4, Q]` as one batched GEMM, their bilinear resize to the
window, `sum_q softmax(class_q)[:-1] * sigmoid(mask_q)` as one GEMM, the windows averaged and resized to the image.
"""

# ruff: noqa: ANN401 - meta / float32 torch tensors of the checkpoint

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_seg, mlx_vit
from fastforks.shared.swin_sr import mlx_net as swin

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LAYOUT = "eomt-v1"
SOURCES = (Path(__file__), Path(mlx_vit.__file__), Path(swin.__file__))
PATCH = 16
TABLES = 8  # RoPE tables kept (one per patch grid)


def rope_table(h: int, w: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """float32 `(cos, sin)` `[h * w, head_dim]` of DINOv3's RoPE (eval), as `features.models.dinov3.net.rope_table`."""
    f32 = np.float32
    ch = np.arange(0.5, h, dtype=f32) / f32(h)
    cw = np.arange(0.5, w, dtype=f32) / f32(w)
    coords = (f32(2.0) * np.stack(np.meshgrid(ch, cw, indexing="ij"), -1).reshape(-1, 2) - f32(1.0)).astype(f32)
    inv = (f32(1.0) / f32(theta) ** np.arange(0, 1, 4 / head_dim, dtype=f32)).astype(f32)
    ang = (f32(2 * math.pi) * coords[:, :, None] * inv[None, None, :]).reshape(h * w, -1)
    ang = np.tile(ang, 2)
    return np.cos(ang), np.sin(ang)


def _np(t: Any) -> np.ndarray:
    return t.detach().double().numpy()


class Network:
    """The EoMT network of a checkpoint's tensors (`config`: the variant's `config.json`, `s`: the stream scale)."""

    def __init__(self, sd: Mapping[str, Any], config: Mapping[str, Any], s: float, dtype: str) -> None:
        """The arrays of the state dict `sd` (float32 tensors, or meta ones on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        self.dtype = dt
        c, self.heads = int(config["hidden_size"]), int(config["num_attention_heads"])
        self.depth, self.nq = int(config["num_hidden_layers"]), int(config["num_queries"])
        self.first = self.depth - int(config["num_blocks"])
        self.prefix = 1 + int(config["num_register_tokens"])
        self.eps = float(config["layer_norm_eps"]) / s**2
        self.theta = float(config.get("rope_theta", 100.0))
        self.rope = "embeddings.position_embeddings.weight" not in sd  # DINOv3
        pe = "embeddings.patch_embeddings." + ("" if self.rope else "projection.")
        w = sd[pe + "weight"]
        self.patch_w = swin.derived((PATCH * PATCH * 3, c), dt, lambda: _np(w).transpose(2, 3, 1, 0).reshape(-1, c) / s)
        bias = sd[pe + "bias"]
        if self.rope:
            self.patch_b = swin.arr(bias, dt, scale=1 / s)
        else:  # the position table with the patch bias folded in
            pos = sd["embeddings.position_embeddings.weight"]
            self.patch_b = swin.derived(tuple(pos.shape), dt, lambda: (_np(pos) + _np(bias)) / s)
        cls, reg = sd["embeddings.cls_token"], sd["embeddings.register_tokens"]
        self.tokens = swin.derived((self.prefix, c), dt, lambda: np.concatenate([_np(cls)[0], _np(reg)[0]]) / s)
        self.query = swin.arr(sd["query.weight"], dt, scale=1 / s)
        self.p: dict[str, mx.array] = {}
        for i in range(self.depth):
            self._layer(sd, i, c, s, dt)
        self.norm = (swin.arr(sd["layernorm.weight"], dt), swin.arr(sd["layernorm.bias"], dt))
        self.up: list[tuple[mx.array, mx.array, mx.array, mx.array, mx.array]] = []
        for j in range(int(config["num_upscale_blocks"])):
            u = f"upscale_block.block.{j}."
            w1, b1 = sd[u + "conv1.weight"], sd[u + "conv1.bias"]  # [in, out, 2, 2]
            self.up.append(
                (
                    swin.derived((c, 4 * c), dt, lambda w1=w1: _np(w1).transpose(0, 2, 3, 1).reshape(c, 4 * c)),
                    swin.derived((4 * c,), dt, lambda b1=b1: np.tile(_np(b1), 4)),
                    swin.derived((3, 3, c), dt, lambda w2=sd[u + "conv2.weight"]: _np(w2)[:, 0].transpose(1, 2, 0)),
                    swin.arr(sd[u + "layernorm2d.weight"], dt),
                    swin.arr(sd[u + "layernorm2d.bias"], dt),
                )
            )
        self.zero = mx.zeros((c,), dt)
        self.mask_head = [swin.linear(_Lin(sd, f"mask_head.fc{k}"), dt) for k in (1, 2, 3)]
        self.classes = swin.linear(_Lin(sd, "class_predictor"), dt)
        self.ln2d_eps = 1e-6  # `EomtLayerNorm2d`'s default
        self._tables: dict[tuple[int, int], tuple[mx.array, mx.array, mx.array, mx.array]] = {}

    def _layer(self, sd: Mapping[str, Any], i: int, c: int, s: float, dt: mx.Dtype) -> None:
        src, b, p = f"layers.{i}.", f"blocks.{i}.", self.p
        for n in ("norm1", "norm2"):
            p[f"{b}{n}.weight"], p[f"{b}{n}.bias"] = (
                swin.arr(sd[f"{src}{n}.weight"], dt),
                swin.arr(sd[f"{src}{n}.bias"], dt),
            )
        a = src + "attention."
        ws = [sd[f"{a}{n}_proj.weight"] for n in "qkv"]
        bs = [sd.get(f"{a}{n}_proj.bias") for n in "qkv"]
        p[b + "qkv.weight"] = swin.derived((c, 3 * c), dt, lambda: np.concatenate([_np(w) for w in ws]).T)
        p[b + "qkv.bias"] = swin.derived(
            (3 * c,), dt, lambda: np.concatenate([np.zeros(c) if v is None else _np(v) for v in bs])
        )
        out = a + ("out_proj." if f"{a}out_proj.weight" in sd else "o_proj.")
        m = src + "mlp."
        fc1, fc2 = (m + "fc1.", m + "fc2.") if f"{m}fc1.weight" in sd else (m + "up_proj.", m + "down_proj.")
        p[b + "fc1.weight"], p[b + "fc1.bias"] = swin.linear(_Lin(sd, fc1[:-1]), dt)
        for name, lin, ls in (("proj", out, "layer_scale1"), ("fc2", fc2, "layer_scale2")):
            lw, lb, g = sd[lin + "weight"], sd[lin + "bias"], sd[f"{src}{ls}.lambda1"]
            p[f"{b}{name}.weight"] = swin.derived(
                tuple(lw.shape[::-1]), dt, lambda lw=lw, g=g: _np(lw).T * (_np(g) / s)
            )
            p[f"{b}{name}.bias"] = swin.derived(tuple(lb.shape), dt, lambda lb=lb, g=g: _np(lb) * (_np(g) / s))

    def tables(self, gh: int, gw: int) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]] | None:
        """RoPE `(cos, sin)` with the prefix rows, before and after the queries join (cached per grid); DINOv2: None."""
        if not self.rope:
            return None
        key = (gh, gw)
        if key not in self._tables:
            if len(self._tables) >= TABLES:
                del self._tables[next(iter(self._tables))]
            cos, sin = (mx.array(a).astype(self.dtype) for a in rope_table(gh, gw, self.p_head_dim, self.theta))
            t = (*mlx_vit.rope_tables(cos, sin, self.prefix), *mlx_vit.rope_tables(cos, sin, self.prefix + self.nq))
            mx.eval(*t)
            self._tables[key] = t
        t = self._tables[key]
        return (t[0], t[1]), (t[2], t[3])

    @property
    def p_head_dim(self) -> int:
        """Channels per head."""
        return self.patch_w.shape[1] // self.heads

    def forward(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Normalised windows `[B, S, S, 3]` -> mask logits `[B, S/4, S/4, Q]`, class logits `[B, Q, classes + 1]`.

        Run dtype, evaluated block by block (`mlx_vit.blocks`' `bounded`).
        """
        b, hh, ww, _ = x.shape
        gh, gw = hh // PATCH, ww // PATCH
        if not self.rope and gh * gw != self.patch_b.shape[0]:
            msg = f"EoMT-DINOv2 runs at its training grid ({self.patch_b.shape[0]} patches), got {gh}x{gw}"
            raise ValueError(msg)
        x = x.astype(self.dtype).reshape(b, gh, PATCH, gw, PATCH, 3).transpose(0, 1, 3, 2, 4, 5)
        t = x.reshape(b, gh * gw, -1) @ self.patch_w + self.patch_b
        c = t.shape[-1]
        t = mx.concatenate([mx.broadcast_to(self.tokens, (b, *self.tokens.shape)), t], axis=1)
        ropes = self.tables(gh, gw)
        prev = None
        for i in range(self.depth):
            if i == self.first:
                t = mx.concatenate([mx.broadcast_to(self.query, (b, *self.query.shape)), t], axis=1)
            rope = None if ropes is None else ropes[i >= self.first]
            t = mlx_vit.block(self.p, i, t, self.heads, self.eps, rope)
            mx.async_eval(t)
            if prev is not None:
                mx.eval(prev)
            prev = t
        t = mx.fast.layer_norm(t, *self.norm, self.eps)
        q = t[:, : self.nq]
        y = t[:, self.nq + self.prefix :].reshape(b, gh, gw, c)
        for w1, b1, w2, lw, lb in self.up:  # 2 x 2 stride 2 transposed conv as a GEMM, GELU, depthwise 3 x 3, LN
            n, h, w, _ = y.shape
            y = mx.addmm(b1, y.reshape(-1, c), w1).reshape(n, h, w, 2, 2, c).transpose(0, 1, 3, 2, 4, 5)
            y = mlx_vit.gelu(y.reshape(n, 2 * h, 2 * w, c))
            y = ops.depthwise(y, w2, self.zero, (1, 1), (1, 1), (1, 1))
            y = mx.fast.layer_norm(y, lw, lb, self.ln2d_eps)
        (w1, b1), (w2, b2), (w3, b3) = self.mask_head
        m = swin.dense(mlx_vit.gelu(swin.dense(mlx_vit.gelu(swin.dense(q, w1, b1)), w2, b2)), w3, b3)
        h4, w4 = y.shape[1:3]
        masks = (y.reshape(b, h4 * w4, c) @ m.transpose(0, 2, 1)).reshape(b, h4, w4, self.nq)
        return masks, swin.dense(q, *self.classes)


class _Lin:
    """An `nn.Linear`'s tensors from the state dict (for `swin.linear`)."""

    def __init__(self, sd: Mapping[str, Any], name: str) -> None:
        self.weight, self.bias = sd[name + ".weight"], sd.get(name + ".bias")


def scores(net: Network, x: mx.array, wins: Sequence[tuple[int, int]], size: int) -> mx.array:
    """Normalised image `[1, rh, rw, 3]` -> float32 semantic scores `[1, rh, rw, C]`, its windows averaged.

    The post-processing after `Network.forward` (a function, not a method: the network is the forward pass).
    """
    crops = mx.concatenate([x[:, y : y + size, z : z + size] for y, z in wins]) if len(wins) > 1 else x
    masks, classes = net.forward(crops)
    masks = mlx_seg.resize(masks, (size, size))
    probs = mx.softmax(classes.astype(mx.float32), axis=-1)[..., :-1].astype(masks.dtype)
    n = masks.shape[0]
    lg = (mx.sigmoid(masks).reshape(n, size * size, -1) @ probs).astype(mx.float32).reshape(n, size, size, -1)
    rh, rw = x.shape[1:3]
    if len(wins) == 1 and (rh, rw) == (size, size):
        return lg
    out = mx.zeros((rh, rw, lg.shape[-1]), mx.float32)
    count = np.zeros((rh, rw, 1), np.float32)
    for k, (y, z) in enumerate(wins):
        out[y : y + size, z : z + size] += lg[k]
        count[y : y + size, z : z + size] += 1
    return (out / mx.array(count))[None]


def network(sd: Mapping[str, Any], config: Mapping[str, Any], s: float, dtype: str) -> Network:
    """The MLX network of an EoMT checkpoint's tensors."""
    return Network(sd, config, s, dtype)
