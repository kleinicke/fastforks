"""The SAM 3 image detector on Apple MLX: `sam3.Sam3` in `mx` operations, weights converted from the checkpoint once.

The PyTorch modules are never built: `model.safetensors` (float32) is opened lazily (`fastforks.shared.mlx_weights`)
and the `detector_model.*` tensors are laid out (`read`) and stored in the dtype they run in, layer by layer, in the
conversion cache (`cached`, `fastforks._core.mlx.convert_cached`); every later load reads that file and not the
checkpoint. Memory holds the detector once (fp16: 1.6 GB) plus the activations. The tracker's keys are read only by
`tracker_neck` (its neck, converted and cached alike on the first `video` call); the tracker heads stay PyTorch.

Precision follows the PyTorch portable path (`model.half_parts`): the trunk, neck, text encoder, DETR encoder, mask
decoder and every 256-wide attention block run in the run dtype `dt`; the geometry encoder's and DETR decoder's
residual streams, norms and MLPs, the text projection and the scoring in float32. LayerNorm, GELU and the group-norm
statistics compute in float32 and round to the dtype of their input, as PyTorch's half kernels do.

Layouts: images are NHWC (`mx.conv2d`); the 14 x 14 patch embedding is a GEMM over the patches; each 2 x 2 stride-2
transposed convolution of the neck is a GEMM to `4 x C_out` followed by a pixel shuffle; 1 x 1 convolutions and the 7
x 7 ROI projection (on a 7 x 7 input) are linear layers. Every linear weight is stored `[in, out]`: MLX's GEMM reads a
transposed view of `[out, in]` 2 to 5 % slower (the exported-graph translation stores them so too), and the GEMMs are
nine tenths of the trunk. q / k / v of the trunk and the CLIP layers are one packed projection. The 256-wide attention
has head dim 32, which MLX's fused attention does not take (it would fall back to materialising the scores: 0.8 GB for
the DETR encoder's 5184 tokens), so q / k / v are zero-padded to 64 (the dot products and the output are unchanged).

Peak: a layer loop's graph is bounded per layer (`_step`), and at the 288 x 288 level the neck and the mask decoder
evaluate between the upsampling, the 3 x 3 convolution and the group norm, so one step's whole-map temporaries are
gone before the next step's come (as one graph the masks peaked 230 MB higher; the time is the same).

Only the unpadded prompt path is implemented (one prompt, its real tokens and exemplar boxes): the padding masks of
the PyTorch net exist for the CUDA graphs of kernels mode.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.mlx import convert_cached, mx
from fastforks.shared.mlx_weights import DTYPES, Weights

from .sam3 import DETR_HEADS, DIM, GRID, HEAD, UNUSED, grid_position_encoding, rope_tables

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fastforks._core.runner import Context

F32 = mx.float32
PREFIX = "detector_model."  # the image detector's keys
NECK = "tracker_neck."  # the video tracker's neck (`tracker_neck`)
PATCH, WINDOW, PRETRAIN = 14, 24, 24  # ViT patch, attention window, stored position grid
VIT_LAYERS, VIT_HEADS, GLOBAL_LAYERS = 32, 16, (7, 15, 23, 31)
CLIP_LAYERS, CLIP_HEADS = 24, 16
GEOMETRY_LAYERS, DETR_LAYERS, QUERIES = 3, 6, 200
_HALF = ("vision_encoder.", "text_encoder.", "detr_encoder.", "mask_decoder.")  # run in `dt` (`model.half_parts`)
_ATTN = re.compile(r"(geometry_encoder|detr_decoder)\.layers\.\d+\.(self_attn|cross_attn|text_cross_attn|vision_cross_attn)\.")
_TABLES = re.compile(  # 2-D weights that are rows to look up or tokens, not linear layers
    r"(token_embedding|position_embedding|label_embed|cls_embed|query_embed|reference_points|presence_token)\.weight$"
)
LAYOUT = "sam3-mlx-v2"  # the cached layout of `read` (this file's hash joins it)
_PACKED = re.compile(r"(vision_encoder\.backbone\.layers\.\d+\.attention|text_encoder\.text_model\.encoder\.layers\.\d+\.self_attn)\.")


# --- weights --------------------------------------------------------------------------------------------------------


def _layout(key: str, a: mx.array) -> mx.array:
    """A checkpoint tensor in the layout the `mx` code reads (see the module docstring)."""
    if not key.endswith("weight"):
        return a
    if a.ndim == 4:  # noqa: PLR2004 - convolution weights
        if "scale_layers" in key:  # ConvTranspose2d [C_in, C_out, 2, 2] -> linear [C_in, (a, b, C_out)]
            return mx.contiguous(a.transpose(0, 2, 3, 1).reshape(a.shape[0], -1))
        if a.shape[-1] == 1 or "patch_embeddings" in key:  # 1 x 1 conv / patches (c, i, j) -> linear [O, I]
            a = a.reshape(a.shape[0], -1)
        elif "boxes_pool_project" in key:  # 7 x 7 conv on a 7 x 7 input -> linear over (p, q, c)
            a = a.transpose(0, 2, 3, 1).reshape(a.shape[0], -1)
        else:
            return a.transpose(0, 2, 3, 1)  # 3 x 3 conv: OIHW -> OHWI (`mx.conv2d`)
    if a.ndim == 2 and not _TABLES.search(key):  # noqa: PLR2004 - a linear layer: [out, in] -> [in, out] (`linear`)
        return mx.contiguous(a.T)
    return a


def _bias(key: str, a: mx.array) -> mx.array:
    """Transposed-convolution biases repeat per output pixel of the 2 x 2 block (`_layout`)."""
    return mx.tile(a, 4) if "scale_layers" in key and key.endswith("bias") else a


def _run_dtype(prefix: str, name: str) -> bool:
    """`name` runs in the run dtype (else float32): the `_HALF` parts and 256-wide attention; all of the tracker neck."""
    return prefix != PREFIX or name.startswith(_HALF) or bool(_ATTN.match(name))


def _group(key: str) -> str:
    """Load group of a key: one transformer layer, else the module (three name components)."""
    m = re.match(r"(.*?\.layers\.\d+\.)", key)
    return m.group(1) if m else ".".join(key.split(".")[:3])


def read(path: str | Path, prefix: str, dt: mx.Dtype, skip: Sequence[str] = ()) -> dict[str, mx.array]:
    """The tensors under `prefix` (minus `skip`), prefix removed, laid out and cast group by group.

    `detector_model.*` parts run in `dt` or float32 (`_run_dtype`); `tracker_neck.*` in `dt`. q / k / v of the trunk
    and the CLIP layers are packed into one `qkv` projection. Only one group's float32 source is alive at a time.
    """
    src = Weights(path)
    groups: dict[str, list[str]] = defaultdict(list)
    for k in src.prefixed(prefix):
        if not k.removeprefix(prefix).startswith(tuple(skip)):
            groups[_group(k.removeprefix(prefix))].append(k)
    out: dict[str, mx.array] = {}
    for keys in groups.values():

        def convert(k: str, a: mx.array) -> mx.array:
            name = k.removeprefix(prefix)
            return _bias(name, _layout(name, a)).astype(dt if _run_dtype(prefix, name) else F32)

        part = {k.removeprefix(prefix): v for k, v in src.take(keys, None, convert).items()}
        for name in [n for n in part if _PACKED.match(n) and ".q_proj." in n]:
            stem, _, leaf = name.partition(".q_proj.")
            names = [f"{stem}.{p}_proj.{leaf}" for p in "qkv"]
            part[f"{stem}.qkv.{leaf}"] = mx.concatenate([part.pop(n) for n in names], axis=-1)
        mx.eval(part)
        out |= part
    return out


def cached(ctx: Context, prefix: str = PREFIX, skip: Sequence[str] = UNUSED) -> dict[str, mx.array]:
    """`read` of the variant's checkpoint in the run dtype `ctx.dtype`, from the conversion cache.

    A miss converts group by group (`read`: the arrays arrive in the dtypes they run in, and are stored so) and writes
    the file; a hit reads only that file. The files of `load(weights=...)` are converted on every load.
    """

    def build() -> tuple[dict[str, mx.array], dict[str, object]]:
        return read(ctx.weights()[0], prefix, DTYPES[ctx.dtype], skip), {}

    found = convert_cached(
        ctx,
        f"{LAYOUT}-{prefix.rstrip('.')}",
        build,
        dtype=ctx.dtype,
        fp32=lambda name: not _run_dtype(prefix, name),
        sources=(Path(__file__),),
    )
    return found.arrays


# --- ops ------------------------------------------------------------------------------------------------------------


def _step(prev: tuple[mx.array, ...], *arrays: mx.array, wait: bool = False) -> tuple[mx.array, ...]:
    """Bound a layer loop's graph: start evaluating `arrays` and wait for `prev`, the layer before.

    A blocking `mx.eval` per layer leaves the GPU idle while Python builds the next layer's graph (5 to 10 % of the
    trunk and of a prompt on an M1 Pro); one lazy graph of the whole loop keeps every layer's temporaries alive.
    `wait` also waits for `arrays`: for the layers whose temporaries are large (the trunk's global attention, the neck
    and mask decoder levels), where two in flight raised the peak by 0.3 to 0.6 GB and saved under 1 % of the time.

    Returns:
        The next call's `prev`.
    """
    if wait:
        mx.eval(*prev, *arrays)
        return ()
    mx.async_eval(*arrays)
    if prev:
        mx.eval(*prev)
    return arrays


def linear(x: mx.array, w: mx.array, b: mx.array | None = None) -> mx.array:
    """`F.linear` with the weight stored `[in, out]` (`_layout`; the bias added in the GEMM's epilogue)."""
    if b is None:
        return mx.matmul(x, w)
    return mx.addmm(b, x.reshape(-1, x.shape[-1]), w).reshape(*x.shape[:-1], w.shape[1])


@mx.compile
def _gelu(x: mx.array) -> mx.array:
    """Exact GELU in float32, rounded to the dtype of `x`."""
    xf = x.astype(F32)
    return (0.5 * xf * (1.0 + mx.erf(xf * (1.0 / math.sqrt(2.0))))).astype(x.dtype)


@mx.compile
def _gn_relu(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """`relu(GroupNorm(8, eps 1e-5)(x))` for NHWC `x`: float32 statistics, the affine in the dtype of `x`."""
    n, h, wd, c = x.shape
    g = 8
    xf = x.astype(F32).reshape(n, h * wd, g, c // g)
    m = mx.mean(xf, axis=(1, 3), keepdims=True)
    v = mx.var(xf, axis=(1, 3), keepdims=True)
    scale = w.astype(F32).reshape(1, 1, g, c // g) * mx.rsqrt(v + 1e-5)
    shift = b.astype(F32).reshape(1, 1, g, c // g) - m * scale
    scale, shift = scale.reshape(n, 1, 1, c).astype(x.dtype), shift.reshape(n, 1, 1, c).astype(x.dtype)
    return mx.maximum(x * scale + shift, 0)


@mx.compile
def _rope(t: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """`[B, N, H, 32, 2]` pairs rotated by `[N, 1, 32]` angles: `(x0 c - x1 s, x1 c + x0 s)` -> `[B, N, H, 64]`."""
    x0, x1 = t[..., 0], t[..., 1]
    out = mx.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], axis=-1)
    return out.reshape(*t.shape[:3], -1)


def sdpa(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | str | None = None) -> mx.array:
    """Fused attention `[B, H, N, d]`; head dim 32 zero-padded to 64 (MLX fuses 64, not 32), scale `d ** -0.5`."""
    d = q.shape[-1]
    if d == HEAD:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5, mask=mask)
    pad = [(0, 0)] * 3 + [(0, HEAD - d)]
    o = mx.fast.scaled_dot_product_attention(mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad), scale=d**-0.5, mask=mask)
    return o[..., :d]


def _sine_dims(features: int = DIM // 2, temperature: int = 10000) -> mx.array:
    i = torch.arange(features, dtype=torch.int64).float()
    return mx.array((temperature ** (2 * torch.div(i, 2, rounding_mode="floor") / features)).numpy())


_DIM_T = _sine_dims()


def sine(x: mx.array) -> mx.array:
    """`sam3._sine`: `[...] -> [..., 128]` interleaved sin / cos (float32)."""
    p = x[..., None] / _DIM_T
    return mx.stack([mx.sin(p[..., 0::2]), mx.cos(p[..., 1::2])], axis=-1).reshape(*x.shape, -1)


def encode_boxes(b: mx.array) -> mx.array:
    """`sam3.encode_boxes`: `(cx, cy, w, h)` -> `[..., 512]` (order y, x, w, h)."""
    s = b * (2 * math.pi)
    return mx.concatenate([sine(s[..., 1]), sine(s[..., 0]), sine(s[..., 2]), sine(s[..., 3])], axis=-1)


def cxcywh_to_xyxy(b: mx.array) -> mx.array:
    """Boxes `(cx, cy, w, h)` -> `(x0, y0, x1, y1)`."""
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return mx.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def inverse_sigmoid(x: mx.array, eps: float = 1e-3) -> mx.array:
    """transformers `inverse_sigmoid`."""
    x = mx.clip(x, 0, 1)
    return mx.log(mx.maximum(x, eps) / mx.maximum(1 - x, eps))


def _shuffle(y: mx.array) -> mx.array:
    """`[B, h, w, (2, 2, C)]` -> `[B, 2h, 2w, C]` (the 2 x 2 transposed convolution's output pixels)."""
    b, h, w, c4 = y.shape
    c = c4 // 4
    return y.reshape(b, h, w, 2, 2, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, 2 * h, 2 * w, c)


def _up2(x: mx.array) -> mx.array:
    """Nearest 2x upsampling of NHWC `x`."""
    return mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)


# --- network --------------------------------------------------------------------------------------------------------


class Net:
    """The detector's weights and its forward passes (`sam3.Sam3` without the padding masks).

    Args:
        ctx: The runner's context: its checkpoint, run dtype (float32 or float16) and conversion cache.
    """

    def __init__(self, ctx: Context) -> None:
        """Read the detector's weights (`cached`), the RoPE tables and the position encoding."""
        self.ctx, self.dt = ctx, DTYPES[ctx.dtype]
        self.w = cached(ctx)
        self.ropes = {WINDOW: self._rope(WINDOW, 1.0), 0: self._rope(GRID, WINDOW / GRID)}
        pos = grid_position_encoding(GRID, GRID).flatten(2).transpose(1, 2)
        self.pos = mx.array(pos.contiguous().numpy())  # [1, 72 * 72, 256] float32
        self.neck_w: dict[str, mx.array] | None = None  # the tracker neck (`tracker_neck`)

    def _rope(self, side: int, scale: float) -> tuple[mx.array, mx.array]:
        cos, sin = rope_tables(side, scale)
        c, s = cos[:, 0::2].contiguous().numpy(), sin[:, 0::2].contiguous().numpy()  # each angle is stored twice
        return mx.array(c)[:, None].astype(self.dt), mx.array(s)[:, None].astype(self.dt)

    def ln(self, x: mx.array, name: str, eps: float = 1e-5) -> mx.array:
        """LayerNorm `name` (float32 statistics; the output in the dtype of `x`)."""
        w, b = self.w[f"{name}.weight"], self.w[f"{name}.bias"]
        return mx.fast.layer_norm(x, w.astype(x.dtype), b.astype(x.dtype), eps)

    def lin(self, x: mx.array, name: str) -> mx.array:
        """Linear `name` (in the dtype of its weight)."""
        return linear(x, self.w[f"{name}.weight"], self.w.get(f"{name}.bias"))

    def mlp(self, x: mx.array, name: str, act: Callable[[mx.array], mx.array]) -> mx.array:
        """`fc2(act(fc1(x)))`."""
        return self.lin(act(self.lin(x, f"{name}.fc1")), f"{name}.fc2")

    def decoder_mlp(self, x: mx.array, name: str, layers: int) -> mx.array:
        """`sam3.DecoderMLP`: 2 or 3 linear layers with ReLU between."""
        for i in range(1, layers):
            x = mx.maximum(self.lin(x, f"{name}.layer{i}"), 0)
        return self.lin(x, f"{name}.layer{layers}")

    # --- vision -----------------------------------------------------------------------------------------------------

    def trunk(self, pixels: mx.array) -> mx.array:
        """`[1, 3, 1008, 1008]` normalised (float32) -> `[1, 72, 72, 1024]` (`sam3.ViT`)."""
        w, p = self.w, "vision_encoder.backbone."
        b, _, hh, ww = pixels.shape
        h, wd = hh // PATCH, ww // PATCH
        patches = pixels.astype(self.dt).reshape(b, 3, h, PATCH, wd, PATCH).transpose(0, 2, 4, 1, 3, 5)
        x = linear(patches.reshape(b, h, wd, -1), w[f"{p}embeddings.patch_embeddings.projection.weight"])
        c = x.shape[-1]
        pe = w[f"{p}embeddings.position_embeddings"].reshape(PRETRAIN, PRETRAIN, c)
        pe = mx.tile(pe, (h // PRETRAIN + 1, wd // PRETRAIN + 1, 1))[:h, :wd]
        x = self.ln(x + pe, f"{p}layer_norm", 1e-6)
        s = WINDOW
        k, j = h // s, wd // s

        def part(z: mx.array) -> mx.array:  # [B, H W, C] -> [B k j, s s, C]
            return z.reshape(b, k, s, j, s, c).transpose(0, 1, 3, 2, 4, 5).reshape(b * k * j, s * s, c)

        def unpart(z: mx.array) -> mx.array:  # [B k j, s s, C] -> [B, H W, C]
            return z.reshape(b, k, j, s, s, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, h * wd, c)

        x, windowed = x.reshape(b, h * wd, c), False  # the residual stays in window order between global layers
        prev: tuple[mx.array, ...] = ()
        for i in range(VIT_LAYERS):
            size = 0 if i in GLOBAL_LAYERS else WINDOW
            if bool(size) != windowed:
                x, windowed = (part(x) if size else unpart(x)), bool(size)
            x = self._vit_layer(x, f"{p}layers.{i}.", *self.ropes[size])
            prev = _step(prev, x, wait=not size)  # one layer's activations at a time
        return (unpart(x) if windowed else x).reshape(b, h, wd, c)

    def _vit_layer(self, x: mx.array, p: str, cos: mx.array, sin: mx.array) -> mx.array:
        b, n, c = x.shape
        qkv = self.lin(self.ln(x, f"{p}layer_norm1", 1e-6), f"{p}attention.qkv").reshape(b, n, 3, VIT_HEADS, -1, 2)
        q, k = _rope(qkv[:, :, 0], cos, sin), _rope(qkv[:, :, 1], cos, sin)
        v = qkv[:, :, 2].reshape(b, n, VIT_HEADS, -1)
        o = sdpa(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3))
        x = x + self.lin(o.transpose(0, 2, 1, 3).reshape(b, n, c), f"{p}attention.o_proj")
        return x + self.mlp(self.ln(x, f"{p}layer_norm2", 1e-6), f"{p}mlp", _gelu)

    def neck(self, t: mx.array, w: dict[str, mx.array] | None = None, p: str = "vision_encoder.neck.") -> list[mx.array]:
        """Trunk output `[1, 72, 72, 1024]` -> the 4x, 2x, 1x levels, NHWC (`sam3.Neck`; `w`: the tracker neck's)."""
        w = self.w if w is None else w
        out = []
        prev: tuple[mx.array, ...] = ()
        for level, ups in enumerate(((0, 2), (0,), ())):
            q, x = f"{p}fpn_layers.{level}.", t
            for n, idx in enumerate(ups):
                if n:
                    x = _gelu(x)
                x = _shuffle(linear(x, w[f"{q}scale_layers.{idx}.weight"], w[f"{q}scale_layers.{idx}.bias"]))
            x = linear(x, w[f"{q}proj1.weight"], w[f"{q}proj1.bias"])
            if not level:  # the 288 x 288 level: the upsampling's temporaries go before the convolution's come
                mx.eval(x)
            out.append(mx.conv2d(x, w[f"{q}proj2.weight"], padding=1) + w[f"{q}proj2.bias"])
            prev = _step(prev, out[-1], wait=True)
        return out

    def encode(self, pixels: mx.array) -> list[mx.array]:
        """`[1, 3, 1008, 1008]` normalised -> the neck levels `[1, 288, 288, 256]`, `[.., 144, ..]`, `[.., 72, ..]`."""
        return self.neck(self.trunk(pixels))

    def tracker_neck(self, t: mx.array) -> list[mx.array]:
        """The video tracker's neck (`tracker_neck.*`, read on the first call) on the trunk output."""
        if self.neck_w is None:
            self.neck_w = cached(self.ctx, NECK, ("fpn_layers.3.",))
        return self.neck(t, self.neck_w, "")

    # --- text -------------------------------------------------------------------------------------------------------

    def text(self, ids: Sequence[int]) -> mx.array:
        """Token ids (real tokens only) -> projected text features `[1, T, 256]` float32 (`Sam3.prompt`)."""
        w, p = self.w, "text_encoder.text_model."
        tok = w[f"{p}embeddings.token_embedding.weight"][mx.array(list(ids), dtype=mx.int32)]
        x = (tok + w[f"{p}embeddings.position_embedding.weight"][: len(ids)])[None]
        b, t, c = x.shape
        for i in range(CLIP_LAYERS):
            q = f"{p}encoder.layers.{i}."
            qkv = self.lin(self.ln(x, f"{q}layer_norm1"), f"{q}self_attn.qkv").reshape(b, t, 3, CLIP_HEADS, -1)
            qkv = qkv.transpose(2, 0, 3, 1, 4)
            o = sdpa(qkv[0], qkv[1], qkv[2], mask="causal")
            x = x + self.lin(o.transpose(0, 2, 1, 3).reshape(b, t, c), f"{q}self_attn.out_proj")
            x = x + self.mlp(self.ln(x, f"{q}layer_norm2"), f"{q}mlp", _gelu)
        return self.lin(self.ln(x, f"{p}final_layer_norm").astype(F32), "text_projection")

    # --- detector ---------------------------------------------------------------------------------------------------

    def attn(self, p: str, q: mx.array, k: mx.array, v: mx.array, bias: mx.array | None = None) -> mx.array:
        """`sam3.Attention` `p`: in the dtype of its weights, the output in the dtype of `q`."""
        out, dt = q.dtype, self.w[f"{p}q_proj.weight"].dtype
        b, nq, c = q.shape
        nk, d = k.shape[1], c // DETR_HEADS

        def heads(x: mx.array, name: str, n: int) -> mx.array:
            return self.lin(x.astype(dt), f"{p}{name}").reshape(b, n, DETR_HEADS, d).transpose(0, 2, 1, 3)

        o = sdpa(heads(q, "q_proj", nq), heads(k, "k_proj", nk), heads(v, "v_proj", nk), bias)
        return self.lin(o.transpose(0, 2, 1, 3).reshape(b, nq, c), f"{p}o_proj").astype(out)

    def _relu(self, x: mx.array) -> mx.array:
        return mx.maximum(x, 0)

    def embed_boxes(self, vision: mx.array, ex: Sequence[mx.array]) -> mx.array:
        """`vision [1, h w, C]` float32, `(boxes, labels, pool_y, pool_x)` -> box tokens `[1, N, C]` (float32)."""
        boxes, labels, pool_y, pool_x = ex
        g, n = "geometry_encoder.", boxes.shape[1]
        h, wd = pool_y.shape[-1], pool_x.shape[-1]
        f = self.ln(vision, f"{g}vision_layer_norm").reshape(h, wd * DIM)
        rows = (pool_y[0].reshape(-1, h) @ f).reshape(n, -1, wd, DIM)  # [N, 7 (p), w, C]
        pooled = (pool_x[0][:, None] @ rows).reshape(1, n, -1)  # [N, 7 (p), 7 (q), C] flattened (p, q, c)
        e = self.lin(boxes, f"{g}boxes_direct_project") + self.lin(pooled, f"{g}boxes_pool_project")
        s = 2 * math.pi
        cx, cy, bw, bh = (boxes[..., i] for i in range(4))
        enc = mx.concatenate([sine(cy * s), sine(cx * s), bh[..., None], bw[..., None]], axis=-1)
        return self.w[f"{g}label_embed.weight"][labels] + (e + self.lin(enc, f"{g}boxes_pos_enc_project"))

    def geometry(self, vision: mx.array, ex: Sequence[mx.array] | None) -> mx.array:
        """The prompt tokens `[1, N + 1, C]` (boxes, then CLS) of the geometry encoder (float32 residual)."""
        g = "geometry_encoder."
        p = self.w[f"{g}cls_embed.weight"][None]
        if ex is not None:
            p = mx.concatenate([self.embed_boxes(vision, ex), p], axis=1)
        p = self.ln(self.lin(p, f"{g}final_proj"), f"{g}prompt_layer_norm")
        key = vision + self.pos
        for i in range(GEOMETRY_LAYERS):
            q = f"{g}layers.{i}."
            h = self.ln(p, f"{q}layer_norm1")
            p = p + self.attn(f"{q}self_attn.", h, h, h)
            p = p + self.attn(f"{q}cross_attn.", self.ln(p, f"{q}layer_norm2"), key, vision)
            p = p + self.mlp(self.ln(p, f"{q}layer_norm3"), f"{q}mlp", self._relu)
        return self.ln(p, f"{g}output_layer_norm")

    def detr_encoder(self, vision: mx.array, prompt: mx.array) -> mx.array:
        """Image tokens attend to themselves and the prompt (`sam3.DetrEncoder`, in the run dtype)."""
        dt = self.w["detr_encoder.layers.0.mlp.fc1.weight"].dtype
        x, pos, prompt = vision.astype(dt), self.pos.astype(dt), prompt.astype(dt)
        prev: tuple[mx.array, ...] = ()
        for i in range(DETR_LAYERS):
            q = f"detr_encoder.layers.{i}."
            h = self.ln(x, f"{q}layer_norm1")
            hp = h + pos
            x = x + self.attn(f"{q}self_attn.", hp, hp, h)
            x = x + self.attn(f"{q}cross_attn.", self.ln(x, f"{q}layer_norm2"), prompt, prompt)
            x = x + self.mlp(self.ln(x, f"{q}layer_norm3"), f"{q}mlp", self._relu)
            prev = _step(prev, x)  # one layer's activations at a time (a longer graph keeps every layer's)
        return x

    def rpb(self, ref: mx.array) -> mx.array:
        """Box relative-position bias `[1, heads, 1 + Q, h w]` (zero row for the presence token; log mode)."""
        d = "detr_decoder."
        b, q, _ = ref.shape
        xyxy = cxcywh_to_xyxy(ref)
        grid = mx.arange(GRID, dtype=F32) / GRID
        dy = grid.reshape(1, 1, -1, 1) - xyxy[..., None, 1::2]  # [B, Q, h, 2]
        dx = grid.reshape(1, 1, -1, 1) - xyxy[..., None, 0::2]

        def log(t: mx.array) -> mx.array:
            t = t * 8
            return mx.sign(t) * mx.log2(mx.abs(t) + 1.0) / math.log2(8)

        ey = self.decoder_mlp(log(dy), f"{d}box_rpb_embed_y", 2).transpose(0, 3, 1, 2)  # [B, heads, Q, h]
        ex = self.decoder_mlp(log(dx), f"{d}box_rpb_embed_x", 2).transpose(0, 3, 1, 2)
        m = (ey[..., None] + ex[..., None, :]).reshape(b, DETR_HEADS, q, -1)
        return mx.concatenate([mx.zeros((b, DETR_HEADS, 1, m.shape[-1]), dtype=m.dtype), m], axis=2)

    def detr_decoder(self, encoded: mx.array, prompt: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Query states `[1, Q, C]`, boxes `[1, Q, 4]` (cxcywh), presence logit `[1, 1]` (`sam3.DetrDecoder`)."""
        d, w = "detr_decoder.", self.w
        ref = mx.sigmoid(w[f"{d}reference_points.weight"])[None]
        h = mx.concatenate([w[f"{d}presence_token.weight"], w[f"{d}query_embed.weight"]])[None]
        key = encoded + self.pos
        dt = w[f"{d}layers.0.vision_cross_attn.q_proj.weight"].dtype
        key_dt, vision_dt = key.astype(dt), encoded.astype(dt)  # cast once, as every layer's attention would
        out = ref
        prev: tuple[mx.array, ...] = ()
        for i in range(DETR_LAYERS):
            q = f"{d}layers.{i}."
            qpos = self.decoder_mlp(encode_boxes(ref), f"{d}ref_point_head", 2)
            qpos = mx.concatenate([mx.zeros((1, 1, DIM), dtype=qpos.dtype), qpos], axis=1)
            bias = self.rpb(ref)
            x = h + qpos
            h = self.ln(h + self.attn(f"{q}self_attn.", x, x, h), f"{q}self_attn_layer_norm")
            h = self.ln(h + self.attn(f"{q}text_cross_attn.", h + qpos, prompt, prompt), f"{q}text_cross_attn_layer_norm")
            o = self.attn(f"{q}vision_cross_attn.", h + qpos, key_dt, vision_dt, bias.astype(dt))
            h = self.ln(h + o, f"{q}vision_cross_attn_layer_norm")
            h = self.ln(h + self.mlp(h, f"{q}mlp", self._relu), f"{q}mlp_layer_norm")
            out = self.ln(h[:, 1:], f"{d}output_layer_norm")
            ref = mx.sigmoid(self.decoder_mlp(out, f"{d}box_head", 3) + inverse_sigmoid(ref))
            prev = _step(prev, h, out, ref)
        presence = self.decoder_mlp(self.ln(h[:, :1], f"{d}presence_layer_norm"), f"{d}presence_head", 3)
        return out, ref, mx.clip(presence[..., 0], -10.0, 10.0)

    def scores(self, queries: mx.array, prompt: mx.array) -> mx.array:
        """Query logits `[1, Q]` from the mean-pooled prompt (`sam3.DotProductScoring`, float32)."""
        s = "dot_product_scoring."
        t = self.ln(self.decoder_mlp(prompt, f"{s}text_mlp", 2) + prompt, f"{s}text_mlp_out_norm")
        pt = self.lin(t.mean(axis=1), f"{s}text_proj")
        logits = (self.lin(queries, f"{s}query_proj") @ pt[:, :, None])[..., 0] / math.sqrt(DIM)
        return mx.clip(logits, -12.0, 12.0)

    def masks(self, queries: mx.array, fpn: Sequence[mx.array], encoded: mx.array, prompt: mx.array) -> mx.array:
        """Mask logits `[1, Q, 288, 288]` (`sam3.MaskDecoder`, in the run dtype; `fpn` NHWC fine to coarse)."""
        m = "mask_decoder."
        dt = self.w[f"{m}instance_projection.weight"].dtype
        enc, prompt = encoded.astype(dt), prompt.astype(dt)
        e = enc + self.attn(f"{m}prompt_cross_attn.", self.ln(enc, f"{m}prompt_cross_attn_norm"), prompt, prompt)
        x = e.reshape(fpn[-1].shape)
        prev: tuple[mx.array, ...] = ()
        for i, f in enumerate(reversed(fpn[:-1])):
            q = f"{m}pixel_decoder."
            x = _up2(x) + f.astype(dt)
            x = mx.conv2d(x, self.w[f"{q}conv_layers.{i}.weight"], padding=1) + self.w[f"{q}conv_layers.{i}.bias"]
            if i:  # 288 x 288: the convolution's temporaries go before the norm's float32 statistics come
                mx.eval(x)
            x = _gn_relu(x, self.w[f"{q}norms.{i}.weight"], self.w[f"{q}norms.{i}.bias"])
            prev = _step(prev, x, wait=True)
        inst = self.lin(x, f"{m}instance_projection")  # [1, 288, 288, C]
        _step(prev, inst)
        emb = queries.astype(dt)
        for i in range(3):
            emb = self.lin(emb, f"{m}mask_embedder.layers.{i}")
            emb = emb if i == 2 else mx.maximum(emb, 0)  # noqa: PLR2004 - the last layer
        b, h, wd, c = inst.shape
        return (emb @ inst.reshape(b, h * wd, c).transpose(0, 2, 1)).reshape(b, -1, h, wd)

    def detect(
        self, fpn: Sequence[mx.array], ids: Sequence[int], ex: Sequence[mx.array] | None
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Scores `[Q]` (sigmoid(logit) * sigmoid(presence)), boxes `[Q, 4]` (normalised xyxy), mask logits
        `[Q, 288, 288]`, all float32 (`Sam3.detect` for one prompt without padding).

        Args:
            fpn: The neck levels fine to coarse, NHWC.
            ids: The prompt's token ids (real tokens only).
            ex: Exemplar boxes `[1, N, 4]` cxcywh, labels `[1, N]`, ROI weights `[1, N, 7, 72]` x 2; or `None`.
        """
        text = self.text(ids)
        vision = fpn[-1].reshape(1, -1, DIM).astype(F32)
        prompt = mx.concatenate([text, self.geometry(vision, ex)], axis=1)
        encoded = self.detr_encoder(vision, prompt).astype(F32)
        _step((), encoded, prompt)
        queries, boxes, presence = self.detr_decoder(encoded, prompt)
        logits = self.scores(queries, prompt)
        masks = self.masks(queries, fpn, encoded, prompt).astype(F32)
        scores = mx.sigmoid(logits) * mx.sigmoid(presence)
        return scores[0], cxcywh_to_xyxy(boxes)[0], masks[0]


def to_mx(t: torch.Tensor) -> mx.array:
    """A CPU tensor as an `mx.array` (int64 -> int32)."""
    t = t.detach().contiguous()
    if t.dtype == torch.int64:
        t = t.to(torch.int32)
    return mx.array(t.numpy())


def nhwc_to_torch(a: mx.array) -> torch.Tensor:
    """An NHWC `mx.array` (evaluated here) as an NCHW CPU tensor of its dtype (channels-last memory)."""
    mx.eval(a)
    return torch.from_numpy(np.array(a, copy=False)).permute(0, 3, 1, 2)


def to_torch(a: mx.array) -> torch.Tensor:
    """An `mx.array` (evaluated here) as a CPU tensor of its dtype."""
    mx.eval(a)
    return torch.from_numpy(np.array(a))
