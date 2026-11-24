"""ViT building blocks on Apple MLX (`device="mlx"`), shared by the MLX ports of the ViT families.

Pure functions of arrays and a flat parameter dict in one layout, which every port's conversion produces:

    blocks.<i>.norm1.weight / .bias, blocks.<i>.norm2.weight / .bias
    blocks.<i>.qkv.weight [C, 3 C] / .bias      q, k, v projections packed (linear weights are stored `[in, out]`)
    blocks.<i>.proj.weight / .bias              attention output projection, LayerScale folded in
    blocks.<i>.fc1.* , blocks.<i>.fc2.*         GELU MLP (LayerScale folded into fc2), or
    blocks.<i>.w12.* , blocks.<i>.w3.*          SwiGLU: `silu(first half of w12) * second half`, then w3

Attention is MLX's fused `scaled_dot_product_attention` (the softmax accumulates in float32 there); LayerNorm is
`mx.fast.layer_norm` (float32 statistics inside its kernel). So float16 weights and activations keep float32 where the
torch fp16 runners have it. RoPE (`rope_tables`, DINOv3) rotates q and k of every token with tables whose prefix rows
are cos 1 / sin 0, which leaves cls and register tokens as they are without slicing the token axis.
"""
# ruff: noqa: ANN001, ANN201, PLR0913, PLR0917 - array-in / array-out maths functions (positional
# arrays, as the torch forwards they mirror); shapes are in the docstrings

from __future__ import annotations

import math
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from fastforks._core.mlx import mx, to_mx, to_torch, tuning

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DinoFeatureEngine",
    "DinoLinearProbe",
    "attention",
    "block",
    "blocks",
    "depth_of",
    "dinov2_layers",
    "dinov2_params",
    "dinov2_table",
    "gelu",
    "patches",
    "rope_tables",
    "rotate",
    "transformer_params",
]


def depth_of(p: dict[str, mx.array]) -> int:
    """Number of blocks in a parameter dict of this layout."""
    return 1 + max(int(k.split(".")[1]) for k in p if k.startswith("blocks."))


def patches(x: mx.array, patch: int) -> mx.array:
    """`[B, 3, H, W]` -> `[B, N, patch * patch * 3]` (rows, columns, channels within a patch).

    With the patch convolution's weight as `[patch * patch * 3, C]`, the embedding is one matmul.
    """
    b, c, h, w = x.shape
    h0, w0 = h // patch, w // patch
    x = x.reshape(b, c, h0, patch, w0, patch).transpose(0, 2, 4, 3, 5, 1)
    return x.reshape(b, h0 * w0, patch * patch * c)


def _gelu(x: mx.array) -> mx.array:
    y = x.astype(mx.float32)
    return (y * (0.5 + 0.5 * mx.erf(y * 0.7071067811865476))).astype(x.dtype)


gelu = mx.compile(_gelu, shapeless=True)
"""Exact (erf) GELU, as torch's default: one fused kernel that computes in float32 (as `mlx_graph`'s; five passes
uncompiled, MoGe ViT-S 994x1482 float16 282 -> 277 ms; `1 + erf` of a negative input cancels in float16)."""


def rope_tables(cos: mx.array, sin: mx.array, prefix: int) -> tuple[mx.array, mx.array]:
    """`(cos, sin)` `[N, D]` of the patch tokens -> `[prefix + N, D]` with identity rows for the prefix tokens."""
    d = cos.shape[1]
    return (
        mx.concatenate([mx.ones((prefix, d), cos.dtype), cos]),
        mx.concatenate([mx.zeros((prefix, d), sin.dtype), sin]),
    )


def rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """RoPE `x * cos + rotate_half(x) * sin` over the last axis (`rotate_half([a, b]) = [-b, a]`)."""
    a, b = mx.split(x, 2, axis=-1)
    return x * cos + mx.concatenate([-b, a], axis=-1) * sin


def attention(qkv: mx.array, heads: int, rope=None) -> mx.array:
    """Packed `[B, N, 3 C]` -> `[B, N, C]`; `rope`: `(cos, sin)` `[N, C / heads]` applied to q and k."""
    b, n, c3 = qkv.shape
    d = c3 // (3 * heads)
    qkv = qkv.reshape(b, n, 3, heads, d).transpose(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    if rope is not None:
        q, k = rotate(q, *rope), rotate(k, *rope)
    out = mx.fast.scaled_dot_product_attention(q, k, qkv[2], scale=d**-0.5)
    return out.transpose(0, 2, 1, 3).reshape(b, n, heads * d)


def block(p: dict[str, mx.array], i: int, x: mx.array, heads: int, eps: float, rope=None) -> mx.array:
    """Pre-norm block `i` on tokens `[B, N, C]`."""
    b = f"blocks.{i}."
    h = mx.fast.layer_norm(x, p[b + "norm1.weight"], p[b + "norm1.bias"], eps)
    h = attention(mx.addmm(p[b + "qkv.bias"], h, p[b + "qkv.weight"]), heads, rope)
    x = x + mx.addmm(p[b + "proj.bias"], h, p[b + "proj.weight"])
    h = mx.fast.layer_norm(x, p[b + "norm2.weight"], p[b + "norm2.bias"], eps)
    if b + "w12.weight" in p:
        x1, x2 = mx.split(mx.addmm(p[b + "w12.bias"], h, p[b + "w12.weight"]), 2, axis=-1)
        return x + mx.addmm(p[b + "w3.bias"], x1 * mx.sigmoid(x1) * x2, p[b + "w3.weight"])
    h = gelu(mx.addmm(p[b + "fc1.bias"], h, p[b + "fc1.weight"]))
    return x + mx.addmm(p[b + "fc2.bias"], h, p[b + "fc2.weight"])


def blocks(p: dict[str, mx.array], depth: int, heads: int, eps: float, t: mx.array, layers, rope=None):
    """All blocks on the embedded tokens `t`: `(tokens after the last block, [tokens at layers])`.

    Layer 0 = `t`, layer `i` = the output of block `i` (1-based, before the final norm), in ascending order.
    """
    hs = [t] if 0 in layers else []
    for i in range(depth):
        t = block(p, i, t, heads, eps, rope)
        if i + 1 in layers:
            hs.append(t)
    return t, hs


def transformer_params(state: dict[str, torch.Tensor]) -> dict[str, mx.array]:
    """Convert timm or DINO transformer blocks, folding LayerScale where the checkpoint has it."""
    out: dict[str, mx.array] = {}

    def take(k: str) -> np.ndarray:
        return state.pop(k).double().numpy()

    def put(k: str, v: np.ndarray) -> None:
        out[k] = mx.array(np.ascontiguousarray(v, dtype=np.float32))

    depth = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
    for i in range(depth):
        b = f"blocks.{i}."
        for n in ("norm1", "norm2"):
            put(f"{b}{n}.weight", take(f"{b}{n}.weight"))
            put(f"{b}{n}.bias", take(f"{b}{n}.bias"))
        g1, g2 = (take(f"{b}ls1.gamma"), take(f"{b}ls2.gamma")) if f"{b}ls1.gamma" in state else (1.0, 1.0)
        put(f"{b}qkv.weight", take(f"{b}attn.qkv.weight").T)
        put(f"{b}qkv.bias", take(f"{b}attn.qkv.bias"))
        put(f"{b}proj.weight", take(f"{b}attn.proj.weight").T * g1)
        put(f"{b}proj.bias", take(f"{b}attn.proj.bias") * g1)
        first, last = ("w12", "w3") if f"{b}mlp.w12.weight" in state else ("fc1", "fc2")  # ViT-g: fused SwiGLU
        put(f"{b}{first}.weight", take(f"{b}mlp.{first}.weight").T)
        put(f"{b}{first}.bias", take(f"{b}mlp.{first}.bias"))
        put(f"{b}{last}.weight", take(f"{b}mlp.{last}.weight").T * g2)
        put(f"{b}{last}.bias", take(f"{b}mlp.{last}.bias") * g2)
    return out


def dinov2_params(state: dict[str, torch.Tensor], patch: int) -> dict[str, mx.array]:
    """DINOv2 checkpoint tensors (hub names) -> the float32 arrays of this layout, plus the embedding's.

    Exact in real arithmetic (float64, stored float32): LayerScale folded, the patch convolution as
    `patch.weight` `[patch * patch * 3, C]` / `patch.bias`, `pos.cls` (cls token + its position) and `pos.grid`
    (the patch positions), `registers` if the checkpoint has them, `norm.*`. Takes its keys out of `state` as it
    goes: ViT-g is 4.4 GB, and two float32 copies plus float64 temporaries would not fit.

    Layout changes require new cache versions in the DINOv2, TIPSv2 and Depth Anything V2 ports.
    """
    out: dict[str, mx.array] = {}

    def take(k: str) -> np.ndarray:
        return state.pop(k).double().numpy()

    def put(k: str, v: np.ndarray) -> None:
        out[k] = mx.array(np.ascontiguousarray(v, dtype=np.float32))

    put("patch.weight", take("patch_embed.proj.weight").transpose(2, 3, 1, 0).reshape(patch * patch * 3, -1))
    put("patch.bias", take("patch_embed.proj.bias"))
    pos = take("pos_embed")[0]
    put("pos.cls", pos[:1] + take("cls_token")[0])
    put("pos.grid", pos[1:])
    if "register_tokens" in state:
        put("registers", take("register_tokens")[0])
    out.update(transformer_params(state))
    put("norm.weight", take("norm.weight"))
    put("norm.bias", take("norm.bias"))
    return out


class DinoLinearProbe:
    """A pretrained DINOv2 ViT and linear probe in MLX, for image or patch classification."""

    def __init__(self, vit, weight, bias, dtype, *, image_logits: bool) -> None:
        """Use the built backbone's position table (already resampled for timm's fixed evaluation crop)."""
        self.patch = vit.patch_embed.proj.kernel_size[0]
        self.pos = vit.pos_embed.detach()[:, 1:].float()
        self.offset = getattr(vit, "interpolate_offset", 0.0)
        self.dtype = getattr(mx, dtype)
        p = {k: v.astype(self.dtype) for k, v in dinov2_params(vit.state_dict(), self.patch).items()}
        del p["pos.grid"]
        depth, heads, eps = depth_of(p), vit.blocks[0].attn.num_heads, vit.norm.eps
        w, b = to_mx(weight.T).astype(self.dtype), to_mx(bias).astype(self.dtype)
        mx.eval(p, w, b)

        patch, run_dtype = self.patch, self.dtype

        def forward(x, table) -> mx.array:
            t = patches(x.astype(run_dtype), patch) @ p["patch.weight"] + (table + p["patch.bias"])
            cls = mx.broadcast_to(p["pos.cls"], (x.shape[0], 1, t.shape[-1]))
            t, _ = blocks(p, depth, heads, eps, mx.concatenate([cls, t], axis=1), ())
            t = mx.fast.layer_norm(t, p["norm.weight"], p["norm.bias"], eps)
            tokens = t[:, 1:]
            if image_logits:
                pooled = mx.mean(tokens.astype(mx.float32), axis=1).astype(run_dtype)
                tokens = mx.concatenate([t[:, 0], pooled], axis=-1)
            return mx.addmm(b, tokens, w)

        self.forward = mx.compile(forward)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Normalised, patch-aligned NCHW image to logits over images or patch tokens."""
        return to_torch(self.logits(image))

    def logits(self, image: torch.Tensor) -> mx.array:
        """`__call__`, the logits left on MLX (lazy, in the run dtype)."""
        h, w = image.shape[-2] // self.patch, image.shape[-1] // self.patch
        side = math.isqrt(self.pos.shape[1])
        table = self.pos
        if (h, w) != (side, side):
            table = (
                F.interpolate(
                    table.reshape(1, side, side, -1).permute(0, 3, 1, 2),
                    scale_factor=((h + self.offset) / side, (w + self.offset) / side),
                    mode="bicubic",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .reshape(1, h * w, -1)
            )
        return self.forward(to_mx(image), to_mx(table).astype(self.dtype))


EPS = 1e-6  # LayerNorm
PATCH = 14
TABLES = 8  # position tables kept (one per grid size, least recently used dropped)
COMPILED = 8  # compiled forwards kept (one per set of hidden-state layers)


def _embed(p: dict[str, mx.array], x: mx.array, table: mx.array) -> mx.array:
    """`[B, 3, H, W]` -> tokens `[B, 1 + R + N, C]` in the weights' dtype (`table`: `Engine.table`, already cast)."""
    t = patches(x.astype(table.dtype), PATCH) @ p["patch.weight"] + table[1:]
    prefix = table[:1] if "registers" not in p else mx.concatenate([table[:1], p["registers"]])
    return mx.concatenate([mx.broadcast_to(prefix, (x.shape[0], *prefix.shape)), t], axis=1)


def dinov2_layers(
    p: dict[str, mx.array], heads: int, x: mx.array, table: mx.array, layers: tuple[int, ...]
) -> list[mx.array]:
    """Normalised `[B, 3, H, W]` -> the tokens `[B, 1 + R + N, C]` after the blocks `layers`, each through the norm.

    DINOv2's `get_intermediate_layers(norm=True)` with 1-based, ascending `layers` (the blocks after the last one do
    not run); `table`: `dinov2_table` in the weights' dtype. Lazy, in that dtype: for a port that writes its head in
    MLX operations too (Depth Anything V2, MoGe).
    """
    _, hs = blocks(p, max(layers), heads, EPS, _embed(p, x, table), layers)
    return [mx.fast.layer_norm(h, p["norm.weight"], p["norm.bias"], EPS) for h in hs]


def dinov2_table(
    cls: torch.Tensor, grid: torch.Tensor, bias: torch.Tensor, h0: int, w0: int, offset: float = 0.1
) -> torch.Tensor:
    """The embedding table `[1 + h0 * w0, C]` of an `h0 x w0` patch grid (float32): what `_embed` adds.

    Row 0 = `cls` (`pos.cls`: class token + its position), the rest = the patch positions `grid` (`pos.grid`, a square
    table) + the patch embedding's `bias`. Another grid size is DINOv2's `interpolate_pos_encoding`: bicubic by scale
    factors, with its `interpolate_offset`.
    """
    side = math.isqrt(grid.shape[0])
    if (h0, w0) != (side, side):
        g = grid.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        g = F.interpolate(g, scale_factor=((h0 + offset) / side, (w0 + offset) / side), mode="bicubic")
        grid = g[0].permute(1, 2, 0).reshape(h0 * w0, -1)
    return torch.cat([cls, grid + bias])


def _feature_forward(
    p: dict[str, mx.array], depth: int, heads: int, layers: tuple[int, ...], x: mx.array, table: mx.array
) -> tuple[mx.array, ...]:
    """Normalised `[B, 3, H, W]` -> float32 `(last_hidden_state [B, N, C], *hidden states at layers)`."""
    t, hs = blocks(p, depth, heads, EPS, _embed(p, x, table), layers)
    last = mx.fast.layer_norm(t, p["norm.weight"], p["norm.bias"], EPS)
    return tuple(a.astype(mx.float32) for a in (last, *hs))


class DinoFeatureEngine:
    """The MLX network of one loaded model: converted weights, position tables and compiled forwards.

    Attributes:
        hidden_size: Channels of every token.
        depth: Number of blocks.
        registers: Number of register tokens.
    """

    def __init__(
        self,
        p32: dict[str, mx.array],
        dtype: str,
        heads: int,
        interpolate: Callable[[torch.Tensor, int, int], torch.Tensor],
    ) -> None:
        """Use the checkpoint's DINOv2 parameters and the family's position interpolation."""
        self.dtype = getattr(mx, dtype)
        self.interpolate = interpolate
        host = {k: torch.from_numpy(np.array(p32.pop(k))) for k in ("pos.cls", "pos.grid", "patch.bias")}
        self.pos = torch.cat([host["pos.cls"], host["pos.grid"]])[None]  # fp32 prefix and patch positions
        self.bias = host["patch.bias"]
        self.p = {k: v.astype(self.dtype) for k, v in p32.items()}
        p32.clear()
        mx.eval(self.p)
        self.heads = heads
        self.depth = depth_of(self.p)
        self.hidden_size = int(self.p["norm.weight"].shape[0])
        self.registers = self.p["registers"].shape[0] if "registers" in self.p else 0
        self.compile = tuning().compile_steps
        self.tables: OrderedDict[tuple[int, int], mx.array] = OrderedDict()
        self.compiled: OrderedDict[tuple[int, ...], Callable[[mx.array, mx.array], tuple[mx.array, ...]]] = (
            OrderedDict()
        )

    def table(self, h0: int, w0: int) -> mx.array:
        """`[1 + h0 * w0, C]` in the run dtype: row 0 = cls token + its position, the rest = positions + patch bias."""
        key = (h0, w0)
        if key not in self.tables:
            t = self.interpolate(self.pos, h0, w0)[0].clone()  # fp32, the torch runners' interpolation
            t[1:] += self.bias
            table = to_mx(t).astype(self.dtype)
            mx.eval(table)
            self.tables[key] = table
            while len(self.tables) > TABLES:
                self.tables.popitem(last=False)
        self.tables.move_to_end(key)
        return self.tables[key]

    def _forward(self, layers: tuple[int, ...]) -> Callable[[mx.array, mx.array], tuple[mx.array, ...]]:
        if layers not in self.compiled:
            params, depth, heads = self.p, self.depth, self.heads

            def run(x: mx.array, table: mx.array) -> tuple[mx.array, ...]:
                return _feature_forward(params, depth, heads, layers, x, table)

            self.compiled[layers] = mx.compile(run) if self.compile else run
            while len(self.compiled) > COMPILED:
                self.compiled.popitem(last=False)
        self.compiled.move_to_end(layers)
        return self.compiled[layers]

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Normalised `[B, 3, H, W]` CPU tensor -> float32 CPU `(last_hidden_state, hidden states at layers)`."""
        table = self.table(image.shape[2] // PATCH, image.shape[3] // PATCH)
        out = self._forward(tuple(layers))(to_mx(image), table)
        mx.eval(*out)
        last, *hs = (torch.from_numpy(np.array(a)) for a in out)
        return last, hs
