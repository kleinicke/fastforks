"""Kernels-mode DINOv2 ViT: fp16 weights and activations, fp16-accumulate GEMMs, flash attention kernel (head_dim 64).

`use_fast_vit(vit)` replaces `get_intermediate_layers` of a `DinoVisionTransformer` (the Depth Anything V2 / DEFOM /
WAFT backbone) by a general version (any input size that is a multiple of the patch size, nothing compiled or tuned
per size). The block maths is the upstream pre-norm block (`x + ls1 * attn(norm1 x)`, `x + ls2 * mlp(norm2 x)`); only
the precision and the kernels change:

- weights and activations in fp16, autocast off inside, cuBLAS GEMMs with fp16 accumulation (GeForce tensor cores run
  twice as fast; `torch.backends.cuda.matmul.allow_fp16_accumulation`)
- attention: the `fastbb` flash kernel reads q / k / v straight from the packed qkv GEMM output and writes
  `[B, N, heads * 64]` for the projection (no transposes). `attn="hh"`: fp16 accumulation for `q k^T` and `p v`
  (fastest), `"fh"`: fp32 `q k^T`, `"ff"`: fp32 for both (SDPA-flash precision); an fp16 overflow (NaN) on the
  first calls switches to the next variant (`fastforks.kernels.fastbb.Flash64`)
- residual + LayerScale as one in-place `addcmul`
- positional embedding interpolated once per grid size (fp32 bicubic, upstream formula), cached

Unsupported structure (head_dim != 64, other activations; the SwiGLU FFN unless `swiglu=True`) raises; there is no
PyTorch fallback.
"""

from __future__ import annotations

import math
import weakref
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import fp16_accumulate as _fp16_accumulate  # re-exported for the ViT families
from fastforks.kernels.dpt_fused import DEEP_K, DptFused
from fastforks.kernels.fastbb import Flash64

from .net.dinov2_layers.mlp import Mlp
from .net.dinov2_layers.swiglu_ffn import SwiGLUFFN

if TYPE_CHECKING:
    import contextlib
    from collections.abc import Sequence

    from .net.dinov2 import DinoVisionTransformer

HEAD_DIM = 64
POS_SIZES = 8  # position tables kept (one per patch grid; video at one size always hits)
FUSED_ROWS = 512
"""`FastViT(fused=...)` runs calls of at least this many tokens (B * N) on the fused-epilogue GEMMs. Measured on DINOv2
(RTX 5060 Ti): 82 / 257 / 362 tokens are up to 13 % slower fused (ViT-S / B; ViT-L even), 577 and more are 0-7 %
(ViT-S at 1370: ~20 %) faster; splitting single GEMMs between the two paths did not move the crossover."""


def _flash(qkv: torch.Tensor, scale: float, attn: Flash64) -> torch.Tensor:
    """softmax(q k^T * scale) v for the packed qkv `[B, N, 3, H, 64]` (fp16) -> `[B, N, H * 64]`."""
    b, n, _, h, d = qkv.shape
    return attn(*qkv.unbind(2), scale).view(b, n, h * d)


class FastViT:
    """The fast `get_intermediate_layers` of one ViT (see `use_fast_vit`)."""

    def __init__(  # noqa: PLR0913 - the switches of the ViT families
        self,
        vit: Any,  # noqa: ANN401 - duck-typed DINOv2 module
        attn: str,
        out_dtype: torch.dtype | None,
        *,
        swiglu: bool = False,
        cache_pos: bool = True,
        fused: DptFused | None = None,
    ) -> None:
        """Check the structure, keep the fp32 position table, convert the ViT to fp16 in place.

        `swiglu`: also accept the SwiGLU FFN (DINOv2 ViT-g; `block` runs it, subclasses with their own blocks may not).
        `cache_pos`: keep the fp32 table and the interpolated tables (False: no persistent memory, the fp16 model's
        table is interpolated in fp32 on each call, as upstream on an fp16 model).
        `fused`: run the block GEMMs of large calls (`FUSED_ROWS`) with the `dpt_fused` epilogues (bias, GELU,
        LayerScale in fp32 + residual, in place): fewer launches, no FFN-output temporaries.
        """
        if vit.embed_dim % vit.num_heads or vit.embed_dim // vit.num_heads != HEAD_DIM:
            msg = f"fast ViT needs head_dim {HEAD_DIM}, got {vit.embed_dim}/{vit.num_heads}"
            raise ValueError(msg)
        blocks = [
            b
            for b in (c for ch in vit.blocks for c in (ch if vit.chunked_blocks else [ch]))
            if not isinstance(b, nn.Identity)
        ]
        for b in blocks:
            mlp = b.mlp
            if swiglu and isinstance(mlp, SwiGLUFFN):
                continue
            if not (isinstance(mlp, Mlp) and isinstance(mlp.act, nn.GELU) and mlp.act.approximate == "none"):
                msg = f"fast ViT supports the Mlp(GELU) FFN only, got {type(mlp).__name__}"
                raise ValueError(msg)
        self.blocks: list[Any] = blocks
        self.cache_pos = cache_pos  # upstream interpolates the fp32 table (unused without the cache)
        self.pos_fp32 = vit.pos_embed.detach().float().clone() if cache_pos else torch.empty(0)
        self.vit = vit.half().eval()
        self.attn, self.out_dtype = Flash64(attn), out_dtype  # ValueError on an unknown variant
        self._pos: dict[tuple[int, int], torch.Tensor] = {}
        self.fused = fused
        self._scales: dict[int, torch.Tensor] = {}  # fp32 LayerScale of the fused epilogues, per LayerScale module

    @property
    def vit(self) -> Any:  # noqa: ANN401 - duck-typed DINOv2 module
        """The ViT this object runs (held weakly after `use_fast_vit`)."""
        vit = self._vit() if isinstance(self._vit, weakref.ref) else self._vit
        if vit is None:
            msg = "the ViT of this FastViT was freed"
            raise ReferenceError(msg)
        return vit

    @vit.setter
    def vit(self, vit: Any) -> None:  # noqa: ANN401 - duck-typed DINOv2 module
        self._vit = vit

    def pos_embed(self, h0: int, w0: int) -> torch.Tensor:
        """The fp16 position embedding for an `h0 x w0` patch grid (cached for `POS_SIZES` grids)."""
        if not self.cache_pos:
            return self.interpolate_pos(h0, w0).half()
        pe = self._pos.get((h0, w0))
        if pe is None:
            if len(self._pos) >= POS_SIZES:
                del self._pos[next(iter(self._pos))]
            pe = self._pos[h0, w0] = self.interpolate_pos(h0, w0).half()
        return pe

    def interpolate_pos(self, h0: int, w0: int) -> torch.Tensor:
        """The fp32 position table for an `h0 x w0` patch grid, as upstream `interpolate_pos_encoding`."""
        vit = self.vit
        pe = self.pos_fp32.to(vit.pos_embed.device) if self.cache_pos else vit.pos_embed.float()
        n = pe.shape[1] - 1
        m = int(math.sqrt(n))
        if n == h0 * w0 and h0 == w0:
            return pe
        p = F.interpolate(
            pe[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2),
            scale_factor=((h0 + vit.interpolate_offset) / m, (w0 + vit.interpolate_offset) / m),
            mode="bicubic",
            antialias=vit.interpolate_antialias,
        )
        return torch.cat([pe[:, :1], p.permute(0, 2, 3, 1).flatten(1, 2)], 1)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patch tokens + cls (+ register) tokens + positions, fp16."""
        vit = self.vit
        b, _, h, w = x.shape
        ps = vit.patch_size
        t = vit.patch_embed(x.half())
        t = torch.cat([vit.cls_token.expand(b, -1, -1), t], 1)
        t = t + self.pos_embed(h // ps, w // ps)
        if vit.register_tokens is not None:
            t = torch.cat([t[:, :1], vit.register_tokens.expand(b, -1, -1), t[:, 1:]], 1)
        return t

    def attend(self, qkv: torch.Tensor, scale: float) -> torch.Tensor:
        """softmax(q k^T * scale) v of the packed qkv `[B, N, 3, H, 64]` -> `[B, N, H * 64]` (the flash kernel)."""
        return _flash(qkv, scale, self.attn)

    def accumulate(self) -> contextlib.AbstractContextManager[None]:
        """Context of the GEMMs: fp16 accumulation (subclasses without the kernels / flag return a null context)."""
        return _fp16_accumulate()

    def block(self, blk: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - duck-typed DINOv2 block
        """One pre-norm block (eval, no drop path)."""
        a, m = blk.attn, blk.mlp
        b, n, c = x.shape
        h = a.num_heads
        if self.fused is not None and b * n >= FUSED_ROWS:
            return self._fused(self.fused, blk, x)
        qkv = F.linear(blk.norm1(x), a.qkv.weight, a.qkv.bias).view(b, n, 3, h, c // h)
        o = F.linear(self.attend(qkv, (c // h) ** -0.5), a.proj.weight, a.proj.bias)
        g1 = getattr(blk.ls1, "gamma", None)
        x = torch.addcmul(x, o, g1) if g1 is not None else x + o
        if isinstance(m, SwiGLUFFN):  # silu(x1) * x2 of the packed w12 output, as upstream
            x1, x2 = F.linear(blk.norm2(x), m.w12.weight, m.w12.bias).chunk(2, dim=-1)
            o = F.linear(F.silu(x1) * x2, m.w3.weight, m.w3.bias)
        else:
            o = F.linear(F.gelu(F.linear(blk.norm2(x), m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)
        g2 = getattr(blk.ls2, "gamma", None)
        return x.addcmul_(o, g2) if g2 is not None else x.add_(o)

    def _scale(self, ls: Any) -> torch.Tensor | None:  # noqa: ANN401 - duck-typed LayerScale / Identity
        """The LayerScale `gamma` in fp32 on its device (the fused epilogue scales in fp32), None without LayerScale."""
        g = getattr(ls, "gamma", None)
        if g is None:
            return None
        s = self._scales.get(id(ls))
        if s is None or s.device != g.device:
            s = self._scales[id(ls)] = g.detach().float()
        return s

    def _fused(self, k: DptFused, blk: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - duck-typed DINOv2 block
        """`block` on the fused-epilogue GEMMs; FFN outputs with `DEEP_K` inputs and LayerScale stay in cuBLAS."""
        a, m = blk.attn, blk.mlp
        b, n, c = x.shape
        h = a.num_heads
        qkv = k.gemm(blk.norm1(x), a.qkv.weight, a.qkv.bias).view(b, n, 3, h, c // h)
        x = k.gemm(self.attend(qkv, (c // h) ** -0.5), a.proj.weight, a.proj.bias, scale=self._scale(blk.ls1), res=x)
        y = blk.norm2(x)
        if isinstance(m, SwiGLUFFN):
            x1, x2 = k.gemm(y, m.w12.weight, m.w12.bias).chunk(2, dim=-1)
            hid, out = F.silu(x1).mul_(x2), m.w3
        else:
            hid, out = k.gemm(y, m.fc1.weight, m.fc1.bias, act=k.gelu), m.fc2
        g2 = self._scale(blk.ls2)
        if out.in_features >= DEEP_K and g2 is not None:
            return x.addcmul_(F.linear(hid, out.weight, out.bias), blk.ls2.gamma)
        return k.gemm(hid, out.weight, out.bias, scale=g2, res=x, out=x)

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | Sequence[int] = 1,
        reshape: bool = False,  # noqa: FBT001, FBT002 - upstream signature
        return_class_token: bool = False,  # noqa: FBT001, FBT002 - upstream signature
        norm: bool = True,  # noqa: FBT001, FBT002 - upstream signature
    ) -> tuple:
        """Same contract as `DinoVisionTransformer.get_intermediate_layers` (outputs in `out_dtype`, default fp16)."""
        nb = len(self.blocks)
        layers = set(range(nb - n, nb)) if isinstance(n, int) else set(n)
        outs = []
        with torch.autocast("cuda", enabled=False), self.accumulate():
            t = self.tokens(x)
            for i, blk in enumerate(self.blocks):
                t = self.block(blk, t)
                if i in layers:
                    outs.append(self.vit.norm(t) if norm else t.clone())
                if i >= max(layers):
                    break
        if len(outs) != len(layers):
            msg = f"only {len(outs)} / {len(layers)} blocks found"
            raise ValueError(msg)
        if self.out_dtype is not None:
            outs = [o.to(self.out_dtype) for o in outs]
        vit = self.vit
        cls = [o[:, 0] for o in outs]
        outs = [o[:, 1 + vit.num_register_tokens :] for o in outs]
        if reshape:
            bs, _, h, w = x.shape
            outs = [
                o.reshape(bs, h // vit.patch_size, w // vit.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for o in outs
            ]
        return tuple(zip(outs, cls, strict=True)) if return_class_token else tuple(outs)


def use_fast_vit(
    vit: DinoVisionTransformer,
    *,
    attn: str = "hh",
    out_dtype: torch.dtype | None = None,
    cache_pos: bool = True,
    fused: DptFused | None = None,
) -> FastViT:
    """Route `vit.get_intermediate_layers` through the fast fp16 path (the ViT is converted to fp16 in place).

    Args:
        vit: A `DinoVisionTransformer` (eval).
        attn: Attention precision, `"hh"` (fastest), `"fh"` or `"ff"`.
        out_dtype: Dtype of the returned tokens (default fp16; DEFOM's bf16 DPT heads want bf16).
        cache_pos: Keep the fp32 position table and the interpolated tables on the device (see `FastViT`).
        fused: The `dpt_fused` library: fused-epilogue block GEMMs in large calls (see `FastViT`).
    """
    fast = FastViT(vit, attn, out_dtype, cache_pos=cache_pos, fused=fused)
    # the ViT holds `fast` through its `get_intermediate_layers`: a strong reference back would be a cycle
    fast._vit = weakref.ref(vit)  # noqa: SLF001
    vit.get_intermediate_layers = fast.get_intermediate_layers  # ty: ignore[invalid-assignment]  # per-instance override
    return fast


def fp16_accumulate() -> contextlib.AbstractContextManager[None]:
    """Context in which cuBLAS GEMMs accumulate in fp16 (`torch.backends.cuda.matmul.allow_fp16_accumulation`)."""
    return _fp16_accumulate()


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, attn: Flash64) -> torch.Tensor:
    """softmax(q k^T * scale) v with the `fastbb` flash kernel (no mask), for ViTs whose blocks `FastViT` can't run.

    Args:
        q: `[B, N, H, 64]` fp16 CUDA, any strides with the last dim contiguous (e.g. a view into a packed qkv, or a
            transposed `[B, H, N, 64]` tensor).
        k: Like `q`.
        v: Like `q`.
        scale: Softmax scale (usually `64 ** -0.5`).
        attn: The model's `Flash64` (accumulation precision and overflow fallback, see the module docstring).

    Returns:
        `[B, N, H, 64]` contiguous; `.view(B, N, H * 64)` feeds the output projection.
    """
    if q.shape[-1] != HEAD_DIM or any(t.stride(-1) != 1 or t.shape != q.shape for t in (q, k, v)):
        msg = f"flash_attention: [B, N, H, {HEAD_DIM}] views with a contiguous last dim, got {q.shape}"
        raise ValueError(msg)
    return attn(q, k, v, scale)
