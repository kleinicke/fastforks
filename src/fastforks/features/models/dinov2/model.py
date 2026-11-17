"""DINOv2 runners: the official hub configurations (facebookresearch/dinov2 @ 7764ea0) on the vendored ViT.

`hub/backbones.py`: image size 518 (37 x 37 position table), patch 14, LayerScale init 1.0, no block chunks; ViT-g
uses the fused SwiGLU FFN; the register variants have 4 registers, antialiased position interpolation and offset 0.
The vendored ViT (Depth Anything V2's copy) predates upstream's offset-0 branch (interpolate to the grid size instead
of by a scale factor), so the position embedding is computed here as upstream 7764ea0 does.

Outputs follow Hugging Face `Dinov2Model`: `hidden_states[0]` = embeddings (patches + position, cls, registers),
`hidden_states[i]` = output of block i (no norm), `last_hidden_state` = final LayerNorm of the last block's output,
`pooler_output` = its cls token.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest import mock

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks.features._runner import Forward, Runner
from fastforks.kernels import dpt_fused
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.shared.depth_anything_v2 import FastViT, vit_base, vit_giant2, vit_large, vit_small
from fastforks.shared.depth_anything_v2.fast_vit import _fp16_accumulate
from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer
from fastforks.shared.depth_anything_v2.net.dinov2_layers import SwiGLUFFN

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.features._runner import Context

_ARCHS: dict[str, Callable[..., Any]] = {"vits": vit_small, "vitb": vit_base, "vitl": vit_large, "vitg": vit_giant2}


@dataclass(frozen=True)
class Params:
    """Variant `params`."""

    arch: str  # "vits", "vitb", "vitl" or "vitg"
    registers: int  # 0 or 4
    fp16_accumulate: bool  # portable: cuBLAS fp16 accumulation in the GEMMs (option)


def _vit(p: Params) -> Any:  # noqa: ANN401 - vendored DinoVisionTransformer
    """The hub `dinov2_<arch>14[_reg]` network (no weights)."""
    return _ARCHS[p.arch](
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="swiglufused" if p.arch == "vitg" else "mlp",
        block_chunks=0,
        num_register_tokens=p.registers,
        interpolate_antialias=bool(p.registers),
        interpolate_offset=0.0 if p.registers else 0.1,
    )


def _pos_embed(vit: Any, table: torch.Tensor, h0: int, w0: int) -> torch.Tensor:  # noqa: ANN401 - DINOv2 module
    """Upstream 7764ea0 `interpolate_pos_encoding` of the fp32 `table` `[1, 1 + M * M, C]` for an `h0 x w0` grid."""
    n = table.shape[1] - 1
    m = int(math.sqrt(n))
    if n == h0 * w0 and h0 == w0:
        return table
    off = vit.interpolate_offset
    kw: dict[str, Any] = (
        {"scale_factor": ((h0 + off) / m, (w0 + off) / m)} if off else {"size": (h0, w0)}
    )  # upstream's (w0, h0) are (rows, columns)
    p = F.interpolate(
        table[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2), mode="bicubic", antialias=vit.interpolate_antialias, **kw
    )
    return torch.cat([table[:, :1], p.permute(0, 2, 3, 1).flatten(1, 2)], 1)


def _interpolate_pos_encoding(self: Any, x: torch.Tensor, w: int, h: int) -> torch.Tensor:  # noqa: ANN401 - DINOv2 module
    """Replacement of the vendored method (same arguments: `w` / `h` = input rows / columns)."""
    ps = self.patch_size
    return _pos_embed(self, self.pos_embed.float(), w // ps, h // ps).to(x.dtype)


class _Vit(DinoVisionTransformer):
    """The vendored ViT with upstream 7764ea0's position interpolation."""

    interpolate_pos_encoding = _interpolate_pos_encoding


def _network(ctx: Context) -> Any:  # noqa: ANN401 - vendored DinoVisionTransformer
    # every tensor is assigned: built on the meta device (no allocation), without upstream's initialisation
    with torch.device("meta"), mock.patch.object(DinoVisionTransformer, "init_weights", return_value=None):
        vit = _vit(ctx.params)
    load_strict(vit, ctx.state_dict(), f"{ctx.variant.ref} checkpoint", assign=True)
    vit.__class__ = _Vit  # not a bound method on the instance: that reference cycle kept the GPU weights after `del`
    return vit.eval()


def _init(runner: Runner, vit: Any) -> None:  # noqa: ANN401 - DINOv2 module
    runner.hidden_size, runner.num_layers = vit.embed_dim, len(vit.blocks)
    runner.num_register_tokens = vit.num_register_tokens
    runner.num_prefix_tokens = 1 + vit.num_register_tokens


def _attention(a: Any, y: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - vendored DINOv2 attention
    """SDPA attention + projection; its temporaries (qkv, the attention output) are freed on return."""
    b, n, c = y.shape
    h = a.num_heads
    qkv = F.linear(y, a.qkv.weight, a.qkv.bias).view(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(b, n, c)
    return F.linear(o, a.proj.weight, a.proj.bias)


def _mlp(m: Any, y: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - vendored DINOv2 Mlp / SwiGLUFFN
    if isinstance(m, SwiGLUFFN):
        x1, x2 = F.linear(y, m.w12.weight, m.w12.bias).chunk(2, dim=-1)
        return F.linear(F.silu(x1) * x2, m.w3.weight, m.w3.bias)
    return F.linear(F.gelu(F.linear(y, m.fc1.weight, m.fc1.bias)), m.fc2.weight, m.fc2.bias)


def _block(blk: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - vendored DINOv2 block
    """One vendored block (eval, no drop path) as functional ops, same maths.

    Small calls are CPU launch bound, and the ~14 module calls per block cost more than its GPU work; the LayerScale +
    residual is one `addcmul` (one fp16 rounding instead of two). No temporaries outlive their half of the block.
    The second one is in place unless grad mode is on (`module()` backward: `norm2` saved `x`).
    """
    x = torch.addcmul(x, _attention(blk.attn, blk.norm1(x)), blk.ls1.gamma)
    o = _mlp(blk.mlp, blk.norm2(x))
    return torch.addcmul(x, o, blk.ls2.gamma) if torch.is_grad_enabled() else x.addcmul_(o, blk.ls2.gamma)


class Portable(Runner):
    """Plain PyTorch, functional blocks with SDPA attention: float32 or fp16 weights / activations (fp32 positions)."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.half = ctx.dtype == "float16"
        self.acc = self.half and ctx.params.fp16_accumulate
        net = _network(ctx)
        if self.half:  # fp16 weights (no per-call autocast casts); the position table stays fp32, as upstream uses it
            pos = net.pos_embed.detach().clone()
            net.half().pos_embed = torch.nn.Parameter(pos, requires_grad=False)
        self.net = net.to(ctx.device)
        _init(self, self.net)

    def _run(self, x: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        net = self.net
        hs = []
        with _fp16_accumulate() if self.acc else contextlib.nullcontext():
            t = net.prepare_tokens_with_masks(x.half() if self.half else x)
            if 0 in layers:
                hs.append(t.float())
            for i, blk in enumerate(net.blocks, 1):
                t = _block(blk, t)
                if i in layers:
                    hs.append(t.float())
            last = net.norm(t).float()
        out = {"last_hidden_state": last, "pooler_output": last[:, 0]}
        if layers:
            out["hidden_states"] = torch.stack(hs, 1)
        return out

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Features of a resized, normalised batch."""
        return self._run(image, layers)

    def module(self) -> torch.nn.Module:
        """The network: normalised `[B, 3, h, w]` (sides multiples of 14) -> `last_hidden_state` `[B, N, C]`."""
        return Forward(self.net, lambda _net, x: self._run(x, ())["last_hidden_state"])


class _FastDinov2(FastViT):
    """`FastViT` with upstream 7764ea0's position interpolation (offset 0: by grid size)."""

    def interpolate_pos(self, h0: int, w0: int) -> torch.Tensor:
        """The fp32 position table of an `h0 x w0` patch grid."""
        return _pos_embed(self.vit, self.pos_fp32.to(self.vit.pos_embed.device), h0, w0)


# flash attention precision per size (fastbb `flash64_*`): "hh" = fp16 q k^T + p v accumulation (fastest), "fh" =
# fp32 q k^T. 006 rel-L2 of the patch tokens vs fp32 at 1022 px: hh S 0.006 / B 0.021 / L 0.007 / g 0.012, fh B 0.008,
# g 0.007; ViT-B and ViT-g use fh (B: as Depth Anything V2's kernels encoder).
_KERNELS_ATTN = {"vits": "hh", "vitb": "fh", "vitl": "hh", "vitg": "fh"}


# CUDA graphs while the call is CPU launch bound: up to this many activations B * N * C (ViT-S up to ~510 px, ViT-L
# up to ~310 px); larger calls are GPU bound, a graph there only keeps its memory pool, so they run eagerly
_GRAPH_WORK = 1 << 19  # fixed: launch-bound threshold (ViT-L 224 graphed; ViT-S 640x480 is GPU bound: no gain)
_GRAPHS = 4  # fixed: graph cache policy, not architecture


class Kernels(Runner):
    """fastbb + dpt_fused: fp16 weights / activations, flash attention kernel (head_dim 64), fp16-accumulate GEMMs.

    Calls of at least `FUSED_ROWS` tokens run the block GEMMs with fused epilogues (bias, GELU, LayerScale + residual);
    smaller ones in cuBLAS with GELU / SwiGLU and the residual in PyTorch. Outputs are returned as float32.
    Small calls without hidden states (`_GRAPH_WORK`) replay a CUDA graph per input shape, captured on its second call
    (launch bound there); `release_graphs()` frees them.
    """

    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, convert it to fp16, move it to the device."""
        super().__init__(ctx)
        vit = _network(ctx)
        _init(self, vit)
        # fp16 in place on the CPU first: an fp32 copy on the device stays reserved by the caching allocator
        self.fast = _FastDinov2(
            vit, _KERNELS_ATTN[ctx.params.arch], None, swiglu=True, fused=dpt_fused.load(ctx.device)
        )
        self.fast.vit, self.fast.pos_fp32 = vit.to(ctx.device), self.fast.pos_fp32.to(ctx.device)
        pixels = _GRAPH_WORK * vit.patch_size**2 // vit.embed_dim
        self.graphs = SmallShapeGraphs(self._last, pixels=pixels, graphs=_GRAPHS)

    def _run(self, image: torch.Tensor, layers: tuple[int, ...]) -> list[torch.Tensor]:
        """`[last_hidden_state (fp16), *hidden states at layers (float32)]`."""
        fast = self.fast
        hs = []
        with torch.autocast("cuda", enabled=False), _fp16_accumulate():
            t = fast.tokens(image)
            if 0 in layers:
                hs.append(t.float())
            for i, blk in enumerate(fast.blocks, 1):
                t = fast.block(blk, t)
                if i in layers:
                    hs.append(t.float())
            return [fast.vit.norm(t), *hs]

    def _last(self, image: torch.Tensor) -> torch.Tensor:
        return self._run(image, ())[0].float()

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self.graphs.release()

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Features of a resized, normalised batch."""
        if layers:
            last, *hs = self._run(image, layers)
            last = last.float()
        else:
            last, hs = self.graphs(image), []
        out = {"last_hidden_state": last, "pooler_output": last[:, 0]}
        if layers:
            out["hidden_states"] = torch.stack(hs, 1)
        return out


RUNNERS = {"portable": Portable, "kernels": Kernels}
