"""BoQ runners: the shared DINOv2 ViT-B/14 + the vendored Bag-of-Queries aggregator (`net/boq.py`).

Inference as the authors' README: the image resized to `size x size` (bicubic, antialiased, aspect ratio not kept),
ImageNet normalisation; the backbone's patch tokens (after the last block, without the final norm, as upstream's
`DinoV2` backbone) as a feature map go through the aggregator: 3x3 convolution, two BoQ blocks (a transformer encoder
layer, then cross-attention of 64 learned queries), a linear layer over the 128 query outputs and an L2 normalisation.
The queries' self-attention does not depend on the image; it is computed once at load (float32).

`portable`: plain PyTorch, float32 or float16 weights / activations (SDPA attention). `kernels`: the backbone as
`FastViT` (fastbb flash attention, `dpt_fused` block GEMMs, fp16 accumulation), the aggregator in fp16; small batches
replay one CUDA graph per batch size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest import mock

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks.kernels import dpt_fused
from fastforks.retrieval._runner import Forward, Runner, scale
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.shared.depth_anything_v2 import FastViT, fp16_accumulate, sdpa_attention, vit_base
from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer

from .net.boq import BoQ

if TYPE_CHECKING:
    from fastforks.retrieval._runner import Context


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        size: Network input side (square).
        dim: Descriptor size.
    """

    size: int
    dim: int


_PATCH = 14  # fixed: the ViT-B/14 patch size of the hub backbone (also its constructor argument)


def _networks(ctx: Context) -> tuple[Any, BoQ]:
    """`(vit, aggregator)` on the CPU in float32: upstream's `DinoV2()` = the hub ViT-B/14 (img 518, offset 0.1)."""
    with mock.patch.object(DinoVisionTransformer, "init_weights", return_value=None):  # every tensor is assigned
        vit = vit_base(
            img_size=518, patch_size=_PATCH, init_values=1.0, ffn_layer="mlp", block_chunks=0, num_register_tokens=0,
            interpolate_antialias=False, interpolate_offset=0.1,
        )  # fmt: skip
    state = ctx.state_dict()
    what = f"{ctx.variant.ref} checkpoint"
    bb = {k.removeprefix("backbone.dino."): v for k, v in state.items() if k.startswith("backbone.dino.")}
    load_strict(vit, bb, what, assign=True)
    agg = BoQ(in_channels=768, proj_channels=384, num_queries=64, num_layers=2, row_dim=ctx.params.dim // 384)
    load_strict(agg, {k.removeprefix("aggregator."): v for k, v in state.items() if k.startswith("aggregator.")}, what)
    return vit.eval(), agg.eval()


def _queries(agg: BoQ) -> list[torch.Tensor]:
    """Each block's queries after their self-attention and norm (`BoQBlock.forward`'s image-independent part), fp32."""
    out = []
    blocks: Any = agg.boqs
    with torch.no_grad():
        for blk in blocks:
            q = blk.queries
            out.append(blk.norm_q(q + blk.self_attn(q, q, q)[0]))
    return out


def _aggregate(agg: BoQ, queries: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Upstream `BoQ.forward` on feature maps `[B, C, h, w]` with the precomputed `queries` -> `[B, dim]` float32."""
    x = agg.norm_input(agg.proj_c(x).flatten(2).permute(0, 2, 1))
    outs = []
    blocks: Any = agg.boqs
    for blk, q in zip(blocks, queries, strict=True):
        x = blk.encoder(x)
        out = blk.cross_attn(q.expand(x.shape[0], -1, -1), x, x, need_weights=False)[0]
        outs.append(blk.norm_out(out))
    out = agg.fc(torch.cat(outs, dim=1).permute(0, 2, 1)).flatten(1)
    return F.normalize(out.float(), p=2, dim=-1)


def _grid(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Patch tokens `[B, 1 + h * w, C]` (cls first) -> `[B, C, h, w]` of the network input `x`."""
    h, w = x.shape[-2] // _PATCH, x.shape[-1] // _PATCH
    return t[:, 1:].transpose(1, 2).reshape(t.shape[0], -1, h, w)


class _BoQ(Runner):
    """Resize, normalise, backbone, aggregator."""

    Params = Params
    batch = None
    agg: BoQ

    def _resize(self, image: torch.Tensor) -> torch.Tensor:
        s = self.ctx.params.size
        x = F.interpolate(image, (s, s), mode="bicubic", align_corners=False, antialias=True)
        return scale(x, self.ctx.io)

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Descriptors `[B, dim]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        return {"descriptor": self._descriptor(self._resize(image))}

    def module(self) -> torch.nn.Module:
        """The network with its pre-processing: `[B, 3, H, W]` in `[0, 1]` -> descriptors `[B, dim]`."""
        return Forward(self.agg, lambda _net, image: self(image)["descriptor"])


class Portable(_BoQ):
    """Plain PyTorch, float32 or float16 (SDPA attention)."""

    dtypes = ("float32", "float16")
    compiled_methods = ("_descriptor",)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        vit, agg = _networks(ctx)
        sdpa_attention(vit)
        queries = _queries(agg)
        if self.dtype != torch.float32:  # the position table stays fp32, as upstream interpolates it
            pos = vit.pos_embed.detach().clone()
            vit.to(self.dtype).pos_embed = torch.nn.Parameter(pos, requires_grad=False)
        self.vit = vit.to(ctx.device)
        self.agg = agg.to(ctx.device, self.dtype)
        self.queries = [q.to(ctx.device, self.dtype) for q in queries]

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        t = self.vit.prepare_tokens_with_masks(x)
        for blk in self.vit.blocks:
            t = blk(t)
        return _aggregate(self.agg, self.queries, _grid(t, x))


# CUDA graphs up to this many network inputs per call (322 x 322: launch bound up to a few images)
_GRAPH_BATCH = 8  # fixed: launch-bound threshold, not architecture
_GRAPHS = 4  # fixed: graph cache policy


class Kernels(_BoQ):
    """fastbb + dpt_fused ViT-B (fp16, flash attention, fused block GEMMs); fp16 aggregator.

    Batches of up to `_GRAPH_BATCH` images replay a CUDA graph per batch size (captured on its second call);
    `release_graphs()` frees them.
    """

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, convert it to fp16, move it to the device."""
        super().__init__(ctx)
        vit, agg = _networks(ctx)
        queries = _queries(agg)
        self.fast = FastViT(vit, "fh", None, fused=dpt_fused.load(ctx.device))  # ViT-B: fp32 q k^T
        self.fast.vit, self.fast.pos_fp32 = vit.to(ctx.device), self.fast.pos_fp32.to(ctx.device)
        self.agg = agg.half().to(ctx.device)
        self.queries = [q.to(ctx.device, torch.float16) for q in queries]
        s = ctx.params.size
        self.graphs = SmallShapeGraphs(self._run, pixels=_GRAPH_BATCH * s * s, graphs=_GRAPHS)

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        fast = self.fast
        with torch.autocast("cuda", enabled=False):
            with fp16_accumulate():
                t = fast.tokens(x)
                for blk in fast.blocks:
                    t = fast.block(blk, t)
            return _aggregate(self.agg, self.queries, _grid(t, x))

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        return self.graphs(x)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
