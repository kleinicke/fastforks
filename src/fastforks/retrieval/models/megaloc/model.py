"""MegaLoc runners: the shared DINOv2 ViT-B/14 + the vendored SALAD-style aggregator (`net/megaloc.py`).

Inference as the authors' README: the image resized to `size x size` (bilinear, antialiased, aspect ratio not kept),
ImageNet normalisation; the backbone's final-norm patch tokens and cls token go through the aggregator (Sinkhorn
optimal transport of the patch features onto 64 clusters, plus a global token), a linear layer to `dim` and an L2
normalisation.

`portable`: plain PyTorch, float32 or float16 weights / activations (SDPA attention); the Sinkhorn assignment and the
cluster sums always run in float32. `kernels`: the backbone as `FastViT` (fastbb flash attention, `dpt_fused` block
GEMMs, fp16 accumulation), the aggregator GEMMs in fp16; small batches replay one CUDA graph per batch size.
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

from .net.megaloc import Aggregator, get_matching_probs

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


_AGG = {"num_channels": 768, "num_clusters": 64, "cluster_dim": 256, "token_dim": 256, "mlp_dim": 512}


def _networks(ctx: Context) -> tuple[Any, Aggregator]:
    """`(vit, aggregator)` on the CPU in float32: upstream's `DINOv2()` = the hub ViT-B/14 (img 518, offset 0.1)."""
    with mock.patch.object(DinoVisionTransformer, "init_weights", return_value=None):  # every tensor is assigned
        vit = vit_base(
            img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0, num_register_tokens=0,
            interpolate_antialias=False, interpolate_offset=0.1,
        )  # fmt: skip
    state = ctx.state_dict()
    bb = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
    what = f"{ctx.variant.ref} checkpoint"
    load_strict(vit, bb, what, not_in_checkpoint=("mask_token",), assign=True)
    agg = Aggregator(ctx.params.dim, {**_AGG, "dropout": 0}, _AGG["num_clusters"] * _AGG["cluster_dim"] + 256)
    load_strict(agg, {k.removeprefix("aggregator."): v for k, v in state.items() if k.startswith("aggregator.")}, what)
    return vit.eval(), agg.eval()


def _mlp(seq: torch.nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    """The aggregator's `conv1x1 -> ReLU -> conv1x1` (or Linear) on tokens `[B, N, C]`."""
    a: Any = seq[0]
    b: Any = seq[-1]
    w1, w2 = a.weight.flatten(1), b.weight.flatten(1)
    return F.linear(F.relu(F.linear(x, w1, a.bias)), w2, b.bias)


def _aggregate(agg: Aggregator, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Upstream `Aggregator.forward` + L2 norm on patch tokens `[B, N, C]` and cls `[B, C]` (no feature map).

    The GEMMs run in the weights' dtype, the Sinkhorn assignment and the cluster sums in float32.
    """
    a = agg.agg
    f = _mlp(a.cluster_features, x).float().transpose(1, 2)  # [B, cluster_dim, N]
    p = _mlp(a.score, x).float().transpose(1, 2)  # [B, clusters, N]
    t = _mlp(a.token_features, t).float()
    p = get_matching_probs(p, a.dust_bin.float(), 3).exp()[:, :-1, :]
    v = torch.cat(
        [F.normalize(t, p=2, dim=-1), F.normalize(torch.einsum("bdn,bkn->bdk", f, p), p=2, dim=1).flatten(1)], -1
    )
    v = F.normalize(v, p=2, dim=-1)
    return F.normalize(agg.linear(v.to(agg.linear.weight.dtype)).float(), p=2, dim=-1)


class _MegaLoc(Runner):
    """Resize, normalise, backbone, aggregator."""

    Params = Params
    batch = None
    agg: Aggregator

    def _resize(self, image: torch.Tensor) -> torch.Tensor:
        s = self.ctx.params.size
        x = F.interpolate(image, (s, s), mode="bilinear", align_corners=False, antialias=True)
        return scale(x, self.ctx.io)

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Descriptors `[B, dim]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        return {"descriptor": self._descriptor(self._resize(image))}

    def module(self) -> torch.nn.Module:
        """The network with its pre-processing: `[B, 3, H, W]` in `[0, 1]` -> descriptors `[B, dim]`."""
        return Forward(self.agg, lambda _net, image: self(image)["descriptor"])


class Portable(_MegaLoc):
    """Plain PyTorch, float32 or float16 (SDPA attention; Sinkhorn in float32)."""

    dtypes = ("float32", "float16")
    compiled_methods = ("_descriptor",)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        vit, agg = _networks(ctx)
        sdpa_attention(vit)
        if self.dtype != torch.float32:  # the position table stays fp32, as upstream interpolates it
            pos = vit.pos_embed.detach().clone()
            vit.to(self.dtype).pos_embed = torch.nn.Parameter(pos, requires_grad=False)
        self.vit = vit.to(ctx.device)
        self.agg = agg.to(ctx.device, self.dtype)

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        t = vit.prepare_tokens_with_masks(x.to(self.dtype))
        for blk in vit.blocks:
            t = blk(t)
        t = vit.norm(t)
        return _aggregate(self.agg, t[:, 1:], t[:, 0])


# CUDA graphs up to this many network inputs per call (322 x 322: launch bound up to a few images)
_GRAPH_BATCH = 8  # fixed: launch-bound threshold, not architecture
_GRAPHS = 4  # fixed: graph cache policy


class Kernels(_MegaLoc):
    """fastbb + dpt_fused ViT-B (fp16, flash attention, fused block GEMMs); fp16 aggregator GEMMs, fp32 Sinkhorn.

    Batches of up to `_GRAPH_BATCH` images replay a CUDA graph per batch size (captured on its second call);
    `release_graphs()` frees them.
    """

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, convert it to fp16, move it to the device."""
        super().__init__(ctx)
        vit, agg = _networks(ctx)
        self.fast = FastViT(vit, "fh", None, fused=dpt_fused.load(ctx.device))  # ViT-B: fp32 q k^T
        self.fast.vit, self.fast.pos_fp32 = vit.to(ctx.device), self.fast.pos_fp32.to(ctx.device)
        self.agg = agg.half().to(ctx.device)
        s = ctx.params.size
        self.graphs = SmallShapeGraphs(self._run, pixels=_GRAPH_BATCH * s * s, graphs=_GRAPHS)

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        fast = self.fast
        with torch.autocast("cuda", enabled=False):
            with fp16_accumulate():
                t = fast.tokens(x)
                for blk in fast.blocks:
                    t = fast.block(blk, t)
                t = fast.vit.norm(t)
            return _aggregate(self.agg, t[:, 1:], t[:, 0])  # fp32 accumulation: the last GEMM sums 16640 terms

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        return self.graphs(x)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
