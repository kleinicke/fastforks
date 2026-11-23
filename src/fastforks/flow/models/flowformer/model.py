"""FlowFormer Small runners using the pinned upstream inference architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.patch import override
from fastforks.flow._runner import Forward, Runner
from fastforks.kernels import flow_upsample
from fastforks.shared import libnet, mlx_wrap
from fastforks.shared.cuda_softmax import softmax_rows_

if TYPE_CHECKING:
    from collections.abc import KeysView

    from fastforks.flow._runner import Context

    from .net.FlowFormer.LatentCostFormer.decoder import MemoryDecoder
    from .net.FlowFormer.LatentCostFormer.encoder import CostPerceiverEncoder, MemoryEncoder
    from .net.FlowFormer.LatentCostFormer.gma import Aggregate, Attention
    from .net.FlowFormer.LatentCostFormer.gru import SepConvGRU

RADIUS = 4  # upstream's `encode_flow_token` window: (2 r + 1)^2 = 81 costs per pixel
LOOKUP_PIXELS = 8192  # the window lookup gathers 100 feature vectors per pixel for this many pixels at a time
# fixed: GMA's attention matrix (1/8 pixels squared, float16) is kept up to this size and multiplied once per iteration;
# larger inputs recompute softmax(q k) v with fused attention instead (1080 x 1920: 12.6 ms per iteration instead of
# 5.9 ms, without the 2 GiB matrix)
GMA_MATRIX_BYTES = 1 << 30


@dataclass(frozen=True)
class Params:
    """The released Small architecture and refinement iteration count."""

    library: str
    pe: str
    dropout: float
    encoder_latent_dim: int
    query_latent_dim: int
    cost_latent_input_dim: int
    cost_latent_token_num: int
    cost_latent_dim: int
    arc_type: str
    cost_heads_num: int
    pretrain: bool
    context_concat: bool
    encoder_depth: int
    feat_cross_attn: bool
    patch_size: int
    patch_embed: str
    no_pe: bool
    gma: str
    kernel_size: int
    rm_res: bool
    vert_c_dim: int
    cost_encoder_res: bool
    cnet: str
    fnet: str
    no_sc: bool
    only_global: bool
    add_flow_token: bool
    use_mlp: bool
    vertical_conv: bool
    iters: int

    @property
    def decoder_depth(self) -> int:
        """Upstream's name for refinement iterations."""
        return self.iters

    def keys(self) -> KeysView[str]:
        """Configuration field names for upstream's augmentation-presence check."""
        return vars(self).keys()


def _forward(net: torch.nn.Module, image1: torch.Tensor, image2: torch.Tensor, *, fp16: bool) -> torch.Tensor:
    """Use upstream operations, with optional autocast confined to the network."""
    with torch.autocast(image1.device.type, torch.float16, enabled=fp16):
        return net(image1, image2)[0]


class Portable(Runner):
    """Float32 upstream inference with torch-only reshape helpers."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Load the released Small checkpoint strictly."""
        super().__init__(ctx)
        libnet.timm()
        from .net.FlowFormer.LatentCostFormer.transformer import FlowFormer  # noqa: PLC0415 - optional timm

        self.net = FlowFormer(ctx.params)
        state = ctx.state_dict()
        check_fits(self.net.state_dict(), state, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(state)
        self.net.eval().to(ctx.device)
        self.net.memory_decoder.update_block.gru.fuse()
        self.fp16 = ctx.dtype == "float16"

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the full-resolution flow, excluding upstream's low-resolution auxiliary flow."""
        return {"flow": _forward(self.net, image1, image2, fp16=self.fp16)}

    def module(self) -> torch.nn.Module:
        """Export the same full-resolution flow."""
        fp16 = self.fp16
        return Forward(self.net, lambda net, a, b: _forward(net, a, b, fp16=fp16))


def _gru_fp32_state(gru: SepConvGRU, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute convolution gates in fp16, and update recurrent state in float32."""
    for gate_name, candidate in (("convzr1", gru.convq1), ("convzr2", gru.convq2)):
        gate = cast("torch.nn.Conv2d", getattr(gru, gate_name))
        gates = gate(torch.cat((h, x), 1).half()).float()
        z, r = gates.sigmoid().chunk(2, dim=1)
        q = candidate(torch.cat((r * h, x), 1).half()).float().tanh()
        h = (1 - z) * h + z * q
    return h


def _upsample(_decoder: torch.nn.Module, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return flow_upsample.convex(flow, mask)


class _Volume:
    """Upstream's all-pairs cost volume of one head, kept as its two feature maps: `f1` [B, N, C], `f2` [B, C, H, W].

    The volume (N x N float32, 4 GiB at 1080 x 1920) is never stored: the encoder makes its cost maps a chunk of query
    pixels at a time, and the decoder's windows are dot products with the sampled `f2` (bilinear sampling is linear).
    """

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor) -> None:
        self.f1 = fmap1.flatten(2).transpose(1, 2)
        self.f2 = fmap2
        self.windows: tuple[torch.Tensor, torch.Tensor] | None = None  # the lookup's float16 copies, made once


def _corr(enc: MemoryEncoder, fmap1: torch.Tensor, fmap2: torch.Tensor) -> _Volume:
    """`MemoryEncoder.corr`, deferred (`_Volume`)."""
    assert enc.cfg.cost_heads_num == 1  # noqa: S101 - the released Small configuration
    return _Volume(fmap1, fmap2)


def _encode(m: CostPerceiverEncoder, vol: _Volume, data: dict, context: torch.Tensor | None = None) -> torch.Tensor:
    """Upstream `CostPerceiverEncoder.forward`, the cost maps made and embedded per chunk of query pixels.

    Each chunk's maps (`f1` rows times `f2`), patch embedding and first latent projection run under float16 autocast
    (on CUDA); only the small latent tokens of the chunk are kept, in float32.
    """
    f1, f2 = vol.f1, vol.f2
    b, _, h, w = f2.shape
    n, k = h * w, m.cfg.cost_latent_token_num
    data["cost_maps"] = vol
    pe = m.patch_embed
    pad = (0, -w % pe.patch_size, 0, -h % pe.patch_size)
    x, size = None, None
    for i in range(b):
        for s in range(0, n, pe.chunk):
            with torch.autocast(f2.device.type, torch.float16, enabled=f2.is_cuda):
                maps = F.pad(torch.matmul(f1[i, s : s + pe.chunk], f2[i].flatten(1)).view(-1, 1, h, w), pad)
                emb, size = pe._embed(maps, share_position=True)  # noqa: SLF001 - upstream's per-chunk embedding
                emb = m.input_layer(m.latent_tokens, emb)
            if x is None:
                x = emb.new_empty((b * n, *emb.shape[1:]), dtype=torch.float32)
            x[i * n + s : i * n + s + emb.shape[0]].copy_(emb)
            del maps, emb
    assert x is not None  # noqa: S101 - at least one pixel
    data["H3W3"] = size
    short_cut = x
    for idx, layer in enumerate(m.encoder_layers):
        x = layer(x)
        x = x.view(b, n, k, -1).permute(0, 2, 1, 3).reshape(b * k, n, -1)
        x = m.vertical_encoder_layers[idx](x, (h, w), context)
        x = x.view(b, k, n, -1).permute(0, 2, 1, 3).reshape(b * n, k, -1)
    return x + short_cut if m.cfg.cost_encoder_res is True else x


def _windows(_decoder: MemoryDecoder, vol: _Volume, coords: torch.Tensor) -> torch.Tensor:
    """Upstream `encode_flow_token`: the (2 r + 1)^2 bilinear costs around each pixel's `coords` (zero outside).

    The 2 r + 2 integer neighbours' costs are gathered dot products of `f1` with `f2` (float16 on CUDA), then
    interpolated; the window index is x offset major, as upstream's grid.
    """
    b, c, h, w = vol.f2.shape
    if vol.windows is None:
        dt = torch.float16 if vol.f2.is_cuda else vol.f2.dtype
        vol.windows = vol.f1.to(dt).contiguous(), vol.f2.to(dt).permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    f1, f2 = vol.windows
    side = 2 * RADIUS + 2
    xy = coords.float().flatten(2)
    out = torch.empty(b, h * w, (side - 1) ** 2, device=coords.device)
    steps = torch.arange(side, device=coords.device)
    for i in range(b):
        for s in range(0, h * w, LOOKUP_PIXELS):
            cx, cy = xy[i, 0, s : s + LOOKUP_PIXELS], xy[i, 1, s : s + LOOKUP_PIXELS]
            fx, fy = cx.floor(), cy.floor()
            wx, wy = (cx - fx)[:, None, None], (cy - fy)[:, None, None]
            gx, gy = fx.long()[:, None] - RADIUS + steps, fy.long()[:, None] - RADIUS + steps
            inside = ((gy >= 0) & (gy < h))[:, :, None] & ((gx >= 0) & (gx < w))[:, None, :]
            idx = gy.clamp(0, h - 1)[:, :, None] * w + gx.clamp(0, w - 1)[:, None, :]
            p = cx.shape[0]
            near = f2[i].index_select(0, idx.view(-1)).view(p, side * side, c)
            d = torch.bmm(near, f1[i, s : s + p, :, None]).view(p, side, side).float() * inside  # [pixel, y, x]
            v = (1 - wy) * ((1 - wx) * d[:, :-1, :-1] + wx * d[:, :-1, 1:]) + wy * (
                (1 - wx) * d[:, 1:, :-1] + wx * d[:, 1:, 1:]
            )
            out[i, s : s + p] = v.transpose(1, 2).reshape(p, -1)
    return out.view(b, h, w, -1).permute(0, 3, 1, 2)


def _gma_attention(att: Attention, fmap: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Upstream GMA `Attention.forward` (one head) in float16: the softmax matrix, or for large inputs `(q, k)`."""
    assert att.heads == 1  # noqa: S101 - FlowFormer's GMA
    q, k = att.to_qk(fmap).chunk(2, dim=1)
    q = (att.scale * q).flatten(2).transpose(1, 2).half()
    k = k.flatten(2).transpose(1, 2).half()
    if q.shape[1] ** 2 * q.element_size() <= GMA_MATRIX_BYTES:
        return (softmax_rows_(torch.bmm(q, k.transpose(1, 2)), 32 << 20),)
    return q[:, None].contiguous(), k[:, None].contiguous()


def _gma_aggregate(agg: Aggregate, attention: tuple[torch.Tensor, ...], fmap: torch.Tensor) -> torch.Tensor:
    """Upstream GMA `Aggregate.forward` with `_gma_attention`'s matrix or fused attention, in float16."""
    b, _, h, w = fmap.shape
    v = agg.to_v(fmap).flatten(2).transpose(1, 2).half()
    if len(attention) == 1:
        out = torch.bmm(attention[0], v)
    else:
        q, k = attention
        out = F.scaled_dot_product_attention(q, k, v[:, None].contiguous(), scale=1.0)[:, 0]
    out = out.float().transpose(1, 2).reshape(b, -1, h, w)
    if agg.project is not None:
        out = agg.project(out)
    return fmap + agg.gamma * out


class Kernels(Portable):
    """Fused convex upsampling, and no all-pairs volume or GMA matrix in float32.

    The cost volume is never stored (`_Volume`): its maps are made per chunk for the encoder, which embeds them in
    float16, and the decoder gathers its windows. GMA's attention runs in float16 (`_gma_attention`). 1080 x 1920: peak
    8699 -> 1800 MiB, 3937 -> 3145 ms.
    """

    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Keep the original FP32 aggregation and fuse the last prediction's upsampling."""
        super().__init__(ctx)
        gru = self.net.get_submodule("memory_decoder.update_block.gru")
        for name in ("convzr1", "convq1", "convzr2", "convq2"):
            gru.get_submodule(name).half().to(memory_format=torch.channels_last)
        override(gru, _gru_fp32_state)
        override(self.net.memory_decoder, _upsample, "upsample_flow")
        override(self.net.memory_encoder, _corr, "corr")
        override(self.net.memory_encoder.cost_perceiver_encoder, _encode)
        override(self.net.memory_decoder, _windows, "encode_flow_token")
        override(self.net.memory_decoder.att, _gma_attention)
        override(self.net.get_submodule("memory_decoder.update_block.aggregator"), _gma_aggregate)


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable),
}
