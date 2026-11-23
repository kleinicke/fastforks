"""FlowFormer Small runners using the pinned upstream inference architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    from .net.FlowFormer.LatentCostFormer.encoder import CostPerceiverEncoder, MemoryEncoder, PatchEmbed
    from .net.FlowFormer.LatentCostFormer.gma import Aggregate, Attention
    from .net.FlowFormer.LatentCostFormer.gru import BasicMotionEncoder, SepConvGRU

RADIUS = 4  # fixed: upstream's `encode_flow_token` window: (2 r + 1)^2 = 81 costs per pixel
LOOKUP_PIXELS = 8192  # fixed: the window lookup gathers 100 feature vectors per pixel for this many pixels at a time
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
    """Compute convolution gates in fp16, and update recurrent state in float32.

    Both passes read one channels-last fp16 `[h | x]` buffer: `x` is cast into it once, then `h` and `r h` are
    written over its first channels (upstream concatenates and casts four times). `(1 - z) h + z q` as `lerp`.
    """
    b, hd, height, width = h.shape
    hx = torch.empty(
        (b, hd + x.shape[1], height, width), dtype=torch.float16, device=h.device, memory_format=torch.channels_last
    )
    hx[:, hd:] = x
    for gate_name, candidate in (("convzr1", gru.convq1), ("convzr2", gru.convq2)):
        gate = cast("torch.nn.Conv2d", getattr(gru, gate_name))
        hx[:, :hd] = h
        z, r = gate(hx).float().sigmoid_().chunk(2, dim=1)
        torch.mul(r, h, out=hx[:, :hd])
        h = torch.lerp(h, candidate(hx).float().tanh_(), z)
    return h


def _half_cl(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16, memory_format=torch.channels_last)


def _motion_encoder(enc: BasicMotionEncoder, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
    """Upstream `BasicMotionEncoder.forward` with fp16 channels-last convs; the features out in float32.

    `convf1` stays float32: the flow it reads reaches hundreds of pixels at 1080p, where float16 steps are 1/8 px.
    """
    cor = torch.relu_(enc.convc2(torch.relu_(enc.convc1(_half_cl(corr)))))
    flo = torch.relu_(enc.convf2(_half_cl(torch.relu_(enc.convf1(flow)))))
    out = torch.relu_(enc.conv(torch.cat([cor, flo], dim=1)))
    return torch.cat([out.float(), flow], dim=1)


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


@dataclass
class _Embed:
    """What `_patch_embed` reads, made once by `_prepare_patch_embed`."""

    convs: tuple[torch.nn.Conv2d, torch.nn.Conv2d, torch.nn.Conv2d]  # `proj`, fp16 channels last
    ffn0: torch.Tensor  # `ffn_with_coord.0`, fp32 `[2 C, 2 C]`: columns `:C` the tokens', `C:` the position's
    ffn0_x: torch.Tensor  # its token half, fp16 `[C, 2 C]` (transposed)
    ffn0_bias: torch.Tensor  # fp32
    ffn2: tuple[torch.Tensor, torch.Tensor]  # `ffn_with_coord.2` as a linear, fp16
    norm: tuple[torch.Tensor, torch.Tensor]  # the LayerNorm's weight and bias, fp16
    position: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)  # `_patch_position`, per grid


def _patch_position(pe: PatchEmbed, e: _Embed, h: int, w: int, device: torch.device) -> torch.Tensor:
    """`ffn_with_coord.0`'s position half for an `h` x `w` patch grid: `W_pos pe + b`, `[h * w, 2 C]` float16."""
    if (h, w) not in e.position:
        from .net.FlowFormer.LatentCostFormer.attention import (  # noqa: PLC0415 - needs timm (libnet)
            ExpPositionEmbeddingSine,
            LinearPositionEmbeddingSine,
        )
        from .net.utils import coords_grid  # noqa: PLC0415 - next to the import above

        coord = coords_grid(1, h, w, device=device) * pe.patch_size + pe.patch_size / 2
        coord = coord.view(1, 2, -1).permute(0, 2, 1)
        enc = (LinearPositionEmbeddingSine if pe.pe == "linear" else ExpPositionEmbeddingSine)(coord, dim=pe.dim)[0]
        e.position[h, w] = torch.addmm(e.ffn0_bias, enc.float(), e.ffn0[:, pe.dim :].t()).half()
    return e.position[h, w]


def _patch_embed(pe: PatchEmbed, x: torch.Tensor, *, share_position: bool = False) -> tuple[torch.Tensor, tuple]:
    """Upstream `PatchEmbed._embed` (`share_position`) on fp16 channels-last activations; tokens `[B, N, 2 C]` fp16.

    The convs run channels last (cuDNN converted every NCHW map to NHWC and back) with in-place ReLUs. The first 1 x 1
    conv of `ffn_with_coord` is split at its concatenation: its position half is the same for every map
    (`_patch_position`, the bias of a GEMM over the maps' tokens), and both 1 x 1 convs are GEMMs over the tokens,
    which channels last already are. The LayerNorm reads fp16 with fp16 weights (autocast ran it on an fp32 copy).
    """
    assert share_position  # noqa: S101 - the kernels encoder's only call
    e: _Embed = pe.__dict__["_kernels"]
    conv0, conv1, conv2 = e.convs
    y = torch.relu_(F.conv2d(x.half(), conv0.weight, conv0.bias, conv0.stride, conv0.padding))
    y = torch.relu_(F.conv2d(y, conv1.weight, conv1.bias, conv1.stride, conv1.padding))
    y = F.conv2d(y, conv2.weight, conv2.bias, conv2.stride, conv2.padding)
    b, c, h, w = y.shape
    tokens = y.permute(0, 2, 3, 1).reshape(b, h * w, c)  # a view: `y` is channels last
    position = _patch_position(pe, e, h, w, y.device)
    y = torch.baddbmm(position.expand(b, -1, -1), tokens, e.ffn0_x.expand(b, -1, -1)).relu_()
    y = F.linear(y, *e.ffn2)
    with torch.autocast(y.device.type, enabled=False):
        return F.layer_norm(y, (y.shape[-1],), *e.norm, pe.norm.eps), (h, w)


def _prepare_patch_embed(pe: PatchEmbed) -> None:
    """fp16 channels-last conv weights and the split `ffn_with_coord.0` for `_patch_embed`."""
    convs = tuple(m for m in pe.proj if isinstance(m, torch.nn.Conv2d))
    for m in convs:
        m.half().to(memory_format=torch.channels_last)
    ffn0, ffn2 = (cast("torch.nn.Conv2d", pe.ffn_with_coord[i]) for i in (0, 2))
    norm = pe.norm
    w0 = ffn0.weight.detach().float().flatten(1)
    conv0, conv1, conv2 = convs  # patch size 8
    assert ffn0.bias is not None and ffn2.bias is not None  # noqa: S101, PT018 - upstream's convs have biases
    pe.__dict__["_kernels"] = _Embed(
        convs=(conv0, conv1, conv2),
        ffn0=w0,
        ffn0_x=w0[:, : cast("int", pe.dim)].t().half().contiguous(),
        ffn0_bias=ffn0.bias.detach().float(),
        ffn2=(ffn2.weight.detach().flatten(1).half(), ffn2.bias.detach().half()),
        norm=(norm.weight.detach().half(), norm.bias.detach().half()),
    )
    override(pe, _patch_embed, "_embed")


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
    """Upstream GMA `Aggregate.forward` with `_gma_attention`'s matrix or fused attention, in float16.

    `to_v` is an fp16 channels-last conv: its `[B, H W, C]` tokens are a view, and so is the result as an image.
    """
    b, _, h, w = fmap.shape
    v = agg.to_v(_half_cl(fmap)).flatten(2).transpose(1, 2)  # contiguous: channels last
    if len(attention) == 1:
        out = torch.bmm(attention[0], v)
    else:
        q, k = attention
        out = F.scaled_dot_product_attention(q, k, v[:, None], scale=1.0)[:, 0]
    out = out.view(b, h, w, -1).permute(0, 3, 1, 2)
    if agg.project is not None:
        out = agg.project(out.float())
    return torch.addcmul(fmap, agg.gamma, out)


class Kernels(Portable):
    """Fused convex upsampling, and no all-pairs volume or GMA matrix in float32.

    The cost volume is never stored (`_Volume`): its maps are made per chunk for the encoder, which embeds them in
    float16, and the decoder gathers its windows. GMA's attention runs in float16 (`_gma_attention`). 1080 x 1920: peak
    8699 -> 1800 MiB, 3937 -> 3145 ms.
    """

    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Run the motion encoder and GMA values in float16 and fuse the last prediction's upsampling."""
        super().__init__(ctx)
        gru = self.net.get_submodule("memory_decoder.update_block.gru")
        for name in ("convzr1", "convq1", "convzr2", "convq2"):
            gru.get_submodule(name).half().to(memory_format=torch.channels_last)
        override(gru, _gru_fp32_state)
        block = self.net.memory_decoder.update_block
        # The flow head stays float32: its updates accumulate over the iterations (1080p: mean 0.00072 -> 0.00062 px).
        block.encoder.half().to(memory_format=torch.channels_last)
        block.encoder.convf1.float()
        override(block.encoder, _motion_encoder)
        block.get_submodule("aggregator.to_v").half().to(memory_format=torch.channels_last)
        override(self.net.memory_decoder, _upsample, "upsample_flow")
        override(self.net.memory_encoder, _corr, "corr")
        override(self.net.memory_encoder.cost_perceiver_encoder, _encode)
        _prepare_patch_embed(self.net.memory_encoder.cost_perceiver_encoder.patch_embed)
        override(self.net.memory_decoder, _windows, "encode_flow_token")
        override(self.net.memory_decoder.att, _gma_attention)
        override(self.net.get_submodule("memory_decoder.update_block.aggregator"), _gma_aggregate)
        from .net.FlowFormer.LatentCostFormer.twins import GlobalSubSampleAttnRPE  # noqa: PLC0415 - needs timm

        for m in self.net.modules():
            if isinstance(m, GlobalSubSampleAttnRPE):
                m.__dict__["_sdpa"] = True  # read by its forward


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable),
}
