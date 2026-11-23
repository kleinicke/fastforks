"""HAT runners: the network of XPixelGroup/HAT and `hat/models/hat_model.py`'s test call (reflect pad, `net_g(img)`).

`HATModel.pre_process` reflect-pads the input bottom / right to a multiple of the window size; `pad` does the same.
Upstream builds the shifted-window mask on the CPU on every call and gathers both relative position biases (the
window attention's and the overlapping cross-attention's) on every call; here the mask is made on the device once per
size, the biases once at load, and both attentions are one `scaled_dot_product_attention` each
(`fastforks.shared.swin_sr`, the head size padded from 30 to 32 in their weights at load: `pad_heads`). `PatchEmbed`
returns contiguous tokens (upstream: a transposed view, which made every residual add strided). The input mean is a
buffer (upstream: a plain attribute, `type_as` per call).

float16: the last residual group's output exceeds the float16 range (820 239 on the golden image, Real-HAT-GAN; its
blocks stay below 25 000), so its final conv and its skip input are divided by `_LAST_SCALE`; only the LayerNorm
`norm` reads it, its `eps` divided by the square (exact).
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks._core.tracing import in_onnx_export
from fastforks.kernels import cudnn_hacc
from fastforks.kernels.shift_windows import rearrange
from fastforks.restoration._runner import Crop, Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.swin_sr import (
    attention,
    checkpoint_upsampler,
    pad_heads,
    patch_embed,
    patch_unembed,
    relative_bias,
    shift_mask,
    shifted_window_regions,
    window_attention,
)

from .net import HAT
from .net.hat_arch import (
    CAB,
    HAB,
    OCAB,
    RHAG,
    PatchEmbed,
    PatchUnEmbed,
    WindowAttention,
    window_partition,
    window_reverse,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` (the `network_g` section of the test options)."""

    scale: int  # 4
    embed_dim: int  # 180 (HAT) or 144 (HAT-S)
    depths: tuple[int, ...]  # hybrid attention blocks per residual group
    heads: tuple[int, ...]  # attention heads per residual group
    compress_ratio: int  # channel attention block: conv to embed_dim // compress_ratio channels
    squeeze_factor: int  # channel attention squeeze
    img_size: int  # training patch size
    key: str  # checkpoint key ("params_ema")
    window: int  # 16: attention window (all released models)


_MASK_PARTITION_MIN_WINDOWS = 256  # fixed: window count from which per-token mask labels beat a dense mask


def _mask(m: HAT, x_size: tuple[int, int]) -> torch.Tensor:
    """Shifted-window masks, stored as per-token labels for large CUDA calls."""
    h, w = x_size
    window, shift = m.window_size, m.shift_size
    weight = m.conv_first.weight
    if (
        weight.is_cuda
        and weight.dtype == torch.float16
        and (h // window) * (w // window) >= _MASK_PARTITION_MIN_WINDOWS
        and not torch.is_grad_enabled()
        and not (torch.compiler.is_compiling() or torch.jit.is_tracing() or in_onnx_export())
    ):
        return shifted_window_regions(h, w, window, shift, weight.device)
    return shift_mask(h, w, window, shift, weight.device, dtype=weight.dtype)


def _attention(
    m: WindowAttention, x: torch.Tensor, rpi: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Window attention using position bias gathered when the model loads."""
    del rpi
    bias = m.get_buffer("bias")
    return window_attention(x, m.qkv, m.proj, heads=m.num_heads, scale=m.scale, bias=bias, mask=mask)


_OVERLAP_TILE_BYTES = 64 * 2**20
_OVERLAP_MIN_PIXELS = 256 * 256


def _overlap_attention(m: OCAB, q: torch.Tensor, padded: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Pack and attend over overlapping key/value windows a few window rows at a time."""
    h, w = size
    b, twice_c, hp, wp = padded.shape
    ws, ow, heads = m.window_size, m.overlap_win_size, m.num_heads
    c, channels = twice_c // 2, q.shape[-1]
    nr, nc = h // ws, w // ws
    kv = padded.view(b, 2, heads, channels, hp, wp)
    windows = kv.unfold(4, ow, ws).unfold(5, ow, ws)
    rows = max(1, _OVERLAP_TILE_BYTES // (nc * twice_c * ow * ow * padded.element_size()))
    bias = m.get_buffer("bias").to(q.dtype)[None]
    out = q.new_empty((q.shape[0], ws * ws, c))
    for batch in range(b):
        for start in range(0, nr, rows):
            stop = min(start + rows, nr)
            local = windows[batch : batch + 1, :, :, :, start:stop]
            local = local.permute(1, 0, 4, 5, 2, 6, 7, 3).reshape(2, -1, heads, ow * ow, channels)
            first, end = (batch * nr + start) * nc, (batch * nr + stop) * nc
            result = attention(q[first:end], local[0], local[1], bias.expand(end - first, -1, -1, -1), m.scale)
            out[first:end] = result.transpose(1, 2).reshape(end - first, ws * ws, c)
            del local, result
    return out


def _ocab(m: OCAB, x: torch.Tensor, x_size: tuple[int, int], rpi: torch.Tensor) -> torch.Tensor:
    """`OCAB.forward` (overlapping cross-attention) with the bias gathered at load and one SDPA call."""
    del rpi
    h, w = x_size
    b, _, dim = x.shape
    c = m.qkv.out_features // 3  # dim, or heads x the padded head size (`pad_heads`)
    ws, ow, heads = m.window_size, m.overlap_win_size, m.num_heads
    shortcut = x
    qkv = m.qkv(m.norm1(x).view(b, h, w, dim))
    q = window_partition(qkv[..., :c], ws).reshape(-1, ws * ws, heads, c // heads).transpose(1, 2)
    kv_image = qkv[..., c:].permute(0, 3, 1, 2)
    if (
        x.is_cuda
        and x.dtype == torch.float16
        and h * w >= _OVERLAP_MIN_PIXELS
        and not torch.is_grad_enabled()
        and not (torch.compiler.is_compiling() or torch.jit.is_tracing() or in_onnx_export())
    ):
        p = (ow - ws) // 2
        padded = F.pad(kv_image, (p, p, p, p))
        del qkv, kv_image
        out = _overlap_attention(m, q, padded, (h, w))
        del padded, q
    else:
        kv = m.unfold(kv_image)  # [b, 2 * c * ow * ow, nW]: k channels, then v channels
        kv = kv.view(b, 2, heads, c // heads, ow * ow, -1).permute(1, 0, 5, 2, 4, 3)
        kv = kv.reshape(2, -1, heads, ow * ow, c // heads)
        bias = m.get_buffer("bias").to(q.dtype)[None].expand(q.shape[0], -1, -1, -1)
        out = attention(q, kv[0], kv[1], bias, m.scale).transpose(1, 2)
    x = window_reverse(out.reshape(-1, ws, ws, c), ws, h, w).view(b, h * w, c)
    x = m.proj(x) + shortcut
    return x + m.mlp(m.norm2(x))


def _prepare(net: HAT) -> None:
    """Gather every block's relative position bias once (upstream: per call); contiguous `PatchEmbed` tokens.

    `PatchUnEmbed` becomes a channels-last view of the tokens (`patch_unembed`).
    """
    sa, oca = net.get_buffer("relative_position_index_SA"), net.get_buffer("relative_position_index_OCA")
    for m in net.modules():
        if isinstance(m, WindowAttention):
            m.register_buffer("bias", relative_bias(m.get_parameter("relative_position_bias_table"), sa))
            del m.relative_position_bias_table
            pad_heads(m.qkv, m.proj, m.num_heads)
            override(m, _attention)
        elif isinstance(m, OCAB):
            t = m.get_parameter("relative_position_bias_table").detach()[oca.reshape(-1)].view(*oca.shape, -1)
            m.register_buffer("bias", t.permute(2, 0, 1).contiguous())
            del m.relative_position_bias_table
            pad_heads(m.qkv, m.proj, m.num_heads)
            override(m, _ocab)
        elif isinstance(m, PatchEmbed):
            override(m, patch_embed)
        elif isinstance(m, PatchUnEmbed):
            override(m, patch_unembed)


_LAST_SCALE = 64.0  # fixed: float16 headroom of the last group's residual sum, not an upstream parameter


def _rhag_scaled(m: RHAG, x: torch.Tensor, x_size: tuple[int, int], params: dict) -> torch.Tensor:
    """`RHAG.forward` / `_LAST_SCALE` (the final conv's weights are divided at load)."""
    return m.patch_embed(m.conv(m.patch_unembed(m.residual_group(x, x_size, params), x_size))) + x / _LAST_SCALE


class Portable(Runner):
    """Plain PyTorch in the card's dtype."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network, gather the relative position biases, cache the shifted-window masks per size."""
        super().__init__(ctx)
        p = ctx.params
        with warnings.catch_warnings():  # upstream's torch.meshgrid without `indexing` ("ij", the default) warns
            warnings.filterwarnings("ignore", "torch.meshgrid", UserWarning)
            self.net = HAT(
                upscale=p.scale, in_chans=3, img_size=p.img_size, window_size=p.window, compress_ratio=p.compress_ratio,
                squeeze_factor=p.squeeze_factor, conv_scale=0.01, overlap_ratio=0.5, img_range=1.0,
                depths=list(p.depths), embed_dim=p.embed_dim, num_heads=list(p.heads), mlp_ratio=2,
                upsampler="pixelshuffle", resi_connection="1conv",
            )  # fmt: skip
        load_strict(self.net, ctx.state_dict(key=p.key), f"{ctx.variant.ref} checkpoint")
        _prepare(self.net)
        override(self.net, _mask, "calculate_mask")
        mean = self.net.mean
        del self.net.mean
        self.net.register_buffer("mean", mean, persistent=False)
        if mlx_wrap.run_dtype(ctx) == "float16":
            last = self.net.layers[-1]
            with torch.no_grad():
                for t in last.get_submodule("conv").parameters():  # weight, bias
                    t.div_(_LAST_SCALE)
            override(last, _rhag_scaled)
            self.net.norm.eps /= _LAST_SCALE**2  # LayerNorm(x / k, eps / k²) = LayerNorm(x, eps)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))
        if ctx.device.type == "cuda" and ctx.dtype == "float16":
            # channels-last weights: cuDNN's fp16 convolutions skip the NCHW<->NHWC conversion around each conv
            # (256x256 x4 portable peak 477 -> 385 MB, kernels 419 -> 385 MB, same speed; RTX 5060 Ti, Windows)
            self.net.to(memory_format=torch.channels_last)

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """`HATModel.pre_process`: reflect-pad bottom / right to a multiple of the window."""
        h, w = x.shape[-2:]
        win = self.ctx.params.window
        x = F.pad(x, (0, -w % win, 0, -h % win), "reflect")
        return x, Crop(0, 0, h, w)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        return {"image": self.net(image.to(getattr(torch, self.ctx.dtype)))}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype."""
        dtype = getattr(torch, self.ctx.dtype)
        return Forward(self.net, lambda net, image: net(image.to(dtype)))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    # the bias inside cuDNN's conv + bias plan: hat-s-x4 512^2 691 -> 671 ms, vs fp32 mean 4.06e-4 -> 4.13e-4
    # (DAT's convs gained nothing: 1045 -> 1047 ms)
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation, fuse_bias=True)  # ty: ignore[invalid-argument-type]


_CHANNEL_ALIGN = 8  # fixed: cuDNN pads NHWC fp16 convolutions to multiples of 8 channels with an extra copy


def _pad_conv(conv: torch.nn.Conv2d, *, cin: int, cout: int) -> None:
    """Zero-pad `conv`'s weights to `cin` inputs and `cout` outputs (the extra outputs are exact zeros)."""
    w, b = conv.weight.detach(), conv.get_parameter("bias").detach()
    o, i = w.shape[:2]
    conv.weight = torch.nn.Parameter(
        F.pad(w, (0, 0, 0, 0, 0, cin - i, 0, cout - o)).contiguous(memory_format=torch.channels_last),
        requires_grad=False,
    )
    conv.bias = torch.nn.Parameter(F.pad(b, (0, cout - o)), requires_grad=False)
    conv.in_channels, conv.out_channels = cin, cout


def _pad_cab(m: CAB) -> None:
    """Pad the CAB's convolutions to multiples of `_CHANNEL_ALIGN` channels, except its input (the block's norm).

    The padded channels stay exactly zero (zero weights and biases, GELU(0) = 0, a zero map times any attention),
    so `_cab` slices them off as a view. 512x512 on an RTX 5060 Ti: 180->60 + 60->180 convolutions 4.6 -> 3.5 ms.
    """
    up, down, squeeze, expand = (c for c in m.modules() if isinstance(c, torch.nn.Conv2d))  # in forward order
    c, mid, s = up.in_channels, up.out_channels, squeeze.out_channels
    pc, pmid = -(-c // _CHANNEL_ALIGN) * _CHANNEL_ALIGN, -(-mid // _CHANNEL_ALIGN) * _CHANNEL_ALIGN
    _pad_conv(up, cin=c, cout=pmid)
    _pad_conv(down, cin=pmid, cout=pc)
    _pad_conv(squeeze, cin=pc, cout=s)
    _pad_conv(expand, cin=s, cout=pc)
    m.channels = c  # ty: ignore[invalid-assignment] - the real channel count for `_cab`
    override(m, _cab)


def _cab(m: CAB, x: torch.Tensor) -> torch.Tensor:
    """`CAB.forward` on the padded convolutions of `_pad_cab`: the real channels as a view."""
    return m.cab(x)[:, : m.channels]


def _shifted_block(
    m: HAB, x: torch.Tensor, x_size: tuple[int, int], rpi_sa: torch.Tensor, attn_mask: torch.Tensor
) -> torch.Tensor:
    """Hybrid attention with (shifted) windows produced and consumed in one pass per direction."""
    h, w = x_size
    normalized = m.norm1(x)
    windows = rearrange(normalized, x_size, m.window_size, m.shift_size)
    att = m.attn(windows, rpi=rpi_sa, mask=attn_mask if m.shift_size else None)
    att = rearrange(att, x_size, m.window_size, m.shift_size, reverse=True)
    conv = m.conv_block(normalized.unflatten(1, (h, w)).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
    x = x + m.drop_path(att) + conv * m.conv_scale
    return x + m.drop_path(m.mlp(m.norm2(x)))


class Kernels(Portable):
    """float16, fp16-accumulate GEMMs and convolutions (`cudnn_hacc`), channel-aligned CABs (`_pad_cab`)."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in list(self.net.modules()):
            if isinstance(m, HAB):
                override(m, _shifted_block)
            if isinstance(m, CAB):
                _pad_cab(m)
            if is_base(m, torch.nn.Conv2d):
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        with fp16_accumulate():
            return {"image": self.net(image.to(torch.float16))}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable),
}

_BLOCK = re.compile(r"layers\.(\d+)\.residual_group\.blocks\.(\d+)\.")


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load(path)`: a HAT state dict as a variant and its params (channels, depths, heads, CAB ratios)."""
    blocks = [(int(m.group(1)), int(m.group(2))) for k in sd if (m := _BLOCK.match(k))]
    if "conv_first.weight" not in sd or "layers.0.residual_group.overlap_attn.qkv.weight" not in sd or not blocks:
        return None
    layers = 1 + max(i for i, _ in blocks)
    g = "layers.0.residual_group."
    embed_dim = sd["conv_first.weight"].shape[0]
    window = (math.isqrt(sd[g + "blocks.0.attn.relative_position_bias_table"].shape[0]) + 1) // 2
    up = checkpoint_upsampler(sd)
    problems = []
    if sd["conv_first.weight"].shape[1] != 3:  # noqa: PLR2004 - RGB
        problems.append("not 3 input channels")
    if up is None or up[0] != "pixelshuffle":
        problems.append("an upsampler other than pixelshuffle")
    if sd[g + "blocks.0.mlp.fc1.weight"].shape[0] != 2 * embed_dim:
        problems.append("an MLP ratio other than 2")
    if math.isqrt(sd[g + "overlap_attn.relative_position_bias_table"].shape[0]) != window + window * 3 // 2 - 1:
        problems.append("an overlap ratio other than 0.5")
    if "absolute_pos_embed" in sd or g + "conv.0.weight" in sd:
        problems.append("absolute position embedding or 3conv residual connections")
    if problems:
        msg = f"HAT checkpoint with {', '.join(problems)}: not supported"
        raise WeightsError(msg)
    tables = [sd[f"layers.{n}.residual_group.blocks.0.attn.relative_position_bias_table"] for n in range(layers)]
    params = {
        "scale": up[1],  # ty: ignore[not-subscriptable]
        "embed_dim": embed_dim,
        "depths": tuple(1 + max(j for i, j in blocks if i == n) for n in range(layers)),
        "heads": tuple(t.shape[1] for t in tables),
        "compress_ratio": embed_dim // sd[g + "blocks.0.conv_block.cab.0.weight"].shape[0],
        "squeeze_factor": embed_dim // sd[g + "blocks.0.conv_block.cab.3.attention.1.weight"].shape[0],
        "window": window,
    }
    return ("hat-s-x4" if embed_dim < 180 else "hat-x4"), params, dict(sd)  # noqa: PLR2004 - HAT-S: 144 channels
