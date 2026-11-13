"""HAT runners: the network of XPixelGroup/HAT and `hat/models/hat_model.py`'s test call (reflect pad, `net_g(img)`).

`HATModel.pre_process` reflect-pads the input bottom / right to a multiple of the window size; `pad` does the same.
Upstream builds the shifted-window mask on the CPU on every call and gathers both relative position biases (the
window attention's and the overlapping cross-attention's) on every call; here the mask is made on the device once per
size, the biases once at load, and both attentions are one `scaled_dot_product_attention` each
(`fastforks.shared.swin_sr`, the head size padded from 30 to 32 on CUDA). `PatchEmbed` returns contiguous tokens
(upstream: a transposed view, which made every residual add strided). The input mean is a buffer (upstream: a plain
attribute, `type_as` per call).

float16: the last residual group's output exceeds the float16 range (820 239 on the golden image, Real-HAT-GAN; its
blocks stay below 25 000), so its final conv and its skip input are divided by `_LAST_SCALE`; only the LayerNorm
`norm` reads it, its `eps` divided by the square (exact).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import override
from fastforks.kernels import cudnn_hacc
from fastforks.restoration._runner import Crop, Forward, Runner
from fastforks.shared.swin_sr import attention, patch_embed, relative_bias, shift_mask, window_attention

from .net import HAT
from .net.hat_arch import OCAB, RHAG, PatchEmbed, WindowAttention, window_partition, window_reverse

if TYPE_CHECKING:
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


def _mask(m: HAT, x_size: tuple[int, int]) -> torch.Tensor:
    """`HAT.calculate_mask` made on the device, once per size."""
    return shift_mask(*x_size, m.window_size, m.shift_size, m.conv_first.weight.device)


def _attention(
    m: WindowAttention, x: torch.Tensor, rpi: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """`WindowAttention.forward` with the bias gathered at load and one SDPA call."""
    del rpi
    return window_attention(x, m.qkv, m.proj, heads=m.num_heads, scale=m.scale, bias=m.bias, mask=mask)  # ty: ignore[invalid-argument-type]


def _ocab(m: OCAB, x: torch.Tensor, x_size: tuple[int, int], rpi: torch.Tensor) -> torch.Tensor:
    """`OCAB.forward` (overlapping cross-attention) with the bias gathered at load and one SDPA call."""
    del rpi
    h, w = x_size
    b, _, c = x.shape
    ws, ow, heads = m.window_size, m.overlap_win_size, m.num_heads
    shortcut = x
    qkv = m.qkv(m.norm1(x).view(b, h, w, c))
    q = window_partition(qkv[..., :c], ws).reshape(-1, ws * ws, heads, c // heads).transpose(1, 2)
    kv = m.unfold(qkv[..., c:].permute(0, 3, 1, 2))  # [b, 2 * c * ow * ow, nW]: k channels, then v channels
    kv = kv.view(b, 2, heads, c // heads, ow * ow, -1).permute(1, 0, 5, 2, 4, 3)
    kv = kv.reshape(2, -1, heads, ow * ow, c // heads)
    bias = m.get_buffer("bias").to(q.dtype)[None].expand(q.shape[0], -1, -1, -1)
    out = attention(q, kv[0], kv[1], bias, m.scale)
    x = window_reverse(out.transpose(1, 2).reshape(-1, ws, ws, c), ws, h, w).view(b, h * w, c)
    x = m.proj(x) + shortcut
    return x + m.mlp(m.norm2(x))


def _prepare(net: HAT) -> None:
    """Gather every block's relative position bias once (upstream: per call), contiguous `PatchEmbed` tokens."""
    sa, oca = net.get_buffer("relative_position_index_SA"), net.get_buffer("relative_position_index_OCA")
    for m in net.modules():
        if isinstance(m, WindowAttention):
            m.register_buffer("bias", relative_bias(m.get_parameter("relative_position_bias_table"), sa))
            del m.relative_position_bias_table
            override(m, _attention)
        elif isinstance(m, OCAB):
            t = m.get_parameter("relative_position_bias_table")[oca.reshape(-1)].view(*oca.shape, -1)
            m.register_buffer("bias", t.permute(2, 0, 1).contiguous())
            del m.relative_position_bias_table
            override(m, _ocab)
        elif isinstance(m, PatchEmbed):
            override(m, patch_embed)


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
        if ctx.dtype == "float16":
            last = self.net.layers[-1]
            with torch.no_grad():
                for t in last.get_submodule("conv").parameters():  # weight, bias
                    t.div_(_LAST_SCALE)
            override(last, _rhag_scaled)
            self.net.norm.eps /= _LAST_SCALE**2  # LayerNorm(x / k, eps / k²) = LayerNorm(x, eps)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))

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
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


class Kernels(Portable):
    """float16, fp16-accumulate GEMMs and convolutions (`cudnn_hacc`)."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if type(m) is torch.nn.Conv2d:
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        with fp16_accumulate():
            return {"image": self.net(image.to(torch.float16))}


RUNNERS = {"portable": Portable, "kernels": Kernels}
