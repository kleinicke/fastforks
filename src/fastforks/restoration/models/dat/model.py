"""DAT runners: the network of zhengchen1999/DAT and `basicsr/models/dat_model.py`'s test call (`net_g(img)`, no tile).

The network pads its window attention's q / k / v with zeros to a multiple of the long window side itself, so there is
no input padding. Upstream evaluates every spatial attention's dynamic position bias MLP (`DynamicPosBias`) on every
call, gathers the bias from it, and builds the shifted-window masks on the CPU on every call for sizes other than the
training size; here the bias is made once at load (the MLP's input is a constant), the masks on the device once per
size, and the window attention is one `scaled_dot_product_attention` call (`fastforks.shared.swin_sr.attention`: the
head size padded to a multiple of 8 on CUDA). Each eval BatchNorm (the attentions' interaction branches) is folded into
the conv before it. The convolutions inside the blocks read channels-last views of the tokens and the tokens stay
contiguous (upstream: a transposed copy before and after every conv, 45 % of the time at 512 x 512). The input mean
is a buffer (upstream: a plain attribute, `type_as` per call).

float16: the last residual group's output exceeds the float16 range (299 266 on the golden image, `dat-x4`; its blocks
stay below 3 400), so its final conv and its skip input are divided by `_LAST_SCALE`; only the LayerNorm `norm` reads
it, its `eps` divided by the square (exact).
"""

from __future__ import annotations

import functools
import math
import re
import types
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import override
from fastforks.kernels import cudnn_hacc
from fastforks.restoration._runner import Crop, Forward, Runner
from fastforks.shared.swin_sr import attention, checkpoint_upsampler

from .net import DAT
from .net.dat_arch import (
    Adaptive_Channel_Attention,
    Adaptive_Spatial_Attention,
    ResidualGroup,
    Spatial_Attention,
    SpatialGate,
    _Tokens,
    windows2img,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` (the `network_g` section of the test options)."""

    scale: int  # 4
    embed_dim: int  # 180 (DAT, DAT-2) or 60 (DAT-light)
    depths: tuple[int, ...]  # blocks per residual group
    heads: tuple[int, ...]  # attention heads per residual group
    split: tuple[int, int]  # rectangle window (8, 32)
    expansion: float  # feed-forward hidden / embed_dim
    resi: str  # residual group conv: "1conv" or "3conv"
    upsampler: str  # "pixelshuffle" or "pixelshuffledirect" (DAT-light)
    img_size: int  # training patch size (the size of the masks upstream registers)
    key: str  # checkpoint key ("params")


@functools.lru_cache(maxsize=32)
def _masks(
    split: tuple[int, int], shift: tuple[int, int], h: int, w: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """`Adaptive_Spatial_Attention.calculate_mask` (both branches' 0 / -100 masks) on the device, once per size."""
    m = types.SimpleNamespace(split_size=list(split), shift_size=list(shift))
    return tuple(t.to(device) for t in Adaptive_Spatial_Attention.calculate_mask(m, h, w))  # ty: ignore[invalid-argument-type]


def _mask(m: Adaptive_Spatial_Attention, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _masks(tuple(m.split_size), tuple(m.shift_size), h, w, m.qkv.weight.device)


def _windows(m: Spatial_Attention, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`Spatial_Attention.im2win` straight from the tokens, one copy (upstream: through a transposed image, three)."""
    hs, ws, heads = m.H_sp, m.W_sp, m.num_heads
    x = x.unflatten(1, (h // hs, hs, w // ws, ws)).unflatten(-1, (heads, -1))  # b, h/hs, hs, w/ws, ws, heads, d
    return x.permute(0, 1, 3, 5, 2, 4, 6).reshape(-1, heads, hs * ws, x.shape[-1])


def _spatial(m: Spatial_Attention, qkv: torch.Tensor, h: int, w: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """`Spatial_Attention.forward` with the position bias made at load and one SDPA call."""
    q, k, v = (_windows(m, t, h, w) for t in qkv.unbind(0))
    bw = q.shape[0]
    add = m.get_buffer("bias").to(q.dtype)[None]
    if mask is not None:
        nw = mask.shape[0]
        add = add + mask.to(q.dtype)[:, None]
        if bw != nw:
            add = add.repeat(bw // nw, 1, 1, 1)
    out = attention(q, k, v, add.expand(bw, -1, -1, -1), m.scale)
    return windows2img(out.transpose(1, 2).reshape(bw, m.H_sp * m.W_sp, -1), m.H_sp, m.W_sp, h, w)


def _image(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Tokens `[b, h * w, c]` as a channels-last `[b, c, h, w]` view (upstream: a transposed copy)."""
    return x.unflatten(1, (h, w)).permute(0, 3, 1, 2)


def _tokens(x: torch.Tensor) -> torch.Tensor:
    """`[b, c, h, w]` -> contiguous tokens `[b, h * w, c]` (free for a channels-last tensor)."""
    return x.permute(0, 2, 3, 1).flatten(1, 2).contiguous()


def _tokens_of(m: _Tokens, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG001 - the overridden module
    return _tokens(x)


def _gate(m: SpatialGate, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`SpatialGate.forward` with the depthwise conv on a channels-last view."""
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * _tokens(m.conv(_image(m.norm(x2), h, w)))


def _spatial_block(m: Adaptive_Spatial_Attention, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`Adaptive_Spatial_Attention.forward` with its convolutions on channels-last views (no transposed copies)."""
    b, n, c = x.shape
    qkv = m.qkv(x).reshape(b, -1, 3, c).permute(2, 0, 1, 3)  # 3, b, hw, c
    v = _image(qkv[2], h, w)
    s = max(m.split_size)
    pad_r, pad_b = -w % s, -h % s
    qkv = F.pad(qkv.reshape(3 * b, h, w, c), (0, 0, 0, pad_r, 0, pad_b)).reshape(3, b, -1, c)
    hp, wp = h + pad_b, w + pad_r
    half = c // 2
    if (m.rg_idx % 2 == 0 and m.b_idx > 0 and (m.b_idx - 2) % 4 == 0) or (m.rg_idx % 2 != 0 and m.b_idx % 4 == 0):
        (s0, s1), grid = m.shift_size, qkv.view(3, b, hp, wp, c)
        q0 = torch.roll(grid[..., :half], shifts=(-s0, -s1), dims=(2, 3)).view(3, b, -1, half)
        q1 = torch.roll(grid[..., half:], shifts=(-s1, -s0), dims=(2, 3)).view(3, b, -1, half)
        masks = m.calculate_mask(hp, wp)
        x1 = torch.roll(m.attns[0](q0, hp, wp, mask=masks[0]), shifts=(s0, s1), dims=(1, 2))
        x2 = torch.roll(m.attns[1](q1, hp, wp, mask=masks[1]), shifts=(s1, s0), dims=(1, 2))
    else:
        x1 = m.attns[0](qkv[..., :half], hp, wp)
        x2 = m.attns[1](qkv[..., half:], hp, wp)
    att = torch.cat([x1[:, :h, :w], x2[:, :h, :w]], dim=3).reshape(b, n, c)
    conv = m.dwconv(v)
    channel_map = m.channel_interaction(conv).view(b, 1, c)
    spatial_map = m.spatial_interaction(_image(att, h, w))
    return m.proj(att * torch.sigmoid(channel_map) + _tokens(torch.sigmoid(spatial_map) * conv))


def _channel_block(m: Adaptive_Channel_Attention, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`Adaptive_Channel_Attention.forward` with its convolutions on channels-last views (no transposed copies)."""
    b, n, c = x.shape
    qkv = m.qkv(x).reshape(b, n, 3, m.num_heads, c // m.num_heads).permute(2, 0, 3, 4, 1)  # 3, b, heads, d, n
    q, k, v = F.normalize(qkv[0], dim=-1), F.normalize(qkv[1], dim=-1), qkv[2]
    attn = ((q @ k.transpose(-2, -1)) * m.temperature).softmax(dim=-1)
    att = (attn @ v).permute(0, 3, 1, 2).reshape(b, n, c)
    conv = m.dwconv(_image(qkv[2].permute(0, 3, 1, 2).reshape(b, n, c), h, w))
    channel_map = m.channel_interaction(_image(att, h, w))
    spatial_map = _tokens(m.spatial_interaction(conv))
    return m.proj(att * torch.sigmoid(spatial_map) + _tokens(conv * torch.sigmoid(channel_map)))


def _group(m: ResidualGroup, x: torch.Tensor, x_size: tuple[int, int], skip: float = 1.0) -> torch.Tensor:
    """`ResidualGroup.forward` (the skip scaled by `skip`), its conv on a channels-last view."""
    res = x
    for blk in m.blocks:
        x = blk(x, x_size)
    return (res / skip if skip != 1.0 else res) + _tokens(m.conv(_image(x, *x_size)))


def _prepare(net: DAT) -> None:
    """Make the position biases once (no MLPs), fold the BatchNorms, masks on the device, channels-last convs."""
    for m in net.modules():
        if isinstance(m, Spatial_Attention):
            n = m.H_sp * m.W_sp
            with torch.no_grad():
                bias = m.pos(m.rpe_biases)[m.relative_position_index.view(-1)].view(n, n, -1)  # ty: ignore[call-non-callable]
            del m.pos, m.rpe_biases, m.relative_position_index
            m.register_buffer("bias", bias.permute(2, 0, 1).contiguous())
            override(m, _spatial)
        elif isinstance(m, Adaptive_Spatial_Attention | Adaptive_Channel_Attention):
            for seq in (m.dwconv, m.channel_interaction, m.spatial_interaction):
                i = next(i for i, b in enumerate(seq) if isinstance(b, nn.BatchNorm2d))
                seq[i - 1], seq[i] = fuse_conv_bn_eval(seq[i - 1], seq[i]), nn.Identity()  # ty: ignore[invalid-argument-type]
            override(m, _spatial_block if isinstance(m, Adaptive_Spatial_Attention) else _channel_block)
            if isinstance(m, Adaptive_Spatial_Attention):
                override(m, _mask, "calculate_mask")
        elif isinstance(m, SpatialGate):
            override(m, _gate)
        elif isinstance(m, ResidualGroup):
            override(m, _group)
        elif isinstance(m, _Tokens):
            override(m, _tokens_of)


_LAST_SCALE = 64.0  # fixed: float16 headroom of the last group's residual sum, not an upstream parameter


def _group_scaled(m: ResidualGroup, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
    """`ResidualGroup.forward` / `_LAST_SCALE` (the final conv's weights are divided at load)."""
    return _group(m, x, x_size, _LAST_SCALE)


def _scale_last_group(net: DAT) -> None:
    """Divide the last residual group's output by `_LAST_SCALE` (float16 range), `norm`'s eps by its square."""
    last = net.layers[-1]
    conv = last.get_submodule("conv" if isinstance(last.conv, nn.Conv2d) else f"conv.{len(last.conv) - 1}")  # ty: ignore[invalid-argument-type]
    with torch.no_grad():  # "1conv" or "3conv": the last convolution
        for t in conv.parameters():  # weight, bias
            t.div_(_LAST_SCALE)
    override(last, _group_scaled)
    net.norm.eps /= _LAST_SCALE**2  # LayerNorm(x / k, eps / k²) = LayerNorm(x, eps)


class Portable(Runner):
    """Plain PyTorch in the card's dtype."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network, make the position biases, fold the BatchNorms."""
        super().__init__(ctx)
        p = ctx.params
        with warnings.catch_warnings():  # upstream's torch.meshgrid without `indexing` ("ij", the default) warns
            warnings.filterwarnings("ignore", "torch.meshgrid", UserWarning)
            self.net = DAT(
                upscale=p.scale, in_chans=3, img_size=p.img_size, img_range=1.0, split_size=list(p.split),
                depth=list(p.depths), embed_dim=p.embed_dim, num_heads=list(p.heads), expansion_factor=p.expansion,
                resi_connection=p.resi, upsampler=p.upsampler,
            )  # fmt: skip
        load_strict(self.net, ctx.state_dict(key=p.key), f"{ctx.variant.ref} checkpoint")
        self.net.eval()
        _prepare(self.net)
        mean = self.net.mean
        del self.net.mean
        self.net.register_buffer("mean", mean, persistent=False)
        if ctx.dtype == "float16":
            _scale_last_group(self.net)
        self.net.to(ctx.device, getattr(torch, ctx.dtype))

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """No padding: the network pads its window attention itself (`dat_model.py` test: `net_g(img)`)."""
        return x, Crop(0, 0, *x.shape[-2:])

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        return {"image": self.net(image.to(getattr(torch, self.ctx.dtype)))}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype."""
        dtype = getattr(torch, self.ctx.dtype)
        return Forward(self.net, lambda net, image: net(image.to(dtype)))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


_HACC_MIN_CHANNELS = 64  # fixed: measured, narrower convolutions run slower through cudnn_hacc


class Kernels(Portable):
    """float16, fp16-accumulate GEMMs and dense convolutions of at least 64 channels in and out (`cudnn_hacc`).

    The narrower ones (DAT-light's 60-channel convolutions, the 1 x 1 convolutions of the channel interactions) were
    slower through `cudnn_hacc` (DAT-light 512 x 512: 300 ms and 2.9 GB vs 268 ms and 1.5 GB; DAT unchanged).
    """

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its dense convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if (
                type(m) is torch.nn.Conv2d
                and m.groups == 1
                and min(m.in_channels, m.out_channels) >= _HACC_MIN_CHANNELS
            ):
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        with fp16_accumulate():
            return {"image": self.net(image.to(torch.float16))}


RUNNERS = {"portable": Portable, "kernels": Kernels}

_BLOCK = re.compile(r"layers\.(\d+)\.blocks\.(\d+)\.")


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load_file`: a DAT state dict as a variant and its params (channels, depths, heads, split, FFN, tail)."""
    a = "layers.0.blocks.0.attn.attns.0."
    blocks = [(int(m.group(1)), int(m.group(2))) for k in sd if (m := _BLOCK.match(k))]
    if "conv_first.weight" not in sd or "before_RG.1.weight" not in sd or a + "rpe_biases" not in sd:
        return None
    layers = 1 + max(i for i, _ in blocks)
    embed_dim = sd["conv_first.weight"].shape[0]
    rpe = sd[a + "rpe_biases"]  # the window's relative offsets: -(split - 1) .. split - 1 per axis
    split = (int(rpe[:, 0].max()) + 1, int(rpe[:, 1].max()) + 1)
    mask = next((v for k, v in sd.items() if k.endswith(".attn.attn_mask_0")), None)  # [windows, N, N] at img_size
    up = checkpoint_upsampler(sd)
    if sd["conv_first.weight"].shape[1] != 3 or up is None or up[0] == "nearest+conv":  # noqa: PLR2004 - RGB
        msg = "DAT checkpoint with not 3 input channels or no pixelshuffle upsampler: not supported"
        raise WeightsError(msg)
    hidden = sd["layers.0.blocks.0.ffn.fc1.weight"].shape[0]
    params = {
        "scale": up[1],
        "embed_dim": embed_dim,
        "depths": tuple(1 + max(j for i, j in blocks if i == n) for n in range(layers)),
        "heads": tuple(2 * sd[f"layers.{n}.blocks.0.attn.attns.0.pos.pos3.2.weight"].shape[0] for n in range(layers)),
        "split": split,
        "expansion": hidden // embed_dim if hidden % embed_dim == 0 else hidden / embed_dim,
        "resi": "3conv" if "layers.0.conv.0.weight" in sd else "1conv",
        "upsampler": up[0],
        "img_size": 64 if mask is None else math.isqrt(mask.shape[0] * split[0] * split[1]),
    }
    if up[0] == "pixelshuffledirect":
        return "dat-light-x4", params, dict(sd)
    return ("dat-2-x4" if hidden == 2 * embed_dim else "dat-x4"), params, dict(sd)
