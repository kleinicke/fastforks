"""DRCT runners: the network of ming053l/DRCT and `drct/models/drct_model.py`'s test call (reflect pad, `net_g(img)`).

`DRCTModel.pre_process` reflect-pads the input bottom / right to a multiple of the window size; `pad` does the same.
Upstream builds the shifted-window mask on the CPU on every call at sizes other than the training one and gathers the
relative position bias on every call; here the mask is made on the device once per size, the bias once at load, and
the window attention is one `scaled_dot_product_attention` (`fastforks.shared.swin_sr`). The dense layers' head sizes
(30, 53, 122, 46, 77 for 180 channels and 6 heads) are zero-padded to multiples of 8 in the weights so the fused
kernel runs. Each residual dense group turns its tokens into a feature map and back around every 1 x 1 `adjust` conv
(`PatchUnEmbed`, conv, `PatchEmbed`: two transposes per layer); here the 1 x 1 convs are `F.linear` on the tokens
(the same products). The input mean is a buffer (upstream: a plain attribute, `type_as` per call).

float16 needs no rescaling: the largest group output on the golden image is 280 (DRCT) / 951 (DRCT-L).
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
from fastforks._core.patch import override
from fastforks.kernels import cudnn_hacc
from fastforks.restoration._runner import Crop, Forward, Runner
from fastforks.shared.swin_sr import (
    checkpoint_upsampler,
    pad_heads,
    patch_embed,
    relative_bias,
    shift_mask,
    window_attention,
)

from .net import DRCT
from .net.drct_arch import RDG, PatchEmbed, SwinTransformerBlock, WindowAttention

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` (the `network_g` section of the test options)."""

    scale: int  # 4
    embed_dim: int  # 180
    heads: tuple[int, ...]  # attention heads per residual dense group (6 groups: DRCT, 12: DRCT-L)
    img_size: int  # training patch size (the size of the stored attention masks)
    key: str  # checkpoint key ("params_ema")
    window: int  # 16: attention window (all released models)


class _NoMask:
    """An unshifted block's mask: upstream's is all zeros and gets `.to(device)`; here attention runs without one."""

    def to(self, *_: object) -> None:
        return None


_NO_MASK = _NoMask()


def _mask(m: SwinTransformerBlock, x_size: tuple[int, int]) -> torch.Tensor | _NoMask:
    """`SwinTransformerBlock.calculate_mask` made on the device, once per size (upstream: all zeros if unshifted)."""
    if not m.shift_size:
        return _NO_MASK
    return shift_mask(*x_size, m.window_size, m.shift_size, m.norm1.weight.device)


def _attention(m: WindowAttention, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """`WindowAttention.forward` with the bias gathered at load and one SDPA call."""
    return window_attention(x, m.qkv, m.proj, heads=m.num_heads, scale=m.scale, bias=m.bias, mask=mask)  # ty: ignore[invalid-argument-type]


def _rdg(m: RDG, x: torch.Tensor, xsize: tuple[int, int]) -> torch.Tensor:
    """`RDG.forward` with the 1 x 1 `adjust` convs as `F.linear` on the tokens (upstream: on an unembedded map)."""
    feats = [x]
    for k in range(1, 5):
        feats.append(F.leaky_relu(_adjust(m, k, torch.cat(feats, -1) if k > 1 else x, xsize), 0.2))
    return _adjust(m, 5, torch.cat(feats, -1), xsize) * 0.2 + x  # the last `adjust` has no LeakyReLU


def _adjust(m: RDG, k: int, x: torch.Tensor, xsize: tuple[int, int]) -> torch.Tensor:
    """`adjust<k>(swin<k>(x))` of a group, the 1 x 1 conv on the tokens."""
    conv = m.get_submodule(f"adjust{k}")
    weight = conv.get_parameter("weight").flatten(1)
    return F.linear(m.get_submodule(f"swin{k}")(x, xsize), weight, conv.get_parameter("bias"))


class Portable(Runner):
    """Plain PyTorch in the card's dtype."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network, gather the relative position biases, pad the heads, cache the masks per size."""
        super().__init__(ctx)
        p = ctx.params
        with warnings.catch_warnings():  # upstream's torch.meshgrid without `indexing` ("ij", the default) warns
            warnings.filterwarnings("ignore", "torch.meshgrid", UserWarning)
            self.net = DRCT(
                upscale=p.scale, in_chans=3, img_size=p.img_size, window_size=p.window, img_range=1.0,
                depths=[6] * len(p.heads), embed_dim=p.embed_dim, num_heads=list(p.heads), mlp_ratio=2,
                upsampler="pixelshuffle", resi_connection="1conv",
            )  # fmt: skip  # depths: only the stochastic depth rates (training); every group has 5 layers
        load_strict(self.net, ctx.state_dict(key=p.key), f"{ctx.variant.ref} checkpoint")
        for m in self.net.modules():
            if isinstance(m, WindowAttention):
                table, index = m.relative_position_bias_table, m.relative_position_index
                m.register_buffer("bias", relative_bias(table, index))  # ty: ignore[invalid-argument-type]
                del m.relative_position_bias_table, m.relative_position_index
                pad_heads(m.qkv, m.proj, m.num_heads)
                override(m, _attention)
            elif isinstance(m, SwinTransformerBlock):
                override(m, _mask, "calculate_mask")
            elif isinstance(m, RDG):
                override(m, _rdg)
            elif isinstance(m, PatchEmbed):
                override(m, patch_embed)
        mean = self.net.mean
        del self.net.mean
        self.net.register_buffer("mean", mean, persistent=False)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """`DRCTModel.pre_process`: reflect-pad bottom / right to a multiple of the window."""
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

_GROUP = re.compile(r"layers\.(\d+)\.swin1\.attn\.relative_position_bias_table")


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load_file`: a DRCT state dict as a variant and its params (channels, groups, heads); keys unchanged."""
    groups = sorted(int(m.group(1)) for k in sd if (m := _GROUP.match(k)))
    if "conv_first.weight" not in sd or "layers.0.adjust1.weight" not in sd or not groups:
        return None
    embed_dim = sd["conv_first.weight"].shape[0]
    tables = [sd[f"layers.{n}.swin1.attn.relative_position_bias_table"] for n in groups]
    window = (math.isqrt(tables[0].shape[0]) + 1) // 2
    mask = sd.get("layers.0.swin2.attn_mask")
    up = checkpoint_upsampler(sd)
    problems = []
    if sd["conv_first.weight"].shape[1] != 3:  # noqa: PLR2004 - RGB
        problems.append("not 3 input channels")
    if up is None or up[0] != "pixelshuffle":
        problems.append("an upsampler other than pixelshuffle")
    if sd["layers.0.adjust1.weight"].shape[0] != 32:  # noqa: PLR2004 - upstream's fixed growth `gc`
        problems.append("a growth other than 32 channels")
    if sd["layers.0.swin1.mlp.fc1.weight"].shape[0] != 2 * embed_dim:
        problems.append("an MLP ratio other than 2")
    if "absolute_pos_embed" in sd or "conv_after_body.weight" not in sd:
        problems.append("absolute position embedding or an identity residual connection")
    if problems:
        msg = f"DRCT checkpoint with {', '.join(problems)}: not supported"
        raise WeightsError(msg)
    params = {
        "scale": up[1],  # ty: ignore[not-subscriptable]
        "embed_dim": embed_dim,
        "heads": tuple(t.shape[1] for t in tables),
        "img_size": 64 if mask is None else math.isqrt(mask.shape[0]) * window,
        "window": window,
    }
    return ("drct-l-x4" if len(groups) > 6 else "drct-x4"), params, dict(sd)  # noqa: PLR2004 - DRCT-L: 12 groups
