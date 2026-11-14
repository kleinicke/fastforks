"""Swin2SR runners: the network of mv-lab/swin2sr and `main_test_swin2sr.py`'s call (mirror padding, `model(lq)`).

`main_test_swin2sr.py` pads the input by mirroring (`cat(x, flip(x))`) to the next multiple of the window size plus one
window when it is already a multiple (as SwinIR's script); `pad` does the same. Upstream's Swin V2 blocks run the
continuous position bias MLP (`cpb_mlp`, 2 -> 512 -> heads) and gather its output on every call, concatenate the q / v
biases per call and rebuild the shifted-window mask on the CPU for every size but the training one. Here the bias
`16 * sigmoid(cpb_mlp(table))[index]` is computed once at load, the q / v biases become the `qkv` bias, the mask is made
on the device once per size, and the cosine attention is one `scaled_dot_product_attention` (q and k L2-normalised,
q times the clamped logit scale, `scale=1`) with the head dimension (30) zero-padded to 32 in the weights so the fused
kernel runs (`fastforks.shared.swin_sr`). Each residual group's 1 x 1 patch-embedding conv (Swin2SR adds one to
SwinIR's `PatchEmbed`) is folded into the 3 x 3 conv before it at load; `PatchEmbed` returns contiguous tokens. The
last group's output is scaled by 1 / 32 before the final `LayerNorm` (its eps by 1 / 32^2): unscaled it overflows
float16 on DIV2K x2 images (upstream's `.half()` too). The input mean is a buffer.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.kernels import cudnn_hacc
from fastforks.restoration._runner import Crop, Forward, Runner
from fastforks.shared.swin_sr import checkpoint_upsampler, pad_heads, shift_mask, window_attention

from .net import Swin2SR
from .net.network_swin2sr import RSTB, PatchEmbed, SwinTransformerBlock, WindowAttention

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` (`main_test_swin2sr.py` `define_model`)."""

    scale: int  # 4 or 2
    embed_dim: int  # 180
    depths: tuple[int, ...]  # Swin V2 layers per residual group
    heads: tuple[int, ...]  # attention heads per residual group
    upsampler: str  # "pixelshuffle" (classical) or "nearest+conv" (real-world)
    resi_connection: str  # "1conv" (released models) or "3conv"
    img_size: int  # training patch size (the size of the stored attention masks)
    window: int  # 8: attention window (all released models)


class _NoMask:
    """An unshifted block's mask: upstream's is all zeros and gets `.to(device)`; here attention runs without one."""

    def to(self, *_: object) -> None:
        return None


_NO_MASK = _NoMask()


def _mask(m: SwinTransformerBlock, x_size: tuple[int, int]) -> torch.Tensor | _NoMask:
    """`SwinTransformerBlock.calculate_mask` made on the device, once per size (upstream: all zeros if unshifted)."""
    if not m.shift_size:
        return _NO_MASK
    return shift_mask(*x_size, m.window_size, m.shift_size, m.norm1.weight.device, dtype=m.norm1.weight.dtype)


def _attention(m: WindowAttention, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """`WindowAttention.forward` (cosine attention) with the bias and logit scale made at load and one SDPA call."""
    bias: torch.Tensor = m.bias  # ty: ignore[invalid-assignment]
    cosine: torch.Tensor = m.cosine  # ty: ignore[invalid-assignment]
    return window_attention(x, m.qkv, m.proj, heads=m.num_heads, scale=1.0, bias=bias, mask=mask, cosine=cosine)


def _fold(m: WindowAttention) -> None:
    """Precompute the position bias and logit scale, fold the q / v biases into `qkv`, drop the per-call parts."""
    coords: torch.Tensor = m.relative_coords_table  # ty: ignore[invalid-assignment]
    index: torch.Tensor = m.relative_position_index  # ty: ignore[invalid-assignment]
    with torch.no_grad():
        table = m.cpb_mlp(coords).view(-1, m.num_heads)
        n = index.shape[0]
        bias = table[index.view(-1)].view(n, n, -1).permute(2, 0, 1)
        m.register_buffer("bias", 16 * torch.sigmoid(bias).contiguous())
        m.register_buffer("cosine", torch.clamp(m.logit_scale, max=math.log(1.0 / 0.01)).exp())
        qkv = torch.nn.Linear(m.qkv.in_features, m.qkv.out_features, bias=m.q_bias is not None)
        qkv.weight.copy_(m.qkv.weight)
        if m.q_bias is not None and m.v_bias is not None and qkv.bias is not None:
            qkv.bias.copy_(torch.cat((m.q_bias, torch.zeros_like(m.v_bias), m.v_bias)))
        m.qkv = qkv.requires_grad_(requires_grad=False)
    for name in ("cpb_mlp", "relative_coords_table", "relative_position_index", "logit_scale", "q_bias", "v_bias"):
        delattr(m, name)
    pad_heads(m.qkv, m.proj, m.num_heads)


def _patch_embed(m: PatchEmbed, x: torch.Tensor) -> torch.Tensor:
    """`PatchEmbed.forward` (Swin2SR's has a 1 x 1 `proj` conv; None where folded) with contiguous tokens."""
    if m.proj is not None:
        x = m.proj(x)
    x = x.flatten(2).transpose(1, 2).contiguous()
    return x if m.norm is None else m.norm(x)


def _fold_proj(m: RSTB) -> None:
    """Fold the group's 1 x 1 `patch_embed.proj` into the 3 x 3 `conv` before it (`1conv`): one conv instead of two."""
    conv, proj = m.conv, m.patch_embed.proj
    if not isinstance(conv, torch.nn.Conv2d) or conv.bias is None or proj.bias is None:
        return
    p = proj.weight[:, :, 0, 0]
    with torch.no_grad():
        conv.weight.copy_(torch.einsum("oc,cikl->oikl", p, conv.weight))
        conv.bias.copy_(p @ conv.bias + proj.bias)
    m.patch_embed.proj = None  # ty: ignore[invalid-assignment]


_LAST_SCALE = 1 / 32  # power of two: exact in the weights


def _scaled_rstb(m: RSTB, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
    """`RSTB.forward` of the last group, output times `_LAST_SCALE` (the conv path's weights carry the factor)."""
    return m.patch_embed(m.conv(m.patch_unembed(m.residual_group(x, x_size), x_size))) + x * _LAST_SCALE


def _scale_last(m: RSTB) -> None:
    """Scale the last group's output down: it reaches ~7e4 on DIV2K x2 images (float16 overflow -> NaN).

    The `LayerNorm` after the groups is scale-invariant once its eps is scaled too, so the output is unchanged.
    """
    last = m.patch_embed.proj
    if last is None:
        last = m.conv[-1] if isinstance(m.conv, torch.nn.Sequential) else m.conv
    with torch.no_grad():
        last.weight.mul_(_LAST_SCALE)
        if last.bias is not None:
            last.bias.mul_(_LAST_SCALE)
    override(m, _scaled_rstb)


class Portable(Runner):
    """Plain PyTorch in the card's dtype."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network, precompute the position biases, pad the heads, cache the masks per size."""
        super().__init__(ctx)
        p = ctx.params
        with warnings.catch_warnings():  # upstream's torch.meshgrid without `indexing` ("ij", the default) warns
            warnings.filterwarnings("ignore", "torch.meshgrid", UserWarning)
            self.net = Swin2SR(
                upscale=p.scale, in_chans=3, img_size=p.img_size, window_size=p.window, img_range=1.0,
                depths=list(p.depths), embed_dim=p.embed_dim, num_heads=list(p.heads), mlp_ratio=2,
                upsampler=p.upsampler, resi_connection=p.resi_connection,
            )  # fmt: skip
        load_strict(self.net, ctx.state_dict(), f"{ctx.variant.ref} checkpoint")
        for m in self.net.modules():
            if isinstance(m, WindowAttention):
                _fold(m)
                override(m, _attention)
            elif isinstance(m, SwinTransformerBlock):
                override(m, _mask, "calculate_mask")
            elif isinstance(m, RSTB):
                _fold_proj(m)
        for m in self.net.modules():
            if isinstance(m, PatchEmbed):
                override(m, _patch_embed)
        _scale_last(self.net.layers[-1])  # ty: ignore[invalid-argument-type]
        self.net.norm.eps *= _LAST_SCALE**2  # LayerNorm(s * x, eps * s^2) == LayerNorm(x, eps)
        mean = self.net.mean
        del self.net.mean
        self.net.register_buffer("mean", mean, persistent=False)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """`main_test_swin2sr.py`: mirror-pad bottom / right to `(size // window + 1) * window`."""
        h, w = x.shape[-2:]
        win = self.ctx.params.window
        x = torch.cat([x, x.flip(2)], 2)[:, :, : (h // win + 1) * win]
        x = torch.cat([x, x.flip(3)], 3)[:, :, :, : (w // win + 1) * win]
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
            if is_base(m, torch.nn.Conv2d):
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        with fp16_accumulate():
            return {"image": self.net(image.to(torch.float16))}


RUNNERS = {"portable": Portable, "kernels": Kernels}

_BLOCK = re.compile(r"layers\.(\d+)\.residual_group\.blocks\.(\d+)\.")


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load(path)`: a Swin2SR state dict as a variant and its params (depths, heads, upsampler); keys unchanged."""
    if "conv_first.weight" not in sd or "layers.0.residual_group.blocks.0.attn.logit_scale" not in sd:
        return None
    blocks = [(int(m.group(1)), int(m.group(2))) for k in sd if (m := _BLOCK.match(k))]
    layers = 1 + max(i for i, _ in blocks)
    depths = tuple(1 + max(j for i, j in blocks if i == n) for n in range(layers))
    heads = tuple(sd[f"layers.{n}.residual_group.blocks.0.attn.logit_scale"].shape[0] for n in range(layers))
    embed_dim = sd["conv_first.weight"].shape[0]
    window = math.isqrt(sd["layers.0.residual_group.blocks.0.attn.relative_position_index"].shape[0])
    mask = sd.get("layers.0.residual_group.blocks.1.attn_mask")
    up = checkpoint_upsampler(sd)
    problems = []
    if sd["conv_first.weight"].shape[1] != 3:  # noqa: PLR2004 - RGB
        problems.append("not 3 input channels")
    if up is None:
        problems.append("no upsampler (denoising / JPEG models)")
    if "conv_aux.weight" in sd or "conv_first_hf.weight" in sd:
        problems.append("the compressed-SR / HF upsampler")
    if "absolute_pos_embed" in sd:
        problems.append("absolute position embedding")
    if sd["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0] != 2 * embed_dim:
        problems.append("an MLP ratio other than 2")
    if problems:
        msg = f"Swin2SR checkpoint with {', '.join(problems)}: not supported"
        raise WeightsError(msg)
    upsampler, scale = up  # ty: ignore[not-iterable]
    params = {
        "scale": scale,
        "embed_dim": embed_dim,
        "depths": depths,
        "heads": heads,
        "upsampler": upsampler,
        "resi_connection": "3conv" if "layers.0.conv.0.weight" in sd else "1conv",
        "img_size": 64 if mask is None else math.isqrt(mask.shape[0]) * window,
        "window": window,
    }
    base = "swin2sr-x4-real" if upsampler == "nearest+conv" else "swin2sr-x2" if scale == 2 else "swin2sr-x4"  # noqa: PLR2004
    return base, params, dict(sd)
