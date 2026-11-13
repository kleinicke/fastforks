"""SwinIR runners: the network of JingyunLiang/SwinIR and `main_test_swinir.py`'s call (mirror padding, `model(lq)`).

`main_test_swinir.py` pads the input by mirroring (`cat(x, flip(x))`) to the next multiple of the window size plus one
window when it is already a multiple; `pad` does the same. Upstream's blocks rebuild the shifted-window mask on the CPU
for every size but the training one, and gather the relative position bias on every call; here the mask is made on the
device once per size and the bias once at load, and the window attention is one `scaled_dot_product_attention` with
the head dimension (30) zero-padded to 32 in the weights so the fused kernel runs (`fastforks.shared.swin_sr`:
unpadded, SDPA's math path takes 2.1 GB and 5x the time per block at 520 x 520). `PatchEmbed` returns contiguous
tokens (upstream: a transposed view, which made every residual add strided). The input mean is a buffer (upstream: a
plain attribute, `type_as` per call).

float16: the last residual group's output exceeds the float16 range (113 520 on the golden image, SwinIR-M x4), so its
final conv and its skip input are divided by `_LAST_SCALE`; only the LayerNorm `norm` reads it, its `eps` divided by
the square (exact).
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
from fastforks._core.patch import override
from fastforks.kernels import cudnn_hacc
from fastforks.restoration._runner import Crop, Forward, Runner
from fastforks.shared.swin_sr import pad_heads, patch_embed, relative_bias, shift_mask, window_attention

from .net import SwinIR
from .net.network_swinir import RSTB, PatchEmbed, SwinTransformerBlock, WindowAttention

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` (`main_test_swinir.py` `define_model`)."""

    scale: int  # 4 or 2
    embed_dim: int  # 180 (M) or 240 (L)
    depths: tuple[int, ...]  # Swin layers per residual group
    heads: tuple[int, ...]  # attention heads per residual group
    upsampler: str  # "pixelshuffle" (classical) or "nearest+conv" (real-world)
    resi_connection: str  # "1conv" or "3conv" (L)
    img_size: int  # training patch size (the size of the stored attention masks)
    key: str  # checkpoint key: "params" or "params_ema"
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
    return shift_mask(*x_size, m.window_size, m.shift_size, m.norm1.weight.device)


def _attention(m: WindowAttention, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """`WindowAttention.forward` with the bias gathered at load and one SDPA call."""
    return window_attention(x, m.qkv, m.proj, heads=m.num_heads, scale=m.scale, bias=m.bias, mask=mask)  # ty: ignore[invalid-argument-type]


_LAST_SCALE = 64.0  # fixed: float16 headroom of the last group's residual sum, not an upstream parameter


def _rstb_scaled(m: RSTB, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
    """`RSTB.forward` / `_LAST_SCALE` (the final conv's weights are divided at load)."""
    return m.patch_embed(m.conv(m.patch_unembed(m.residual_group(x, x_size), x_size))) + x / _LAST_SCALE


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
            self.net = SwinIR(
                upscale=p.scale, in_chans=3, img_size=p.img_size, window_size=p.window, img_range=1.0,
                depths=list(p.depths), embed_dim=p.embed_dim, num_heads=list(p.heads), mlp_ratio=2,
                upsampler=p.upsampler, resi_connection=p.resi_connection,
            )  # fmt: skip
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
            elif isinstance(m, PatchEmbed):
                override(m, patch_embed)
        mean = self.net.mean
        del self.net.mean
        self.net.register_buffer("mean", mean, persistent=False)
        if ctx.dtype == "float16":
            last = self.net.layers[-1]
            *_, conv = (c for c in last.modules() if isinstance(c, torch.nn.Conv2d))  # its `conv` (3conv: the last)
            with torch.no_grad():
                conv.weight.div_(_LAST_SCALE)
                if conv.bias is not None:
                    conv.bias.div_(_LAST_SCALE)
            override(last, _rstb_scaled)
            self.net.norm.eps /= _LAST_SCALE**2  # LayerNorm(x / k, eps / k²) = LayerNorm(x, eps)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """`main_test_swinir.py`: mirror-pad bottom / right to `(size // window + 1) * window`."""
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
            if type(m) is torch.nn.Conv2d:
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        with fp16_accumulate():
            return {"image": self.net(image.to(torch.float16))}


RUNNERS = {"portable": Portable, "kernels": Kernels}

_BLOCK = re.compile(r"layers\.(\d+)\.residual_group\.blocks\.(\d+)\.")


def _upsampler(sd: Mapping[str, torch.Tensor]) -> tuple[str, int] | None:
    """`(upsampler, scale)` from the reconstruction layers, or None for one fastforks does not run (denoising)."""
    if "conv_up1.weight" in sd:
        return "nearest+conv", 4 if "conv_up2.weight" in sd else 2
    convs = [v for k, v in sd.items() if k.startswith("upsample.") and k.endswith(".weight")]
    if not convs:
        return None
    if "conv_before_upsample.0.weight" not in sd:  # one conv to scale² x 3 channels
        return "pixelshuffledirect", round(math.sqrt(convs[0].shape[0] / 3))
    ratio = convs[0].shape[0] // sd["conv_before_upsample.0.weight"].shape[0]
    return "pixelshuffle", 3 if ratio == 9 else 2 ** len(convs)  # noqa: PLR2004 - x3: one conv to 9 x channels


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load_file`: a SwinIR state dict as a variant and its params (size, depths, heads, upsampler); keys unchanged."""
    blocks = [(int(m.group(1)), int(m.group(2))) for k in sd if (m := _BLOCK.match(k))]
    if "conv_first.weight" not in sd or not blocks:
        return None
    layers = 1 + max(i for i, _ in blocks)
    depths = tuple(1 + max(j for i, j in blocks if i == n) for n in range(layers))
    tables = [sd[f"layers.{n}.residual_group.blocks.0.attn.relative_position_bias_table"] for n in range(layers)]
    embed_dim = sd["conv_first.weight"].shape[0]
    window = (math.isqrt(tables[0].shape[0]) + 1) // 2
    mask = sd.get("layers.0.residual_group.blocks.1.attn_mask")
    up = _upsampler(sd)
    problems = []
    if sd["conv_first.weight"].shape[1] != 3:  # noqa: PLR2004 - RGB
        problems.append("not 3 input channels")
    if up is None:
        problems.append("no upsampler (denoising / JPEG models)")
    if "absolute_pos_embed" in sd:
        problems.append("absolute position embedding")
    if sd["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0] != 2 * embed_dim:
        problems.append("an MLP ratio other than 2")
    if problems:
        msg = f"SwinIR checkpoint with {', '.join(problems)}: not supported"
        raise WeightsError(msg)
    upsampler, scale = up  # ty: ignore[not-iterable]
    params = {
        "scale": scale,
        "embed_dim": embed_dim,
        "depths": depths,
        "heads": tuple(t.shape[1] for t in tables),
        "upsampler": upsampler,
        "resi_connection": "3conv" if "layers.0.conv.0.weight" in sd else "1conv",
        "img_size": 64 if mask is None else math.isqrt(mask.shape[0]) * window,
        "window": window,
    }
    base = "swinir-m-x4-real" if upsampler == "nearest+conv" else "swinir-m-x2" if scale == 2 else "swinir-m-x4"  # noqa: PLR2004
    return base, params, dict(sd)
