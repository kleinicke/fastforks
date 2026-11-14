"""GFPGAN runners: TencentARC/GFPGAN's `GFPGANv1Clean` and `GFPGANer.enhance`'s call on one aligned face.

Upstream normalises the aligned 512 x 512 face crop to [-1, 1], runs `net(x, return_rgb=False)[0]` and maps the
output from [-1, 1] back (`tensor2img(min_max=(-1, 1))`); face detection, alignment and pasting back are not part of
the network and not part of this family: the input is the aligned crop.

Changes: the StyleGAN2 noise is the checkpoint's stored noise (`randomize_noise=False`; upstream draws new noise on
every call, so its output varies from call to call: noise weights <= 0.06, see the results file); the intermediate
RGB outputs (`return_rgb`) are not computed and the SFT conditions are not cloned; the SFT of the second channel half
runs in place; the modulated convolutions take the noise-free bias as the convolution bias and fold StyleConv's
sqrt(2) into the modulated weight (modulation and demodulation in float32); the unet's upsampling ResBlocks apply
their bias-free 1x1 skip convolution before the bilinear upsampling (both linear: the same result at a quarter of
the cost). CUDA: channels-last.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import InputError, WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import override
from fastforks.kernels import cudnn_hacc
from fastforks.restoration._runner import Crop, Forward, Runner

from .net import GFPGANv1Clean

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fastforks.restoration._runner import Context

    from .net.gfpganv1_clean_arch import ResBlock
    from .net.stylegan2_clean_arch import ModulatedConv2d

    Conv = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class Params:
    """Card `[params]` (`GFPGANer`'s clean v1.3 / v1.4 network)."""

    scale: int  # 1: the restored face has the input's size
    size: int  # 512: the aligned face crop (out_size)
    key: str  # checkpoint key ("params_ema")


def _torch_conv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, padding: int, groups: int) -> torch.Tensor:
    return F.conv2d(x, w, b, padding=padding, groups=groups)


def _hacc(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, padding: int, groups: int) -> torch.Tensor:
    if groups != 1:  # Kernels runs one face at a time
        msg = "cudnn_hacc convolutions have groups 1"
        raise ValueError(msg)
    w = w.contiguous(memory_format=torch.channels_last)
    return cudnn_hacc.conv_half_acc(x, w, b, (padding, padding), (1, 1))


def _modconv(  # noqa: PLR0913, PLR0917 - the StyleConv / ToRGB call sites pass all six
    m: ModulatedConv2d, x: torch.Tensor, style: torch.Tensor, gain: float, bias: torch.Tensor | None, conv: Conv
) -> torch.Tensor:
    """`ModulatedConv2d.forward(x, style) * gain + bias` (bias `[C]`): the batch as groups, as upstream."""
    b, c = x.shape[:2]
    s = m.modulation(style).float().view(b, 1, c, 1, 1)
    w = m.weight.float() * s
    if m.demodulate:
        w = w * torch.rsqrt(w.pow(2).sum([2, 3, 4]) + m.eps).view(b, -1, 1, 1, 1)
    w = (w * gain).to(x.dtype).view(b * m.out_channels, c, m.kernel_size, m.kernel_size)
    if m.sample_mode == "upsample":
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    h, wd = x.shape[-2:]
    bias = None if bias is None else bias.to(x.dtype).repeat(b)
    out = conv(x.reshape(1, b * c, h, wd), w, bias, m.padding, b)
    return out.reshape(b, m.out_channels, *out.shape[-2:])


def _resblock(
    m: ResBlock, x: torch.Tensor, conv: Callable[[torch.nn.Conv2d, torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """`ResBlock.forward`; upsampling: the 1x1 skip before the bilinear upsampling (both linear, no bias)."""
    out = F.leaky_relu_(conv(m.conv1, x), 0.2)
    out = F.interpolate(out, scale_factor=m.scale_factor, mode="bilinear", align_corners=False)
    out = F.leaky_relu_(conv(m.conv2, out), 0.2)
    if m.scale_factor > 1:
        skip = F.interpolate(conv(m.skip, x), scale_factor=m.scale_factor, mode="bilinear", align_corners=False)
    else:
        skip = conv(m.skip, F.interpolate(x, scale_factor=m.scale_factor, mode="bilinear", align_corners=False))
    return out + skip


def _module_conv(m: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return m(x)


def _forward(m: GFPGANv1Clean, x: torch.Tensor, noise: list[torch.Tensor], conv: Conv) -> torch.Tensor:
    """`GFPGANv1Clean.forward(x, return_rgb=False, randomize_noise=False)[0]` (v1.3 / v1.4 configs).

    v1.3 / v1.4: input_is_latent, different_w, sft_half; `noise` = the decoder's noise buffers times each StyleConv's
    noise weight.
    """
    mc = _module_conv
    feat = F.leaky_relu_(m.conv_body_first(x), 0.2)
    skips = []
    for block in m.conv_body_down:
        feat = _resblock(block, feat, mc)  # ty: ignore[invalid-argument-type]
        skips.insert(0, feat)
    feat = F.leaky_relu_(m.final_conv(feat), 0.2)
    latent = m.final_linear(feat.flatten(1)).view(feat.shape[0], -1, m.num_style_feat)
    conditions = []
    for i, block in enumerate(m.conv_body_up):
        feat = _resblock(block, feat + skips[i], mc)  # ty: ignore[invalid-argument-type]
        conditions += [m.condition_scale[i](feat), m.condition_shift[i](feat)]

    d = m.stylegan_decoder
    sqrt2 = 2**0.5

    def style_conv(sc: Any, out: torch.Tensor, style: torch.Tensor, n: torch.Tensor) -> torch.Tensor:  # noqa: ANN401
        out = _modconv(sc.modulated_conv, out, style, sqrt2, sc.bias.view(-1), conv)
        return F.leaky_relu_(out.add_(n), 0.2)

    def to_rgb(tr: Any, out: torch.Tensor, style: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:  # noqa: ANN401
        out = _modconv(tr.modulated_conv, out, style, 1.0, tr.bias.view(-1), conv)
        if skip is None:
            return out
        return out.add_(F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False))

    out = d.constant_input.weight.repeat(latent.shape[0], 1, 1, 1)
    out = style_conv(d.style_conv1, out, latent[:, 0], noise[0])
    skip = to_rgb(d.to_rgb1, out, latent[:, 1], None)
    convs = d.style_convs
    for j, rgb in enumerate(d.to_rgbs):
        i = 2 * j + 1
        out = style_conv(convs[2 * j], out, latent[:, i], noise[i])
        if i < len(conditions):  # sft_half: scale and shift the second half of the channels
            half = out.shape[1] // 2
            out[:, half:].mul_(conditions[i - 1]).add_(conditions[i])
        out = style_conv(convs[2 * j + 1], out, latent[:, i + 1], noise[i + 1])
        skip = to_rgb(rgb, out, latent[:, i + 2], skip)
    return skip


def _noise(net: GFPGANv1Clean) -> list[torch.Tensor]:
    d = net.stylegan_decoder
    layers = [d.style_conv1, *d.style_convs]
    return [sc.weight * getattr(d.noises, f"noise{i}") for i, sc in enumerate(layers)]


class Portable(Runner):
    """Plain PyTorch in the card's dtype, channels-last on CUDA."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network (`GFPGANer`'s clean arguments) and load `params_ema`."""
        super().__init__(ctx)
        p = ctx.params
        self.net = _build(p.size)
        load_strict(self.net, ctx.state_dict(key=p.key), f"{ctx.variant.ref} checkpoint")
        self.fmt = torch.channels_last if torch.device(ctx.device).type == "cuda" else torch.contiguous_format
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))
        for m in self.net.modules():  # not `net.to(memory_format=)`: the modulated weights are 5-D
            if isinstance(m, torch.nn.Conv2d):
                m.to(memory_format=self.fmt)
        with torch.no_grad():
            self.noise = _noise(self.net)

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """No padding: the network takes exactly the aligned `size` x `size` face crop."""
        h, w = x.shape[-2:]
        n = self.ctx.params.size
        if (h, w) != (n, n):
            msg = f"GFPGAN restores an aligned {n} x {n} face crop, got {h} x {w} (resize or align the face first)"
            raise InputError(msg)
        return x, Crop(0, 0, h, w)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        x = (image.to(getattr(torch, self.ctx.dtype)) * 2 - 1).contiguous(memory_format=self.fmt)
        return (_forward(self.net, x, self.noise, _torch_conv).float() + 1) / 2

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored face in [0, 1] (the network works in [-1, 1])."""
        return {"image": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype, [0, 1] in and out."""
        dtype, fmt, noise = getattr(torch, self.ctx.dtype), self.fmt, self.noise
        return Forward(
            self.net,
            lambda net, image: (
                (_forward(net, (image.to(dtype) * 2 - 1).contiguous(memory_format=fmt), noise, _torch_conv).float() + 1)
                / 2
            ),
        )


def _hacc_module(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convolutions (`cudnn_hacc`, modulated ones included); one face a call."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if type(m) is torch.nn.Conv2d:
                override(m, _hacc_module)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored faces in [0, 1]."""
        x = (image.to(torch.float16) * 2 - 1).contiguous(memory_format=self.fmt)
        with fp16_accumulate():
            out = [_forward(self.net, x[i : i + 1], self.noise, _hacc) for i in range(x.shape[0])]
        return {"image": (torch.cat(out).float() + 1) / 2}


RUNNERS = {"portable": Portable, "kernels": Kernels}


def _build(size: int) -> GFPGANv1Clean:
    return GFPGANv1Clean(out_size=size, num_style_feat=512, channel_multiplier=2, decoder_load_path=None,
                         fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True, narrow=1,
                         sft_half=True)  # fmt: skip


_SHAPES = {  # the clean 512 network of v1.3 / v1.4 (GFPGANer, arch "clean"): key -> leading dims
    "conv_body_down.6.conv2.weight": (256, 256),  # out_size 512: 7 downsampling blocks
    "final_linear.weight": (16 * 512, 256 * 16),  # different_w: one style per latent
    "condition_scale.0.2.weight": (256, 256),  # sft_half
    "stylegan_decoder.style_convs.8.modulated_conv.weight": (1, 256),  # channel_multiplier 2 (128 x 128)
}


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load_file`: a clean GFPGAN (v1.3 / v1.4 layout) as a variant, params, keys."""
    if "stylegan_decoder.style_convs.0.modulated_conv.weight" not in sd or "conv_body_first.weight" not in sd:
        return None
    bad = [k for k, dims in _SHAPES.items() if k not in sd or tuple(sd[k].shape[: len(dims)]) != dims]
    if bad:
        msg = (
            f"GFPGAN checkpoint differs from the clean v1.3 / v1.4 network (out_size 512, channel_multiplier 2,"
            f" different_w, sft_half) at {', '.join(bad)}: not supported"
        )
        raise WeightsError(msg)
    return "gfpgan-v1.4", {}, dict(sd)
