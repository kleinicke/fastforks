"""Real-ESRGAN runners: BasicSR's `RRDBNet` and the call of upstream `RealESRGANer.enhance`.

As upstream (`inference_realesrgan.py`, no tiling, `pre_pad` 0): the RGB image in `[0, 1]` goes through the network
(x2: reflect-padded at the bottom and right to even sizes, then pixel-unshuffled), the output is cropped and clamped to
`[0, 1]`. Upstream runs float16 by default (`--fp32` for float32).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.kernels import cudnn_hacc, sr_ops
from fastforks.restoration._runner import Crop, Forward, Runner

from .net import RRDBNet
from .net.rrdbnet_arch import RRDB, ResidualDenseBlock

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`."""

    scale: int  # 4, 2 or 1 (x2 / x1: the input is pixel-unshuffled by 2 / 4)
    num_block: int  # RRDB blocks: 23, 6 in the anime model
    num_feat: int  # trunk channels (64 in the release models)
    num_grow: int  # dense-block growth channels (32 in the release models)


_TAIL_TILE_BYTES = 64 * 1024**2  # upper target per high-resolution 64-channel feature map


def _upsample(m: RRDBNet, feature: torch.Tensor) -> torch.Tensor:
    y = m.lrelu(m.conv_up1(F.interpolate(feature, scale_factor=2, mode="nearest")))
    y = m.lrelu(m.conv_up2(F.interpolate(y, scale_factor=2, mode="nearest")))
    return m.conv_last(m.lrelu(m.conv_hr(y)))


def _upsample_stripes(m: RRDBNet, feature: torch.Tensor) -> torch.Tensor:
    """The two 2x upconvolutions and two output convolutions need just two input halo rows on either side."""
    b, c, h, w = feature.shape
    rows = max(1, _TAIL_TILE_BYTES // (b * c * 16 * w * feature.element_size()))
    if rows >= h:
        return _upsample(m, feature)
    out = torch.empty(
        (b, m.conv_last.out_channels, 4 * h, 4 * w),
        device=feature.device,
        dtype=feature.dtype,
        memory_format=torch.channels_last,
    )
    for start in range(0, h, rows):
        end = min(start + rows, h)
        lo, hi = max(0, start - 2), min(h, end + 2)
        tile = _upsample(m, feature[:, :, lo:hi])
        out[:, :, 4 * start : 4 * end].copy_(tile[:, :, 4 * (start - lo) : 4 * (end - lo)])
    return out


def _stripe_forward(m: RRDBNet, x: torch.Tensor) -> torch.Tensor:
    feature = F.pixel_unshuffle(x, 4 // m.scale) if m.scale in (1, 2) else x
    feature = m.conv_first(feature)
    feature = feature + m.conv_body(m.body(feature))
    return _upsample_stripes(m, feature)


class Portable(Runner):
    """Plain PyTorch in the card's dtype, channels-last on CUDA.

    cuDNN's float16 convolutions at 512²: 631 -> 470 ms, 2.2 -> 1.2 GB. On the CPU, channels-last changes the float32
    rounding (4e-5 against the golden's 1e-5), so the layout stays.
    """

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        p = ctx.params
        self.net = RRDBNet(3, 3, scale=p.scale, num_feat=p.num_feat, num_block=p.num_block, num_grow_ch=p.num_grow)
        load_strict(self.net, ctx.state_dict(key="params_ema"), f"{ctx.variant.ref} checkpoint")
        self.fmt = torch.channels_last if torch.device(ctx.device).type == "cuda" else torch.contiguous_format
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype)).to(memory_format=self.fmt)

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """x2 / x1: reflect padding at the bottom and right to multiples of 2 / 4 (`RealESRGANer.pre_process`)."""
        h, w = x.shape[-2:]
        mod = {2: 2, 1: 4}.get(self.ctx.params.scale)
        if mod and (h % mod or w % mod):
            x = F.pad(x, (0, -w % mod, 0, -h % mod), mode="reflect")
        return x, Crop(0, 0, h, w)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        x = image.to(getattr(torch, self.ctx.dtype)).contiguous(memory_format=self.fmt)
        return {"image": _stripe_forward(self.net, x) if x.is_cuda else self.net(x)}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype."""
        dtype, fmt = getattr(torch, self.ctx.dtype), self.fmt
        return Forward(self.net, lambda net, image: net(image.to(dtype).contiguous(memory_format=fmt)))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


def _dense(m: ResidualDenseBlock, x: torch.Tensor) -> torch.Tensor:
    """Write each growth convolution directly into its final channel slice instead of rebuilding dense prefixes."""
    b, c, h, w = x.shape
    packed = torch.empty(
        (b, m.conv5.in_channels, h, w), device=x.device, dtype=x.dtype, memory_format=torch.channels_last
    )
    packed[:, :c].copy_(x)
    backend = cudnn_hacc.load(x.device)
    for conv in (m.conv1, m.conv2, m.conv3, m.conv4):
        tail = packed[:, c : c + conv.out_channels]
        backend.conv_into(packed[:, :c], conv.weight, tail, padding=1, stride=1)
        sr_ops.bias_activation(tail, conv.get_parameter("bias"))
        c += conv.out_channels
    return sr_ops.residual(m.conv5(packed), x)


def _rrdb(m: RRDB, x: torch.Tensor) -> torch.Tensor:
    return sr_ops.residual(m.rdb3(m.rdb2(m.rdb1(x))), x)


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convolutions (`cudnn_hacc`)."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        sr_ops.load(ctx.device)
        for m in self.net.modules():
            if isinstance(m, ResidualDenseBlock):
                override(m, _dense)
            elif isinstance(m, RRDB):
                override(m, _rrdb)
            if is_base(m, torch.nn.Conv2d):
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        x = image.to(torch.float16).contiguous(memory_format=self.fmt)
        with fp16_accumulate():
            return {"image": _stripe_forward(self.net, x)}


RUNNERS = {"portable": Portable, "kernels": Kernels}

# ESRGAN's original `model.N` layout (old-arch ESRGAN files, e.g. 4x-UltraSharp) -> BasicSR's RRDBNet names
_OLD = {"model.0.": "conv_first.", "model.3.": "conv_up1.", "model.6.": "conv_up2.", "model.8.": "conv_hr."}
_OLD_RDB = re.compile(r"model\.1\.sub\.(\d+)\.RDB(\d)\.conv(\d)\.0\.")
_SCALES = {3: 4, 12: 2, 48: 1}  # conv_first input channels -> scale (x2, x1: pixel-unshuffled input)
_VARIANTS = {4: "realesrgan-x4plus", 2: "realesrgan-x2plus"}


def _new_keys(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The old ESRGAN layout under RRDBNet's keys (x4 with two upsampling convs; others keep a `model.` key)."""
    blocks = {int(m.group(1)) for k in sd if (m := _OLD_RDB.match(k))}
    trunk = f"model.1.sub.{len(blocks)}."
    out = {}
    for k, v in sd.items():
        if m := _OLD_RDB.match(k):
            out[f"body.{m.group(1)}.rdb{m.group(2)}.conv{m.group(3)}.{k[m.end() :]}"] = v
            continue
        prefix = next((p for p in (*_OLD, trunk, "model.10.") if k.startswith(p)), None)
        name = {trunk: "conv_body.", "model.10.": "conv_last.", **_OLD}.get(prefix) if prefix else None
        out[name + k[len(prefix) :] if name and prefix else k] = v
    return out


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load(path)`: an RRDBNet (Real-ESRGAN / ESRGAN; BasicSR or old `model.N` keys) as a variant, params, keys."""
    if "model.1.sub.0.RDB1.conv1.0.weight" in sd:
        sd = _new_keys(sd)
    elif "body.0.rdb1.conv1.weight" not in sd:
        return None
    blocks = len({k.split(".")[1] for k in sd if k.startswith("body.")})
    first, grow = sd["conv_first.weight"], sd["body.0.rdb1.conv1.weight"]
    last = sd.get("conv_last.weight")
    problems = [
        f"{what} {got} (fastforks: {want})"
        for what, got, want in (("output channels", None if last is None else last.shape[0], 3),)
        if got != want
    ]
    if first.shape[1] not in _SCALES:
        problems.append(f"input channels {first.shape[1]} (fastforks: 3 for x4, 12 for x2, 48 for x1)")
    if "conv_up2.weight" not in sd or any(k.startswith("model.") for k in sd):
        problems.append("not two x2 upsampling stages (fastforks: x4, or x2 / x1 with a pixel-unshuffled input)")
    if problems:
        msg = f"RRDBNet (ESRGAN) checkpoint with {', '.join(problems)}: not supported"
        raise WeightsError(msg)
    scale = _SCALES[first.shape[1]]
    params = {"scale": scale, "num_block": blocks, "num_feat": first.shape[0], "num_grow": grow.shape[0]}
    return _VARIANTS.get(scale, "realesrgan-x4plus"), params, dict(sd)
