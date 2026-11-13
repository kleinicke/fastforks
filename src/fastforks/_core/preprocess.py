"""Input handling every task shares, as the card's `[io]` says: value range, padding, resizing and cropping.

Plain functions on `[B, C, H, W]` tensors (traceable by dynamo and exportable), used by the task cores and by
families that handle their own inputs (`io = "raw"`).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from fastforks._core.cards import Io

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Crop(NamedTuple):  # not a frozen dataclass: dynamo of torch < 2.10 cannot build one in a traced graph
    """Region of the original image inside a padded one."""

    top: int
    left: int
    height: int
    width: int


def crop(x: torch.Tensor, region: Crop) -> torch.Tensor:
    """Crop the last two dims of `x` to `region` (a function, not a method: traceable by dynamo of torch < 2.10)."""
    return x[..., region.top : region.top + region.height, region.left : region.left + region.width]


def pad_widths(h: int, w: int, io: Io) -> tuple[int, int, int, int]:
    """`F.pad` widths (left, right, top, bottom) that bring `h, w` to multiples of `io.pad_multiple`.

    `centre` matches upstream `InputPadder` ('sintel' mode); `top_right` pads on top and on the right, `bottom_right`
    below and on the right; `none` pads nothing.
    """
    if io.pad_mode == "none":
        return 0, 0, 0, 0
    ph, pw = (-h) % io.pad_multiple, (-w) % io.pad_multiple
    if io.pad_mode == "top_right":
        return 0, pw, ph, 0
    if io.pad_mode == "bottom_right":
        return 0, pw, 0, ph
    return pw // 2, pw - pw // 2, ph // 2, ph - ph // 2


def pad_to(x: torch.Tensor, io: Io) -> tuple[torch.Tensor, Crop]:
    """Pad `[B, C, H, W]` as `io` says; returns the padded tensor and the crop that undoes it."""
    h, w = x.shape[-2:]
    pads = pad_widths(h, w, io)
    if any(pads):
        x = F.pad(x, pads, mode="replicate") if io.pad_fill == "replicate" else F.pad(x, pads)
    return x, Crop(pads[2], pads[0], h, w)


@functools.cache
def _channels(values: tuple[float, ...], device: torch.device) -> torch.Tensor:
    """`[1, C, 1, 1]` constant on `device`, made once (a new CUDA tensor from a list is a blocking host copy)."""
    return torch.tensor(values, device=device, dtype=torch.float32).view(1, -1, 1, 1)


def scale(x: torch.Tensor, io: Io) -> torch.Tensor:
    """`[0, 1]` RGB -> the value range the network expects (`io.input`)."""
    if io.input == "0-255":
        return x * 255.0
    if io.input in {"imagenet", "mean_std"}:
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if io.input == "imagenet" else (io.mean, io.std)
        if torch.compiler.is_compiling() or torch.jit.is_tracing():  # traced graphs keep their constants inline
            m, s = (torch.tensor(v, device=x.device, dtype=torch.float32).view(1, 3, 1, 1) for v in (mean, std))
        else:
            m, s = _channels(tuple(mean), x.device), _channels(tuple(std), x.device)
        return (x - m) / s
    return x


def resize_crop(x: torch.Tensor, io: Io) -> torch.Tensor:
    """Resize the shorter side to `io.resize` (antialiased), then centre-crop `io.crop` x `io.crop` (0: skip either).

    The torchvision / timm eval transform (`Resize(resize)` + `CenterCrop(crop)`), on the GPU. With `io.pad_square`
    the image is first pasted centred on a square of that value (PIL `paste` at `((side - w) // 2, (side - h) // 2)`);
    with `io.stretch` both sides are resized to `io.resize` (Hugging Face `ViTImageProcessor`).
    """
    if io.pad_square is not None:
        h, w = x.shape[-2:]
        side = max(h, w)
        left, top = (side - w) // 2, (side - h) // 2
        if side != min(h, w):
            x = F.pad(x, (left, side - w - left, top, side - h - top), value=io.pad_square)
    if io.resize:
        h, w = x.shape[-2:]
        if io.stretch:
            size = (io.resize, io.resize)
        else:  # torchvision truncates the long side: 427 x 640 -> 256 x 383
            size = (
                (io.resize, max(1, int(w * io.resize / h))) if h <= w else (max(1, int(h * io.resize / w)), io.resize)
            )
        if size != (h, w):
            antialias = io.interpolation != "nearest"
            x = F.interpolate(x, size=size, mode=io.interpolation, antialias=antialias, align_corners=None)
            if io.interpolation == "bicubic":  # bicubic overshoots; PIL / torchvision on uint8 clamp to the range
                x = x.clamp(0, 1)
    if io.crop:
        h, w = x.shape[-2:]
        if h < io.crop or w < io.crop:  # zero-pad like torchvision's CenterCrop
            pw, ph = max(io.crop - w, 0), max(io.crop - h, 0)
            x = F.pad(x, (pw // 2, (pw + 1) // 2, ph // 2, (ph + 1) // 2))
            h, w = x.shape[-2:]
        top, left = round((h - io.crop) / 2), round((w - io.crop) / 2)
        x = x[..., top : top + io.crop, left : left + io.crop]
    return x
