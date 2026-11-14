"""AnyLine runners: weights -> network and the portable forward pass.

Inference as controlnet_aux's `AnylineDetector` with the float blend of TheMistoAI/ComfyUI-Anyline, all at the detect
size (`edges._runner.to_detect`, 0..255 floats), then resized to the input size (bilinear):

- base: MTEED's four outputs bilinear to the detect size, the sigmoid of their mean, `safe_step(edge, 2)`;
- line art: `min_c(gaussian(x, sigma 2) - x)` (OpenCV's 17-tap kernel, reflect-101 border) clipped to [0, 255], divided
  by `max(16, median of the values > 3)` per image, times 127, truncated to uint8 levels; 4-connected specks of fewer
  than 12 pixels removed (upstream: `remove_small_objects(min_size=36, connectivity=1)` on the 3-channel copy);
- out: `1 - (1 - top) * (1 - base)` where the line art `top` is drawn, else `base`.

`portable`: upstream's modules in plain PyTorch, fp32 or fp16; the line art in float32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.edges._runner import Forward, Runner, safe_step, to_detect, to_input
from fastforks.shared.teed import TED, edge_map

if TYPE_CHECKING:
    from fastforks.edges._runner import Context

SIGMA = 2.0  # fixed: upstream hard-codes guassian_sigma
TAPS = 17  # fixed: OpenCV's float32 kernel size for SIGMA: round(sigma * 4 * 2 + 1) | 1
THRESHOLD = 3  # fixed: upstream hard-codes intensity_threshold
MIN_PIXELS = 12  # fixed: upstream remove_small_objects(min_size=36) on 3 identical channels with connectivity 1


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`.

    Attributes:
        resolution: Detect resolution: the short side of the network input (rounded to a multiple of 64).
    """

    resolution: int


def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    """OpenCV `GaussianBlur(x, (0, 0), 2.0)` of `[B, C, H, W]` float32 (separable, reflect-101 border)."""
    t = torch.arange(TAPS, device=x.device, dtype=torch.float64) - TAPS // 2
    k = torch.exp(-(t**2) / (2 * SIGMA**2))
    k = (k / k.sum()).float()
    c, p = x.shape[1], TAPS // 2
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, TAPS), groups=c)
    return F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, TAPS, 1), groups=c)


def median_above(v: torch.Tensor, threshold: float) -> torch.Tensor:
    """Numpy `median(v[v > threshold])` per row of `[B, N]` (the mean of the two middle values); NaN if none."""
    valid = v > threshold
    n = valid.sum(-1, keepdim=True)
    s = torch.where(valid, v, math.inf).sort(-1).values
    lo, hi = ((n - 1).clamp(min=0) // 2), (n // 2).clamp(max=v.shape[-1] - 1)
    med = (s.gather(-1, lo) + s.gather(-1, hi))[:, 0] / 2
    return torch.where(n[:, 0] > 0, med, math.nan)


def _shifted(t: torch.Tensor, fill: int) -> list[torch.Tensor]:
    """The 4 neighbours of every pixel of `[B, H, W]` (`fill` outside)."""
    p = F.pad(t, (1, 1, 1, 1), value=fill)
    return [p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]]


def remove_small(fg: torch.Tensor, min_pixels: int = MIN_PIXELS) -> torch.Tensor:
    """`fg` `[B, H, W]` bool without its 4-connected components of fewer than `min_pixels` pixels.

    skimage `remove_small_objects`, exact: after `min_pixels - 1` min-label propagation steps every small component
    carries a single label (its diameter is at most `min_pixels - 1`); a label group with no differently labelled
    neighbour is a whole component, so its size is that component's.
    """
    n = fg.numel()
    label = torch.where(fg, torch.arange(n, device=fg.device).view(fg.shape), n)
    for _ in range(min_pixels - 1):
        label = torch.where(fg, torch.minimum(label, torch.stack(_shifted(label, n)).amin(0)), n)
    differs = torch.zeros_like(fg)
    for nb in _shifted(label, n):
        differs |= (nb != label) & (nb < n)
    li = label.clamp(max=n - 1)
    count = torch.bincount(li[fg], minlength=n)
    opened = torch.zeros(n, dtype=torch.bool, device=fg.device)
    opened[li[differs & fg]] = True
    return fg & ((count[li] >= min_pixels) | opened[li])


def line_art(x: torch.Tensor) -> torch.Tensor:
    """controlnet_aux `lineart_standard` levels of `[B, 3, h, w]` 0..255 floats with the specks removed, in `[0, 1]`."""
    x = x.float()
    intensity = (gaussian_blur(x) - x).amin(1).clamp_(0, 255)
    med = median_above(intensity.flatten(1), THRESHOLD).nan_to_num(16).clamp(min=16)
    lin = (intensity / med[:, None, None] * 127).clamp_(0, 255).floor_()
    return torch.where(remove_small(lin > 0), lin / 255, 0)


def anyline(outputs: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """MTEED outputs + the detect-size input `x` (0..255) -> the AnyLine map `[B, h, w]` in `[0, 1]`."""
    base = safe_step(edge_map(outputs), 2)
    top = line_art(x)
    return torch.where(top > 0, 1 - (1 - top) * (1 - base), base)


class Portable(Runner):
    """Upstream's network in plain PyTorch, float32 or float16 (NCHW); the line art in float32."""

    Params = Params
    dtypes = ("float32", "float16")
    compiled_methods = ("net",)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        net = TED()
        net.load_state_dict(ctx.state_dict())
        # NCHW: channels-last measured 15-25 % slower and +20 % memory at 1024 px (RTX 5060 Ti)
        self.net = net.eval().to(ctx.device, self.dtype)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """AnyLine map `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        x = to_detect(image, self.ctx.params.resolution)
        return {"edges": to_input(anyline(self.net(x.to(self.dtype)), x), image)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> edges `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["edges"])


RUNNERS = {"portable": Portable}
