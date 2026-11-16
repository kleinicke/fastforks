"""The interface an edges family implements: one `Runner` per backend, in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    Params = Params  # with `resolution` (the card's [options])

    def __call__(self, image):  # [B, 3, H, W] RGB in [0, 1]
        x = to_detect(image, self.ctx.params.resolution)  # 0..255 at the detect size, as controlnet_aux
        return {"edges": to_input(self.net(x)[:, 0], image)}  # [B, H, W] in [0, 1], 1 = line
```

Families run at controlnet_aux's detect resolution (`resize_image`: the short side scaled to `resolution`, both sides
rounded to a multiple of 64) and return the map at the input size. Families that find line segments (M-LSD) also
return `lines`: a list of `[N, 4]` float32 tensors (x0, y0, x1, y1 in input pixels), one per image.
The core does image conversion and checks, batching, output checks, locking and the device guard.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core import runner as core
from fastforks._core.runner import Context, Forward, NoParams
from fastforks.shared.cv2_resize import cv2_resize

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["Context", "Forward", "NoParams", "Runner", "detect_size", "safe_step", "to_detect", "to_input"]

MULTIPLE = 64  # fixed: controlnet_aux `resize_image` rounds both sides to a multiple of 64


def detect_size(height: int, width: int, resolution: int) -> tuple[int, int]:
    """controlnet_aux `resize_image`'s size: short side scaled to `resolution`, sides rounded to multiples of 64."""
    k = resolution / min(height, width)
    return round(height * k / MULTIPLE) * MULTIPLE, round(width * k / MULTIPLE) * MULTIPLE


def to_detect(image: torch.Tensor, resolution: int) -> torch.Tensor:
    """`[B, 3, H, W]` in `[0, 1]` -> uint8 levels (0..255, float32) at the detect size, as controlnet_aux.

    OpenCV's `INTER_AREA` when shrinking (exact), `INTER_CUBIC` when enlarging (upstream: `INTER_LANCZOS4`).
    """
    h, w = image.shape[-2:]
    size = detect_size(h, w, resolution)
    levels = (image * 255).round_().clamp_(0, 255)
    if size == (h, w):
        return levels
    return cv2_resize(levels, *size, "cubic" if resolution > min(h, w) else "area")


def safe_step(x: torch.Tensor, step: int = 2) -> torch.Tensor:
    """controlnet_aux `safe_step` (`int(x * (step + 1)) / step`, x >= 0) with its 8-bit output's clip to 1."""
    return ((x * (step + 1)).floor() / step).clamp_(max=1)


def to_input(edges: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """`[B, h, w]` map at the detect size -> `[B, H, W]` at the image size (bilinear, OpenCV's `INTER_LINEAR`)."""
    if edges.shape[-2:] == image.shape[-2:]:
        return edges
    return F.interpolate(edges[:, None], image.shape[-2:], mode="bilinear", align_corners=False)[:, 0]


class Runner(core.Runner):
    """One backend of an edges family."""

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor) -> Mapping[str, torch.Tensor | list[torch.Tensor]]:
        """Run a batch.

        Args:
            image: `[B, 3, H, W]` float32 RGB in `[0, 1]` on `ctx.device`.

        Returns:
            `{"edges": [B, H, W]}` (any float dtype) in `[0, 1]` (1 = line) at the input size; line-segment families
            also `"lines"`: `B` float32 tensors `[N, 4]` (x0, y0, x1, y1 in input pixels).
        """
