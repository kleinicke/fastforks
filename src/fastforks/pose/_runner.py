"""The interface a pose estimation family implements: one `Runner` per backend, in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...

    def __call__(self, image, boxes):  # [1, 3, H, W] RGB in [0, 1]; [P, 4] person boxes (x0, y0, x1, y1)
        return {"keypoints": ..., "keypoint_scores": ..., "scores": ...}  # P people, K keypoints
```

Top-down families (`needs_boxes = True`) crop each person box themselves (each family has its own crop rule, e.g.
ViTPose's 1.25x padded box at 256x192) and map the keypoints back to the input. The core converts images, checks the
boxes, runs a batch one image at a time, checks shapes, locks the model and guards the device.

Portable runners also implement `module()`: the network on person crops (`[B, 3, h, w]` RGB in `[0, 1]` at the
network size) -> heatmaps; cropping and decoding stay outside.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, ClassVar

import torch

from fastforks._core import runner as core
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["BUCKETS", "Context", "Forward", "NoParams", "Runner", "in_buckets"]

BUCKETS = (1, 2, 4, 8, 16, 32)  # fixed: graph cache policy (people per network call), not architecture


def in_buckets(forward: Callable[[torch.Tensor], torch.Tensor], crops: torch.Tensor) -> torch.Tensor:
    """`forward` over person crops `[P, ...]` in batches of a `BUCKETS` size: `[P, ...]` outputs in crop order.

    More people than the largest bucket run as several batches; zero crops pad a batch up to its bucket (their rows
    are dropped), so a cached graph per bucket serves any number of people.
    """
    outs = []
    for chunk in crops.split(BUCKETS[-1]):
        p = chunk.shape[0]
        bucket = next(n for n in BUCKETS if n >= p)
        padded = chunk if bucket == p else torch.cat([chunk, chunk.new_zeros(bucket - p, *chunk.shape[1:])])
        outs.append(forward(padded)[:p])
    return outs[0] if len(outs) == 1 else torch.cat(outs)


class Runner(core.Runner):
    """One backend of a pose estimation family."""

    batch = 1
    needs_boxes: ClassVar[bool] = True
    """Top-down: the call needs one box per person (one-stage families find the people themselves)."""

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Estimate the keypoints of the people in one image.

        Args:
            image: `[1, 3, H, W]` RGB float32 in `[0, 1]` on `ctx.device`.
            boxes: `[P, 4]` float32 person boxes `(x0, y0, x1, y1)` in input pixels on `ctx.device` (top-down), or
                `None` (one-stage families).

        Returns:
            `keypoints` `[P, K, 2]` (x, y in input pixels), `keypoint_scores` `[P, K]`, `scores` `[P]` (person
            confidence) and, for one-stage families, `boxes` `[P, 4]`.
        """
