"""The interface an object detection family implements: one `Runner` per backend, in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...

    def __call__(self, image, score_threshold):  # [1, 3, H, W] RGB in [0, 1], the input size
        return {"boxes": ..., "scores": ..., "labels": ..., "masks": ...}  # N instances, best first
```

Detectors resize, normalise and pad the image themselves (each family has its own resize rule, e.g. shortest edge
800), and map boxes and masks back to the input size. The core converts images, runs a batch one image at a time,
checks shapes, locks the model and guards the device.

Portable runners also implement `module()`: the dense part of the network (backbone + feature pyramid) as an
`nn.Module`; proposal selection, NMS and mask pasting are data-dependent and stay outside.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from fastforks._core import runner as core
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    import torch

__all__ = ["Context", "Forward", "NoParams", "Runner"]


class Runner(core.Runner):
    """One backend of an object detection family."""

    batch = 1
    open_vocabulary = False  # True: `__call__` also takes `text` (class names; labels index into them)

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor, score_threshold: float) -> dict[str, torch.Tensor]:
        """Detect the instances of one image.

        Args:
            image: `[1, 3, H, W]` RGB float32 in `[0, 1]` on `ctx.device`.
            score_threshold: Minimum class score (the call's, else the load option `score_threshold`).

        Open-vocabulary runners (`open_vocabulary = True`) also take `text`: a tuple of class names, given at call
        time; without it they detect the card's label set.

        Returns:
            The card's outputs for `N` instances sorted by descending score: `boxes` `[N, 4]` (`x0, y0, x1, y1` in
            input pixels), `scores` `[N]`, `labels` `[N]` int64 (index into the card's label set), `masks`
            `[N, H, W]` bool (mask families).
        """
