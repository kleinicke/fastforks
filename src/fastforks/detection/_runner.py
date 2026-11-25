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
800), and map boxes and masks back to the input size. The core converts images, checks shapes, locks the model and
guards the device. A batch runs one image at a time, unless the runner sets `batch` to `None` (or a chunk size) and
implements `detect_batch`, which detects in several images of the same size with one network call.

Portable runners also implement `module()`: the dense part of the network (backbone + feature pyramid) as an
`nn.Module`; proposal selection, NMS and mask pasting are data-dependent and stay outside.
"""

from __future__ import annotations

import abc

import torch

from fastforks._core import runner as core
from fastforks._core.runner import Context, Forward, NoParams

__all__ = ["Context", "Forward", "NoParams", "Runner", "split_kept"]


class Runner(core.Runner):
    """One backend of an object detection family."""

    batch = 1  # images per `detect_batch` call (None: any number); 1: the core calls `__call__` per image
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

    def detect_batch(self, images: torch.Tensor, score_threshold: float) -> list[dict[str, torch.Tensor]]:
        """`__call__` for every image of `images` `[B, 3, H, W]`; runners with `batch != 1` batch the network call."""
        return [self(images[i : i + 1], score_threshold) for i in range(images.shape[0])]


def split_kept(keep: torch.Tensor, max_detections: int, **outputs: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """Per image of `keep` `[B, Q]` (bool), the rows of `outputs` `[B, Q, ...]` it keeps, at most `max_detections`.

    The rows stay in query order. One host sync reads the kept positions (two for a batch: the counts per image).
    """
    nz = keep.nonzero()  # [K, 2]
    b = keep.shape[0]
    rows = [nz[:, 1]] if b == 1 else nz[:, 1].split(torch.bincount(nz[:, 0], minlength=b).tolist())
    return [{k: v[b][r[:max_detections]] for k, v in outputs.items()} for b, r in enumerate(rows)]
