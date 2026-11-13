"""The interface a calibration family implements: one `Runner` per backend, in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...

    def __call__(self, image):  # [B, 3, H, W] RGB in [0, 1]
        k, g = self.net(resize(image))
        return {"intrinsics": undo_resize(k), "gravity": g}  # [B, 3, 3] pixels of `image`, [B, 3]
```

Calibration families resize the image to their network size themselves and return the camera of the input image
(intrinsics in its pixels). `roll`, `pitch` and `vfov` are in radians.
The core does image conversion and checks, batching, output checks, locking and the device guard.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from fastforks._core import runner as core
from fastforks._core.preprocess import scale
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    import torch

__all__ = ["Context", "Forward", "NoParams", "Runner", "scale"]


class Runner(core.Runner):
    """One backend of a calibration family."""

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            image: `[B, 3, H, W]` float32 RGB in `[0, 1]` on `ctx.device`.

        Returns:
            `{"intrinsics": [B, 3, 3]}` (pixels of `image`) and any of `gravity` `[B, 3]` (unit vector in the camera
            frame: x right, y down, z forward), `roll`, `pitch`, `vfov` `[B]` (radians) and the card's extras.
        """
