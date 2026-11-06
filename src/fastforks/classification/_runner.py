"""The interface an image classification family implements: one `Runner` per backend, in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    batch = None  # any batch size
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...

    def __call__(self, image):  # [B, 3, crop, crop], resized, cropped and normalised as the card's [io] says
        return {"logits": self.net(image)}  # [B, classes]
```

The core does image conversion and checks, the eval transform of the card's `[io]` (`resize` the shorter side,
centre `crop`, `input` normalisation; on the model's device), batching, output checks, locking and the device guard.
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
    """One backend of an image classification family."""

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            image: `[B, 3, h, w]` float32 on `ctx.device`, transformed as the card's `[io]` says.

        Returns:
            `{"logits": [B, classes]}` (any float dtype), classes in the order of the card's label set.
        """
