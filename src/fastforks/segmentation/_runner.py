"""The interface a semantic segmentation family implements: one `Runner` per backend, in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...

    def __call__(self, image):  # [B, 3, H, W] RGB in [0, 1]
        x = scale(image, self.ctx.io)  # + the family's resize / sliding window
        return {"logits": upsample(self.net(x), image.shape[-2:])}  # [B, classes, H, W]
```

Segmentation families differ in their test-time input handling (resize, sliding window), so a runner gets the raw
image and returns logits at its size; `fastforks._core.preprocess` has the shared helpers. The core does image
conversion and checks, batching, the per-pixel argmax, output checks, locking and the device guard. A family whose
logits are upsampled may override `labels` (the labels-only call, the default) to take the argmax without the
full-resolution logits (`fastforks.shared.seginfer.upsample_argmax`).
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, ClassVar

from fastforks._core import runner as core
from fastforks._core.preprocess import scale
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    import torch

__all__ = ["Context", "Forward", "NoParams", "Runner", "scale"]


class Runner(core.Runner):
    """One backend of a semantic segmentation family.

    Attributes:
        probabilities: The family returns per-class probabilities instead of logits (mask classification: class
            softmax times mask sigmoid, summed over the queries, as HF / detectron2 `semantic_inference`); the API
            hands out their log as `logits`.
    """

    probabilities: ClassVar[bool] = False
    compiled_methods: ClassVar[tuple[str, ...]] = ("__call__", "labels")  # labels: the default (labels-only) call

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            image: `[B, 3, H, W]` float32 RGB in `[0, 1]` on `ctx.device`.

        Returns:
            `{"logits": [B, classes, H, W]}` (any float dtype; probabilities if `probabilities`) at the input size,
            classes in the order of the card's label set.
        """

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Per-pixel class `[B, H, W]` (int64) of a batch: the argmax over the classes of `self(image)["logits"]`.

        The API's labels-only call. An override must give that argmax up to near-ties (pixels whose two best classes
        differ by about the rounding of the logits' dtype may take either; exact ties: the first index), only cheaper.
        """
        return self(image)["logits"].argmax(1)
