"""The interface a dense-matching model family implements: one `Runner` per backend, listed in `model.py` as `RUNNERS`.

```python
# matching/models/<family>/model.py
class Portable(Runner):
    dtypes = ("float32", "float16")  # CUDA dtypes this code implements; the card picks the validated one
    Params = Params

    def __call__(self, image1, image2):  # scaled [B, 3, H1, W1], [B, 3, H2, W2]
        return {"warp": ..., "certainty": ...}  # [B, 2, H1, W1] in image2 pixels, [B, H1, W1] in [0, 1]


RUNNERS = {"portable": Portable}
```

The runner resizes the images to its network size and the maps back itself (the two images may differ in size). The
core does the rest: image conversion and checks, scaling as the card's `[io]` says, output checks, the cast to
float32, batching, locking and the device guard.
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
    """One backend of a dense-matching model family."""

    @abc.abstractmethod
    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            image1: `[B, 3, H1, W1]` float32 on `ctx.device`, scaled as the card's `[io]` says.
            image2: `[B, 3, H2, W2]`, the image to match into (any size).

        Returns:
            `{"warp": [B, 2, H1, W1]}` (any float dtype): `(x, y)` in pixels of `image2` (pixel centres at integers) for
            every pixel of `image1`, plus `"certainty"` `[B, H1, W1]` in `[0, 1]` if the card lists it.
        """

    def module(self) -> torch.nn.Module:
        """The plain-PyTorch network as an `nn.Module`, as `torch.export` / ONNX export see it.

        It maps `(image1, image2)` scaled `[B, 3, H, W]` to the runner's `{name: map}`. Portable runners implement it.

        Raises:
            NotImplementedError: The backend has no exportable module.
        """
        return super().module()
