"""The interface a frame-interpolation family implements: one `Runner` per backend, listed in `model.py` as `RUNNERS`.

```python
# interpolation/models/<family>/model.py
class Portable(Runner):
    dtypes = ("float32", "float16")  # CUDA dtypes this code implements; the card picks the validated one

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...  # build from ctx.state_dict(), move to ctx.device

    def __call__(self, image1, image2, timestep):  # padded, scaled [B, 3, H, W] -> {"frame": [B, 3, H, W]}
        return {"frame": self.net(image1, image2, timestep)}


RUNNERS = {"portable": Portable}
```

The core does everything else: image conversion and checks, scaling and padding as the card's `[io]` says, cropping,
output checks, the cast to float32, batching, locking and the device guard.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from fastforks._core import runner as core
from fastforks._core.preprocess import Crop, crop, pad_to, scale
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    import torch

__all__ = ["Context", "Crop", "Forward", "NoParams", "Runner", "crop", "pad_to", "scale"]


class Runner(core.Runner):
    """One backend of a frame-interpolation model family."""

    @abc.abstractmethod
    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            image1: `[B, 3, H, W]` float32 on `ctx.device`, scaled and padded as the card's `[io]` says.
            image2: Same shape: the next frame.
            timestep: Time of the output frame, 0 = `image1`, 1 = `image2`.

        Returns:
            `{"frame": [B, 3, H, W]}` (any float dtype) in the input's value range.
        """

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """Pad a scaled `[B, 3, H, W]` batch as the card's `[io] pad` says."""
        return pad_to(x, self.ctx.io)

    def module(self) -> torch.nn.Module:
        """The plain-PyTorch network as an `nn.Module`, as `torch.export` / ONNX export see it.

        It maps `(image1, image2)` padded and scaled `[B, 3, H, W]` to the middle frame `[B, 3, H, W]` (time 0.5).
        Portable runners implement it.

        Raises:
            NotImplementedError: The backend has no exportable module.
        """
        return super().module()
