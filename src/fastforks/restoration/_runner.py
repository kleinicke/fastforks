"""The interface a restoration family implements: one `Runner` per backend, listed in `model.py` as `RUNNERS`.

```python
# restoration/models/<family>/model.py
@dataclass(frozen=True)
class Params:
    scale: int  # output size / input size (every family has it)


class Portable(Runner):
    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...  # build from ctx.state_dict(), move to ctx.device

    def __call__(self, image):  # padded, scaled [B, 3, H, W] -> {"image": [B, 3, scale * H, scale * W]}
        return {"image": self.net(image)}


RUNNERS = {"portable": Portable}
```

The core does everything else: image conversion and checks, scaling and padding as the card's `[io]` says, tiling,
cropping (`scale` x the padding), clamping to `[0, 1]`, output checks, the cast to float32, batching, locking and the
device guard.
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
    """One backend of a restoration family."""

    @property
    def scale(self) -> int:
        """Output size / input size (the family's `Params.scale`)."""
        return self.ctx.params.scale

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            image: `[B, 3, H, W]` float32 on `ctx.device`, scaled and padded as the card's `[io]` says.

        Returns:
            `{"image": [B, 3, scale * H, scale * W]}` (any float dtype) in the input's value range.
        """

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """Pad a scaled `[B, 3, H, W]` batch as the card's `[io] pad` says (fill `family`: override this)."""
        return pad_to(x, self.ctx.io)

    def module(self) -> torch.nn.Module:
        """The plain-PyTorch network as an `nn.Module`, as `torch.export` / ONNX export see it.

        It maps a padded and scaled `[B, 3, H, W]` batch to `[B, 3, scale * H, scale * W]`. Portable runners implement
        it.

        Raises:
            NotImplementedError: The backend has no exportable module.
        """
        return super().module()
