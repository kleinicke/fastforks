"""The interface a stereo model family implements: one `Runner` per backend, listed in `model.py` as `RUNNERS`.

```python
# stereo/models/<family>/model.py
@dataclass(frozen=True)
class Params:  # card [params] + variant params + [options] (+ load(options=)), checked at load
    iters: int


class Portable(Runner):
    dtypes = ("float32", "float16")  # CUDA dtypes this code implements; the card picks the validated one
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...  # build from ctx.state_dict(), move to ctx.device

    def __call__(self, left, right):  # padded, scaled [B, 3, H, W] -> {"disparity": [B, H, W], ...}
        return {"disparity": self.net(left, right)[:, 0]}


RUNNERS = {"portable": Portable, "kernels": Kernels}
```

The core does everything else: image conversion and checks, scaling and padding as the card's `[io]` says (unless
`io = "raw"`), cropping, output checks, the cast to float32, batching, locking and the device guard.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, ClassVar, Literal

from fastforks._core import runner as core
from fastforks._core.preprocess import Crop, crop, pad_to, pad_widths, scale
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    import torch

__all__ = ["Context", "Crop", "Forward", "NoParams", "Runner", "crop", "pad_to", "pad_widths", "scale"]


class Runner(core.Runner):
    """One backend of a stereo model family.

    Class attributes (besides those of `fastforks._core.runner.Runner`):
        io: `"core"`: `__call__` gets scaled, padded inputs and returns padded maps. `"raw"`: it gets `[B, 3, H, W]`
            RGB in `[0, 1]` and returns maps of that size (for kernels that fuse scaling and padding).
    """

    io: ClassVar[Literal["core", "raw"]] = "core"

    @abc.abstractmethod
    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            left: `[B, 3, H, W]` float32 on `ctx.device`: scaled and padded (`io = "core"`) or raw `[0, 1]`.
            right: Same shape.

        Returns:
            Output maps `[B, H, W]` by name (any float dtype); `"disparity"` is required, all must be in
            `card.outputs`.
        """

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """Pad a scaled `[B, 3, H, W]` batch; override for `pad.fill = "family"`."""
        return pad_to(x, self.ctx.io)

    def module(self) -> torch.nn.Module:
        """The plain-PyTorch network as an `nn.Module`, as `torch.export` / ONNX export see it.

        It maps `(left, right)` padded and scaled `[B, 3, H, W]` to the disparity `[B, H, W]`. Portable runners
        implement it.

        Raises:
            NotImplementedError: The backend has no exportable module (e.g. `kernels`).
        """
        return super().module()
