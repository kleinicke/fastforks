"""The interface a volume segmentation family implements: one `Runner` per backend, in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.class_names = {1: "spleen", ...}
        self.channels = 1

    def __call__(self, volume, spacing):  # [C, D, H, W] float32 on ctx.device, (z, y, x) mm
        return {"labels": ...}  # [D, H, W] integer labels on the same grid
```

The core converts the input (numpy / torch, `[D, H, W]` or `[C, D, H, W]`), checks the spacing and the outputs, and
holds the locks and the device guard. `fastforks.volume._nnunet` has the shared nnU-Net v2 engine.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, ClassVar

from fastforks._core import runner as core
from fastforks._core.runner import Context, NoParams

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch

__all__ = ["Context", "NoParams", "Runner"]


class Runner(core.Runner):
    """One backend of a volume segmentation family.

    Attributes:
        class_names: Label -> class name (background 0 not listed).
        channels: Input channels (modalities) the model takes.
    """

    batch: ClassVar[int | None] = None  # one volume per call
    class_names: Mapping[int, str]
    channels: int = 1

    @abc.abstractmethod
    def __call__(self, volume: torch.Tensor, spacing: Sequence[float]) -> dict[str, torch.Tensor]:
        """Segment one volume.

        Args:
            volume: `[C, D, H, W]` float32 on `ctx.device` (CT: Hounsfield units).
            spacing: Voxel size in mm along `D`, `H`, `W`.

        Returns:
            `{"labels": [D, H, W]}` uint8 (int16 above 255 classes) on the input grid.
        """
