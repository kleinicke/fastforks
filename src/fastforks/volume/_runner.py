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
holds the locks and the device guard. `fastforks.volume._nnunet` has the shared nnU-Net v2 engine. Encoder families
(card outputs `["features"]`) implement `EncoderRunner` instead: `volume -> {"features": (map per stride, ...)}`;
promptable families (card outputs `["mask"]`) `InteractiveRunner`: a session state, prompts, the network patch call.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar

from fastforks._core import runner as core
from fastforks._core.runner import Context, NoParams

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch

__all__ = ["Context", "EncoderRunner", "InteractiveRunner", "NoParams", "Runner"]


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


class EncoderRunner(core.Runner):
    """One backend of a volume encoder family (pretrained feature maps, no labels).

    Attributes:
        channels: Input channels (modalities) the model takes.
        strides: Downsampling factor of every feature map, in output order.
        widths: Channels of every feature map, in output order.
        max_size: Largest side (voxels) a call takes, or `None`.
    """

    batch: ClassVar[int | None] = None  # one volume per call
    channels: int = 1
    strides: tuple[int, ...]
    widths: tuple[int, ...]
    max_size: int | None = None

    @abc.abstractmethod
    def __call__(self, volume: torch.Tensor) -> dict[str, tuple[torch.Tensor, ...]]:
        """Encode one volume.

        Args:
            volume: `[C, D, H, W]` float32 on `ctx.device` (raw intensities; the runner normalises).

        Returns:
            `{"features": (map, ...)}`, map `i` `[widths[i], ceil(D / strides[i]), ...]` on the device.
        """


class InteractiveRunner(core.Runner):
    """One backend of a promptable volume family: a session per volume, refined by prompts.

    The session methods are eager host code (prompt maps, patch placement); they run the network through
    `self(x)`, the only compiled entry point. Coordinates are `(z, y, x)` voxels of the input volume.

    Attributes:
        channels: Input channels (modalities) the model takes.
    """

    batch: ClassVar[int | None] = None  # one volume per session
    channels: int = 1

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """The network on one patch: input channels `[C, *patch]` float32 -> mask `[*patch]` uint8."""

    @abc.abstractmethod
    def start(self, volume: torch.Tensor) -> Any:  # noqa: ANN401 - the family's session state
        """A new session on a `[C, D, H, W]` float32 volume on `ctx.device` (raw intensities)."""

    @abc.abstractmethod
    def reset(self, s: Any) -> None:  # noqa: ANN401
        """Forget every prompt and the mask."""

    @abc.abstractmethod
    def point(self, s: Any, zyx: Sequence[float], positive: bool) -> None:  # noqa: ANN401, FBT001
        """Add a point prompt and update the mask."""

    @abc.abstractmethod
    def box(self, s: Any, box: Sequence[Sequence[float]], positive: bool) -> None:  # noqa: ANN401, FBT001
        """Add a box prompt `[[z0, z1), [y0, y1), [x0, x1)]` and update the mask."""

    @abc.abstractmethod
    def scribble(self, s: Any, mask: torch.Tensor, positive: bool) -> None:  # noqa: ANN401, FBT001
        """Add a scribble (`[D, H, W]` 0/1 on the device) and update the mask."""

    @abc.abstractmethod
    def mask(self, s: Any) -> torch.Tensor:  # noqa: ANN401
        """The current mask `[D, H, W]` bool on the input grid."""
