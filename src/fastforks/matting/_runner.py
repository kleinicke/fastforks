"""The interface a matting family implements: one `Runner` per backend, in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...

    def __call__(self, image, trimap):  # [B, 3, H, W] RGB in [0, 1]; None (trimap = "unused")
        x = scale(resize(image, self.ctx.params.size), self.ctx.io)
        return {"alpha": resize(self.net(x).sigmoid(), image.shape[-2:])[:, 0]}  # [B, H, W]
```

Matting families resize the image to their network size themselves, so a runner gets the raw image and returns the
alpha at its size. Trimap-guided families set `trimap = "required"` and get the trimap, mask-guided ones (video
matting) `mask = "required"` and get the mask in the same argument, the others `None`. Video families set `video` and
implement `track`.
The core does image conversion and checks, batching, output checks, locking and the device guard.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from fastforks._core import runner as core
from fastforks._core.preprocess import scale
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

__all__ = ["Context", "Forward", "NoParams", "Runner", "scale"]


class Runner(core.Runner):
    """One backend of a matting family.

    Class attributes (besides those of `fastforks._core.runner.Runner`):
        trimap: `"unused"`: the network takes the image only (passing a trimap is an error); `__call__` gets `None`.
            `"required"`: it needs a trimap; `__call__` gets it.
        mask: `"required"`: the network takes a binary mask of the object instead of a trimap (video matting from a
            first-frame mask); `__call__` gets it in place of the trimap. `"unused"`: no mask.
        video: The runner implements `track` (video matting from a first-frame mask).
    """

    trimap: ClassVar[Literal["unused", "required"]] = "unused"
    mask: ClassVar[Literal["unused", "required"]] = "unused"
    video: ClassVar[bool] = False

    @classmethod
    def input_use(cls, params: Any) -> dict[str, Any]:  # noqa: ANN401, ARG003 - the family's `Params`
        """`{"trimap": cls.trimap, "mask": cls.mask}`.

        Args:
            params: The variant's `Params`.
        """
        return {"trimap": cls.trimap, "mask": cls.mask}

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            image: `[B, 3, H, W]` float32 RGB in `[0, 1]` on `ctx.device`.
            trimap: With `trimap = "required"`: `[B, 1, H, W]` float32 in `[0, 1]` on `ctx.device` (0 = background,
                1 = foreground, 0.5 = unknown); with `mask = "required"`: the binary mask `[B, 1, H, W]` float32 in
                `{0, 1}` (1 = object); else `None`.

        Returns:
            `{"alpha": [B, H, W]}` (any float dtype) in `[0, 1]` at the input size.
        """

    def track(
        self, frames: Callable[[int], torch.Tensor], n_frames: int, size: tuple[int, int], mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Matte the object masked on frame 0 through a video (runners with `video = True`).

        Args:
            frames: Frame `t` as `[1, 3, H, W]` float32 RGB in `[0, 1]` on `ctx.device` (called once per frame, in
                order).
            n_frames: Number of frames `T`.
            size: `(H, W)` of the frames.
            mask: `[H, W]` float32 binary mask of the object on frame 0 (1 = object) on `ctx.device`.

        Returns:
            `{"alpha": [T, H, W] float32}` in `[0, 1]` on the CPU, filled per frame (device memory must not grow
            with `T`).

        Raises:
            NotImplementedError: The model does not matte video.
        """
        msg = f"{type(self).__name__} does not matte video"
        raise NotImplementedError(msg)
