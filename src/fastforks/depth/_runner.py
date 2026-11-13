"""The interface a monocular depth family implements: one `Runner` per backend, listed in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    Params = Params
    intrinsics = "optional"  # the network can use known intrinsics

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...

    def __call__(self, image, intrinsics):  # [B, 3, H, W] RGB in [0, 1], [B, 3, 3] or None
        x = resize_to_multiple(scale(image, self.ctx.io), 14)  # family-specific input handling
        return {"depth": upsample(self.net(x), image.shape[-2:])}  # maps at the input size
```

Depth families differ in how they resize their input (fixed size, multiple of 14, keep aspect...), so a runner gets
the raw image and returns maps at its size; `fastforks._core.preprocess` has the shared helpers. The core does image
conversion and checks, batching, output checks, dtype casts, locking and the device guard.

Portable runners also implement `module()` (the plain network for export); kernels runners may omit it.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from fastforks._core import runner as core
from fastforks._core.preprocess import scale
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    import torch

__all__ = ["Context", "Forward", "NoParams", "Runner", "scale"]


class Runner(core.Runner):
    """One backend of a monocular depth family.

    Class attributes (besides those of `fastforks._core.runner.Runner`):
        intrinsics: `"unused"`: the network ignores camera intrinsics (passing them is an error). `"optional"`: it
            uses them if given (else it estimates them or assumes a default). `"required"`: it needs them.
    """

    intrinsics: ClassVar[Literal["unused", "optional", "required"]] = "unused"
    prompt_depth: ClassVar[Literal["unused", "required"]] = "unused"

    @classmethod
    def input_use(cls, params: Any) -> dict[str, Any]:  # noqa: ANN401, ARG003 - the family's `Params`
        """`{"intrinsics": cls.intrinsics}`, plus `"prompt_depth"` for runners that take one.

        Args:
            params: The variant's `Params`.
        """
        if cls.prompt_depth == "unused":
            return {"intrinsics": cls.intrinsics}
        return {"intrinsics": cls.intrinsics, "prompt_depth": cls.prompt_depth}

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            image: `[B, 3, H, W]` float32 RGB in `[0, 1]` on `ctx.device`.
            intrinsics: `[B, 3, 3]` float32 pinhole matrices in pixels of `image`, or `None`.

        Returns:
            Maps at the input size, among the card's `outputs`: `"depth"`, `"inverse_depth"`, `"confidence"`
            (`[B, H, W]`), `"mask"` (`[B, H, W]` bool), `"points"`, `"normals"` (`[B, H, W, 3]`) and `"intrinsics"`
            (`[B, 3, 3]`, estimated).
        """
