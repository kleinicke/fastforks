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
alpha at its size. Trimap-guided families set `trimap = "required"` and get the trimap, the others `None`.
The core does image conversion and checks, batching, output checks, locking and the device guard.
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
    """One backend of a matting family.

    Class attributes (besides those of `fastforks._core.runner.Runner`):
        trimap: `"unused"`: the network takes the image only (passing a trimap is an error); `__call__` gets `None`.
            `"required"`: it needs a trimap; `__call__` gets it.
    """

    trimap: ClassVar[Literal["unused", "required"]] = "unused"

    @classmethod
    def input_use(cls, params: Any) -> dict[str, Any]:  # noqa: ANN401, ARG003 - the family's `Params`
        """`{"trimap": cls.trimap}`.

        Args:
            params: The variant's `Params`.
        """
        return {"trimap": cls.trimap}

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            image: `[B, 3, H, W]` float32 RGB in `[0, 1]` on `ctx.device`.
            trimap: With `trimap = "required"`: `[B, 1, H, W]` float32 in `[0, 1]` on `ctx.device` (0 = background,
                1 = foreground, 0.5 = unknown); else `None`.

        Returns:
            `{"alpha": [B, H, W]}` (any float dtype) in `[0, 1]` at the input size.
        """
