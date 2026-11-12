"""The interface a point-tracking model family implements: one `Runner` per backend, listed in `model.py` as `RUNNERS`.

```python
# tracking/models/<family>/model.py
class Portable(Runner):
    dtypes = ("float32", "float16")  # CUDA dtypes this code implements; the card picks the validated one
    Params = Params

    def __call__(self, frames, n, size, queries):  # frame(t) -> [1, 3, H, W]; queries [Q, 3] (t, x, y)
        return {"tracks": ..., "visibility": ...}  # [T, Q, 2] (x, y) input pixels, [T, Q] in [0, 1]


RUNNERS = {"portable": Portable}
```

The runner reads the frames it needs, resizes them to its network size and maps the tracks back itself. The core does
the rest: video and query conversion and checks, output checks, the cast to float32, locking and the device guard.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from fastforks._core import runner as core
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

__all__ = ["Context", "Forward", "NoParams", "Runner"]


class Runner(core.Runner):
    """One backend of a point-tracking model family."""

    @abc.abstractmethod
    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Track the query points through the video.

        Args:
            frames: `frames(t)` is frame `t` as `[1, 3, H, W]` float32 in `[0, 1]` on `ctx.device`.
            n: Number of frames `T`.
            size: `(H, W)` of every frame.
            queries: `[Q, 3]` float32 `(t, x, y)` on `ctx.device`: frame index (integer valued, `0 <= t < T`) and
                position in input pixels (pixel centres at integer coordinates).

        Returns:
            `{"tracks": [T, Q, 2]}` (any float dtype), `(x, y)` input pixels of every query in every frame, plus
            `"visibility"` `[T, Q]` in `[0, 1]` if the card lists it.
        """

    def module(self) -> torch.nn.Module:
        """The plain-PyTorch network as an `nn.Module`, as `torch.export` / ONNX export see it.

        It maps `(video [T, 3, H, W] in [0, 1], queries [Q, 3])` to the runner's `{name: tensor}`. Portable runners
        implement it.

        Raises:
            NotImplementedError: The backend has no exportable module.
        """
        return super().module()
