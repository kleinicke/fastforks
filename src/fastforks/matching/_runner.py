"""The interface a matching model family implements: one `Runner` per backend, listed in `model.py` as `RUNNERS`.

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
from typing import TYPE_CHECKING, Any

from fastforks._core import runner as core
from fastforks._core.preprocess import scale
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    import torch

__all__ = ["Context", "Forward", "NoParams", "Runner", "scale"]


class Runner(core.Runner):
    """One backend of a matching model family (dense or sparse)."""

    @abc.abstractmethod
    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            image1: `[B, 3, H1, W1]` float32 on `ctx.device`, scaled as the card's `[io]` says.
            image2: `[B, 3, H2, W2]`, the image to match into (any size).

        Returns:
            `{"warp": [B, 2, H1, W1]}` (any float dtype): `(x, y)` in pixels of `image2` (pixel centres at integers) for
            every pixel of `image1`, plus `"certainty"` `[B, H1, W1]` in `[0, 1]` if the card lists it. Sparse
            families (card outputs `pairs`) get `B = 1` and return `{"keypoints1": [N, 2], "keypoints2": [M, 2],
            "pairs": [K, 2] (integer indices into the two keypoint sets), "scores": [K]}`, keypoints in input pixels.
        """

    match_reads_images = True
    """Whether `match` reads the images; if not, `match_pairs` decodes only images whose `describe` is not kept."""

    match_streams = 1
    """Pairs `match_pairs` matches at once on CUDA (a `match` that does not read the images), each on its own thread
    and stream: for a `match` that waits for the GPU often (LightGlue: adaptive depth and width sync every layer),
    the others launch meanwhile. `match` must be thread-safe."""

    def describe(self, image: torch.Tensor) -> object:  # noqa: ARG002 - the default: no per-image stage
        """The part of a call that depends on one image only (`[1, 3, H, W]`, scaled), or `None` if there is none.

        `MatchingModel.match_pairs` computes it once per image and passes the two results to `match`. Whatever it
        returns (tensors, lists / tuples / dicts of them) must not be modified by `match`.
        """
        return None

    def match(self, image1: torch.Tensor, image2: torch.Tensor, described: tuple[Any, Any]) -> dict[str, torch.Tensor]:
        """`__call__(image1, image2)`, bit-identical, from the two images' `describe` results."""
        raise NotImplementedError

    def module(self) -> torch.nn.Module:
        """The plain-PyTorch network as an `nn.Module`, as `torch.export` / ONNX export see it.

        It maps `(image1, image2)` scaled `[B, 3, H, W]` to the runner's `{name: map}`. Portable runners implement it.

        Raises:
            NotImplementedError: The backend has no exportable module.
        """
        return super().module()
