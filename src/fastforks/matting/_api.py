"""Matting API: `list_models`, `model_card`, `load`, and the model / result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.image import ImageLike, to_rgb_batch, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import MapOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    from fastforks.matting._runner import Runner

__all__ = ["Backend", "MattingModel", "MattingResult", "list_models", "load", "model_card"]

TASK = "matting"


@dataclass(frozen=True)
class MattingResult(MapOutputs):
    """Output of a matting model, on the model's device, aligned with the input image.

    `save(path)` writes `alpha` (`MapOutputs.save`): `.tif`, `.npy`, `.pfm`; for an 8-bit PNG mask write
    `fastforks.save("mask.png", (result.alpha * 255).round().byte())`.

    Attributes:
        alpha: Foreground alpha per pixel `[H, W]` float32 in `[0, 1]` (`[B, H, W]` for a batch); 1 = foreground.
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    alpha: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    _save_fields = ("alpha",)


class MattingModel(core.Model):
    """A loaded matting model. Call it with an image (or a batch).

    Has the attributes of `fastforks.Model`.
    """

    _runner: Runner

    def __call__(self, image: ImageLike) -> MattingResult:
        """Foreground alpha of an image.

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).

        Returns:
            The result; batched inputs give batched maps.

        Raises:
            InputError: Unsupported or too small image.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)

        def run(chunk: torch.Tensor) -> dict[str, torch.Tensor]:
            return self._check_outputs(self._runner(chunk), {"alpha": (chunk.shape[0], *chunk.shape[-2:])})

        maps = unbatch(self._batched(run, x), image)
        return MattingResult(maps["alpha"], self.variant.ref, self.backend, self.options)


def bench_inputs(model: MattingModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]`."""
    return (synthetic_image(size.height, size.width, model.device),), {}


list_models, model_card, load = task_api(TASK, MattingModel)
