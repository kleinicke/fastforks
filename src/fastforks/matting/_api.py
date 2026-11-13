"""Matting API: `list_models`, `model_card`, `load`, and the model / result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, to_rgb_batch, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import MapOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping

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

    def __call__(self, image: ImageLike, *, trimap: ImageLike | None = None) -> MattingResult:
        """Foreground alpha of an image.

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).
            trimap: Only for trimap-guided models (`contract().inputs["trimap"] == "required"`): a gray image of the
                same size (and batch, or one for all), 0 = background, 255 (1.0 for float) = foreground, 128 (0.5) =
                unknown; read like `image` (colour: the first channel).

        Returns:
            The result; batched inputs give batched maps.

        Raises:
            InputError: Unsupported or too small image; a trimap missing for a model that needs one, given to one that
                does not use it, or of another size or batch.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)
        t = self._trimap(trimap, x)

        def run(chunk: torch.Tensor, tri: torch.Tensor | None) -> dict[str, torch.Tensor]:
            out = self._runner(chunk, tri)
            return self._check_outputs(out, {"alpha": (chunk.shape[0], *chunk.shape[-2:])})

        maps = unbatch(self._batched(run, x, t), image)
        return MattingResult(maps["alpha"], self.variant.ref, self.backend, self.options)

    def _trimap(self, trimap: ImageLike | None, x: torch.Tensor) -> torch.Tensor | None:
        need = self._runner.trimap == "required"
        if trimap is None:
            if need:
                msg = f"{self.variant.ref} needs a trimap (gray image: 0 background, 128 unknown, 255 foreground)"
                raise InputError(msg)
            return None
        if not need:
            msg = f"{self.variant.ref} does not use a trimap; call it without one"
            raise InputError(msg)
        t = to_rgb_batch(trimap, self.device)[:, :1]
        if t.shape[-2:] != x.shape[-2:] or t.shape[0] not in {1, x.shape[0]}:
            msg = (
                f"trimap: {t.shape[0]} map(s) of {tuple(t.shape[-2:])} px do not match the image(s): {x.shape[0]} of "
                f"{tuple(x.shape[-2:])} px"
            )
            raise InputError(msg)
        return t.expand(x.shape[0], -1, -1, -1).contiguous()

    def example_inputs(
        self, height: int | None = None, width: int | None = None, *, batch: int = 1
    ) -> tuple[torch.Tensor, ...]:
        """Example inputs of `module()` (as the exports trace them): the synthetic image, plus a trimap if required.

        The trimap `[batch, 1, height, width]` marks a centred box as unknown (0.5), a ring around it as foreground (1)
        and the border as background (0).
        """
        (image,) = super().example_inputs(height, width, batch=batch)[:1]
        if self._runner.trimap != "required":
            return (image,)
        return image, synthetic_trimap(*image.shape[-2:], self.device).expand(image.shape[0], -1, -1, -1).contiguous()


def synthetic_trimap(height: int, width: int, device: torch.device) -> torch.Tensor:
    """`[1, 1, H, W]` float32 trimap: background border, foreground ring, unknown centre (`example_inputs`, bench)."""
    t = torch.zeros(1, 1, height, width, device=device)
    t[..., height // 8 : height - height // 8, width // 8 : width - width // 8] = 1.0
    t[..., height // 4 : height - height // 4, width // 4 : width - width // 4] = 0.5
    return t


def bench_inputs(model: MattingModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]`, plus a synthetic trimap if the model needs one."""
    image = synthetic_image(size.height, size.width, model.device)
    if model._runner.trimap != "required":  # noqa: SLF001 - the task's own runner contract
        return (image,), {}
    return (image,), {"trimap": synthetic_trimap(size.height, size.width, model.device)}


list_models, model_card, load = task_api(TASK, MattingModel)
