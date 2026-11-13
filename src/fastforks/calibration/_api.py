"""Calibration API: `list_models`, `model_card`, `load` and the model / result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.image import ImageLike, to_rgb_batch, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    from fastforks.calibration._runner import Runner

__all__ = ["Backend", "CalibrationModel", "CalibrationResult", "list_models", "load", "model_card"]

TASK = "calibration"

_STANDARD = ("intrinsics", "gravity", "roll", "pitch", "vfov")
_SHAPES = {"intrinsics": (3, 3), "gravity": (3,), "roll": (), "pitch": (), "vfov": ()}


@dataclass(frozen=True)
class CalibrationResult(NumpyOutputs):
    """Output of a calibration model, on the model's device (a batch adds a leading `B` to every field).

    Attributes:
        intrinsics: Pinhole matrix `[3, 3]` float32 in pixels of the input image; the principal point as the
            model fits it (GeoCalib: fixed at the image centre `(W / 2, H / 2)`).
        gravity: Unit gravity direction `[3]` in the camera frame (x right, y down, z forward), or `None`.
        roll: Roll in radians `[]` (rotation about the optical axis; 0 = level horizon), or `None`.
        pitch: Pitch in radians `[]` (tilt of the optical axis against the horizon, sign as upstream), or `None`.
        vfov: Vertical field of view in radians `[]`, or `None`.
        extras: Further model outputs by name (e.g. distortion parameters); leading dim as the image (none or `B`).
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    intrinsics: torch.Tensor
    gravity: torch.Tensor | None
    roll: torch.Tensor | None
    pitch: torch.Tensor | None
    vfov: torch.Tensor | None
    extras: Mapping[str, torch.Tensor]
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)


class CalibrationModel(core.Model):
    """A loaded calibration model. Call it with an image (or a batch)."""

    _runner: Runner

    def __call__(self, image: ImageLike) -> CalibrationResult:
        """Camera intrinsics (and gravity, if the model estimates it) of an image.

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`). Each image of
                a batch is calibrated on its own.

        Returns:
            The result.

        Raises:
            InputError: Unsupported or too small image.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)

        def run(chunk: torch.Tensor) -> dict[str, torch.Tensor]:
            out = self._runner(chunk)
            b = chunk.shape[0]
            shapes = {k: (b, *_SHAPES[k]) if k in _SHAPES else (b, *v.shape[1:]) for k, v in out.items()}
            return self._check_outputs(out, shapes)

        out = unbatch(self._batched(run, x), image)
        return CalibrationResult(
            **{name: out.pop(name, None) for name in _STANDARD},  # ty: ignore[invalid-argument-type]
            extras=out,
            model=self.variant.ref,
            backend=self.backend,
            options=self.options,
        )


def bench_inputs(model: CalibrationModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]` (resized by the model as usual)."""
    return (synthetic_image(size.height, size.width, model.device),), {}


list_models, model_card, load = task_api(TASK, CalibrationModel)
