"""Edges API: `list_models`, `model_card`, `load`, and the model / result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.image import ImageLike, is_batch, to_rgb_batch, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import MapOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    from fastforks.edges._runner import Runner

__all__ = ["Backend", "EdgesModel", "EdgesResult", "list_models", "load", "model_card"]

TASK = "edges"


@dataclass(frozen=True)
class EdgesResult(MapOutputs):
    """Output of an edges model, on the model's device, aligned with the input image.

    `save(path)` writes `edges` (`MapOutputs.save`): `.tif`, `.npy`, `.pfm`; for an 8-bit ControlNet hint write
    `fastforks.save("hint.png", (result.edges * 255).round().byte())`.

    Attributes:
        edges: Line map `[H, W]` float32 in `[0, 1]` (`[B, H, W]` for a batch); 1 = line (white lines on black, the
            ControlNet convention).
        lines: Line-segment models (M-LSD) only, else `None`: `[N, 4]` float32 segments `(x0, y0, x1, y1)` in input
            pixels (a list of them for a batch).
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    edges: torch.Tensor
    model: str
    backend: Backend
    lines: torch.Tensor | list[torch.Tensor] | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    _save_fields = ("edges",)


class EdgesModel(core.Model):
    """A loaded edges model. Call it with an image (or a batch).

    Has the attributes of `fastforks.Model`.
    """

    _runner: Runner

    def __call__(self, image: ImageLike) -> EdgesResult:
        """Line map of an image.

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).

        Returns:
            The result; batched inputs give batched maps (and a list of `lines`).

        Raises:
            InputError: Unsupported or too small image.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)
        segments: list[torch.Tensor] = []

        def run(chunk: torch.Tensor) -> dict[str, torch.Tensor]:
            out = dict(self._runner(chunk))
            lines = out.pop("lines", None)
            if lines is not None:
                segments.extend(t.float() for t in lines)
            return self._check_outputs(out, {"edges": (chunk.shape[0], *chunk.shape[-2:])})  # ty: ignore[invalid-argument-type] - lines popped

        maps = unbatch(self._batched(run, x), image)
        lines = (segments if is_batch(image) else segments[0]) if segments else None
        return EdgesResult(maps["edges"], self.variant.ref, self.backend, lines, self.options)


def bench_inputs(model: EdgesModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]`."""
    return (synthetic_image(size.height, size.width, model.device),), {}


list_models, model_card, load = task_api(TASK, EdgesModel)
