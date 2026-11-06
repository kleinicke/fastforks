"""Semantic segmentation API: `list_models`, `model_card`, `load`, and the model / result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import CardError
from fastforks._core.image import ImageLike, to_rgb_batch, unbatch
from fastforks._core.labels import label_set
from fastforks._core.model import Backend
from fastforks._core.outputs import MapOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.segmentation._runner import Runner

__all__ = ["Backend", "SegmentationModel", "SegmentationResult", "list_models", "load", "model_card"]

TASK = "segmentation"


@dataclass(frozen=True)
class SegmentationResult(MapOutputs):
    """Output of a semantic segmentation model, on the model's device, aligned with the input image.

    `save(path)` writes `labels` (`MapOutputs.save`): `.png` 8-bit (exact class indices), `.npy`, `.tif`, `.pfm`.

    Attributes:
        labels: Class index per pixel `[H, W]` int64 (`[B, H, W]` for a batch).
        logits: Class logits `[classes, H, W]` float32 if requested (`model(image, logits=True)`), else `None`; softmax
            over the classes gives a per-pixel distribution. Mask-classification models (Mask2Former, OneFormer,
            EoMT) score a class by its probability summed over the queries (HF / detectron2 `semantic_inference`);
            their `logits` are the log of that score (floored at 1e-30). `labels` is the argmax of these logits up to
            near-ties: a pixel whose two best classes differ by about the rounding of the logits may take either (the
            labels-only call can resize in another precision or kernel).
        class_names: Name of every class, index i = class i (the card's label set, e.g. ADE20K-150).
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    labels: torch.Tensor
    logits: torch.Tensor | None
    class_names: tuple[str, ...]
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    _save_fields = ("labels",)


class SegmentationModel(core.Model):
    """A loaded semantic segmentation model. Call it with an image (or a batch).

    Besides the attributes of `fastforks.Model`:

    Attributes:
        class_names: Name of every class, index i = class i.
    """

    _runner: Runner

    def __init__(self, runner: Runner, backend: Backend, options: Mapping[str, Any]) -> None:
        """Wrap a runner (built by `load`)."""
        super().__init__(runner, backend, options)
        self.class_names = label_set(self.variant.output_info["label_set"])

    def __call__(self, image: ImageLike, *, logits: bool = False) -> SegmentationResult:
        """Label every pixel of an image.

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).
            logits: Also return the class logits (`[classes, H, W]` float32 per image: large for big images).

        Returns:
            The result; batched inputs give batched maps.

        Raises:
            InputError: Unsupported or too small image.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)
        (h, w), c = x.shape[-2:], len(self.class_names)
        log_space = self._runner.probabilities

        def run(chunk: torch.Tensor) -> dict[str, torch.Tensor]:  # argmax per chunk: only labels are kept
            if not logits:  # the runner may skip the full-resolution logits
                labels = self._runner.labels(chunk)
                if tuple(labels.shape) != (chunk.shape[0], h, w) or labels.dtype != torch.int64:
                    want = (chunk.shape[0], h, w)
                    msg = f"{self.variant.ref}: labels {labels.dtype} {tuple(labels.shape)}, expected int64 {want}"
                    raise CardError(msg)
                return {"labels": labels}
            out = self._runner(chunk)
            scores = self._check_outputs(out, {"logits": (chunk.shape[0], c, h, w)}, cast=False)["logits"]
            labels = scores.argmax(1)  # on the raw scores: the floor below cannot create ties
            return {"labels": labels, "logits": scores.float().clamp_min(1e-30).log() if log_space else scores.float()}

        maps = unbatch(self._batched(run, x), image)
        return SegmentationResult(
            maps["labels"], maps.get("logits"), self.class_names, self.variant.ref, self.backend, self.options
        )


def bench_inputs(model: SegmentationModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]`, labels only (the default call)."""
    return (synthetic_image(size.height, size.width, model.device),), {}


list_models, model_card, load = task_api(TASK, SegmentationModel)
