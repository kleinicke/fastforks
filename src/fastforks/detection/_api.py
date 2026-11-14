"""Object detection API: `list_models`, `model_card`, `load`, and the model / result types."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, is_batch, to_rgb_batch
from fastforks._core.instances import Instances
from fastforks._core.labels import label_set
from fastforks._core.model import Backend

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fastforks.detection._runner import Runner

__all__ = ["Backend", "DetectionModel", "DetectionResult", "list_models", "load", "model_card"]

TASK = "detection"


@dataclass(frozen=True)
class DetectionResult(Instances):
    """Detections of one image, sorted by descending score, on the model's device.

    The fields are those of `fastforks._core.instances.Instances`; their names are the keys of the Hugging Face
    transformers `post_process_object_detection` dicts (`scores`, `labels`, `boxes`; mask models add `masks`).
    `len(result)` is the number of detections; `result.to_list()` gives one plain dict per detection.
    """


class DetectionModel(core.Model):
    """A loaded object detector. Call it with an image (or a batch).

    `module()` (portable) is the dense part of the network (backbone + feature pyramid): an RGB image `[B, 3, h, w]` in
    `[0, 1]` already at the network size (normalised and padded inside) -> the pyramid levels; the resize, proposal
    selection, NMS and mask pasting are data-dependent and not part of it.

    Besides the attributes of `fastforks.Model`:

    Attributes:
        class_names: Name of every class (index = `labels` value).
    """

    _runner: Runner

    def __init__(self, runner: Runner, backend: Backend, options: Mapping[str, Any]) -> None:
        """Wrap a runner (built by `load`)."""
        super().__init__(runner, backend, options)
        self.class_names = label_set(self.variant.output_info["label_set"])

    def __call__(
        self, image: ImageLike, *, score_threshold: float | None = None, text: Sequence[str] | None = None
    ) -> DetectionResult | list[DetectionResult]:
        """Detect the instances in an image (resized as the model was evaluated; see the card's `[params]`).

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).
            score_threshold: Minimum class score for this call (default: the load option `score_threshold`). It
                applies where the original post-processing applies it (R-CNN: before the per-class NMS).
            text: Open-vocabulary models only (`model.open_vocabulary`): the class names to detect, e.g.
                `["cat", "remote control"]`; `labels` then index into it and `class_names` is it. Default: the
                card's label set (e.g. the 80 COCO class names).

        Returns:
            One `DetectionResult` (its `options` include a call-time `score_threshold`); a batch (BHWC / BCHW)
            gives a list with one per image.

        Raises:
            InputError: Unsupported or too small image, a `score_threshold` that is not a finite number >= 0, or
                `text` that is empty, not a list of non-empty strings, or given to a closed-vocabulary model.
        """
        threshold = self._threshold(score_threshold)
        names = self._queries(text)
        x = to_rgb_batch(image, self.device, keep_uint8=self._runner.uint8_input)
        self._check_size(x)
        changed = score_threshold is not None
        options = core.ReadOnlyDict(self.options, score_threshold=threshold) if changed else self.options
        results = [self._detect(x[i : i + 1], threshold, options, names) for i in range(x.shape[0])]
        return results if is_batch(image) else results[0]

    @property
    def open_vocabulary(self) -> bool:
        """True if the model takes `text` class names at call time (OWLv2, Grounding DINO)."""
        return self._runner.open_vocabulary

    def _queries(self, text: Sequence[str] | None) -> tuple[str, ...] | None:
        if text is None:
            return None
        if not self.open_vocabulary:
            msg = f"{self.variant.ref} has a fixed label set and takes no text (open-vocabulary models do)"
            raise InputError(msg)
        names = (text,) if isinstance(text, str) else tuple(text)
        if not names or not all(isinstance(s, str) and s.strip() for s in names):
            msg = f"text must be a non-empty list of non-empty strings, got {text!r}"
            raise InputError(msg)
        return names

    def _threshold(self, value: float | None) -> float:
        if value is None:
            return getattr(self._runner.ctx.params, "score_threshold", 0.0)
        low = self.card.option_min.get("score_threshold", 0.0)
        high = self.card.option_max.get("score_threshold", 1.0)
        if (
            not isinstance(value, numbers.Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            msg = f"score_threshold must be a finite number from {low} to {high}, got {value!r}"
            raise InputError(msg)
        return float(value)

    def _detect(
        self, x: torch.Tensor, threshold: float, options: Mapping[str, Any], names: tuple[str, ...] | None
    ) -> DetectionResult:
        with self._session():
            out = self._runner(x, threshold) if names is None else self._runner(x, threshold, text=names)  # ty: ignore[unknown-argument]
        n = out["boxes"].shape[0] if "boxes" in out else 0
        shapes = {"boxes": (n, 4), "scores": (n,), "labels": (n,), "masks": (n, *x.shape[-2:])}
        self._check_outputs(out, shapes)
        masks = out.get("masks")
        return DetectionResult(
            boxes=out["boxes"].float(),
            scores=out["scores"].float(),
            labels=out["labels"].long(),
            masks=None if masks is None else masks.bool(),
            class_names=self.class_names if names is None else names,
            model=self.variant.ref,
            backend=self.backend,
            options=options,
        )


def bench_inputs(model: DetectionModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic uint8 image `[3, H, W]` (a photo's type; resized by the model as usual)."""
    return ((synthetic_image(size.height, size.width, model.device)[0] * 255).round().to(torch.uint8),), {}


list_models, model_card, load = task_api(
    TASK, DetectionModel, note="Load options (card `[options]`) include the post-processing, e.g. `score_threshold`."
)
