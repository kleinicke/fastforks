"""Pose estimation API: `list_models`, `model_card`, `load`, and the model / result types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, is_batch, to_rgb_batch
from fastforks._core.instances import Instances
from fastforks._core.labels import label_set
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.pose._runner import Runner

__all__ = ["Backend", "PoseModel", "PoseResult", "list_models", "load", "model_card"]

TASK = "pose"


def _pairs(limbs: str) -> tuple[tuple[int, int], ...]:
    """`"a-b c-d"` -> `((a, b), (c, d))`."""
    return tuple((int(a), int(b)) for a, b in (limb.split("-") for limb in limbs.split()))


SKELETONS: dict[str, tuple[tuple[int, int], ...]] = {  # limbs as keypoint index pairs, per keypoint label set
    "coco-keypoints-17": _pairs(
        "15-13 13-11 16-14 14-12 11-12 5-11 6-12 5-6 5-7 6-8 7-9 8-10 1-2 0-1 0-2 1-3 2-4 3-5 4-6"
    ),
    # mmpose `configs/_base_/datasets/coco_wholebody.py` `skeleton_info`: the body, ankle -> toes / heel, and each
    # hand root -> its five fingers joint by joint (the 68 face points have no limbs)
    "coco-wholebody-133": _pairs(
        "15-13 13-11 16-14 14-12 11-12 5-11 6-12 5-6 5-7 6-8 7-9 8-10 1-2 0-1 0-2 1-3 2-4 3-5 4-6"
        " 15-17 15-18 15-19 16-20 16-21 16-22"
        " 91-92 92-93 93-94 94-95 91-96 96-97 97-98 98-99 91-100 100-101 101-102 102-103"
        " 91-104 104-105 105-106 106-107 91-108 108-109 109-110 110-111"
        " 112-113 113-114 114-115 115-116 112-117 117-118 118-119 119-120 112-121 121-122 122-123 123-124"
        " 112-125 125-126 126-127 127-128 112-129 129-130 130-131 131-132"
    ),
    # mmpose `halpe26.py`, `coco_wholebody_hand.py`, `ap10k.py` `skeleton_info`; `lapa.py` has none (face landmarks)
    "halpe-26": _pairs(
        "15-13 13-11 11-19 16-14 14-12 12-19 17-18 18-19 18-5 5-7 7-9 18-6 6-8 8-10 1-2 0-1 0-2 1-3 2-4 3-5 4-6"
        " 15-20 15-22 15-24 16-21 16-23 16-25"
    ),
    "coco-wholebody-hand-21": _pairs(
        "0-1 1-2 2-3 3-4 0-5 5-6 6-7 7-8 0-9 9-10 10-11 11-12 0-13 13-14 14-15 15-16 0-17 17-18 18-19 19-20"
    ),
    "lapa-106": (),
    "ap10k-17": _pairs("0-1 0-2 1-2 2-3 3-4 3-5 5-6 6-7 3-8 8-9 9-10 4-11 11-12 12-13 4-14 14-15 15-16"),
}

BoxesLike = torch.Tensor | np.ndarray | Sequence[Sequence[float]] | Instances
"""Person boxes `[P, 4]` `(x0, y0, x1, y1)` in input pixels: a tensor, array, nested list, or a detection result."""


@dataclass(frozen=True)
class PoseResult(NumpyOutputs):
    """Keypoints of the people in one image, on the model's device.

    Attributes:
        keypoints: `[P, K, 2]` float32 keypoint positions `(x, y)` in pixels of the input image (pixel centres at
            integer coordinates).
        keypoint_scores: `[P, K]` float32 keypoint confidence (heatmap models: the heatmap peak, about 0..1).
        boxes: `[P, 4]` float32 person boxes `(x0, y0, x1, y1)` in input pixels (top-down: the boxes given).
        scores: `[P]` float32 person confidence (top-down: the mean keypoint score).
        keypoint_names: Name of every keypoint (the card's label set), index = keypoint `K`.
        skeleton: Limbs as pairs of keypoint indices (for drawing).
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    keypoints: torch.Tensor
    keypoint_scores: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    keypoint_names: tuple[str, ...]
    skeleton: tuple[tuple[int, int], ...]
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of people."""
        return int(self.scores.shape[0])

    def to_list(self) -> list[dict[str, Any]]:
        """One dict per person: `box`, `score` and `keypoints` (`{name: (x, y, score)}`)."""
        kps, kss = self.keypoints.tolist(), self.keypoint_scores.tolist()
        return [
            {
                "box": box,
                "score": score,
                "keypoints": {n: (*xy, s) for n, xy, s in zip(self.keypoint_names, kp, ks, strict=True)},
            }
            for box, score, kp, ks in zip(self.boxes.tolist(), self.scores.tolist(), kps, kss, strict=True)
        ]


class PoseModel(core.Model):
    """A loaded pose estimator. Call it with an image (or a batch) and, for top-down models, the person boxes.

    `module()` (portable) is the network on person crops: RGB `[B, 3, h, w]` in `[0, 1]` at the network size (e.g.
    256x192) -> heatmaps `[B, K, h / 4, w / 4]`; the cropping and the keypoint decoding are not part of it.

    Besides the attributes of `fastforks.Model`:

    Attributes:
        keypoint_names: Name of every keypoint (index = keypoint).
        skeleton: Limbs as pairs of keypoint indices.
        needs_boxes: Whether calls need person boxes (top-down models).
    """

    _runner: Runner

    def __init__(self, runner: Runner, backend: Backend, options: Mapping[str, Any]) -> None:
        """Wrap a runner (built by `load`)."""
        super().__init__(runner, backend, options)
        names = self.variant.output_info["label_set"]
        self.keypoint_names = label_set(names)
        self.skeleton = SKELETONS[names]  # a new label set comes with its limbs
        self.needs_boxes = runner.needs_boxes

    def __call__(
        self, image: ImageLike, boxes: BoxesLike | Sequence[BoxesLike] | None = None
    ) -> PoseResult | list[PoseResult]:
        """Estimate the keypoints of the people in an image.

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).
            boxes: Top-down models: one box per person, `[P, 4]` `(x0, y0, x1, y1)` in input pixels (a tensor, array,
                list, or a `fastforks.detection` result); a batch takes a list with one per image. `[]` (no people)
                gives an empty result.

        Returns:
            One `PoseResult` (the people in box order); a batch (BHWC / BCHW) gives a list with one per image.

        Raises:
            InputError: Unsupported image, missing, malformed or flipped boxes, or boxes for a model that finds the
                people itself.
        """
        x = to_rgb_batch(image, self.device)
        per_image = self._boxes(boxes, x.shape[0], batch=is_batch(image))
        results = [self._estimate(x[i : i + 1], b) for i, b in enumerate(per_image)]
        return results if is_batch(image) else results[0]

    def _boxes(self, boxes: Any, n: int, *, batch: bool) -> list[torch.Tensor | None]:  # noqa: ANN401
        if not self.needs_boxes:
            if boxes is not None:
                msg = f"{self.variant.ref} finds the people itself; it takes no boxes"
                raise InputError(msg)
            return [None] * n
        if boxes is None:
            msg = (
                f"{self.variant.ref} is a top-down model: pass boxes=[P, 4] person boxes (x0, y0, x1, y1), e.g. "
                "from fastforks.detection (the person class) or ground truth"
            )
            raise InputError(msg)
        if not batch:
            return [_boxes_tensor(boxes, self.device)]
        if isinstance(boxes, Instances | torch.Tensor | np.ndarray) or not isinstance(boxes, Sequence):
            msg = f"a batch of {n} images takes a list of {n} box arrays (one per image), got {type(boxes).__name__}"
            raise InputError(msg)
        if len(boxes) != n:
            msg = f"a batch of {n} images takes {n} box arrays, got {len(boxes)}"
            raise InputError(msg)
        return [_boxes_tensor(b, self.device) for b in boxes]

    def _estimate(self, x: torch.Tensor, boxes: torch.Tensor | None) -> PoseResult:
        k = len(self.keypoint_names)
        if boxes is not None and boxes.shape[0] == 0:
            out = {"keypoints": x.new_zeros(0, k, 2), "keypoint_scores": x.new_zeros(0, k), "scores": x.new_zeros(0)}
        else:
            with self._session():
                out = self._runner(x, boxes)
        p = out["keypoints"].shape[0]
        shapes = {"keypoints": (p, k, 2), "keypoint_scores": (p, k), "boxes": (p, 4), "scores": (p,)}
        out = self._check_outputs(out, shapes)
        return PoseResult(
            keypoints=out["keypoints"],
            keypoint_scores=out["keypoint_scores"],
            boxes=out["boxes"] if boxes is None else boxes.clone(),  # not the caller's tensor (a view of their input)
            scores=out["scores"],
            keypoint_names=self.keypoint_names,
            skeleton=self.skeleton,
            model=self.variant.ref,
            backend=self.backend,
            options=self.options,
        )


def _boxes_tensor(value: Any, device: torch.device) -> torch.Tensor:  # noqa: ANN401
    """Person boxes -> `[P, 4]` float32 on `device`.

    Raises:
        InputError: Not a finite real `[P, 4]` array, or a flipped box (`x1 < x0` or `y1 < y0`).
    """
    if isinstance(value, Instances):
        value = value.boxes
    try:
        t = torch.as_tensor(np.asarray(value, dtype=np.float64) if isinstance(value, list | tuple) else value)
    except (TypeError, ValueError, RuntimeError) as e:
        msg = f"boxes: not a numeric array ({e})"
        raise InputError(msg) from e
    if t.numel() == 0:
        return torch.zeros(0, 4, device=device)
    if t.is_complex() or t.dtype == torch.bool or t.ndim != 2 or t.shape[1] != 4:  # noqa: PLR2004 - (x0, y0, x1, y1)
        msg = f"boxes must be a real [P, 4] array (x0, y0, x1, y1), got {t.dtype} {list(t.shape)}"
        raise InputError(msg)
    t = t.to(dtype=torch.float32)  # checked where it is (host boxes: no device sync), then moved
    if not bool(torch.isfinite(t).all()) or not bool(torch.isfinite(t[:, 2:] - t[:, :2]).all()):
        msg = "boxes has non-finite values (or a width / height that overflows float32)"
        raise InputError(msg)
    if bad := torch.nonzero((t[:, 2] < t[:, 0]) | (t[:, 3] < t[:, 1]))[:, 0].tolist():
        msg = f"boxes are (x0, y0, x1, y1) with x1 >= x0 and y1 >= y0; box {bad[0]} is {t[bad[0]].tolist()}"
        raise InputError(msg)
    return t.to(device)


def bench_inputs(model: PoseModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[3, H, W]` and (top-down) one person box per 128 px of width."""
    h, w = size.height, size.width
    if not model.needs_boxes:
        return (synthetic_image(h, w, model.device)[0],), {}
    n = max(1, w // 128)
    x0 = torch.arange(n, dtype=torch.float32) * (w / n)
    boxes = torch.stack((x0, torch.full_like(x0, h * 0.1), x0 + w / n * 0.9, torch.full_like(x0, h * 0.95)), -1)
    return (synthetic_image(h, w, model.device)[0],), {"boxes": boxes.to(model.device)}


list_models, model_card, load = task_api(TASK, PoseModel)
