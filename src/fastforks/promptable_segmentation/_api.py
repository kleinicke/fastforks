"""Promptable segmentation API: `list_models`, `model_card`, `load`, and the model / embedding / result types."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.arguments import typo_checked_load
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError, ModelNotFoundError
from fastforks._core.image import ImageLike, is_batch, to_rgb_batch
from fastforks._core.instances import Instances
from fastforks._core.model import Backend, Dtype, ModeArg
from fastforks._core.outputs import MapOutputs, NumpyOutputs

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping, Sequence

    from fastforks.promptable_segmentation._runner import Runner

__all__ = [
    "Backend",
    "ConceptModelId",
    "ConceptSegmentationModel",
    "ConceptSegmentationResult",
    "ImageEmbeddings",
    "InContextModelId",
    "InContextSegmentationModel",
    "InContextSegmentationResult",
    "PromptLike",
    "PromptableSegmentationModel",
    "PromptableSegmentationResult",
    "VideoSegmentationResult",
    "list_models",
    "load",
    "model_card",
]

TASK = "promptable_segmentation"
PromptLike = torch.Tensor | np.ndarray | list[Any] | tuple[Any, ...]
"""A prompt array: torch tensor, numpy array or nested lists / tuples of numbers."""
ConceptModelId = Literal["sam3", "sam3@1"]
"""Ids of the concept models (text / exemplar prompts): `load` returns a `ConceptSegmentationModel` for them."""
InContextModelId = Literal["insid3-vitl16", "insid3-vitl16@1"]
"""Ids of the in-context models (reference images + masks): `load` returns an `InContextSegmentationModel` for them."""
_MULTIMASK = 3  # candidate masks per prompt of SAM / SAM 2 with multimask_output (num_multimask_outputs)
MAX_EXEMPLARS = 256  # fixed: exemplar boxes per call (the box tokens, ROI pooling and attention grow with N)


CONCEPT_THRESHOLD = 0.5  # fixed: SAM 3's default instance score threshold (scores are in [0, 1])


def _threshold(value: object) -> float:
    """The concept threshold: a number in [0, 1] (the score range; as detection's `score_threshold`)."""
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or not 0 <= float(value) <= 1:
        msg = f"threshold must be a number in [0, 1] (scores are probabilities), got {value!r}"
        raise InputError(msg)
    return float(value)


def _boxes_tensor(t: torch.Tensor) -> torch.Tensor:
    """Box prompts: no boxes (`[]` from a detector that found nothing) become `[0, 4]`; flipped boxes raise."""
    if t.numel() == 0:
        return t.reshape(0, 4)
    if t.device.type == "cpu" and t.shape[-1] == 4:  # noqa: PLR2004 - no device sync for a prompt already on the GPU
        _check_flipped(t.reshape(-1, 4))
    return t


def _check_flipped(boxes: torch.Tensor) -> None:
    """Raise `InputError` for a box `(x0, y0, x1, y1)` with `x1 < x0` or `y1 < y0` (a flipped box)."""
    if bad := torch.nonzero((boxes[:, 2] < boxes[:, 0]) | (boxes[:, 3] < boxes[:, 1]))[:, 0].tolist():
        msg = f"input_boxes are (x0, y0, x1, y1) with x1 >= x0 and y1 >= y0; box {bad[0]} is {boxes[bad[0]].tolist()}"
        raise InputError(msg)


_PROMPTS = {  # argument -> (prompt kind, dims with the prompt batch P, dtype)
    "input_points": ("points", 3, torch.float32),
    "input_labels": ("points", 2, torch.int64),
    "input_boxes": ("boxes", 2, torch.float32),
    "input_masks": ("masks", 4, torch.float32),
}


@dataclass(frozen=True)
class ImageEmbeddings:
    """Image features (HF: `get_image_embeddings`), reusable across prompt calls of the same loaded model.

    `torch.load` of a saved one needs `torch.serialization.safe_globals([ImageEmbeddings])` (or `weights_only=False`
    for your own file); results load once their task module is imported.

    One object per `get_image_embeddings` call; for a batch it holds every image (`len(emb)`), and `emb[i]` is the
    embedding of image `i`. A prompt call takes a single-image embedding.

    Attributes:
        features: The backend's feature tensors, image batch first, on the model's device (opaque; layout differs per
            family).
        image_size: `(H, W)` of the embedded image(s).
        model: Versioned model id that computed them.
        backend: The backend that computed them.
        options: Load options that differ from the validated defaults (prompt calls must use the same).
    """

    features: Mapping[str, torch.Tensor]
    image_size: tuple[int, int]
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of images."""
        return int(next(iter(self.features.values())).shape[0])

    def __getitem__(self, i: int) -> ImageEmbeddings:
        """The embedding of image `i`."""
        n = len(self)
        if not -n <= i < n:
            msg = f"image index {i} out of range for {n} embedded image(s)"
            raise IndexError(msg)
        i %= n
        return replace(self, features={k: v[i : i + 1] for k, v in self.features.items()})


@dataclass(frozen=True)
class PromptableSegmentationResult(MapOutputs):
    """Masks for a set of prompts on one image, on the model's device.

    `save(path)` writes the mask of a one-prompt, one-mask result (`MapOutputs.save`; `.png` is a 1-bit black / white
    image); for several masks select one: `fastforks.save("m.png", result.masks[p, m])`.

    `P` (prompt batch) is present if the prompts had it (`input_points [P, N, 2]`, `input_boxes [P, 4]`, ...), `M` is
    3 for `multimask_output=True` (SAM / SAM 2: three candidate masks per prompt), else 1.

    Attributes:
        masks: `[P, M, H, W]` bool masks at the input size (logits upsampled and thresholded at 0, as the original
            `postprocess_masks` + `mask_threshold`; HF `post_process_masks`).
        iou_scores: `[P, M]` float32 predicted mask IoU (HF: `iou_scores`).
        pred_masks: `[P, M, h, w]` float32 low-resolution mask logits (HF `pred_masks`, without the image batch);
            pass one as `input_masks` (`[P, 1, h, w]`) to refine.
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    masks: torch.Tensor
    iou_scores: torch.Tensor
    pred_masks: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    _save_fields = ("masks",)


@dataclass(frozen=True)
class VideoSegmentationResult(NumpyOutputs):
    """Masks of the objects prompted on frame 0, for every frame of a video (`track`), on the CPU.

    The CPU, not the model's device: the result grows with the video length, the device memory does not.

    `P` (objects) is present if the prompts had it (`input_points [P, N, 2]`, `input_boxes [P, 4]`,
    `input_masks [P, H, W]`).

    Attributes:
        masks: `[T, P, H, W]` bool masks at the video size (the original `propagate_in_video` masks `> 0`).
        pred_masks: `[T, P, h, w]` float32 low-resolution mask logits (after the original small-hole filling).
        object_score_logits: `[T, P]` float32 object presence logits (`> 0`: the object is visible).
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    masks: torch.Tensor
    pred_masks: torch.Tensor
    object_score_logits: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConceptSegmentationResult(Instances):
    """Every instance of a concept prompt in one image (SAM 3), sorted by descending score, on the model's device.

    The fields are those of `fastforks._core.instances.Instances` (`labels` is `None`, `class_names` empty); the names
    are the keys of the Hugging Face transformers SAM 3 `post_process_instance_segmentation` dicts. `masks` are
    `[N, H, W]` bool at the input size.
    """


@dataclass(frozen=True)
class InContextSegmentationResult(MapOutputs):
    """The target-image mask of an in-context call (INSID3), on the model's device.

    Attributes:
        masks: `[H, W]` bool mask at the target image size.
        similarity: `[h, w]` float32 cosine similarity of each target patch to the reference prototype (after the
            positional debiasing; `> 0` is the forward match), at the backbone's patch grid.
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    masks: torch.Tensor
    similarity: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    _save_fields = ("masks",)


class PromptableSegmentationModel(core.Model):
    """A loaded promptable segmentation model: embed an image once, then segment it with prompts.

    SAM / SAM 2 take point / box / mask prompts and return masks per prompt; SAM 3 takes a concept prompt (`text=`
    and / or exemplar boxes `input_boxes=`, positive or negative) and returns every instance of it.

    Besides the attributes of `fastforks.Model`:

    Attributes:
        prompts: Prompt kinds the model takes (`"points"`, `"boxes"`, `"masks"`; `"text"`: a concept model, whose
            `"boxes"` are exemplar boxes).
    """

    _runner: Runner

    def __init__(self, runner: Runner, backend: Backend, options: Mapping[str, Any]) -> None:
        """Wrap a runner (built by `load`)."""
        super().__init__(runner, backend, options)
        self.prompts = runner.prompts

    def get_image_embeddings(self, image: ImageLike) -> ImageEmbeddings:
        """Compute the image features once (the expensive part); prompt calls on them are cheap.

        Args:
            image: Image (path, PIL, numpy HWC, torch CHW; see `fastforks._core.image`). A batch (`BHWC` / `BCHW`)
                gives one object holding every image; index it (`emb[i]`) for the prompt calls. The encoder runs
                in chunks of the runner's batch size (one image at a time for SAM / SAM 2).

        Raises:
            InputError: Unsupported or too small image.
        """
        return self._embed_tensor(self._to_batch(image))

    def _to_batch(self, image: ImageLike) -> torch.Tensor:
        """`image` as a checked `[B, 3, H, W]` batch on the device."""
        x = to_rgb_batch(image, self.device)
        self._check_size(x)
        return x

    def _embed_tensor(self, x: torch.Tensor) -> ImageEmbeddings:
        """The image encoder on a batch from `_to_batch`."""
        size = (int(x.shape[-2]), int(x.shape[-1]))
        b = max(1, self._runner.batch or x.shape[0])
        with self._session():
            parts = [self._runner.embed(x[i : i + b]) for i in range(0, x.shape[0], b)]
        features = parts[0] if len(parts) == 1 else {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        return ImageEmbeddings(features, size, self.variant.ref, self.backend, self.options)

    @overload
    def __call__(
        self,
        image: ImageLike | ImageEmbeddings,
        *,
        input_points: PromptLike | None = None,
        input_labels: PromptLike | None = None,
        input_boxes: PromptLike | None = None,
        input_masks: PromptLike | None = None,
        multimask_output: bool = True,
        text: None = None,
    ) -> PromptableSegmentationResult: ...

    @overload
    def __call__(
        self,
        image: ImageLike | ImageEmbeddings,
        *,
        text: str,
        input_boxes: PromptLike | None = None,
        input_boxes_labels: PromptLike | None = None,
        threshold: float | None = None,
    ) -> ConceptSegmentationResult: ...

    @overload
    def __call__(
        self,
        image: ImageLike | ImageEmbeddings,
        *,
        input_boxes: PromptLike,
        input_boxes_labels: PromptLike,
        threshold: float | None = None,
    ) -> ConceptSegmentationResult: ...

    def __call__(  # noqa: PLR0913 - one argument per prompt kind (HF names)
        self,
        image: ImageLike | ImageEmbeddings,
        *,
        input_points: PromptLike | None = None,
        input_labels: PromptLike | None = None,
        input_boxes: PromptLike | None = None,
        input_masks: PromptLike | None = None,
        multimask_output: bool = True,
        text: str | None = None,
        input_boxes_labels: PromptLike | None = None,
        threshold: float | None = None,
    ) -> PromptableSegmentationResult | ConceptSegmentationResult:
        """Segment one image with prompts (HF `SamModel` / `Sam3Model` argument names, one image).

        Concept models (`"text"` in `prompts`, SAM 3) take `text` and / or exemplar `input_boxes` (+
        `input_boxes_labels`) and return every instance of the concept; the other models take point / box / mask
        prompts and return masks per prompt.

        Args:
            image: One image, or its `get_image_embeddings` (reuse it for many prompt calls).
            input_points: `[P, N, 2]` (or `[N, 2]`) point `(x, y)` in pixels of the input image.
            input_labels: `[P, N]` (or `[N]`) point labels: 1 foreground, 0 background, -1 padding.
            input_boxes: `[P, 4]` (or `[4]`) box `(x0, y0, x1, y1)` in pixels of the input image. Concept models:
                exemplar boxes `[N, 4]` (or `[4]`), alone (the original text "visual") or with `text`.
            input_masks: `[P, 1, h, w]` (or `[1, h, w]`) low-resolution mask logits, e.g. a previous `pred_masks`.
            multimask_output: Three candidate masks per prompt (best for a single ambiguous point) instead of one.
            text: Concept prompt (SAM 3, the original `set_text_prompt`), e.g. `"yellow school bus"`.
            input_boxes_labels: Concept models: `[N]` (or a scalar) exemplar labels, 1 positive, 0 negative
                (default: all positive).
            threshold: Concept models: keep instances scoring above this, in [0, 1] (default 0.5, as the original
                `confidence_threshold`); `result.options["threshold"]` records the one used.

        Returns:
            The masks; `P` is dropped if no prompt had it, and an empty prompt batch (`P = 0`, e.g. boxes `[0, 4]`)
            gives an empty result. Concept models: the instances.

        Raises:
            InputError: Unsupported image, missing / inconsistent / unsupported prompts, or embeddings of another
                model.
        """
        given = {
            "input_points": input_points,
            "input_labels": input_labels,
            "input_boxes": input_boxes,
            "input_masks": input_masks,
        }
        given = {k: v for k, v in given.items() if v is not None}
        if "text" in self.prompts:
            return self._concept(image, text, given, input_boxes_labels, threshold)
        concept = {"text": text, "input_boxes_labels": input_boxes_labels, "threshold": threshold}
        if text is not None:
            msg = f"{self.variant.ref} takes no text prompts (takes: {sorted(self.prompts)})"
            raise InputError(msg)
        if extra := sorted(k for k, v in concept.items() if v is not None):
            msg = f"{self.variant.ref} is not a concept model: {extra} apply to text / exemplar prompts only"
            raise InputError(msg)
        prompts, batched = self._prompts(given)
        emb = self._embedding(image)
        p = next(iter(prompts.values())).shape[0]
        if p == 0:  # an empty prompt batch (e.g. a detector found nothing): an empty result, no model run
            return self._no_masks(emb, multimask_output)
        with self._session():
            out = self._runner.predict(
                dict(emb.features),
                emb.image_size,
                points=prompts.get("input_points"),
                labels=prompts.get("input_labels"),
                boxes=prompts.get("input_boxes"),
                masks=prompts.get("input_masks"),
                multimask=multimask_output,
            )
            m = out["iou_scores"].shape[-1] if "iou_scores" in out and multimask_output else 1
            self._check_outputs(out, {"pred_masks": (p, m, *self._runner.mask_size), "iou_scores": (p, m)})
            low = out["pred_masks"].float()
            masks = self._runner.upsample(low, emb.image_size) > 0
            if (c := self._runner.low_res_clamp) is not None:
                low = low.clamp(-c, c)
        maps = {"masks": masks, "iou_scores": out["iou_scores"].float(), "pred_masks": low}
        if not batched:
            maps = {k: v[0] for k, v in maps.items()}
        return PromptableSegmentationResult(
            maps["masks"], maps["iou_scores"], maps["pred_masks"], self.variant.ref, self.backend, self.options
        )

    def track(  # noqa: C901 - input checks per prompt kind
        self,
        video: Sequence[ImageLike] | np.ndarray | torch.Tensor,
        *,
        input_points: PromptLike | None = None,
        input_labels: PromptLike | None = None,
        input_boxes: PromptLike | None = None,
        input_masks: PromptLike | None = None,
    ) -> VideoSegmentationResult:
        """Track objects prompted on the first frame through a video (models with `video`, SAM 2).

        The original `SAM2VideoPredictor`: the prompts go to frame 0 (`add_new_points_or_box` / `add_new_mask`,
        one object per prompt), then `propagate_in_video` forward. Frames are encoded one at a time.

        Args:
            video: The frames in order: a sequence of images (paths, PIL, numpy HWC, torch CHW; one size), or a
                `[T, H, W, 3]` numpy / `[T, 3, H, W]` torch batch.
            input_points: `[P, N, 2]` (or `[N, 2]`) point `(x, y)` in pixels of frame 0.
            input_labels: `[P, N]` (or `[N]`) point labels: 1 foreground, 0 background, -1 padding.
            input_boxes: `[P, 4]` (or `[4]`) box `(x0, y0, x1, y1)` in pixels of frame 0.
            input_masks: `[P, H, W]` (or `[H, W]`) binary masks of frame 0 (nonzero = object), instead of points /
                boxes.

        Returns:
            The masks per frame and object; `P` is dropped if no prompt had it.

        Raises:
            InputError: The model does not track video, an empty or mixed-size video, or missing / inconsistent
                prompts.
        """
        if not self._runner.video:
            msg = f"{self.variant.ref} does not track video"
            raise InputError(msg)
        if isinstance(video, str | bytes) or not hasattr(video, "__len__") or not hasattr(video, "__getitem__"):
            msg = f"video must be a sequence of frames or a [T, ...] frame batch, got {type(video).__name__}"
            raise InputError(msg)
        n = len(video)
        if n == 0:
            msg = "video has no frames"
            raise InputError(msg)
        first = self._frame(video, 0)
        size = (int(first.shape[-2]), int(first.shape[-1]))

        def frame(t: int) -> torch.Tensor:
            x = first if t == 0 else self._frame(video, t)
            if tuple(x.shape[-2:]) != size:
                msg = f"video frames must have one size: frame 0 is {size}, frame {t} is {tuple(x.shape[-2:])}"
                raise InputError(msg)
            return x

        if input_masks is not None:
            if any(v is not None for v in (input_points, input_labels, input_boxes)):
                msg = "track takes input_masks or input_points / input_boxes, not both"
                raise InputError(msg)
            m = _numeric("input_masks", input_masks, torch.float32)
            batched = m.ndim == 3  # noqa: PLR2004 - [P, H, W]
            if m.ndim not in {2, 3} or tuple(m.shape[-2:]) != size:
                msg = f"input_masks must be [P, H, W] (or [H, W]) at the frame size {size}, got {tuple(m.shape)}"
                raise InputError(msg)
            prompts = {"input_masks": (m if batched else m[None]).ne(0).float().to(self.device)}
        else:
            given = {"input_points": input_points, "input_labels": input_labels, "input_boxes": input_boxes}
            prompts, batched = self._prompts({k: v for k, v in given.items() if v is not None})
        if next(iter(prompts.values())).shape[0] == 0:
            msg = "track needs at least one object prompt"
            raise InputError(msg)
        with self._session():
            out = self._runner.track(
                frame,
                n,
                size,
                points=prompts.get("input_points"),
                labels=prompts.get("input_labels"),
                boxes=prompts.get("input_boxes"),
                masks=prompts.get("input_masks"),
            )
        if not batched:
            out = {k: v[:, 0] for k, v in out.items()}
        return VideoSegmentationResult(
            out["masks"], out["pred_masks"], out["object_score_logits"], self.variant.ref, self.backend, self.options
        )

    def _frame(self, video: Sequence[ImageLike] | np.ndarray | torch.Tensor, t: int) -> torch.Tensor:
        """Frame `t` as a checked `[1, 3, H, W]` batch on the device."""
        x = self._to_batch(cast("ImageLike", video[t]))
        if x.shape[0] != 1:
            msg = f"video frame {t} is a batch of {x.shape[0]} images; pass one image per frame"
            raise InputError(msg)
        return x

    def _concept(
        self,
        image: ImageLike | ImageEmbeddings,
        text: str | None,
        given: dict[str, PromptLike],
        labels: PromptLike | None,
        threshold: float | None,
    ) -> ConceptSegmentationResult:
        if other := sorted(set(given) - {"input_boxes"}):
            msg = f"{self.variant.ref} takes text and / or exemplar input_boxes, not {other}"
            raise InputError(msg)
        boxes = given.get("input_boxes")
        if text is None and boxes is None:
            msg = "no prompt given: pass text and / or exemplar input_boxes"
            raise InputError(msg)
        if text is not None and (not isinstance(text, str) or not text.strip()):
            msg = f"text must be a non-empty string, got {text!r}"
            raise InputError(msg)
        if boxes is None and labels is not None:
            msg = "input_boxes_labels needs input_boxes"
            raise InputError(msg)
        threshold = CONCEPT_THRESHOLD if threshold is None else _threshold(threshold)
        options = core.ReadOnlyDict(self.options, threshold=threshold)  # the threshold used, also the default
        ex_boxes, ex_labels = self._check_exemplars(boxes, labels, text)  # before the (expensive) image encoder
        x = None
        if isinstance(image, ImageEmbeddings):
            size = image.image_size
        else:  # the image size is known before the encoder runs
            if is_batch(image):
                msg = "prompt calls take one image; use get_image_embeddings() for a batch and call once per emb[i]"
                raise InputError(msg)
            x = self._to_batch(image)
            size = (int(x.shape[-2]), int(x.shape[-1]))
        ex_boxes = _normalised(ex_boxes, size)
        emb = self._embedding(image if x is None else self._embed_tensor(x))
        (h, w), size = emb.image_size, self._runner.mask_size
        with self._session():
            out = self._runner.segment_concept(dict(emb.features), text, ex_boxes, ex_labels)
            q = out["scores"].shape[0]  # the runner's raw maps (not the card outputs, which are the result fields)
            shapes = {"scores": (q,), "boxes": (q, 4), "pred_masks": (q, *size)}
            if (got := {k: tuple(v.shape) for k, v in out.items()}) != shapes:
                msg = f"{self.variant.ref}: runner returned {got}, expected {shapes}"
                raise RuntimeError(msg)
            out = {k: v.float() for k, v in out.items()}
            scores = out["scores"]
            idx = torch.nonzero(scores > threshold)[:, 0]
            idx = idx[torch.argsort(scores[idx], descending=True)]
            boxes = out["boxes"][idx] * torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)
            if len(idx):
                masks = self._runner.upsample(out["pred_masks"][idx, None], (h, w))[:, 0] > 0
            else:  # nothing found: no zero-batch interpolation
                masks = torch.zeros((0, h, w), dtype=torch.bool, device=self.device)
        return ConceptSegmentationResult(boxes, scores[idx], None, masks, (), self.variant.ref, self.backend, options)

    def _check_exemplars(
        self, boxes: PromptLike | None, labels: PromptLike | None, text: str | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Checked exemplar boxes `[N, 4]` in pixels and labels `[N]` (long) on the CPU; `None` for no boxes.

        Raises:
            InputError: Not a `boxes` model, bad shape / labels, more than `MAX_EXEMPLARS` boxes, a box with
                `x1 <= x0` or `y1 <= y0`, or an empty box set without text.
        """
        if boxes is None:
            return None, None
        if "boxes" not in self.prompts:
            msg = f"{self.variant.ref} takes no boxes prompts (takes: {sorted(self.prompts)})"
            raise InputError(msg)
        b = _numeric("input_boxes", boxes, torch.float32).cpu()
        if b.numel() == 0:  # an empty exemplar set (e.g. a detector found nothing)
            b = b.reshape(0, 4)
        elif b.ndim == 1:
            b = b[None]
        if b.ndim != 2 or b.shape[-1] != 4:  # noqa: PLR2004 - (x0, y0, x1, y1)
            msg = f"input_boxes (exemplars of one image) must be [N, 4] or [4], got shape {tuple(b.shape)}"
            raise InputError(msg)
        n = b.shape[0]
        if n > MAX_EXEMPLARS:
            msg = f"at most {MAX_EXEMPLARS} exemplar boxes per call, got {n}"
            raise InputError(msg)
        lab = None if labels is None else _exemplar_labels(labels, n)
        if bad := torch.nonzero((b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1]))[:, 0].tolist():
            msg = f"input_boxes need x1 > x0 and y1 > y0; box {bad[0]} is {b[bad[0]].tolist()}"
            raise InputError(msg)
        if n == 0:
            if text is None:
                msg = "input_boxes is empty and no text given: pass at least one exemplar box or text"
                raise InputError(msg)
            return None, None
        return b, lab

    def _embedding(self, image: ImageLike | ImageEmbeddings) -> ImageEmbeddings:
        """The single-image embedding of a prompt call (computed if `image` is an image)."""
        emb = image if isinstance(image, ImageEmbeddings) else self._embed_one(image)
        if emb.model != self.variant.ref or emb.backend != self.backend or dict(emb.options) != dict(self.options):
            msg = (
                f"embeddings of {emb.model} ({emb.backend.mode}, options {dict(emb.options)}, checkpoint "
                f"{list(emb.backend.checkpoint) or 'recorded'}) used with {self.variant.ref} ({self.backend.mode}, "
                f"options {dict(self.options)}, checkpoint {list(self.backend.checkpoint) or 'recorded'}); "
                "embed with this model"
            )
            raise InputError(msg)
        if len(emb) != 1:
            msg = f"prompt calls take one image; these embeddings hold {len(emb)}: pass emb[i]"
            raise InputError(msg)
        return emb

    def _embed_one(self, image: ImageLike) -> ImageEmbeddings:
        if is_batch(image):
            msg = "prompt calls take one image; use get_image_embeddings() for a batch and call once per emb[i]"
            raise InputError(msg)
        return self.get_image_embeddings(image)

    def _no_masks(self, emb: ImageEmbeddings, multimask: bool) -> PromptableSegmentationResult:  # noqa: FBT001
        """The result of an empty prompt batch (`P = 0`): `[0, M, H, W]` masks, `[0, M]` scores."""
        m, (h, w), (mh, mw) = _MULTIMASK if multimask else 1, emb.image_size, self._runner.mask_size
        return PromptableSegmentationResult(
            torch.zeros((0, m, h, w), dtype=torch.bool, device=self.device),
            torch.zeros((0, m), device=self.device),
            torch.zeros((0, m, mh, mw), device=self.device),
            self.variant.ref,
            self.backend,
            self.options,
        )

    def _prompts(self, given: dict[str, PromptLike]) -> tuple[dict[str, torch.Tensor], bool]:
        """Prompt arrays as tensors on the device with a leading `P`, and whether the caller passed `P`.

        An empty prompt batch (`P = 0`, e.g. `input_boxes` `[0, 4]`) is kept (the call returns an empty result);
        points with no point per prompt (`[P, 0, 2]`) are dropped, and raise if they were the only prompt.
        """
        if not given:
            msg = "no prompt given: pass input_points + input_labels, input_boxes and / or input_masks (or text)"
            raise InputError(msg)
        if ("input_points" in given) != ("input_labels" in given):
            msg = "input_points and input_labels go together"
            raise InputError(msg)
        out, batched, caller = {}, set(), {}
        for name, value in given.items():
            kind, dims, dtype = _PROMPTS[name]
            if kind not in self.prompts:
                msg = f"{self.variant.ref} takes no {kind} prompts (takes: {sorted(self.prompts)})"
                raise InputError(msg)
            t = _numeric(name, value, dtype)
            if name == "input_boxes":
                t = _boxes_tensor(t)
            caller[name] = tuple(t.shape)
            if name == "input_labels":
                _check_point_labels(value)
            if t.ndim not in {dims, dims - 1}:
                msg = f"{name} must have {dims} (or {dims - 1}, without P) dims, got shape {tuple(t.shape)}"
                raise InputError(msg)
            batched.add(t.ndim == dims)
            out[name] = (t if t.ndim == dims else t[None]).to(self.device)
        self._check_prompt_shapes(out, caller)
        points = out.get("input_points")
        if points is not None and points.shape[0] and not points.shape[1]:  # P prompts with no point each
            del out["input_points"], out["input_labels"]
            if not out:
                msg = f"no prompt given: input_points has no points (shape {caller['input_points']})"
                raise InputError(msg)
        return out, True in batched

    def _check_prompt_shapes(self, p: dict[str, torch.Tensor], caller: dict[str, tuple[int, ...]]) -> None:
        """Check the batched prompts `p`; errors name the shapes the caller passed (`caller`)."""
        shapes = {k: tuple(v.shape) for k, v in p.items()}
        if len({s[0] for s in shapes.values()}) != 1:
            msg = f"prompts disagree on the prompt batch P: {caller}"
            raise InputError(msg)
        want = {"input_points": (2,), "input_boxes": (4,), "input_masks": (1, *self._runner.mask_size)}
        for k, tail in want.items():
            if k in shapes and shapes[k][-len(tail) :] != tail:
                msg = f"{k} must end in {tail}, got shape {caller[k]}"
                raise InputError(msg)
        if "input_points" in shapes and shapes["input_labels"] != shapes["input_points"][:2]:
            msg = f"input_labels {caller['input_labels']} must be input_points {caller['input_points']} without the 2"
            raise InputError(msg)


class ConceptSegmentationModel(PromptableSegmentationModel):
    """A loaded concept model (SAM 3, `load` of a `ConceptModelId`): text and / or exemplar boxes -> every instance.

    The same model as `PromptableSegmentationModel` with calls typed as `ConceptSegmentationResult`.
    """

    @overload
    def __call__(
        self,
        image: ImageLike | ImageEmbeddings,
        *,
        text: str,
        input_boxes: PromptLike | None = None,
        input_boxes_labels: PromptLike | None = None,
        threshold: float | None = None,
    ) -> ConceptSegmentationResult: ...

    @overload
    def __call__(
        self,
        image: ImageLike | ImageEmbeddings,
        *,
        input_boxes: PromptLike,
        input_boxes_labels: PromptLike | None = None,
        threshold: float | None = None,
    ) -> ConceptSegmentationResult: ...

    def __call__(  # noqa: PLR0913  # ty: ignore[invalid-method-override] - typed for concept prompts only
        self,
        image: ImageLike | ImageEmbeddings,
        *,
        input_points: PromptLike | None = None,
        input_labels: PromptLike | None = None,
        input_boxes: PromptLike | None = None,
        input_masks: PromptLike | None = None,
        multimask_output: bool = True,  # noqa: ARG002 - accepted like the base class; no effect on concept prompts
        text: str | None = None,
        input_boxes_labels: PromptLike | None = None,
        threshold: float | None = None,
    ) -> ConceptSegmentationResult:
        """Segment every instance of a concept: `text` and / or exemplar `input_boxes` (+ `input_boxes_labels`).

        See `PromptableSegmentationModel.__call__` for the arguments; point / mask prompts raise `InputError`.

        Raises:
            InputError: Unsupported image, missing / inconsistent / unsupported prompts, or embeddings of another
                model.
        """
        given = {"input_points": input_points, "input_labels": input_labels, "input_boxes": input_boxes}
        given = {k: v for k, v in (given | {"input_masks": input_masks}).items() if v is not None}
        return self._concept(image, text, given, input_boxes_labels, threshold)


class InContextSegmentationModel(PromptableSegmentationModel):
    """A loaded in-context model (INSID3, `load` of an `InContextModelId`): reference images + masks -> target mask.

    The model segments in a target image whatever the reference masks mark in the reference images (one or more
    shots); no point / box / text prompts.
    """

    def __call__(  # ty: ignore[invalid-method-override] - in-context prompts only
        self,
        image: ImageLike | ImageEmbeddings,
        *,
        reference_images: ImageLike | ImageEmbeddings | Sequence[ImageLike],
        reference_masks: PromptLike | Sequence[PromptLike],
    ) -> InContextSegmentationResult:
        """Segment the target `image` like the references.

        Args:
            image: The target image, or its `get_image_embeddings`.
            reference_images: One reference image, a list of them, an image batch, or their `get_image_embeddings`.
            reference_masks: One binary mask per reference (nonzero = the concept), each at its reference's size:
                `[H, W]` for one reference, else `[S, H, W]` (references of one size) or a list.

        Returns:
            The target mask.

        Raises:
            InputError: Unsupported image, no references, a mask count or size that does not fit the references, or
                embeddings of another model.
        """
        emb = self._embedding(image)
        refs, sizes = self._references(reference_images)
        masks = _reference_masks(reference_masks, sizes, self.device)
        with self._session():
            out = self._runner.segment_reference(emb.features["features"], refs, masks)
            h, w = out["similarity"].shape
            self._check_outputs(out, {"masks": (h, w), "similarity": (h, w)})
            mask = self._runner.upsample(out["masks"], emb.image_size)
        return InContextSegmentationResult(
            mask, out["similarity"].float(), self.variant.ref, self.backend, self.options
        )

    def _references(
        self, images: ImageLike | ImageEmbeddings | Sequence[ImageLike]
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Reference features `[S, ...]` and the size of each reference image."""
        if isinstance(images, ImageEmbeddings):
            embs = [self._checked(images)]
        elif isinstance(images, list | tuple):
            if not images:
                msg = "reference_images is empty: pass at least one reference image"
                raise InputError(msg)
            embs = [
                self._checked(i) if isinstance(i, ImageEmbeddings) else self.get_image_embeddings(i) for i in images
            ]
        else:
            embs = [self.get_image_embeddings(cast("ImageLike", images))]
        sizes = [e.image_size for e in embs for _ in range(len(e))]
        return torch.cat([e.features["features"] for e in embs]), sizes

    def _checked(self, emb: ImageEmbeddings) -> ImageEmbeddings:
        """Embeddings of this model (any count)."""
        if emb.model != self.variant.ref or emb.backend != self.backend or dict(emb.options) != dict(self.options):
            msg = f"reference embeddings of {emb.model} ({emb.backend.mode}) used with {self.variant.ref}; embed here"
            raise InputError(msg)
        return emb


def _reference_masks(
    masks: PromptLike | Sequence[PromptLike], sizes: list[tuple[int, int]], device: torch.device
) -> list[torch.Tensor]:
    """One `[H, W]` bool mask per reference on `device`.

    Raises:
        InputError: Not one mask per reference, or a mask not at its reference's size.
    """
    if isinstance(masks, list | tuple) and len(masks) == len(sizes) and not all(np.isscalar(m) for m in masks):
        items = [_numeric("reference_masks", m, torch.float32) for m in masks]
    else:
        t = _numeric("reference_masks", cast("PromptLike", masks), torch.float32)
        items = [t] if t.ndim == 2 else list(t)  # noqa: PLR2004 - [H, W]
    if len(items) != len(sizes):
        msg = f"reference_masks: {len(items)} mask(s) for {len(sizes)} reference image(s)"
        raise InputError(msg)
    for i, (m, size) in enumerate(zip(items, sizes, strict=True)):
        if tuple(m.shape) != size:
            msg = f"reference_masks[{i}] must be [H, W] at its reference's size {size}, got {tuple(m.shape)}"
            raise InputError(msg)
    return [m.to(device).ne(0) for m in items]


def _model_class(model: str) -> type[PromptableSegmentationModel]:
    """`ConceptSegmentationModel` / `InContextSegmentationModel` by the runner's prompts, else the point / box model."""
    try:
        card = core.model_card(TASK, model)
    except ModelNotFoundError:
        return PromptableSegmentationModel  # `load_model` raises the error
    prompts = cast("type[Runner]", core.runner_class(card, card.modes[0])).prompts
    if "reference" in prompts:
        return InContextSegmentationModel
    return ConceptSegmentationModel if "text" in prompts else PromptableSegmentationModel


def _exemplar_labels(labels: PromptLike, n: int) -> torch.Tensor:
    """Exemplar labels `[n]` (long, CPU) of `n` boxes.

    Raises:
        InputError: Not `n` labels, or a label other than 1 / 0.
    """
    lab = _numeric("input_boxes_labels", labels, torch.float32).cpu().reshape(-1)
    if lab.shape[0] != n:
        msg = f"input_boxes_labels has {lab.shape[0]} labels for {n} boxes"
        raise InputError(msg)
    if not bool(((lab == 0) | (lab == 1)).all()):
        msg = "input_boxes_labels must be 1 (positive) or 0 (negative); pass only real boxes (no padding)"
        raise InputError(msg)
    return lab.long()


def _check_point_labels(value: PromptLike) -> None:
    """Point labels must be 1 (foreground), 0 (background) or -1 (padding): `.to(int64)` would truncate 0.7 and pass 2.

    Raises:
        InputError: Any other value.
    """
    lab = _numeric("input_labels", value, torch.float32)
    if not bool(((lab == 1) | (lab == 0) | (lab == -1)).all()):
        msg = "input_labels must be 1 (foreground), 0 (background) or -1 (padding)"
        raise InputError(msg)


def _normalised(boxes: torch.Tensor | None, size: tuple[int, int]) -> torch.Tensor | None:
    """Exemplar boxes in pixels -> normalised by the image size (as `Sam3Processor`: x / W, y / H).

    Raises:
        InputError: A box reaches further than one image size outside the image.
    """
    if boxes is None:
        return None
    h, w = size
    x, y = boxes[:, 0::2], boxes[:, 1::2]
    if bool((x < -w).any() or (x > 2 * w).any() or (y < -h).any() or (y > 2 * h).any()):
        msg = f"exemplar boxes must lie within one image size of the {w} x {h} image (x in [-W, 2W], y in [-H, 2H])"
        raise InputError(msg)
    return boxes / torch.tensor([w, h, w, h], dtype=torch.float32)


def _numeric(name: str, value: PromptLike, dtype: torch.dtype) -> torch.Tensor:
    """A prompt array as a finite real tensor of `dtype` (on its own device: numpy / lists on the CPU).

    Raises:
        InputError: Not numeric, complex, or not finite.
    """
    try:
        t = torch.as_tensor(np.asarray(value) if isinstance(value, list | tuple) else value)
    except (TypeError, ValueError, RuntimeError) as e:
        msg = f"{name}: not a numeric array ({e})"
        raise InputError(msg) from e
    if t.is_complex():
        msg = f"{name} must be real, got {t.dtype}"
        raise InputError(msg)
    t = t.to(dtype=dtype)
    if not bool(torch.isfinite(t).all()):
        msg = f"{name} has non-finite values"
        raise InputError(msg)
    return t


def bench_inputs(model: PromptableSegmentationModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[3, H, W]` and a foreground point at its centre.

    Text models get the text prompt "object" instead of the point; in-context models the image itself as the reference
    with its centre quarter as the mask. Times the full call (image encoder + one prompt,
    three candidate masks); `get_image_embeddings` alone is the encoder.
    """
    h, w = size.height, size.width
    image = synthetic_image(h, w, model.device)[0]
    if "text" in model.prompts:
        return (image,), {"text": "object"}
    if "reference" in model.prompts:
        mask = torch.zeros(h, w, dtype=torch.bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return (image,), {"reference_images": image, "reference_masks": mask}
    return (image,), {"input_points": [[(w - 1) / 2, (h - 1) / 2]], "input_labels": [1]}


list_models, model_card, _ = task_api(TASK, _model_class)


@overload
def load(
    model: ConceptModelId,
    *,
    mode: ModeArg = "auto",
    device: str | torch.device | None = None,
    dtype: Dtype | torch.dtype | None = None,
    options: Mapping[str, Any] | None = None,
    weights: str | os.PathLike[str] | None = None,
) -> ConceptSegmentationModel: ...


@overload
def load(
    model: InContextModelId,
    *,
    mode: ModeArg = "auto",
    device: str | torch.device | None = None,
    dtype: Dtype | torch.dtype | None = None,
    options: Mapping[str, Any] | None = None,
    weights: str | os.PathLike[str] | None = None,
) -> InContextSegmentationModel: ...


@overload
def load(
    model: str,
    *,
    mode: ModeArg = "auto",
    device: str | torch.device | None = None,
    dtype: Dtype | torch.dtype | None = None,
    options: Mapping[str, Any] | None = None,
    weights: str | os.PathLike[str] | None = None,
) -> PromptableSegmentationModel: ...


@typo_checked_load
def load(  # noqa: PLR0913 - the same arguments in every task
    model: str,
    *,
    mode: ModeArg = "auto",
    device: str | torch.device | None = None,
    dtype: Dtype | torch.dtype | None = None,
    options: Mapping[str, Any] | None = None,
    weights: str | os.PathLike[str] | None = None,
) -> PromptableSegmentationModel:
    """Load a promptable segmentation model; the arguments and errors are those of `fastforks.stereo.load`.

    Args:
        model: Model id, optionally pinned to a revision.
        mode: `"auto"`, `"portable"` or `"kernels"`.
        device: Default: `"cuda"` if available, else `"mps"` for a family validated on it, else `"cpu"`.
        dtype: `None` (the validated one), `"float32"` (portable only), `"float16"` or `"bfloat16"`; see
            `fastforks.stereo.load`.
        options: Overrides of the model's load options (card `[options]`).
        weights: Your own weights file (or directory) instead of the recorded ones, e.g. a fine-tune; not
            hash-checked, loaded strictly (see `fastforks.stereo.load`).

    Raises:
        ModelNotFoundError: Unknown id or revision.
        OptionsError: Invalid option, mode, dtype or device.
        BackendUnavailableError: `mode="kernels"` cannot run here. Never raised for `"auto"`.
        WeightsError: Weights cannot be found, downloaded or verified, or `weights` does not fit.
    """
    return core.load_model(
        TASK,
        model,
        _model_class(model),
        mode=mode,
        device=device,
        dtype=dtype,
        options=options,
        weights=weights,
    )
