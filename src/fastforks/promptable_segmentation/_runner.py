"""The interface a promptable segmentation family implements: one `Runner` per backend, in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    Params = Params
    prompts = frozenset({"points", "boxes", "masks"})

    def embed(self, image):  # [1, 3, H, W] RGB in [0, 1]
        return {"image_embeddings": self.encoder(resize_and_pad(image))}

    def predict(self, features, size, *, points, labels, boxes, masks, multimask):  # prompts in input pixels
        return {"pred_masks": ..., "iou_scores": ...}  # [P, M, h, w] logits, [P, M]

    def upsample(self, pred_masks, size):  # -> [P, M, H, W] logits at the input size
        ...
```

Families differ in how they resize the image (SAM: longest side 1024 + pad; SAM 2: square 1024) and so in how prompt
coordinates and low-resolution masks map back to the input, so the runner owns both directions. The core converts
images and prompts, checks shapes, thresholds the upsampled logits at 0 (upstream `mask_threshold`), locks the model
and guards the device.

Portable runners also implement `module()`: the image encoder with its input handling, `[1, 3, H, W]` in `[0, 1]` ->
the image embedding(s), named by the `Forward`'s `output_names` (`image_embeddings`; SAM 3: `fpn0` .. `fpn2`).
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar

from fastforks._core import runner as core
from fastforks._core.preprocess import scale
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

__all__ = ["Context", "Forward", "NoParams", "Runner", "scale"]


class Runner(core.Runner):
    """One backend of a promptable segmentation family.

    Class attributes (besides those of `fastforks._core.runner.Runner`):
        prompts: Prompt kinds the model takes: `"points"`, `"boxes"`, `"masks"` (low-resolution mask logits), or
            `"text"` (concept models: `segment_concept`; with `"boxes"`: exemplar boxes too).
        mask_size: `(h, w)` of the low-resolution masks (`predict` output and `masks` prompt).
        low_res_clamp: If set, the returned `pred_masks` are clamped to `[-c, c]` after `upsample` has seen the
            unclamped logits (SAM 2: 32).
        video: The runner implements `track` (video object tracking from frame-0 prompts).
    """

    prompts: ClassVar[frozenset[str]] = frozenset({"points", "boxes", "masks"})
    mask_size: ClassVar[tuple[int, int]] = (256, 256)
    low_res_clamp: ClassVar[float | None] = None
    compiled_methods: ClassVar[tuple[str, ...]] = ("embed", "predict")
    video: ClassVar[bool] = False

    @classmethod
    def input_use(cls, params: Any) -> dict[str, Any]:  # noqa: ANN401, ARG003 - the family's `Params`
        """`{"prompts": sorted prompt kinds}`.

        Args:
            params: The variant's `Params`.
        """
        return {"prompts": tuple(sorted(cls.prompts))}

    @abc.abstractmethod
    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Image features for `predict`.

        Args:
            image: `[1, 3, H, W]` float32 RGB in `[0, 1]` on `ctx.device`.

        Returns:
            Named feature tensors (opaque to the core; handed back to `predict` unchanged).
        """

    @abc.abstractmethod
    def predict(  # noqa: PLR0913 - one argument per prompt kind
        self,
        features: dict[str, torch.Tensor],
        size: tuple[int, int],
        *,
        points: torch.Tensor | None,
        labels: torch.Tensor | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
        multimask: bool,
    ) -> dict[str, torch.Tensor]:
        """Masks for `P` prompts on one image.

        Args:
            features: What `embed` returned.
            size: `(H, W)` of the input image.
            points: `[P, N, 2]` float32 `(x, y)` in input pixels, or `None`.
            labels: `[P, N]` int64 point labels (1 foreground, 0 background, -1 padding), with `points`.
            boxes: `[P, 4]` float32 `(x0, y0, x1, y1)` in input pixels, or `None`.
            masks: `[P, 1, h, w]` float32 low-resolution mask logits (a previous `pred_masks`), or `None`.
            multimask: Return the model's multiple-mask output (`M > 1`) instead of the single mask (`M = 1`).

        Returns:
            `{"pred_masks": [P, M, h, w], "iou_scores": [P, M]}` (any float dtype).
        """

    def segment_concept(
        self,
        features: dict[str, torch.Tensor],
        text: str | None,
        boxes: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Every query's prediction for a concept prompt on one image (models with the `"text"` prompt kind).

        Args:
            features: What `embed` returned.
            text: The text prompt; `None`: exemplar boxes only.
            boxes: Exemplar boxes `[N, 4]` `(x0, y0, x1, y1)` normalised by the input size (CPU float32), or `None`
                (concept models with the `"boxes"` prompt kind).
            labels: `[N]` int64, 1 for a positive and 0 for a negative exemplar; `None`: all positive.

        Returns:
            `{"scores": [Q], "boxes": [Q, 4], "pred_masks": [Q, h, w]}`: probabilities, boxes `(x0, y0, x1, y1)`
            normalised by the input size, low-resolution mask logits. The core thresholds, sorts, scales and
            upsamples (`upsample`).

        Raises:
            NotImplementedError: The model takes no text prompts.
        """
        msg = f"{type(self).__name__} takes no text prompts"
        raise NotImplementedError(msg)

    def track(  # noqa: PLR0913 - one argument per prompt kind
        self,
        frames: Callable[[int], torch.Tensor],
        n_frames: int,
        size: tuple[int, int],
        *,
        points: torch.Tensor | None,
        labels: torch.Tensor | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Track `P` objects prompted on frame 0 through a video (models with `video = True`).

        Args:
            frames: Frame `t` as `[1, 3, H, W]` float32 RGB in `[0, 1]` on `ctx.device` (called once per frame, in
                order).
            n_frames: Number of frames `T`.
            size: `(H, W)` of the frames.
            points: As `predict`, on frame 0.
            labels: As `predict`.
            boxes: As `predict`.
            masks: `[P, H, W]` float32 binary masks on frame 0 (instead of points / boxes), or `None`.

        Returns:
            `{"masks": [T, P, H, W] bool, "pred_masks": [T, P, h, w], "object_score_logits": [T, P]}`.

        Raises:
            NotImplementedError: The model does not track video.
        """
        msg = f"{type(self).__name__} does not track video"
        raise NotImplementedError(msg)

    @abc.abstractmethod
    def upsample(self, pred_masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Low-resolution mask logits `[P, M, h, w]` -> logits `[P, M, H, W]` at the input size `size`."""
