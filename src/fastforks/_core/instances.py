"""`Instances`: the detections of one image (boxes, scores, labels, masks), shared by the instance-level tasks.

`fastforks.detection` returns one per image; `fastforks.promptable_segmentation` returns one for concept prompts
(text / exemplar boxes). The attribute names are the keys of the Hugging Face transformers post-processing dicts
(`post_process_object_detection`: `scores` / `labels` / `boxes`; SAM 3 `post_process_instance_segmentation`: + `masks`).
"""

from __future__ import annotations

import dataclasses
import numbers
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import torch

from fastforks._core.outputs import NumpyOutputs

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from fastforks._core.model import Backend


@dataclass(frozen=True)
class Instances(NumpyOutputs):
    """Instances found in one image, sorted by descending score, on the model's device.

    Attributes:
        boxes: `[N, 4]` float32 boxes `(x0, y0, x1, y1)` in pixels of the input image.
        scores: `[N]` float32 confidence.
        labels: `[N]` int64 class indices into `class_names`, or `None` (class-agnostic, e.g. a text prompt).
        masks: `[N, H, W]` bool instance masks at the input size, or `None` (box-only models).
        class_names: Name of every class (the card's label set); empty for class-agnostic results.
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor | None
    masks: torch.Tensor | None
    class_names: tuple[str, ...]
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of instances."""
        return int(self.scores.shape[0])

    def __getitem__(self, index: int | slice | torch.Tensor) -> Self:
        """The selected instances (same type), e.g. `result[result.scores > 0.5]` or `result[:10]`.

        An int selects one instance (as a result of length 1); out-of-range ints raise `IndexError`, negative ones count
        from the end.
        """
        if isinstance(index, torch.Tensor):
            if index.ndim == 0 and not index.dtype.is_floating_point and index.dtype != torch.bool:
                index = int(index.item())  # a 0-d integer tensor selects like an int (keeps the instance dimension)
        elif isinstance(index, numbers.Integral):  # numpy integer scalars (bool is an Integral too: like Python, 0 / 1)
            index = int(index)
        if isinstance(index, int):
            n = len(self)
            if not -n <= index < n:
                msg = f"instance index {index} out of range for {n} instances"
                raise IndexError(msg)
            index %= n
            index = slice(index, index + 1)

        def pick(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t[index]

        return dataclasses.replace(
            self, boxes=self.boxes[index], scores=self.scores[index], labels=pick(self.labels), masks=pick(self.masks)
        )

    def __iter__(self) -> Iterator[Self]:
        """One result of length 1 per instance, in score order (`for det in result: ...`)."""
        for i in range(len(self)):
            yield self[i]

    def cpu(self) -> Self:
        """The instances on the host (same type); device tensors go through pinned staging buffers.

        Faster than `.cpu()` on each field: pageable copies of full-image masks cost about 1 ms each. The returned
        tensors live in page-locked memory (released when they are freed).
        """

        def host(t: torch.Tensor) -> torch.Tensor:
            if t.device.type != "cuda":
                return t.cpu()
            return torch.empty(t.shape, dtype=t.dtype, pin_memory=True).copy_(t)

        return self._moved(host)

    def to_list(self) -> list[dict[str, Any]]:
        """One dict per instance: `box` (list), `score`, `label` + `class_name`, `mask` ([H, W] tensor)."""
        boxes, scores = self.boxes.tolist(), self.scores.tolist()
        labels = self.labels.tolist() if self.labels is not None else [None] * len(scores)
        out: list[dict[str, Any]] = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels, strict=True)):
            d: dict[str, Any] = {"box": box, "score": score}
            if label is not None:
                d |= {"label": label, "class_name": self.class_names[label] if self.class_names else None}
            if self.masks is not None:
                d["mask"] = self.masks[i]
            out.append(d)
        return out
