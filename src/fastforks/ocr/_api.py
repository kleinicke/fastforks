"""OCR API: `list_models`, `model_card`, `load`, and the model / result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.image import ImageLike, is_batch, to_rgb_batch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks._core.bench import BenchInputs, BenchSize
    from fastforks.ocr._runner import Runner

__all__ = ["Backend", "OCRModel", "OCRResult", "list_models", "load", "model_card"]

TASK = "ocr"


@dataclass(frozen=True)
class OCRResult(NumpyOutputs):
    """Text lines of one image in reading order (top to bottom, then left to right), on the model's device.

    Attributes:
        boxes: `[N, 4, 2]` float32 quads: corners `(x, y)` in input pixels, clockwise from the top-left one.
            Recognisers without a detector (PARSeq) return the whole image as one quad.
        texts: The `N` recognised texts.
        scores: `[N]` float32 confidence in `[0, 1]` (the mean probability of the decoded characters; 0 for an empty
            text).
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    boxes: torch.Tensor
    texts: list[str]
    scores: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of text lines."""
        return len(self.texts)


class OCRModel(core.Model):
    """A loaded OCR model. Call it with an image (or a batch).

    `module()` (portable) is the text detector (recognisers: the recogniser) with its normalisation: an RGB image
    `[B, 3, h, w]` in `[0, 1]` already at the network size -> the network output; the resize, box extraction,
    cropping, recognition batches and decoding are data-dependent and not part of it.

    Has the attributes of `fastforks.Model`.
    """

    _runner: Runner

    def __call__(self, image: ImageLike) -> OCRResult | list[OCRResult]:
        """Find and read the text lines of an image.

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).

        Returns:
            One `OCRResult`; a batch (BHWC / BCHW) gives a list with one per image.

        Raises:
            InputError: Unsupported or too small image.
        """
        x = to_rgb_batch(image, self.device, keep_uint8=self._runner.uint8_input)
        self._check_size(x)
        results = [self._read(x[i : i + 1]) for i in range(x.shape[0])]
        return results if is_batch(image) else results[0]

    def _read(self, x: torch.Tensor) -> OCRResult:
        with self._session():
            out = dict(self._runner(x))
        texts = list(out.pop("texts"))
        n = len(texts)
        maps = self._check_outputs(out, {"boxes": (n, 4, 2), "scores": (n,)})  # ty: ignore[invalid-argument-type] - texts popped
        return OCRResult(maps["boxes"], texts, maps["scores"], self.variant.ref, self.backend, self.options)


_WORDS = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "page",
    "2026",
    "total",
    "amount",
    "invoice",
    "no.",
    "4711",
    "date",
    "report",
    "section",
    "figure",
    "table",
    "result",
    "value",
    "speed",
    "memory",
    "model",
    "text",
    "line",
    "read",
    "find",
    "order",
    "price",
    "EUR",
    "12.50",
    "kg",
]


def text_page(height: int, width: int) -> np.ndarray:
    """A synthetic page `[H, W, 3]` uint8: black lines of random words (fixed seed) on white (22 lines at 1080 px).

    The OCR bench input: a texture without text would leave the detector nothing to box and the recogniser idle.
    """
    size = max(8, round(min(0.6 * height, max(height / 36, 16))))
    font = ImageFont.load_default(size=size)
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    rng = np.random.default_rng(0)
    margin, y = max(2, size // 4), max(2, size // 4)
    while y + size <= height:
        x = margin
        while True:
            word = _WORDS[rng.integers(len(_WORDS))]
            right = x + draw.textlength(word, font=font)
            if right > width - margin and x > margin:
                break
            draw.text((x, y), word, fill="black", font=font)
            x = right + size // 2
        y += round(size * 1.6)
    return np.asarray(page).copy()


def bench_inputs(model: OCRModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic text page `[3, H, W]` uint8 (`text_page`)."""
    return (torch.from_numpy(text_page(size.height, size.width)).permute(2, 0, 1).to(model.device),), {}


list_models, model_card, load = task_api(
    TASK, OCRModel, note="Load options (card `[options]`) include the detector post-processing, e.g. `box_threshold`."
)
