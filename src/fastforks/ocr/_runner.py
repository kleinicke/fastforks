"""The interface an OCR family implements: one `Runner` per backend, in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    Params = Params

    def __call__(self, image):  # [1, 3, H, W] uint8 RGB (`uint8_input`)
        ...
        return {"boxes": quads, "texts": texts, "scores": scores}  # [N, 4, 2], N strings, [N]
```

The shared steps of detector + recogniser pipelines are here, as PaddleX's OCR pipeline runs them (bit-exact where
OpenCV / NumPy decide integers): the detector resize (`det_size`, `det_input`), the 8-connected components of the
binary map (`components`, on the device), the DB boxes (`fastforks.ocr._db`), the reading order and the crops
(`fastforks.ocr._crop`), the recogniser batches (`rec_batches`) and the CTC decoding (`ctc_decode`).
The core does image conversion and checks, the per-image loop, output checks, locking and the device guard.
"""

from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core import runner as core
from fastforks._core.runner import Context, Forward, NoParams
from fastforks.detection._cv2_resize import cv2_linear
from fastforks.ocr import _crop, _db

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

__all__ = [
    "Context",
    "Forward",
    "NoParams",
    "Runner",
    "components",
    "ctc_decode",
    "det_input",
    "det_size",
    "levels",
    "line_crops",
    "rec_batches",
    "text_boxes",
]

STRIDE = 32  # fixed: the detectors' input sides are multiples of 32
DET_MIN_SIDE = 64  # fixed: PaddleX `limit_type="min"`, `limit_side_len=64` (PP-OCRv5 defaults)
DET_MAX_SIDE = 4000  # fixed: PaddleX `max_side_limit`
DET_THRESH = 0.3  # fixed: PaddleX `thresh` of the binary map
MAX_CANDIDATES = 1000  # fixed: PaddleX DBPostProcess
DET_MEAN = (0.485, 0.456, 0.406)  # applied to the B, G, R channels in this order, as upstream
DET_STD = (0.229, 0.224, 0.225)
REC_HEIGHT = 48  # fixed: PP-OCRv5 recogniser input height
REC_WIDTH = 320  # fixed: minimum recogniser input width (`rec_image_shape`)
REC_MAX_WIDTH = 3200  # fixed: PaddleX `OCRReisizeNormImg.max_imgW`


def det_size(height: int, width: int) -> tuple[int, int]:
    """PaddleX `DetResizeForTest.resize_image_type0` (limit type "min" 64, max side 4000): the detector input size."""
    ratio = DET_MIN_SIDE / (min(width, height)) if min(height, width) < DET_MIN_SIDE else 1.0
    h, w = int(height * ratio), int(width * ratio)
    if max(h, w) > DET_MAX_SIDE:
        ratio = DET_MAX_SIDE / max(h, w)
        h, w = int(h * ratio), int(w * ratio)
    return max(round(h / STRIDE) * STRIDE, STRIDE), max(round(w / STRIDE) * STRIDE, STRIDE)


def det_input(bgr: torch.Tensor) -> torch.Tensor:
    """`[1, 3, H, W]` uint8 BGR -> the detector input `[1, 3, h, w]` float32 (OpenCV linear resize, normalised)."""
    size = det_size(*bgr.shape[-2:])
    x = cv2_linear(bgr, *size).float() if size != tuple(bgr.shape[-2:]) else bgr.float()
    for c in range(3):  # NormalizeImage: x * (1/255/std) + (-mean/std), both float32 roundings
        x[:, c].mul_(1.0 / 255.0 / DET_STD[c]).add_(-DET_MEAN[c] / DET_STD[c])
    return x


def components(mask: torch.Tensor) -> list[_db.Rows]:
    """8-connected components of a `[h, w]` bool map, in OpenCV's label order (raster order of the first pixel).

    Each component as its rows: `(y ascending, leftmost x, rightmost x)` (`fastforks.ocr._db.Rows`). The labelling
    runs on the device: every pixel takes the smallest pixel index among its 3 x 3 neighbours, then the smallest
    of its horizontal and of its vertical run, then follows its label's label (pointer jumping), until nothing
    changes. The fixed point is each component's smallest pixel index whatever the passes; the run passes make
    a long text line converge in a few rounds instead of one per pixel of its length.
    """
    if mask.device.type == "mps":  # int64 scatter_reduce atomics do not compile on Apple GPUs
        mask = mask.cpu()
    h, w = mask.shape
    big = h * w
    idx = torch.arange(big, device=mask.device).view(h, w)
    label = torch.where(mask, idx, big)
    ys, xs = mask.nonzero(as_tuple=True)
    flat_pos = ys * w + xs
    left = torch.nn.functional.pad(mask[:, :-1], (1, 0), value=False)
    up = torch.nn.functional.pad(mask[:-1], (0, 0, 1, 0), value=False)
    h_run = (mask & ~left).view(-1).cumsum(0)[flat_pos] - 1  # run index of each set pixel, raster order
    v_run = (mask & ~up).t().reshape(-1).cumsum(0)[xs * h + ys] - 1  # column-major order
    runs = int(max(h_run.max().item(), v_run.max().item())) + 1 if len(ys) else 0
    while True:
        pad = torch.nn.functional.pad(label, (1, 1, 1, 1), value=big)
        low = label
        for dy in range(3):
            for dx in range(3):
                low = torch.minimum(low, pad[dy : dy + h, dx : dx + w])
        lab = low[ys, xs]
        for run in (h_run, v_run):
            lab = lab.new_full((runs,), big).scatter_reduce_(0, run, lab, "amin")[run]
        flat = label.new_full((big + 1,), big)
        flat[flat_pos] = lab
        for _ in range(4):
            flat = flat[flat]
        new = flat[:-1].view(h, w)
        if torch.equal(new, label):
            break
        label = new
    keys, inverse = torch.unique(label[ys, xs] * h + ys, return_inverse=True)
    x0 = torch.full_like(keys, big).scatter_reduce_(0, inverse, xs, "amin")
    x1 = torch.full_like(keys, -1).scatter_reduce_(0, inverse, xs, "amax")
    keys, x0, x1 = (t.cpu().numpy() for t in (keys, x0, x1))
    comp, row = keys // h, keys % h
    starts = np.flatnonzero(np.diff(comp, prepend=-1))
    ends = [*starts[1:], len(comp)][: len(starts)]  # none for an empty map
    return [(row[a:b], x0[a:b], x1[a:b]) for a, b in zip(starts, ends, strict=True)]


def text_boxes(pred: torch.Tensor, size: tuple[int, int], *, box_threshold: float, unclip_ratio: float) -> np.ndarray:
    """PaddleX `DBPostProcess` + `SortQuadBoxes`: int32 quads `[N, 4, 2]` in `size` (h, w) pixels, reading order."""
    comps = components(pred > DET_THRESH)
    quads, _ = _db.boxes_from_rows(
        pred.float().cpu().numpy(),
        comps,
        size,
        box_thresh=box_threshold,
        unclip_ratio=unclip_ratio,
        max_candidates=MAX_CANDIDATES,
    )
    return quads[_crop.sort_quads(quads)].astype(np.int32)


def line_crops(bgr: torch.Tensor, quads: np.ndarray) -> list[torch.Tensor]:
    """PaddleX `CropByPolys` (quad): `[3, h, w]` uint8 crops of `[1, 3, H, W]` uint8 per quad, tall ones rotated."""
    src, sizes = _crop.crop_sources(quads)
    return [_crop.rotate_tall(c) for c in _crop.crop_quads(bgr[0], src, sizes)]


def _rec_width(h: int, w: int) -> tuple[int, int]:
    """PaddleX `OCRReisizeNormImg.resize`: (resized width, padded width) of an `h x w` crop."""
    width = int(REC_HEIGHT * max(REC_WIDTH / REC_HEIGHT, w * 1.0 / h))
    if width > REC_MAX_WIDTH:
        return REC_MAX_WIDTH, REC_MAX_WIDTH
    return (min(math.ceil(REC_HEIGHT * (w / float(h))), width)), width


def rec_batches(crops: Sequence[torch.Tensor], batch: int) -> Iterator[tuple[list[int], torch.Tensor]]:
    """PaddleX's recogniser batches: crops sorted by aspect ratio, `batch` at a time, each zero-padded to the widest.

    Yields:
        The crop indices and their input `[b, 3, 48, W]` float32 in [-1, 1] (each crop resized to height 48 with
        OpenCV's linear resize, `x / 255`, `- 0.5`, `/ 0.5`; zeros on the right).
    """
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[2] / float(crops[i].shape[1]))
    for start in range(0, len(order), batch):
        ids = order[start : start + batch]
        sizes = [_rec_width(*crops[i].shape[1:]) for i in ids]
        x = crops[ids[0]].new_zeros((len(ids), 3, REC_HEIGHT, max(p for _, p in sizes)), dtype=torch.float32)
        for k, (i, (rw, _)) in enumerate(zip(ids, sizes, strict=True)):
            line = cv2_linear(crops[i][None], REC_HEIGHT, rw)[0].float()
            x[k, :, :, :rw] = line.div_(255).sub_(0.5).div_(0.5)
        yield ids, x


def ctc_decode(probs: torch.Tensor, chars: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """PaddleX `CTCLabelDecode`: greedy decoding of `[B, T, classes]` probabilities (class 0 = blank).

    Repeated classes merge, blanks drop; the score is the float32 mean probability of the kept steps (0 if none).
    """
    p, k = probs.float().max(-1)
    p, k = p.cpu().numpy(), k.cpu().numpy()
    texts, scores = [], np.zeros(len(k), np.float32)
    for b in range(len(k)):
        keep = np.ones(k.shape[1], bool)
        keep[1:] = k[b, 1:] != k[b, :-1]
        keep &= k[b] != 0
        texts.append("".join(chars[c] for c in k[b, keep]))
        if keep.any():
            scores[b] = np.mean(p[b, keep])
    return texts, scores


class Runner(core.Runner):
    """One backend of an OCR family."""

    uint8_input = True

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor) -> Mapping[str, torch.Tensor | list[str]]:
        """Read one image.

        Args:
            image: `[1, 3, H, W]` RGB on `ctx.device`: uint8 levels for uint8 inputs, else float32 in `[0, 1]`.

        Returns:
            `{"boxes": [N, 4, 2], "texts": N strings, "scores": [N]}` in reading order (boxes in input pixels).
        """


def levels(image: torch.Tensor) -> torch.Tensor:
    """`[B, 3, H, W]` RGB (uint8 or float in [0, 1]) -> uint8 BGR, as OpenCV reads an image file."""
    x = image if image.dtype == torch.uint8 else image.mul(255).round_().clamp_(0, 255).to(torch.uint8)
    return x.flip(1)
