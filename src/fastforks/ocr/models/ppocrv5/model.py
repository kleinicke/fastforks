"""PP-OCRv5 runners: Paddle inference weights -> networks and the portable pipeline.

Inference as PaddleX's `OCR` pipeline with the document preprocessor and the text-line orientation classifier off
(PaddleOCR 3 `PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
use_textline_orientation=False)`): DB detector at the input size rounded to multiples of 32 (short side at least
64), DB boxes, reading order, perspective crops of the text lines, CTC recogniser in batches of 6 crops sorted by
aspect ratio, greedy CTC decoding (`fastforks.ocr._runner`). Both networks see BGR, as upstream reads images with
OpenCV.

`portable`: the networks in plain PyTorch (`net/`), float32 or float16; the integer steps (resizes, boxes, crops)
are bit-exact to OpenCV / pyclipper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.runner import host
from fastforks.ocr._runner import (
    DET_MEAN,
    DET_STD,
    Forward,
    Runner,
    ctc_decode,
    det_input,
    levels,
    line_crops,
    rec_batches,
    text_boxes,
)

from . import convert
from .net import build

if TYPE_CHECKING:
    from pathlib import Path

    from fastforks.ocr._runner import Context

REC_BATCH = 6  # fixed: PaddleX OCR pipeline `TextRecognition.batch_size`
FP16_NECK_SCALE = 256.0  # fixed: a power of two (exact); the server neck's ~1.5e6 peaks -> ~6e3 in float16


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`.

    Attributes:
        server: Server networks (PP-HGNetV2) instead of the mobile ones (PP-LCNetV3).
        box_threshold: Minimum mean detector score inside a box (PaddleX `text_det_box_thresh`).
        unclip_ratio: Box growth (PaddleX `text_det_unclip_ratio`): offset = area * ratio / perimeter.
    """

    server: bool
    box_threshold: float
    unclip_ratio: float


def _net(kind: str, program: Path, params: Path, *, server: bool, classes: int = 0) -> torch.nn.Module:
    net, layers = build(kind, server, classes)
    arrays = convert.state_dict(convert.read(program, params), layers)
    state = {k: torch.from_numpy(v.copy()) for k, v in arrays.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected or any(not k.endswith("num_batches_tracked") for k in missing):
        msg = f"PP-OCRv5 {kind}: missing {missing}, unexpected {unexpected}"
        raise ValueError(msg)
    return net.eval()


class Portable(Runner):
    """The PP-OCRv5 networks in plain PyTorch, float32 or float16."""

    Params = Params
    dtypes = ("float32", "float16")
    compiled_methods = ("_detect", "_recognise")  # the networks; boxes, crops and decoding stay eager

    def __init__(self, ctx: Context) -> None:
        """Build both networks."""
        super().__init__(ctx)
        det_program, det_params, rec_program, rec_params, rec_config = ctx.weights()
        chars = json.loads(rec_config.read_text(encoding="utf-8"))["PostProcess"]["character_dict"]
        self.chars = ["", *chars, " "]  # blank, the dictionary, the space (`use_space_char`)
        self.dtype = ctx.torch_dtype
        server = ctx.params.server
        det = _net("det", det_program, det_params, server=server)
        if self.dtype == torch.float16:
            det.scale_neck(FP16_NECK_SCALE)  # ty: ignore[call-non-callable] - Detector
        self.det = det.to(ctx.device, self.dtype)
        rec = _net("rec", rec_program, rec_params, server=server, classes=len(self.chars))
        self.rec = rec.to(ctx.device, self.dtype)

    @host
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor | list[str]]:
        """Text lines of a `[1, 3, H, W]` RGB image."""
        p = self.ctx.params
        bgr = levels(image)
        pred = self._detect(det_input(bgr))
        quads = text_boxes(
            pred, (bgr.shape[-2], bgr.shape[-1]), box_threshold=p.box_threshold, unclip_ratio=p.unclip_ratio
        )
        crops = line_crops(bgr, quads)
        texts, scores = [""] * len(crops), np.zeros(len(crops), np.float32)
        for ids, x in rec_batches(crops, REC_BATCH):
            t, s = ctc_decode(self._recognise(x), self.chars)
            for k, i in enumerate(ids):
                texts[i], scores[i] = t[k], s[k]
        dev = image.device
        return {
            "boxes": torch.from_numpy(quads.astype(np.float32)).to(dev),
            "texts": texts,
            "scores": torch.from_numpy(scores).to(dev),
        }

    def _detect(self, x: torch.Tensor) -> torch.Tensor:
        """Detector map `[h, w]` of the normalised input `[1, 3, h, w]`."""
        return self.det(x.to(self.dtype))[0]

    def _recognise(self, x: torch.Tensor) -> torch.Tensor:
        """Class probabilities `[b, T, classes]` of a recogniser batch `[b, 3, 48, W]`."""
        return self.rec(x.to(self.dtype))

    def module(self) -> torch.nn.Module:
        """The detector with its normalisation: `[B, 3, h, w]` RGB in `[0, 1]` (multiples of 32) -> map `[B, h, w]`."""
        mean = torch.tensor(DET_MEAN[::-1], device=self.ctx.device).view(1, 3, 1, 1)
        std = torch.tensor(DET_STD[::-1], device=self.ctx.device).view(1, 3, 1, 1)

        def forward(net: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
            x = ((image - mean) / std).flip(1)  # RGB statistics reversed, then BGR as the detector takes it
            return net(x.to(self.dtype)).float()

        return Forward(self.det, forward)


RUNNERS = {"portable": Portable}
