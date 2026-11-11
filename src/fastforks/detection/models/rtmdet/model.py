"""RTMDet runners: image -> boxes, scores and classes (mmdetection `inference_detector`).

Follows the test pipeline and `RTMDetHead.predict` of the reference:

- input: the uint8 image is resized to fit `size x size` keeping its aspect ratio (factor `size / longer side`, new
  sides `int(side * factor + 0.5)`; OpenCV's 8-bit `INTER_LINEAR`, reproduced on the device: bit-exact with OpenCV
  4.10 on arm64 for 36 size pairs of noise and smooth images, measured), put top-left on a `size x size` canvas of
  114s and normalised with the mmdet means / stds (0..255). Small images are enlarged
- network: CSPNeXt, PAFPN, head -> per cell of the stride 8 / 16 / 32 maps a logit per class and four distances
- postprocess: sigmoid scores; per level the candidates at or above the score threshold (the best `nms_pre`); boxes
  = the cell's point -+ the distances, clamped to the canvas, divided by the resize factors (width and height each
  their own, as the rounded sizes give them), empty boxes dropped; class-wise NMS (`nms_iou`, torchvision's
  operator on boxes shifted by `label * (largest coordinate + 1)`, as mmcv's `batched_nms`; from 10000 boxes on in
  several calls, `classwise_nms`), best `max_detections`

The reference configs test with score threshold 0.001 and 300 detections (COCO protocol; the person detectors 0.05 and
100); here `score_threshold` (0.05) and `max_detections` (100) are options as in the other detection families. NMS is
greedy, so thresholding before it equals thresholding after it. Boxes are not clamped to the image, as upstream: an
object at the right or bottom border can reach into the padding.

Portable runs the network in float16 on CUDA (channels-last), scores and boxes in float32. Kernels runs the whole
network through `fastforks._optimize` (fused conv + SiLU / residual / concat kernels, fp16, channels-last) and replays
normalisation, network and box decoding from one CUDA graph (the input is always `size x size`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import NUMPY_ARRAYS, Inert, latin1_encode, numeric_scalar
from fastforks._core.errors import InputError
from fastforks._core.patch import override
from fastforks._optimize import optimize
from fastforks.detection._cv2_resize import cv2_linear
from fastforks.detection._ops import ops
from fastforks.detection._runner import Forward, Runner
from fastforks.shared.cspnext import Attention
from fastforks.shared.cuda_graphs import GraphCache

from .net import STRIDES, Arch, RTMDet, convert, decode, points

if TYPE_CHECKING:
    from fastforks.detection._runner import Context

MEAN = (123.675, 116.28, 103.53)  # fixed: reference data_preprocessor mean (RGB, 0..255)
STD = (58.395, 57.12, 57.375)  # fixed: reference data_preprocessor std (RGB, 0..255)
PAD = 114.0  # fixed: reference test pipeline `Pad(pad_val=dict(img=(114, 114, 114)))`
SPLIT = 10000  # fixed: mmcv `batched_nms` `split_thr`: from this many boxes on, NMS runs in several calls


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    size: int  # square network input (the image is fitted into it)
    nms_pre: int  # candidates per level that enter NMS
    nms_iou: float  # NMS IoU threshold
    score_threshold: float  # option: minimum class score
    max_detections: int  # option: detections per image
    cuda_graphs: bool  # option: kernels replays the network from a CUDA graph


def classwise_nms(boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, iou: float) -> torch.Tensor:
    """Indices that class-wise NMS keeps, best score first (mmcv `batched_nms`: boxes shifted by their class).

    Below `SPLIT` boxes one NMS call; from there on one call per run of whole classes with at most `SPLIT` boxes (a
    larger class alone): the same boxes survive, and the operator's memory and time, which grow with the square of
    the box count, stay bounded. Measured on COCO at the reference's score threshold 0.001 (about 90 000 candidates
    an image, RTX 5060 Ti): the whole call 95 ms and 5 to 7 GB reserved with one NMS call, 20 ms and under 1.2 GB so.
    """
    shifted = boxes + (labels.to(boxes) * (boxes.max() + 1))[:, None]
    if boxes.shape[0] < SPLIT:
        return ops().nms(shifted, scores, iou)
    counts = torch.bincount(labels).tolist()
    keep, lo, n = [], 0, 0
    for c in range(len(counts) + 1):
        k = counts[c] if c < len(counts) else SPLIT + 1  # past the last class: flush
        if n and n + k > SPLIT:
            m = ((labels >= lo) & (labels < c)).nonzero()[:, 0]
            keep.append(m[ops().nms(shifted[m], scores[m], iou)])
            lo, n = c, 0
        n += k
    kept = torch.cat(keep).sort().values
    return kept[scores[kept].argsort(descending=True, stable=True)]


def fit(h: int, w: int, size: int) -> tuple[int, int]:
    """The resized `(height, width)` of an `h x w` image (mmcv `rescale_size`; at least one pixel per side)."""
    s = size / max(h, w)
    return max(int(h * s + 0.5), 1), max(int(w * s + 0.5), 1)


def canvas(image: torch.Tensor, size: int) -> tuple[torch.Tensor, tuple[float, float]]:
    """`[1, 3, H, W]` RGB in [0, 1] -> the padded `[1, 3, size, size]` canvas (0..255) and the (x, y) factors."""
    h, w = image.shape[-2:]
    nh, nw = fit(h, w, size)
    x = image.new_full((1, 3, size, size), PAD)
    x[..., :nh, :nw] = cv2_linear(image.mul(255).round_(), nh, nw)
    return x, (nw / w, nh / h)


class Portable(Runner):
    """Plain PyTorch: float16 network on CUDA (channels-last), float32 scores, boxes and NMS."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network with the checkpoint (BatchNorm folded) on the device."""
        super().__init__(ctx)
        p: Params = ctx.params
        net = RTMDet(p)
        net.load_state_dict(convert(ctx.state_dict(key="state_dict", strip="", extra=extra())))
        self.mf = torch.channels_last if ctx.device.type == "cuda" else torch.contiguous_format
        self.net = net.eval().to(ctx.device, ctx.torch_dtype).to(memory_format=self.mf)
        self.mean = torch.tensor(MEAN, device=ctx.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=ctx.device).view(1, 3, 1, 1)
        self.grid = points(p.size, p.size, ctx.device)
        cells = [(p.size // s) ** 2 for s in STRIDES]
        self.levels = [(sum(cells[:i]), sum(cells[: i + 1])) for i in range(len(cells))]  # cell ranges per level
        if ctx.device.type == "mps":  # MPS reduces over two axes ~30x slower than over one (same result)
            for m in self.net.modules():
                if isinstance(m, Attention):
                    override(m, _attention_mps)

    def run(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The head maps of a normalised float32 input."""
        return self.net(x.to(self.ctx.torch_dtype).contiguous(memory_format=self.mf))

    def predict(self, x: torch.Tensor, grid: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """float32 class logits `[B, N, classes]` and boxes `[B, N, 4]` (network pixels) of `x` in 0..255."""
        return decode(self.run((x - self.mean) / self.std), *grid, exp_on_reg=self.ctx.params.exp_on_reg)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits `[N, classes]` and boxes `[N, 4]` of the canvas."""
        logits, boxes = self.predict(x, self.grid)
        return logits[0], boxes[0]

    def candidates(self, logits: torch.Tensor, score_threshold: float) -> tuple[torch.Tensor, ...]:
        """Cell, label and score of every class score at or above the threshold (cells ascending)."""
        scores = logits.sigmoid()
        cell, label = (scores >= score_threshold).nonzero(as_tuple=True)
        return cell, label, scores[cell, label]

    def __call__(self, image: torch.Tensor, score_threshold: float) -> dict[str, torch.Tensor]:
        """The reference's test pipeline and box postprocessing on one image."""
        p: Params = self.ctx.params
        x, (sx, sy) = canvas(image, p.size)
        logits, boxes = self._forward(x)
        cell, label, score = self.candidates(logits, score_threshold)
        if cell.numel() > p.nms_pre:  # only then can a level hold more than `nms_pre`
            parts = []
            for lo, hi in self.levels:
                m = ((cell >= lo) & (cell < hi)).nonzero()[:, 0]
                parts.append(m[score[m].topk(p.nms_pre).indices] if m.numel() > p.nms_pre else m)
            sel = torch.cat(parts)
            cell, label, score = cell[sel], label[sel], score[sel]
        out = boxes[cell].clamp_(0, p.size) / boxes.new_tensor([sx, sy, sx, sy])
        ok = (out[:, 2] > out[:, 0]) & (out[:, 3] > out[:, 1])  # the reference's `min_bbox_size = 0`
        out, label, score = out[ok], label[ok], score[ok]
        if out.numel():
            keep = classwise_nms(out, label, score, p.nms_iou)[: p.max_detections]
            out, label, score = out[keep], label[keep], score[keep]
        return {"boxes": out, "scores": score, "labels": label}

    def module(self) -> torch.nn.Module:
        """The network: RGB `[B, 3, H, W]` in `[0, 1]`, H and W multiples of 32 (no resize, no padding; default 640).

        Returns the class logits `[B, N, classes]` (apply a sigmoid) and the boxes `[B, N, 4]` (`x0, y0, x1, y1` in
        pixels of the given input, before clamping and NMS); N = the cells of the stride 8, 16 and 32 maps.
        """

        def fn(_net: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401 - the network
            h, w = x.shape[-2:]
            if h % STRIDES[-1] or w % STRIDES[-1]:
                msg = (
                    f"RTMDet module() / export takes H and W that are multiples of {STRIDES[-1]} (it does not pad), "
                    f"got {h}x{w}; the model call fits and pads any size"
                )
                raise InputError(msg)
            grid = self.grid if (h, w) == (self.ctx.params.size,) * 2 else points(h, w, x.device)
            return self.predict(x.mul(255).round(), grid)

        return Forward(self.net, fn, output_names=("logits", "boxes"))


class Kernels(Portable):
    """CUDA: the network through `fastforks._optimize`, replayed with normalisation and box decoding from a graph."""

    def __init__(self, ctx: Context) -> None:
        """The portable network, then its optimized copy and the graph cache."""
        super().__init__(ctx)
        self.net = optimize(copy.deepcopy(self.net.float()), mode="kernels", device=ctx.device, family="cnn")
        self.graphs = GraphCache(self._eager, graphs=1) if ctx.params.cuda_graphs else None

    def run(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The head maps of a normalised float32 input (the optimized network converts dtype and layout itself)."""
        return self.net(x)

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        """Logits and boxes of the canvas as one float32 `[N, classes + 4]` tensor (one graph output)."""
        logits, boxes = self.predict(x, self.grid)
        return torch.cat((logits[0], boxes[0]), -1)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.graphs.run((x.device, tuple(x.shape)), x) if self.graphs is not None else None
        if out is None:
            out = self._eager(x)
        return out[:, :-4], out[:, -4:]

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        if self.graphs is not None:
            self.graphs.release()


def _attention_mps(self: Attention, x: torch.Tensor) -> torch.Tensor:
    """`Attention.forward` with the spatial mean over one flattened axis (MPS: `mean((2, 3))` is ~30x slower)."""
    return x * F.hardsigmoid(self.fc(x.flatten(2).mean(-1)[..., None, None]))


def extra() -> dict[tuple[str, str], object]:
    """What the mmdet checkpoints pickle besides tensors: numpy arrays / scalars of the meta, and the training log."""
    logs = {
        ("mmengine.logging.history_buffer", "HistoryBuffer"): Inert,
        ("mmengine.logging.message_hub", "MessageHub"): Inert,
    }
    scalar = numeric_scalar
    scalars = {(m, "scalar"): scalar for m in ("numpy.core.multiarray", "numpy._core.multiarray")}
    return {**NUMPY_ARRAYS, ("_codecs", "encode"): latin1_encode, **logs, **scalars}


RUNNERS = {"portable": Portable, "kernels": Kernels}
