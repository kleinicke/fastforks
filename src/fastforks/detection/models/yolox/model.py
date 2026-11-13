"""YOLOX runners: image -> boxes, scores and classes (the reference's `ValTransform` + `postprocess`).

Follows `tools/demo.py` / the COCO evaluator of the reference:

- input: the uint8 image is resized to fit `size x size` keeping its aspect ratio (factor `r = min(size / h,
  size / w)`, new sides `int(side * r)`; OpenCV's 8-bit `INTER_LINEAR`, reproduced on the device), put top-left on a
  `size x size` canvas of 114s, 0..255 without normalisation. Small images are enlarged
- network: CSPDarknet, PAFPN, decoupled head -> per cell of the stride 8 / 16 / 32 maps the box (centre, size),
  an objectness logit and a logit per class
- postprocess: each cell's score is `sigmoid(objectness) * max sigmoid(class)` with that class as its label; the
  cells at or above the score threshold, boxes divided by `r`, class-wise NMS (`nms_iou`, torchvision's
  `batched_nms`, the operator the reference calls), best `max_detections`

The reference keeps every box after NMS and tests COCO with a score threshold of 0.01; here `score_threshold` (0.05)
and `max_detections` (100) are options as in the other detection families. Boxes are not clamped to the image, as
upstream.

Portable runs the network in float16 on CUDA (channels-last), scores and boxes in float32. Kernels runs the whole
network through `fastforks._optimize` (fused conv + SiLU / residual / concat kernels, fp16, channels-last) and replays
network and box decoding from one CUDA graph (the input is always `size x size`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.errors import InputError
from fastforks._optimize import optimize
from fastforks.detection._cv2_resize import cv2_linear
from fastforks.detection._ops import ops
from fastforks.detection._runner import Forward, Runner
from fastforks.shared.cuda_graphs import GraphCache

from .net import STRIDES, YOLOX, Arch, cells, convert, decode

if TYPE_CHECKING:
    from fastforks.detection._runner import Context

PAD = 114.0  # fixed: reference `preproc` canvas value


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    size: int  # square network input (the image is fitted into it)
    nms_iou: float  # NMS IoU threshold
    score_threshold: float  # option: minimum score (objectness x class)
    max_detections: int  # option: detections per image
    cuda_graphs: bool  # option: kernels replays the network from a CUDA graph


def fit(h: int, w: int, size: int) -> tuple[int, int]:
    """The resized `(height, width)` of an `h x w` image (reference `preproc`: truncated; at least one pixel)."""
    r = min(size / h, size / w)
    return max(int(h * r), 1), max(int(w * r), 1)


def canvas(image: torch.Tensor, size: int) -> tuple[torch.Tensor, float]:
    """`[1, 3, H, W]` RGB in [0, 1] or uint8 -> the padded `[1, 3, size, size]` canvas (0..255), resize factor."""
    h, w = image.shape[-2:]
    nh, nw = fit(h, w, size)
    x = image.new_full((1, 3, size, size), PAD, dtype=torch.float32)
    x[..., :nh, :nw] = cv2_linear(image, nh, nw, 1.0 if image.dtype == torch.uint8 else 255.0)
    return x, min(size / h, size / w)


class Portable(Runner):
    """Plain PyTorch: float16 network on CUDA (channels-last), float32 scores, boxes and NMS."""

    dtypes = ("float32", "float16")
    Params = Params
    uint8_input = True

    def __init__(self, ctx: Context) -> None:
        """Build the network with the checkpoint (BatchNorm folded) on the device."""
        super().__init__(ctx)
        p: Params = ctx.params
        net = YOLOX(p)
        net.load_state_dict(convert(ctx.state_dict(key="model", strip="")))
        self.mf = torch.channels_last if ctx.device.type == "cuda" else torch.contiguous_format
        self.net = net.eval().to(ctx.device, ctx.torch_dtype).to(memory_format=self.mf)
        self.grid = cells(p.size, p.size, ctx.device)

    def run(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The head maps of a float32 input in 0..255."""
        return self.net(x.to(self.ctx.torch_dtype).contiguous(memory_format=self.mf))

    def predict(
        self, x: torch.Tensor, grid: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """float32 boxes `[B, N, 4]` (network pixels), scores `[B, N]` and labels `[B, N]` of `x` in 0..255."""
        boxes, obj, cls = decode(self.run(x), *grid)
        conf, label = cls.max(-1)
        return boxes, obj * conf, label

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Boxes `[N, 4]`, scores `[N]` and labels `[N]` of the canvas."""
        boxes, score, label = self.predict(x, self.grid)
        return boxes[0], score[0], label[0]

    def __call__(self, image: torch.Tensor, score_threshold: float) -> dict[str, torch.Tensor]:
        """The reference's preprocessing and postprocessing on one image."""
        p: Params = self.ctx.params
        x, r = canvas(image, p.size)
        boxes, score, label = self._forward(x)
        keep = (score >= score_threshold).nonzero()[:, 0]
        boxes, score, label = boxes[keep] / r, score[keep], label[keep]
        if boxes.numel():
            keep = ops().batched_nms(boxes, score, label, p.nms_iou)[: p.max_detections]
            boxes, score, label = boxes[keep], score[keep], label[keep]
        return {"boxes": boxes, "scores": score, "labels": label}

    def module(self) -> torch.nn.Module:
        """The network: RGB `[B, 3, H, W]` in `[0, 1]`, H and W multiples of 32 (no resize, no padding; default 640).

        Returns the boxes `[B, N, 4]` (`x0, y0, x1, y1` in pixels of the given input, before NMS), the scores `[B, N]`
        (objectness x best class probability) and that class `[B, N]`; N = the cells of the stride 8, 16 and 32 maps.
        """

        def fn(_net: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: ANN401 - the network
            h, w = x.shape[-2:]
            if h % STRIDES[-1] or w % STRIDES[-1]:
                msg = (
                    f"YOLOX module() / export takes H and W that are multiples of {STRIDES[-1]} (it does not pad), "
                    f"got {h}x{w}; the model call fits and pads any size"
                )
                raise InputError(msg)
            grid = self.grid if (h, w) == (self.ctx.params.size,) * 2 else cells(h, w, x.device)
            return self.predict(x.mul(255).round(), grid)

        return Forward(self.net, fn, output_names=("boxes", "scores", "labels"))


class Kernels(Portable):
    """CUDA: the network through `fastforks._optimize`, replayed with box decoding from a graph."""

    def __init__(self, ctx: Context) -> None:
        """The portable network, then its optimized copy and the graph cache."""
        super().__init__(ctx)
        self.net = optimize(copy.deepcopy(self.net.float()), mode="kernels", device=ctx.device, family="cnn")
        self.graphs = GraphCache(self._eager, graphs=1) if ctx.params.cuda_graphs else None

    def run(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The head maps of a float32 input (the optimized network converts dtype and layout itself)."""
        return self.net(x)

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        """Boxes, score and label of the canvas as one float32 `[N, 6]` tensor (one graph output)."""
        boxes, score, label = self.predict(x, self.grid)
        return torch.cat((boxes[0], score[0, :, None], label[0, :, None].float()), -1)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.graphs.run((x.device, tuple(x.shape)), x) if self.graphs is not None else None
        if out is None:
            out = self._eager(x)
        return out[:, :4], out[:, 4], out[:, 5].long()

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        if self.graphs is not None:
            self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
