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
network and box decoding from one CUDA graph per batch size (the input is always `size x size`; one image and one
batch size are kept). The resize to the canvas replays from a graph of its own per input shape (two kept), captured
when a shape comes a second time.
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
    """`[B, 3, H, W]` RGB in [0, 1] or uint8 -> the padded `[B, 3, size, size]` canvas (0..255), resize factor."""
    h, w = image.shape[-2:]
    nh, nw = fit(h, w, size)
    x = image.new_full((image.shape[0], 3, size, size), PAD, dtype=torch.float32)
    x[..., :nh, :nw] = cv2_linear(image, nh, nw, 1.0 if image.dtype == torch.uint8 else 255.0)
    return x, min(size / h, size / w)


class Portable(Runner):
    """Plain PyTorch: float16 network on CUDA (channels-last), float32 scores, boxes and NMS."""

    dtypes = ("float32", "float16")
    Params = Params
    uint8_input = True
    batch = None  # `detect_batch`: a batch of images in one network call (NMS per image)
    compiled_methods = ("__call__", "detect_batch")

    def __init__(self, ctx: Context) -> None:
        """Build the network with the checkpoint (BatchNorm folded) on the device."""
        super().__init__(ctx)
        self.mf = torch.channels_last if ctx.device.type == "cuda" else torch.contiguous_format
        self.net = _network(ctx).to(ctx.device, ctx.torch_dtype).to(memory_format=self.mf)
        self.grid = cells(ctx.params.size, ctx.params.size, ctx.device)

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
        """Boxes `[B, N, 4]`, scores `[B, N]` and labels `[B, N]` of the canvases."""
        return self.predict(x, self.grid)

    def __call__(self, image: torch.Tensor, score_threshold: float) -> dict[str, torch.Tensor]:
        """The reference's preprocessing and postprocessing on one image."""
        return self.detect_batch(image, score_threshold)[0]

    def detect_batch(self, images: torch.Tensor, score_threshold: float) -> list[dict[str, torch.Tensor]]:
        """`__call__` on each image of `[B, 3, H, W]`, the network run once on the batch."""
        x, r = self.fit_canvas(images)
        boxes, score, label = self._forward(x)
        return [self._finish(boxes[i], score[i], label[i], r, score_threshold) for i in range(len(boxes))]

    def fit_canvas(self, images: torch.Tensor) -> tuple[torch.Tensor, float]:
        """The padded canvases of `[B, 3, H, W]` images and the resize factor (see `canvas`)."""
        return canvas(images, self.ctx.params.size)

    def _finish(
        self, boxes: torch.Tensor, score: torch.Tensor, label: torch.Tensor, r: float, score_threshold: float
    ) -> dict[str, torch.Tensor]:
        """Threshold, NMS and the best `max_detections` of one image's cells."""
        p: Params = self.ctx.params
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
        self.graphs = GraphCache(self._eager, graphs=2) if ctx.params.cuda_graphs else None  # one image + one batch
        # the resize to the canvas, a graph of its own per input shape: about 40 small launches; images of varying
        # sizes still replay the network graph, which only depends on the batch size
        self.canvases = GraphCache(self._canvas, graphs=2) if ctx.params.cuda_graphs else None

    def run(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The head maps of a float32 input (the optimized network converts dtype and layout itself)."""
        return self.net(x)

    def _canvas(self, images: torch.Tensor) -> torch.Tensor:
        return canvas(images, self.ctx.params.size)[0]

    def fit_canvas(self, images: torch.Tensor) -> tuple[torch.Tensor, float]:
        """The canvases from the input shape's graph once it is captured, else eagerly (and the factors)."""
        key = (images.device, images.dtype, tuple(images.shape))
        x = self.canvases.run(key, images) if self.canvases is not None else None
        if x is None:
            return canvas(images, self.ctx.params.size)
        (h, w), size = images.shape[-2:], self.ctx.params.size
        return x, min(size / h, size / w)

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        """Boxes, score and label of the canvases as one float32 `[B, N, 6]` tensor (one graph output)."""
        boxes, score, label = self.predict(x, self.grid)
        return torch.cat((boxes, score[..., None], label[..., None].float()), -1)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.graphs.run((x.device, tuple(x.shape)), x) if self.graphs is not None else None
        if out is None:
            out = self._eager(x)
        return out[..., :4], out[..., 4], out[..., 5].long()

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pools."""
        for cache in (self.graphs, self.canvases):
            if cache is not None:
                cache.release()


class _MlxPortable(Portable):
    """Apple MLX: the network written in MLX operations (`mlx_net`), canvases, decoding and NMS as portable.

    Kernels mode dispatches the convs to the `conv_mlx` kernels as the graph translator does. 1 x 1 convs stay
    `mx.conv2d` (not the translator's GEMM): they gave ~3 % on yolox-s, and the yolox-x kernels float16 golden
    drifted past its limit with them (boxes mean |diff| 0.80 of 0.2).
    """

    module = Runner.module  # the MLX network is not exportable

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts the portable network's weights)."""
        Runner.__init__(self, ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx
        from fastforks.shared.cspnext import mlx_net as csp  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        self.graph = "kernels" if ctx.mode == "kernels" else "portable"

        def make(net: YOLOX) -> mlx_net.MlxYOLOX:
            with csp.run_dtype_silu():  # as the exported network: the float32 one drifts yolox-m / -s float16 (csp)
                return mlx_net.MlxYOLOX(net, ctx.dtype, self.graph)

        self.mlx = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            make,
            real=lambda: _network(ctx),
            skeleton=lambda: YOLOX(ctx.params),
            sources=mlx_net.SOURCES,
        )
        self.net = None
        self.grid = cells(ctx.params.size, ctx.params.size, ctx.device)

    def run(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The float32 NCHW head maps of a float32 input in 0..255."""
        from fastforks._core.mlx import mx, to_mx, to_torch  # noqa: PLC0415 - imports mlx
        from fastforks.shared.cspnext import mlx_net as csp  # noqa: PLC0415 - imports mlx

        csp.options(self.graph, gemm1x1=False)
        maps = self.mlx(to_mx(x.float().permute(0, 2, 3, 1)))
        mx.eval(*maps)
        return tuple(to_torch(m).permute(0, 3, 1, 2) for m in maps)


def _network(ctx: Context) -> YOLOX:
    """The float32 CPU network with the checkpoint, its batch norms folded."""
    net = YOLOX(ctx.params)
    net.load_state_dict(convert(ctx.state_dict(key="model", strip="")))
    return net.eval()


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
    "kernels_mlx": _MlxPortable,
}
