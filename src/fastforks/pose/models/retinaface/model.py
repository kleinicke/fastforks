"""RetinaFace runners: one image -> the faces in it, each with a box and five landmarks (facexlib `detect_faces`).

- input (`use_origin_size=True`, facexlib's default): the image at its own size, BGR, 0..255 minus the mean
  (104, 117, 123); no resize
- network (`net.RetinaFace`): per cell of the stride 8 / 16 / 32 maps and per anchor (sizes 16, 32 / 64, 128 /
  256, 512 px) a face logit pair, a box and 5 landmark offsets
- faces: softmax face probability above `score_threshold`, best first, greedy NMS on the boxes (`nms_iou`); boxes
  `centre + offset * 0.1 * size`, `size * exp(offset * 0.2)`, landmarks `centre + offset * 0.1 * size` (the anchor
  centre `((j, i) + 0.5) * stride`, `size` its side)

`scores` is the face probability; RetinaFace has no per-landmark score, so `keypoint_scores` repeats it. Landmarks in
upstream order: the eye, then the mouth corner, on the image's left first (the person's right on a frontal face).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from fastforks._core.upload import device_tensor
from fastforks._optimize import optimize
from fastforks.kernels import box_nms, cudnn_hacc
from fastforks.pose._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import ANCHORS, HEAD, Arch, RetinaFace, convert

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.pose._runner import Context

MEAN_BGR = (104.0, 117.0, 123.0)  # fixed: reference input mean (BGR)
STEPS = (8, 16, 32)  # fixed: reference feature strides
MIN_SIZES = ((16, 32), (64, 128), (256, 512))  # fixed: reference anchor sides per level
VARIANCE = (0.1, 0.2)  # fixed: reference box coding variances (centre, size)
LANDMARKS = 5  # fixed: reference landmarks per face


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    nms_iou: float  # NMS IoU threshold
    score_threshold: float  # option: minimum face probability
    cuda_graphs: bool  # option: kernels replays the network from CUDA graphs for small repeated shapes


def anchors(h: int, w: int, device: torch.device) -> torch.Tensor:
    """`[N, 4]` (cx, cy, side, side) in input pixels, upstream `PriorBox` order: level, row, column, anchor."""
    out = []
    for step, sizes in zip(STEPS, MIN_SIZES, strict=True):
        ys = (torch.arange(math.ceil(h / step), device=device, dtype=torch.float32) + 0.5) * step
        xs = (torch.arange(math.ceil(w / step), device=device, dtype=torch.float32) + 0.5) * step
        s = torch.cat([torch.full((1,), float(v), device=device) for v in sizes])  # no host copy: graph capturable
        cy, cx, side = torch.meshgrid(ys, xs, s, indexing="ij")
        out.append(torch.stack([cx, cy, side, side], -1).reshape(-1, 4))
    return torch.cat(out)


def nms(boxes: torch.Tensor, iou: float) -> torch.Tensor:
    """Indices greedy NMS keeps of `[n, 4]` boxes sorted best first (a box goes if a kept one overlaps it > `iou`)."""
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    rows = [torch.zeros(0, boxes.shape[0], dtype=torch.bool)]
    for s in range(0, boxes.shape[0], 256):  # row blocks: n x n float pairs at low thresholds are 100s of MB
        b, a = boxes[s : s + 256], area[s : s + 256, None]
        wh = torch.minimum(b[:, None, 2:], boxes[None, :, 2:]) - torch.maximum(b[:, None, :2], boxes[None, :, :2])
        overlap = wh.clamp(min=0).prod(-1)
        rows.append((overlap / (a + area[None] - overlap) > iou).cpu())
    over = torch.cat(rows)
    gone = torch.zeros(boxes.shape[0], dtype=torch.bool)
    keep = []
    for i in range(boxes.shape[0]):
        if not gone[i]:
            keep.append(i)
            gone |= over[i]
    return device_tensor(keep, dtype=torch.long, device=boxes.device)


def split(maps: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Head maps of one image -> (face logits `[N, 2]`, box offsets `[N, 4]`, landmark offsets `[N, 10]`)."""
    logits, box, lms = split_batch(maps)
    return logits[0], box[0], lms[0]


def split_batch(maps: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Head maps `[B, HEAD, h, w]` -> (face logits `[B, N, 2]`, box offsets `[B, N, 4]`, landmarks `[B, N, 10]`)."""
    rows = torch.cat([m.permute(0, 2, 3, 1).reshape(m.shape[0], -1, HEAD) for m in maps], 1)
    bn, n = rows.shape[0], rows.shape[1] * ANCHORS
    c, b = 2 * ANCHORS, 6 * ANCHORS
    return rows[..., :c].reshape(bn, n, 2), rows[..., c:b].reshape(bn, n, 4), rows[..., b:].reshape(bn, n, 10)


def decode(maps: tuple[torch.Tensor, ...], h: int, w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every anchor of float32 head maps `[B, HEAD, h, w]` of `h` x `w` images decoded (module docstring).

    Scores `[B, N]`, boxes `[B, N, 4]` and landmarks `[B, N, LANDMARKS, 2]`; elementwise per anchor, so selecting
    anchors afterwards gives the same values as decoding only the selected ones.
    """
    logits, box, lms = split_batch(maps)
    score = (logits[..., 1] - logits[..., 0]).sigmoid()  # = softmax(logits)[..., 1]
    prior = anchors(h, w, score.device)
    centre, side = prior[:, :2], prior[:, 2:]
    xy = centre + box[..., :2] * VARIANCE[0] * side
    wh = side * torch.exp(box[..., 2:] * VARIANCE[1])
    boxes = torch.cat([xy - wh / 2, xy + wh / 2], -1)
    kp = centre[:, None] + lms.unflatten(-1, (LANDMARKS, 2)) * VARIANCE[0] * side[:, None]
    return score, boxes, kp


def select(
    cells: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    threshold: float,
    iou: float,
    suppress: Callable[[torch.Tensor, float], torch.Tensor] = nms,
) -> dict[str, torch.Tensor]:
    """The faces of one image from its decoded anchors (`decode`, one image): threshold, best first, suppression."""
    score, boxes, kp = cells
    idx = (score > threshold).nonzero()[:, 0]
    idx = idx[score[idx].argsort(descending=True, stable=True)]
    keep = idx[suppress(boxes[idx], iou)]
    sc = score[keep]
    return {
        "keypoints": kp[keep],
        "keypoint_scores": sc[:, None].expand(-1, LANDMARKS).contiguous(),
        "scores": sc,
        "boxes": boxes[keep],
    }


def faces(  # noqa: PLR0913 - image dimensions, thresholds and suppression rule
    maps: tuple[torch.Tensor, ...],
    h: int,
    w: int,
    threshold: float,
    iou: float,
    *,
    suppress: Callable[[torch.Tensor, float], torch.Tensor] = nms,
) -> dict[str, torch.Tensor]:
    """The faces of one image from its float32 head maps (module docstring)."""
    score, boxes, kp = decode(maps, h, w)
    return select((score[0], boxes[0], kp[0]), threshold, iou, suppress)


class Portable(Runner):
    """`fastforks._optimize` on the network (BatchNorm folded, fp16 channels-last on CUDA); float32 decoding."""

    dtypes = ("float32", "float16")
    Params = Params
    suppress = staticmethod(nms)
    needs_boxes: ClassVar[bool] = False
    compiled_methods = ("__call__", "estimate_batch")
    batch = None
    # images x H x W per network call (`estimate_batch`): launch bound (RTX 5060 Ti, Windows, fp16,
    # 8 frames 480 x 640, ms per frame:
    # mobilenet0.25 7.0 -> 3.2 (kernels with CUDA graphs 1.4 -> 0.92), resnet50 9.3 -> 5.6 (3.5 -> 2.3)); 1080p runs
    # one image per call
    batch_pixels = 2_100_000

    def __init__(self, ctx: Context) -> None:
        """Build the network with the checkpoint and optimise it for the mode."""
        super().__init__(ctx)
        net = RetinaFace(ctx.params)
        net.load_state_dict(convert(ctx.state_dict()))
        self.net = optimize(net.eval(), mode=ctx.mode, device=ctx.device, fp16=ctx.dtype == "float16", family="cnn")
        self.mean = torch.tensor(MEAN_BGR, device=ctx.device).view(1, 3, 1, 1)
        if ctx.device.type == "cuda":  # the fp32 weights of the optimise passes stay cached otherwise
            torch.cuda.empty_cache()

    def dense(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """float32 head maps of `image` `[B, 3, H, W]` RGB in [0, 1]."""
        return tuple(self.net(image.flip(1) * 255 - self.mean))

    def decoded(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every anchor of `image` `[B, 3, H, W]` decoded (`decode`): scores, boxes, landmarks."""
        return decode(self.dense(image), *image.shape[-2:])

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """The faces in the image with their landmarks."""
        assert boxes is None  # noqa: S101 - the task API rejects boxes for one-stage families
        return self._faces(image)[0]

    def estimate_batch(self, images: torch.Tensor, boxes: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """The faces of each image of `[B, 3, H, W]`: one network call and decoding, selected per image."""
        assert all(b is None for b in boxes)  # noqa: S101 - one-stage: the task API passes no boxes
        return self._faces(images)

    def _faces(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        a: Params = self.ctx.params
        score, box, kp = self.decoded(images)
        return [
            select((score[i], box[i], kp[i]), a.score_threshold, a.nms_iou, self.suppress)
            for i in range(images.shape[0])
        ]

    def module(self) -> torch.nn.Module:
        """The network: images `[B, 3, H, W]` RGB in [0, 1] -> the stride 8 / 16 / 32 head maps."""

        def fn(_net: Any, x: torch.Tensor) -> tuple[torch.Tensor, ...]:  # noqa: ANN401 - the network
            return self.dense(x)

        return Forward(self.net, fn, output_names=("head_s8", "head_s16", "head_s32"))


# Network + every anchor's decoding: launch bound (RTX 5060 Ti, Windows, fp16, mobilenet0.25, ms per image, eager ->
# graph: 480 x 640 5.0 -> 1.4, 8 x 480 x 640 1.7 -> 0.92, 1080p 5.3 -> 3.1; +190 MB reserved at 480 x 640)
_GRAPH_PIXELS = 2_100_000  # fixed: batch x H x W, a speed threshold (not the architecture)
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Portable):
    """Run greedy suppression on the device with a bounded overlap buffer.

    A repeated input shape up to `_GRAPH_PIXELS` replays the network and the decoding of every anchor from a CUDA
    graph (`fastforks.shared.cuda_graphs`); selecting the faces (a data-dependent number) stays eager.
    """

    suppress = staticmethod(box_nms.indices)

    def __init__(self, ctx: Context) -> None:
        """Build the network as portable does, with a CUDA graph cache for it."""
        super().__init__(ctx)
        self.graphs = SmallShapeGraphs(
            super().decoded,
            pixels=_GRAPH_PIXELS,
            graphs=_GRAPHS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )

    def decoded(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every anchor of `image` decoded, network and decoding from a graph for a small repeated shape."""
        return self.graphs(image)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self.graphs.release()


class _OneImage(Portable):
    """Portable for the MLX port: one image per call, as before batching (not measured on Apple GPUs)."""

    batch = 1


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_OneImage),
}
