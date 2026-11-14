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

from fastforks._optimize import optimize
from fastforks.pose._runner import Forward, Runner

from .net import ANCHORS, HEAD, Arch, RetinaFace, convert

if TYPE_CHECKING:
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


def anchors(h: int, w: int, device: torch.device) -> torch.Tensor:
    """`[N, 4]` (cx, cy, side, side) in input pixels, upstream `PriorBox` order: level, row, column, anchor."""
    out = []
    for step, sizes in zip(STEPS, MIN_SIZES, strict=True):
        ys = (torch.arange(math.ceil(h / step), device=device, dtype=torch.float32) + 0.5) * step
        xs = (torch.arange(math.ceil(w / step), device=device, dtype=torch.float32) + 0.5) * step
        s = torch.tensor(sizes, device=device, dtype=torch.float32)
        cy, cx, side = torch.meshgrid(ys, xs, s, indexing="ij")
        out.append(torch.stack([cx, cy, side, side], -1).reshape(-1, 4))
    return torch.cat(out)


def nms(boxes: torch.Tensor, iou: float) -> torch.Tensor:
    """Indices greedy NMS keeps of `[n, 4]` boxes sorted best first (a box goes if a kept one overlaps it > `iou`)."""
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    wh = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:]) - torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    overlap = wh.clamp(min=0).prod(-1)
    over = (overlap / (area[:, None] + area[None] - overlap) > iou).cpu()
    gone = torch.zeros(boxes.shape[0], dtype=torch.bool)
    keep = []
    for i in range(boxes.shape[0]):
        if not gone[i]:
            keep.append(i)
            gone |= over[i]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def split(maps: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Head maps of one image -> (face logits `[N, 2]`, box offsets `[N, 4]`, landmark offsets `[N, 10]`)."""
    rows = torch.cat([m[0].permute(1, 2, 0).reshape(-1, HEAD) for m in maps])
    n = rows.shape[0] * ANCHORS
    c, b = 2 * ANCHORS, 6 * ANCHORS
    return rows[:, :c].reshape(n, 2), rows[:, c:b].reshape(n, 4), rows[:, b:].reshape(n, 2 * LANDMARKS)


def faces(maps: tuple[torch.Tensor, ...], h: int, w: int, threshold: float, iou: float) -> dict[str, torch.Tensor]:
    """The faces of one image from its float32 head maps (module docstring)."""
    logits, box, lms = split(maps)
    score = (logits[:, 1] - logits[:, 0]).sigmoid()  # = softmax(logits)[:, 1]
    idx = (score > threshold).nonzero()[:, 0]
    idx = idx[score[idx].argsort(descending=True, stable=True)]
    prior = anchors(h, w, score.device)[idx]
    centre, side = prior[:, :2], prior[:, 2:]
    xy = centre + box[idx, :2] * VARIANCE[0] * side
    wh = side * torch.exp(box[idx, 2:] * VARIANCE[1])
    boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)
    keep = nms(boxes, iou)
    idx, boxes, centre, side = idx[keep], boxes[keep], centre[keep], side[keep]
    kp = centre[:, None] + lms[idx].view(-1, LANDMARKS, 2) * VARIANCE[0] * side[:, None]
    return {
        "keypoints": kp,
        "keypoint_scores": score[idx, None].expand(-1, LANDMARKS).contiguous(),
        "scores": score[idx],
        "boxes": boxes,
    }


class Portable(Runner):
    """`fastforks._optimize` on the network (BatchNorm folded, fp16 channels-last on CUDA); float32 decoding."""

    dtypes = ("float32", "float16")
    Params = Params
    needs_boxes: ClassVar[bool] = False

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

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """The faces in the image with their landmarks."""
        assert boxes is None  # noqa: S101 - the task API rejects boxes for one-stage families
        a: Params = self.ctx.params
        return faces(self.dense(image), *image.shape[-2:], a.score_threshold, a.nms_iou)

    def module(self) -> torch.nn.Module:
        """The network: images `[B, 3, H, W]` RGB in [0, 1] -> the stride 8 / 16 / 32 head maps."""

        def fn(_net: Any, x: torch.Tensor) -> tuple[torch.Tensor, ...]:  # noqa: ANN401 - the network
            return self.dense(x)

        return Forward(self.net, fn)


RUNNERS = {"portable": Portable, "kernels": Portable}
