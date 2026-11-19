"""YuNet runners: one image -> the faces in it, each with a box and five landmarks.

As libfacedetection.train's `eval_widerface --mode origin` input and `YuNetPostprocessor`:

- input: the image at its own size, BGR, 0..255 (no mean, no scale), zero-padded bottom / right to multiples of 32
- network (`net.YuNet`): per cell of the stride 8 / 16 / 32 maps (anchor-free, one prediction per cell) a class and an
  objectness logit, a box and 5 landmark offsets
- faces: `sigmoid(class) * sigmoid(objectness)` at or above `score_threshold`, best first, greedy NMS on the boxes
  (`nms_iou`); the cell's point `(j, i) * stride` (no half-cell offset) plus `offset * stride` for the box centre and
  the landmarks, `exp(offset) * stride` for the box size

`scores` is the face score; YuNet has no per-landmark score, so `keypoint_scores` repeats it. Landmarks in upstream
order: right eye, left eye, nose tip, right and left mouth corner (of the person).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F

from fastforks._optimize import optimize
from fastforks.kernels import box_nms
from fastforks.pose._runner import Forward, Runner
from fastforks.pose.models.retinaface.model import nms
from fastforks.shared import mlx_wrap

from .net import HEAD, Arch, YuNet, convert

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.pose._runner import Context

STRIDES = (8, 16, 32)  # fixed: reference feature strides
DIVISOR = 32  # fixed: reference evaluation padding (`Pad(size_divisor=32)`)
LANDMARKS = 5  # fixed: reference landmarks per face


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    nms_iou: float  # NMS IoU threshold
    score_threshold: float  # option: minimum face score


def points(maps: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """`[N, 3]` (x, y, stride) per cell, upstream `MlvlPointGenerator(offset=0)` order: level, row, column."""
    out = []
    for m, s in zip(maps, STRIDES, strict=True):
        h, w = m.shape[-2:]
        ys = torch.arange(h, device=m.device, dtype=torch.float32) * s
        xs = torch.arange(w, device=m.device, dtype=torch.float32) * s
        y, x = torch.meshgrid(ys, xs, indexing="ij")
        out.append(torch.stack([x, y, torch.full_like(x, s)], -1).reshape(-1, 3))
    return torch.cat(out)


def faces(
    maps: tuple[torch.Tensor, ...],
    threshold: float,
    iou: float,
    suppress: Callable[[torch.Tensor, float], torch.Tensor] = nms,
) -> dict[str, torch.Tensor]:
    """The faces of one image from its float32 head maps (module docstring)."""
    rows = torch.cat([m[0].permute(1, 2, 0).reshape(-1, HEAD) for m in maps])
    score = rows[:, 0].sigmoid() * rows[:, 1].sigmoid()
    idx = (score >= threshold).nonzero()[:, 0]
    idx = idx[score[idx].argsort(descending=True, stable=True)]
    pt = points(maps)[idx]
    xy, s = pt[:, :2], pt[:, 2:]
    r = rows[idx]
    centre, wh = r[:, 2:4] * s + xy, r[:, 4:6].exp() * s
    boxes = torch.cat([centre - wh / 2, centre + wh / 2], 1)
    keep = suppress(boxes, iou)
    xy, s = xy[keep], s[keep]
    kp = r[keep, 6:].view(-1, LANDMARKS, 2) * s[:, None] + xy[:, None]
    sc = score[idx[keep]]
    return {
        "keypoints": kp,
        "keypoint_scores": sc[:, None].expand(-1, LANDMARKS).contiguous(),
        "scores": sc,
        "boxes": boxes[keep],
    }


class Portable(Runner):
    """`fastforks._optimize` on the network (BatchNorm folded, fp16 channels-last on CUDA); float32 decoding."""

    dtypes = ("float32", "float16")
    Params = Params
    suppress = staticmethod(nms)
    needs_boxes: ClassVar[bool] = False

    def __init__(self, ctx: Context) -> None:
        """Build the network with the checkpoint (heads merged) and optimise it for the mode."""
        super().__init__(ctx)
        net = YuNet(ctx.params)
        sd = ctx.state_dict()
        net.load_state_dict(convert(sd.get("state_dict", sd)))  # ty: ignore[invalid-argument-type]
        self.net = optimize(net.eval(), mode=ctx.mode, device=ctx.device, fp16=ctx.dtype == "float16", family="cnn")
        if ctx.device.type == "cuda":  # the fp32 weights of the optimise passes stay cached otherwise
            torch.cuda.empty_cache()

    def dense(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """float32 head maps of `image` `[B, 3, H, W]` RGB in [0, 1] (padded to multiples of 32 here)."""
        h, w = image.shape[-2:]
        x = F.pad(image.flip(1) * 255, (0, -w % DIVISOR, 0, -h % DIVISOR))
        return tuple(m.float() for m in self.net(x))

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """The faces in the image with their landmarks."""
        assert boxes is None  # noqa: S101 - the task API rejects boxes for one-stage families
        a: Params = self.ctx.params
        return faces(self.dense(image), a.score_threshold, a.nms_iou, self.suppress)

    def module(self) -> torch.nn.Module:
        """The network: images `[B, 3, H, W]` RGB in [0, 1] -> the stride 8 / 16 / 32 head maps."""

        def fn(_net: Any, x: torch.Tensor) -> tuple[torch.Tensor, ...]:  # noqa: ANN401 - the network
            return self.dense(x)

        return Forward(self.net, fn, output_names=("head_s8", "head_s16", "head_s32"))


class Kernels(Portable):
    """Run greedy suppression on the device with a bounded overlap buffer."""

    suppress = staticmethod(box_nms.indices)


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable),
}
