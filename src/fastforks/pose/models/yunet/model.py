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
from fastforks.kernels import box_nms, cudnn_hacc
from fastforks.pose._runner import Forward, Runner
from fastforks.pose.models.retinaface.model import nms
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs

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
    cuda_graphs: bool  # option: kernels replays the network from CUDA graphs for small repeated shapes


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


def decode(maps: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every cell of float32 head maps `[B, HEAD, h, w]` decoded (module docstring).

    Scores `[B, N]`, boxes `[B, N, 4]` and landmarks `[B, N, LANDMARKS, 2]`; elementwise per cell, so selecting
    cells afterwards gives the same values as decoding only the selected cells.
    """
    rows = torch.cat([m.permute(0, 2, 3, 1).reshape(m.shape[0], -1, HEAD) for m in maps], 1)
    score = rows[..., 0].sigmoid() * rows[..., 1].sigmoid()
    pt = points(maps)
    xy, s = pt[:, :2], pt[:, 2:]
    centre, wh = rows[..., 2:4] * s + xy, rows[..., 4:6].exp() * s
    boxes = torch.cat([centre - wh / 2, centre + wh / 2], -1)
    kp = rows[..., 6:].unflatten(-1, (LANDMARKS, 2)) * s[:, None] + xy[:, None]
    return score, boxes, kp


def select(
    cells: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    threshold: float,
    iou: float,
    suppress: Callable[[torch.Tensor, float], torch.Tensor] = nms,
) -> dict[str, torch.Tensor]:
    """The faces of one image from its decoded cells (`decode`, one image): threshold, best first, suppression."""
    score, boxes, kp = cells
    idx = (score >= threshold).nonzero()[:, 0]
    idx = idx[score[idx].argsort(descending=True, stable=True)]
    keep = idx[suppress(boxes[idx], iou)]
    sc = score[keep]
    return {
        "keypoints": kp[keep],
        "keypoint_scores": sc[:, None].expand(-1, LANDMARKS).contiguous(),
        "scores": sc,
        "boxes": boxes[keep],
    }


def faces(
    maps: tuple[torch.Tensor, ...],
    threshold: float,
    iou: float,
    suppress: Callable[[torch.Tensor, float], torch.Tensor] = nms,
) -> dict[str, torch.Tensor]:
    """The faces of one image from its float32 head maps (module docstring)."""
    score, boxes, kp = decode(maps)
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
    # yunet-n 5.3 -> 2.0 (kernels with CUDA graphs 1.2 -> 0.87), yunet-s 5.0 -> 2.5 (1.2 -> 0.85)); 1080p runs one image
    # per call
    batch_pixels = 2_100_000

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

    def decoded(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every cell of `image` `[B, 3, H, W]` decoded (`decode`): scores, boxes, landmarks."""
        return decode(self.dense(image))

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


# Network + every cell's decoding, ~60 kernels: launch bound (RTX 5060 Ti, Windows, fp16, ms per image, eager -> graph:
# 480 x 640 2.5 -> 1.2, 8 x 480 x 640 1.03 -> 0.87, 1080p 3.8 -> 2.9; +124 MB reserved at 480 x 640)
_GRAPH_PIXELS = 2_100_000  # fixed: batch x H x W, a speed threshold (not the architecture)
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Portable):
    """Run greedy suppression on the device with a bounded overlap buffer.

    A repeated input shape up to `_GRAPH_PIXELS` replays the network and the decoding of every cell from a CUDA graph
    (`fastforks.shared.cuda_graphs`); selecting the faces (a data-dependent number) stays eager.
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
        """Every cell of `image` decoded, network and decoding from a graph for a small repeated shape."""
        return self.graphs(image)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self.graphs.release()


class _OneImage(Portable):
    """Portable for the MLX port: one image per call, as before batching (not measured on Apple GPUs)."""

    batch = 1


def _network(ctx: Context) -> YuNet:
    """The network with the checkpoint (heads merged): float32, CPU, BatchNorms unfolded."""
    net = YuNet(ctx.params)
    sd = ctx.state_dict()
    net.load_state_dict(convert(sd.get("state_dict", sd)))  # ty: ignore[invalid-argument-type]
    return net.eval()


class _MlxPortable(_OneImage):
    """Apple MLX: the network written in MLX operations (`mlx_net`), padding, decoding and NMS as portable."""

    module = Runner.module  # the MLX network is not exportable
    CACHE_MB: ClassVar[dict[str, int]] = {}  # the buffer cache this mode asks for, by variant ('*': all)

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts the checkpoint's weights)."""
        Runner.__init__(self, ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        self.mlx = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            lambda net: mlx_net.network(net, ctx.dtype),
            real=lambda: _network(ctx),
            skeleton=lambda: YuNet(ctx.params),
            sources=mlx_net.SOURCES,
        )
        self.net = None
        if mb := self.CACHE_MB.get(ctx.variant.id, self.CACHE_MB.get("*", 0)):  # held while this runner lives
            from fastforks._core.mlx import request_cache  # noqa: PLC0415 - MLX only

            request_cache(self, mb)

    def dense(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """float32 head maps of `image` `[B, 3, H, W]` RGB in [0, 1] (padded to multiples of 32 here)."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        h, w = image.shape[-2:]
        x = F.pad(image.flip(1) * 255, (0, -w % DIVISOR, 0, -h % DIVISOR))
        return tuple(to_torch(m) for m in self.mlx(to_mx(x.float())))


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(_OneImage)}
