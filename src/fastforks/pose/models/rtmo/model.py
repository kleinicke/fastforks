"""RTMO runners: one image -> the people in it with their keypoints (mmpose `inference_bottomup`), no person boxes.

- letterbox (`BottomupResize`, `resize_mode="fit"`): the image scaled by `r = min(S / W, S / H)` into the centre of
  the square `S x S` network input, grey (114) around it; output pixel `(u, v)` samples
  `((u, v) - S / 2) / r + (W, H) / 2` bilinearly, the way OpenCV `warpAffine` does: source coordinates in 1/32 px
  (columns: offset and step rounded to 1/1024 one by one, rows: together), pixels outside the image count as 114
  and blend in at the border, the result rounded to uint8. The network reads the 0..255 values unnormalised
- network (`net.RTMO`): per grid cell of the stride-16 and stride-32 maps a person logit, a box, keypoint visibility
  logits and a pose vector
- people (`RTMOHead.predict`): cells with a person score above `score_threshold`, best first, greedy NMS on their
  boxes (`nms_iou`); their keypoints from the pose vector by `net.DCC` over bins that span the box x 1.25
- back to the image: `p / r + (W, H) / 2 - S / (2 r)` for keypoints and boxes (boxes clipped to the network input
  first, as the reference)

`scores` is the person score, `keypoint_scores` the visibility probabilities; people come best first.
"""

from __future__ import annotations

import copy
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from fastforks._core.checkpoint import NUMPY_ARRAYS, latin1_encode
from fastforks._core.upload import device_tensor
from fastforks._optimize import optimize
from fastforks.kernels import box_nms, cudnn_hacc, pose_warp
from fastforks.pose._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import DCC, RTMO, STRIDES, Arch, convert

if TYPE_CHECKING:
    from fastforks.pose._runner import Context

PAD = 114.0  # fixed: reference letterbox fill (BottomupResize pad_val)
PADDING = 1.25  # fixed: reference box padding for the DCC bins (bbox_padding)
WARP_BITS = 5  # fixed: OpenCV INTER_BITS (bilinear source coordinates in 1/32 px)
AB_BITS = 10  # fixed: OpenCV AB_BITS (warpAffine fixed point before the 1/32 rounding)
IOU_EPS = 1e-6  # fixed: reference lower bound of the union in the NMS IoU


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    nms_iou: float  # NMS IoU threshold
    score_threshold: float  # option: minimum person score
    cuda_graphs: bool = True


def _to_warp(q: torch.Tensor) -> torch.Tensor:
    """1/1024 px integers as OpenCV `warpAffine` rounds them to 1/32 px."""
    shift = 1 << (AB_BITS - WARP_BITS)
    return torch.floor((q + shift // 2) / shift) / (1 << WARP_BITS)


def _coordinates(ih: int, iw: int, size: int, r: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Rounded source columns and rows of the square OpenCV warp."""
    one = float(1 << AB_BITS)
    i = torch.arange(size, dtype=torch.float64)  # float64 on the CPU: the reference's double arithmetic
    x = _to_warp(round((iw / 2 - size / (2 * r)) * one) + (i / r * one).round())
    y = _to_warp(((ih / 2 - size / (2 * r) + i / r) * one).round())
    return x, y


@functools.lru_cache(maxsize=16)
def _cuda_coordinates(ih: int, iw: int, size: int, device: torch.device) -> torch.Tensor:
    r = min(size / iw, size / ih)
    return torch.stack(_coordinates(ih, iw, size, r)).to(device=device, dtype=torch.float32)


def _cuda_letterbox(image: torch.Tensor, size: int) -> tuple[torch.Tensor, float]:
    ih, iw = image.shape[-2:]
    return pose_warp.letterbox(image, _cuda_coordinates(ih, iw, size, image.device)), min(size / iw, size / ih)


def letterbox(image: torch.Tensor, size: int) -> tuple[torch.Tensor, float]:
    """(`[1, 3, size, size]` in 0..255 (integers), scale `r`) of `image` `[1, 3, H, W]` in [0, 1] (module docstring)."""
    ih, iw = image.shape[-2:]
    r = min(size / iw, size / ih)
    if image.device.type == "cuda":
        x, y = _cuda_coordinates(ih, iw, size, image.device).unbind()
    else:
        x, y = _coordinates(ih, iw, size, r)
    x0, y0 = x.floor(), y.floor()
    fx, fy = (x - x0).float().to(image.device), (y - y0).float().to(image.device)
    # One border pixel of the fill value around the image: every tap outside reads it
    src = torch.nn.functional.pad((image[0].float() * 255).round(), (1, 1, 1, 1), value=PAD)
    xi, xj = ((t.clamp(-1, iw).long() + 1).to(image.device) for t in (x0, x0 + 1))
    yi, yj = ((t.clamp(-1, ih).long() + 1).to(image.device) for t in (y0, y0 + 1))
    # Weights are multiples of 1/32: both blends are exact in float32, as OpenCV's integers
    rows = src[:, yi] * (1 - fy)[:, None] + src[:, yj] * fy[:, None]
    out = rows[:, :, xi] * (1 - fx) + rows[:, :, xj] * fx
    return torch.floor(out + 0.5)[None], r  # OpenCV rounds half up to uint8


def priors(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(cell centres `[N, 2]` x, y in network input pixels, strides `[N]`), the maps row by row, stride 16 first."""
    points, strides = [], []
    for s in STRIDES:
        c = torch.arange(size // s, device=device, dtype=torch.float32) * s + (s - 1) / 2
        points.append(torch.stack([c.expand(len(c), -1), c[:, None].expand(-1, len(c))], -1).reshape(-1, 2))
        strides.append(torch.full((len(c) ** 2,), float(s), device=device))
    return torch.cat(points), torch.cat(strides)


def nms(boxes: torch.Tensor, iou: float) -> torch.Tensor:
    """Indices greedy NMS keeps of `[n, 4]` boxes sorted best first (a box goes if a kept one overlaps it > `iou`)."""
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    wh = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:]) - torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    overlap = wh.clamp(min=0).prod(-1)
    over = (overlap / (area[:, None] + area[None] - overlap).clamp(min=IOU_EPS) > iou).cpu()
    gone = torch.zeros(boxes.shape[0], dtype=torch.bool)
    keep = []
    for i in range(boxes.shape[0]):  # the people of one image: short
        if not gone[i]:
            keep.append(i)
            gone |= over[i]
    return device_tensor(keep, dtype=torch.long, device=boxes.device)


class Portable(Runner):
    """Plain PyTorch: letterbox on the device, float16 network on CUDA, float32 decoding."""

    dtypes = ("float32", "float16")
    Params = Params
    needs_boxes: ClassVar[bool] = False
    compiled_methods = ("__call__", "estimate_batch")
    # images per network call (`estimate_batch`): launch bound (RTX 5060 Ti, Windows, fp16, 8 frames 480 x 640, ms per
    # frame: rtmo-t 12.7 -> 4.8, rtmo-s 12.8 -> 5.6, rtmo-m 14.4 -> 7.2, rtmo-l 18.3 -> 9.5); the network input is
    # the fixed letterbox size, so memory does not grow with the image
    batch = 8
    resize = staticmethod(letterbox)
    suppress = staticmethod(nms)

    def __init__(self, ctx: Context) -> None:
        """Build the network with the checkpoint (BatchNorm folded) on the device."""
        super().__init__(ctx)
        a: Params = ctx.params
        net_sd, dcc_sd = convert(ctx.state_dict(key="state_dict", strip="", extra=extra()))
        net = RTMO(a)
        net.load_state_dict(net_sd)
        self.net = net.eval().to(ctx.device, ctx.torch_dtype)
        if ctx.mode == "portable" and ctx.device.type == "cuda" and ctx.dtype == "float16":
            for conv in self.net.modules():
                if isinstance(conv, torch.nn.Conv2d):
                    conv.to(memory_format=torch.channels_last)
        dcc = DCC(a)
        dcc.load_state_dict(dcc_sd)
        self.dcc = dcc.eval().to(ctx.device)
        self.prior, self.stride = priors(a.size, ctx.device)

    def dense(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """float32 per-cell outputs (`net.Head.forward`) of the letterboxed image `[B, 3, S, S]` in 0..255."""
        return tuple(t.float() for t in self.net(x.to(self.ctx.torch_dtype)))

    def people(self, dense: tuple[torch.Tensor, ...], r: float, height: int, width: int) -> dict[str, torch.Tensor]:
        """The people of one image from its per-cell outputs (module docstring)."""
        a: Params = self.ctx.params
        cls, box, vis, vec = (t[0] for t in dense)
        score = cls.sigmoid()
        idx = (score > a.score_threshold).nonzero()[:, 0]
        idx = idx[score[idx].argsort(descending=True, stable=True)]
        stride = self.stride[idx, None]
        xy = box[idx, :2] * stride + self.prior[idx]
        wh = box[idx, 2:].exp() * stride
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)
        keep = self.suppress(boxes, a.nms_iou)
        idx, boxes = idx[keep], boxes[keep]
        centre = (boxes[:, :2] + boxes[:, 2:]) / 2
        kp = self.keypoints(vec[idx], centre, (boxes[:, 2:] - boxes[:, :2]) * PADDING, self.prior[idx])
        offset = device_tensor([width, height], device=kp.device) / 2 - a.size / (2 * r)
        return {
            "keypoints": kp / r + offset,
            "keypoint_scores": vis[idx].sigmoid(),
            "scores": score[idx],
            "boxes": boxes.clamp(0, a.size) / r + offset.repeat(2),
        }

    def keypoints(
        self, vec: torch.Tensor, centre: torch.Tensor, scale: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        """Keypoints `[P, K, 2]` of the kept people (`DCC.forward`)."""
        return self.dcc(vec, centre, scale, grid)

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """The people in the image with their keypoints."""
        assert boxes is None  # noqa: S101 - the task API rejects boxes for one-stage families
        x, r = self.resize(image, self.ctx.params.size)
        return self.people(self.dense(x), r, *image.shape[-2:])

    def estimate_batch(self, images: torch.Tensor, boxes: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """The people of each image of `[B, 3, H, W]`: letterboxed one by one, one network call, decoded per image."""
        assert all(b is None for b in boxes)  # noqa: S101 - one-stage: the task API passes no boxes
        size = self.ctx.params.size
        boxed = [self.resize(images[i : i + 1], size) for i in range(images.shape[0])]
        dense = self.dense(torch.cat([x for x, _ in boxed]))
        h, w = images.shape[-2:]
        return [self.people(tuple(t[i : i + 1] for t in dense), r, h, w) for i, (_, r) in enumerate(boxed)]

    def module(self) -> torch.nn.Module:
        """The network: images `[B, 3, S, S]` RGB in [0, 1] at the network size -> `net.Head.forward` per cell."""

        def fn(_net: Any, x: torch.Tensor) -> tuple[torch.Tensor, ...]:  # noqa: ANN401 - the network
            return self.dense(x * 255)

        return Forward(self.net, fn, output_names=("class_logits", "box_offsets", "visibility_logits", "pose_vectors"))


# people per DCC call in kernels mode: every call pads its people to this many and replays one CUDA graph (the
# decoding is ~60 tiny kernels; eager they cost rtmo-s ~0.7 ms of host launches per image on Windows); more run eagerly
DCC_ROWS = 64


class _GraphedDCC:
    """`DCC` on the kept people padded to `DCC_ROWS` rows (copies of the first), from one CUDA graph.

    The DCC is per person (its attention runs over one person's keypoints), so the padding rows change no real row.
    """

    def __init__(self, dcc: DCC, *, enabled: bool, capture_after: int) -> None:
        self.dcc = dcc
        self.graphs = SmallShapeGraphs(
            dcc, pixels=DCC_ROWS, graphs=1, enabled=enabled, capture_after=capture_after, size=lambda v, *_: v.shape[0]
        )

    def __call__(
        self, vec: torch.Tensor, centre: torch.Tensor, scale: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        """Keypoints `[P, K, 2]` (`DCC.forward`)."""
        p = vec.shape[0]
        if p == 0 or p > DCC_ROWS:
            return self.dcc(vec, centre, scale, grid)
        padded = [torch.cat([t, t[:1].expand(DCC_ROWS - p, *t.shape[1:])]) for t in (vec, centre, scale, grid)]
        return self.graphs(*padded)[:p]


class Kernels(Portable):
    """CUDA: the CSPDarknet backbone through `fastforks._optimize` (fused conv kernels, fp16, channels-last).

    The neck and the head stay the portable float16 modules; decoding as in `Portable`.
    """

    batch = 1  # the network graph is captured for one image
    resize = staticmethod(_cuda_letterbox)

    @staticmethod
    def suppress(boxes: torch.Tensor, iou: float) -> torch.Tensor:
        """Use the training implementation's lower bound on the union area."""
        return box_nms.indices(boxes, iou, IOU_EPS)

    def __init__(self, ctx: Context) -> None:
        """The portable network, then the optimized backbone."""
        super().__init__(ctx)
        net = self.net
        self.backbone = optimize(copy.deepcopy(net.backbone).float(), mode="kernels", device=ctx.device, family="cnn")
        self.neck, self.head = net.neck.half(), net.head.half()
        del net.backbone  # the optimized copy replaces it
        self.graphs = SmallShapeGraphs(
            self._dense,
            pixels=ctx.params.size**2,
            graphs=1,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )
        self.dcc_graph = _GraphedDCC(self.dcc, enabled=ctx.params.cuda_graphs, capture_after=cudnn_hacc.capture_after())

    def release_graphs(self) -> None:
        """Release the fixed-size network graph, the DCC graph and their workspaces."""
        self.graphs.release()
        self.dcc_graph.graphs.release()

    def keypoints(
        self, vec: torch.Tensor, centre: torch.Tensor, scale: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        """Keypoints `[P, K, 2]` of the kept people, from the DCC graph."""
        return self.dcc_graph(vec, centre, scale, grid)

    def dense(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Replay the fixed-size network while keeping the variable-length person decoding eager."""
        return self.graphs(x)

    def _dense(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """float32 per-cell outputs of the letterboxed image `[B, 3, S, S]` in 0..255."""
        c3, c4, c5 = self.backbone.forward_native(x)
        return tuple(t.float() for t in self.head(self.neck((c3, c4, c5))))


def extra() -> dict[tuple[str, str], object]:
    """What the mmpose checkpoints pickle besides tensors: the meta's numpy arrays and their latin-1 byte strings."""
    return {**NUMPY_ARRAYS, ("_codecs", "encode"): latin1_encode}


class _OneImage(Portable):
    """Portable for the MLX port: one image per call, as before batching (not measured on Apple GPUs)."""

    batch = 1


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_OneImage),
}
