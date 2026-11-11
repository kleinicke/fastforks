"""RTMPose runners: person crops -> SimCC logits -> keypoints in the image (mmpose `inference_topdown`).

Per person box (`x0, y0, x1, y1`):

- crop (`GetBBoxCenterScale` + `TopdownAffine`): the box centre, its size x 1.25 with the shorter side widened to
  the network's aspect (3 : 4; hand, face and animal models 1 : 1); output pixel `(u, v)` samples
  `centre + ((u, v) - (w, h) / 2) * size / (w, h)` bilinearly, the way OpenCV `warpAffine` does: source coordinates
  in 1/32 px, pixels outside the image count as 0 and blend in at the border, the crop rounded to uint8.
  Not bit-exact with OpenCV 4.10 (900 random boxes, measured):
  rows and columns whose coordinate sits at a 1/32 rounding tie land one step off (OpenCV rounds a row's offset and
  step together, `_fixed` one by one), 0.4 % of the pixels of a noise image (up to 11 levels) and 0.02 % of a
  smooth one (up to 5); `crop` itself is the exact bilinear blend at its coordinates. RGB normalised with the
  mmpose means / stds in 0..255
- decode (`SimCCLabel.decode`, no softmax): argmax of each x / y logit vector, divided by the 2 bins per pixel;
  keypoint score = the smaller of the two maxima; mapped back with `p / (w, h) * size + centre - size / 2`
- `flip_test` (option, off by default as in rtmlib; mmpose's COCO AP protocol turns it on): the logits of the
  mirrored crop, left / right keypoints swapped and the x logits reversed, averaged with the plain ones
- `scores`: the mean keypoint score (the reference gives none)

The reference marks keypoints whose score is <= 0 with -1 before mapping them back; here they keep their argmax: the
score says they were not found.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import NUMPY_ARRAYS, empty_bytes, latin1_encode
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import override
from fastforks._optimize import conv_ops, ops, optimize
from fastforks.pose._runner import BUCKETS, Forward, Runner, in_buckets
from fastforks.shared.cspnext import Attention, fold_bn
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import SPLIT, Arch, RTMPose, Trunk

if TYPE_CHECKING:
    from fastforks.pose._runner import Context

PADDING = 1.25  # fixed: reference crop padding (GetBBoxCenterScale)
MEAN = (123.675, 116.28, 103.53)  # fixed: reference data_preprocessor mean (RGB, 0..255)
STD = (58.395, 57.12, 57.375)  # fixed: reference data_preprocessor std (RGB, 0..255)
WARP_BITS = 5  # fixed: OpenCV INTER_BITS (bilinear source coordinates in 1/32 px)
AB_BITS = 10  # fixed: OpenCV AB_BITS (warpAffine fixed point before the 1/32 rounding)
# Left / right partner of each keypoint by label set (`flip_indices` of the reference dataset metas)
# fmt: off
FLIP = {  # fixed: the label sets' keypoint order
    "coco-keypoints-17": (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15),
    "coco-wholebody-133": (  # body, feet, face, left hand, right hand
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 20, 21, 22, 17, 18, 19, 39, 38, 37, 36, 35, 34,
        33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 50, 51, 52, 53, 58, 57,
        56, 55, 54, 68, 67, 66, 65, 70, 69, 62, 61, 60, 59, 64, 63, 77, 76, 75, 74, 73, 72, 71, 82, 81, 80, 79, 78,
        87, 86, 85, 84, 83, 90, 89, 88,
        *range(112, 133), *range(91, 112),
    ),
    "halpe-26": (
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18, 19, 21, 20, 23, 22, 25, 24,
    ),
    "coco-wholebody-hand-21": tuple(range(21)),  # one hand: the mirrored crop keeps every keypoint
    "lapa-106": (
        *range(32, -1, -1), 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52, 53, 54,
        *range(65, 54, -1), 79, 78, 77, 76, 75, 82, 81, 80, 83, 70, 69, 68, 67, 66, 73, 72, 71, 74,
        *range(90, 83, -1), *range(95, 90, -1), *range(100, 95, -1), 103, 102, 101, 105, 104,
    ),
    "ap10k-17": (1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13),
}
# fmt: on


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    flip_test: bool  # average with the mirrored crop (option; the reference's COCO AP protocol)
    conv_accumulation_channels: int = 0  # zero keeps the standard kernel recipe


def crop_boxes(boxes: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(centres `[P, 2]`, crop sizes `[P, 2]` w, h in pixels) of `x0, y0, x1, y1` boxes for a `height x width` net."""
    wh = (boxes[:, 2:] - boxes[:, :2]) * PADDING
    centre = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    aspect = width / height
    w, h = wh.unbind(1)
    wide = w > h * aspect
    return centre, torch.stack([torch.where(wide, w, h * aspect), torch.where(wide, w / aspect, h)], 1)


def _fixed(offset: torch.Tensor, step: torch.Tensor, n: int) -> torch.Tensor:
    """Source coordinates `offset + step * i` (i < n) as OpenCV `warpAffine` rounds a column: 1/1024, then 1/32."""
    i = torch.arange(n, device=offset.device, dtype=torch.float32)  # float32: exact integers here (< 2^24)
    one = float(1 << AB_BITS)
    q = (offset[:, None] * one).round() + (step[:, None] * i * one).round()
    shift = 1 << (AB_BITS - WARP_BITS)
    return torch.floor((q + shift // 2) / shift) / (1 << WARP_BITS)


def crop(image: torch.Tensor, centre: torch.Tensor, size: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Crops `[P, 3, height, width]` in 0..255 (integers) of `image` `[1, 3, H, W]` in [0, 1] (module docstring)."""
    ih, iw = image.shape[-2:]
    s = size[:, 0] / width  # source pixels per crop pixel (the same along y: the aspect is the network's)
    x = _fixed(centre[:, 0] - s * (width / 2), s, width)  # [P, w]
    y = _fixed(centre[:, 1] - s * (height / 2), s, height)  # [P, h]
    x0, y0 = x.floor(), y.floor()
    fx, fy = x - x0, y - y0  # multiples of 1/32: the tap weights below are exact in float32, as OpenCV's integers
    pixels = (image[0].float() * 255).round().flatten(1)  # [3, H * W] uint8 levels
    out = torch.zeros(3, centre.shape[0], height, width, device=image.device)
    for dy, ty in ((0, 1 - fy), (1, fy)):
        yy = y0 + dy
        wy = (ty * ((yy >= 0) & (yy < ih)))[:, :, None]  # pixels outside the image count as 0
        row = yy.clamp(0, ih - 1).long()[:, :, None] * iw  # int64: a float32 index is inexact above 2^24 pixels
        for dx, tx in ((0, 1 - fx), (1, fx)):
            xx = x0 + dx
            wx = (tx * ((xx >= 0) & (xx < iw)))[:, None, :]
            idx = row + xx.clamp(0, iw - 1).long()[:, None, :]  # [P, h, w]
            out += pixels[:, idx] * (wy * wx)
    return torch.floor(out.transpose(0, 1) + 0.5)  # OpenCV rounds half up to uint8


def decode(sx: torch.Tensor, sy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(crop coordinates `[P, K, 2]` x, y in network input pixels, scores `[P, K]`) of float32 SimCC logits."""
    vx, ix = sx.max(-1)
    vy, iy = sy.max(-1)
    return torch.stack([ix, iy], -1).float() / SPLIT, torch.minimum(vx, vy)


class Portable(Runner):
    """Plain PyTorch: crops on the device, float16 network on CUDA, float32 decode."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network with the checkpoint (BatchNorm folded) on the device."""
        super().__init__(ctx)
        a: Params = ctx.params
        net = RTMPose(a)
        net.load_state_dict(fold_bn(ctx.state_dict(key="state_dict", strip="", extra=extra())))
        self.net = net.eval().to(ctx.device, ctx.torch_dtype)
        self.mean = torch.tensor(MEAN, device=ctx.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=ctx.device).view(1, 3, 1, 1)
        self.flip = torch.tensor(FLIP[ctx.variant.output_info["label_set"]], device=ctx.device)
        self.net_size = torch.tensor([a.width, a.height], device=ctx.device, dtype=torch.float32)
        if ctx.device.type == "mps":  # MPS reduces over two axes ~30x slower than over one (same result)
            for m in self.net.modules():
                if isinstance(m, Attention):
                    override(m, _attention_mps)

    def logits(self, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """float32 SimCC logits (x `[P, K, 2 w]`, y `[P, K, 2 h]`) of crops `[P, 3, h, w]` in 0..255."""
        x = ((crops - self.mean) / self.std).to(self.ctx.torch_dtype)
        if not self.ctx.params.flip_test:
            sx, sy = self.net(x)
            return sx.float(), sy.float()
        p = x.shape[0]
        sx, sy = (t.float() for t in self.net(torch.cat([x, x.flip(-1)])))
        fx = sx[p:].index_select(1, self.flip).flip(-1)
        fy = sy[p:].index_select(1, self.flip)
        return (sx[:p] + fx) * 0.5, (sy[:p] + fy) * 0.5

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Keypoints of the people in `boxes`."""
        assert boxes is not None  # noqa: S101 - the task API checks top-down calls
        a: Params = self.ctx.params
        centre, size = crop_boxes(boxes.float(), a.height, a.width)
        coords, score = decode(*self.logits(crop(image, centre, size, a.height, a.width)))
        kp = coords * (size / self.net_size)[:, None] + (centre - size * 0.5)[:, None]
        return {"keypoints": kp, "keypoint_scores": score, "scores": score.mean(1)}

    def module(self) -> torch.nn.Module:
        """The network: crops `[B, 3, h, w]` RGB in [0, 1] at the network size -> SimCC logits (x, y)."""

        def fn(_net: Any, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401 - the network
            return self.logits(crops * 255)

        return Forward(self.net, fn)


class Kernels(Portable):
    """CUDA: an optimized backbone, an fp16 head and CUDA graphs.

    The CSPNeXt backbone (RTMW: with its neck) runs through `fastforks._optimize` (fused conv + SiLU / residual /
    concat kernels, fp16, channels-last), the SimCC head in fp16 with fp16-accumulating GEMMs, and the crops ->
    logits -> argmax path as one CUDA graph per person bucket (`BUCKETS`, zero crops pad a batch up). Cropping and the
    mapping back run eagerly.
    """

    def __init__(self, ctx: Context) -> None:
        """The portable network, then the optimized backbone and the fp16 head."""
        super().__init__(ctx)
        net = self.net.float()
        trunk = Trunk(net.backbone, net.neck)
        self.backbone = optimize(copy.deepcopy(trunk), mode="kernels", device=ctx.device, family="cnn")
        if ctx.params.conv_accumulation_channels:
            _precise_backbone(self.backbone, ctx.params.conv_accumulation_channels)
        self.head = net.head.half()
        del net.backbone  # the optimized copy replaces it
        net.neck = None
        a: Params = ctx.params
        self.graphs = SmallShapeGraphs(self._decoded, pixels=BUCKETS[-1] * a.height * a.width)

    def _decoded(self, crops: torch.Tensor) -> torch.Tensor:
        """`[P, K, 3]` float32 (x, y in network input pixels, score) of crops `[P, 3, h, w]` in 0..255."""
        x = (crops - self.mean) / self.std
        if self.ctx.params.flip_test:
            x = torch.cat([x, x.flip(-1)])
        with fp16_accumulate():
            sx, sy = (t.float() for t in self.head(_half(self.backbone(x))))
        if self.ctx.params.flip_test:
            p = crops.shape[0]
            sx = (sx[:p] + sx[p:].index_select(1, self.flip).flip(-1)) * 0.5
            sy = (sy[:p] + sy[p:].index_select(1, self.flip)) * 0.5
        coords, score = decode(sx, sy)
        return torch.cat([coords, score[..., None]], -1)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self.graphs.release()

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Keypoints of the people in `boxes`."""
        assert boxes is not None  # noqa: S101 - the task API checks top-down calls
        a: Params = self.ctx.params
        centre, size = crop_boxes(boxes.float(), a.height, a.width)
        crops = crop(image, centre, size, a.height, a.width)
        out = in_buckets(self.graphs, crops)
        kp = out[..., :2] * (size / self.net_size)[:, None] + (centre - size * 0.5)[:, None]
        score = out[..., 2]
        return {"keypoints": kp, "keypoint_scores": score, "scores": score.mean(1)}

    def module(self) -> torch.nn.Module:
        """The optimized network: crops `[B, 3, h, w]` RGB in [0, 1] at the network size -> SimCC logits (x, y)."""

        def fn(_net: Any, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401 - the network
            x = (crops * 255 - self.mean) / self.std
            with fp16_accumulate():
                sx, sy = self.head(_half(self.backbone(x)))
            return sx.float(), sy.float()

        return Forward(self.net, fn)


class MetalKernels(Portable):
    """Apple MPS: the portable runner with the crops as one Metal kernel (`pose_metal.warp_crop`, the pixels of `crop`).

    The crop's gathers and blends are ~20 small ops per call on MPS; the network stays the portable one (MPS
    convolutions are compute bound; fp16 is not faster there).
    """

    module = Runner.module  # the Metal kernel is not exportable

    def __init__(self, ctx: Context) -> None:
        """The portable network; the Metal library."""
        super().__init__(ctx)
        from fastforks.kernels import pose_metal  # noqa: PLC0415 - needs the kernel library

        self.lib = pose_metal.load(ctx.device)

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Keypoints of the people in `boxes`."""
        from fastforks.kernels import pose_metal  # noqa: PLC0415 - loaded in __init__

        assert boxes is not None  # noqa: S101 - the task API checks top-down calls
        a: Params = self.ctx.params
        centre, size = crop_boxes(boxes.float(), a.height, a.width)
        s = size[:, 0] / a.width  # as `crop`
        x = _fixed(centre[:, 0] - s * (a.width / 2), s, a.width)
        y = _fixed(centre[:, 1] - s * (a.height / 2), s, a.height)
        coords, score = decode(*self.logits(pose_metal.warp_crop(self.lib, image, x, y)))
        kp = coords * (size / self.net_size)[:, None] + (centre - size * 0.5)[:, None]
        return {"keypoints": kp, "keypoint_scores": score, "scores": score.mean(1)}


def _precise_conv(  # noqa: PLR0913, PLR0917 - fused kernel signature
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    residual: torch.Tensor | None,
    padding: tuple[int, int] | list[int],
    stride: tuple[int, int] | list[int],
    act: int,
    cfg: int = -1,
    *,
    channels: int,
) -> torch.Tensor:
    """Wide half-input convolutions with cuDNN accumulation and float32 residual/activation."""
    if w.shape[1] < channels:
        return conv_ops.conv_fused(x, w, b, residual, padding, stride, act, cfg)
    y = F.conv2d(x, w, b, stride, padding).float()
    if residual is not None:
        y = y + residual.float()
    return conv_ops._ACTS[act](y).half().contiguous(memory_format=torch.channels_last)  # noqa: SLF001 - shared fused-op activation codes


def _precise_backbone(backbone: torch.nn.Module, channels: int) -> None:
    """Apply the measured RTMW-X384 recipe before CUDA graph capture.

    Half accumulation moved confident wrist/elbow peaks by >100 px. Wide cuDNN convolutions and the
    float-accumulating depthwise kernel reduce golden p99 to 2.92 px without changing weights or preprocessing.
    """
    for module in backbone.modules():
        if isinstance(module, torch.fx.GraphModule):
            for node in module.graph.nodes:
                if node.op == "call_function" and node.target is ops.conv_fused:
                    node.target = _precise_conv
                    node.kwargs = {**node.kwargs, "channels": channels}
                if node.op == "call_function" and node.target is ops.dwconv2:
                    node.target = ops.dwconv
            module.recompile()


def _half(y: torch.Tensor | tuple[torch.Tensor, ...]) -> Any:  # noqa: ANN401 - what the head takes
    """The optimized backbone's map (RTMW: the neck's two maps) as float16."""
    return y.half() if isinstance(y, torch.Tensor) else tuple(t.half() for t in y)


def _attention_mps(self: Attention, x: torch.Tensor) -> torch.Tensor:
    """`Attention.forward` with the spatial mean over one flattened axis (MPS: `mean((2, 3))` is ~30x slower)."""
    return x * F.hardsigmoid(self.fc(x.flatten(2).mean(-1)[..., None, None]))


def extra() -> dict[tuple[str, str], object]:
    """What the mmpose checkpoints pickle besides tensors: the meta's numpy arrays and their byte strings."""
    return {**NUMPY_ARRAYS, ("_codecs", "encode"): latin1_encode, ("__builtin__", "bytes"): empty_bytes}


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
