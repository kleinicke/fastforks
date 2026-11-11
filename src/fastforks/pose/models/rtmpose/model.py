"""RTMPose runners: person crops -> SimCC logits -> keypoints in the image (mmpose `inference_topdown`).

Per person box (`x0, y0, x1, y1`):

- crop (`fastforks.shared.topdown`, `GetBBoxCenterScale` + `TopdownAffine`): the box centre, its size x 1.25 with
  the shorter side widened to the network's aspect (3 : 4; hand, face and animal models 1 : 1), sampled bilinearly
  the way OpenCV `warpAffine` does and rounded to uint8; RGB normalised with the mmpose means / stds in 0..255
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
from fastforks.shared.topdown import FLIP, MEAN, STD, crop, crop_boxes, fixed

from .net import SPLIT, Arch, RTMPose, Trunk

if TYPE_CHECKING:
    from fastforks.pose._runner import Context


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    flip_test: bool  # average with the mirrored crop (option; the reference's COCO AP protocol)
    conv_accumulation_channels: int = 0  # zero keeps the standard kernel recipe


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
        x = fixed(centre[:, 0] - s * (a.width / 2), s, a.width)
        y = fixed(centre[:, 1] - s * (a.height / 2), s, a.height)
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
