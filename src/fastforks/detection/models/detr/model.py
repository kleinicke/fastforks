"""DETR (ResNet-50 / 101) runner: weights -> network, and the portable inference pass.

Follows the official repo's COCO evaluation (`main.py --eval`, batch size 1, so no padding):

- input: shortest edge 800, longest at most 1333 (`RandomResize([800], max_size=1333)`: PIL bilinear on the uint8
  image, reproduced bit-exactly in torch on every device), ImageNet mean / std
- ResNet (FrozenBatchNorm folded) -> 1x1 projection -> sine position embedding -> 6 encoder and 6 decoder layers
  (post-norm, position added to queries / keys only) over 100 learned queries -> class logits and sigmoid boxes
- `PostProcess`: softmax, best of the 91 class ids per query, boxes centre / size -> corners times the input size.
  Queries whose best class is one of the 11 unused COCO ids ("N/A") are dropped (COCO evaluation ignores them), the
  others are mapped to the 80 contiguous `coco-80` labels, then `score_threshold` and `max_detections` apply.

Portable runs the backbone and transformer in float16 on CUDA (channels-last); the heads, softmax and box decoding
stay float32. Kernels runs the ResNet convs as cuDNN fused conv + bias + ReLU (+ residual) or fp16-accumulate convs
(`cudnn_hacc`) and the transformer GEMMs with fp16 accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.half import fp16_accumulate
from fastforks.detection._convs import Convs, FusedConvs
from fastforks.detection._resize import pil_bilinear
from fastforks.detection._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc

from .net import Arch, Detr, fold_frozen_bn

if TYPE_CHECKING:
    from fastforks.detection._runner import Context

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # fixed: ImageNet statistics of the DETR input transform
UNUSED_IDS = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83)  # fixed: COCO category ids without a class (1..90)


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    min_size: int  # shortest edge after resizing
    max_size: int  # longest edge cap
    score_threshold: float  # option: minimum class score
    max_detections: int  # option: detections per image


def class_map(num_ids: int) -> torch.Tensor:
    """COCO category id (index) -> `coco-80` label, -1 for the unused ids and the padding id 0 (and ids above 90)."""
    out, nxt = torch.full((num_ids,), -1, dtype=torch.int64), 0
    for i in range(1, min(num_ids, 91)):
        if i not in UNUSED_IDS:
            out[i], nxt = nxt, nxt + 1
    return out


def resized_size(h: int, w: int, short: int, long: int) -> tuple[int, int]:
    """DETR `RandomResize.get_size_with_aspect_ratio` (truncating, not rounding)."""
    size = short
    if max(h, w) / min(h, w) * size > long:
        size = round(long * min(h, w) / max(h, w))
    if min(h, w) == size:
        return h, w
    if w < h:
        return int(size * h / w), size
    return size, int(size * w / h)


def prepare(image: torch.Tensor, p: Params) -> torch.Tensor:
    """`[1, 3, H, W]` RGB in `[0, 1]` -> the normalised, resized network input (float32)."""
    size = resized_size(*image.shape[-2:], p.min_size, p.max_size)
    x = image.mul(255).round_()
    if size != tuple(image.shape[-2:]):
        x = pil_bilinear(x, size)
    return normalise(x / 255)


def normalise(x: torch.Tensor) -> torch.Tensor:
    """RGB `[B, 3, h, w]` in `[0, 1]` -> minus the ImageNet mean over its std."""
    mean = torch.tensor(MEAN, device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / torch.tensor(STD, device=x.device, dtype=torch.float32).view(1, 3, 1, 1)


class Portable(Runner):
    """Plain PyTorch; float16 backbone and transformer on CUDA."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network (FrozenBatchNorm folded), convert it to the compute dtype on the device."""
        super().__init__(ctx)
        net = Detr(ctx.params)
        net.load_state_dict(fold_frozen_bn(ctx.state_dict(key="model", strip="")))
        net.eval()
        self.mf = torch.channels_last if ctx.device.type == "cuda" else torch.contiguous_format
        self.dt = ctx.torch_dtype
        self.net = net.to(ctx.device, self.dt).to(memory_format=self.mf)
        self.net.class_embed.float()
        self.net.bbox_embed.float()
        self.ids = class_map(ctx.params.num_classes).to(ctx.device)

    def _logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x.to(self.dt).contiguous(memory_format=self.mf))

    def module(self) -> torch.nn.Module:
        """The network: RGB `[B, 3, h, w]` in `[0, 1]` at the network size (no resize).

        Returns the class logits `[B, 100, 92]` (the last is "no object") and the boxes `[B, 100, 4]` (centre x / y,
        width, height; 0..1).
        """
        return Forward(
            self.net,
            lambda net, x: net(normalise(x.mul(255).round() / 255).to(self.dt)),
            output_names=("logits", "boxes"),
            query_scores=lambda logits: logits.softmax(-1)[..., :-1].amax(-1),
        )

    def __call__(self, image: torch.Tensor, score_threshold: float) -> dict[str, torch.Tensor]:
        """Official `PostProcess` on one image, with the `coco-80` label mapping."""
        p: Params = self.ctx.params
        h, w = image.shape[-2:]
        logits, boxes = self._logits(prepare(image, p))
        scores, ids = logits.float().softmax(-1)[0, :, :-1].max(-1)
        cx, cy, bw, bh = boxes[0].float().unbind(-1)
        boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1) * boxes.new_tensor([w, h, w, h])
        labels = self.ids[ids]
        keep = (labels >= 0) & (scores >= score_threshold)
        scores, order = scores[keep].sort(descending=True)
        order = order[: p.max_detections]
        scores = scores[: p.max_detections]
        return {"boxes": boxes[keep][order], "scores": scores, "labels": labels[keep][order]}


def features(net: Detr, x: torch.Tensor, cv: Convs) -> torch.Tensor:
    """`net.input_proj(net.backbone(x))` with the convs run by `cv`."""
    bb = net.backbone
    x = F.max_pool2d(cv.conv_relu(bb.conv1, x), 3, 2, 1)
    for layer in (bb.layer1, bb.layer2, bb.layer3, bb.layer4):
        for b in layer:
            out = cv.conv_relu(b.conv2, cv.conv_relu(b.conv1, x))
            x = cv.conv_add_relu(b.conv3, out, x if b.downsample is None else cv.conv(b.downsample[0], x))
    return cv.conv(net.input_proj, x)


class Kernels(Portable):
    """cuDNN fused conv + ReLU (+ residual), fp16-accumulate convs (`cudnn_hacc`) and GEMMs."""

    def __init__(self, ctx: Context) -> None:
        """Build the network, load `cudnn_hacc`."""
        super().__init__(ctx)
        self.cv = FusedConvs(cudnn_hacc.load(ctx.device))

    def _logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with fp16_accumulate():
            return self.net.head(features(self.net, x.to(self.dt).contiguous(memory_format=self.mf), self.cv))


RUNNERS = {"portable": Portable, "kernels": Kernels}
