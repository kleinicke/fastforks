"""detectron2 Mask / Faster R-CNN (R50-FPN) runners: weights -> network, and the portable / kernels inference passes.

Both follow upstream `DefaultPredictor` + `GeneralizedRCNN.inference` (model-zoo `Base-RCNN-FPN.yaml` test settings):

- input: RGB -> BGR 0..255 (upstream takes uint8 images, so values are rounded), `ResizeShortestEdge` (PIL bilinear,
  reproduced bit-exactly in torch on every device: Pillow's fixed-point two-pass filter and pass order), minus the
  pixel mean, zero-padded to a multiple of `size_divisibility`
- ResNet + FPN (P2-P6, P6 = stride-2 subsampling of P5); RPN: shared head, anchors per level, the top
  `pre_nms_topk` per level decoded, clipped, empty boxes dropped, NMS across levels (per level id), `post_nms_topk`
- box head: ROIAlign (aligned, sampling ratio 0) on P2-P5 by box size, 2 FCs, class softmax; per-class boxes clipped,
  `score_threshold`, per-class NMS, `max_detections`
- mask head (mask variants): ROIAlign 14 x 14 on the detected boxes, the predicted class's 28 x 28 mask, sigmoid
- output: boxes scaled to the input size and clipped, empty ones dropped; masks pasted into the image
  (`grid_sample`) and thresholded at 0.5 (CPU: per box, CUDA: whole image in 1 GB chunks, as upstream)

Portable runs the convolutions and FCs in float16 on CUDA (channels-last; FrozenBatchNorm folded); ROIAlign, box
decoding, NMS and mask pasting stay float32 (box coordinates need float32). Kernels adds cuDNN fused conv + bias + ReLU
(+ residual) for the ResNet, RPN and mask-head convs, fp16-accumulate cuDNN convs (`cudnn_hacc`) for the shortcut and
FPN convs and fp16-accumulate GEMMs for the box head (kernels path).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import NUMPY_ARRAYS
from fastforks._core.half import fp16_accumulate
from fastforks.detection._convs import Convs, FusedConvs
from fastforks.detection._ops import ops
from fastforks.detection._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.pil_resize import pil_bilinear

from .net import Arch, Rcnn, fold_frozen_bn

if TYPE_CHECKING:
    from types import ModuleType

    from fastforks.detection._runner import Context

PIXEL_MEAN_BGR = (103.530, 116.280, 123.675)  # fixed: detectron2 MODEL.PIXEL_MEAN (BGR, 0..255; PIXEL_STD is 1)
RPN_WEIGHTS = (1.0, 1.0, 1.0, 1.0)  # fixed: detectron2 MODEL.RPN.BBOX_REG_WEIGHTS
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)  # fixed: detectron2 MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
SCALE_CLAMP = math.log(1000.0 / 16)  # fixed: detectron2 _DEFAULT_SCALE_CLAMP (largest log-scale of a box delta)
CANONICAL_LEVEL, CANONICAL_SIZE = 4, 224  # fixed: detectron2 ROIPooler (box of side 224 -> level 4)
MASK_THRESHOLD = 0.5  # fixed: detectron2 detector_postprocess
PASTE_LIMIT = 1024**3  # fixed: detectron2 GPU_MEM_LIMIT, bytes of float32 masks pasted at once on a GPU


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    min_size: int  # shortest edge after resizing
    max_size: int  # longest edge cap
    size_divisibility: int  # padding multiple
    anchor_sizes: list[int]  # anchor side per level P2..P6
    aspect_ratios: list[float]  # anchor height / width
    pre_nms_topk: int  # RPN proposals per level before NMS
    post_nms_topk: int  # RPN proposals after NMS
    rpn_nms: float  # RPN NMS IoU
    box_nms: float  # per-class NMS IoU of the detections
    score_threshold: float  # option: minimum class score
    max_detections: int  # option: detections per image


def resized_size(h: int, w: int, short: int, long: int) -> tuple[int, int]:
    """detectron2 `ResizeShortestEdge.get_output_shape`."""
    scale = short / min(h, w)
    nh, nw = (float(short), scale * w) if h < w else (scale * h, float(short))
    if max(nh, nw) > long:
        s = long / max(nh, nw)
        nh, nw = nh * s, nw * s
    return int(nh + 0.5), int(nw + 0.5)


def prepare(image: torch.Tensor, p: Params) -> tuple[torch.Tensor, tuple[int, int]]:
    """`[1, 3, H, W]` RGB in `[0, 1]` -> the padded BGR network input (float32) and the resized `(h, w)`."""
    size = resized_size(*image.shape[-2:], p.min_size, p.max_size)
    x = image.mul(255).round_()
    if size != tuple(image.shape[-2:]):
        x = pil_bilinear(x, size)
    return normalise(x, p), size


def normalise(x: torch.Tensor, p: Params) -> torch.Tensor:
    """RGB `0..255` `[B, 3, h, w]` -> BGR minus the pixel mean, zero-padded to a multiple of `size_divisibility`."""
    x = x.flip(1) - torch.tensor(PIXEL_MEAN_BGR, device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
    d = p.size_divisibility
    return F.pad(x, (0, -x.shape[-1] % d, 0, -x.shape[-2] % d))


def apply_deltas(deltas: torch.Tensor, boxes: torch.Tensor, weights: tuple[float, ...]) -> torch.Tensor:
    """detectron2 `Box2BoxTransform.apply_deltas`: `[R, K * 4]` deltas on `[R, 4]` boxes -> `[R, K * 4]` boxes."""
    deltas = deltas.float()
    boxes = boxes.to(deltas.dtype)
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    ctr_x, ctr_y = boxes[:, 0] + 0.5 * widths, boxes[:, 1] + 0.5 * heights
    wx, wy, ww, wh = weights
    dx, dy = deltas[:, 0::4] / wx, deltas[:, 1::4] / wy
    dw = torch.clamp(deltas[:, 2::4] / ww, max=SCALE_CLAMP)
    dh = torch.clamp(deltas[:, 3::4] / wh, max=SCALE_CLAMP)
    px, py = dx * widths[:, None] + ctr_x[:, None], dy * heights[:, None] + ctr_y[:, None]
    pw, ph = torch.exp(dw) * widths[:, None], torch.exp(dh) * heights[:, None]
    out = torch.stack((px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph), dim=-1)
    return out.reshape(deltas.shape)


def clip(boxes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """detectron2 `Boxes.clip` of `[R, 4]` boxes to `(h, w)`."""
    h, w = size
    x0, y0 = boxes[:, 0].clamp(min=0, max=w), boxes[:, 1].clamp(min=0, max=h)
    x1, y1 = boxes[:, 2].clamp(min=0, max=w), boxes[:, 3].clamp(min=0, max=h)
    return torch.stack((x0, y0, x1, y1), dim=-1)


def nonempty(boxes: torch.Tensor) -> torch.Tensor:
    """detectron2 `Boxes.nonempty` (threshold 0)."""
    return (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)


def cell_anchors(size: int, ratios: list[float]) -> torch.Tensor:
    """detectron2 `DefaultAnchorGenerator.generate_cell_anchors` for one size: `[len(ratios), 4]`."""
    out = []
    for ar in ratios:
        w = math.sqrt(size**2.0 / ar)
        h = ar * w
        out.append([-w / 2.0, -h / 2.0, w / 2.0, h / 2.0])
    return torch.tensor(out)


def grid_anchors(cell: torch.Tensor, hw: tuple[int, int], stride: int) -> torch.Tensor:
    """detectron2 `DefaultAnchorGenerator._grid_anchors` (offset 0) of one level: `[h * w * A, 4]`."""
    sy = torch.arange(0, hw[0] * stride, step=stride, dtype=torch.float32, device=cell.device)
    sx = torch.arange(0, hw[1] * stride, step=stride, dtype=torch.float32, device=cell.device)
    y, x = torch.meshgrid(sy, sx, indexing="ij")
    x, y = x.reshape(-1), y.reshape(-1)
    shifts = torch.stack((x, y, x, y), dim=1)
    return (shifts.view(-1, 1, 4) + cell.view(1, -1, 4)).reshape(-1, 4)


def pool(feats: list[torch.Tensor], boxes: torch.Tensor, side: int, tv: ModuleType) -> torch.Tensor:
    """detectron2 `ROIPooler` (ROIAlignV2) over P2..P5 (float32 maps): `[R, 4]` boxes -> `[R, C, side, side]`."""
    rois = torch.cat([boxes.new_zeros(boxes.shape[0], 1), boxes], 1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    lvl = torch.floor(CANONICAL_LEVEL + torch.log2(torch.sqrt(area) / CANONICAL_SIZE + 1e-8))
    lvl = lvl.clamp(min=2, max=1 + len(feats)).to(torch.int64) - 2
    out = feats[0].new_zeros(boxes.shape[0], feats[0].shape[1], side, side)
    for i, f in enumerate(feats):
        inds = torch.nonzero(lvl == i, as_tuple=True)[0]
        r = rois[inds].to(f.dtype)
        out.index_put_((inds,), tv.roi_align(f, r, (side, side), 1.0 / 2 ** (i + 2), 0, aligned=True))
    return out


def _paste(
    masks: torch.Tensor, boxes: torch.Tensor, h: int, w: int, *, skip_empty: bool
) -> tuple[torch.Tensor, tuple[slice, ...]]:
    """detectron2 `_do_paste_mask`: `[N, 1, M, M]` probabilities into `[N, h', w']` (boxes' region if skip_empty)."""
    dev = masks.device
    if skip_empty:
        x0i, y0i = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(dtype=torch.int32).tolist()
        x1i = int(torch.clamp(boxes[:, 2].max().ceil() + 1, max=w).to(dtype=torch.int32))
        y1i = int(torch.clamp(boxes[:, 3].max().ceil() + 1, max=h).to(dtype=torch.int32))
    else:
        x0i, y0i, x1i, y1i = 0, 0, w, h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)
    n = masks.shape[0]
    iy = torch.arange(y0i, y1i, device=dev, dtype=torch.float32) + 0.5
    ix = torch.arange(x0i, x1i, device=dev, dtype=torch.float32) + 0.5
    iy = (iy - y0) / (y1 - y0) * 2 - 1
    ix = (ix - x0) / (x1 - x0) * 2 - 1
    gx = ix[:, None, :].expand(n, iy.size(1), ix.size(1))
    gy = iy[:, :, None].expand(n, iy.size(1), ix.size(1))
    out = F.grid_sample(masks, torch.stack([gx, gy], dim=3).to(masks.dtype), align_corners=False)
    return out[:, 0], ((slice(y0i, y1i), slice(x0i, x1i)) if skip_empty else ())


def paste_masks(masks: torch.Tensor, boxes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """detectron2 `paste_masks_in_image` (threshold 0.5): `[N, 1, M, M]` -> `[N, h, w]` bool."""
    n, (h, w) = masks.shape[0], size
    cpu = boxes.device.type == "cpu"
    out = torch.zeros(n, h, w, device=boxes.device, dtype=torch.bool)
    if n == 0:
        return out
    chunks = n if cpu else math.ceil(n * h * w * 4 / PASTE_LIMIT)
    for inds in torch.chunk(torch.arange(n, device=boxes.device), chunks):
        m, region = _paste(masks[inds], boxes[inds], h, w, skip_empty=cpu)
        out[(inds, *region)] = m >= MASK_THRESHOLD
    return out


def features(net: Rcnn, x: torch.Tensor, cv: Convs) -> list[torch.Tensor]:
    """ResNet + FPN: the network input -> P2..P6 (detectron2 `FPN.forward`, `sum` fusion, `LastLevelMaxPool`)."""
    bu = net.backbone.bottom_up
    x = F.max_pool2d(cv.conv_relu(bu.stem.conv1, x), kernel_size=3, stride=2, padding=1)
    stages = []
    for stage in bu.stages():
        for b in stage:
            out = cv.conv_relu(b.conv2, cv.conv_relu(b.conv1, x))
            x = cv.conv_add_relu(b.conv3, out, x if b.shortcut is None else cv.conv(b.shortcut, x))
        stages.append(x)
    fpn = net.backbone.fpn()
    prev = cv.conv(fpn[-1][0], stages[-1])
    levels = [cv.conv(fpn[-1][1], prev)]
    for (lat, outc), f in zip(reversed(fpn[:-1]), reversed(stages[:-1]), strict=True):
        prev = cv.conv(lat, f) + F.interpolate(prev, scale_factor=2.0, mode="nearest")
        levels.insert(0, cv.conv(outc, prev))
    levels.append(F.max_pool2d(levels[-1], kernel_size=1, stride=2, padding=0))
    return levels


class _Rcnn(Runner):
    """Inference shared by both backends; `cv` and the float16 parts differ."""

    dtypes = ("float32", "float16")
    Params = Params
    cv: Convs

    def __init__(self, ctx: Context) -> None:
        """Build the network (FrozenBatchNorm folded), convert it to the compute dtype on the device."""
        super().__init__(ctx)
        self.tv = ops()
        p: Params = ctx.params
        net = Rcnn(p)
        net.load_state_dict(fold_frozen_bn(ctx.state_dict(key="model", strip="", extra=NUMPY_ARRAYS)))
        net.eval()
        dt = ctx.torch_dtype
        mf = torch.channels_last if ctx.device.type == "cuda" else torch.contiguous_format
        self.net = net.to(ctx.device, dt).to(memory_format=mf)
        self.predictor = net.roi_heads.box_predictor.float()  # float32 (the fast path)
        self.mf, self.dt = mf, dt
        self.cells = [cell_anchors(s, p.aspect_ratios).to(ctx.device) for s in p.anchor_sizes]

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.dt).contiguous(memory_format=self.mf)

    def _proposals(self, feats: list[torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
        """detectron2 `RPN.forward` + `find_top_rpn_proposals` (inference, one image): `[R, 4]` boxes."""
        p, head = self.ctx.params, self.net.proposal_generator.rpn_head
        boxes, scores, lvls = [], [], []
        for i, f in enumerate(feats):
            t = self.cv.conv_relu(head.conv, f)
            obj = Convs.conv(self.cv, head.objectness_logits, t).float()
            dl = Convs.conv(self.cv, head.anchor_deltas, t).float()
            h, w = obj.shape[-2:]
            obj = obj.permute(0, 2, 3, 1).flatten(1)[0]
            dl = dl.view(1, -1, 4, h, w).permute(0, 3, 4, 1, 2).flatten(1, -2)[0]
            s, idx = obj.topk(min(p.pre_nms_topk, obj.numel()))
            anchors = grid_anchors(self.cells[i], (h, w), 2 ** (i + 2))
            boxes.append(apply_deltas(dl[idx], anchors[idx], RPN_WEIGHTS))
            scores.append(s)
            lvls.append(torch.full((s.numel(),), i, dtype=torch.int64, device=s.device))
        b, s, lv = torch.cat(boxes), torch.cat(scores), torch.cat(lvls)
        valid = torch.isfinite(b).all(dim=1) & torch.isfinite(s)
        if not bool(valid.all()):
            b, s, lv = b[valid], s[valid], lv[valid]
        b = clip(b, size)
        keep = nonempty(b)
        if not bool(keep.all()):
            b, s, lv = b[keep], s[keep], lv[keep]
        keep = self.tv.batched_nms(b.float(), s, lv, p.rpn_nms)[: p.post_nms_topk]
        return b[keep]

    def _box_head(self, x: torch.Tensor) -> torch.Tensor:
        bh = self.net.roi_heads.box_head
        x = F.relu_(F.linear(x.flatten(1).to(self.dt), bh.fc1.weight, bh.fc1.bias))
        return F.relu_(F.linear(x, bh.fc2.weight, bh.fc2.bias)).float()

    def _detections(
        self, levels: list[torch.Tensor], proposals: torch.Tensor, size: tuple[int, int], threshold: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """detectron2 box head + `fast_rcnn_inference_single_image`: boxes, scores, classes (resized image)."""
        p = self.ctx.params
        x = self._box_head(pool(levels, proposals, p.box_pool, self.tv))
        logits, deltas = self.predictor.cls_score(x), self.predictor.bbox_pred(x)
        boxes = apply_deltas(deltas, proposals, BOX_WEIGHTS)
        scores = F.softmax(logits, dim=-1)
        valid = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        if not bool(valid.all()):
            boxes, scores = boxes[valid], scores[valid]
        scores = scores[:, :-1]
        boxes = clip(boxes.reshape(-1, 4), size).view(-1, boxes.shape[1] // 4, 4)
        mask = scores > threshold
        inds = mask.nonzero()
        boxes, scores = boxes[mask], scores[mask]
        keep = self.tv.batched_nms(boxes.float(), scores, inds[:, 1], p.box_nms)[: p.max_detections]
        return boxes[keep], scores[keep], inds[keep, 1]

    def _masks(self, levels: list[torch.Tensor], boxes: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """detectron2 mask head + `mask_rcnn_inference`: `[N, 1, 28, 28]` probabilities of the predicted classes."""
        mh = self.net.roi_heads.mask_head
        if mh is None:
            msg = f"{self.ctx.variant.ref} has no mask head"
            raise RuntimeError(msg)
        if boxes.shape[0] == 0:  # no detections: the fused cuDNN / fp16-accumulate convs reject an empty batch
            side = 2 * self.ctx.params.mask_pool
            return levels[0].new_zeros(0, 1, side, side)
        x = self._input(pool(levels, boxes, self.ctx.params.mask_pool, self.tv))
        for c in mh.convs():
            x = self.cv.conv_relu(c, x)
        x = F.relu_(F.conv_transpose2d(x, mh.deconv.weight, mh.deconv.bias, stride=2))
        logits = Convs.conv(self.cv, mh.predictor, x).float()
        return logits[torch.arange(logits.shape[0], device=logits.device), classes][:, None].sigmoid()

    def __call__(self, image: torch.Tensor, score_threshold: float) -> dict[str, torch.Tensor]:
        """detectron2 `DefaultPredictor` on one image."""
        h, w = image.shape[-2:]
        x, size = prepare(image, self.ctx.params)
        feats = features(self.net, self._input(x), self.cv)
        # float32 NCHW once for both poolers (torchvision ROIAlign would copy channels-last levels per call)
        levels = [f.to(torch.float32, memory_format=torch.contiguous_format) for f in feats[:4]]
        boxes, scores, classes = self._detections(levels, self._proposals(feats, size), size, score_threshold)
        probs = self._masks(levels, boxes, classes) if self.ctx.params.mask else None
        # detector_postprocess: to the input size, clip, drop empty boxes, paste the masks
        boxes = boxes.clone()
        boxes[:, 0::2] *= w / size[1]
        boxes[:, 1::2] *= h / size[0]
        boxes = clip(boxes, (h, w))
        keep = nonempty(boxes)
        out = {"boxes": boxes[keep], "scores": scores[keep], "labels": classes[keep]}
        if probs is not None:
            out["masks"] = paste_masks(probs[keep], out["boxes"], (h, w))
        return out


class Portable(_Rcnn):
    """Plain PyTorch + torchvision operators; float16 convs / FCs on CUDA."""

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.cv = Convs()

    def module(self) -> torch.nn.Module:
        """ResNet + FPN: RGB `[B, 3, h, w]` in `[0, 1]` at the network size (no resize) -> `p2` .. `p6`.

        Normalised and padded inside, as the model call does after its resize.
        """
        p = self.ctx.params
        return Forward(
            self.net,
            lambda net, x: tuple(features(net, self._input(normalise(x.mul(255).round(), p)), self.cv)),
            output_names=("p2", "p3", "p4", "p5", "p6"),
        )


class Kernels(_Rcnn):
    """cuDNN fused conv + ReLU (+ residual), fp16-accumulate convs (`cudnn_hacc`) and GEMMs."""

    def __init__(self, ctx: Context) -> None:
        """Build the network, load `cudnn_hacc`."""
        super().__init__(ctx)
        self.cv = FusedConvs(cudnn_hacc.load(ctx.device))

    def __call__(self, image: torch.Tensor, score_threshold: float) -> dict[str, torch.Tensor]:
        """As portable, with fp16-accumulate cuBLAS GEMMs for the box head."""
        with fp16_accumulate():
            return super().__call__(image, score_threshold)


RUNNERS = {"portable": Portable, "kernels": Kernels}
