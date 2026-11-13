"""LW-DETR runners: transformers' `LwDetrForObjectDetection` network, our forward and the authors' PostProcess.

As the Hugging Face `DeformableDetrImageProcessor` of the weights (transformers 5.19, torchvision backend) and the
authors' `SquareResize`: the image is resized to a `size` x `size` square (aspect ratio ignored; bilinear with
antialiasing on uint8 values: torch's integer kernel on the CPU, elsewhere in float and rounded, as torchvision does),
ImageNet mean / std. The network is the ViT backbone (interleaved window / global attention) -> the C2f projector (one
or two levels) -> the two-stage query selection (the encoder's `num_queries` best proposals) -> 3 deformable decoder
layers; one image, so no padding and every pixel valid (the forward below is `LwDetrModel.forward` without the masks,
so it runs as one CUDA graph). The authors' `PostProcess`: the best `num_select` of the `Q x 91` sigmoid scores; the
COCO category ids are mapped to `coco-80` and the unused ids dropped, then `score_threshold` and `max_detections`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.detection._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.seginfer import pretrained

if TYPE_CHECKING:
    from fastforks.detection._runner import Context

MEAN = (0.485, 0.456, 0.406)  # fixed: ImageNet normalisation (the image processor's)
STD = (0.229, 0.224, 0.225)  # fixed: ImageNet normalisation
UNUSED_IDS = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83)  # fixed: COCO category ids without a class (1..90)


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` + `[options]`."""

    arch: str  # transformers class
    size: int  # square network input
    num_select: int  # PostProcess: the best (query, class) pairs kept before the threshold
    score_threshold: float
    max_detections: int
    cuda_graphs: bool  # option: kernels replays the network from a CUDA graph


def class_map(num_ids: int) -> torch.Tensor:
    """COCO category id (index) -> `coco-80` label; -1 for id 0 and the unused ids."""
    out, nxt = torch.full((num_ids,), -1, dtype=torch.int64), 0
    for i in range(1, min(num_ids, 91)):
        if i not in UNUSED_IDS:
            out[i], nxt = nxt, nxt + 1
    return out


def square(image: torch.Tensor, size: int) -> torch.Tensor:
    """`[B, 3, H, W]` in [0, 1] -> `[B, 3, size, size]` uint8 values as float32 (the processor's resize)."""
    x = (image.float() * 255).round().clamp(0, 255)
    if tuple(x.shape[-2:]) == (size, size):
        return x
    if x.device.type == "cpu":  # torchvision's uint8 path: torch's integer antialiasing kernel (CPU only)
        return F.interpolate(x.to(torch.uint8), size=(size, size), mode="bilinear", antialias=True).float()
    return F.interpolate(x, size=(size, size), mode="bilinear", antialias=True).round().clamp(0, 255)


def fold_batch_norms(net: torch.nn.Module) -> int:
    """Fold every conv + BatchNorm pair (transformers' `LwDetrConvNormLayer`) into the conv (float32); their count.

    Needed for float16, not only faster: some projector BatchNorms of the medium / large / xlarge weights hold running
    variances above the float16 range (up to 1.2e7; xlarge even one `inf`), which a float16 BatchNorm turns into `inf`
    and so zeroes those channels. Folded, the conv weights carry `gamma / sqrt(var + eps)` and stay in range.
    """
    n = 0
    for m in net.modules():
        conv, bn = getattr(m, "conv", None), getattr(m, "norm", None)
        if is_base(conv, torch.nn.Conv2d) and is_base(bn, torch.nn.BatchNorm2d):
            m.conv, m.norm = fuse_conv_bn_eval(conv.float(), bn.float()), torch.nn.Identity()
            n += 1
    return n


def _refine(ref: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """The transformers `refine_bboxes`: centre shifted by `delta * size`, size scaled by `exp(delta)`."""
    return torch.cat((delta[..., :2] * ref[..., 2:] + ref[..., :2], delta[..., 2:].exp() * ref[..., 2:]), -1)


class Network(torch.nn.Module):
    """`LwDetrForObjectDetection.forward` (inference, one query group) with every pixel valid.

    Normalised `[B, 3, H, W]` -> class logits `[B, Q, 91]` (index = COCO category id) and boxes `[B, Q, 4]` (centre
    x / y, width, height; 0..1), both float32. The level shapes and start indices are kept per input shape, so a
    captured forward copies nothing from the host.
    """

    def __init__(self, net: Any) -> None:  # noqa: ANN401 - transformers model
        """Wrap the loaded model."""
        super().__init__()
        self.net = net
        self._shapes: dict[tuple[tuple[int, int], ...], tuple[torch.Tensor, torch.Tensor]] = {}

    def _levels(self, shapes: tuple[tuple[int, int], ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        hit = self._shapes.get(shapes)
        if hit is None:
            t = torch.tensor(shapes, dtype=torch.long, device=device)
            hit = (t, torch.cat((t.new_zeros(1), t.prod(1).cumsum(0)[:-1])))
            self._shapes[shapes] = hit
        return hit

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits and boxes of the decoder's last layer."""
        m = self.net.model
        feats = m.backbone.projector(m.backbone.backbone(x).feature_maps)
        shapes = tuple((int(f.shape[-2]), int(f.shape[-1])) for f in feats)
        spatial, start = self._levels(shapes, x.device)
        src = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], 1)
        b, n = src.shape[:2]
        query, proposals, invalid = m.gen_encoder_output_proposals(
            src, torch.zeros(b, n, dtype=torch.bool, device=x.device), list(shapes)
        )
        query = m.enc_output_norm[0](m.enc_output[0](query))
        scores = m.enc_out_class_embed[0](query).masked_fill(invalid, float("-inf")).amax(-1)
        coords = _refine(proposals, m.enc_out_bbox_embed[0](query))
        top = scores.topk(m.num_queries, dim=1)[1]
        coords = coords.gather(1, top[..., None].expand(-1, -1, 4))
        ref = _refine(coords, m.reference_point_embed.weight[: m.num_queries][None].expand(b, -1, -1))
        target = m.query_feat.weight[: m.num_queries][None].expand(b, -1, -1)
        dec = m.decoder(
            inputs_embeds=target,
            reference_points=ref,
            spatial_shapes=spatial,
            spatial_shapes_list=list(shapes),
            level_start_index=start,
            valid_ratios=torch.ones(b, len(shapes), 2, dtype=src.dtype, device=x.device),
            encoder_hidden_states=src,
            encoder_attention_mask=None,
        )
        h = dec.last_hidden_state
        boxes = _refine(dec.intermediate_reference_points[-1], self.net.bbox_embed(h))
        return self.net.class_embed(h).float(), boxes.float()


class Portable(Runner):
    """transformers' network in the compute dtype; PostProcess in float32."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network from the variant's config and weights."""
        super().__init__(ctx)
        net: Any = pretrained(ctx, config=0, weights=1)
        if not fold_batch_norms(net.eval()):
            msg = "LW-DETR: no conv + BatchNorm pair was folded (the transformers module tree changed)"
            raise RuntimeError(msg)
        self.dt = ctx.torch_dtype
        self.net = Network(net.to(ctx.device, self.dt))
        self.ids = class_map(net.config.num_labels).to(ctx.device)
        self.mean = torch.tensor(MEAN, device=ctx.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=ctx.device).view(1, 3, 1, 1)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 values (float) -> the network input in the compute dtype."""
        return ((x / 255 - self.mean) / self.std).to(self.dt)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x)

    def __call__(self, image: torch.Tensor, score_threshold: float) -> dict[str, torch.Tensor]:
        """Every detection above the threshold (best first, at most `max_detections`), in input pixels."""
        p: Any = self.ctx.params
        h, w = image.shape[-2:]
        logits, boxes = self._forward(self.normalise(square(image, p.size)))
        scores, idx = logits[0].sigmoid().flatten().topk(p.num_select)
        classes = logits.shape[-1]
        labels, query = self.ids[idx % classes], idx // classes
        cx, cy, bw, bh = boxes[0][query].unbind(-1)
        out = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1) * boxes.new_tensor([w, h, w, h])
        keep = ((labels >= 0) & (scores >= score_threshold)).nonzero()[:, 0][: p.max_detections]
        return {"boxes": out[keep], "scores": scores[keep], "labels": labels[keep]}

    def module(self) -> torch.nn.Module:
        """The network: RGB `[B, 3, H, W]` in `[0, 1]` (no resize; the model call takes `size` x `size`, default 640).

        Returns the class logits `[B, Q, 91]` (apply a sigmoid; index = COCO category id) and the boxes `[B, Q, 4]`
        (centre x / y, width, height; 0..1). The queries are the encoder's top-k, so their order is not stable under
        small changes.
        """
        return Forward(
            self.net,
            lambda net, x: net(self.normalise(x.mul(255).round())),
            output_names=("logits", "boxes"),
            query_scores=lambda logits: logits.sigmoid().amax(-1),
        )


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


class Kernels(Portable):
    """float16 with fp16-accumulate convs (`cudnn_hacc`) and GEMMs; the network replays from a CUDA graph per shape."""

    module = Runner.module  # fp16-accumulate GEMMs are a cuBLAS mode, not an exportable operator

    def __init__(self, ctx: Context) -> None:
        """Build the network, route its convolutions (groups 1) through `cudnn_hacc`, set up the graph cache."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if is_base(m, torch.nn.Conv2d) and m.groups == 1 and m.padding_mode == "zeros" and m.padding != "same":
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                override(m, _hacc_conv)
        after = cudnn_hacc.capture_after()
        self.graphs = GraphCache(self._eager, graphs=1, capture_after=after) if ctx.params.cuda_graphs else None

    def _eager(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with fp16_accumulate():
            return self.net(x)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = None if self.graphs is None else self.graphs.run((x.device, x.dtype, tuple(x.shape)), x)
        return self._eager(x) if out is None else out

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        if self.graphs is not None:
            self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
