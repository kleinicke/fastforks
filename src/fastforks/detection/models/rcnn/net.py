"""detectron2's R-CNN R50-FPN modules with their state-dict names (inference only; FrozenBatchNorm folded at load).

Ported from facebookresearch/detectron2 (`modeling/backbone/{resnet,fpn}.py`, `modeling/proposal_generator/rpn.py`,
`modeling/roi_heads/{box_head,fast_rcnn,mask_head}.py`; Apache-2.0, see LICENSE). Only the modules and their
parameters: the forward pass is in `model.py`. Layout as the model-zoo configs `Base-RCNN-FPN.yaml` (ResNet with
`STRIDE_IN_1X1`, stride on the first 1 x 1 conv of a stage; FrozenBatchNorm; FPN with `sum` fusion and a max-pooled P6;
a shared RPN head; a 2-FC box head; a 4-conv mask head with a 2x deconvolution).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fastforks._core.errors import CheckpointMismatchError

FROZEN_BN_EPS = 1e-5  # fixed: detectron2 FrozenBatchNorm2d default eps
STEM_WIDTH = 64  # fixed: ResNet stem width (all detectron2 ResNets)


@dataclass(frozen=True)
class Arch:
    """Network shape (card `[params]`)."""

    blocks: list[int]  # bottleneck blocks per stage res2..res5
    fpn_dim: int  # FPN / RPN width
    num_anchors: int  # anchors per location (aspect ratios)
    box_pool: int  # box ROIAlign output side
    mask_pool: int  # mask ROIAlign output side
    fc_dim: int  # box head FC width
    num_classes: int
    mask: bool  # whether the variant has a mask head


def _conv(cin: int, cout: int, k: int, stride: int = 1) -> nn.Conv2d:
    """A conv with a bias (the folded FrozenBatchNorm, or the conv's own)."""
    return nn.Conv2d(cin, cout, k, stride, k // 2)


class Bottleneck(nn.Module):
    """detectron2 `BottleneckBlock` (stride in the first 1 x 1 conv)."""

    def __init__(self, cin: int, mid: int, cout: int, stride: int) -> None:
        """Convs of one block; a projection shortcut where the width changes."""
        super().__init__()
        self.shortcut = _conv(cin, cout, 1, stride) if cin != cout else None
        self.conv1 = _conv(cin, mid, 1, stride)
        self.conv2 = _conv(mid, mid, 3)
        self.conv3 = _conv(mid, cout, 1)


class Stem(nn.Module):
    """detectron2 `BasicStem`: 7 x 7 stride-2 conv (+ ReLU, 3 x 3 stride-2 max pooling in the forward)."""

    def __init__(self) -> None:
        """The stem conv."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, STEM_WIDTH, 7, 2, 3)


class ResNet(nn.Module):
    """`bottom_up`: the stem and the stages `res2` .. `res5`."""

    def __init__(self, a: Arch) -> None:
        """Stages of `a.blocks` bottlenecks (widths 64 -> 256, x 2 per stage)."""
        super().__init__()
        self.stem = Stem()
        cin, mid = STEM_WIDTH, STEM_WIDTH
        self._stages: list[list[Bottleneck]] = []
        for i, n in enumerate(a.blocks):
            cout = mid * 4
            blocks = [Bottleneck(cin if j == 0 else cout, mid, cout, 2 if i > 0 and j == 0 else 1) for j in range(n)]
            self.add_module(f"res{i + 2}", nn.Sequential(*blocks))
            self._stages.append(blocks)
            cin, mid = cout, mid * 2

    def stages(self) -> list[list[Bottleneck]]:
        """The blocks of `res2` .. `res5`."""
        return self._stages


class Backbone(nn.Module):
    """detectron2 `FPN` over the ResNet (`fpn_lateral{2..5}`, `fpn_output{2..5}`)."""

    def __init__(self, a: Arch) -> None:
        """ResNet + lateral 1 x 1 and output 3 x 3 convs per level."""
        super().__init__()
        self.bottom_up = ResNet(a)
        for i in range(len(a.blocks)):
            cin = STEM_WIDTH * 4 * 2**i
            self.add_module(f"fpn_lateral{i + 2}", nn.Conv2d(cin, a.fpn_dim, 1))
            self.add_module(f"fpn_output{i + 2}", nn.Conv2d(a.fpn_dim, a.fpn_dim, 3, 1, 1))

    def fpn(self) -> list[tuple[nn.Conv2d, nn.Conv2d]]:
        """`(lateral, output)` convs of levels 2 .. 5."""
        n = len(self.bottom_up.stages())
        return [(getattr(self, f"fpn_lateral{i + 2}"), getattr(self, f"fpn_output{i + 2}")) for i in range(n)]


class RpnHead(nn.Module):
    """detectron2 `StandardRPNHead`: a shared 3 x 3 conv + ReLU, objectness and box-delta 1 x 1 convs."""

    def __init__(self, a: Arch) -> None:
        """Convs for `a.num_anchors` anchors per location."""
        super().__init__()
        self.conv = nn.Conv2d(a.fpn_dim, a.fpn_dim, 3, 1, 1)
        self.objectness_logits = nn.Conv2d(a.fpn_dim, a.num_anchors, 1)
        self.anchor_deltas = nn.Conv2d(a.fpn_dim, a.num_anchors * 4, 1)


class BoxHead(nn.Module):
    """detectron2 `FastRCNNConvFCHead` with two FCs (+ ReLU)."""

    def __init__(self, a: Arch) -> None:
        """fc1 on the flattened pooled features, fc2."""
        super().__init__()
        self.fc1 = nn.Linear(a.fpn_dim * a.box_pool**2, a.fc_dim)
        self.fc2 = nn.Linear(a.fc_dim, a.fc_dim)


class BoxPredictor(nn.Module):
    """detectron2 `FastRCNNOutputLayers`: class scores (+ background) and per-class box deltas."""

    def __init__(self, a: Arch) -> None:
        """The two linear layers."""
        super().__init__()
        self.cls_score = nn.Linear(a.fc_dim, a.num_classes + 1)
        self.bbox_pred = nn.Linear(a.fc_dim, a.num_classes * 4)


class MaskHead(nn.Module):
    """detectron2 `MaskRCNNConvUpsampleHead`: 4 x (3 x 3 conv + ReLU), 2x deconv + ReLU, per-class 1 x 1 predictor."""

    def __init__(self, a: Arch) -> None:
        """The convs."""
        super().__init__()
        for i in range(4):
            self.add_module(f"mask_fcn{i + 1}", nn.Conv2d(a.fpn_dim, a.fpn_dim, 3, 1, 1))
        self.deconv = nn.ConvTranspose2d(a.fpn_dim, a.fpn_dim, 2, 2)
        self.predictor = nn.Conv2d(a.fpn_dim, a.num_classes, 1)

    def convs(self) -> list[nn.Conv2d]:
        """`mask_fcn1` .. `mask_fcn4`."""
        return [getattr(self, f"mask_fcn{i + 1}") for i in range(4)]


class RoiHeads(nn.Module):
    """detectron2 `StandardROIHeads` parameters."""

    def __init__(self, a: Arch) -> None:
        """Box head and predictor, and the mask head for mask variants."""
        super().__init__()
        self.box_head = BoxHead(a)
        self.box_predictor = BoxPredictor(a)
        self.mask_head = MaskHead(a) if a.mask else None


class ProposalGenerator(nn.Module):
    """detectron2 `RPN` parameters."""

    def __init__(self, a: Arch) -> None:
        """The RPN head."""
        super().__init__()
        self.rpn_head = RpnHead(a)


class Rcnn(nn.Module):
    """detectron2 `GeneralizedRCNN` parameters (same state-dict names, FrozenBatchNorm folded)."""

    def __init__(self, a: Arch) -> None:
        """Untrained network of shape `a`."""
        super().__init__()
        self.backbone = Backbone(a)
        self.proposal_generator = ProposalGenerator(a)
        self.roi_heads = RoiHeads(a)


_BN_PARTS = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")  # the last is dropped


def fold_frozen_bn(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """detectron2 state dict -> the same with every `<conv>.norm` (FrozenBatchNorm) folded into `<conv>.bias`.

    `y = (conv(x) - mean) * w / sqrt(var + eps) + b` (computed in float64). Anchor buffers (`cell_anchors`) are dropped:
    `model.py` computes them from the card.
    """
    stray = [k for k in sd if ".norm." in k and k.rsplit(".", 1)[1] not in _BN_PARTS]
    if stray:
        what, details = "rcnn checkpoint", f"unexpected FrozenBatchNorm keys ({', '.join(stray[:5])})"
        raise CheckpointMismatchError(what, details)
    out = {k: v for k, v in sd.items() if ".norm." not in k and "anchor_generator" not in k}
    for k in sd:
        if k.endswith(".norm.weight"):
            conv = k.removesuffix(".norm.weight")
            g, b, m, v = (sd[f"{conv}.norm.{n}"].double() for n in ("weight", "bias", "running_mean", "running_var"))
            s = g * (v + FROZEN_BN_EPS).rsqrt()
            w = sd[f"{conv}.weight"]
            out[f"{conv}.weight"] = (w.double() * s.view(-1, 1, 1, 1)).to(w.dtype)
            out[f"{conv}.bias"] = (b - m * s).to(w.dtype)
    return out
