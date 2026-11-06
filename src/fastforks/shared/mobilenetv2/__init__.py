"""MobileNetV2 (1.0x) feature stages with timm's module names, so timm-trained checkpoints load unchanged.

Many stereo nets (IGEV-Stereo and its derivatives) build their feature encoder from
`timm.create_model("mobilenetv2_100", features_only=True)` and keep `conv_stem`, `bn1` and slices of `blocks` in their
state dict. This module rebuilds that structure without timm (a heavy dependency for ~100 lines of layers): same
module tree, parameter names and eval-mode math. The layer layout follows timm's `mobilenetv2_100` definition
(Apache-2.0, see LICENSE).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    import torch

# timm arch string per stage: (block type, repeats, stride, expansion, out channels), all 3x3 depthwise
STAGES = (
    ("ds", 1, 1, 1, 16),
    ("ir", 2, 2, 6, 24),
    ("ir", 3, 2, 6, 32),
    ("ir", 4, 2, 6, 64),
    ("ir", 3, 1, 6, 96),
    ("ir", 3, 2, 6, 160),
    ("ir", 1, 1, 6, 320),
)
STEM_CHANNELS = 32


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm followed by an optional ReLU6 (timm's `BatchNormAct2d`; `drop` is identity at inference)."""

    def __init__(self, channels: int, *, act: bool = True) -> None:
        """Create the norm; `act=False` leaves the activation as identity."""
        super().__init__(channels)
        self.drop = nn.Identity()
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - the overridden method's parameter name
        """Normalize, then activate."""
        return self.act(self.drop(super().forward(input)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 -> BN + ReLU6 -> pointwise -> BN (timm `ds` block; MobileNetV2's first stage has no skip)."""

    def __init__(self, cin: int, cout: int, stride: int) -> None:
        """Create the block."""
        super().__init__()
        self.has_skip = stride == 1 and cin == cout
        self.conv_dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = BatchNormAct2d(cin)
        self.aa = nn.Identity()
        self.se = nn.Identity()
        self.conv_pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = BatchNormAct2d(cout, act=False)
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        y = self.bn2(self.conv_pw(self.se(self.aa(self.bn1(self.conv_dw(x))))))
        return self.drop_path(y) + x if self.has_skip else y


class InvertedResidual(nn.Module):
    """Pointwise expand -> depthwise 3x3 -> pointwise linear projection, residual at stride 1 (timm `ir` block)."""

    def __init__(self, cin: int, cout: int, stride: int, expansion: int) -> None:
        """Create the block."""
        super().__init__()
        mid = cin * expansion
        self.has_skip = stride == 1 and cin == cout
        self.conv_pw = nn.Conv2d(cin, mid, 1, bias=False)
        self.bn1 = BatchNormAct2d(mid)
        self.conv_dw = nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False)
        self.bn2 = BatchNormAct2d(mid)
        self.aa = nn.Identity()
        self.se = nn.Identity()
        self.conv_pwl = nn.Conv2d(mid, cout, 1, bias=False)
        self.bn3 = BatchNormAct2d(cout, act=False)
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        y = self.bn2(self.conv_dw(self.bn1(self.conv_pw(x))))
        y = self.bn3(self.conv_pwl(self.se(self.aa(y))))
        return self.drop_path(y) + x if self.has_skip else y


class MobileNetV2Features(nn.Module):
    """`conv_stem`, `bn1` (with ReLU6) and `blocks` (one `nn.Sequential` per stage), as timm's features model.

    Attributes:
        channels: Output channels of each stage.
    """

    def __init__(self) -> None:
        """Create the network (untrained; load a checkpoint that contains it)."""
        super().__init__()
        self.conv_stem = nn.Conv2d(3, STEM_CHANNELS, 3, 2, 1, bias=False)
        self.bn1 = BatchNormAct2d(STEM_CHANNELS)
        stages: list[nn.Module] = []
        cin = STEM_CHANNELS
        for kind, repeats, stride, expansion, cout in STAGES:
            blocks: list[nn.Module] = []
            for i in range(repeats):
                s = stride if i == 0 else 1
                blocks.append(
                    DepthwiseSeparableConv(cin, cout, s) if kind == "ds" else InvertedResidual(cin, cout, s, expansion)
                )
                cin = cout
            stages.append(nn.Sequential(*blocks))
        self.blocks = nn.Sequential(*stages)
        self.channels = [c for *_, c in STAGES]
