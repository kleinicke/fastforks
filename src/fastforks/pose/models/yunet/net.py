"""YuNet network (ShiqiYu/libfacedetection.train `yunet_train`: `YuNetBackbone`, `TFPN`, `YuNetHead`), inference only.

- unit (`ConvDPUnit`): 1 x 1 conv, 3 x 3 depthwise conv (both with bias), then BatchNorm + ReLU (not in the heads)
- backbone: a stride-2 3 x 3 stem conv + BN + ReLU and one unit (`model0`), then five blocks of two units
  (`model1..5`), 2 x 2 max pooling after `model0`, `model2`, `model3` and `model4`; `model3..5` are the stride 8 / 16 /
  32 maps
- neck (`TFPN`): top-down, each coarser map through its lateral unit, nearest-upsampled x 2 and added to the finer one
- heads: per map an optional shared unit (`yunet_n`), then one head unit per output: here the class (1), objectness
  (1), box (4) and landmark (10) units of a level are one unit with 16 outputs (`convert` concatenates the checkpoint's
  four; the 1 x 1 and depthwise convs are per channel, so this is exact)

Module names follow the checkpoint (`convert` only merges the heads); `fastforks._optimize` folds the BatchNorms.
The input height and width must be multiples of 32 (the runner pads, as upstream's evaluation `Pad(size_divisor=32)`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

LEVELS = 3  # fixed: reference stride 8 / 16 / 32 maps
HEAD = 1 + 1 + 4 + 10  # merged head channels: class, objectness, box, 5 landmarks
PARTS = ("cls", "obj", "bbox", "kps")  # checkpoint head names in merged channel order


@dataclass(frozen=True)
class Arch:
    """Card `[params]` + variant `params`: the two released sizes differ in two places."""

    stage1: int  # channels out of `model1` (yunet_n 64, yunet_s 32)
    shared_convs: int  # shared head units per level (yunet_n 1, yunet_s 0)


class Unit(nn.Module):
    """`ConvDPUnit`: 1 x 1 conv, 3 x 3 depthwise conv, optionally BatchNorm + ReLU."""

    def __init__(self, cin: int, cout: int, *, bn: bool = True) -> None:
        """`cin` -> `cout` channels."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, groups=cout)
        self.bn = nn.BatchNorm2d(cout) if bn else None
        self.relu = nn.ReLU() if bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The unit."""
        x = self.conv2(self.conv1(x))
        return x if self.bn is None or self.relu is None else self.relu(self.bn(x))


class Stem(nn.Module):
    """`Conv_head`: stride-2 3 x 3 conv + BN + ReLU, then one unit."""

    def __init__(self, cin: int, mid: int, cout: int) -> None:
        """`cin` -> `mid` (stride 2) -> `cout` channels."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, mid, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.relu1 = nn.ReLU()
        self.conv2 = Unit(mid, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The stem."""
        return self.conv2(self.relu1(self.bn1(self.conv1(x))))


class Block(nn.Module):
    """`Conv4layerBlock`: two units."""

    def __init__(self, cin: int, cout: int) -> None:
        """`cin` -> `cin` -> `cout` channels."""
        super().__init__()
        self.conv1 = Unit(cin, cin)
        self.conv2 = Unit(cin, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The block."""
        return self.conv2(self.conv1(x))


class Backbone(nn.Module):
    """`YuNetBackbone`: the stride 8 / 16 / 32 maps (64 channels each)."""

    def __init__(self, a: Arch) -> None:
        """The stem and five blocks."""
        super().__init__()
        self.model0 = Stem(3, 16, 16)
        self.model1 = Block(16, a.stage1)
        self.model2 = Block(a.stage1, 64)
        self.model3 = Block(64, 64)
        self.model4 = Block(64, 64)
        self.model5 = Block(64, 64)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stride 8, 16, 32 maps."""
        x = F.max_pool2d(self.model0(x), 2)
        x = F.max_pool2d(self.model2(self.model1(x)), 2)
        c3 = self.model3(x)
        c4 = self.model4(F.max_pool2d(c3, 2))
        c5 = self.model5(F.max_pool2d(c4, 2))
        return c3, c4, c5


class Neck(nn.Module):
    """`TFPN`: top-down sums of the lateral units' outputs."""

    def __init__(self) -> None:
        """One lateral unit per level."""
        super().__init__()
        self.lateral_convs = nn.ModuleList(Unit(64, 64) for _ in range(LEVELS))

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The top-down sums, finest first."""
        p5 = self.lateral_convs[2](c5)
        p4 = self.lateral_convs[1](c4 + F.interpolate(p5, scale_factor=2.0, mode="nearest"))
        p3 = self.lateral_convs[0](c3 + F.interpolate(p4, scale_factor=2.0, mode="nearest"))
        return p3, p4, p5


class Head(nn.Module):
    """`YuNetHead` (`stacked_convs = 0`) with the four output units of a level merged (`heads`)."""

    def __init__(self, a: Arch) -> None:
        """The shared units (if any) and the merged output units."""
        super().__init__()
        self.multi_level_share_convs = (
            nn.ModuleList(nn.Sequential(*(Unit(64, 64) for _ in range(a.shared_convs))) for _ in range(LEVELS))
            if a.shared_convs
            else None
        )
        self.heads = nn.ModuleList(Unit(64, HEAD, bn=False) for _ in range(LEVELS))

    def forward(self, feats: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Per level the 16 head channels: class, objectness, box (4), landmarks (10)."""
        if self.multi_level_share_convs is not None:
            feats = tuple(m(f) for m, f in zip(self.multi_level_share_convs, feats, strict=True))
        return tuple(m(f) for m, f in zip(self.heads, feats, strict=True))


class YuNet(nn.Module):
    """Image `[B, 3, H, W]` (BGR, 0..255, H and W multiples of 32) -> per level `[B, 16, H / s, W / s]` head maps."""

    def __init__(self, a: Arch) -> None:
        """Backbone, neck and head."""
        super().__init__()
        self.backbone = Backbone(a)
        self.neck = Neck()
        self.bbox_head = Head(a)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The head maps."""
        return self.bbox_head(self.neck(*self.backbone(x)))


def convert(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A checkpoint (`module.` prefix and `state_dict` wrapper allowed) with each level's four head units merged."""
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    out = {k: v for k, v in sd.items() if not k.startswith(tuple(f"bbox_head.multi_level_{p}." for p in PARTS))}
    for level in range(LEVELS):
        for name in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"):
            parts = [sd[f"bbox_head.multi_level_{p}.{level}.{name}"] for p in PARTS]
            out[f"bbox_head.heads.{level}.{name}"] = torch.cat(parts)
    return out
