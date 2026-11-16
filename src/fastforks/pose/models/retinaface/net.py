"""RetinaFace network (facexlib 0.3.0 `RetinaFace`, from biubug6/Pytorch_Retinaface), inference only.

- backbone: torchvision ResNet-50 (`layer2..4`, strides 8 / 16 / 32) or the 0.25-width MobileNetV1 (`stage1..3`)
- `FPN`: 1 x 1 projections to `out` channels, top-down nearest upsampling + add, a 3 x 3 merge conv per finer map
- `SSH` per map: a 3 x 3 branch (out / 2) and a 5 x 5 and a 7 x 7 branch made of stacked 3 x 3 convs (out / 4
  each), concatenated, ReLU
- heads: per map one 1 x 1 conv with the class (2 x 2), box (2 x 4) and landmark (2 x 10) outputs of the two anchors
  concatenated (`convert` merges upstream's three heads), channel `a * n + c` of each part as upstream's `view`

Module names follow the checkpoint, so `load_state_dict` reads it as is; `fastforks._optimize` folds the BatchNorms.
Activations are LeakyReLU(0.1) where upstream has `out_channels <= 64` (MobileNet) and ReLU otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

LEAKY = 0.1  # fixed: reference slope of the MobileNet backbone and of FPN / SSH when out_channels <= 64
ANCHORS = 2  # fixed: reference anchors per cell (min_sizes per level)
HEAD = ANCHORS * (2 + 4 + 10)  # merged head channels: class, box, 5 landmarks
RESNET_BLOCKS = (3, 4, 6, 3)  # fixed: torchvision ResNet-50
MOBILENET = (  # fixed: reference MobileNetV1 x 0.25 (stage, out channels, stride) after the stem conv (8, stride 2)
    (1, 16, 1), (1, 32, 2), (1, 32, 1), (1, 64, 2), (1, 64, 1),
    (2, 128, 2), (2, 128, 1), (2, 128, 1), (2, 128, 1), (2, 128, 1), (2, 128, 1),
    (3, 256, 2), (3, 256, 1),
)  # fmt: skip


@dataclass(frozen=True)
class Arch:
    """Card `[params]` + variant `params`."""

    backbone: str  # "resnet50" or "mobilenet0.25"
    out_channels: int  # FPN / SSH / head width


def _act(leaky: float) -> nn.Module:
    return nn.LeakyReLU(leaky, inplace=True) if leaky else nn.ReLU(inplace=True)


def conv_bn(  # noqa: PLR0913 - the conv and its activation
    cin: int, cout: int, k: int = 3, *, stride: int = 1, groups: int = 1, act: float | None = None
) -> nn.Sequential:
    """Conv (no bias) + BatchNorm (+ LeakyReLU(`act`) / ReLU for `act == 0`), upstream `conv_bn*` (`.0` / `.1` keys)."""
    layers = [nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    return nn.Sequential(*layers, *([] if act is None else [_act(act)]))


class Bottleneck(nn.Module):
    """torchvision ResNet `Bottleneck` (stride on the 3 x 3 conv)."""

    def __init__(self, cin: int, width: int, stride: int) -> None:
        """`cin` -> `4 width` channels; a projection shortcut where the shape changes."""
        super().__init__()
        cout = width * 4
        self.conv1, self.bn1 = nn.Conv2d(cin, width, 1, bias=False), nn.BatchNorm2d(width)
        self.conv2, self.bn2 = nn.Conv2d(width, width, 3, stride, 1, bias=False), nn.BatchNorm2d(width)
        self.conv3, self.bn3 = nn.Conv2d(width, cout, 1, bias=False), nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The residual block."""
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + (x if self.downsample is None else self.downsample(x)))


class ResNet50(nn.Module):
    """torchvision `resnet50` up to `layer4`; returns the `layer2..4` maps."""

    def __init__(self) -> None:
        """The stem and four stages."""
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        cin, layers = 64, []
        for i, n in enumerate(RESNET_BLOCKS):
            width = 64 << i
            blocks = [Bottleneck(cin if b == 0 else width * 4, width, 2 if i and b == 0 else 1) for b in range(n)]
            layers.append(nn.Sequential(*blocks))
            cin = width * 4
        self.layer1, self.layer2, self.layer3, self.layer4 = layers

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stride 8, 16, 32 maps."""
        x = self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        return c3, c4, self.layer4(c4)


class MobileNet(nn.Module):
    """Upstream `MobileNetV1` (0.25 width) without its classifier; returns the `stage1..3` maps."""

    def __init__(self) -> None:
        """The stem conv and the depthwise-separable blocks of each stage."""
        super().__init__()
        stages: list[list[nn.Module]] = [[conv_bn(3, 8, stride=2, act=LEAKY)], [], []]
        cin = 8
        for stage, cout, stride in MOBILENET:
            dw = conv_bn(cin, cin, stride=stride, groups=cin, act=LEAKY)
            pw = conv_bn(cin, cout, 1, act=LEAKY)
            stages[stage - 1].append(nn.Sequential(*dw, *pw))  # upstream `conv_dw`: keys .0/.1 and .3/.4
            cin = cout
        self.stage1, self.stage2, self.stage3 = (nn.Sequential(*s) for s in stages)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stride 8, 16, 32 maps."""
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        return c3, c4, self.stage3(c4)


def _up(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Nearest upsampling of `x` to `like`'s size (one row / column less on odd maps).

    Upstream's `F.interpolate(size=...)` gives source index `floor(i * n / m)` = `i // 2` for `m` in {2n - 1, 2n}.
    """
    return F.interpolate(x, scale_factor=2.0)[:, :, : like.shape[2], : like.shape[3]]


class FPN(nn.Module):
    """Upstream `FPN`."""

    def __init__(self, cins: tuple[int, int, int], out: int, leaky: float) -> None:
        """Projections and merge convs."""
        super().__init__()
        self.output1, self.output2, self.output3 = (conv_bn(c, out, 1, act=leaky) for c in cins)
        self.merge1, self.merge2 = conv_bn(out, out, act=leaky), conv_bn(out, out, act=leaky)

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Top-down pass."""
        p5 = self.output3(c5)
        p4 = self.output2(c4)
        p4 = self.merge2(p4 + _up(p5, p4))
        p3 = self.output1(c3)
        return self.merge1(p3 + _up(p4, p3)), p4, p5


class SSH(nn.Module):
    """Upstream `SSH` context module (its key names, incl. `conv7x7_3`)."""

    def __init__(self, c: int, leaky: float) -> None:
        """Three branches of 3 x 3 convs."""
        super().__init__()
        self.conv3X3 = conv_bn(c, c // 2)
        self.conv5X5_1 = conv_bn(c, c // 4, act=leaky)
        self.conv5X5_2 = conv_bn(c // 4, c // 4)
        self.conv7X7_2 = conv_bn(c // 4, c // 4, act=leaky)
        self.conv7x7_3 = conv_bn(c // 4, c // 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenated branches, ReLU."""
        t = self.conv5X5_1(x)
        return self.relu(torch.cat([self.conv3X3(x), self.conv5X5_2(t), self.conv7x7_3(self.conv7X7_2(t))], 1))


class RetinaFace(nn.Module):
    """Image `[B, 3, H, W]` (BGR, mean-subtracted 0..255) -> per level `[B, HEAD, ceil(H / s), ceil(W / s)]`."""

    def __init__(self, a: Arch) -> None:
        """Backbone, FPN, SSH and the merged heads."""
        super().__init__()
        mobile = a.backbone == "mobilenet0.25"
        self.body = MobileNet() if mobile else ResNet50()
        cin = 32 if mobile else 256  # upstream `in_channel`: stage maps have 2, 4, 8 times it
        leaky = LEAKY if a.out_channels <= 64 else 0.0  # noqa: PLR2004 - upstream's rule
        self.fpn = FPN((cin * 2, cin * 4, cin * 8), a.out_channels, leaky)
        self.ssh1, self.ssh2, self.ssh3 = (SSH(a.out_channels, leaky) for _ in range(3))
        self.head = nn.ModuleList(nn.Conv2d(a.out_channels, HEAD, 1) for _ in range(3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raw head maps of the stride 8, 16, 32 levels."""
        p3, p4, p5 = self.fpn(*self.body(x))
        return self.head[0](self.ssh1(p3)), self.head[1](self.ssh2(p4)), self.head[2](self.ssh3(p5))


def convert(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Upstream checkpoint -> `RetinaFace` state dict.

    The three heads of each level are merged into `head.<i>`; the unused MobileNet classifier is dropped.
    """
    out = {k: v for k, v in sd.items() if not k.startswith(("ClassHead.", "BboxHead.", "LandmarkHead.", "body.fc."))}
    for i in range(3):
        for p in ("weight", "bias"):
            heads = ("ClassHead", "BboxHead", "LandmarkHead")
            out[f"head.{i}.{p}"] = torch.cat([sd[f"{h}.{i}.conv1x1.{p}"] for h in heads])
    return out
