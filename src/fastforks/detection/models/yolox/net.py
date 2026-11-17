"""YOLOX network (Megvii `YOLOX`: CSPDarknet backbone, `YOLOPAFPN` neck, decoupled `YOLOXHead`), inference only.

- CSPDarknet: a Focus stem (2 x 2 space-to-depth, then a 3 x 3 conv), then four stages, each a 3 x 3 stride-2 conv
  and a `CSPLayer`; the last stage has an SPP bottleneck (max pools 5, 9, 13) before its `CSPLayer` (no residuals
  there). A `CSPLayer` splits into a main branch (1 x 1 conv, bottlenecks: 1 x 1 conv, 3 x 3 conv, residual) and a
  1 x 1 shortcut, concatenates them and applies a final 1 x 1 conv. The stride 8 / 16 / 32 maps go on
- `YOLOPAFPN`: top-down (1 x 1 lateral conv, nearest x2, concat, `CSPLayer` without residuals), bottom-up (3 x 3
  stride-2 conv, concat, `CSPLayer`)
- head, per level: a 1 x 1 stem conv to the head width, then two 3 x 3 convs and a 1 x 1 conv to the class logits,
  and two 3 x 3 convs and a 1 x 1 conv to the box (centre offset, log size; in strides) and objectness logits

Every conv except the predictions is followed by BatchNorm (eps 1e-3, set by the reference's `init_yolo`) and SiLU;
BatchNorm is folded into the convs at load. Depth and width scale with `depth` / `width`. The reference network takes
BGR in 0..255; `convert` flips the stem conv's input channels, so this one takes RGB.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.cspnext import SPP, ConvAct, fold_bn

BN_EPS = 1e-3  # fixed: architecture (`init_yolo` sets every BatchNorm's eps)
STRIDES = (8, 16, 32)  # fixed: architecture (the three head levels)
WIDTHS = (256, 512, 1024)  # fixed: the stride 8 / 16 / 32 maps at width 1.0
HEAD_WIDTH = 256  # fixed: head channels at width 1.0


@dataclass(frozen=True)
class Arch:
    """Card `[params]` + variant `params`."""

    depth: float  # depth factor (bottlenecks per `CSPLayer`)
    width: float  # width factor (channels of backbone, neck and head)
    num_classes: int  # class logits per cell


class Bottleneck(nn.Module):
    """1 x 1 conv, 3 x 3 conv, residual if `add`."""

    def __init__(self, c: int, *, add: bool) -> None:
        """In, hidden and out `c` channels."""
        super().__init__()
        self.conv1 = ConvAct(c, c, 1)
        self.conv2 = ConvAct(c, c, 3)
        self.add = add

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv2(conv1(x)) (+ x)."""
        y = self.conv2(self.conv1(x))
        return y + x if self.add else y


class CSPLayer(nn.Module):
    """YOLOv5's C3: main branch (1 x 1 conv, `n` bottlenecks) and a 1 x 1 shortcut, concatenated, 1 x 1 conv."""

    def __init__(self, cin: int, cout: int, n: int, *, residual: bool = True) -> None:
        """`n` bottlenecks on `cout // 2` channels."""
        super().__init__()
        hidden = int(cout * 0.5)
        self.conv1 = ConvAct(cin, hidden, 1)
        self.conv2 = ConvAct(cin, hidden, 1)
        self.conv3 = ConvAct(2 * hidden, cout, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden, add=residual) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv3(cat(m(conv1(x)), conv2(x)))."""
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))


class Focus(nn.Module):
    """2 x 2 space-to-depth (top-left, bottom-left, top-right, bottom-right) and a 3 x 3 conv."""

    def __init__(self, cout: int) -> None:
        """RGB in, `cout` channels out at half resolution."""
        super().__init__()
        self.conv = ConvAct(12, cout, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The conv of the four phases, channel-concatenated."""
        p = (x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2])
        return self.conv(torch.cat(p, 1))


class CSPDarknet(nn.Module):
    """Focus stem + four stages; the maps of the last three."""

    def __init__(self, a: Arch) -> None:
        """Layers of the variant `a`."""
        super().__init__()
        c = int(a.width * 64)
        n = max(round(a.depth * 3), 1)
        self.stem = Focus(c)
        self.dark2 = nn.Sequential(ConvAct(c, 2 * c, 3, 2), CSPLayer(2 * c, 2 * c, n))
        self.dark3 = nn.Sequential(ConvAct(2 * c, 4 * c, 3, 2), CSPLayer(4 * c, 4 * c, 3 * n))
        self.dark4 = nn.Sequential(ConvAct(4 * c, 8 * c, 3, 2), CSPLayer(8 * c, 8 * c, 3 * n))
        self.dark5 = nn.Sequential(
            ConvAct(8 * c, 16 * c, 3, 2), SPP(16 * c), CSPLayer(16 * c, 16 * c, n, residual=False)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The stride 8 / 16 / 32 maps."""
        c3 = self.dark3(self.dark2(self.stem(x)))
        c4 = self.dark4(c3)
        return c3, c4, self.dark5(c4)


class PAFPN(nn.Module):
    """`YOLOPAFPN`: the backbone, then top-down and bottom-up paths over its three maps."""

    def __init__(self, a: Arch) -> None:
        """Layers of the variant `a`; `round(3 * depth)` bottlenecks per `CSPLayer`."""
        super().__init__()
        c3, c4, c5 = (int(c * a.width) for c in WIDTHS)
        n = round(3 * a.depth)
        self.backbone = CSPDarknet(a)
        self.lateral_conv0 = ConvAct(c5, c4, 1)
        self.C3_p4 = CSPLayer(2 * c4, c4, n, residual=False)
        self.reduce_conv1 = ConvAct(c4, c3, 1)
        self.C3_p3 = CSPLayer(2 * c3, c3, n, residual=False)
        self.bu_conv2 = ConvAct(c3, c3, 3, 2)
        self.C3_n3 = CSPLayer(2 * c3, c4, n, residual=False)
        self.bu_conv1 = ConvAct(c4, c4, 3, 2)
        self.C3_n4 = CSPLayer(2 * c4, c5, n, residual=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three head inputs (stride 8, 16, 32)."""
        x2, x1, x0 = self.backbone(x)
        p5 = self.lateral_conv0(x0)
        f4 = self.C3_p4(torch.cat((F.interpolate(p5, scale_factor=2.0, mode="nearest"), x1), 1))
        p4 = self.reduce_conv1(f4)
        n3 = self.C3_p3(torch.cat((F.interpolate(p4, scale_factor=2.0, mode="nearest"), x2), 1))
        n4 = self.C3_n3(torch.cat((self.bu_conv2(n3), p4), 1))
        n5 = self.C3_n4(torch.cat((self.bu_conv1(n4), p5), 1))
        return n3, n4, n5


class Head(nn.Module):
    """`YOLOXHead`: per level a stem, a class branch and a box branch; box and objectness share one 1 x 1 conv."""

    def __init__(self, a: Arch) -> None:
        """Layers of the variant `a`."""
        super().__init__()
        c = int(HEAD_WIDTH * a.width)

        def tower() -> nn.Sequential:
            return nn.Sequential(ConvAct(c, c, 3), ConvAct(c, c, 3))

        self.stems = nn.ModuleList([ConvAct(int(w * a.width), c, 1) for w in WIDTHS])
        self.cls_convs = nn.ModuleList([tower() for _ in STRIDES])
        self.reg_convs = nn.ModuleList([tower() for _ in STRIDES])
        self.cls_preds = nn.ModuleList([nn.Conv2d(c, a.num_classes, 1) for _ in STRIDES])
        self.reg_obj_preds = nn.ModuleList([nn.Conv2d(c, 5, 1) for _ in STRIDES])  # reg_preds (4) + obj_preds (1)

    def forward(self, feats: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """(box + objectness `[B, 5, h, w]`, class logits `[B, classes, h, w]`) of level 0, 1, 2, flat."""
        out: list[torch.Tensor] = []
        for i, x in enumerate(feats):
            x = self.stems[i](x)  # noqa: PLW2901 - the level's stem output replaces its input
            out.append(self.reg_obj_preds[i](self.reg_convs[i](x)))
            out.append(self.cls_preds[i](self.cls_convs[i](x)))
        return tuple(out)


class YOLOX(nn.Module):
    """Backbone + neck and head: an RGB image in 0..255 -> the six head maps."""

    def __init__(self, a: Arch) -> None:
        """Layers of the variant `a`."""
        super().__init__()
        self.backbone = PAFPN(a)
        self.head = Head(a)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """`[B, 3, H, W]` (H, W multiples of 32) -> (box + objectness, class logits) per level, flat."""
        return self.head(self.backbone(x))


def cells(h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The cells' grid positions `[N, 2]` (x, y; in cells) and strides `[N]` of an `h x w` input, level by level."""
    grid, strides = [], []
    for s in STRIDES:
        ys, xs = torch.meshgrid(
            torch.arange(h // s, device=device, dtype=torch.float32),
            torch.arange(w // s, device=device, dtype=torch.float32),
            indexing="ij",
        )
        grid.append(torch.stack((xs.reshape(-1), ys.reshape(-1)), -1))
        strides.append(torch.full((grid[-1].shape[0],), float(s), device=device))
    return torch.cat(grid), torch.cat(strides)


def decode(
    maps: tuple[torch.Tensor, ...], grid: torch.Tensor, strides: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """float32 boxes `[B, N, 4]` (x0, y0, x1, y1; not clamped), objectness and class probabilities of the head maps.

    `YOLOXHead.decode_outputs`: centre `(offset + cell) * stride`, size `exp(output) * stride`.
    """
    box = torch.cat([m.float().flatten(2) for m in maps[0::2]], 2).transpose(1, 2)
    cls = torch.cat([m.float().flatten(2) for m in maps[1::2]], 2).transpose(1, 2)
    s = strides[:, None]
    centre, half = (box[..., :2] + grid) * s, box[..., 2:4].exp() * s / 2
    return torch.cat((centre - half, centre + half), -1), box[..., 4].sigmoid(), cls.sigmoid()


def convert(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The reference keys as `YOLOX` keys: every `BaseConv`'s BatchNorm folded into its conv (weight and bias).

    The stem conv's input channels are flipped within each space-to-depth phase (the reference feeds BGR); the box and
    objectness predictions are stacked into one conv per level.
    """
    out = fold_bn(sd, BN_EPS)
    w = out["backbone.backbone.stem.conv.conv.weight"]
    out["backbone.backbone.stem.conv.conv.weight"] = w.unflatten(1, (4, 3)).flip(2).flatten(1, 2)
    for i in range(len(STRIDES)):
        for k in ("weight", "bias"):
            reg, obj = out.pop(f"head.reg_preds.{i}.{k}"), out.pop(f"head.obj_preds.{i}.{k}")
            out[f"head.reg_obj_preds.{i}.{k}"] = torch.cat((reg, obj))
    return out
