"""RTMDet network (mmdetection 3.3.0 `RTMDet`: CSPNeXt backbone, CSPNeXt-PAFPN neck, `RTMDetSepBNHead`), inference only.

- CSPNeXt (arch P5): a 3-conv stem (stride 2), then four stages, each a 3 x 3 stride-2 conv and a `CSPLayer`; the last
  stage has an SPP bottleneck (max pools 5, 9, 13) before it. A `CSPLayer` splits into a main branch (1 x 1 conv,
  `CSPNeXtBlock`s: 3 x 3 conv, then 5 x 5 depthwise + 1 x 1 pointwise, residual) and a 1 x 1 shortcut, concatenates
  them, applies channel attention (global average pool, 1 x 1 conv with bias, hard sigmoid, multiply; backbone only)
  and a final 1 x 1 conv. The stride 8 / 16 / 32 maps go on
- CSPNeXt-PAFPN: top-down (1 x 1 reduce, nearest x2, concat, `CSPLayer` without shortcut), bottom-up (3 x 3 stride-2
  conv, concat, `CSPLayer`), then a 3 x 3 conv per level to the head width
- head, per level: two 3 x 3 convs and a 1 x 1 conv to the class logits; two 3 x 3 convs and a 1 x 1 conv to the four
  distances (left, top, right, bottom) from the cell's point, in strides (`exp` of the output for M / L / X). The
  reference shares the 3 x 3 convs between the levels with a BatchNorm per level; folded, each level has its own

Every conv is followed by BatchNorm (eps 1e-5) and SiLU; BatchNorm is folded into the convs at load. Depth and width
scale with `deepen` / `widen`; `depthwise` (nano) replaces the 3 x 3 convs outside the stem by depthwise + 1 x 1.
The reference network takes BGR; `convert` flips the first conv, so this one takes RGB.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# (in, out, blocks, identity shortcut in the blocks); the last stage also has the SPP bottleneck
STAGES = (
    (64, 128, 3, True),
    (128, 256, 6, True),
    (256, 512, 6, True),
    (512, 1024, 3, False),
)  # fixed: CSPNeXt P5 at 1.0
SPP_POOLS = (5, 9, 13)  # fixed: architecture (SPP bottleneck kernels of the last stage)
DW_KERNEL = 5  # fixed: architecture (CSPNeXtBlock depthwise kernel)
BN_EPS = 1e-5  # fixed: architecture (SyncBN default of the configs)
STRIDES = (8, 16, 32)  # fixed: architecture (the three head levels)
STACKED = 2  # fixed: architecture (head `stacked_convs`)


@dataclass(frozen=True)
class Arch:
    """Card `[params]` + variant `params`."""

    deepen: float  # CSPNeXt depth factor (blocks per stage)
    widen: float  # width factor (channels of backbone, neck and head)
    neck_blocks: int  # blocks per neck `CSPLayer`
    num_classes: int  # class logits per cell
    exp_on_reg: bool  # distances are exp(output) (M / L / X) rather than the output
    depthwise: bool  # depthwise-separable 3 x 3 convs (nano)


class ConvAct(nn.Module):
    """mmcv `ConvModule` with its BatchNorm folded: conv (with the folded bias) + SiLU."""

    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, groups: int = 1) -> None:
        """A `k x k` conv with `k // 2` padding."""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU(conv(x)) (traced by `fastforks._optimize` as a conv module + SiLU)."""
        return F.silu(self.conv(x))


class Separable(nn.Module):
    """mmcv `DepthwiseSeparableConvModule`: `k x k` depthwise conv + SiLU, then 1 x 1 conv + SiLU."""

    def __init__(self, cin: int, cout: int, k: int, stride: int = 1) -> None:
        """Depthwise on `cin` channels, pointwise to `cout`."""
        super().__init__()
        self.depthwise_conv = ConvAct(cin, cin, k, stride, groups=cin)
        self.pointwise_conv = ConvAct(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pointwise(depthwise(x))."""
        return self.pointwise_conv(self.depthwise_conv(x))


def conv3(cin: int, cout: int, stride: int = 1, *, depthwise: bool) -> nn.Module:
    """A 3 x 3 conv + SiLU, depthwise-separable for nano."""
    return Separable(cin, cout, 3, stride) if depthwise else ConvAct(cin, cout, 3, stride)


class Block(nn.Module):
    """`CSPNeXtBlock` (expansion 1): dwsep5x5(conv3x3(x)), plus x with the identity shortcut."""

    def __init__(self, c: int, *, identity: bool, depthwise: bool) -> None:
        """Layers on `c` channels."""
        super().__init__()
        self.conv1 = conv3(c, c, depthwise=depthwise)
        self.conv2 = Separable(c, c, DW_KERNEL)
        self.identity = identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The block's output."""
        y = self.conv2(self.conv1(x))
        return x + y if self.identity else y


class Attention(nn.Module):
    """Channel attention: x * hardsigmoid(conv1x1(mean over HW))."""

    def __init__(self, c: int) -> None:
        """The 1 x 1 conv with bias."""
        super().__init__()
        self.fc = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reweighted channels."""
        return x * F.hardsigmoid(self.fc(x.mean((2, 3), keepdim=True)))


class CSPLayer(nn.Module):
    """`CSPLayer` with CSPNeXt blocks; channel attention in the backbone."""

    def __init__(  # noqa: PLR0913 - one layer, keyword-only switches
        self, cin: int, cout: int, blocks: int, *, identity: bool, attention: bool, depthwise: bool
    ) -> None:
        """`cin` -> `cout` channels, `blocks` blocks on `cout // 2`."""
        super().__init__()
        mid = cout // 2
        self.main_conv = ConvAct(cin, mid, 1)
        self.short_conv = ConvAct(cin, mid, 1)
        self.blocks = nn.Sequential(*(Block(mid, identity=identity, depthwise=depthwise) for _ in range(blocks)))
        self.attention = Attention(2 * mid) if attention else None
        self.final_conv = ConvAct(2 * mid, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """final(attention(cat(blocks(main(x)), short(x))))."""
        y = torch.cat([self.blocks(self.main_conv(x)), self.short_conv(x)], 1)
        if self.attention is not None:
            y = self.attention(y)
        return self.final_conv(y)


class SPP(nn.Module):
    """SPP bottleneck: 1 x 1 conv to half, cat with max pools (stride 1, same size), 1 x 1 conv."""

    def __init__(self, c: int) -> None:
        """In and out `c` channels."""
        super().__init__()
        self.conv1 = ConvAct(c, c // 2, 1)
        self.conv2 = ConvAct(c // 2 * (len(SPP_POOLS) + 1), c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv2(cat(x, pools(x)))."""
        x = self.conv1(x)
        return self.conv2(torch.cat([x, *(F.max_pool2d(x, k, 1, k // 2) for k in SPP_POOLS)], 1))


def blocks_of(n: int, deepen: float) -> int:
    """Blocks of a stage: `max(round(n * deepen), 1)` (Python's round, as the reference)."""
    return max(round(n * deepen), 1)


def widths(a: Arch) -> tuple[int, int, int]:
    """Channels of the stride 8 / 16 / 32 backbone maps."""
    return int(STAGES[1][1] * a.widen), int(STAGES[2][1] * a.widen), int(STAGES[3][1] * a.widen)


class CSPNeXt(nn.Module):
    """The backbone: stem + 4 stages; the maps of the last three stages."""

    def __init__(self, a: Arch) -> None:
        """Layers scaled by `a.deepen` / `a.widen`."""
        super().__init__()
        w = a.widen
        c0 = int(STAGES[0][0] * w // 2)
        self.stem = nn.Sequential(ConvAct(3, c0, 3, 2), ConvAct(c0, c0, 3), ConvAct(c0, int(STAGES[0][0] * w), 3))
        stages: list[nn.Sequential] = []
        for i, (cin, cout, n, identity) in enumerate(STAGES):
            ci, co = int(cin * w), int(cout * w)
            layers: list[nn.Module] = [conv3(ci, co, 2, depthwise=a.depthwise)]
            if i == len(STAGES) - 1:
                layers.append(SPP(co))
            csp = CSPLayer(co, co, blocks_of(n, a.deepen), identity=identity, attention=True, depthwise=a.depthwise)
            layers.append(csp)
            stages.append(nn.Sequential(*layers))
        self.stage1, self.stage2, self.stage3, self.stage4 = stages  # the mmdet names

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The stride 8, 16 and 32 maps."""
        x = self.stage1(self.stem(x))
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        return c3, c4, self.stage4(c4)


class PAFPN(nn.Module):
    """`CSPNeXtPAFPN`: top-down and bottom-up paths over the three maps, then a conv per level to the head width."""

    def __init__(self, a: Arch) -> None:
        """Layers on the backbone's widths; `a.neck_blocks` blocks per `CSPLayer`."""
        super().__init__()
        c3, c4, c5 = widths(a)
        dw = a.depthwise

        def csp(cin: int, cout: int) -> CSPLayer:
            return CSPLayer(cin, cout, a.neck_blocks, identity=False, attention=False, depthwise=dw)

        self.reduce_layers = nn.ModuleList([ConvAct(c5, c4, 1), ConvAct(c4, c3, 1)])
        self.top_down_blocks = nn.ModuleList([csp(2 * c4, c4), csp(2 * c3, c3)])
        self.downsamples = nn.ModuleList([conv3(c3, c3, 2, depthwise=dw), conv3(c4, c4, 2, depthwise=dw)])
        self.bottom_up_blocks = nn.ModuleList([csp(2 * c3, c4), csp(2 * c4, c5)])
        self.out_convs = nn.ModuleList([conv3(c, c3, depthwise=dw) for c in (c3, c4, c5)])

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The three head inputs (stride 8, 16, 32)."""
        p5 = self.reduce_layers[0](c5)
        p4 = self.top_down_blocks[0](torch.cat([F.interpolate(p5, scale_factor=2.0, mode="nearest"), c4], 1))
        p4 = self.reduce_layers[1](p4)
        n3 = self.top_down_blocks[1](torch.cat([F.interpolate(p4, scale_factor=2.0, mode="nearest"), c3], 1))
        n4 = self.bottom_up_blocks[0](torch.cat([self.downsamples[0](n3), p4], 1))
        n5 = self.bottom_up_blocks[1](torch.cat([self.downsamples[1](n4), p5], 1))
        return self.out_convs[0](n3), self.out_convs[1](n4), self.out_convs[2](n5)


class Head(nn.Module):
    """`RTMDetSepBNHead` without objectness: class logits and box distances per level."""

    def __init__(self, a: Arch) -> None:
        """Two 3 x 3 convs and a 1 x 1 prediction conv per branch and level."""
        super().__init__()
        c = widths(a)[0]

        def tower() -> nn.Sequential:
            return nn.Sequential(*(conv3(c, c, depthwise=a.depthwise) for _ in range(STACKED)))

        self.cls_convs = nn.ModuleList([tower() for _ in STRIDES])
        self.reg_convs = nn.ModuleList([tower() for _ in STRIDES])
        self.rtm_cls = nn.ModuleList([nn.Conv2d(c, a.num_classes, 1) for _ in STRIDES])
        self.rtm_reg = nn.ModuleList([nn.Conv2d(c, 4, 1) for _ in STRIDES])

    def forward(self, feats: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """(class logits `[B, classes, h, w]`, raw box output `[B, 4, h, w]`) of level 0, 1, 2, flat."""
        out: list[torch.Tensor] = []
        for i, x in enumerate(feats):
            out.append(self.rtm_cls[i](self.cls_convs[i](x)))
            out.append(self.rtm_reg[i](self.reg_convs[i](x)))
        return tuple(out)


class RTMDet(nn.Module):
    """Backbone, neck and head: a normalised RGB image -> the six head maps."""

    def __init__(self, a: Arch) -> None:
        """Layers of the variant `a`."""
        super().__init__()
        self.backbone = CSPNeXt(a)
        self.neck = PAFPN(a)
        self.bbox_head = Head(a)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """`[B, 3, H, W]` (H, W multiples of 32) -> (class logits, raw box output) per level, flat."""
        return self.bbox_head(self.neck(*self.backbone(x)))


def points(h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The cells' points `[N, 2]` (x, y; network pixels) and strides `[N]` of an `h x w` input, level by level.

    mmdet's `MlvlPointGenerator` with offset 0: the top-left corner of each cell, rows first.
    """
    pts, strides = [], []
    for s in STRIDES:
        ys, xs = torch.meshgrid(
            torch.arange(h // s, device=device, dtype=torch.float32) * s,
            torch.arange(w // s, device=device, dtype=torch.float32) * s,
            indexing="ij",
        )
        pts.append(torch.stack((xs.reshape(-1), ys.reshape(-1)), -1))
        strides.append(torch.full((pts[-1].shape[0],), float(s), device=device))
    return torch.cat(pts), torch.cat(strides)


def decode(
    maps: tuple[torch.Tensor, ...], pts: torch.Tensor, strides: torch.Tensor, *, exp_on_reg: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """The head maps as float32 class logits `[B, N, classes]` and boxes `[B, N, 4]` (x0, y0, x1, y1; not clamped)."""
    logits = torch.cat([m.float().flatten(2) for m in maps[0::2]], 2).transpose(1, 2)
    dist = torch.cat([m.float().flatten(2) for m in maps[1::2]], 2).transpose(1, 2)
    dist = (dist.exp() if exp_on_reg else dist) * strides[:, None]
    return logits, torch.cat((pts - dist[..., :2], pts + dist[..., 2:]), -1)


def convert(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The mmdet keys as `RTMDet` keys: every `ConvModule`'s BatchNorm folded into its conv (weight and bias).

    The first conv's input channels are flipped (the reference feeds BGR). Some checkpoints also save the
    preprocessor's mean / std buffers (the runner's constants); they are dropped.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if ".bn." in k or k.startswith("data_preprocessor."):
            continue
        if k.endswith(".conv.weight"):
            p = k[: -len(".conv.weight")]
            std = (sd[f"{p}.bn.running_var"].double() + BN_EPS).sqrt()
            g = sd[f"{p}.bn.weight"].double() / std
            out[k] = (v.double() * g.view(-1, 1, 1, 1)).float()
            out[f"{p}.conv.bias"] = (sd[f"{p}.bn.bias"].double() - sd[f"{p}.bn.running_mean"].double() * g).float()
        else:
            out[k] = v
    out["backbone.stem.0.conv.weight"] = out["backbone.stem.0.conv.weight"].flip(1)
    return out
