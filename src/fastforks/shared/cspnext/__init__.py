"""CSPNeXt (mmdetection 3.3.0, arch P5), the backbone of RTMDet and RTMPose, inference only.

A 3-conv stem (stride 2), then four stages, each a 3 x 3 stride-2 conv and a `CSPLayer`; the last stage has an SPP
bottleneck (max pools 5, 9, 13) before it. A `CSPLayer` splits into a main branch (1 x 1 conv, `CSPNeXtBlock`s: 3 x 3
conv, then 5 x 5 depthwise + 1 x 1 pointwise, residual) and a 1 x 1 shortcut, concatenates them, applies channel
attention (global average pool, 1 x 1 conv with bias, hard sigmoid, multiply; backbone only) and a final 1 x 1 conv.

Every conv is followed by BatchNorm (eps 1e-5) and SiLU; `fold_bn` folds the BatchNorm of a reference state dict into
the convs, so the modules here have none. Depth and width scale with `deepen` / `widen`; `depthwise` (RTMDet nano)
replaces the 3 x 3 convs outside the stem by depthwise + 1 x 1. The module names are mmdet's.
"""

from __future__ import annotations

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


class CSPNeXt(nn.Module):
    """The backbone: stem + 4 stages; the maps of the last three stages."""

    def __init__(self, deepen: float, widen: float, *, depthwise: bool = False) -> None:
        """Layers with `deepen` x the blocks per stage and `widen` x the channels."""
        super().__init__()
        w = widen
        c0 = int(STAGES[0][0] * w // 2)
        self.stem = nn.Sequential(ConvAct(3, c0, 3, 2), ConvAct(c0, c0, 3), ConvAct(c0, int(STAGES[0][0] * w), 3))
        stages: list[nn.Sequential] = []
        for i, (cin, cout, n, identity) in enumerate(STAGES):
            ci, co = int(cin * w), int(cout * w)
            layers: list[nn.Module] = [conv3(ci, co, 2, depthwise=depthwise)]
            if i == len(STAGES) - 1:
                layers.append(SPP(co))
            csp = CSPLayer(co, co, blocks_of(n, deepen), identity=identity, attention=True, depthwise=depthwise)
            layers.append(csp)
            stages.append(nn.Sequential(*layers))
        self.stage1, self.stage2, self.stage3, self.stage4 = stages  # the mmdet names

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The stride 8, 16 and 32 maps."""
        x = self.stage1(self.stem(x))
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        return c3, c4, self.stage4(c4)


def fold_bn(sd: dict[str, torch.Tensor], eps: float = BN_EPS) -> dict[str, torch.Tensor]:
    """A reference state dict with every `ConvModule`'s BatchNorm (eps `eps`) folded into its conv (weight, bias)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if ".bn." in k:
            continue
        if k.endswith(".conv.weight"):
            p = k[: -len(".conv.weight")]
            std = (sd[f"{p}.bn.running_var"].double() + eps).sqrt()
            g = sd[f"{p}.bn.weight"].double() / std
            out[k] = (v.double() * g.view(-1, 1, 1, 1)).float()
            out[f"{p}.conv.bias"] = (sd[f"{p}.bn.bias"].double() - sd[f"{p}.bn.running_mean"].double() * g).float()
        else:
            out[k] = v
    return out
