# Copyright 2020 The MMSegmentation Authors (Apache-2.0, see LICENSE in this directory); condensed from
# mmsegmentation v1.2.2 (mmseg/models/backbones/{resnet,hrnet}.py, mmseg/models/decode_heads/{fcn,psp,aspp,
# sep_aspp}_head.py, mmseg/models/utils/res_layer.py) and mmcv 2.1.0 (ConvModule, DepthwiseSeparableConvModule):
# inference only, BatchNorm + ReLU fixed, no mmcv / mmengine.
"""The mmsegmentation CNN classics as plain PyTorch modules, state-dict compatible with the authors' checkpoints.

- `ResNetV1c`: ResNet with the 3-conv deep stem; `-D8`: output stride 8 (stages 3 / 4 dilated 2 / 4, the first block
  of each dilated stage at half the dilation, `contract_dilation`).
- `HRNet`: the high-resolution network (four parallel branches, repeated multi-resolution fusion).
- Decode heads `FCNHead`, `PSPHead` (pyramid pooling), `ASPPHead` (DeepLabV3), `DepthwiseSeparableASPPHead`
  (DeepLabV3+, with the stride-4 skip); the auxiliary heads are training-only and not built.

`EncoderDecoder(backbone, decode_head)` maps a normalised image to the head's logits at its own (1/4 or 1/8)
resolution, as mmseg's `extract_feat` + `decode_head.forward`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

BN_EPS = 1e-5  # fixed: mmcv BatchNorm default


def resize(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Mmseg `resize`: bilinear, `align_corners=False` (all heads here)."""
    return F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False)


def cat(ts: list[torch.Tensor]) -> torch.Tensor:
    """`torch.cat` along channels, channels-last if the largest input is.

    The pooled branches (image pooling, PSP bins) come back NCHW from their 1 x 1 .. 6 x 6 maps; mixed inputs give an
    NCHW concatenation, copied back to channels-last by the next conv (DeepLabV3+ 512 x 512: 0.25 ms).
    """
    big = max(ts, key=torch.Tensor.numel)
    if not big.is_contiguous() and big.is_contiguous(memory_format=torch.channels_last):
        ts = [t.contiguous(memory_format=torch.channels_last) for t in ts]
    return torch.cat(ts, 1)


def _bins(n: int, s: int, like: torch.Tensor) -> torch.Tensor:
    """`[s, n]` averaging matrix of adaptive pooling: bin `i` covers `floor(i n / s)` .. `ceil((i + 1) n / s)`."""
    m = torch.zeros(s, n, dtype=like.dtype)
    for i in range(s):
        a, b = i * n // s, -(-(i + 1) * n // s)
        m[i, a:b] = 1 / (b - a)
    return m.to(like.device)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """`nn.AdaptiveAvgPool2d(s)`; on MPS, sizes `s` does not divide as the same bin averages by matrix products."""

    def __init__(self, s: int) -> None:
        """Output `s x s`."""
        super().__init__(s)
        self.s = s

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - the parent's argument name
        """`[N, C, H, W]` -> `[N, C, s, s]`."""
        s, (h, w) = self.s, input.shape[-2:]
        if input.device.type != "mps" or (h % s == 0 and w % s == 0):
            return super().forward(input)
        return _bins(h, s, input) @ input @ _bins(w, s, input).T  # MPS implements divisible sizes only (pytorch#96056)


class ConvModule(nn.Module):
    """mmcv `ConvModule` with `norm_cfg=BN`: conv (no bias) -> BatchNorm -> ReLU (`act=False`: no ReLU)."""

    def __init__(  # noqa: PLR0913 - the conv geometry
        self,
        cin: int,
        cout: int,
        k: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        """Conv `cin -> cout`, kernel `k`."""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=BN_EPS)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv -> BN [-> ReLU]."""
        y = self.bn(self.conv(x))
        return F.relu(y) if self.act else y


class DepthwiseSeparableConvModule(nn.Module):
    """mmcv `DepthwiseSeparableConvModule`: depthwise `ConvModule` then pointwise `ConvModule` (both BN + ReLU)."""

    def __init__(self, cin: int, cout: int, k: int, *, padding: int = 0, dilation: int = 1) -> None:
        """Depthwise `k x k` (dilated) on `cin`, then 1x1 to `cout`."""
        super().__init__()
        self.depthwise_conv = ConvModule(cin, cin, k, padding=padding, dilation=dilation, groups=cin)
        self.pointwise_conv = ConvModule(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise -> pointwise."""
        return self.pointwise_conv(self.depthwise_conv(x))


# ------------------------------------------------------------------------------------------------ ResNet / HRNet
class BasicBlock(nn.Module):
    """ResNet basic block: 3x3 -> BN -> ReLU -> 3x3 -> BN, plus the shortcut, ReLU."""

    expansion = 1

    def __init__(self, cin: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        """Block `cin -> planes`."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=BN_EPS)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=BN_EPS)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block."""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + (x if self.downsample is None else self.downsample(x)))


class Bottleneck(nn.Module):
    """ResNet bottleneck (`style="pytorch"`: the stride on the 3x3): 1x1 -> 3x3 (dilated) -> 1x1, plus the shortcut."""

    expansion = 4

    def __init__(
        self, cin: int, planes: int, stride: int = 1, downsample: nn.Module | None = None, dilation: int = 1
    ) -> None:
        """Block `cin -> 4 * planes`."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=BN_EPS)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=BN_EPS)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=BN_EPS)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block."""
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return F.relu(y + (x if self.downsample is None else self.downsample(x)))


def _shortcut(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout, eps=BN_EPS))


def res_layer(  # noqa: PLR0913, PLR0917 - mmseg ResLayer arguments
    block: type[BasicBlock | Bottleneck],
    cin: int,
    planes: int,
    blocks: int,
    stride: int = 1,
    dilation: int = 1,
    *,
    contract_dilation: bool = False,
) -> nn.Sequential:
    """Mmseg `ResLayer`: `blocks` blocks, the first one strided (and at half the dilation with `contract_dilation`)."""
    cout = planes * block.expansion
    down = _shortcut(cin, cout, stride) if stride != 1 or cin != cout else None
    first = dilation // 2 if dilation > 1 and contract_dilation else dilation
    layers: list[nn.Module] = [
        Bottleneck(cin, planes, stride, down, first) if block is Bottleneck else BasicBlock(cin, planes, stride, down)
    ]
    layers.extend(
        Bottleneck(cout, planes, 1, None, dilation) if block is Bottleneck else BasicBlock(cout, planes)
        for _ in range(1, blocks)
    )
    return nn.Sequential(*layers)


class ResNetV1c(nn.Module):
    """ResNet-V1c (deep stem: three 3x3 convs); returns the four stage outputs."""

    DEPTHS: ClassVar[dict[int, tuple[int, ...]]] = {
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
    }  # fixed: torchvision / mmseg stage depths

    def __init__(
        self, depth: int = 101, strides: Sequence[int] = (1, 2, 1, 1), dilations: Sequence[int] = (1, 1, 2, 4)
    ) -> None:
        """Mmseg `ResNetV1c(depth, strides, dilations, contract_dilation=True)`."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32, eps=BN_EPS),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32, eps=BN_EPS),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64, eps=BN_EPS),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        cin = 64
        self.layers: list[str] = []
        for i, (n, s, d) in enumerate(zip(self.DEPTHS[depth], strides, dilations, strict=True)):
            planes = 64 * 2**i
            self.add_module(f"layer{i + 1}", res_layer(Bottleneck, cin, planes, n, s, d, contract_dilation=True))
            self.layers.append(f"layer{i + 1}")
            cin = planes * Bottleneck.expansion

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Image -> the four stage outputs (strides 4, 8, 8, 8 for D8)."""
        x = self.maxpool(self.stem(x))
        outs = []
        for name in self.layers:
            x = getattr(self, name)(x)
            outs.append(x)
        return outs


class _Upsample(nn.Module):
    """mmseg `Upsample(scale_factor, bilinear)`: output size `int(size * scale)`."""

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resize(x, [int(t * float(self.scale)) for t in x.shape[-2:]])


class HRModule(nn.Module):
    """One HRNet module: per-branch residual blocks, then every output branch sums all branches (resampled), ReLU."""

    def __init__(self, block: type[BasicBlock | Bottleneck], blocks: Sequence[int], channels: list[int]) -> None:
        """Branches of `channels` (in = out), `blocks` blocks each, multi-scale output."""
        super().__init__()
        nb = len(channels)
        self.branches = nn.ModuleList(
            nn.Sequential(*[block(c, c) for _ in range(n)]) for n, c in zip(blocks, channels, strict=True)
        )
        self.fuse_layers: nn.ModuleList | None = None
        if nb == 1:
            return
        fuse = []
        for i in range(nb):
            row: list[nn.Module | None] = []
            for j in range(nb):
                if j > i:
                    row.append(
                        nn.Sequential(
                            nn.Conv2d(channels[j], channels[i], 1, bias=False),
                            nn.BatchNorm2d(channels[i], eps=BN_EPS),
                            _Upsample(2 ** (j - i)),
                        )
                    )
                elif j == i:
                    row.append(None)
                else:
                    downs: list[nn.Module] = []
                    for k in range(i - j):
                        last = k == i - j - 1
                        cout = channels[i] if last else channels[j]
                        parts: list[nn.Module] = [
                            nn.Conv2d(channels[j], cout, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(cout, eps=BN_EPS),
                        ]
                        if not last:
                            parts.append(nn.ReLU())
                        downs.append(nn.Sequential(*parts))
                    row.append(nn.Sequential(*downs))
            fuse.append(nn.ModuleList(row))  # ty: ignore[invalid-argument-type] - None slots, as mmseg
        self.fuse_layers = nn.ModuleList(fuse)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Branch features -> fused branch features."""
        x = [br(xi) for br, xi in zip(self.branches, x, strict=True)]
        if self.fuse_layers is None:
            return x
        out = []
        for i, row in enumerate(self.fuse_layers):
            y: torch.Tensor | None = None
            for j, f in enumerate(row):  # ty: ignore[invalid-argument-type] - rows are ModuleLists
                if j == i:
                    t = x[j]
                elif j > i:
                    t = resize(f(x[j]), x[i].shape[2:])
                else:
                    t = f(x[j])
                y = t if y is None else y + t
            assert y is not None  # noqa: S101 - at least one branch
            out.append(F.relu(y))
        return out


def _transition(pre: list[int], cur: list[int]) -> nn.ModuleList:
    """HRNet transition: 3x3 convs where a branch's width changes, strided 3x3 convs for each new branch."""
    layers: list[nn.Module | None] = []
    for i, c in enumerate(cur):
        if i < len(pre):
            layers.append(
                None
                if c == pre[i]
                else nn.Sequential(nn.Conv2d(pre[i], c, 3, 1, 1, bias=False), nn.BatchNorm2d(c, eps=BN_EPS), nn.ReLU())
            )
            continue
        downs = []
        for j in range(i + 1 - len(pre)):
            cin = pre[-1]
            cout = c if j == i - len(pre) else cin
            downs.append(
                nn.Sequential(nn.Conv2d(cin, cout, 3, 2, 1, bias=False), nn.BatchNorm2d(cout, eps=BN_EPS), nn.ReLU())
            )
        layers.append(nn.Sequential(*downs))
    return nn.ModuleList(layers)  # ty: ignore[invalid-argument-type] - None slots, as mmseg


class HRNet(nn.Module):
    """HRNet-W`width` (the mmseg `fcn_hr*` configs): stem, a bottleneck stage, then stages of 2 / 3 / 4 branches."""

    def __init__(self, width: int = 48) -> None:
        """Branch widths `width * (1, 2, 4, 8)`; 1 / 4 / 3 modules of 4 basic blocks in stages 2 / 3 / 4."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=BN_EPS)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=BN_EPS)
        self.layer1 = res_layer(Bottleneck, 64, 64, 4)
        widths = [width * 2**i for i in range(4)]
        pre = [256]
        for s, modules in ((2, 1), (3, 4), (4, 3)):
            cur = widths[:s]
            self.add_module(f"transition{s - 1}", _transition(pre, cur))
            self.add_module(f"stage{s}", nn.Sequential(*[HRModule(BasicBlock, [4] * s, cur) for _ in range(modules)]))
            pre = cur

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Image -> four branch outputs (strides 4, 8, 16, 32)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        ys = [self.layer1(x)]
        for s in (2, 3, 4):
            trans = getattr(self, f"transition{s - 1}")
            xs = [ys[i] if t is None else t(ys[-1] if i >= len(ys) - 1 else ys[i]) for i, t in enumerate(trans)]
            ys = getattr(self, f"stage{s}")(xs)
        return ys


# ------------------------------------------------------------------------------------------------ decode heads
class FCNHead(nn.Module):
    """mmseg `FCNHead`: `num_convs` ConvModules [+ concat with the input and one more], 1x1 classifier.

    `in_index` a list: mmseg's `resize_concat` (all selected inputs resized to the first one's size, concatenated).
    """

    def __init__(  # noqa: PLR0913 - mmseg FCNHead arguments
        self,
        cin: int,
        channels: int,
        classes: int,
        *,
        num_convs: int = 2,
        kernel_size: int = 3,
        concat_input: bool = True,
        in_index: int | list[int] = 3,
    ) -> None:
        """Head on `cin` channels."""
        super().__init__()
        self.in_index = in_index
        p = kernel_size // 2
        self.convs = nn.Sequential(
            *[ConvModule(cin if i == 0 else channels, channels, kernel_size, padding=p) for i in range(num_convs)]
        )
        self.conv_cat = ConvModule(cin + channels, channels, kernel_size, padding=p) if concat_input else None
        self.conv_seg = nn.Conv2d(channels, classes, 1)

    def inputs(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """The selected (resized, concatenated) input."""
        if isinstance(self.in_index, list):
            xs = [feats[i] for i in self.in_index]
            return torch.cat([xs[0], *[resize(t, xs[0].shape[2:]) for t in xs[1:]]], 1)
        return feats[self.in_index]

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Backbone features -> logits."""
        x = self.inputs(feats)
        y = self.convs(x)
        if self.conv_cat is not None:
            y = self.conv_cat(torch.cat([x, y], 1))
        return self.conv_seg(y)


class PSPHead(nn.Module):
    """mmseg `PSPHead`: pyramid pooling (adaptive average pools, 1x1 ConvModules, upsampled), 3x3 bottleneck."""

    def __init__(self, cin: int, channels: int, classes: int, pool_scales: Sequence[int] = (1, 2, 3, 6)) -> None:
        """Head on stage 4 (`cin` channels)."""
        super().__init__()
        self.psp_modules = nn.ModuleList(
            nn.Sequential(AdaptiveAvgPool2d(s), ConvModule(cin, channels, 1)) for s in pool_scales
        )
        self.bottleneck = ConvModule(cin + len(pool_scales) * channels, channels, 3, padding=1)
        self.conv_seg = nn.Conv2d(channels, classes, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Backbone features -> logits."""
        x = feats[3]
        outs = [x, *[resize(m(x), x.shape[2:]) for m in self.psp_modules]]
        return self.conv_seg(self.bottleneck(cat(outs)))


class ASPPHead(nn.Module):
    """mmseg `ASPPHead` (DeepLabV3): image pooling + 1x1 + dilated 3x3 ConvModules, concatenated, 3x3 bottleneck."""

    def __init__(self, cin: int, channels: int, classes: int, dilations: Sequence[int] = (1, 12, 24, 36)) -> None:
        """Head on stage 4 (`cin` channels)."""
        super().__init__()
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvModule(cin, channels, 1))
        self.aspp_modules = nn.ModuleList(
            ConvModule(cin, channels, 1) if d == 1 else ConvModule(cin, channels, 3, padding=d, dilation=d)
            for d in dilations
        )
        self.bottleneck = ConvModule((len(dilations) + 1) * channels, channels, 3, padding=1)
        self.conv_seg = nn.Conv2d(channels, classes, 1)

    def aspp(self, x: torch.Tensor) -> torch.Tensor:
        """The bottleneck output of the ASPP branches."""
        outs = [resize(self.image_pool(x), x.shape[2:]), *[m(x) for m in self.aspp_modules]]
        return self.bottleneck(cat(outs))

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Backbone features -> logits."""
        return self.conv_seg(self.aspp(feats[3]))


class DepthwiseSeparableASPPHead(ASPPHead):
    """mmseg `DepthwiseSeparableASPPHead` (DeepLabV3+): ASPP with separable dilated branches, stride-4 skip decoder."""

    def __init__(  # noqa: PLR0913, PLR0917 - mmseg head arguments
        self,
        cin: int,
        channels: int,
        classes: int,
        dilations: Sequence[int] = (1, 12, 24, 36),
        c1_in: int = 256,
        c1_channels: int = 48,
    ) -> None:
        """Head on stage 4 (`cin` channels) with the stage-1 skip (`c1_in` channels)."""
        super().__init__(cin, channels, classes, dilations)
        self.aspp_modules = nn.ModuleList(
            ConvModule(cin, channels, 1)
            if d == 1
            else DepthwiseSeparableConvModule(cin, channels, 3, padding=d, dilation=d)
            for d in dilations
        )
        self.c1_bottleneck = ConvModule(c1_in, c1_channels, 1)
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(channels + c1_channels, channels, 3, padding=1),
            DepthwiseSeparableConvModule(channels, channels, 3, padding=1),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Backbone features -> logits (stride 4)."""
        y = self.aspp(feats[3])
        c1 = self.c1_bottleneck(feats[0])
        y = torch.cat([resize(y, c1.shape[2:]), c1], 1)
        return self.conv_seg(self.sep_bottleneck(y))


class EncoderDecoder(nn.Module):
    """mmseg `EncoderDecoder` at inference: backbone -> decode head (state-dict keys `backbone.*`, `decode_head.*`)."""

    def __init__(self, backbone: nn.Module, decode_head: nn.Module) -> None:
        """Wrap the two parts."""
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised image -> logits at the head's resolution."""
        return self.decode_head(self.backbone(x))


def build(arch: str, classes: int = 150) -> EncoderDecoder:
    """The network of a variant's `arch`: `fcn-r101-d8`, `psp-r101-d8`, `aspp-r101-d8`, `sepaspp-r101-d8`, `fcn-hr48`.

    Raises:
        ValueError: Unknown `arch`.
    """
    head, _, bb = arch.partition("-")
    if bb == "hr48":
        return EncoderDecoder(
            HRNet(48),
            FCNHead(720, 720, classes, num_convs=1, kernel_size=1, concat_input=False, in_index=[0, 1, 2, 3]),
        )
    if bb != "r101-d8":
        msg = f"unknown mmseg arch {arch!r}"
        raise ValueError(msg)
    heads: dict[str, nn.Module] = {
        "fcn": FCNHead(2048, 512, classes),
        "psp": PSPHead(2048, 512, classes),
        "aspp": ASPPHead(2048, 512, classes),
        "sepaspp": DepthwiseSeparableASPPHead(2048, 512, classes),
    }
    if head not in heads:
        msg = f"unknown mmseg arch {arch!r}"
        raise ValueError(msg)
    return EncoderDecoder(ResNetV1c(101), heads[head])
