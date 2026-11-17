"""ReCoVEr network (Kiefhaber, Roth & Schaub-Meyer, ICCV 2025) as visinf/ReCoVEr builds it (`models/recover.py`).

Inference only, as upstream runs it (`disable_cost=True`, its default): no feature encoder and no cost volume.

Ported from visinf/ReCoVEr (BSD-3-Clause, see `LICENSE`) with the same module tree, so the released checkpoints load
as they are (their unused `fnet.*` tensors are dropped). Changes: no torchvision import (upstream builds its
extractors from torchvision blocks and loads ImageNet weights at construction, overwritten by the checkpoint); the
input is already scaled to [-1, 1] and padded (the core does both, as the card's `[io]` says); the flow is upsampled
(and the upsampling weights predicted) after the last iteration only, which is upstream's `final`; the uncertainty
(`info`) channels of the flow head are not computed.

- Context encoder on both images stacked as 6 channels (ResNet34 stages for RN, MobileNetV3-Large for MN,
  ConvNeXt-T for CX; stride 8), then a 3 x 3 conv: the initial hidden state and the context features. The flow head
  on the hidden state gives the initial flow.
- Update block (SEA-RAFT's): motion encoder (correlation + flow convs), two ConvNeXt blocks on
  [hidden | context | motion]; the flow head adds a delta flow at 1/8 resolution. Upstream feeds an all-zero
  correlation, so the correlation branch of the motion encoder is the same map every call: `corr_term` computes it
  on a 5 x 5 canvas (border rows / columns and the interior) and expands it to the feature size, which is exact.
- Upsampling: convex combination of 3 x 3 neighbours with predicted weights.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

FACTOR = 8  # fixed: the context encoders downsample by 8
DIM = 128  # fixed: hidden state and context channels (upstream args.dim)
CORR = 4 * 9 * 9  # fixed: upstream's correlation channels (4 levels, radius 4), all zero at inference
Arch = Literal["rn", "mn", "cx"]


# --- ResNet34 stages (upstream models/resnet.py ResNetFPN, models/layer.py BasicBlock)


class BasicBlock(nn.Module):
    """Two 3 x 3 conv-bn-relu with a residual (1 x 1 conv-bn projection when the shape changes) and a final relu."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        """Build the block."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Sequential | None = None
        if stride != 1 or cin != cout:
            self.bn3 = nn.BatchNorm2d(cout)  # upstream: the same module as downsample.1 (saved twice)
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride), self.bn3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block."""
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class ResNetFPN(nn.Module):
    """7 x 7 stride-2 stem (conv-bn-relu), ResNet34 stages (3, 4, 6 blocks; strides 1, 2, 2), 1 x 1 output conv."""

    def __init__(self, cin: int, cout: int) -> None:
        """Build the encoder."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        dims = (64, 64, 128, 256)
        self.layer1, self.layer2, self.layer3 = (
            nn.Sequential(*(BasicBlock(dims[i] if j == 0 else dims[i + 1], dims[i + 1], s if j == 0 else 1)
                            for j in range(n)))
            for i, (n, s) in enumerate(zip((3, 4, 6), (1, 2, 2), strict=True))
        )  # fmt: skip
        self.final_conv = nn.Conv2d(256, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, cin, H, W]` -> `[B, cout, H / 8, W / 8]`."""
        x = self.relu(self.bn1(self.conv1(x)))
        return self.final_conv(self.layer3(self.layer2(self.layer1(x))))


# --- MobileNetV3-Large (upstream models/mobilenet.py, torchvision's InvertedResidual)


def _divisible(v: float, divisor: int = 8) -> int:
    """The `_make_divisible` of torchvision."""
    out = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return out + divisor if out < 0.9 * v else out


def _cna(  # noqa: PLR0913, PLR0917 - torchvision's signature
    cin: int, cout: int, k: int, stride: int = 1, groups: int = 1, act: type[nn.Module] | None = None
) -> nn.Sequential:
    """The `Conv2dNormActivation` of torchvision: conv (no bias), batch norm (eps 1e-3), activation."""
    layers: list[nn.Module] = [nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, groups=groups, bias=False),
                               nn.BatchNorm2d(cout, eps=0.001)]  # fmt: skip
    if act is not None:
        layers.append(act())
    return nn.Sequential(*layers)


class SqueezeExcitation(nn.Module):
    """torchvision SE: global mean, 1 x 1 conv-relu-conv, hard sigmoid scale."""

    def __init__(self, c: int, squeeze: int) -> None:
        """Build the gate."""
        super().__init__()
        self.fc1 = nn.Conv2d(c, squeeze, 1)
        self.fc2 = nn.Conv2d(squeeze, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Channel-scaled `x`."""
        s = F.relu(self.fc1(x.mean((2, 3), keepdim=True)))
        return x * F.hardsigmoid(self.fc2(s))


class InvertedResidual(nn.Module):
    """torchvision MobileNetV3 block: expand (1 x 1), depthwise k x k, optional SE, project; residual at stride 1."""

    def __init__(self, cin: int, k: int, exp: int, cout: int, se: bool, hs: bool, stride: int) -> None:  # noqa: FBT001, PLR0913, PLR0917
        """Build the block (`hs`: hard swish, else relu)."""
        super().__init__()
        act = nn.Hardswish if hs else nn.ReLU
        layers: list[nn.Module] = []
        if exp != cin:
            layers.append(_cna(cin, exp, 1, act=act))
        layers.append(_cna(exp, exp, k, stride, groups=exp, act=act))
        if se:
            layers.append(SqueezeExcitation(exp, _divisible(exp // 4)))
        layers.append(_cna(exp, cout, 1))
        self.block = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Block output."""
        y = self.block(x)
        return x + y if self.residual else y


MOBILENET_L = (
    (16, 3, 16, 16, False, False, 1), (16, 3, 64, 24, False, False, 2), (24, 3, 72, 24, False, False, 1),
    (24, 5, 72, 40, True, False, 2), (40, 5, 120, 40, True, False, 1), (40, 5, 120, 40, True, False, 1),
    (40, 3, 240, 80, False, True, 1), (80, 3, 200, 80, False, True, 1), (80, 3, 184, 80, False, True, 1),
    (80, 3, 184, 80, False, True, 1), (80, 3, 480, 112, True, True, 1), (112, 3, 672, 112, True, True, 1),
    (112, 5, 672, 160, True, True, 1), (160, 5, 960, 160, True, True, 1), (160, 5, 960, 160, True, True, 1),
)  # fmt: skip  # fixed: upstream size "l" (cin, kernel, expanded, cout, SE, hard swish, stride)


class MobileNetV3(nn.Module):
    """MobileNetV3-Large features (stride 8: the 6-channel stem and two stride-2 blocks) + a 1 x 1 conv to `cout`."""

    def __init__(self, cin: int, cout: int) -> None:
        """Build the encoder."""
        super().__init__()
        layers: list[nn.Module] = [_cna(cin, 16, 3, 2, act=nn.Hardswish)]
        layers += [InvertedResidual(*c) for c in MOBILENET_L]
        layers.append(_cna(160, 960, 1, act=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.final = nn.Conv2d(960, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, cin, H, W]` -> `[B, cout, H / 8, W / 8]`."""
        return self.final(self.features(x))


# --- ConvNeXt-T (upstream models/convnext.py, torchvision's CNBlock)


class LayerNorm2d(nn.LayerNorm):
    """torchvision `LayerNorm2d`: LayerNorm over the channels of `[B, C, H, W]`."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - LayerNorm.forward's parameter name
        """Normalised `input`."""
        y = F.layer_norm(input.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2)


class _Permute(nn.Module):
    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class CNBlock(nn.Module):
    """torchvision ConvNeXt block: 7 x 7 depthwise conv, LayerNorm, Linear-GELU-Linear, layer scale, residual."""

    def __init__(self, c: int) -> None:
        """Build the block."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 7, padding=3, groups=c), _Permute(0, 2, 3, 1), nn.LayerNorm(c, eps=1e-6),
            nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c), _Permute(0, 3, 1, 2),
        )  # fmt: skip
        self.layer_scale = nn.Parameter(torch.ones(c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block."""
        return x + self.layer_scale * self.block(x)


class ConvNeXt(nn.Module):
    """ConvNeXt-T stages (96, 192, 384, 768 channels; 3, 3, 9, 3 blocks) at stride 8.

    A stride-4 stem, one stride-2 downsampling, then upstream's stride-1 "downsamplings" (LayerNorm, zero pad right /
    bottom, 2 x 2 conv) between the stages and after the last one.
    """

    def __init__(self, cin: int, cout: int) -> None:
        """Build the encoder."""
        super().__init__()
        dims, depths = (96, 192, 384, 768), (3, 3, 9, 3)
        layers: list[nn.Module] = [nn.Sequential(nn.Conv2d(cin, 96, 4, 4), LayerNorm2d(96, eps=1e-6))]
        for i, (c, n) in enumerate(zip(dims, depths, strict=True)):
            layers.append(nn.Sequential(*(CNBlock(c) for _ in range(n))))
            nxt = dims[i + 1] if i < len(dims) - 1 else cout
            if i == 0:
                layers.append(nn.Sequential(LayerNorm2d(c, eps=1e-6), nn.Conv2d(c, nxt, 2, 2)))
            else:
                layers.append(nn.Sequential(LayerNorm2d(c, eps=1e-6), nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(c, nxt, 2)))
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, cin, H, W]` -> `[B, cout, H / 8, W / 8]`."""
        return self.features(x)


# --- update block (upstream models/update.py, models/layer.py ConvNextBlock: SEA-RAFT's)


class LayerNorm(nn.Module):
    """LayerNorm over the last (channel) dimension, with upstream's parameter names."""

    def __init__(self, dim: int, eps: float) -> None:
        """Build the norm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised `[..., C]`."""
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class ConvNextBlock(nn.Module):
    """7 x 7 depthwise conv, LayerNorm, Linear-GELU-Linear MLP, layer scale, residual, then a 1 x 1 conv to `cout`."""

    def __init__(self, dim: int, cout: int) -> None:
        """Build the block."""
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * cout)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * cout, dim)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.final = nn.Conv2d(dim, cout, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, dim, h, w]` -> `[B, cout, h, w]`."""
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.gamma * self.pwconv2(self.act(self.pwconv1(self.norm(y))))
        return self.final(x + y.permute(0, 3, 1, 2))


class BasicMotionEncoder(nn.Module):
    """Encodes the (all-zero) correlation and the current flow; the flow is appended to the output."""

    def __init__(self, dim: int) -> None:
        """Build the encoder."""
        super().__init__()
        self.convc1 = nn.Conv2d(CORR, dim * 2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim * 2, dim + dim // 2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim * 2, dim - 2, 3, padding=1)

    def corr_term(self, h: int, w: int) -> torch.Tensor:
        """`conv`'s correlation half (and its bias) for a zero correlation: `[1, dim - 2, h, w]`, h and w >= 5.

        `convc1` of zeros is its bias everywhere; the two 3 x 3 convs after it only see the zero padding within two
        pixels of the border, so a 5 x 5 canvas holds every distinct value: rows / columns 0, 1 and the last two are
        the border ones, the middle one is the interior. The gather is exact (same arithmetic per pixel).
        """
        c1 = cast("torch.Tensor", self.convc1.bias)
        cor = F.relu(c1).view(1, -1, 1, 1).expand(1, c1.shape[0], 5, 5)
        cor = F.relu(self.convc2(cor))
        wc = self.conv.weight[:, : cor.shape[1]]
        small = F.conv2d(cor, wc, self.conv.bias, padding=1)
        iy = torch.tensor([0, 1] + [2] * (h - 4) + [3, 4], device=small.device)
        ix = torch.tensor([0, 1] + [2] * (w - 4) + [3, 4], device=small.device)
        return small.index_select(2, iy).index_select(3, ix)

    def forward(self, flow: torch.Tensor, corr_term: torch.Tensor) -> torch.Tensor:
        """Motion features `[B, dim, h, w]` from the flow and `corr_term`."""
        flo = F.relu(self.convf2(F.relu(self.convf1(flow))))
        wf = self.conv.weight[:, self.conv.weight.shape[1] - flo.shape[1] :]
        out = F.relu(F.conv2d(flo, wf, padding=1) + corr_term)
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    """Motion encoder, then ConvNeXt blocks on [hidden | context | motion] -> new hidden state."""

    def __init__(self, dim: int, num_blocks: int = 2) -> None:
        """Build the block."""
        super().__init__()
        self.encoder = BasicMotionEncoder(dim)
        self.refine = nn.ModuleList(ConvNextBlock(3 * dim, dim) for _ in range(num_blocks))

    def forward(
        self, net: torch.Tensor, inp: torch.Tensor, corr_term: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        """Updated hidden state."""
        inp = torch.cat([inp, self.encoder(flow, corr_term)], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net


def _upsample(flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Flow at 1/8 -> full resolution: convex combination of 3 x 3 neighbours."""
    b, c, h, w = flow.shape
    m = torch.softmax(mask.view(b, 1, 9, FACTOR, FACTOR, h, w), dim=2)
    up = F.unfold(FACTOR * flow, kernel_size=3, padding=1).view(b, c, 9, 1, 1, h, w)
    up = torch.sum(m * up, dim=2)
    return up.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * FACTOR, w * FACTOR)


class ReCoVEr(nn.Module):
    """ReCoVEr: `(image1, image2)` in `[-1, 1]`, H and W multiples of 8 -> flow `[B, 2, H, W]` (image1 to image2)."""

    def __init__(self, arch: Arch) -> None:
        """Build the network for one context encoder (`rn`, `mn`, `cx`)."""
        super().__init__()
        enc = {"rn": ResNetFPN, "mn": MobileNetV3, "cx": ConvNeXt}[arch]
        self.cnet = enc(6, 2 * DIM)
        self.init_conv = nn.Conv2d(2 * DIM, 2 * DIM, 3, padding=1)
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(DIM, DIM * 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(DIM * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(DIM, 2 * DIM, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(2 * DIM, 6, 3, padding=1)
        )
        self.update_block = BasicUpdateBlock(DIM)

    def delta(self, net: torch.Tensor) -> torch.Tensor:
        """The flow channels of the flow head (its 4 uncertainty channels are not computed)."""
        head, out = self.flow_head[0], cast("nn.Conv2d", self.flow_head[2])
        return F.conv2d(F.relu(head(net)), out.weight[:2], cast("torch.Tensor", out.bias)[:2], padding=1)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
        """Flow after `iters` updates, float32 `[B, 2, H, W]`; runs in the dtype of the weights."""
        dt = self.init_conv.weight.dtype
        x = torch.cat([image1, image2], dim=1).to(dt)
        net, context = self.init_conv(self.cnet(x)).split(DIM, dim=1)
        flow = self.delta(net).float()
        corr_term = self.update_block.encoder.corr_term(*net.shape[-2:])
        for _ in range(iters):
            net = self.update_block(net, context, corr_term, flow.to(dt))
            flow = flow + self.delta(net).float()
        mask = 0.25 * self.upsample_weight(net)
        return _upsample(flow, mask.float())
