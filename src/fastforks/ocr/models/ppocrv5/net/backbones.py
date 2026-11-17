"""PP-HGNetV2-B4 (server models) and PP-LCNetV3 (mobile models), as PaddleOCR's `rec_pphgnetv2.py` and
`rec_lcnetv3.py` in their exported (inference) form."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .layers import ConvBN, LearnableAffineBlock

# PP-HGNetV2-B4: in, mid, out, blocks, downsample, light block, kernel, layers, stride
HGNET_DET = (
    (48, 48, 128, 1, False, False, 3, 6, 2),
    (128, 96, 512, 1, True, False, 3, 6, 2),
    (512, 192, 1024, 3, True, True, 5, 6, 2),
    (1024, 384, 2048, 1, True, True, 5, 6, 2),
)
HGNET_REC = (
    (48, 48, 128, 1, True, False, 3, 6, (2, 1)),
    (128, 96, 512, 1, True, False, 3, 6, (1, 2)),
    (512, 192, 1024, 3, True, True, 5, 6, (2, 1)),
    (1024, 384, 2048, 1, True, True, 5, 6, (2, 1)),
)


class _LightConvBN(nn.Module):
    def __init__(self, cin: int, cout: int, k: int) -> None:
        super().__init__()
        self.conv1 = ConvBN(cin, cout, 1)
        self.conv2 = ConvBN(cout, cout, k, groups=cout, act=nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class _Stem(nn.Module):
    def __init__(self, cin: int, mid: int, cout: int, text_rec: bool) -> None:
        super().__init__()
        self.stem1 = ConvBN(cin, mid, 3, 2, act=nn.ReLU())
        self.stem2a = ConvBN(mid, mid // 2, 2, act=nn.ReLU(), padding=0)  # Paddle "SAME": pad right / bottom by 1
        self.stem2b = ConvBN(mid // 2, mid, 2, act=nn.ReLU(), padding=0)
        self.stem3 = ConvBN(mid * 2, mid, 3, 1 if text_rec else 2, act=nn.ReLU())
        self.stem4 = ConvBN(mid, cout, 1, act=nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x2 = self.stem2b(F.pad(self.stem2a(F.pad(x, (0, 1, 0, 1))), (0, 1, 0, 1)))
        x1 = F.max_pool2d(F.pad(x, (0, 1, 0, 1)), 2, 1)  # SAME, ceil; x >= 0 after ReLU, so zero padding is exact
        return self.stem4(self.stem3(torch.cat([x1, x2], 1)))


class _HGBlock(nn.Module):
    def __init__(self, cin: int, mid: int, cout: int, k: int, n: int, identity: bool, light: bool) -> None:
        super().__init__()
        self.identity = identity
        self.layers = nn.ModuleList(
            _LightConvBN(cin if i == 0 else mid, mid, k) if light else ConvBN(cin if i == 0 else mid, mid, k, act=nn.ReLU())
            for i in range(n)
        )
        self.squeeze = ConvBN(cin + n * mid, cout // 2, 1, act=nn.ReLU())
        self.excitation = ConvBN(cout // 2, cout, 1, act=nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x]
        y = x
        for layer in self.layers:
            y = layer(y)
            outs.append(y)
        y = self.excitation(self.squeeze(torch.cat(outs, 1)))
        return y + x if self.identity else y


class _HGStage(nn.Module):
    def __init__(
        self, cin: int, mid: int, cout: int, blocks: int, down: bool, light: bool, k: int, n: int, stride: int | tuple
    ) -> None:
        super().__init__()
        self.downsample = ConvBN(cin, cin, 3, stride, groups=cin) if down else nn.Identity()
        self.blocks = nn.Sequential(
            *(_HGBlock(cin if i == 0 else cout, mid, cout, k, n, i > 0, light) for i in range(blocks))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


class PPHGNetV2B4(nn.Module):
    """PP-HGNetV2-B4: `det` returns the 4 stage outputs (strides 4..32), else the recognition feature `[B, 2048, 1, W/8]`."""

    def __init__(self, det: bool) -> None:
        super().__init__()
        self.det = det
        self.stem = _Stem(3, 32, 48, text_rec=not det)
        self.stages = nn.ModuleList(_HGStage(*c) for c in (HGNET_DET if det else HGNET_REC))
        if not det:  # upstream creates (unused) fc layers here; the exported model drops them
            self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs if self.det else F.avg_pool2d(x, (3, 2))


def make_divisible(v: float, divisor: int = 16) -> int:
    """PaddleOCR's channel rounding."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


LCNET_DET = {  # k, in, out, stride, se
    "blocks2": ((3, 16, 32, 1, False),),
    "blocks3": ((3, 32, 64, 2, False), (3, 64, 64, 1, False)),
    "blocks4": ((3, 64, 128, 2, False), (3, 128, 128, 1, False)),
    "blocks5": ((3, 128, 256, 2, False), *((5, 256, 256, 1, False),) * 4),
    "blocks6": ((5, 256, 512, 2, True), (5, 512, 512, 1, True), (5, 512, 512, 1, False), (5, 512, 512, 1, False)),
}
LCNET_REC = {
    "blocks2": ((3, 16, 32, 1, False),),
    "blocks3": ((3, 32, 64, 1, False), (3, 64, 64, 1, False)),
    "blocks4": ((3, 64, 128, (2, 1), False), (3, 128, 128, 1, False)),
    "blocks5": ((3, 128, 256, (1, 2), False), *((5, 256, 256, 1, False),) * 4),
    "blocks6": (
        (5, 256, 512, (2, 1), True),
        (5, 512, 512, 1, True),
        (5, 512, 512, (2, 1), False),
        (5, 512, 512, 1, False),
    ),
}


class _RepLayer(nn.Module):
    """Re-parameterised `LearnableRepLayer`: conv (with bias) -> LAB (-> hardswish -> LAB unless stride 2)."""

    def __init__(self, cin: int, cout: int, k: int, stride: int | tuple, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, groups=groups)
        self.conv.late = True  # created by Paddle's export (`rep`) after the network
        self.lab = LearnableAffineBlock()
        self.act = stride != 2  # a tuple stride (2, 1) != 2: upstream applies the activation
        self.act_lab = LearnableAffineBlock() if self.act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lab(self.conv(x))
        return self.act_lab(F.hardswish(x)) if self.act_lab is not None else x


class _SE(nn.Module):
    """SE with PyTorch's hardsigmoid (x / 6 + 0.5); `slope` / `offset` for det_mobilenet_v3's (0.2, 0.5)."""

    def __init__(self, c: int, slope: float = 1 / 6) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c // 4, 1)
        self.conv2 = nn.Conv2d(c // 4, c, 1)
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.conv2(F.relu(self.conv1(x.mean((2, 3), keepdim=True))))
        s = F.hardsigmoid(s) if self.slope == 1 / 6 else (s * self.slope + 0.5).clamp_(0, 1)
        return x * s


class _LCBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int | tuple, k: int, se: bool) -> None:
        super().__init__()
        self.dw_conv = _RepLayer(cin, cin, k, stride, groups=cin)
        self.se = _SE(cin) if se else nn.Identity()
        self.pw_conv = _RepLayer(cin, cout, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw_conv(self.se(self.dw_conv(x)))


class PPLCNetV3(nn.Module):
    """PP-LCNetV3: `det` (scale 0.75) returns 4 projected stage outputs, else (scale 0.95) `[B, C, 1, W/8]`."""

    def __init__(self, scale: float, det: bool) -> None:
        super().__init__()
        self.det = det
        cfg = LCNET_DET if det else LCNET_REC
        self.conv1 = ConvBN(3, make_divisible(16 * scale), 3, 2)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                *(_LCBlock(make_divisible(i * scale), make_divisible(o * scale), s, k, se) for k, i, o, s, se in rows)
            )
            for rows in cfg.values()
        )
        if det:
            chans = [make_divisible(cfg[b][-1][2] * scale) for b in ("blocks3", "blocks4", "blocks5", "blocks6")]
            self.layer_list = nn.ModuleList(
                nn.Conv2d(c, int(m * scale), 1) for c, m in zip(chans, (16, 24, 56, 480), strict=True)
            )
            self.out_channels = [int(m * scale) for m in (16, 24, 56, 480)]
        else:
            self.out_channels = make_divisible(512 * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x = self.blocks[0](self.conv1(x))
        outs = []
        for blk in self.blocks[1:]:
            x = blk(x)
            outs.append(x)
        if self.det:
            return [conv(o) for conv, o in zip(self.layer_list, outs, strict=True)]
        return F.avg_pool2d(x, (3, 2))
