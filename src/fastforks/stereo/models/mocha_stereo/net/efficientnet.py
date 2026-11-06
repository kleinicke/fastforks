"""fastforks: timm's `tf_efficientnetv2_l` (features_only), stem and stages 0-5, without timm.

Same module names as timm 1.0 (`conv_stem`, `bn1`, `blocks.<stage>.<block>.<conv_pw | bn1 | ...>`), so MoCha's
checkpoint keys load unchanged, and the same maths: the `tf_` variant (BatchNorm eps 1e-3, TensorFlow "same" padding:
dynamic, asymmetric for the stride-2 convs, static k // 2 for stride 1), SiLU, squeeze-excite with 1/4 of the block
input channels (checked bit-exact against timm 1.0.30, fp32 CPU). Stage 6 (640 channels) is not built: MoCha's
`Feature` uses stages 0-5 only (its checkpoints hold no stage 6).
"""
import math

import torch.nn.functional as F
from torch import nn

# timm _gen_efficientnetv2_l: (block type, repeats, stride, expansion, out channels, squeeze-excite ratio)
_STAGES = (
    ("cn", 4, 1, 1, 32, None),
    ("er", 7, 2, 4, 64, None),
    ("er", 7, 2, 4, 96, None),
    ("ir", 10, 2, 4, 192, 0.25),
    ("ir", 19, 1, 6, 224, 0.25),
    ("ir", 25, 2, 6, 384, 0.25),
)


def pad_same(x, k, s, d=1):
    """timm pad_same: TensorFlow "same" padding (extra pixel at the bottom / right)."""
    ih, iw = x.shape[-2:]
    ph = max((math.ceil(ih / s) - 1) * s + (k - 1) * d + 1 - ih, 0)
    pw = max((math.ceil(iw / s) - 1) * s + (k - 1) * d + 1 - iw, 0)
    if ph > 0 or pw > 0:
        x = F.pad(x, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2])
    return x


class Conv2dSame(nn.Conv2d):
    """timm Conv2dSame: TensorFlow "same" padding for the stride-2 convs (input-size dependent)."""

    def __init__(self, cin, cout, k, stride, groups=1, bias=False):
        super().__init__(cin, cout, k, stride, 0, 1, groups, bias)

    def forward(self, x):
        x = pad_same(x, self.kernel_size[0], self.stride[0], self.dilation[0])
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


def conv(cin, cout, k, stride=1, groups=1):
    if stride > 1:
        return Conv2dSame(cin, cout, k, stride, groups)
    return nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False)


class BatchNormAct2d(nn.BatchNorm2d):
    """timm BatchNormAct2d (eval): BatchNorm, then the activation (SiLU or none)."""

    def __init__(self, c, act=True):
        super().__init__(c, eps=1e-3)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(super().forward(x))


class SqueezeExcite(nn.Module):
    def __init__(self, c, rd):
        super().__init__()
        self.conv_reduce = nn.Conv2d(c, rd, 1)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_expand = nn.Conv2d(rd, c, 1)

    def forward(self, x):
        s = self.conv_expand(self.act1(self.conv_reduce(x.mean((2, 3), keepdim=True))))
        return x * s.sigmoid()


class ConvBnAct(nn.Module):
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.has_skip = stride == 1 and cin == cout  # the stage is "_skip"
        self.conv = conv(cin, cout, k, stride)
        self.bn1 = BatchNormAct2d(cout)

    def forward(self, x):
        y = self.bn1(self.conv(x))
        return y + x if self.has_skip else y


class EdgeResidual(nn.Module):
    def __init__(self, cin, cout, k, stride, exp):
        super().__init__()
        mid = cin * exp
        self.has_skip = stride == 1 and cin == cout
        self.conv_exp = conv(cin, mid, k, stride)
        self.bn1 = BatchNormAct2d(mid)
        self.conv_pwl = nn.Conv2d(mid, cout, 1, bias=False)
        self.bn2 = BatchNormAct2d(cout, act=False)

    def forward(self, x):
        y = self.bn2(self.conv_pwl(self.bn1(self.conv_exp(x))))
        return y + x if self.has_skip else y


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, k, stride, exp, se):
        super().__init__()
        mid = cin * exp
        self.has_skip = stride == 1 and cin == cout
        self.conv_pw = nn.Conv2d(cin, mid, 1, bias=False)
        self.bn1 = BatchNormAct2d(mid)
        self.conv_dw = conv(mid, mid, k, stride, groups=mid)
        self.bn2 = BatchNormAct2d(mid)
        self.se = SqueezeExcite(mid, round(cin * se))
        self.conv_pwl = nn.Conv2d(mid, cout, 1, bias=False)
        self.bn3 = BatchNormAct2d(cout, act=False)

    def forward(self, x):
        y = self.se(self.bn2(self.conv_dw(self.bn1(self.conv_pw(x)))))
        y = self.bn3(self.conv_pwl(y))
        return y + x if self.has_skip else y


class EfficientNetFeatures(nn.Module):
    def __init__(self, stages=_STAGES, stem=32, k=3):
        super().__init__()
        self.conv_stem = conv(3, stem, k, 2)
        self.bn1 = BatchNormAct2d(stem)
        blocks, cin = [], stem
        for kind, n, stride, exp, cout, se in stages:
            stage = []
            for i in range(n):
                s = stride if i == 0 else 1
                if kind == "cn":
                    stage.append(ConvBnAct(cin, cout, k, s))
                elif kind == "er":
                    stage.append(EdgeResidual(cin, cout, k, s, exp))
                else:
                    stage.append(InvertedResidual(cin, cout, k, s, exp, se))
                cin = cout
            blocks.append(nn.Sequential(*stage))
        self.blocks = nn.Sequential(*blocks)


def tf_efficientnetv2_l_features():
    return EfficientNetFeatures()
