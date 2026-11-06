"""CoEx: MobileNetV2 features, attention correlation volume, 3D aggregation with guided cost excitation, top-k
soft-argmax over a 1/4-resolution volume, superpixel upsampling to full resolution."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.mobilenetv2 import MobileNetV2Features

CHANS = (16, 24, 32, 96, 160)  # mobilenetv2_100 stage widths (1/2 ... 1/32)
BLOCKS = (1, 2, 3, 5, 6)  # stage boundaries of `MobileNetV2Features.blocks` that upstream groups into block0 ... block4
SPIXEL = (32, 48)  # width of the image-stem branches at 1/2 and 1/4
AGG = (16, 32, 48)  # 3D aggregation widths
BLOCKS_PER_LEVEL = (2, 2, 2)


class BasicConv(nn.Module):
    """conv (no bias) -> BatchNorm -> LeakyReLU(0.01), 2D or 3D, optionally transposed."""

    def __init__(self, cin, cout, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super().__init__()
        self.relu, self.use_bn = relu, bn
        conv = (nn.ConvTranspose3d if deconv else nn.Conv3d) if is_3d else (nn.ConvTranspose2d if deconv else nn.Conv2d)
        self.conv = conv(cin, cout, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(cout) if is_3d else nn.BatchNorm2d(cout)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = self.LeakyReLU(x)
        return x


class Conv2x(nn.Module):
    """Strided (de)convolution, merge with the skip tensor by concatenation (nearest-resized if needed), conv."""

    def __init__(self, cin, cout, deconv=False):
        super().__init__()
        self.conv1 = BasicConv(cin, cout, deconv, False, kernel_size=4 if deconv else 3, stride=2, padding=1)
        self.conv2 = BasicConv(cout * 2, cout * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=rem.shape[-2:], mode="nearest")
        return self.conv2(torch.cat((x, rem), 1))


class Feature(nn.Module):
    """MobileNetV2 stages as `block0 ... block4` (state-dict names of upstream's timm-based module)."""

    def __init__(self):
        super().__init__()
        model = MobileNetV2Features()
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        for i in range(5):
            setattr(self, f"block{i}", nn.Sequential(*model.blocks[(BLOCKS[i - 1] if i else 0) : BLOCKS[i]]))

    def forward(self, x):
        x = self.bn1(self.conv_stem(x))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return x2, [x4, x8, x16, x32]


class FeatUp(nn.Module):
    """Top-down merge of the 1/32 ... 1/4 features."""

    def __init__(self):
        super().__init__()
        self.deconv32_16 = Conv2x(CHANS[4], CHANS[3], deconv=True)
        self.deconv16_8 = Conv2x(CHANS[3] * 2, CHANS[2], deconv=True)
        self.deconv8_4 = Conv2x(CHANS[2] * 2, CHANS[1], deconv=True)
        self.conv4 = BasicConv(CHANS[1] * 2, CHANS[1] * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, feats):
        x4, x8, x16, x32 = feats
        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.conv4(self.deconv8_4(x8, x4))
        return [x4, x8, x16, x32]


class CostVolume(nn.Module):
    """Single-head correlation of `x` with `y` shifted by 0 ... `maxdisp` px, `[B, 1, maxdisp + 1, H, W]` (flipped)."""

    def __init__(self, maxdisp):
        super().__init__()
        self.maxdisp = maxdisp + 1
        self.unfold = nn.Unfold((1, maxdisp + 1), 1, 0, 1)
        self.left_pad = nn.ZeroPad2d((maxdisp, 0, 0, 0))

    def forward(self, x, y):
        b, c, h, w = x.shape
        unfolded = self.unfold(self.left_pad(y)).reshape(b, 1, c, self.maxdisp, h, w)
        cost = (x.reshape(b, 1, c, 1, h, w) * unfolded).sum(2)
        return torch.flip(cost, [2])


class AttentionCostVolume(nn.Module):
    """Features -> conv -> 1x1 descriptor, L2-normalised, correlated over disparity."""

    def __init__(self, max_disp, cin, hidden):
        super().__init__()
        self.costVolume = CostVolume(max_disp // 4)
        self.conv = BasicConv(cin, hidden, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(hidden, hidden, kernel_size=1, padding=0, stride=1)

    def forward(self, left, right):
        x, y = self.desc(self.conv(left)), self.desc(self.conv(right))
        return self.costVolume(_l2(x), _l2(y))


def _l2(x):
    """Channel-wise L2 normalisation, computed in float32 (a no-op cast for float32 inputs)."""
    xf = x.float()
    return (xf / torch.norm(xf, 2, 1, True)).to(x.dtype)


class ChannelAtt(nn.Module):
    """Guided cost excitation: image features gate the channels of a cost-volume level."""

    def __init__(self, cv_chan, im_chan):
        super().__init__()
        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0), nn.Conv2d(im_chan // 2, cv_chan, 1)
        )

    def forward(self, cv, im):
        return torch.sigmoid(self.im_att(im).unsqueeze(2)) * cv


class Aggregation(nn.Module):
    """3D hourglass over the cost volume with guided cost excitation at every level."""

    def __init__(self, max_disp):
        super().__init__()
        self.D = max_disp // 4
        self.conv_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.channelAttStem = ChannelAtt(8, 2 * CHANS[1] + SPIXEL[1])
        self.channelAtt, self.channelAttDown = nn.ModuleList(), nn.ModuleList()
        self.conv_down, self.conv_up, self.conv_skip, self.conv_agg = (nn.ModuleList() for _ in range(4))
        channels = (8, *AGG)
        inp = channels[0]
        for i in range(3):
            convs = []
            for n in range(BLOCKS_PER_LEVEL[i]):
                stride = (2, 2, 2) if n == 0 else 1
                convs.append(BasicConv(inp, channels[i + 1], is_3d=True, kernel_size=3, padding=1, stride=stride))
                inp = channels[i + 1]
            self.conv_down.append(nn.Sequential(*convs))
            self.channelAttDown.append(ChannelAtt(channels[i + 1], (1 if i == 2 else 2) * CHANS[i + 2]))
            out, bn, relu = (1, False, False) if i == 0 else (channels[i], True, True)
            self.conv_up.append(
                BasicConv(
                    channels[i + 1], out, deconv=True, is_3d=True, bn=bn, relu=relu,
                    kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2),
                )
            )  # fmt: skip
            self.conv_agg.append(
                nn.Sequential(
                    BasicConv(channels[i], channels[i], is_3d=True, kernel_size=3, padding=1, stride=1),
                    BasicConv(channels[i], channels[i], is_3d=True, kernel_size=3, padding=1, stride=1),
                )
            )
            self.conv_skip.append(
                BasicConv(2 * channels[i], channels[i], is_3d=True, kernel_size=1, padding=0, stride=1)
            )
            self.channelAtt.append(ChannelAtt(channels[i], 2 * CHANS[i + 1]))

    def forward(self, img, cost):
        b, _, h, w = img[0].shape
        cost = self.channelAttStem(self.conv_stem(cost.reshape(b, -1, self.D, h, w)), img[0])
        feats = [cost]
        for i in range(3):
            cost = self.channelAttDown[i](self.conv_down[i](cost), img[i + 1])
            feats.append(cost)
        for i in range(3):
            cost = self.conv_up[-i - 1](cost)
            if cost.shape != feats[-i - 2].shape:
                cost = F.interpolate(cost, size=feats[-i - 2].shape[-3:], mode="nearest")
            if i == 2:
                break
            cost = self.conv_skip[-i - 1](torch.cat([cost, feats[-i - 2]], 1))
            cost = self.channelAtt[-i - 1](self.conv_agg[-i - 1](cost), img[-i - 2])
        return cost


def upfeat(x, prob, up_h=4, up_w=4):
    """Superpixel upsampling: every output pixel mixes the 3x3 neighbourhood of its coarse pixel with `prob` weights."""
    b, _, h, w = x.shape
    feat = F.unfold(x, 3, 1, 1).reshape(b, -1, h, w)
    feat = F.interpolate(feat, (h * up_h, w * up_w), mode="nearest").reshape(b, -1, 9, h * up_h, w * up_w)
    return (feat * prob.unsqueeze(1)).sum(2)


class CoEx(nn.Module):
    """Stereo network; `forward(left, right)` returns the full-resolution disparity `[B, H, W]` (float32)."""

    def __init__(self, max_disp: int = 192, top_k: int = 2):
        super().__init__()
        self.D = max_disp // 4
        self.top_k = top_k
        self.feature = Feature()
        self.up = FeatUp()
        self.cost_volume = AttentionCostVolume(max_disp, CHANS[1] * 2 + SPIXEL[1], CHANS[1] * 2)
        self.cost_agg = Aggregation(max_disp)
        self.stem_2 = nn.Sequential(
            BasicConv(3, SPIXEL[0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(SPIXEL[0], SPIXEL[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(SPIXEL[0]),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            BasicConv(SPIXEL[0], SPIXEL[1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(SPIXEL[1], SPIXEL[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(SPIXEL[1]),
            nn.ReLU(),
        )
        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1))
        self.spx_2 = Conv2x(CHANS[1], 32, deconv=True)
        self.spx_4 = nn.Sequential(
            BasicConv(CHANS[1] * 2 + SPIXEL[1], CHANS[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(CHANS[1], CHANS[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(CHANS[1]),
            nn.ReLU(),
        )

    def features(self, left, right):
        """Both images through the backbone, the top-down merge and the image stems: (x list, y list, x/2 stem)."""
        b = left.shape[0]
        both = torch.cat([left, right], 0)
        _, v = self.feature(both)
        v = self.up(v)
        stem_2 = self.stem_2(both)
        stem_4 = self.stem_4(stem_2)
        x = [t[:b] for t in v]
        y = [t[b:] for t in v]
        x[0] = torch.cat((x[0], stem_4[:b]), 1)
        y[0] = torch.cat((y[0], stem_4[b:]), 1)
        return x, y, stem_2[:b]

    def spixel_weights(self, x0, stem_2x):
        """Softmax over the 9 neighbours for the 4x upsampling."""
        xspx = self.spx_2(self.spx_4(x0), stem_2x)
        return F.softmax(self.spx(xspx).float(), 1)

    def regress(self, cost, spg):
        """Top-k soft-argmin over the disparity axis of `cost` (`[B, 1, D, h, w]`), then superpixel upsampling."""
        cost = cost.float()
        k = self.top_k
        ind = cost.sort(2, descending=True)[1][:, :, :k]
        corr = F.softmax(torch.gather(cost, 2, ind), 2)
        disp_4 = torch.sum(corr * ind, 2, keepdim=True)
        disp_4 = disp_4.reshape(cost.shape[0], 1, *disp_4.shape[-2:])
        return upfeat(disp_4, spg, 4, 4).squeeze(1) * 4

    def forward(self, left, right):
        """Disparity `[B, H, W]` of a padded pair (multiples of 32)."""
        x, y, stem_2x = self.features(left, right)
        cost = self.cost_volume(x[0], y[0])[:, :, :-1]
        cost = self.cost_agg(x, cost)
        return self.regress(cost, self.spixel_weights(x[0], stem_2x))
