import torch
import torch.nn as nn
from torch.nn import ReLU6
# fastforks: timm's `fasternet_t0` features_only model is replaced by the timm-free re-implementation (same module
# names, so the checkpoints load unchanged); `import timm` removed.
from fastforks.shared.fasternet import FasterNetFeatures
from .submodule import *


class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = BasicConv(
            in_channels,
            out_channels,
            deconv=True,
            bn=True,
            relu=True,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.concat = BasicConv(
            out_channels * 2,
            out_channels * 2,
            bn=True,
            relu=True,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, y):
        x = self.deconv(x)
        xy = torch.cat([x, y], 1)
        out = self.concat(xy)
        return out


FASTERNET_T0_MODEL = "fasternet_t0"
LAS1_FEATURE_CHANNELS = [24, 32, 96, 160]
LAS2_FEATURE_CHANNELS = [40, 80, 160, 320]


# fastforks: LAS1's `FeatureNet` (timm mobilenetv2_100, not used by the LAS2 models) removed.


class FeatureNetFasterNet(nn.Module):
    """FasterNet feature pyramid used by the simplified LAS2 release model."""

    def __init__(self, pretrained=True):
        super().__init__()

        # fastforks: was timm.create_model(FASTERNET_T0_MODEL, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        self.backbone = FasterNetFeatures()

        self.feature_channels = list(self.backbone.channels)
        if self.feature_channels != LAS2_FEATURE_CHANNELS:
            raise ValueError(
                f"Expected {FASTERNET_T0_MODEL} channels {LAS2_FEATURE_CHANNELS}, "
                f"got {self.feature_channels}"
            )

        channels = self.feature_channels
        self.fpn_layer4 = FPNLayer(channels[3], channels[2])
        self.fpn_layer3 = FPNLayer(channels[2], channels[1])
        self.fpn_layer2 = FPNLayer(channels[1], channels[0])

        self.out_conv = BasicConv2d(
            channels[0],
            channels[0],
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            norm_layer=nn.InstanceNorm2d,
        )

    def forward(self, images):
        c2, c3, c4, c5 = self.backbone(images)

        p4 = self.fpn_layer4(c5, c4)
        p3 = self.fpn_layer3(p4, c3)
        p2 = self.fpn_layer2(p3, c2)
        p2 = self.out_conv(p2)

        return [p2, p3, p4, c5]
