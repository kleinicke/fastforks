# MVSAnywhere (c) Niantic, Inc. 2025, non-commercial licence (LICENSE); BlurPool and the antialiased ResNet stem
# (c) 2019 Adobe, CC-BY-NC-SA-4.0 (LICENSE-antialiased-cnns.md), built on torchvision's ResNet (BSD-3-Clause).
import numpy as np
import torch
import torch.nn as nn

from fastforks.shared.depth_anything_v2 import FeatureFusionBlock, _make_scratch


# ---------------------------------------------------------------------------------------------- layers.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.Identity):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bias = norm_layer == nn.Identity
        self.conv1 = conv3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=bias)
        self.bn2 = norm_layer(planes)
        if inplanes == planes * self.expansion and stride == 1:
            self.downsample = None
        else:
            conv = conv1x1 if stride == 1 else conv3x3
            self.downsample = nn.Sequential(
                conv(inplanes, planes * self.expansion, bias=bias, stride=stride),
                norm_layer(planes * self.expansion),
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ---------------------------------------------------------------------------------------------- networks.py
class MLP(nn.Module):
    def __init__(self, channel_list, disable_final_activation=False):
        super(MLP, self).__init__()

        layer_list = []
        for layer_index in list(range(len(channel_list)))[:-1]:
            layer_list.append(nn.Linear(channel_list[layer_index], channel_list[layer_index + 1]))
            layer_list.append(nn.LeakyReLU(inplace=True))

        if disable_final_activation:
            layer_list = layer_list[:-1]

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)


# antialiased_cnns (blurpool.py / resnet.py): the pieces of resnet18(filter_size=4, pool_only=True) up to layer1
class BlurPool(nn.Module):
    def __init__(self, channels, filt_size=4, stride=2):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_sizes = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))] * 2
        self.stride = stride
        self.channels = channels
        a = {3: [1.0, 2.0, 1.0], 4: [1.0, 3.0, 3.0, 1.0], 5: [1.0, 4.0, 6.0, 4.0, 1.0]}[filt_size]
        a = np.array(a)
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = nn.ReflectionPad2d(self.pad_sizes)

    def forward(self, inp):
        return nn.functional.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class ResNetBasicBlock(nn.Module):
    """torchvision / antialiased_cnns BasicBlock, stride 1 (layer1)."""

    expansion = 1

    def __init__(self, inplanes, planes):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResnetMatchingEncoder(nn.Module):
    """Pytorch module for a resnet encoder (antialiased ResNet-18 stem + layer1)."""

    def __init__(self, num_ch_out=16):
        super().__init__()

        self.num_ch_enc = np.array([64, 64])
        self.num_ch_out = num_ch_out

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1), BlurPool(64, filt_size=4, stride=2)),
            nn.Sequential(ResNetBasicBlock(64, 64), ResNetBasicBlock(64, 64)),
            nn.Conv2d(self.num_ch_enc[-1], 128, (1, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, self.num_ch_out, (3, 3), padding=1, padding_mode="replicate"),
            nn.InstanceNorm2d(self.num_ch_out),
        )

    def forward(self, input_image):
        return self.net(input_image)


# ---------------------------------------------------------------------------------------------- vit_modules.py
class CostVolumePatchEmbed(nn.Module):
    def __init__(
        self,
        num_ch_cv,
        num_feats,
        num_ch_outs=[128, 256],
        num_ch_proj=[96, 192],
        patch_size=[14, 14],
    ):
        super().__init__()
        self.num_ch_outs = num_ch_outs
        self.num_ch_cv = num_ch_cv
        self.patch_size = patch_size
        self.num_feats = num_feats
        self.convs = nn.ModuleDict()

        for i in range(3):
            num_ch_in = num_ch_cv if i == 0 else num_ch_outs[i - 1]
            num_ch_out = (num_ch_outs + [self.num_feats])[i]
            self.convs[f"ds_conv_{i}"] = BasicBlock(num_ch_in, num_ch_out, stride=1 if i == 0 else 2)

            if i < 2:
                self.convs[f"conv_{i}"] = nn.Sequential(
                    BasicBlock(num_ch_proj[i] + num_ch_out, num_ch_out, stride=1),
                    BasicBlock(num_ch_out, num_ch_out, stride=1),
                )

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=num_feats,
                    out_channels=chns,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for chns in num_ch_proj
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=num_ch_proj[0], out_channels=num_ch_proj[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=num_ch_proj[1], out_channels=num_ch_proj[1], kernel_size=2, stride=2, padding=0
                ),
            ]
        )

    def forward(self, x, img_feats):
        # resize such that 2 downsamples will give 1/14th resolution
        B, C, H, W = x.shape

        for i in range(3):
            x = self.convs[f"ds_conv_{i}"](x)

            if i < 2:
                img_feat = img_feats[i][0].reshape((B, H * 4 // 16, W * 4 // 16, self.num_feats))
                img_feat = img_feat.permute(0, 3, 1, 2)
                img_feat = self.projects[i](img_feat)
                img_feat = self.resize_layers[i](img_feat)
                x = torch.cat([x, img_feat], dim=1)
                x = self.convs[f"conv_{i}"](x)

        return x.flatten(2).transpose(1, 2)  # B HW C


# ---------------------------------------------------------------------------------------------- depth_anything_blocks.py
def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


def _head(features, prediction_scale, nclass=1):
    head_features_1 = features
    head_features_2 = 32
    return nn.Sequential(
        nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
        nn.Upsample(scale_factor=(16 * prediction_scale) / 8, mode="bilinear", align_corners=True),
        nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(head_features_2, nclass, kernel_size=1, stride=1, padding=0),
        nn.Identity(),
    )


class DPTHead(nn.Module):
    """dinov2_vitb14 DPT head with the scale-0 depth and sky heads (`output_convs[0]`, `sky_convs[0]`)."""

    def __init__(self, in_channels=768, features=128, out_channels=[96, 192, 384, 768], prediction_scale=1.0):
        super(DPTHead, self).__init__()

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, False)
        self.scratch.refinenet2 = _make_fusion_block(features, False)
        self.scratch.refinenet3 = _make_fusion_block(features, False)
        self.scratch.refinenet4 = _make_fusion_block(features, False)

        self.output_convs = nn.ModuleList([_head(features, prediction_scale)])
        self.sky_convs = nn.ModuleList([_head(features, prediction_scale)])

    def forward(self, out_features, patch_h, patch_w, sky=True):
        """Patch tokens `[B, hw, C]` of 4 layers -> depth logits `[B, 1, H, W]` (and sky logits, or None)."""
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        return self.output_convs[0](path_1), self.sky_convs[0](path_1) if sky else None
