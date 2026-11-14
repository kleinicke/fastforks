"""timm's ResNet `BasicBlock` stages (ResNet-18 / -34 `layer1..4`), written out so `timm` is not a dependency.

fastforks addition (not upstream code): the module tree and parameter names of timm 1.0 `models/resnet.py`
(Apache-2.0, Ross Wightman) for its default ResNet (3x3 BasicBlocks, ReLU, BatchNorm, 1x1 strided conv + BatchNorm
shortcut, no anti-aliasing / attention / drop path), so `timm.create_model("resnet18...", features_only=True).layerN`
state dicts load unchanged.
"""

import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.drop_block = nn.Identity()
        self.act1 = nn.ReLU(inplace=True)
        self.aa = nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = None
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = None

    def forward(self, x):
        shortcut = x
        x = self.aa(self.act1(self.drop_block(self.bn1(self.conv1(x)))))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        return self.act2(x)


def make_layer(inplanes, planes, blocks, stride):
    """timm `make_blocks` for one stage: `blocks` BasicBlocks, the first with stride and a 1x1 conv shortcut if needed."""
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False), nn.BatchNorm2d(planes)
        )
    layers = [BasicBlock(inplanes, planes, stride, downsample)]
    layers += [BasicBlock(planes, planes) for _ in range(1, blocks)]
    return nn.Sequential(*layers)


RESNET18 = (2, 2, 2, 2)  # blocks per stage


def resnet_layers(depths=RESNET18):
    """`layer1..layer4` of a timm BasicBlock ResNet (64, 128, 256, 512 channels; strides 1, 2, 2, 2)."""
    chans = (64, 128, 256, 512)
    return [make_layer(64 if i == 0 else chans[i - 1], chans[i], d, 1 if i == 0 else 2) for i, d in enumerate(depths)]
