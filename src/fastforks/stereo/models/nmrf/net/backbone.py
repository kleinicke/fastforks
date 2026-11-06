import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)


        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not (stride == 1 and in_planes == planes):
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        identity = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))

        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x + identity)


class Backbone(nn.Module):
    def __init__(self, output_dim=128, norm_layer=nn.InstanceNorm2d):
        super(Backbone, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # 1/2
        self.norm1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1, norm_layer=norm_layer) # 1/2
        self.layer2 = self._make_layer(96, stride=2, norm_layer=norm_layer) # 1/4
        self.layer3 = self._make_layer(128, stride=1, norm_layer=norm_layer) # 1/4

        self.conv2 = nn.Conv2d(128, output_dim, 1, 1, 0)

        self.output_dim = output_dim

        # fastforks: no weight initialisation (`_init_weights`): the checkpoint sets every parameter

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = 2 * (x / 255.0) - 1.0
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        x = self.layer2(x)  # 1/4
        x = self.layer3(x)  # 1/4
        x = self.conv2(x)

        out = [x, F.avg_pool2d(x, kernel_size=2, stride=2)]  # high to low res

        return out


# fastforks: `checkpoint_filter_fn` and `create_backbone`'s Swin branch (SwinAdaptor + MSDeformAttn) are not vendored;
# only the ResNet backbone of the released Scene Flow model is used
