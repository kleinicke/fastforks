# fastforks: the fine features of upstream `romav2/features.py`: `torchvision.models.vgg19_bn().features[:27]` built
# from plain layers (same indices, so the checkpoint keys `refiner_features.layers.N` load unchanged). The DINOv3
# descriptor is in `dinov3.py`.
import torch
from torch import nn

from .amp import autocast
from .normalizers import imagenet


def _vgg19bn_layers() -> list[nn.Module]:
    layers: list[nn.Module] = []
    c_in = 3
    for v in (64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M"):
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(c_in, v, kernel_size=3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            c_in = v
    return layers


class VGG19BN(nn.Module):
    """Features at strides 1, 2 and 4 as `{stride: [B, H, W, C]}`."""

    amp: torch.dtype | None = None  # fastforks: autocast dtype of upstream's bf16 region (amp.py)
    channels_last: bool = False  # fastforks: NHWC activations (cuDNN; the stride maps come out as contiguous BHWC)

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_vgg19bn_layers())

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x = imagenet(x)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with autocast(x, self.amp):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x.permute(0, 2, 3, 1)
                    scale = scale * 2
                    if scale > 4:  # the last pooled map is not used
                        break
                x = layer(x)
            return feats
