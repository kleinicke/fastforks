"""PP-OCRv5 networks in PyTorch: the text detectors (DB) and the CTC text recognisers."""

from __future__ import annotations

import torch
from torch import nn

from .backbones import PPHGNetV2B4, PPLCNetV3
from .heads import CTCHead, DBHead
from .layers import creation_order, paddle_layers
from .necks import LKPAN, RSEFPN

__all__ = ["Detector", "Recognizer", "build"]


class Detector(nn.Module):
    """BGR image `[B, 3, H, W]` normalised (H, W multiples of 32) -> DB shrink map `[B, H, W]` in [0, 1]."""

    def __init__(self, server: bool) -> None:
        super().__init__()
        if server:
            self.backbone = PPHGNetV2B4(det=True)
            self.neck = LKPAN([128, 512, 1024, 2048], 256)
            self.head = DBHead(256, local=True)
        else:
            self.backbone = PPLCNetV3(0.75, det=True)
            self.neck = RSEFPN(self.backbone.out_channels, 96)
            self.head = DBHead(96, local=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.neck(self.backbone(x)))[:, 0]

    @torch.no_grad()
    def scale_neck(self, s: float) -> None:
        """Divide the LKPAN neck's activations by `s` and fold the head's first BatchNorm into its conv; same maths.

        The server neck and head reach ~1.5e6 on text photos (and that BatchNorm's variance more), past float16's
        65504. The neck is linear up to the IntraCL blocks, whose biases and BatchNorm shifts scale with it (ReLU
        commutes with a positive scale); the folded head conv takes the factor back.
        """
        if not isinstance(self.neck, LKPAN):
            return
        for conv in self.neck.ins_conv:
            conv.weight.div_(s)
        for block in self.neck.incl:
            for m in block.modules():
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    m.bias.div_(s)
            block.bn.running_mean.div_(s)
            block.bn.bias.div_(s)
        branch = self.head.binarize
        conv, bn = branch.conv1, branch.conv_bn1
        k = bn.weight.double() / (bn.running_var.double() + bn.eps).sqrt()
        fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, padding=conv.padding)
        fused.weight.copy_(conv.weight.double() * (k * s).view(-1, 1, 1, 1))
        fused.bias.copy_(bn.bias.double() - bn.running_mean.double() * k)
        branch.conv1, branch.conv_bn1 = fused, nn.Identity()


class Recognizer(nn.Module):
    """BGR text line `[B, 3, 48, W]` in [-1, 1] -> CTC probabilities `[B, W / 8, classes]` (class 0 = blank)."""

    def __init__(self, server: bool, classes: int) -> None:
        super().__init__()
        self.backbone = PPHGNetV2B4(det=False) if server else PPLCNetV3(0.95, det=False)
        self.head = CTCHead(self.backbone.out_channels, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build(kind: str, server: bool, classes: int = 0) -> tuple[nn.Module, dict[str, list[tuple[str, nn.Module]]]]:
    """A network (`"det"` / `"rec"`) and its parameter layers by Paddle type in Paddle's numbering order."""
    with creation_order() as created:
        net = Detector(server) if kind == "det" else Recognizer(server, classes)
    return net, paddle_layers(net, created)
