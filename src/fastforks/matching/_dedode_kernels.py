"""Fused CUDA convolutions for DeDoDe encoders and refiners."""

import torch
from torch import nn

from fastforks._core.patch import override
from fastforks.shared import cnn_kernels
from fastforks.shared.dedode import VGG, ConvRefiner, Descriptor, Detector, VGGDINOv2


def prepare(network: Detector | Descriptor, device: torch.device) -> None:
    """Prepare one VGG encoder and its convolutional decoder for half-precision inference."""
    encoder = network.encoder
    vgg = encoder.vgg if isinstance(encoder, VGGDINOv2) else encoder
    vgg.layers = nn.Sequential(*vgg.layers)  # ty: ignore[invalid-assignment] - fuse adjacent convolution blocks
    cnn_kernels.prepare_generic(vgg, device)
    override(vgg, _vgg_forward)
    for block in network.decoder.modules():
        if isinstance(block, ConvRefiner):
            cnn_kernels.prepare_generic(block, device)
            override(block, _refiner_forward)


def _vgg_forward(self: VGG, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Size]]:
    """`VGG.forward` on an fp16 channels-last input."""
    x = x.half().contiguous(memory_format=torch.channels_last)
    feats, sizes = [], []
    for layer in self.layers:
        if isinstance(layer, nn.MaxPool2d):
            feats.append(x)
            sizes.append(x.shape[-2:])
        x = layer(x)
    return feats, sizes


def _refiner_forward(self: ConvRefiner, feats: torch.Tensor) -> torch.Tensor:
    """`ConvRefiner.forward` on an fp16 channels-last input."""
    x0 = self.block1(feats.half().contiguous(memory_format=torch.channels_last))
    return self.out_conv((self.hidden_blocks(x0) + x0) / 1.4)
