"""Perception Encoder runners: timm's network walked by `fastforks.shared.libnet.features`.

On Apple MPS, kernels is the portable network with the interleaved RoPE as a Metal kernel (`vit_metal`).
"""

from fastforks.shared.libnet import features

RUNNERS = {"portable": features.Portable, "kernels": features.Kernels, "kernels_mps": features.MetalKernels}
