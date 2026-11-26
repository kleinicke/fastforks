"""Perception Encoder runners: timm's network walked by `fastforks.shared.libnet.features`.

On Apple MPS, kernels is the portable network with the interleaved RoPE as a Metal kernel (`vit_metal`). On Apple MLX
the network is written by hand (`features.MlxPortable`, `mlx_features`).
"""

from dataclasses import dataclass

from fastforks.shared.libnet import features


@dataclass(frozen=True)
class Params(features.Params):
    """Perception Encoder options, including CUDA weight streaming."""

    stream: bool = False


class Portable(features.Portable):
    """PyTorch inference with optional block streaming."""

    Params = Params


class Kernels(features.Kernels):
    """Fused CUDA inference with optional block streaming."""

    Params = Params


class MetalKernels(features.MetalKernels):
    """Metal RoPE with resident weights."""

    Params = Params


class MlxPortable(features.MlxPortable):
    """Apple MLX: the hand-written Eva (absolute table + 2D RoPE)."""

    Params = Params


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "kernels_mps": MetalKernels,
    "portable_mlx": MlxPortable,
}
