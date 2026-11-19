"""CLIP / OpenCLIP runners: timm's network walked by `fastforks.shared.libnet.features` (portable / kernels)."""

from fastforks.shared import mlx_wrap
from fastforks.shared.libnet import features

RUNNERS = {
    "portable": features.Portable,
    "kernels": features.Kernels,
    "portable_mlx": mlx_wrap.port(features.Portable, attrs=(), methods=("_forward",)),
}
