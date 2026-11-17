"""CLIP / OpenCLIP runners: timm's network walked by `fastforks.shared.libnet.features` (portable / kernels)."""

from fastforks.shared.libnet import features

RUNNERS = {"portable": features.Portable, "kernels": features.Kernels}
