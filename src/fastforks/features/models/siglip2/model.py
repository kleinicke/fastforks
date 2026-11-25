"""SigLIP 2 runners: timm's network walked by `fastforks.shared.libnet.features` (MLX: `features.MlxPortable`)."""

from fastforks.shared.libnet import features

RUNNERS = {
    "portable": features.Portable,
    "kernels": features.Kernels,
    "portable_mlx": features.MlxPortable,
}
