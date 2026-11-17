"""Per-channel image normalization for CUDA BCHW tensors."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"normalize": "pppplil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Subtract the channel mean and divide by its standard deviation in float32."""
    if x.ndim != len("BCHW") or mean.numel() != x.shape[1] or std.numel() != x.shape[1]:
        msg = "Expected BCHW input with one mean and standard deviation per channel"
        raise ValueError(msg)
    x = x.float().contiguous()
    out = torch.empty_like(x)
    load(x.device).launch(
        "normalize",
        *linear(x.numel()),
        x,
        mean.float().contiguous(),
        std.float().contiguous(),
        out,
        x.shape[2] * x.shape[3],
        x.shape[1],
        x.numel(),
    )
    return out
