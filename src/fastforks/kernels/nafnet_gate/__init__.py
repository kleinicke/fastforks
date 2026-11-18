"""NAFNet's depthwise 3 x 3 convolution and SimpleGate in one CUDA launch."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"depthwise_gate": "ppppiiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def depthwise_gate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Gate an NCHW tensor using kernel-major weights [9, 2C] and bias [2C]."""
    n, twice, h, w = x.shape
    c = twice // 2
    if twice % 2 or weight.shape != (9, twice) or bias.shape != (twice,):
        msg = "Expected an even input width, [9, 2C] weights and [2C] biases"
        raise ValueError(msg)
    if any(v.dtype != torch.float16 or v.device != x.device for v in (x, weight, bias)):
        msg = "Expected fp16 activations and weights on the same device"
        raise ValueError(msg)
    out = torch.empty((n, c, h, w), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    load(x.device).launch(
        "depthwise_gate",
        *linear(out.numel()),
        x.contiguous(memory_format=torch.channels_last),
        weight.contiguous(),
        bias.contiguous(),
        out,
        h,
        w,
        c,
        out.numel(),
    )
    return out
