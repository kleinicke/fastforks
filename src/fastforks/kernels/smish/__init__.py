"""Fused fp16 Smish activation for TED."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"smish": "ppl"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


class Activation(torch.nn.Module):
    """Smish on a dense CUDA fp16 tensor."""

    def __init__(self, device: torch.device) -> None:
        """Load the CUDA kernel for the network device."""
        super().__init__()
        self.lib = load(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute x * tanh(log(1 + sigmoid(x))) in one pass."""
        if x.dtype != torch.float16:
            msg = "The fused Smish kernel expects fp16 input"
            raise ValueError(msg)
        if x.device != self.lib.device:
            msg = f"The fused Smish kernel is bound to {self.lib.device}, got input on {x.device}"
            raise ValueError(msg)
        x = x.contiguous()
        out = torch.empty_like(x)
        self.lib.launch("smish", *linear(x.numel()), x, out, x.numel())
        return out
