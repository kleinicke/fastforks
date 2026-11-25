"""Apply the CUDA row LayerNorm to large half-precision restoration token maps."""

import torch
from torch import nn

from fastforks._core.patch import override
from fastforks.kernels import sr_norm

_MIN_ELEMENTS = 65536


def _layer_norm(m: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    if (
        x.is_cuda
        and x.dtype == torch.float16
        and x.ndim > 0
        and x.shape[-1:] == m.normalized_shape
        and x.numel() > _MIN_ELEMENTS
        and x.is_contiguous()
        and torch.is_inference_mode_enabled()
        and not torch.compiler.is_compiling()
        and x.data_ptr() % (16 if x.shape[-1] % 8 == 0 else 8) == 0
    ):
        return sr_norm.layer_norm(x, m.get_buffer("_sr_weight"), m.get_buffer("_sr_bias"), m.eps)
    return nn.LayerNorm.forward(m, x)


def fast_layer_norm(net: nn.Module, device: torch.device) -> None:
    """Install the row kernel on affine, one-dimensional LayerNorm layers."""
    sr_norm.load(device)
    for m in net.modules():
        if (
            isinstance(m, nn.LayerNorm)
            and m.weight is not None
            and m.bias is not None
            and len(m.normalized_shape) == 1
            and sr_norm.supported(m.normalized_shape[0])
        ):
            m.register_buffer("_sr_weight", m.weight.detach().float(), persistent=False)
            m.register_buffer("_sr_bias", m.bias.detach().float(), persistent=False)
            override(m, _layer_norm)
