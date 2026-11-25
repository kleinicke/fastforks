"""Apply the CUDA row LayerNorm to large half-precision restoration token maps."""

import torch
from torch import nn

from fastforks._core.patch import override
from fastforks.kernels import sr_norm

_MIN_ELEMENTS = 65536


def _aligned_half(x: torch.Tensor, affine: torch.Tensor) -> bool:
    return (
        affine.dtype == x.dtype and affine.device == x.device and affine.is_contiguous() and affine.data_ptr() % 8 == 0
    )


def _layer_norm(m: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    torch_order = m.normalized_shape == (sr_norm.TORCH_ORDER_CHANNELS,)
    if (
        x.is_cuda
        and x.dtype == torch.float16
        and x.ndim > 0
        and x.shape[-1:] == m.normalized_shape
        and x.numel() > _MIN_ELEMENTS
        and x.is_contiguous()
        and torch.is_inference_mode_enabled()
        and not torch.compiler.is_compiling()
        and x.data_ptr() % (16 if not torch_order and x.shape[-1] % 8 == 0 else 8) == 0
    ):
        if not torch_order:
            return sr_norm.layer_norm(x, m.get_buffer("_sr_weight"), m.get_buffer("_sr_bias"), m.eps)
        weight, bias = m.weight, m.bias
        if weight is not None and bias is not None and _aligned_half(x, weight) and _aligned_half(x, bias):
            return sr_norm.layer_norm_240(x, weight, bias, m.eps)
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
            if m.normalized_shape != (sr_norm.TORCH_ORDER_CHANNELS,):
                m.register_buffer("_sr_weight", m.weight.detach().float(), persistent=False)
                m.register_buffer("_sr_bias", m.bias.detach().float(), persistent=False)
            override(m, _layer_norm)
