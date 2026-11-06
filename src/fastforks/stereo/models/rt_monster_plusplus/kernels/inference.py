"""Kernels inference path for RT-MonSter++ (general: any input size /32, no per-size compilation / tuning).

The portable path (`net/inference.py` `forward_fast`: fp16 weights, fp16 autocast, NHWC) on the shared kernels:
  - the Depth Anything V2 ViT-S as the fp16 DINOv2 of `fastforks.shared.depth_anything_v2` (`use_fast_vit`:
    fp16-accumulate GEMMs, `fastbb` flash attention)
  - every other conv as an fp16-accumulate cuDNN graph (`cudnn_hacc`) or CUTLASS conv with the bias / BatchNorm /
    activation in the epilogue (`cutlass_conv`, `igev`; `fastforks.shared.cnn_kernels.prepare_generic`: conv -> BN -> act
    runs as one kernel). Inputs arrive as autocast would pass them to a conv (fp32 disparity / lookup features are
    cast to fp16; no conv runs in the fp32 regions)
The disparity states, the lookups, the mono scale / shift fit, the warps and the upsampling stay fp32 as portable.
"""

import torch
from torch import nn

from fastforks._core.patch import override
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import conv_ep
from fastforks.shared.depth_anything_v2 import use_fast_vit

from ..net.inference import forward_fast, prepare
from ..net.monster import Monster


def _conv_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """cnn_kernels conv with autocast's input cast (fp16, channels-last)."""
    return conv_ep(self, x.half().contiguous(memory_format=cnn_kernels.memory_format(x.dim())))


@torch.no_grad()
def prepare_kernels(model: Monster, device: torch.device) -> Monster:
    """The portable fp16 preparation, the fp16 DINOv2 and the convs on the fp16-accumulate kernels.

    Raises:
        KernelUnavailableError: A kernel library (or cuDNN) cannot be used on `device`.
        RuntimeError: No conv was routed through the kernels (the module tree changed).
    """
    cnn_kernels.load(device)
    prepare(model, half=True, channels_last=True)
    use_fast_vit(model.mono_encoder)
    for name, child in model.named_children():
        if name != "mono_encoder":
            cnn_kernels.prepare_generic(child, device)
    n = 0
    for m in model.modules():
        if getattr(m.forward, "__func__", None) is cnn_kernels._conv_forward:
            override(m, _conv_forward)  # replaces cnn_kernels's override
            n += 1
    if not n or not any(getattr(m, "_hacc", False) for m in model.modules()):
        msg = "RT-MonSter++ kernels: no conv was routed through the fp16-accumulate kernels"
        raise RuntimeError(msg)
    return model


def forward_kernels(model: Monster, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The portable test-mode forward; returns the disparity [B, H, W] (fp32)."""
    return forward_fast(model, left, right)[:, 0]
