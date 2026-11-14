"""Packed-QKV fastbb attention for RF-DETR's windowed DINOv2 backbone, fvm_conv depthwise for its Seg head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import KernelUnavailableError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import base_type, override
from fastforks._optimize.conv_ops import dwconv
from fastforks.kernels.fastbb import Flash64
from fastforks.shared.rfdetr.build import export_path

LIBRARY = "fastbb"
HEAD_DIM = 64  # fixed: fastbb's supported head dimension


class _Attention(nn.Module):
    """One packed projection and fp32-accumulating flash attention, preserving the upstream tuple API."""

    def __init__(self, source: Any) -> None:  # noqa: ANN401 - vendored attention module
        super().__init__()
        self.heads = source.num_attention_heads
        self.dim = source.all_head_size
        if source.attention_head_size != HEAD_DIM:
            msg = f"RF-DETR fastbb needs head dimension {HEAD_DIM}, got {source.attention_head_size}"
            raise KernelUnavailableError(LIBRARY, msg)
        w = source.query.weight
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=source.query.bias is not None, device=w.device, dtype=w.dtype)
        with torch.no_grad():
            self.qkv.weight.copy_(torch.cat([source.query.weight, source.key.weight, source.value.weight]))
            if self.qkv.bias is not None:
                self.qkv.bias.copy_(torch.cat([source.query.bias, source.key.bias, source.value.bias]))
        self.flash = Flash64("ff")

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> tuple[torch.Tensor, None]:  # noqa: FBT001, FBT002 - upstream attention signature
        """Return context tokens; attention-weight output is unsupported by the pinned windowed backbone."""
        if output_attentions:
            msg = "RF-DETR fastbb does not return attention weights"
            raise KernelUnavailableError(LIBRARY, msg)
        batch, tokens, _ = hidden_states.shape
        qkv = F.linear(hidden_states, self.qkv.weight, self.qkv.bias).view(batch, tokens, 3, self.heads, HEAD_DIM)
        context = self.flash(*qkv.unbind(2), HEAD_DIM**-0.5)
        return context.reshape(batch, tokens, self.dim), None


def use_fast_attention(net: nn.Module) -> None:
    """Replace only the DINOv2 self-attention projections/attention, after loading the original checkpoint."""
    from .net.models.backbone.dinov2_with_windowed_attn import (  # noqa: PLC0415 - optional transformers
        Dinov2WithRegistersAttention,
    )

    count = 0
    for module in net.modules():
        if isinstance(module, Dinov2WithRegistersAttention):
            module.set_submodule("attention", _Attention(module.attention))
            count += 1
    if not count:
        msg = "RF-DETR backbone contains no supported DINOv2 attention modules"
        raise KernelUnavailableError(LIBRARY, msg)


def _block(self: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - vendored DepthwiseConvBlock
    """`DepthwiseConvBlock.forward` in channels-last memory: fvm_conv depthwise, no layout copies around the norm."""
    y = dwconv(x, self.w_rsc, self.dwconv.bias, [1, 1], [1, 1], 0).permute(0, 2, 3, 1)
    y = self.act(self.pwconv1(self.norm(y)))
    if self.gamma is not None:
        y = self.gamma * y
    return y.permute(0, 3, 1, 2) + x


def use_fast_blocks(net: nn.Module) -> None:
    """Run the segmentation head's depthwise blocks (upstream: PyTorch's non-cuDNN depthwise kernel) by `_block`."""
    from .net.models.heads.segmentation import DepthwiseConvBlock  # noqa: PLC0415 - optional transformers

    for module in net.modules():
        if isinstance(module, DepthwiseConvBlock) and module.dwconv.groups % 8 == 0:
            module.register_buffer(
                "w_rsc", module.dwconv.weight.detach()[:, 0].permute(1, 2, 0).contiguous(), persistent=False
            )
            override(module, _block)


def _encoder(self: nn.Module, x: torch.Tensor) -> Any:  # noqa: ANN401 - vendored DINOv2 feature list
    """The ViT encoder with fp16-accumulate GEMMs (2x the fp32-accumulate rate on GeForce)."""
    with fp16_accumulate():
        return base_type(self).forward(self, x)


def use_fp16_accumulate(net: Any) -> None:  # noqa: ANN401 - lazy vendored LWDETR API
    """Accumulate the DINOv2 GEMMs in fp16; the decoder and heads keep fp32 accumulation (masks and boxes drift)."""
    override(net.backbone[0].encoder, _encoder)


def prepare_graph(net: Any) -> None:  # noqa: ANN401 - lazy vendored LWDETR API
    """Select the authors' graph-safe inference path without bound-method reference cycles."""
    net.transformer.enable_cuda_graph_capture()
    export_path(net)
