"""Packed-QKV fastbb attention for RF-DETR's windowed DINOv2 backbone."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import KernelUnavailableError
from fastforks._core.patch import base_type, override
from fastforks.kernels.fastbb import Flash64

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


def prepare_graph(net: Any) -> None:  # noqa: ANN401 - lazy vendored LWDETR API
    """Select the authors' graph-safe inference path without bound-method reference cycles."""
    net.transformer.enable_cuda_graph_capture()
    net.export()
    for module in net.modules():
        if "_forward_origin" in module.__dict__:
            # Upstream export saves its bound forward and installs a bound forward_export.
            # In inference no code restores that saved method; class overrides preserve eager destruction.
            override(module, base_type(module).forward_export)
            del module.__dict__["_forward_origin"]
