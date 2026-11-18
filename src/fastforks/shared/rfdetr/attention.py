"""Packed DINOv2 attention projections for fixed RF-DETR inference networks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class PackedAttention(nn.Module):
    """Project Q, K and V in one GEMM and attend using PyTorch SDPA."""

    def __init__(self, source: Any) -> None:  # noqa: ANN401 - vendored DINOv2 attention
        """Pack the three loaded projections in query, key, value order."""
        super().__init__()
        self.heads = source.num_attention_heads
        self.dim = source.all_head_size
        self.head_dim = source.attention_head_size
        w = source.query.weight
        layers = (source.query, source.key, source.value)
        self.qkv = nn.Linear(
            w.shape[1], 3 * self.dim, bias=source.query.bias is not None, device=w.device, dtype=w.dtype
        )
        with torch.no_grad():
            self.qkv.weight.copy_(torch.cat([layer.weight for layer in layers]))
            if self.qkv.bias is not None:
                self.qkv.bias.copy_(torch.cat([layer.bias for layer in layers]))

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> tuple[torch.Tensor, None]:  # noqa: FBT001, FBT002 - upstream signature
        """Return the projected context in upstream batch, token, embedding order."""
        if output_attentions:
            msg = "Packed RF-DETR inference does not return attention weights"
            raise ValueError(msg)
        b, n, _ = hidden_states.shape
        q, k, v = self.qkv(hidden_states).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v)
        return y.transpose(1, 2).reshape(b, n, self.dim), None


def pack_attention(net: nn.Module) -> None:
    """Replace the independent backbone projections after loading the upstream checkpoint."""
    from .net.models.backbone.dinov2_with_windowed_attn import (  # noqa: PLC0415 - optional transformers
        Dinov2WithRegistersAttention,
    )

    for module in net.modules():
        if isinstance(module, Dinov2WithRegistersAttention):
            module.set_submodule("attention", PackedAttention(module.attention))
