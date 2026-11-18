"""DINOv2 on Apple MLX (`device="mlx"`): every hub variant (ViT-S / B / L / g, with and without registers).

The maths of `model.py` (hub configuration, upstream 7764ea0 position interpolation), with the outputs of the torch
runners (blocks: `fastforks.shared.mlx_vit`): `last_hidden_state`, and the hidden states HF numbers 0 (embeddings) to
L (after block L, no norm).

What is prepared once, at conversion (exact in real arithmetic, done in float64, stored float32):
- LayerScale folded into `attn.proj` and the FFN's last layer (weights and biases times gamma).
- The patch embedding as one matmul on unfolded patches (`[14 * 14 * 3, C]`).
- The cls token merged with its position; all linear weights stored transposed (`[in, out]`).

Per grid size (cached): the position table, interpolated in float32 by the same PyTorch call as the torch runners
(`model._pos_embed`), with the patch embedding's bias added, so the embedding is `patches @ W + table`.

float16: weights and activations fp16; the position table is built in fp32; LayerNorm statistics are fp32 inside
`mx.fast.layer_norm`'s kernel and the attention's softmax accumulates in fp32 inside MLX's fused attention; the
residual stream is fp16, as in the torch fp16 runner. Outputs are float32.

There is no kernels mode: a Metal erf-GELU kernel, `addmm`, token padding and bfloat16 gave nothing measurable on the
M1 over MLX's own operations (the time is in matmul and attention).
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING

from fastforks._core.mlx import converted
from fastforks.shared.mlx_vit import DinoFeatureEngine, dinov2_params

from .model import _pos_embed

if TYPE_CHECKING:
    from fastforks.features._runner import Context

HEADS = {"vits": 6, "vitb": 12, "vitl": 16, "vitg": 24}
LAYOUT = "v1"


class Engine(DinoFeatureEngine):
    """DINOv2 features with the hub checkpoint's position interpolation."""

    def __init__(self, ctx: Context) -> None:
        """Convert or read the cached DINOv2 weights."""
        params = converted(ctx, LAYOUT, lambda: dinov2_params(ctx.state_dict(), 14))
        registers = ctx.params.registers
        interp = SimpleNamespace(interpolate_offset=0.0 if registers else 0.1, interpolate_antialias=bool(registers))
        super().__init__(params, ctx.dtype, HEADS[ctx.params.arch], partial(_pos_embed, interp))
