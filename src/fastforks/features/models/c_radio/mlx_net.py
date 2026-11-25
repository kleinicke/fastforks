"""Apple MLX C-RADIOv3 / v4 feature encoder with the original square-then-crop position interpolation.

Hand-written on `fastforks.shared.mlx_vit` (no timm network is built). The checkpoint is validated and converted once
(LayerScale folded in float64, linear weights `[in, out]`, the patch embedding as one matmul) and cached as float32
safetensors (`fastforks._core.mlx.converted`); a load reads that file lazily and casts array by array, so only the
run-dtype copy is ever resident (ViT-g float16: 2.3 GB instead of the 4.6 GB float32 state beside it).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.errors import CheckpointMismatchError
from fastforks._core.mlx import converted, mx, to_mx, to_torch
from fastforks.shared.mlx_vit import blocks, depth_of, patches, transformer_params

from .model import _PATCH, _checkpoint, _resample

if TYPE_CHECKING:
    from fastforks.features._runner import Context

LAYOUT = "v1"  # change with `_convert` or `mlx_vit.transformer_params`
_EPS = 1e-6  # timm's VisionTransformer LayerNorm


def params(
    state: dict[str, torch.Tensor], pos: torch.Tensor, tok: torch.Tensor, indices: list[int]
) -> dict[str, mx.array]:
    """`model._checkpoint`'s tensors as float32 arrays: `mlx_vit`'s blocks, `prefix`, `patch.weight`, `pos`, `summary`.

    Takes its keys out of `state`; what is left is not part of the network.
    """
    patch = state.pop("patch_embed.proj.weight")
    patch = patch.permute(2, 3, 1, 0).reshape(-1, patch.shape[0])
    out = transformer_params(state)
    out["prefix"] = to_mx(tok.float())
    out["patch.weight"] = to_mx(patch.float())
    out["pos"] = to_mx(pos.float())
    out["summary"] = mx.array(indices, dtype=mx.int32)
    return out


def _convert(ctx: Context) -> dict[str, mx.array]:
    state, pos, tok, indices = _checkpoint(ctx)
    out = params(state, pos, tok, indices)
    if state or depth_of(out) != ctx.params.depth:
        what, msg = f"{ctx.variant.ref} checkpoint", f"{depth_of(out)} blocks, unexpected keys {sorted(state)[:8]}"
        raise CheckpointMismatchError(what, msg)
    return out


def engine(ctx: Context) -> Engine:
    """The engine of `ctx.variant`: its weights converted once, then read from the cache."""
    p = ctx.params
    return Engine(converted(ctx, LAYOUT, lambda: _convert(ctx)), p.num_heads, ctx.dtype, align=p.pos_align_corners)


class Engine:
    """Converted weights, per-grid position tables and forwards for the requested hidden states.

    Attributes:
        prefix_tokens: Summary plus register tokens before the patches.
        summaries: Summary tokens concatenated into `pooler_output`.
    """

    def __init__(self, p32: dict[str, mx.array], heads: int, dtype_name: str, *, align: bool) -> None:
        """Take `params`' arrays (emptied) into the run dtype; positions stay float32 on the host for resampling."""
        dtype = getattr(mx, dtype_name)
        pos = torch.from_numpy(np.array(p32.pop("pos")))
        summary = p32.pop("summary")
        params: dict[str, mx.array] = {}
        for key in list(p32):  # one by one: a lazy cast keeps its float32 source until it is evaluated
            params[key] = p32.pop(key).astype(dtype)
            mx.eval(params[key])
        prefix, weight = params.pop("prefix")[None], params.pop("patch.weight")
        depth = depth_of(params)
        self.prefix_tokens, self.summaries = int(prefix.shape[1]), int(summary.shape[0])

        @lru_cache(maxsize=8)
        def table(h: int, w: int) -> mx.array:
            return to_mx(_resample(pos, h, w, align_corners=align)).astype(dtype)

        def run(image: mx.array, positions: mx.array, layers: tuple[int, ...]) -> tuple[mx.array, ...]:
            batch = image.shape[0]
            tokens = patches(image.astype(dtype), _PATCH) @ weight + positions
            tokens = mx.concatenate([mx.broadcast_to(prefix, (batch, *prefix.shape[1:])), tokens], axis=1)
            last, hidden = blocks(params, depth, heads, _EPS, tokens, layers, bounded=True)
            pooled = mx.take(last, summary, axis=1).reshape(batch, -1)
            return (last, pooled, mx.stack(hidden, axis=1)) if layers else (last, pooled)

        self.table = table
        self.run = run

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Return the checkpoint's teacher summaries and tokens for a normalised image batch."""
        h, w = image.shape[-2:]
        arrays = self.run(to_mx(image), self.table(h // _PATCH, w // _PATCH), tuple(layers))
        mx.eval(*arrays)
        keys = (
            ("last_hidden_state", "pooler_output", "hidden_states")
            if layers
            else ("last_hidden_state", "pooler_output")
        )
        return {key: to_torch(arr) for key, arr in zip(keys, arrays, strict=True)}
