"""Apple MLX C-RADIOv3 feature encoder with the original square-then-crop position interpolation."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx, to_mx, to_torch, tuning
from fastforks.shared.mlx_vit import blocks, patches, transformer_params

from .model import _PATCH, _resample

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from fastforks.features._runner import Context


class Engine:
    """Converted weights, per-grid position tables and forwards for the requested hidden states."""

    def __init__(self, net: Any, pos: torch.Tensor, indices: list[int], ctx: Context) -> None:  # noqa: ANN401 - timm ViT
        """Prepare weights for channels-last patches and attention, keeping positions in float32 for resampling."""
        heads, depth, width = ctx.params.num_heads, ctx.params.depth, ctx.params.embed_dim
        eps = net.blocks[0].norm1.eps
        state = net.state_dict()
        dtype = getattr(mx, ctx.dtype)
        prefix = to_mx(state.pop("cls_token")).astype(dtype)
        patch = state.pop("patch_embed.proj.weight").permute(2, 3, 1, 0).reshape(-1, width)
        weight = to_mx(patch).astype(dtype)
        params = {k: v.astype(dtype) for k, v in transformer_params(state).items()}
        summary = mx.array(indices)
        mx.eval(params, prefix, weight, summary)

        @lru_cache(maxsize=8)
        def table(h: int, w: int) -> mx.array:
            grid = _resample(pos, h, w, align_corners=ctx.params.pos_align_corners)
            return to_mx(grid).astype(dtype)

        @lru_cache(maxsize=8)
        def program(layers: tuple[int, ...]) -> Callable[..., tuple[mx.array, ...]]:
            def run(image: mx.array, positions: mx.array) -> tuple[mx.array, ...]:
                batch = image.shape[0]
                tokens = patches(image.astype(dtype), _PATCH) @ weight + positions
                pre = mx.broadcast_to(prefix, (batch, *prefix.shape[1:]))
                last, hidden = blocks(params, depth, heads, eps, mx.concatenate([pre, tokens], axis=1), layers)
                pooled = mx.take(last, summary, axis=1).reshape(batch, -1)
                return (last, pooled, mx.stack(hidden, axis=1)) if layers else (last, pooled)

            return mx.compile(run) if tuning().compile_steps else run

        self.table = table
        self.program = program

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Return the checkpoint's teacher summaries and tokens for a normalised image batch."""
        h, w = image.shape[-2:]
        arrays = self.program(layers)(to_mx(image), self.table(h // _PATCH, w // _PATCH))
        mx.eval(*arrays)
        keys = (
            ("last_hidden_state", "pooler_output", "hidden_states")
            if layers
            else ("last_hidden_state", "pooler_output")
        )
        return {key: to_torch(arr) for key, arr in zip(keys, arrays, strict=True)}
