"""FocalNet (timm `FocalNet`, focal modulation) written in MLX, channels last.

The exported-graph translation took 0.8-1.5 s of the first call (export, translation, segment tuning); written out,
the first call is `mx.compile`'s trace and the shader builds only. The 1x1 convolutions are matmuls over the channels;
LayerScale (if the checkpoint has it) folds into the projection before it; GELU and the global context's mean are
computed in float32.
"""

# ruff: noqa: ANN401 - timm's architecture-specific layers

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from fastforks._core.mlx import mx
from fastforks.shared.libnet.mlx_cnn import _chain, _conv, _linear, _norm

if TYPE_CHECKING:
    from collections.abc import Callable


def _gelu(x: mx.array) -> mx.array:
    """Exact (erf) GELU computed in float32 (`1 + erf` of a negative input cancels in float16)."""
    y = x.astype(mx.float32)
    return (y * (0.5 + 0.5 * mx.erf(y * 0.7071067811865476))).astype(x.dtype)


def _maybe_norm(layer: nn.Module, dtype: mx.Dtype) -> Callable | None:
    return None if isinstance(layer, nn.Identity) else _norm(layer, dtype)


def _gamma(layer: Any) -> Any:
    return None if isinstance(layer, nn.Identity) else layer.gamma.detach()


def _modulation(mod: Any, dtype: mx.Dtype, gamma: Any) -> Callable:
    """`FocalModulation`: per-level depthwise context with gates, a gated global context, `q * h(context)`."""
    c, levels = mod.dim, mod.focal_level
    f, h = _linear(mod.f, dtype), _linear(mod.h, dtype)
    focal = [_conv(layer[0], None, dtype) for layer in mod.focal_layers]
    post = _maybe_norm(mod.norm, dtype)
    proj = _linear(mod.proj, dtype, gamma)
    scale = 1.0 / (levels + 1) if mod.normalize_modulator else None

    def apply(x: mx.array) -> mx.array:
        y = f(x)
        q, ctx, gates = y[..., :c], y[..., c : 2 * c], y[..., 2 * c :]
        total = None
        for level, conv in enumerate(focal):
            ctx = _gelu(conv(ctx))
            term = ctx * gates[..., level : level + 1]
            total = term if total is None else total + term
        glob = _gelu(mx.mean(ctx.astype(mx.float32), axis=(1, 2), keepdims=True)).astype(ctx.dtype)
        total = glob * gates[..., levels:] if total is None else total + glob * gates[..., levels:]
        if scale is not None:
            total = total * scale
        out = q * h(total)
        return proj(post(out) if post is not None else out)

    return apply


def _block(block: Any, dtype: mx.Dtype) -> Callable:
    """`FocalNetBlock`: pre-norm (or post-norm) modulation and MLP residuals, LayerScale folded when it can be."""
    norm1, norm2 = _maybe_norm(block.norm1, dtype), _maybe_norm(block.norm2, dtype)
    post1, post2 = _maybe_norm(block.norm1_post, dtype), _maybe_norm(block.norm2_post, dtype)
    g1, g2 = _gamma(block.ls1), _gamma(block.ls2)
    # LayerScale folds into the last projection unless a post-norm sits between them
    modulate = _modulation(block.modulation, dtype, g1 if post1 is None else None)
    mlp = block.mlp
    fc1, fc2 = _linear(mlp.fc1, dtype), _linear(mlp.fc2, dtype, g2 if post2 is None else None)
    if not isinstance(mlp.norm, nn.Identity):
        msg = "FocalNet MLP norm has no MLX port"
        raise NotImplementedError(msg)
    s1 = None if post1 is None or g1 is None else mx.array(g1.numpy()).astype(dtype)
    s2 = None if post2 is None or g2 is None else mx.array(g2.numpy()).astype(dtype)

    def branch(y: mx.array, post: Callable | None, s: mx.array | None) -> mx.array:
        if post is not None:
            y = post(y)
        return y * s if s is not None else y

    def apply(x: mx.array) -> mx.array:
        x = x + branch(modulate(norm1(x) if norm1 is not None else x), post1, s1)
        y = fc2(_gelu(fc1(norm2(x) if norm2 is not None else x)))
        return x + branch(y, post2, s2)

    return apply


def _downsample(layer: Any, dtype: mx.Dtype) -> Callable:
    conv = _conv(layer.proj, None, dtype)
    norm = _maybe_norm(layer.norm, dtype)
    return conv if norm is None else _chain(conv, norm)


def focalnet(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a timm `FocalNet`: NHWC images -> normalised map, pooled `[B, C]` -> logits."""
    if not isinstance(net.norm, nn.Identity) and type(net.head).__name__ != "ClassifierHead":
        msg = "FocalNet with an MLP head has no MLX port"
        raise NotImplementedError(msg)
    ops = [_downsample(net.stem, dtype)]
    for stage in net.layers:
        if not isinstance(stage.downsample, nn.Identity):
            ops.append(_downsample(stage.downsample, dtype))
        ops.extend(_block(block, dtype) for block in stage.blocks)
    if not isinstance(net.norm, nn.Identity):
        ops.append(_norm(net.norm, dtype))
    return _chain(*ops), _linear(net.head.fc, dtype)
