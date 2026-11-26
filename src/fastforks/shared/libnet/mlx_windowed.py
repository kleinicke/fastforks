"""Windowed hierarchical vision transformers (timm Swin and DaViT) written in MLX, channels last.

The exported-graph translation spent 1.1-2.5 s of the first call exporting and translating; written out, the first
call is `mx.compile`'s trace and the shader builds only. The input sizes are the cards' fixed crops, so every
window's attention bias (Swin: the relative position bias plus the shifted-window mask; DaViT has none) is gathered
once at load, in the run dtype. Window attention is `mx.fast.scaled_dot_product_attention` over
`[B * windows, heads, N, d]` (Swin: `[B, windows * heads, N, d]`, so its per-window mask broadcasts over the batch);
DaViT's channel attention (`softmax(k^T v)` over the channels, then `q @ attn^T`) is two matmuls, the first and the
softmax in float32. BatchNorm folds into the convolutions (`mlx_cnn._conv`); GELU is `mlx_vit.gelu` (float32
inside, as torch rounds once).
"""

# ruff: noqa: ANN401 - timm's architecture-specific layers

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from fastforks._core.mlx import attention, mx, to_mx
from fastforks.shared.libnet.mlx_cnn import _chain, _conv, _conv_k, _norm
from fastforks.shared.mlx_vit import gelu

if TYPE_CHECKING:
    from collections.abc import Callable


def _dense(layer: nn.Linear, dtype: mx.Dtype) -> Callable:
    w = to_mx(layer.weight.detach().T).astype(dtype)
    if layer.bias is None:
        mx.eval(w)
        return lambda x: x @ w
    b = to_mx(layer.bias.detach()).astype(dtype)
    mx.eval(w, b)
    return lambda x: mx.addmm(b, x, w)


def _mlp(mlp: Any, dtype: mx.Dtype) -> Callable:
    """The timm `Mlp` (or TinyViT's `NormMlp`, its LayerNorm first) with an exact GELU."""
    if type(mlp.act).__name__ != "GELU" or getattr(mlp.act, "approximate", "none") != "none":
        msg = f"unsupported MLP activation {type(mlp.act).__name__}"
        raise NotImplementedError(msg)
    norm = None if isinstance(mlp.norm, nn.Identity) else _norm(mlp.norm, dtype)
    fc1, fc2 = _dense(mlp.fc1, dtype), _dense(mlp.fc2, dtype)

    def apply(x: mx.array) -> mx.array:
        return fc2(gelu(fc1(norm(x) if norm is not None else x)))

    return apply


def _pad_to(x: mx.array, win: int) -> mx.array:
    """`[B, H, W, C]` zero-padded at the bottom / right to multiples of `win` (as timm pads before partitioning)."""
    ph, pw = -x.shape[1] % win, -x.shape[2] % win
    return mx.pad(x, [(0, 0), (0, ph), (0, pw), (0, 0)]) if ph or pw else x


def _windows(x: mx.array, win: int) -> mx.array:
    """`[B, H, W, C]` (multiples of `win`) -> `[B, windows, win * win, C]`, windows row-major."""
    b, h, w, c = x.shape
    x = x.reshape(b, h // win, win, w // win, win, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, (h // win) * (w // win), win * win, c)


def _unwindows(x: mx.array, win: int, h: int, w: int) -> mx.array:
    """`_windows`' inverse: `[B, windows, win * win, C]` -> `[B, h, w, C]`."""
    b, _, _, c = x.shape
    x = x.reshape(b, h // win, w // win, win, win, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, w, c)


def _split_qkv(qkv: mx.array, heads: int) -> tuple[mx.array, mx.array, mx.array]:
    """`[B, windows, N, 3 C]` -> q, k, v `[B, windows, heads, N, d]`."""
    b, nw, n, c3 = qkv.shape
    qkv = qkv.reshape(b, nw, n, 3, heads, c3 // (3 * heads)).transpose(3, 0, 1, 4, 2, 5)
    return qkv[0], qkv[1], qkv[2]


def _window_attention(
    qkv: Callable, proj: Callable, heads: int, bias: mx.array | None, *, per_window: bool
) -> Callable:
    """Windows `[B, windows, N, C]` -> attention output `[B, windows, N, C]`.

    `bias`: the additive attention bias, `[windows * heads, N, N]` if `per_window` (Swin's shifted-window mask),
    else `[heads, N, N]`.
    """

    def apply(x: mx.array) -> mx.array:
        b, nw, n, _ = x.shape
        q, k, v = _split_qkv(qkv(x), heads)
        d = q.shape[-1]
        if per_window:  # windows into the heads: the mask broadcasts over the batch
            q, k, v = (t.reshape(b, nw * heads, n, d) for t in (q, k, v))
        else:
            q, k, v = (t.reshape(b * nw, heads, n, d) for t in (q, k, v))
        y = attention(q, k, v, scale=d**-0.5, mask=bias)
        y = y.reshape(b, nw, heads, n, d).transpose(0, 1, 3, 2, 4).reshape(b, nw, n, heads * d)
        return proj(y)

    return apply


# --- Swin ---------------------------------------------------------------------------------------------------------


def _swin_bias(block: Any, dtype: mx.Dtype) -> mx.array:
    """The block's relative position bias plus its shifted-window mask, `[windows * heads, N, N]` (or `[heads, ..]`)."""
    at = block.attn
    n = at.window_area
    bias = at.relative_position_bias_table.detach()[at.relative_position_index.view(-1)].view(n, n, -1)
    bias = bias.permute(2, 0, 1).float()  # [heads, N, N]
    if block.attn_mask is not None:  # [windows, N, N] of 0 / -100
        bias = (bias[None] + block.attn_mask.float()[:, None]).flatten(0, 1)
    out = to_mx(bias.contiguous()).astype(dtype)
    mx.eval(out)
    return out


def _swin_block(block: Any, dtype: mx.Dtype) -> Callable:
    if getattr(block, "dynamic_mask", False):
        msg = "Swin with a dynamic mask has no MLX port"
        raise NotImplementedError(msg)
    win = block.window_size[0]
    if block.window_size[0] != block.window_size[1] or block.shift_size[0] != block.shift_size[1]:
        msg = "Swin with non-square windows has no MLX port"
        raise NotImplementedError(msg)
    shift = block.shift_size[0]
    norm1, norm2 = _norm(block.norm1, dtype), _norm(block.norm2, dtype)
    attn = _window_attention(
        _dense(block.attn.qkv, dtype),
        _dense(block.attn.proj, dtype),
        block.attn.num_heads,
        _swin_bias(block, dtype),
        per_window=block.attn_mask is not None,
    )
    mlp = _mlp(block.mlp, dtype)
    res = tuple(block.input_resolution)

    def apply(x: mx.array) -> mx.array:
        _, h, w, _ = x.shape
        if (h, w) != res:
            msg = f"the Swin block takes {res} maps, not {(h, w)}"
            raise ValueError(msg)
        y = norm1(x)
        if shift:
            y = mx.roll(y, shift=(-shift, -shift), axis=(1, 2))
        y = _pad_to(y, win)
        y = _unwindows(attn(_windows(y, win)), win, y.shape[1], y.shape[2])[:, :h, :w]
        if shift:
            y = mx.roll(y, shift=(shift, shift), axis=(1, 2))
        x = x + y
        return x + mlp(norm2(x))

    return apply


def _swin_merge(layer: Any, dtype: mx.Dtype) -> Callable:
    """`PatchMerging`: each 2 x 2 neighbourhood's channels (column-major: (0,0), (1,0), (0,1), (1,1)), norm, matmul."""
    norm, reduction = _norm(layer.norm, dtype), _dense(layer.reduction, dtype)

    def apply(x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (0, x.shape[1] % 2), (0, x.shape[2] % 2), (0, 0)])
        b, h, w, c = x.shape
        x = x.reshape(b, h // 2, 2, w // 2, 2, c).transpose(0, 1, 3, 4, 2, 5).reshape(b, h // 2, w // 2, 4 * c)
        return reduction(norm(x))

    return apply


def swin(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a timm `SwinTransformer`: NHWC images -> normalised map, pooled `[B, C]` -> logits."""
    pe = net.patch_embed
    if getattr(pe, "output_fmt", None) is not None and str(pe.output_fmt) not in {"NHWC", "Format.NHWC"}:
        msg = "Swin patch embedding must be NHWC"
        raise NotImplementedError(msg)
    ops: list[Callable] = [_conv(pe.proj, None, dtype)]
    if not isinstance(pe.norm, nn.Identity):
        ops.append(_norm(pe.norm, dtype))
    for stage in net.layers:
        if not isinstance(stage.downsample, nn.Identity):
            ops.append(_swin_merge(stage.downsample, dtype))
        ops.extend(_swin_block(block, dtype) for block in stage.blocks)
    ops.append(_norm(net.norm, dtype))
    return _chain(*ops), _dense(net.head.fc, dtype)


# --- DaViT --------------------------------------------------------------------------------------------------------


def _cpe(layer: Any, dtype: mx.Dtype) -> Callable:
    """`ConvPosEnc`: `x + act(depthwise conv(x))`, the depthwise conv `graph_mlx`'s kernel."""
    conv = _conv_k(layer.proj, None, dtype)
    act = None if isinstance(layer.act, nn.Identity) else gelu
    return lambda x: x + (act(conv(x)) if act is not None else conv(x))


def _davit_spatial(block: Any, dtype: mx.Dtype) -> Callable:
    win = block.window_size[0]
    cpe1, cpe2, norm1 = _cpe(block.cpe1, dtype), _cpe(block.cpe2, dtype), _norm(block.norm1, dtype)
    attn = _window_attention(
        _dense(block.attn.qkv, dtype), _dense(block.attn.proj, dtype), block.num_heads, None, per_window=False
    )
    mlp = None if block.mlp is None else (_norm(block.norm2, dtype), _mlp(block.mlp, dtype))

    def apply(x: mx.array) -> mx.array:
        _, h, w, _ = x.shape
        x = cpe1(x)
        y = _pad_to(norm1(x), win)
        x = x + _unwindows(attn(_windows(y, win)), win, y.shape[1], y.shape[2])[:, :h, :w]
        x = cpe2(x)
        return x if mlp is None else x + mlp[1](mlp[0](x))

    return apply


def _davit_channel(block: Any, dtype: mx.Dtype) -> Callable:
    if type(block.attn).__name__ != "ChannelAttention":
        msg = f"DaViT {type(block.attn).__name__} has no MLX port"
        raise NotImplementedError(msg)
    heads = block.attn.num_heads
    cpe1, cpe2, norm1 = _cpe(block.cpe1, dtype), _cpe(block.cpe2, dtype), _norm(block.norm1, dtype)
    qkv, proj = _dense(block.attn.qkv, dtype), _dense(block.attn.proj, dtype)
    mlp = None if block.mlp is None else (_norm(block.norm2, dtype), _mlp(block.mlp, dtype))

    def attention(x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        d = c // heads
        t = qkv(x.reshape(b, h * w, c)).reshape(b, h * w, 3, heads, d).transpose(2, 0, 3, 1, 4)
        q, k, v = t[0], t[1], t[2]  # [B, heads, N, d]
        # [B, heads, d, d], in float32: a sum over every pixel (3136 at 224 x 224) can leave float16's range
        kv = (k.astype(mx.float32) * d**-0.5).swapaxes(-1, -2) @ v.astype(mx.float32)
        attn = mx.softmax(kv, axis=-1).astype(q.dtype)
        y = (q @ attn.swapaxes(-1, -2)).transpose(0, 2, 1, 3).reshape(b, h, w, c)
        return proj(y)

    def apply(x: mx.array) -> mx.array:
        x = cpe1(x)
        x = x + attention(norm1(x))
        x = cpe2(x)
        return x if mlp is None else x + mlp[1](mlp[0](x))

    return apply


def _davit_down(layer: Any, dtype: mx.Dtype) -> Callable:
    """`Downsample`: LayerNorm2d, then the strided convolution (an even kernel pads the map to its multiple)."""
    norm, conv = _norm(layer.norm, dtype), _conv(layer.conv, None, dtype)
    k = layer.conv.kernel_size[0] if layer.even_k else 1
    return lambda x: conv(_pad_to(norm(x), k))


def davit(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a timm `DaVit`: NHWC images -> last stage's map, pooled `[B, C]` -> logits."""
    if not isinstance(net.norm_pre, nn.Identity) or not isinstance(net.head.pre_logits, nn.Identity):
        msg = "DaViT with the norm before the pool or an MLP head has no MLX port"
        raise NotImplementedError(msg)
    stem_conv, stem_norm = _conv(net.stem.conv, None, dtype), _norm(net.stem.norm, dtype)
    ops: list[Callable] = [lambda x: stem_norm(stem_conv(_pad_to(x, 4)))]
    for stage in net.stages:
        if not isinstance(stage.downsample, nn.Identity):
            ops.append(_davit_down(stage.downsample, dtype))
        for pair in stage.blocks:
            for block in pair:
                spatial = type(block).__name__ == "SpatialBlock"
                ops.append(_davit_spatial(block, dtype) if spatial else _davit_channel(block, dtype))
    return _chain(*ops), _chain(_norm(net.head.norm, dtype), _dense(net.head.fc, dtype))
