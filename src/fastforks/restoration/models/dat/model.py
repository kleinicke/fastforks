"""DAT runners: the network of zhengchen1999/DAT and `basicsr/models/dat_model.py`'s test call (`net_g(img)`, no tile).

The network pads its window attention's q / k / v with zeros to a multiple of the long window side itself, so there is
no input padding. Upstream evaluates every spatial attention's dynamic position bias MLP (`DynamicPosBias`) on every
call, gathers the bias from it, and builds the shifted-window masks on the CPU on every call for sizes other than the
training size; here the bias is made once at load (the MLP's input is a constant), the masks on the device once per
size, and the window attention is one `scaled_dot_product_attention` call (`fastforks.shared.swin_sr.attention`: the
head size padded to a multiple of 8 on CUDA). Each eval BatchNorm (the attentions' interaction branches) is folded into
the conv before it. The convolutions inside the blocks read channels-last views of the tokens and the tokens stay
contiguous (upstream: a transposed copy before and after every conv, 45 % of the time at 512 x 512). The input mean
is a buffer (upstream: a plain attribute, `type_as` per call).

float16: the last residual group's output exceeds the float16 range (299 266 on the golden image, `dat-x4`; its blocks
stay below 3 400), so its final conv and its skip input are divided by `_LAST_SCALE`; only the LayerNorm `norm` reads
it, its `eps` divided by the square (exact).
"""

from __future__ import annotations

import functools
import math
import re
import types
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.half import fp16_accumulate, fp32_accumulate
from fastforks._core.patch import is_base, override
from fastforks.kernels import cudnn_hacc, dat_interaction
from fastforks.restoration._runner import Crop, Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.sr_layer_norm import fast_layer_norm
from fastforks.shared.swin_sr import attention, checkpoint_upsampler, roll, stripe_pixelshuffle

from .net import DAT
from .net.dat_arch import (
    SGFN,
    Adaptive_Channel_Attention,
    Adaptive_Spatial_Attention,
    ResidualGroup,
    Spatial_Attention,
    SpatialGate,
    _Tokens,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` (the `network_g` section of the test options)."""

    scale: int  # 4
    embed_dim: int  # 180 (DAT, DAT-2) or 60 (DAT-light)
    depths: tuple[int, ...]  # blocks per residual group
    heads: tuple[int, ...]  # attention heads per residual group
    split: tuple[int, int]  # rectangle window (8, 32)
    expansion: float  # feed-forward hidden / embed_dim
    resi: str  # residual group conv: "1conv" or "3conv"
    upsampler: str  # "pixelshuffle" or "pixelshuffledirect" (DAT-light)
    img_size: int  # training patch size (the size of the masks upstream registers)
    key: str  # checkpoint key ("params")


@functools.lru_cache(maxsize=32)
def _masks(
    split: tuple[int, int], shift: tuple[int, int], h: int, w: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """`Adaptive_Spatial_Attention.calculate_mask` (both branches' 0 / -100 masks) on the device, once per size."""
    m = types.SimpleNamespace(split_size=list(split), shift_size=list(shift))
    return tuple(t.to(device) for t in Adaptive_Spatial_Attention.calculate_mask(m, h, w))  # ty: ignore[invalid-argument-type]


def _mask(m: Adaptive_Spatial_Attention, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.compiler.is_compiling():  # dynamo cannot trace through lru_cache; the graph holds the masks anyway
        masks = Adaptive_Spatial_Attention.calculate_mask(m, h, w)  # the class function: the instance's is this one
        return tuple(t.to(m.qkv.weight.device) for t in masks)
    return _masks(tuple(m.split_size), tuple(m.shift_size), h, w, m.qkv.weight.device)


def _windows(m: Spatial_Attention, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`Spatial_Attention.im2win` straight from the tokens, one copy (upstream: through a transposed image, three).

    On CUDA the head size is zero-padded to a multiple of 8 in the same copy (`attention` would copy q, k and v again
    to pad them); the attention output's extra columns are sliced off by the callers.
    """
    hs, ws, heads = m.H_sp, m.W_sp, m.num_heads
    x = x.unflatten(1, (h // hs, hs, w // ws, ws)).unflatten(-1, (heads, -1))  # b, h/hs, hs, w/ws, ws, heads, d
    win = x.permute(0, 1, 3, 5, 2, 4, 6)
    d = x.shape[-1]
    pad = -d % 8 if x.is_cuda else 0
    if not pad:
        return win.reshape(-1, heads, hs * ws, d)
    out = x.new_empty((*win.shape[:-1], d + pad))
    out[..., d:].zero_()
    out[..., :d] = win
    return out.view(-1, heads, hs * ws, d + pad)


def _to_image(win: torch.Tensor, hs: int, ws: int, into: torch.Tensor) -> torch.Tensor:
    """`windows2img` of an attention output `[b * rows * cols, heads, hs * ws, >= d]` in one copy.

    `into` is the image `[b, rows * hs, cols * ws, heads * d]`, any view (a region, a channel half); upstream copies
    twice (heads to channels, windows to rows).
    """
    b, h, w, c = into.shape
    heads = win.shape[1]
    d = c // heads
    src = win[..., :d].view(b, h // hs, w // ws, heads, hs, ws, d).permute(0, 1, 4, 2, 5, 3, 6)
    into.view(b, h // hs, hs, w // ws, ws, heads, d).copy_(src)
    return into


def _spatial(  # noqa: PLR0913 - upstream's arguments and the output
    m: Spatial_Attention,
    qkv: torch.Tensor,
    h: int,
    w: int,
    mask: torch.Tensor | None = None,
    *,
    into: torch.Tensor | None = None,
) -> torch.Tensor:
    """`Spatial_Attention.forward` with the position bias made at load and one SDPA call, written into `into`."""
    q, k, v = (_windows(m, t, h, w) for t in qkv.unbind(0))
    bw = q.shape[0]
    add = m.get_buffer("bias").to(q.dtype)[None]
    if mask is not None:
        nw = mask.shape[0]
        add = add + mask.to(q.dtype)[:, None]
        if bw != nw:
            add = add.repeat(bw // nw, 1, 1, 1)
    out = attention(q, k, v, add.expand(bw, -1, -1, -1), m.scale)
    if into is None:
        into = qkv.new_empty((qkv.shape[1], h, w, qkv.shape[-1]))
    return _to_image(out, m.H_sp, m.W_sp, into)


def _attend(attn: nn.Module, qkv: torch.Tensor, h: int, w: int, into: torch.Tensor) -> None:
    """`attn(qkv, h, w)` written into `into` (`_spatial` itself writes there; an upstream module is copied)."""
    if dict(getattr(type(attn), "_fastforks_methods", ())).get("forward") is _spatial:
        attn(qkv, h, w, into=into)
    else:
        into.copy_(attn(qkv, h, w))


def _pieces(n: int, s: int, sh: int, lo: int, hi: int) -> list[tuple[int, int, int, int, int]]:
    """Windows `lo .. hi` of a grid of `n` windows of size `s` along a dim rolled by `-sh` (0 <= sh < s), as blocks.

    Window `r` offset `i` holds image index `(r * s + i + sh) % (n * s)`. A block `(r0, nr, i0, ni, y0)`: windows
    `r0 .. r0 + nr`, offsets `i0 .. i0 + ni`, image `y0 .. y0 + nr * ni` (`ni == s` when `nr > 1`).
    """
    if not sh:
        return [(lo, hi - lo, 0, s, lo * s)]
    out = [(lo, min(hi, n - 1) - lo, 0, s, lo * s + sh)] if lo < n - 1 else []
    if hi == n:
        out += [(n - 1, 1, 0, s - sh, (n - 1) * s + sh), (n - 1, 1, s - sh, sh, 0)]
    return out


def _rolled_blocks(  # noqa: PLR0913 - the window grid and its shift
    win: torch.Tensor,
    img: torch.Tensor,
    *,
    grid: tuple[int, int],
    rows: tuple[int, int],
    cols: tuple[int, int],
    shift: tuple[int, int],
    to_image: bool,
) -> None:
    """Copy between windows and the image whose `roll(-shift)` they partition (`to_image`: windows into the image).

    `win` `[b, nr, nc, heads, hs, ws, d]`: the windows of grid rows `rows` and cols `cols`; `img` `[b, h, w, heads * d]`
    (any view).
    """
    hs, ws, heads, d = win.shape[4], win.shape[5], win.shape[3], win.shape[6]
    for r0, nr, i0, ni, y0 in _pieces(grid[0], hs, shift[0], *rows):
        for c0, nc, j0, nj, x0 in _pieces(grid[1], ws, shift[1], *cols):
            a = img[:, y0 : y0 + nr * ni, x0 : x0 + nc * nj].unflatten(1, (nr, ni)).unflatten(3, (nc, nj))
            a = a.unflatten(-1, (heads, d)).permute(0, 1, 3, 5, 2, 4, 6)
            r, c = r0 - rows[0], c0 - cols[0]
            b = win[:, r : r + nr, c : c + nc, :, i0 : i0 + ni, j0 : j0 + nj]
            (a if to_image else b).copy_(b if to_image else a)


def _shifted_window(  # noqa: PLR0913 - upstream's arguments and the output
    m: Spatial_Attention, qkv: torch.Tensor, h: int, w: int, shift: tuple[int, int], *, into: torch.Tensor
) -> torch.Tensor:
    """The attention of `roll(qkv, -shift)` rolled back by `shift` into `into`, without either roll's copy.

    `qkv` `[3, b, h, w, c]` (any view), `into` `[b, h, w, c]` (any view). The windows are copied straight from the
    unrolled image and the outputs straight into `into` (`_rolled_blocks`), and the attention mask is shared within the
    interior, final row, final column and corner of the shifted grid.
    """
    hs, ws = m.H_sp, m.W_sp
    grid = (h // hs, w // ws)
    heads, b = m.num_heads, qkv.shape[1]
    d = qkv.shape[-1] // heads
    pad = -d % 8 if qkv.is_cuda else 0
    bias = m.get_buffer("bias").to(qkv.dtype)
    token = torch.arange(hs * ws, device=qkv.device)
    row_zone = (token // ws >= hs - shift[0]).int() * 2
    col_zone = (token % ws >= ws - shift[1]).int()
    for r in (False, True):
        rows = (grid[0] - 1, grid[0]) if r else (0, grid[0] - 1)
        for c in (False, True):
            cols = (grid[1] - 1, grid[1]) if c else (0, grid[1] - 1)
            nr, nc = rows[1] - rows[0], cols[1] - cols[0]
            if not nr or not nc:
                continue
            pos = bias
            if r or c:
                # Tokens straddling either wrapped edge receive the upstream additive bias.
                zones = row_zone + col_zone if r and c else row_zone if r else col_zone
                pos = bias + (zones[:, None] != zones[None, :]).to(qkv.dtype) * -100.0
            win = qkv.new_empty((3, b, nr, nc, heads, hs, ws, d + pad))  # this section's q, k, v windows
            if pad:
                win[..., d:].zero_()
            for t, wt in zip(qkv.unbind(0), win.unbind(0), strict=True):
                _rolled_blocks(wt[..., :d], t, grid=grid, rows=rows, cols=cols, shift=shift, to_image=False)
            q, k, v = win.view(3, b * nr * nc, heads, hs * ws, d + pad).unbind(0)
            y = attention(q, k, v, pos.expand(q.shape[0], -1, -1, -1), m.scale)
            del win, q, k, v
            y7 = y[..., :d].unflatten(2, (hs, ws)).unflatten(0, (b, nr, nc))
            _rolled_blocks(y7, into, grid=grid, rows=rows, cols=cols, shift=shift, to_image=True)
            del y, y7
    return into


def _image(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Tokens `[b, h * w, c]` as a channels-last `[b, c, h, w]` view (upstream: a transposed copy)."""
    return x.unflatten(1, (h, w)).permute(0, 3, 1, 2)


def _tokens(x: torch.Tensor) -> torch.Tensor:
    """`[b, c, h, w]` -> contiguous tokens `[b, h * w, c]` (free for a channels-last tensor)."""
    return x.permute(0, 2, 3, 1).flatten(1, 2).contiguous()


def _tokens_of(m: _Tokens, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG001 - the overridden module
    return _tokens(x)


def _gate(m: SpatialGate, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`SpatialGate.forward` with the depthwise conv on a channels-last view."""
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * _tokens(m.conv(_image(m.norm(x2), h, w)))


def _ffn(m: SGFN, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`SGFN.forward` with `fc1` as two GEMMs, one per gate half: the gate's LayerNorm reads a contiguous half.

    Upstream's `x.chunk(2, -1)` of the fc1 output made the norm copy its strided half first (DAT x4 512 x 512: 36 copies
    of 189 MB).
    """
    half, fc1, sg = m.fc2.in_features, m.fc1, m.sg
    x1 = m.act(F.linear(x, fc1.weight[:half], None if fc1.bias is None else fc1.bias[:half]))
    x2 = m.act(F.linear(x, fc1.weight[half:], None if fc1.bias is None else fc1.bias[half:]))
    return m.fc2(x1 * _tokens(sg.conv(_image(sg.norm(x2), h, w))))


def _cuda_ffn(m: SGFN, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Finish the spatial gate before allocating the independent value branch."""
    act = m.act
    if (
        type(act) is not nn.GELU
        or m.training
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or torch.jit.is_tracing()
    ):
        return _ffn(m, x, h, w)
    half, fc1, sg = m.fc2.in_features, m.fc1, m.sg
    bias = fc1.bias
    gate = F.linear(x, fc1.weight[half:], None if bias is None else bias[half:])
    gate = sg.norm(torch.ops.aten.gelu_(gate, approximate=act.approximate))
    gate = _tokens(sg.conv(_image(gate, h, w)))
    value = F.linear(x, fc1.weight[:half], None if bias is None else bias[:half])
    torch.ops.aten.gelu_(value, approximate=act.approximate)
    value.mul_(gate)
    del gate
    return m.fc2(value)


def _interact(
    channel: torch.Tensor, spatial: torch.Tensor, channel_map: torch.Tensor, spatial_map: torch.Tensor
) -> torch.Tensor:
    return channel * channel_map.sigmoid() + spatial * spatial_map.sigmoid()


_COMPACT_MASK_PIXELS = 256**2


def _spatial_block(  # noqa: PLR0913 - separate CUDA gate and attention mask specializations
    m: Adaptive_Spatial_Attention,
    x: torch.Tensor,
    h: int,
    w: int,
    *,
    mix: Callable = _interact,
    compact_mask: bool = False,
) -> torch.Tensor:
    """`Adaptive_Spatial_Attention.forward` with its convolutions on channels-last views (no transposed copies)."""
    b, n, c = x.shape
    qkv = m.qkv(x).reshape(b, -1, 3, c).permute(2, 0, 1, 3)  # 3, b, hw, c
    s = max(m.split_size)
    pad_r, pad_b = -w % s, -h % s
    # Use a view of the padded values on CUDA so the original QKV storage can be reclaimed.
    v = None if compact_mask and (pad_r or pad_b) else _image(qkv[2], h, w)
    if pad_r or pad_b:  # no padded copy of q, k, v when the size divides (peak memory)
        qkv = F.pad(qkv.reshape(3 * b, h, w, c), (0, 0, 0, pad_r, 0, pad_b)).reshape(3, b, -1, c)
    hp, wp = h + pad_b, w + pad_r
    if v is None:
        v = qkv[2].view(b, hp, wp, c)[:, :h, :w].permute(0, 3, 1, 2)
    half = c // 2
    heads = qkv.new_empty((b, hp, wp, c))  # both branches' outputs, written in place (upstream: a cat)
    if (m.rg_idx % 2 == 0 and m.b_idx > 0 and (m.b_idx - 2) % 4 == 0) or (m.rg_idx % 2 != 0 and m.b_idx % 4 == 0):
        s0, s1 = m.shift_size
        use_compact = compact_mask or (
            x.is_cuda
            and not (torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.is_grad_enabled())
            and hp * wp >= _COMPACT_MASK_PIXELS
        )
        masks = None if use_compact else m.calculate_mask(hp, wp)
        for i, (channels, sh, sw) in enumerate(((slice(None, half), s0, s1), (slice(half, None), s1, s0))):
            attn = m.attns[i]
            part = qkv[..., channels].view(3, b, hp, wp, -1)
            if masks is None:
                _shifted_window(cast("Spatial_Attention", attn), part, hp, wp, (sh, sw), into=heads[..., channels])
                continue
            rolled = roll(part, (-sh, -sw), (2, 3)).view(3, b, -1, half)
            out = attn(rolled, hp, wp, mask=masks[i])
            del part, rolled
            roll(out, (sh, sw), (1, 2), out=heads[..., channels])
            del out
    else:
        _attend(m.attns[0], qkv[..., :half], hp, wp, heads[..., :half])
        _attend(m.attns[1], qkv[..., half:], hp, wp, heads[..., half:])
    att = heads[:, :h, :w].reshape(b, n, c)  # a copy only for a padded size
    del heads, qkv
    conv = m.dwconv(v)
    del v
    channel_map = m.channel_interaction(conv).view(b, 1, c)
    spatial_map = m.spatial_interaction(_image(att, h, w))
    return m.proj(mix(att, conv.flatten(2).transpose(1, 2), channel_map, spatial_map.flatten(2).transpose(1, 2)))


_SPLIT_K = 4096  # fixed: measured (RTX 5060 Ti), 2048-16384 tokens per partial product within 10 %


def _split_k_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Per-head `q @ k^T` of `[B, heads, d, N]` over all `N` tokens as split-K float32-output GEMMs.

    cuBLAS runs the `d x d` products with K = N (262144 at 512 x 512) on one slow kernel (1.6 ms for d = 10, 6 heads);
    here the all-heads `[heads d, N]` rows (a unit-stride view of the token layout) multiply in `N / 4096` chunks with
    float32 outputs, are summed in float32 and the per-head diagonal blocks are kept (0.2 ms). The scores stay
    float32-accumulated, as `fp32_accumulate` gave, and round to the input dtype once.
    """
    b, heads, d, n = q.shape
    c = heads * d
    qm, km = q.reshape(b, c, n), k.reshape(b, c, n)
    full = n // _SPLIT_K * _SPLIT_K
    out = []
    for i in range(b):
        qa, ka = qm[i], km[i]
        s = None
        with fp32_accumulate():  # float32 GEMM outputs need it under `fp16_accumulate`
            if full:
                qs = qa[:, :full].unflatten(1, (-1, _SPLIT_K)).transpose(0, 1)  # [chunks, heads d, 4096]
                ks = ka[:, :full].unflatten(1, (-1, _SPLIT_K)).permute(1, 2, 0)  # [chunks, 4096, heads d]
                s = torch.bmm(qs, ks, out_dtype=torch.float32).sum(0)
            if n > full:
                r = torch.mm(qa[:, full:], ka[:, full:].T, out_dtype=torch.float32)
                s = r if s is None else s + r
        out.append(cast("torch.Tensor", s).view(heads, d, heads, d).diagonal(dim1=0, dim2=2).permute(2, 0, 1))
    return torch.stack(out).to(q.dtype)


def _channel_block(  # noqa: PLR0913 - a CUDA gate and the fp32 scores, like `_spatial_block`
    m: Adaptive_Channel_Attention,
    x: torch.Tensor,
    h: int,
    w: int,
    *,
    mix: Callable = _interact,
    fp32_scores: bool = False,
) -> torch.Tensor:
    """`Adaptive_Channel_Attention.forward` with its convolutions on channels-last views (no transposed copies).

    `fp32_scores`: q k^T (a sum over all h * w tokens) accumulates in fp32 also under `fp16_accumulate`
    (`_split_k_scores`). An fp16 accumulator cannot add the many small products of the normalized q and k (a 512 x 512
    tile: 262144 terms, sum at most 1): DAT x4 kernels then differed from portable by up to 112 / 255 on a photo.
    """
    b, n, c = x.shape
    qkv = m.qkv(x).reshape(b, n, 3, m.num_heads, c // m.num_heads).permute(2, 0, 3, 4, 1)  # 3, b, heads, d, n
    q, k, v = F.normalize(qkv[0], dim=-1), F.normalize(qkv[1], dim=-1), qkv[2]
    scores = _split_k_scores(q, k) if fp32_scores else q @ k.transpose(-2, -1)
    attn = (scores * m.temperature).softmax(dim=-1)
    del q, k
    att = (attn @ v).permute(0, 3, 1, 2).reshape(b, n, c)
    conv = m.dwconv(_image(qkv[2].permute(0, 3, 1, 2).reshape(b, n, c), h, w))
    del v, qkv, attn
    channel_map = m.channel_interaction(_image(att, h, w))
    spatial_map = _tokens(m.spatial_interaction(conv))
    return m.proj(mix(conv.flatten(2).transpose(1, 2), att, channel_map.flatten(1)[:, None], spatial_map))


def _group(m: ResidualGroup, x: torch.Tensor, x_size: tuple[int, int], skip: float = 1.0) -> torch.Tensor:
    """`ResidualGroup.forward` (the skip scaled by `skip`), its conv on a channels-last view."""
    res = x
    for blk in m.blocks:
        x = blk(x, x_size)
    return (res / skip if skip != 1.0 else res) + _tokens(m.conv(_image(x, *x_size)))


def _prepare(net: DAT) -> None:
    """Make the position biases once (no MLPs), fold the BatchNorms, masks on the device, channels-last convs."""
    for m in net.modules():
        if isinstance(m, Spatial_Attention):
            n = m.H_sp * m.W_sp
            with torch.no_grad():
                bias = m.pos(m.rpe_biases)[m.relative_position_index.view(-1)].view(n, n, -1)  # ty: ignore[call-non-callable]
            del m.pos, m.rpe_biases, m.relative_position_index
            m.register_buffer("bias", bias.permute(2, 0, 1).contiguous())
            override(m, _spatial)
        elif isinstance(m, Adaptive_Spatial_Attention | Adaptive_Channel_Attention):
            for seq in (m.dwconv, m.channel_interaction, m.spatial_interaction):
                i = next(i for i, b in enumerate(seq) if isinstance(b, nn.BatchNorm2d))
                seq[i - 1], seq[i] = fuse_conv_bn_eval(seq[i - 1], seq[i]), nn.Identity()  # ty: ignore[invalid-argument-type]
            override(m, _spatial_block if isinstance(m, Adaptive_Spatial_Attention) else _channel_block)
            if isinstance(m, Adaptive_Spatial_Attention):
                override(m, _mask, "calculate_mask")
        elif isinstance(m, SpatialGate):
            override(m, _gate)
        elif isinstance(m, SGFN):
            override(m, _ffn)
        elif isinstance(m, ResidualGroup):
            override(m, _group)
        elif isinstance(m, _Tokens):
            override(m, _tokens_of)


_LAST_SCALE = 64.0  # fixed: float16 headroom of the last group's residual sum, not an upstream parameter


def _group_scaled(m: ResidualGroup, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
    """`ResidualGroup.forward` / `_LAST_SCALE` (the final conv's weights are divided at load)."""
    return _group(m, x, x_size, _LAST_SCALE)


def _scale_last_group(net: DAT) -> None:
    """Divide the last residual group's output by `_LAST_SCALE` (float16 range), `norm`'s eps by its square."""
    last = net.layers[-1]
    conv = last.get_submodule("conv" if isinstance(last.conv, nn.Conv2d) else f"conv.{len(last.conv) - 1}")  # ty: ignore[invalid-argument-type]
    with torch.no_grad():  # "1conv" or "3conv": the last convolution
        for t in conv.parameters():  # weight, bias
            t.div_(_LAST_SCALE)
    override(last, _group_scaled)
    net.norm.eps /= _LAST_SCALE**2  # LayerNorm(x / k, eps / k²) = LayerNorm(x, eps)


class Portable(Runner):
    """Plain PyTorch in the card's dtype."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network, make the position biases, fold the BatchNorms."""
        super().__init__(ctx)
        p = ctx.params
        with warnings.catch_warnings():  # upstream's torch.meshgrid without `indexing` ("ij", the default) warns
            warnings.filterwarnings("ignore", "torch.meshgrid", UserWarning)
            self.net = DAT(
                upscale=p.scale, in_chans=3, img_size=p.img_size, img_range=1.0, split_size=list(p.split),
                depth=list(p.depths), embed_dim=p.embed_dim, num_heads=list(p.heads), expansion_factor=p.expansion,
                resi_connection=p.resi, upsampler=p.upsampler,
            )  # fmt: skip
        load_strict(self.net, ctx.state_dict(key=p.key), f"{ctx.variant.ref} checkpoint")
        self.net.eval()
        _prepare(self.net)
        mean = self.net.mean
        del self.net.mean
        self.net.register_buffer("mean", mean, persistent=False)
        if mlx_wrap.run_dtype(ctx) == "float16":
            _scale_last_group(self.net)
        self.net.to(ctx.device, getattr(torch, ctx.dtype))
        if ctx.device.type == "cuda" and ctx.dtype == "float16":
            # channels-last weights: cuDNN's fp16 convolutions skip the NCHW<->NHWC conversion around each conv
            # (256x256 x4 dat-light peak 251 -> 195 MB, same speed; RTX 5060 Ti, Windows)
            self.net.to(memory_format=torch.channels_last)

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """No padding: the network pads its window attention itself (`dat_model.py` test: `net_g(img)`)."""
        return x, Crop(0, 0, *x.shape[-2:])

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        return {"image": self.net(image.to(getattr(torch, self.ctx.dtype)))}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype."""
        dtype = getattr(torch, self.ctx.dtype)
        return Forward(self.net, lambda net, image: net(image.to(dtype)))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


_HACC_MIN_CHANNELS = 64  # fixed: measured, narrower convolutions run slower through cudnn_hacc


def _cuda_spatial(m: Adaptive_Spatial_Attention, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return _spatial_block(m, x, h, w, mix=dat_interaction.interaction, compact_mask=True)


def _cuda_channel(m: Adaptive_Channel_Attention, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return _channel_block(m, x, h, w, mix=dat_interaction.interaction, fp32_scores=True)


_GN_MAX_C = 96  # fixed: measured (RTX 5060 Ti), group_norm beats F.layer_norm up to 96 channels, not at 144 / 180
_GN_MIN_ROWS = 65536  # fixed: measured, below this F.layer_norm is faster (4096 rows: 0.02 vs 0.06 ms)


def _narrow_layer_norm(m: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """`LayerNorm` over a narrow last dim as a one-group `group_norm` of `[rows, C, 1]` (the same statistics + affine).

    PyTorch's CUDA LayerNorm runs about 6x below the memory bound on short rows: DAT-light's 60 channels at 512 x 512
    (262 144 rows) take 1.02 ms against 0.62 ms here.
    """
    c = m.normalized_shape[0]
    if x.numel() // c < _GN_MIN_ROWS:
        return F.layer_norm(x, m.normalized_shape, m.weight, m.bias, m.eps)
    return F.group_norm(x.reshape(-1, c, 1), 1, m.weight, m.bias, m.eps).view(x.shape)


class Kernels(Portable):
    """float16, fp16-accumulate GEMMs and dense convolutions of at least 64 channels in and out (`cudnn_hacc`).

    The narrower ones (DAT-light's 60-channel convolutions, the 1 x 1 convolutions of the channel interactions) were
    slower through `cudnn_hacc` (DAT-light 512 x 512: 300 ms and 2.9 GB vs 268 ms and 1.5 GB; DAT unchanged).
    LayerNorms of at most `_GN_MAX_C` channels run as group norms (`_narrow_layer_norm`).
    """

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its dense convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        fast_layer_norm(self.net, ctx.device)
        for m in self.net.modules():
            if isinstance(m, Adaptive_Spatial_Attention):
                override(m, _cuda_spatial)
            elif isinstance(m, Adaptive_Channel_Attention):
                override(m, _cuda_channel)
            elif isinstance(m, SGFN):
                override(m, _cuda_ffn)
            elif (
                is_base(m, torch.nn.Conv2d)
                and m.groups == 1
                and min(m.in_channels, m.out_channels) >= _HACC_MIN_CHANNELS
            ):
                override(m, _hacc_conv)
            elif is_base(m, nn.LayerNorm) and len(m.normalized_shape) == 1 and m.normalized_shape[0] <= _GN_MAX_C:
                override(m, _narrow_layer_norm)
        if self.net.upsampler == "pixelshuffle":
            stripe_pixelshuffle(self.net, self.net.upscale)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        with fp16_accumulate():
            return {"image": self.net(image.to(torch.float16))}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable),
}

_BLOCK = re.compile(r"layers\.(\d+)\.blocks\.(\d+)\.")


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load(path)`: a DAT state dict as a variant and its params (channels, depths, heads, split, FFN, tail)."""
    a = "layers.0.blocks.0.attn.attns.0."
    blocks = [(int(m.group(1)), int(m.group(2))) for k in sd if (m := _BLOCK.match(k))]
    if "conv_first.weight" not in sd or "before_RG.1.weight" not in sd or a + "rpe_biases" not in sd:
        return None
    layers = 1 + max(i for i, _ in blocks)
    embed_dim = sd["conv_first.weight"].shape[0]
    rpe = sd[a + "rpe_biases"]  # the window's relative offsets: -(split - 1) .. split - 1 per axis
    split = (int(rpe[:, 0].max()) + 1, int(rpe[:, 1].max()) + 1)
    mask = next((v for k, v in sd.items() if k.endswith(".attn.attn_mask_0")), None)  # [windows, N, N] at img_size
    up = checkpoint_upsampler(sd)
    if sd["conv_first.weight"].shape[1] != 3 or up is None or up[0] == "nearest+conv":  # noqa: PLR2004 - RGB
        msg = "DAT checkpoint with not 3 input channels or no pixelshuffle upsampler: not supported"
        raise WeightsError(msg)
    hidden = sd["layers.0.blocks.0.ffn.fc1.weight"].shape[0]
    params = {
        "scale": up[1],
        "embed_dim": embed_dim,
        "depths": tuple(1 + max(j for i, j in blocks if i == n) for n in range(layers)),
        "heads": tuple(2 * sd[f"layers.{n}.blocks.0.attn.attns.0.pos.pos3.2.weight"].shape[0] for n in range(layers)),
        "split": split,
        "expansion": hidden // embed_dim if hidden % embed_dim == 0 else hidden / embed_dim,
        "resi": "3conv" if "layers.0.conv.0.weight" in sd else "1conv",
        "upsampler": up[0],
        "img_size": 64 if mask is None else math.isqrt(mask.shape[0] * split[0] * split[1]),
    }
    if up[0] == "pixelshuffledirect":
        return "dat-light-x4", params, dict(sd)
    return ("dat-2-x4" if hidden == 2 * embed_dim else "dat-x4"), params, dict(sd)
