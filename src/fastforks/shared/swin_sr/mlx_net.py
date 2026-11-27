"""Swin super-resolution networks on Apple MLX: the layers the hand-written ports share (SwinIR, Swin2SR, HAT, DRCT).

Written in `mlx.core` operations on NHWC maps; a map `[B, H, W, C]` is also the token sequence the vendored networks
keep as `[B, H * W, C]` (`PatchEmbed` / `PatchUnEmbed` are layout changes only and vanish). Every weight is read
through `arr` / `derived` (`fastforks.shared.mlx_port.tensor`), so a port built by `mlx_port.recorded` loads its arrays
from the conversion cache, in the run dtype.

- `window_attention`: one (shifted) window attention of a Swin block, `proj(softmax(q k^T * scale + bias (+ mask)) v)`,
  the relative position bias gathered once at load (`Attention`), Swin V2's cosine attention (q and k L2-normalised
  in float32, q times the clamped logit scale) as an option, through `fastforks._core.mlx.attention` (MLX's scaled
  dot-product attention, bias and mask summed per chunk) as the graph translation runs it. The head size (30) stays
  as it is there: MLX fuses no attention under 64, and the 64 x 64 scores of a window are cheaper than padded copies;
- `ln`: the channel layer norm (`graph_mlx.ops.row_layer_norm`, as the translation);
- `shift_mask`: the 0 / -100 shifted-window mask, made once per map size; `border_mask`: the same as its border
  windows (the only ones spanning several regions), which `attend` runs again with their masks;
- `Block` / `Group`: a Swin block in bands of window rows on token gathers (consecutive blocks of a group hand their
  tokens on in window order), an RSTB's blocks, convs and skip;
- `windows` / `unwindows`: the cyclic shift and the window partition (and back; OCAB and DAT);
- `conv`, `seq`: 3 x 3 convs through the graph translator's `mlx_graph._conv2d` (its channel padding onto MLX's fast
  tiles, as the graph translation ran them), 1 x 1 ones as matrix products, `nn.Sequential`s of convs, LeakyReLUs and
  PixelShuffles;
- `reconstruct`: the upsamplers (`pixelshuffle`, `nearest+conv`) in row bands with their halo where the output is
  large (`TAIL_BAND_BYTES`), each band its own step;
- `stepped` / `steps`: compiled pieces evaluated one by one, one in flight (MLX frees a command buffer's transients
  only when it completes, and one lazy graph of the whole network holds far more than its live set).
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core import mlx as mlx_core
from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_graph as mg
from fastforks.shared import mlx_port

if TYPE_CHECKING:
    from collections.abc import Callable

    Layer = Callable[[mx.array], mx.array]

__all__ = [
    "TAIL_BAND_BYTES",
    "Attention",
    "Block",
    "Border",
    "Group",
    "Network",
    "OverlapBlock",
    "arr",
    "attend",
    "bands",
    "border_mask",
    "conv",
    "dense",
    "derived",
    "gelu",
    "layer_norm",
    "linear",
    "ln",
    "lrelu",
    "pixel_shuffle",
    "reconstruct",
    "seq",
    "shift_mask",
    "stepped",
    "steps",
    "unwindows",
    "window_attention",
    "windows",
]

ONE = (1, 1)


def _cast(a: np.ndarray, dtype: mx.Dtype) -> mx.array:
    """A float64 numpy array as an evaluated MLX array in `dtype` (no double rounding to float16)."""
    out = mx.array(np.ascontiguousarray(a, dtype=np.float32)).astype(dtype)
    mx.eval(out)
    return out


def arr(t: Any, dtype: mx.Dtype, *, scale: float = 1.0) -> mx.array:
    """A torch tensor (times `scale`, in float64) as an MLX array in `dtype`, read through `mlx_port.tensor`.

    `t` may be computed from the modules' tensors (`layer.weight / 64`, a gathered bias): on a cache hit it is a meta
    tensor of the same shape, and only its shape is read (`scale` is applied in the conversion: torch arithmetic on a
    meta tensor imports `torch._dynamo`, about a second of load).
    """
    return mlx_port.tensor(t, dtype, lambda: _cast(t.detach().double().numpy() * scale, dtype))


def derived(shape: tuple[int, ...], dtype: mx.Dtype, make: Callable[[], np.ndarray]) -> mx.array:
    """An array of `shape` that `make()` computes from weight values (numpy), cached as `arr`'s (not run on a hit)."""
    return mlx_port.tensor(SimpleNamespace(shape=shape), dtype, lambda: _cast(make(), dtype))


def linear(layer: Any, dtype: mx.Dtype) -> tuple[mx.array, mx.array | None]:
    """An `nn.Linear` as (`[in, out]` weight, bias or None)."""
    return arr(layer.weight.t(), dtype), None if layer.bias is None else arr(layer.bias, dtype)


def dense(x: mx.array, w: mx.array, b: mx.array | None) -> mx.array:
    """`x @ w + b` over the last axis (the bias fused, `mx.addmm`), any leading shape."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    y = x2 @ w if b is None else mx.addmm(b, x2, w)
    return y.reshape(*lead, w.shape[1])


def layer_norm(layer: Any, dtype: mx.Dtype, *, eps_scale: float = 1.0) -> Layer:
    """An `nn.LayerNorm` over the channels (`eps_scale`: its epsilon times this, for a scaled input)."""
    w, b, eps = arr(layer.weight, dtype), arr(layer.bias, dtype), float(layer.eps) * eps_scale
    return lambda x: ln(x, w, b, eps)


def ln(x: mx.array, w: mx.array | None, b: mx.array | None, eps: float) -> mx.array:
    """Layer norm over the channels, as the graph translation runs it.

    `graph_mlx.ops.row_layer_norm` up to its width: `mx.fast.layer_norm` is several times slower at these widths
    (180: 8.1 against 1.3 ms per 47M elements).
    """
    if x.shape[-1] <= ops.ROW_LN_MAX:
        return ops.row_layer_norm(x, w, b, eps)
    return mx.fast.layer_norm(x, w, b, eps)


def gelu(x: mx.array) -> mx.array:
    """`nn.GELU()` (erf), computed in float32 (fused into its neighbours by `mx.compile`)."""
    y = x.astype(mx.float32)
    return (0.5 * y * (1 + mx.erf(y * 0.7071067811865476))).astype(x.dtype)


def lrelu(x: mx.array, slope: float) -> mx.array:
    """`LeakyReLU(slope)`."""
    return mx.maximum(x, x * slope)


def pixel_shuffle(x: mx.array, r: int) -> mx.array:
    """`PixelShuffle(r)` of an NHWC map (input channel `c * r * r + i * r + j`)."""
    n, h, w, c = x.shape
    x = x.reshape(n, h, w, c // (r * r), r, r).transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(n, h * r, w * r, c // (r * r))


def up2(x: mx.array) -> mx.array:
    """`F.interpolate(x, scale_factor=2, mode="nearest")` of an NHWC map."""
    n, h, w, c = x.shape
    return mx.broadcast_to(x[:, :, None, :, None], (n, h, 2, w, 2, c)).reshape(n, 2 * h, 2 * w, c)


def conv(layer: Any, dtype: mx.Dtype, *, scale: float = 1.0) -> Layer:
    """An `nn.Conv2d` (stride 1, with bias) on NHWC maps, its weights and bias times `scale` (a power of two)."""
    if tuple(layer.stride) != ONE or layer.groups != 1 or layer.bias is None:
        msg = f"Swin SR MLX: a stride-1, ungrouped conv with bias, got {layer}"
        raise NotImplementedError(msg)
    w, b = layer.weight, layer.bias
    bias = arr(b, dtype, scale=scale)
    if tuple(layer.kernel_size) == ONE:
        m = arr(w[:, :, 0, 0].t(), dtype, scale=scale)
        return lambda x: dense(x, m, bias)
    ohwi, padding = arr(w.permute(0, 2, 3, 1), dtype, scale=scale), (int(layer.padding[0]), int(layer.padding[1]))
    return lambda x: mg._conv2d(x, ohwi, ONE, padding, ONE, 1) + bias  # noqa: SLF001 - the translator's dispatch


def stages(layer: Any, dtype: mx.Dtype) -> list[list[Layer]]:
    """An `nn.Conv2d` or an `nn.Sequential` of convs, LeakyReLUs and PixelShuffles as stages.

    A stage per conv, the activations and shuffles after it in its stage.
    """
    from torch import nn  # noqa: PLC0415 - the module types of a built network (torch is imported by then)

    mods = list(layer) if isinstance(layer, nn.Sequential) else [layer]
    out: list[list[Layer]] = []
    for m in mods:
        if isinstance(m, nn.Conv2d):
            out.append([conv(m, dtype)])
            continue
        if isinstance(m, nn.LeakyReLU):
            op: Layer = functools.partial(lrelu, slope=float(m.negative_slope))
        elif isinstance(m, nn.PixelShuffle):
            op = functools.partial(pixel_shuffle, r=int(m.upscale_factor))
        else:
            msg = f"Swin SR MLX: unsupported layer {type(m).__name__}"
            raise NotImplementedError(msg)
        if not out:
            out.append([])
        out[-1].append(op)
    return out


def _chain(ops: list[Layer]) -> Layer:
    def apply(x: mx.array) -> mx.array:
        for op in ops:
            x = op(x)
        return x

    return apply


def _convs3(layer: Any) -> int:
    """The 3 x 3 (wider than 1 x 1) convs of an `nn.Conv2d` or an `nn.Sequential`."""
    from torch import nn  # noqa: PLC0415 - the module types of a built network (torch is imported by then)

    mods = list(layer) if isinstance(layer, nn.Sequential) else [layer]
    return sum(isinstance(m, nn.Conv2d) and m.kernel_size[0] > 1 for m in mods)


def seq(layer: Any, dtype: mx.Dtype) -> tuple[Layer, int]:
    """An `nn.Conv2d` or an `nn.Sequential` of convs, LeakyReLUs and PixelShuffles; and its 3 x 3 conv count."""
    return _chain([op for stage in stages(layer, dtype) for op in stage]), _convs3(layer)


# --- windows ---------------------------------------------------------------------------------------------------------


Size = int | tuple[int, int]


def _pair(v: Size) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else (int(v[0]), int(v[1]))


def windows(x: mx.array, window: Size, shift: Size) -> mx.array:
    """`window_partition(roll(x, -shift))`: `[B, H, W, C]` -> `[B * nW, wh * ww, C]` (square or (wh, ww) windows)."""
    (wh, ww), (sh, sw) = _pair(window), _pair(shift)
    if sh or sw:
        x = mx.roll(x, (-sh, -sw), axis=(1, 2))
    b, h, w, c = x.shape
    x = x.reshape(b, h // wh, wh, w // ww, ww, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, wh * ww, c)


def unwindows(y: mx.array, size: tuple[int, int, int], window: Size, shift: Size) -> mx.array:
    """`roll(window_reverse(y), shift)`: `[B * nW, wh * ww, C]` -> `[B, H, W, C]` (`size` = (B, H, W))."""
    (wh, ww), (sh, sw) = _pair(window), _pair(shift)
    b, h, w = size
    c = y.shape[-1]
    y = y.reshape(b, h // wh, w // ww, wh, ww, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)
    return mx.roll(y, (sh, sw), axis=(1, 2)) if sh or sw else y


@functools.lru_cache(maxsize=16)
def _shift_mask(h: int, w: int, window: tuple[int, int], shift: tuple[int, int]) -> np.ndarray:
    """`SwinTransformerBlock.calculate_mask` (DAT's for its rectangles) of an `h` x `w` map.

    `[nW, N, N]`: 0 within a shifted window's region, -100 across regions.
    """
    (wh, ww), (sh, sw) = window, shift
    img = np.zeros((h, w), np.float32)
    cnt = 0
    for hs in (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)):
        for ws in (slice(0, -ww), slice(-ww, -sw), slice(-sw, None)):
            img[hs, ws] = cnt
            cnt += 1
    m = img.reshape(h // wh, wh, w // ww, ww).transpose(0, 2, 1, 3).reshape(-1, wh * ww)
    return np.where(m[:, None, :] != m[:, :, None], -100.0, 0.0).astype(np.float32)


@functools.lru_cache(maxsize=8)
def _shift_mask_mx(h: int, w: int, window: tuple[int, int], shift: tuple[int, int], dtype: mx.Dtype) -> mx.array:
    out = mx.array(_shift_mask(h, w, window, shift)[:, None]).astype(dtype)
    mx.eval(out)
    return out


def shift_mask(h: int, w: int, window: Size, shift: Size, dtype: mx.Dtype) -> mx.array:
    """`[nW, 1, N, N]` shifted-window mask in `dtype` (float16 holds 0 / -100 exactly), made once per map size."""
    return _shift_mask_mx(h, w, _pair(window), _pair(shift), dtype)


Border = tuple[mx.array, mx.array]
"""A shifted-window mask as its border windows: (`[nb]` int32 window indices over the batch, `[nb, 1, N, N]` their
masks); every other window spans one region (its mask all zeros)."""


@functools.lru_cache(maxsize=8)
def _border_mask(
    size: tuple[int, int, int], window: tuple[int, int], shift: tuple[int, int], dtype: mx.Dtype
) -> Border:
    batch, h, w = size
    m = _shift_mask(h, w, window, shift)
    border = np.flatnonzero(m.any(axis=(1, 2)))
    index = (np.arange(batch)[:, None] * m.shape[0] + border[None]).reshape(-1).astype(np.int32)
    out = mx.array(index), mx.array(np.tile(m[border][:, None], (batch, 1, 1, 1))).astype(dtype)
    mx.eval(*out)
    return out


def border_mask(size: tuple[int, int, int], window: Size, shift: Size, dtype: mx.Dtype) -> Border:
    """`shift_mask` of a batch of `size` = (B, H, W) maps as its border windows (`Border`), made once per size.

    The shifted windows span several regions only along the map's last window row and column (SwinIR at 520 x 520:
    129 of 4225): `attend` runs every window with the position bias alone and those again with their mask, instead of
    a bias + mask sum as large as the scores.
    """
    return _border_mask((int(size[0]), int(size[1]), int(size[2])), _pair(window), _pair(shift), dtype)


class Attention:
    """A `WindowAttention`'s arrays: `qkv` / `proj` weights, the relative position bias `[nH, N, N]`, the scale.

    `bias` is the gathered bias as a torch tensor (computed from the module's tensors, meta on a cache hit); `cosine`
    the clamped, exponentiated logit scale `[nH, 1, 1]` (Swin V2, float32, its `scale` 1).
    """

    def __init__(  # noqa: PLR0913 - the projections, the bias and the head geometry
        self,
        qkv: Any,
        proj: Any,
        bias: Any,
        dtype: mx.Dtype,
        *,
        heads: int,
        scale: float,
        qkv_bias: Any = None,
        cosine: Any = None,
    ) -> None:
        """The projections and the relative position bias of one attention in `dtype`."""
        self.qkv_w = arr(qkv.weight.t(), dtype)
        b = qkv.bias if qkv_bias is None else qkv_bias
        self.qkv_b = None if b is None else arr(b, dtype)
        self.proj_w, self.proj_b = linear(proj, dtype)
        self.bias = arr(bias, dtype)
        self.cosine = None if cosine is None else arr(cosine, mx.float32)
        self.heads, self.scale = heads, scale

    def params(self) -> dict[str, mx.array]:
        """The arrays as a compiled function's argument."""
        p = {"qkv_w": self.qkv_w, "proj_w": self.proj_w, "proj_b": self.proj_b, "bias": self.bias}
        if self.qkv_b is not None:
            p["qkv_b"] = self.qkv_b
        if self.cosine is not None:
            p["cosine"] = self.cosine
        return p


def _normalize(x: mx.array) -> mx.array:
    """`F.normalize(x, dim=-1)` in float32."""
    y = x.astype(mx.float32)
    return y * mx.rsqrt(mx.maximum(mx.sum(y * y, axis=-1, keepdims=True), 1e-24))


SCORE_CHUNK_BYTES = 256 << 20
"""Largest float32 score block of one attention chunk (`attend`): the windows are attended in chunks of whole window
rows below it (SwinIR at 520 x 520: one chunk; HAT's 256 x 576 overlapping windows: 8 rows of 32 per chunk)."""


def _chunks(windows: int, row: int, per_window: int, limit: int = SCORE_CHUNK_BYTES) -> list[tuple[int, int]]:
    """Window ranges of at most `limit` bytes of scores (`per_window` bytes each), whole window rows."""
    step = max(1, limit // (per_window * row)) * row
    return [(a, min(a + step, windows)) for a in range(0, windows, step)]


def attend(  # noqa: PLR0913 - the attention's operands and geometry
    q: mx.array,
    k: mx.array,
    v: mx.array,
    bias: mx.array,
    mask: Border | None,
    *,
    heads: int,
    row: int,
    cosine: mx.array | None = None,
) -> mx.array:
    """`softmax(q k^T + bias (+ mask)) v` of `[B * nW, N, C]` windows (q already scaled), heads split and merged.

    `mask`: a shifted-window mask (`border_mask`); `cosine`: Swin V2's logit scale (q and k normalised first); `row`:
    windows per window row (the chunks are whole rows: `SCORE_CHUNK_BYTES`).
    """
    b_, n, c = q.shape
    m = k.shape[1]
    d = c // heads
    q = q.reshape(b_, n, heads, d).transpose(0, 2, 1, 3)
    k = k.reshape(b_, m, heads, d).transpose(0, 2, 1, 3)
    v = v.reshape(b_, m, heads, d).transpose(0, 2, 1, 3)
    if cosine is not None:  # Swin V2: cos(q, k) * logit scale
        q = (_normalize(q) * cosine).astype(v.dtype)
        k = _normalize(k).astype(v.dtype)
    outs = [mlx_core.attention(q[a:e], k[a:e], v[a:e], scale=1.0, mask=bias[None]) for a, e in
            _chunks(b_, row, heads * n * m * 4)]  # fmt: skip
    o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
    if mask is not None:  # the windows across regions again, with their masks
        index, border = mask
        o[index] = mlx_core.attention(q[index], k[index], v[index], scale=1.0, mask=bias + border)
    return o.transpose(0, 2, 1, 3).reshape(b_, n, c)


def window_attention(  # noqa: PLR0913 - the windows, the arrays and the geometry
    t: mx.array, p: dict[str, mx.array], mask: Border | None, *, heads: int, scale: float, row: int
) -> mx.array:
    """`WindowAttention.forward` of `[B * nW, N, C]` windows (`p`: `Attention.params()`, `mask`: `border_mask`)."""
    c = t.shape[-1]
    qkv = dense(t, p["qkv_w"], p.get("qkv_b"))
    q, k, v = qkv[..., :c], qkv[..., c : 2 * c], qkv[..., 2 * c :]
    if "cosine" not in p:
        q = q * scale
    o = attend(q, k, v, p["bias"], mask, heads=heads, row=row, cosine=p.get("cosine"))
    return dense(o, p["proj_w"], p["proj_b"])


# --- steps -----------------------------------------------------------------------------------------------------------

_PENDING: list[Any] = []  # the step in flight (`stepped`)
PIPELINE = True
"""Schedule a step before awaiting the one in flight (depth 1); off: each step awaited, two steps' transients never
held at once (SwinIR-M x2 float16 at 520 x 520: depth 2 4 % faster at 1.34 times the peak)."""


def stepped(f: Callable[..., Any], *, depth: int = 1) -> Callable[..., Any]:
    """`f` compiled, each result scheduled as a step, the steps before it awaited (`PIPELINE`) down to `depth - 1`.

    `depth` 0: a step of whole maps, run alone (the steps before it awaited first, it awaited before the next).
    """
    g = mx.compile(f)

    def run(*args: Any) -> Any:
        y = g(*args)
        if not PIPELINE or not depth:
            mx.eval(*_PENDING)
            _PENDING.clear()
            mx.eval(y)
            return y
        mx.async_eval(y)
        while len(_PENDING) >= depth:
            mx.eval(_PENDING.pop(0))
        _PENDING.append(y)
        return y

    return run


def steps(f: Callable[..., Any], *args: Any) -> Any:
    """`f(*args)` with its `stepped` pieces, the pipeline emptied afterwards (also on an error)."""
    try:
        return f(*args)
    finally:
        _PENDING.clear()


# --- reconstruction --------------------------------------------------------------------------------------------------

TAIL_BAND_BYTES = 128 << 20
"""Largest map of one row band of the upsampler (`reconstruct`): a band's live set is a few of its maps (the x4
pixel-shuffle tail's 256-channel conv outputs and their shuffles), so SwinIR-M x4 at 520 x 520 (the 2080 x 2080 x 64
map: 554 MB in float16) runs in five bands, in nine in float32."""


def bands(  # noqa: PLR0913 - the tail, its input and its geometry
    run: Callable[[mx.array], mx.array], x: mx.array, *, scale: int, halo: int, row_bytes: int, limit: int
) -> mx.array:
    """`run(x)` in row bands of at most `limit` bytes of its largest map, each band its own call.

    `run`: a `scale`-times upsampling tail needing `halo` input rows of context; `row_bytes`: its largest map's bytes
    per input row.
    """
    h = x.shape[1]
    rows = max(1, limit // row_bytes)
    if rows >= h:
        return run(x)
    out = []
    for start in range(0, h, rows):
        end = min(start + rows, h)
        lo, hi = max(0, start - halo), min(h, end + halo)
        out.append(run(x[:, lo:hi])[:, (start - lo) * scale : (end - lo) * scale])
    return mx.concatenate(out, axis=1)


def reconstruct(net: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    """The upsampler of SwinIR / Swin2SR / HAT / DRCT / DAT after `conv_after_body` (+ the skip).

    `[B, h, w, C]` features -> `[B, s h, s w, 3]`, run in row bands of at most `TAIL_BAND_BYTES` for its largest map
    (`net.upsampler`: `pixelshuffle`, `nearest+conv` or `pixelshuffledirect`).
    """
    scale = int(net.upscale)
    parts: list[list[Layer]] = []
    if net.upsampler != "pixelshuffledirect":
        n_before = _convs3(net.conv_before_upsample)
        parts += stages(net.conv_before_upsample, dtype)
    else:
        n_before = 0
    if net.upsampler == "pixelshuffle":
        n_up = _convs3(net.upsample)
        parts += [*stages(net.upsample, dtype), [conv(net.conv_last, dtype)]]
        feat = net.conv_last.in_channels
        halo = n_before + n_up + 1

    elif net.upsampler == "nearest+conv":
        slope = functools.partial(lrelu, slope=float(net.lrelu.negative_slope))
        parts.append([up2, conv(net.conv_up1, dtype), slope])
        if scale == 4:  # noqa: PLR2004 - x4: two nearest steps
            parts.append([up2, conv(net.conv_up2, dtype), slope])
        parts += [[conv(net.conv_hr, dtype), slope], [conv(net.conv_last, dtype)]]
        feat = net.conv_last.in_channels
        halo = n_before + 4

    elif net.upsampler == "pixelshuffledirect":  # DAT-light: one conv and pixel shuffle
        parts += stages(net.upsample, dtype)
        feat, halo = 3, 1

    else:
        msg = f"Swin SR MLX: upsampler {net.upsampler!r} is not supported"
        raise NotImplementedError(msg)
    runs = [stepped(_chain(ops)) for ops in parts]  # a conv per step: a band's maps are not all held at once

    def run(x: mx.array) -> mx.array:
        for r in runs:
            x = r(x)
        return x

    size = 2 if dtype == mx.float16 else 4

    def apply(x: mx.array) -> mx.array:
        b, _, w, _ = x.shape
        row = b * feat * scale * scale * w * size  # bytes of one input row's largest map
        return bands(run, x, scale=scale, halo=halo, row_bytes=row, limit=TAIL_BAND_BYTES)

    return apply


# --- the Swin block --------------------------------------------------------------------------------------------------


def _conv3(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """A 3 x 3 conv (padding 1, OHWI weights) through the translator's `_conv2d`."""
    return mg._conv2d(x, w, ONE, ONE, ONE, 1) + b  # noqa: SLF001 - the translator's dispatch


def _cab(y: mx.array, p: dict[str, mx.array]) -> mx.array:
    """HAT's `CAB`: conv, GELU, conv, channel attention (`x * sigmoid(fc(relu(fc(mean(x)))))`)."""
    y = _conv3(gelu(_conv3(y, p["c1_w"], p["c1_b"])), p["c2_w"], p["c2_b"])
    a = mx.mean(y.astype(mx.float32), axis=(1, 2), keepdims=True).astype(y.dtype)
    a = mx.sigmoid(dense(mx.maximum(dense(a, p["a1_w"], p["a1_b"]), 0), p["a2_w"], p["a2_b"]))
    return y * a


BLOCK_BAND_BYTES = 32 << 20
"""Largest float32 score block of one band of a Swin block (`_block`): the block runs on its windows in bands of whole
window rows below it, each band's attention, projection and MLP one step (SwinIR-M x2 float16 at 520 x 520, one band
in flight: 64 MB 3.46 s / 956 MB, 128 MB 3.51 s / 1226 MB, one band 3.65 s / 1608 MB, 16 MB 3.70 s)."""

MAP_DEPTH = 1
"""Steps in flight when a block's step back onto the map is scheduled (`stepped`)."""
TAIL_DEPTH = 0
"""Steps in flight when a group's tail is scheduled (`stepped`): 0, run alone, the block's way back awaited first
(SwinIR-M x2 at 520 x 520, against 1: float32 4.64 s / 1.58 GB against 4.40 s / 1.91 GB, float16 3.09 s / 0.90 GB
against 3.04 s / 0.96 GB; the way back folded into the tail or the tail in row bands held more or as much)."""
BAND_DEPTH = 2
"""Steps in flight when a block's band is scheduled (`stepped`), the larger steps around them one: SwinIR-M x2
float16 at 520 x 520 with 32 MB bands 3.04 s / 980 MB, one band 3.15 s / 956 MB, 64 MB bands two in flight
3.00 s / 1135 MB."""


@functools.lru_cache(maxsize=16)
def _window_order(size: tuple[int, int, int], window: int, shift: int) -> tuple[mx.array, mx.array]:
    """`windows` / `unwindows` of a `size` = (B, H, W) map as token gathers: (the windows' token order, its inverse).

    One row gather of the flat tokens, three times faster than the roll and the six-axis transpose (SwinIR-M's
    520 x 520 x 180 float16: 1.3 against 4.6 ms; back 8.4 ms).
    """
    b, h, w = size
    order = np.array(windows(mx.array(np.arange(b * h * w, dtype=np.int32).reshape(b, h, w, 1)), window, shift))
    order = order.reshape(-1)
    out = mx.array(order), mx.array(np.argsort(order).astype(np.int32))
    mx.eval(*out)
    return out


@functools.lru_cache(maxsize=16)
def _gather_index(size: tuple[int, int, int], window: int, src: int | None, shift: int) -> mx.array:
    """The token gather onto the windows of `shift`: from a raster map (`src` None) or from tokens in another order.

    `src`: the shift of a previous block whose output tokens stayed in its window order (never put back onto the map).
    """
    order = _window_order(size, window, shift)[0]
    if src is None:
        return order
    out = mx.take(_window_order(size, window, src)[1], order)
    mx.eval(out)
    return out


@functools.lru_cache(maxsize=64)
def _band_border(border: Border, lo: int, hi: int) -> Border | None:
    """The border windows of `border` within windows `lo:hi`, indexed from `lo` (`None`: none there)."""
    index = np.array(border[0])
    sel = np.flatnonzero((index >= lo) & (index < hi))
    if not sel.size:
        return None
    out = mx.array((index[sel] - lo).astype(np.int32)), border[1][mx.array(sel.astype(np.int32))]
    mx.eval(*out)
    return out


@functools.lru_cache(maxsize=64)
def _block(  # noqa: C901, PLR0913 - the block's static geometry, its variants' steps
    heads: int,
    scale: float,
    window: int,
    shift: int,
    eps: tuple[float, float],
    *,
    post: bool,
    conv_scale: float,
) -> Callable[..., mx.array]:
    """One Swin block as a run of small steps: `(x, (B, H, W), arrays, mask, src, keep=) -> x`.

    `post`: Swin V2's post-norm; `conv_scale`: HAT's `CAB` branch on the normed input, times this, 0 without one.
    Every op but the attention is per token, so the block runs on its (shifted) windows: a step for the `CAB` branch,
    one per band of window rows (`BLOCK_BAND_BYTES`: the band's token gather (`_gather_index`), norm, qkv, attention,
    the border windows attended again with their masks, projection, skip, MLP) and one back onto the map, unless
    `keep`: then the bands' tokens stay in the block's window order and the next block gathers from them (`src`).
    A command buffer returns its transients only when it completes, and with the next step queued while one runs
    (`stepped`) two steps' transients are held: one step per block held SwinIR-M's network peak 1.5 times the graph
    translation's. The gathers inside the bands and the skipped way back measured 9 % faster (SwinIR-M x2 float16,
    520 x 520: 3.43 -> 3.11 s at the same peak) than a gather step and a step back per block.
    """

    def prep(x: mx.array, p: dict[str, Any]) -> mx.array:
        return _cab(ln(x, p["n1_w"], p["n1_b"], eps[0]), p["cab"]) * conv_scale

    def gather(x: mx.array, order: mx.array) -> mx.array:
        c = x.shape[-1]
        return mx.take(x.reshape(-1, c), order, axis=0).reshape(-1, window * window, c)

    def split(qkv: mx.array, a: dict[str, mx.array]) -> tuple[mx.array, mx.array, mx.array]:
        n_, n, c3 = qkv.shape
        c = c3 // 3
        q = qkv[..., :c] if "cosine" in a else qkv[..., :c] * scale
        q, k, v = (t.reshape(n_, n, heads, c // heads).transpose(0, 2, 1, 3) for t in (q, qkv[..., c : 2 * c],
                                                                                        qkv[..., 2 * c :]))  # fmt: skip
        if "cosine" in a:  # Swin V2: cos(q, k) * logit scale
            q = (_normalize(q) * a["cosine"]).astype(v.dtype)
            k = _normalize(k).astype(v.dtype)
        return q, k, v

    def mlp_(t: mx.array, p: dict[str, Any]) -> mx.array:
        return dense(gelu(dense(t, p["fc1_w"], p["fc1_b"])), p["fc2_w"], p["fc2_b"])

    def band(x: mx.array, order: mx.array, p: dict[str, Any], cab: mx.array | None, border: Border | None) -> mx.array:
        a = p["attn"]
        xw = gather(x, order)
        t = xw if post else ln(xw, p["n1_w"], p["n1_b"], eps[0])
        qkv = dense(t, a["qkv_w"], a.get("qkv_b"))
        o = mlx_core.attention(*split(qkv, a), scale=1.0, mask=a["bias"][None])
        if border is not None:  # the windows across regions, attended again with their masks
            index, masks = border
            o[index] = mlx_core.attention(*split(qkv[index], a), scale=1.0, mask=a["bias"] + masks)
        n_, _, n, c = o.shape
        y = dense(o.transpose(0, 2, 1, 3).reshape(n_, n, heads * c), a["proj_w"], a["proj_b"])
        if post:  # Swin V2: x + norm(attn(x)), then + norm(mlp(.))
            y = xw + ln(y, p["n1_w"], p["n1_b"], eps[0])
            return y + ln(mlp_(y, p), p["n2_w"], p["n2_b"], eps[1])
        y = xw + y if cab is None else xw + y + gather(cab, order)
        return y + mlp_(ln(y, p["n2_w"], p["n2_b"], eps[1]), p)

    def back(outs: list[mx.array], inverse: mx.array, size: tuple[int, int, int]) -> mx.array:
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        c = o.shape[-1]
        return mx.take(o.reshape(-1, c), inverse, axis=0).reshape(*size, c)

    prep_, back_ = stepped(prep), stepped(back, depth=MAP_DEPTH)
    bands_ = {
        (c, m): stepped(functools.partial(_band_args, band, cab=c, border=m), depth=BAND_DEPTH)
        for c in (False, True)
        for m in (False, True)
    }

    def f(  # noqa: PLR0913 - the input, its geometry, the arrays, the mask, the input's order and the output's
        x: mx.array, size: tuple[int, int, int], p: dict[str, Any], mask: Border | None, src: int | None, *, keep: bool
    ) -> mx.array:
        b, h, w = size
        order = _gather_index(size, window, src, shift)
        cab = prep_(x, p) if conv_scale else None
        n = window * window
        outs = []
        for lo, hi in _chunks(b * h * w // n, w // window, heads * n * n * 4, BLOCK_BAND_BYTES):
            border = _band_border(mask, lo, hi) if mask is not None else None
            args = (
                [x, order[lo * n : hi * n], p] + ([cab] if cab is not None else []) + (list(border) if border else [])
            )
            outs.append(bands_[cab is not None, border is not None](*args))
        if keep:  # the tokens in this block's window order, joined in the next step
            return mx.concatenate(outs, axis=0).reshape(-1, outs[0].shape[-1]) if len(outs) > 1 else outs[0]
        return back_(outs, _window_order(size, window, shift)[1], size)

    return f


def _band_args(  # noqa: PLR0913 - the band's map, windows and arrays, its variant
    band: Callable[..., mx.array], x: mx.array, order: mx.array, p: Any, *rest: mx.array, cab: bool, border: bool
) -> mx.array:
    """`band` on flat arguments (a compiled function's inputs are arrays and trees)."""
    c = rest[0] if cab else None
    m = (rest[-2], rest[-1]) if border else None
    return band(x, order, p, c, m)


def _mlp_norms(m: Any, dtype: mx.Dtype) -> dict[str, mx.array]:
    """A block's `norm1`, `norm2` and `mlp` arrays."""
    fc1, fc2 = linear(m.mlp.fc1, dtype), linear(m.mlp.fc2, dtype)
    return {
        "n1_w": arr(m.norm1.weight, dtype),
        "n1_b": arr(m.norm1.bias, dtype),
        "n2_w": arr(m.norm2.weight, dtype),
        "n2_b": arr(m.norm2.bias, dtype),
        "fc1_w": fc1[0],
        "fc1_b": fc1[1],
        "fc2_w": fc2[0],
        "fc2_b": fc2[1],
    }


def _ohwi(layer: Any, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    """A 3 x 3 `nn.Conv2d` (padding 1) as OHWI weights and bias."""
    return arr(layer.weight.permute(0, 2, 3, 1), dtype), arr(layer.bias, dtype)


def _fc(layer: Any, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    """A 1 x 1 `nn.Conv2d` as `[in, out]` weights and bias."""
    return arr(layer.weight[:, :, 0, 0].t(), dtype), arr(layer.bias, dtype)


class Block:
    """A `SwinTransformerBlock`: `[B, H, W, C]` -> the same.

    SwinIR, Swin2SR, DRCT's Swin layers; HAT's `HAB` with `cab`, its `CAB` module, and `conv_scale`. `attention`: its
    `Attention`; `post`: post-norm (Swin V2).
    """

    def __init__(  # noqa: PLR0913 - the block, its attention and the variants' options
        self,
        m: Any,
        attention: Attention,
        dtype: mx.Dtype,
        *,
        post: bool = False,
        shift: int | None = None,
        cab: Any = None,
        conv_scale: float = 0.0,
    ) -> None:
        """The arrays of the block `m` and its attention in `dtype`."""
        self.window = int(m.window_size)
        self.shift = int(m.shift_size if shift is None else shift)
        self.p: dict[str, Any] = {"attn": attention.params(), **_mlp_norms(m, dtype)}
        if cab is not None:
            seq_ = cab.cab
            c1, c2, ca = seq_[0], seq_[2], seq_[3].attention
            (c1w, c1b), (c2w, c2b) = _ohwi(c1, dtype), _ohwi(c2, dtype)
            (a1w, a1b), (a2w, a2b) = _fc(ca[1], dtype), _fc(ca[3], dtype)
            self.p["cab"] = {"c1_w": c1w, "c1_b": c1b, "c2_w": c2w, "c2_b": c2b, "a1_w": a1w, "a1_b": a1b,
                             "a2_w": a2w, "a2_b": a2b}  # fmt: skip
        self.dtype = dtype
        self.f = _block(
            attention.heads,
            attention.scale,
            self.window,
            self.shift,
            (float(m.norm1.eps), float(m.norm2.eps)),
            post=post,
            conv_scale=float(conv_scale) if cab is not None else 0.0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """The block's output map."""
        return self.chain(x, tuple(x.shape[:3]), None, keep=False)

    def chain(self, x: mx.array, size: tuple[int, int, int], src: int | None, *, keep: bool) -> mx.array:
        """The block on a `size` map, or on the tokens of a previous block (`src`: its shift).

        `keep`: the output tokens left in this block's window order (`Group`: a next block without a `CAB` gathers
        from them directly).
        """
        mask = border_mask(size, self.window, self.shift, self.dtype) if self.shift else None
        return self.f(x, size, self.p, mask, src, keep=keep)


def overlap_windows(kv: mx.array, window: int, overlap: int) -> mx.array:
    """`nn.Unfold(overlap, stride=window, padding=(overlap - window) // 2)` of an NHWC map, as HAT's OCAB reads it.

    `[B, H, W, C]` -> `[B * nW, overlap**2, C]` (zero padding).
    """
    b, h, w, c = kv.shape
    pad = (overlap - window) // 2
    kv = mx.pad(kv, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    nh, nw = h // window, w // window
    rows = mx.array((np.arange(nh)[:, None] * window + np.arange(overlap)[None]).reshape(-1).astype(np.int32))
    cols = mx.array((np.arange(nw)[:, None] * window + np.arange(overlap)[None]).reshape(-1).astype(np.int32))
    kv = mx.take(mx.take(kv, rows, axis=1), cols, axis=2)  # [B, nh * overlap, nw * overlap, C]
    kv = kv.reshape(b, nh, overlap, nw, overlap, c).transpose(0, 1, 3, 2, 4, 5)
    return kv.reshape(b * nh * nw, overlap * overlap, c)


@functools.lru_cache(maxsize=16)
def _ocab(heads: int, scale: float, window: int, overlap: int, eps: tuple[float, float]) -> Callable[..., mx.array]:
    """HAT's `OCAB` as a step: overlapping cross-attention, its projection and the skip, then the MLP.

    Each window's queries attend to the keys / values of a larger, zero-padded window around it.
    """

    def f(x: mx.array, p: dict[str, Any]) -> mx.array:
        b, h, w, c = x.shape
        qkv = dense(ln(x, p["n1_w"], p["n1_b"], eps[0]), p["qkv_w"], p["qkv_b"])
        q = windows(qkv[..., :c] * scale, window, 0)
        kv = overlap_windows(qkv[..., c:], window, overlap)
        o = attend(q, kv[..., :c], kv[..., c:], p["bias"], None, heads=heads, row=w // window)
        x = dense(unwindows(o, (b, h, w), window, 0), p["proj_w"], p["proj_b"]) + x
        return x + dense(gelu(dense(ln(x, p["n2_w"], p["n2_b"], eps[1]), p["fc1_w"], p["fc1_b"])),
                         p["fc2_w"], p["fc2_b"])  # fmt: skip

    return stepped(f)


class OverlapBlock:
    """HAT's `OCAB` (`bias`: its gathered relative position bias `[nH, window**2, overlap**2]`, a torch tensor)."""

    def __init__(self, m: Any, bias: Any, dtype: mx.Dtype) -> None:
        """The arrays of HAT's `OCAB` `m` and its position `bias` in `dtype`."""
        w, b = linear(m.qkv, dtype)
        pw, pb = linear(m.proj, dtype)
        self.p = {"qkv_w": w, "qkv_b": b, "proj_w": pw, "proj_b": pb, "bias": arr(bias, dtype), **_mlp_norms(m, dtype)}
        self.f = _ocab(
            int(m.num_heads),
            float(m.scale),
            int(m.window_size),
            int(m.overlap_win_size),
            (float(m.norm1.eps), float(m.norm2.eps)),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """The block's output (one step)."""
        return self.f(x, self.p)


class Group:
    """An RSTB: `convs(blocks(x)) + x`, the last conv and the skip times `scale` (float16 headroom of a last group).

    `convs`: its `conv`, a conv or `resi_connection` 3conv's Sequential, and any 1 x 1 projection after it.
    """

    def __init__(self, blocks: list[Any], convs: list[Any], dtype: mx.Dtype, *, scale: float = 1.0) -> None:
        """The group's built `blocks` and the arrays of its `convs` in `dtype`."""
        from torch import nn  # noqa: PLC0415 - the module types of a built network

        self.blocks = blocks
        mods = [c for layer in convs for c in (layer if isinstance(layer, nn.Sequential) else [layer])]
        last = max(i for i, c in enumerate(mods) if isinstance(c, nn.Conv2d))
        ops: list[Layer] = []
        for i, c in enumerate(mods):
            if isinstance(c, nn.Conv2d):
                ops.append(conv(c, dtype, scale=scale if i == last else 1.0))
            elif isinstance(c, nn.LeakyReLU):
                ops.append(functools.partial(lrelu, slope=float(c.negative_slope)))
            else:
                msg = f"Swin SR MLX: unsupported layer {type(c).__name__} in a group's conv"
                raise NotImplementedError(msg)

        def tail(y: mx.array, x: mx.array) -> mx.array:
            for op in ops:
                y = op(y)
            return y + (x * scale if scale != 1.0 else x)

        self.tail = stepped(tail, depth=TAIL_DEPTH)

    def __call__(self, x: mx.array) -> mx.array:
        """The group's output."""
        y, src, size = x, None, tuple(x.shape[:3])
        for b, nxt in zip(self.blocks, [*self.blocks[1:], None], strict=True):
            if not isinstance(b, Block):
                y, src = b(y), None
                continue
            keep = isinstance(nxt, Block) and not nxt.p.get("cab") and nxt.window == b.window
            y = b.chain(y, size, src, keep=keep)
            src = b.shift if keep else None
        return self.tail(y, x)


class Network:
    """The SwinIR-style shell: `conv_first`, the groups, `norm`, `conv_after_body` + skip, the upsampler.

    `groups`: the residual groups as callables on `[B, H, W, C]` maps; `norm_eps_scale`: the final norm's epsilon
    factor (a scaled last group); `img_range` and the input mean as the network's.
    """

    def __init__(self, net: Any, groups: list[Any], dtype: mx.Dtype, *, norm_eps_scale: float = 1.0) -> None:
        """The head, the built `groups`, the norm and the tail of `net` in `dtype`."""
        self.groups = groups
        self.dtype = dtype
        first = conv(net.conv_first, dtype)
        embed = getattr(net, "patch_embed", None)
        proj = getattr(embed, "proj", None)  # Swin2SR: a 1 x 1 conv
        embed_proj = None if proj is None else conv(proj, dtype)
        norm0 = net.before_RG[1] if embed is None else embed.norm  # DAT: `before_RG`'s LayerNorm after the tokens
        embed_norm = None if norm0 is None else layer_norm(norm0, dtype)
        self.mean = arr(net.mean.reshape(-1), mx.float32)
        self.img_range = float(net.img_range)
        rng = self.img_range

        def head(x: mx.array, mean: mx.array) -> tuple[mx.array, mx.array]:
            f = first(((x - mean) * rng).astype(dtype))
            y = f if embed_proj is None else embed_proj(f)
            return f, y if embed_norm is None else embed_norm(y)

        self.head = stepped(head)
        norm = layer_norm(net.norm, dtype, eps_scale=norm_eps_scale)
        after, _ = seq(net.conv_after_body, dtype)
        self.body_tail = stepped(lambda y, f: after(norm(y)) + f)
        self.tail = reconstruct(net, dtype)

    def forward(self, x: mx.array) -> mx.array:
        """`[B, 3, H, W]` float32 (padded) -> `[B, 3, sH, sW]` float32."""
        f, y = self.head(x.transpose(0, 2, 3, 1), self.mean)
        for g in self.groups:
            y = g(y)
        y = self.body_tail(y, f)
        out = self.tail(y).transpose(0, 3, 1, 2)
        return out.astype(mx.float32) / self.img_range + self.mean.reshape(-1, 1, 1)

    def __call__(self, x: mx.array) -> mx.array:
        """`forward` with its steps (the result evaluated)."""
        mg._KERNELS, mg._GEMM1X1, mg._HALF_WINO_F32, mg._HALF_WINO_PAD = False, True, False, False  # noqa: SLF001 - the translator's portable defaults
        out = steps(self.forward, x)
        mx.eval(out)
        return out
