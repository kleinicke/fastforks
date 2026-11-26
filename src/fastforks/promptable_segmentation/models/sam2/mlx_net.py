"""SAM 2's image encoder (Hiera trunk + the folded FPN neck of `fast.py`) written in MLX, channels last.

Written out instead of replayed from an exported graph: the export, translation and segment tuning were most of the
first call (1.6 to 3.8 s on an M1 Pro; 0.15 to 0.6 s like this).

- A block is `fast._block_f32_stream`: in float16 the residual stream stays float32 (the norms read it, attention and
  the MLP run in float16, the adds are float32), and a stage transition's projection, whose pooled output *is* the
  stream there, runs in float32 with float32 weights.
- Attention is MLX's fused `scaled_dot_product_attention` (float32 softmax inside), which has a fused kernel at head
  dims 64, 80 and 128 only. Hiera's head dim is 96 (Tiny, Small), 56 (Base Plus) or 72 (Large). 56 is padded to 64 in
  the weights (`PAD_TO`), as zero output columns of the q / k / v projection and zero input rows of the output
  projection: the scores and the result are the same sums plus zeros, and Base Plus is 10 % faster for it. Padding
  96 to 128 or 72 to 80 was measured slower than the unfused path (the wider GEMMs cost more than the kernel saves),
  so those stay as they are.
- The layer norms are Metal kernels of our own (`_NORM`), one SIMD group per token: `mx.fast.layer_norm` costs 3.6 ms on
  the 256 x 256 x 96 map of stage 1 (a quarter of that block) and ours 0.4 ms. Each kernel also does the work around
  the norm in the same pass: `_LN_PART` adds the MLP output of the block before to the stream, normalises, casts and
  writes the tokens in window order (zero rows where the windows overhang the map, which are keys as upstream's);
  `_ADD_LN` reads the attention output in window order, adds it to the stream and normalises for the MLP. A block is
  then two of these, four GEMMs, the attention and the GELU (Tiny's stage-1 block 14.8 -> 6.8 ms, Large's stage-3
  block 10.0 -> 8.8 ms, of which the GEMMs are 7.1).
- So a block hands on its stream and its MLP output apart (`(x, m)`), and the block after it adds them.
- The query pooling of a stage's first block is a 2 x 2 max over each window.
- The blocks are evaluated one by one, since one lazy graph of the trunk keeps every block's temporaries alive at
  once. From stage 2 on a block is started (`mx.async_eval`) before the one before it is waited for, so the GPU never
  idles while Python builds the next graph. The stage-1 blocks (256 x 256 tokens, 0.3 to 0.5 GB of temporaries each)
  are evaluated alone: two of them in flight were 3 to 5 % faster and 0.3 to 0.5 GB more peak memory.
"""

# ruff: noqa: ANN401 - the upstream trunk's torch modules

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import metal_kernel, mx, to_mx, to_torch
from fastforks.shared.mlx_vit import gelu

if TYPE_CHECKING:
    import torch

    from .fast import Lin, Neck

__all__ = ["MlxEncoder"]

SYNC_TOKENS = 1 << 16  # blocks on maps this large are evaluated alone (see the module docstring)
PAD_TO = {56: 64}  # head dims padded to one MLX's attention runs fused, where that was measured faster


LANES = 32  # fixed: threads of a SIMD group on Apple GPUs (the kernels' thread group; `_kernels` checks it)

# One SIMD group per token: lane `l` reads the channels l, l + 32, ...; `simd_sum` adds the lanes' partial sums.
# `SRC(c)` is the token's channel c (the stream, or the stream plus a branch), `OUT(c)` stores it where it is a sum.
_NORM = """
    float acc = 0;
    for (uint c = lane; c < C; c += 32) acc += SRC(c);
    float mean = simd_sum(acc) / float(C);
    acc = 0;
    for (uint c = lane; c < C; c += 32) { float d = SRC(c) - mean; acc += d * d; }
    float inv = rsqrt(simd_sum(acc) / float(C) + eps[0]);
    for (uint c = lane; c < C; c += 32) { OUT(c) h[o + c] = (SRC(c) - mean) * inv * g[c] + b[c]; }
#undef SRC
#undef OUT
"""
# dims: channels, map height and width, window height and width, windows down and across (of the padded map)
_DIMS = """
    uint C = dims[0], H = dims[1], W = dims[2], wh = dims[3], ww = dims[4], nh = dims[5], nw = dims[6];
    uint lane = thread_position_in_grid.x, r = thread_position_in_grid.y;
"""
# A thread group per row of the window-ordered output `h` `[B nh nw, wh, ww, C]`; `s` is its token in the map.
_LN_PART = (
    _DIMS
    + """
    uint t = r % (wh * ww), wi = r / (wh * ww);
    uint yy = ((wi / nw) % nh) * wh + t / ww, xx = (wi % nw) * ww + t % ww;
    uint o = r * C;
    if (yy >= H || xx >= W) { for (uint c = lane; c < C; c += 32) h[o + c] = 0; return; }
    uint s = (((wi / (nw * nh)) * H + yy) * W + xx) * C;
"""
)
# A thread group per token of the map `[B, H, W, C]`; `s` is its row in the window-ordered `y`.
_ADD_LN = (
    _DIMS
    + """
    uint xx = r % W, yy = (r / W) % H;
    uint s = ((((r / (W * H) * nh + yy / wh) * nw + xx / ww) * wh + yy % wh) * ww + xx % ww) * C;
    uint o = r * C;
#define SRC(c) (x[o + (c)] + float(y[s + (c)]))
#define OUT(c) xo[o + (c)] = SRC(c);
"""
    + _NORM
)


@cache
def _kernels() -> tuple[Callable[..., list[mx.array]], Callable[..., list[mx.array]], Callable[..., list[mx.array]]]:
    """The norm kernels: `ln(x) -> h`, `ln_add(x, m) -> xo, h` (both window-ordered `h`), `add_ln(x, y) -> xo, h`.

    Raises if a thread group of `LANES` threads is not one SIMD group here (`simd_sum` would add fewer lanes).
    """
    tail = ["g", "b", "eps", "dims"]
    plain = "#define SRC(c) x[s + (c)]\n#define OUT(c)\n"
    added = "#define SRC(c) (x[s + (c)] + float(m[s + (c)]))\n#define OUT(c) xo[s + (c)] = SRC(c);\n"
    ln = metal_kernel(name="sam2_ln", input_names=["x", *tail], output_names=["h"], source=_LN_PART + plain + _NORM)
    ln_add = metal_kernel(
        name="sam2_ln_add", input_names=["x", "m", *tail], output_names=["xo", "h"], source=_LN_PART + added + _NORM
    )
    add_ln = metal_kernel(name="sam2_add_ln", input_names=["x", "y", *tail], output_names=["xo", "h"], source=_ADD_LN)
    x = mx.arange(3 * 70, dtype=mx.float32).reshape(1, 1, 3, 70) % 7
    one, zero = mx.ones(70), mx.zeros(70)
    (got,) = ln(
        inputs=[x, one, zero, mx.array([1e-6]), mx.array([70, 1, 3, 1, 3, 1, 1], dtype=mx.uint32)],
        grid=(LANES, 3, 1),
        threadgroup=(LANES, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
    )
    if mx.abs(got - mx.fast.layer_norm(x, one, zero, 1e-6)).max().item() > 1e-4:  # noqa: PLR2004 - float32 rounding
        msg = f"SAM 2 on MLX: this GPU's SIMD groups are not {LANES} threads wide (the layer-norm kernels need that)"
        raise RuntimeError(msg)
    return ln, ln_add, add_ln


def _dims(c: int, size: tuple[int, int], window: int) -> tuple[mx.array, tuple[int, int, int]]:
    """A kernel's `dims` for a map of `size` in windows of `window` (0: the whole map) and `(windows, wh, ww)`."""
    h, w = size
    wh, ww = (window, window) if window else (h, w)
    nh, nw = -(-h // wh), -(-w // ww)
    return mx.array([c, h, w, wh, ww, nh, nw], dtype=mx.uint32), (nh * nw, wh, ww)


def arr(t: torch.Tensor | np.ndarray, dtype: mx.Dtype) -> mx.array:
    """A weight as an evaluated MLX array of `dtype`."""
    v = t if isinstance(t, np.ndarray) else t.detach().double().cpu().numpy()
    a = mx.array(np.ascontiguousarray(v, dtype=np.float32)).astype(dtype)
    mx.eval(a)
    return a


def linear(layer: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    """`torch.nn.Linear` as one `addmm` in `dtype`."""
    w, b = arr(layer.weight.T, dtype), arr(layer.bias, dtype)
    return lambda x: mx.addmm(b, x, w)


def layer_norm(norm: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    """`torch.nn.LayerNorm` (or upstream's `LayerNorm2d`, channels last here) over the last axis, output in `dtype`."""
    ln, _, _ = _kernels()
    g, b, eps = arr(norm.weight, mx.float32), arr(norm.bias, mx.float32), mx.array([float(norm.eps)])
    c = g.shape[0]

    def run(x: mx.array) -> mx.array:
        rows = x.size // c
        dims = mx.array([c, rows, 1, rows, 1, 1, 1], dtype=mx.uint32)
        (h,) = ln(
            inputs=[x, g, b, eps, dims],
            grid=(LANES, rows, 1),
            threadgroup=(LANES, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[dtype],
        )
        return h

    return run


class Twins:
    """MLX arrays `[B, H, W, C]` and the float32 NCHW tensors made of them, found again by the tensor's memory."""

    def __init__(self, keep: int = 8) -> None:
        """Remember the last `keep` maps (two frames' features and conditioned maps)."""
        self.items: deque[tuple[torch.Tensor, mx.array]] = deque(maxlen=keep)

    def tensor(self, a: mx.array) -> torch.Tensor:
        """`a` as a float32 NCHW tensor (a view of the channels-last copy), remembered."""
        t = to_torch(a).permute(0, 3, 1, 2)
        self.items.append((t, a))
        return t

    def array(self, t: torch.Tensor) -> mx.array:
        """The float32 channels-last MLX array of NCHW `t`: the remembered one if `t` is such a tensor or its `expand`.

        An expanded batch (stride 0: every row the same memory) comes back with batch 1.
        """
        for c, a in self.items:
            same = (
                t.untyped_storage().data_ptr() == c.untyped_storage().data_ptr()
                and t.storage_offset() == c.storage_offset()
                and t.shape[1:] == c.shape[1:]
                and t.stride()[1:] == c.stride()[1:]
            )
            if same and (t.shape[0] == c.shape[0] or (c.shape[0] == 1 and t.stride(0) == 0)):
                return a
        if t.shape[0] > 1 and t.stride(0) == 0:
            t = t[:1]
        return to_mx(t.float().permute(0, 2, 3, 1))


def _pool(x: mx.array) -> mx.array:
    """2 x 2 / stride-2 max pooling of `[B, H, W, C]` (even sides)."""
    b, h, w, c = x.shape
    return x.reshape(b, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def _attention(attn: Any, dtype: mx.Dtype, pad_to: dict[int, int]) -> Callable[[mx.array], mx.array]:
    """`MultiScaleAttention.forward` on windows `[B, H, W, C]` -> `[B, H', W', C_out]` (the queries pooled or not)."""
    heads = attn.num_heads
    d = attn.dim_out // heads
    dp = pad_to.get(d, d)
    w = attn.qkv.weight.detach().double().numpy().reshape(3, heads, d, attn.dim)
    b = attn.qkv.bias.detach().double().numpy().reshape(3, heads, d)
    o = attn.proj.weight.detach().double().numpy().T.reshape(heads, d, attn.dim_out)
    if dp != d:
        w = np.pad(w, ((0, 0), (0, 0), (0, dp - d), (0, 0)))
        b = np.pad(b, ((0, 0), (0, 0), (0, dp - d)))
        o = np.pad(o, ((0, 0), (0, dp - d), (0, 0)))
    qkv_w, qkv_b = arr(w.reshape(3 * heads * dp, attn.dim).T, dtype), arr(b.reshape(-1), dtype)
    out_w, out_b = arr(o.reshape(heads * dp, attn.dim_out), dtype), arr(attn.proj.bias, dtype)
    pooled, scale = attn.q_pool is not None, d**-0.5

    def run(x: mx.array) -> mx.array:
        n, h, wd, _ = x.shape
        qkv = mx.addmm(qkv_b, x.reshape(n, h * wd, -1), qkv_w).reshape(n, h * wd, 3, heads, dp)
        q = qkv[:, :, 0]
        if pooled:
            q = _pool(q.reshape(n, h, wd, heads * dp))
            h, wd = h // 2, wd // 2
            q = q.reshape(n, h * wd, heads, dp)
        k, v = qkv[:, :, 1].transpose(0, 2, 1, 3), qkv[:, :, 2].transpose(0, 2, 1, 3)
        y = mx.fast.scaled_dot_product_attention(q.transpose(0, 2, 1, 3), k, v, scale=scale)
        return mx.addmm(out_b, y.transpose(0, 2, 1, 3).reshape(n, h, wd, heads * dp), out_w)

    return run


Block = Callable[[mx.array, mx.array | None], tuple[mx.array, mx.array, mx.array]]


def _block(blk: Any, dtype: mx.Dtype, stream: mx.Dtype, pad_to: dict[int, int]) -> Block:
    """`MultiScaleBlock.forward` on a `stream`-dtype map, attention and MLP in `dtype`.

    `run(x, m)` takes the stream and the MLP output of the block before (`None`: nothing to add) and returns its input
    `x + m`, its stream after the attention and its MLP output: the block's result is the sum of the last two.
    """
    ln, ln_add, add_ln = _kernels()
    g1, b1, g2, b2 = (arr(t, stream) for t in (blk.norm1.weight, blk.norm1.bias, blk.norm2.weight, blk.norm2.bias))
    eps1, eps2 = mx.array([float(blk.norm1.eps)]), mx.array([float(blk.norm2.eps)])
    proj = linear(blk.proj, stream) if blk.dim != blk.dim_out else None
    attn = _attention(blk.attn, dtype, pad_to)
    fc1, fc2 = (linear(layer, dtype) for layer in blk.mlp.layers)
    window, pooled, cin, cout = blk.window_size, blk.q_stride is not None, blk.dim, blk.dim_out

    def run(x: mx.array, m: mx.array | None) -> tuple[mx.array, mx.array, mx.array]:
        bsz, size = x.shape[0], (x.shape[1], x.shape[2])
        # a projected block reads the norm twice (shortcut in the stream's dtype, attention in windows): whole map
        dims, (n, wh, ww) = _dims(cin, size, 0 if proj is not None else window)
        rows, shape = bsz * n * wh * ww, (bsz * n, wh, ww, cin)
        kind = stream if proj is not None else dtype
        grid = {"grid": (LANES, rows, 1), "threadgroup": (LANES, 1, 1)}
        if m is None:
            (h,) = ln(inputs=[x, g1, b1, eps1, dims], output_shapes=[shape], output_dtypes=[kind], **grid)
        else:
            x, h = ln_add(
                inputs=[x, m, g1, b1, eps1, dims], output_shapes=[x.shape, shape], output_dtypes=[stream, kind], **grid
            )
        entry, shortcut = x, x
        if proj is not None:
            shortcut = proj(h)
            shortcut = _pool(shortcut) if pooled else shortcut
            h = h.astype(dtype)
            if window:
                ph, pw = -size[0] % window, -size[1] % window
                if ph or pw:
                    h = mx.pad(h, ((0, 0), (0, ph), (0, pw), (0, 0)))
                hp, wp = size[0] + ph, size[1] + pw
                h = h.reshape(bsz, hp // window, window, wp // window, window, cin).transpose(0, 1, 3, 2, 4, 5)
                h = h.reshape(-1, window, window, cin)
        y = attn(h)
        size = (shortcut.shape[1], shortcut.shape[2])
        dims, _ = _dims(cout, size, window // 2 if pooled else window)
        x, h = add_ln(
            inputs=[shortcut, y, g2, b2, eps2, dims],
            grid=(LANES, bsz * size[0] * size[1], 1),
            threadgroup=(LANES, 1, 1),
            output_shapes=[shortcut.shape, shortcut.shape],
            output_dtypes=[stream, dtype],
        )
        return entry, x, fc2(gelu(fc1(h)))

    return run


class MlxEncoder:
    """Hiera trunk and neck of one loaded SAM 2 on MLX: `fast.trunk` + `fast.neck`."""

    def __init__(  # noqa: PLR0913 - the network's parts and two options
        self,
        trunk: Any,
        neck: Neck,
        pos: torch.Tensor,
        dtype_name: str,
        *,
        pad_to: dict[int, int] | None = None,
        twins: Twins | None = None,
    ) -> None:
        """Copy the weights of upstream `Hiera` (`trunk`), the folded `neck` and the position map `[1, T, T, C]`.

        `pad_to` replaces `PAD_TO` (the tests pad their small trunk's head dims). With `twins` the outputs are views
        of channels-last tensors, remembered there with their MLX arrays.
        """
        self.twins = twins
        dtype = getattr(mx, dtype_name)
        stream = mx.float32
        self.dtype, self.stream = dtype, stream
        conv = trunk.patch_embed.proj
        self.stride, self.padding = conv.stride[0], conv.padding[0]
        self.patch_w = arr(conv.weight.permute(0, 2, 3, 1), dtype)
        self.patch_b = arr(conv.bias, dtype)
        self.pos = arr(pos, dtype)
        self.blocks = [_block(blk, dtype, stream, PAD_TO if pad_to is None else pad_to) for blk in trunk.blocks]
        self.stage_ends = set(trunk.stage_ends)

        def lin(t: Lin) -> Callable[[mx.array], mx.array]:
            w, b = arr(t[0].T, dtype), arr(t[1], dtype)
            return lambda x: mx.addmm(b, x.astype(dtype), w).astype(mx.float32)

        self.top, self.mid, self.s1, self.s0 = lin(neck.top), lin(neck.mid), lin(neck.s1), lin(neck.s0)
        self.no_mem = arr(neck.no_mem, mx.float32)

    def features(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Normalised `[B, S, S, 3]` -> float32 channels-last `(image_embed, feat_s1, feat_s0)`."""
        x = mx.conv2d(x.astype(self.dtype), self.patch_w, stride=self.stride, padding=self.padding)
        x = (x + self.patch_b + self.pos).astype(self.stream)
        feats, prev, m = [], None, None
        for i, blk in enumerate(self.blocks):
            big = x.shape[1] * x.shape[2] >= SYNC_TOKENS
            entry, x, m = blk(x, m)
            if i - 1 in self.stage_ends:
                feats.append(entry)  # the stage's output: this block's kernel added the last MLP output to the stream
            if big:
                mx.eval(x, m)
                prev = None
            else:
                mx.async_eval(x, m)
                if prev is not None:
                    mx.eval(*prev)
                prev = (x, m)
        if m is not None:
            feats.append(x + m.astype(self.stream))
        f0, f1, f2, f3 = feats
        top = self.top(f3)
        b, h, w, c = top.shape
        up = mx.broadcast_to(top[:, :, None, :, None], (b, h, 2, w, 2, c)).reshape(b, 2 * h, 2 * w, c)
        return self.mid(f2) + up + self.no_mem, self.s1(f1), self.s0(f0)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Normalised NCHW `[B, 3, S, S]` -> float32 NCHW `image_embed`, `feat_s1`, `feat_s0` (as `fast.neck`)."""
        out = self.features(to_mx(image.float().permute(0, 2, 3, 1)))
        mx.eval(*out)
        keys = ("image_embed", "feat_s1", "feat_s0")
        if self.twins is not None:  # views of the channels-last copies; the MLX arrays are found again by them
            return {k: self.twins.tensor(a) for k, a in zip(keys, out, strict=True)}
        return {k: to_torch(a).permute(0, 3, 1, 2).contiguous() for k, a in zip(keys, out, strict=True)}
