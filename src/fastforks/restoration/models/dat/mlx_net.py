"""DAT on Apple MLX, hand-written (`fastforks.shared.swin_sr.mlx_net`'s shell and attention): NHWC maps, the run dtype.

The network of `model.Portable`. Blocks alternate between

- spatial (`Adaptive_Spatial_Attention`): q / k / v zero-padded after the projection to a multiple of the long window
  side, two branches on the channel halves with transposed rectangle windows (8 x 32 and 32 x 8), shifted in every
  fourth block as upstream's `rg_idx` / `b_idx` rule says, the dynamic position biases made at load (the MLP's input
  is a constant), the masks once per size; the projection's rows are reordered at load so each branch's q, k and v
  are one contiguous slice;
- channel (`Adaptive_Channel_Attention`): the per-head `d x d` attention over all tokens as one `C x C` product with
  the off-head entries masked out of the softmax (a block-diagonal attention matrix, applied as one product), q and k
  L2-normalised over the tokens in float32 first;

each with its depthwise-conv branch and the adaptive interaction (the eval BatchNorms folded into the convs before
them), then the spatial-gate feed-forward (`fc1` as two products, one per gate half). float16 divides the last group's
final conv and its skip by `model._LAST_SCALE` as Portable does, the final norm's epsilon by its square.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared.swin_sr import mlx_net as swin

if TYPE_CHECKING:
    from collections.abc import Callable

LAYOUT = "dat-v1"
SOURCES = (Path(__file__), Path(swin.__file__))

_LAST_SCALE = 64.0  # as `model._LAST_SCALE` (float16 only)


def _fold(conv: Any, bn: Any) -> tuple[Any, Any]:
    """`fuse_conv_bn_eval` of a conv and its eval BatchNorm as torch tensors (on meta, a replay: the conv's shapes).

    No arithmetic on meta tensors: it imports `torch._dynamo` (about a second of load).
    """
    if conv.weight.is_meta:
        return conv.weight, conv.bias
    k = bn.weight / (bn.running_var + bn.eps).sqrt()
    return conv.weight * k.reshape(-1, 1, 1, 1), (conv.bias - bn.running_mean) * k + bn.bias


def _dw(w: Any, b: Any, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    """A depthwise 3 x 3 conv `[C, 1, 3, 3]` as `[3, 3, C]` taps (`ops.depthwise`) and bias."""
    return swin.arr(w[:, 0].permute(1, 2, 0), dtype), swin.arr(b, dtype)


def _fc(w: Any, b: Any, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    """A 1 x 1 conv `[out, in, 1, 1]` as `[in, out]` weights and bias."""
    return swin.arr(w[:, :, 0, 0].t(), dtype), swin.arr(b, dtype)


def _interaction(m: Any, dtype: mx.Dtype) -> dict[str, mx.array]:
    """The arrays of an attention's conv branch and its channel / spatial interactions (BatchNorms folded)."""
    dw, ci, si = m.dwconv, m.channel_interaction, m.spatial_interaction
    p = {}
    p["dw_w"], p["dw_b"] = _dw(*_fold(dw[0], dw[1]), dtype)
    p["c1_w"], p["c1_b"] = _fc(*_fold(ci[1], ci[2]), dtype)
    p["c2_w"], p["c2_b"] = _fc(ci[4].weight, ci[4].bias, dtype)
    p["s1_w"], p["s1_b"] = _fc(*_fold(si[0], si[1]), dtype)
    p["s2_w"], p["s2_b"] = _fc(si[3].weight, si[3].bias, dtype)
    p["proj_w"], p["proj_b"] = swin.linear(m.proj, dtype)
    return p


def _depthwise(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """A depthwise 3 x 3 conv, padding 1, on the graph arm's Metal kernel (`mx.conv2d(groups=C)`: far slower at 180)."""
    return ops.depthwise(x, w, b, swin.ONE, swin.ONE, swin.ONE)


def _channel_map(x: mx.array, p: dict[str, mx.array]) -> mx.array:
    """`channel_interaction`: `[B, H, W, C]` -> `[B, 1, 1, C]` (before the sigmoid)."""
    a = mx.mean(x.astype(mx.float32), axis=(1, 2), keepdims=True).astype(x.dtype)
    return swin.dense(swin.gelu(swin.dense(a, p["c1_w"], p["c1_b"])), p["c2_w"], p["c2_b"])


def _spatial_map(x: mx.array, p: dict[str, mx.array]) -> mx.array:
    """`spatial_interaction`: `[B, H, W, C]` -> `[B, H, W, 1]` (before the sigmoid)."""
    return swin.dense(swin.gelu(swin.dense(x, p["s1_w"], p["s1_b"])), p["s2_w"], p["s2_b"])


def _spatial(  # noqa: PLR0913 - the block's static geometry
    x: mx.array,
    p: dict[str, Any],
    masks: tuple[mx.array, ...],
    *,
    heads: int,
    scale: float,
    split: tuple[int, int],
    shift: tuple[int, int] | None,
) -> mx.array:
    """`Adaptive_Spatial_Attention.forward` of a normed `[B, H, W, C]` map (`p`: its arrays, q / k / v reordered)."""
    b, h, w, c = x.shape
    half = c // 2
    qkv = swin.dense(x, p["qkv_w"], p["qkv_b"])  # [q0 k0 v0 q1 k1 v1]
    v = mx.concatenate([qkv[..., 2 * half : 3 * half], qkv[..., 5 * half :]], axis=-1)
    s = max(split)
    pad_b, pad_r = -h % s, -w % s
    if pad_b or pad_r:
        qkv = mx.pad(qkv, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
    hp, wp = h + pad_b, w + pad_r
    outs = []
    for i, window in enumerate((split, split[::-1])):
        part = qkv[..., 3 * half * i : 3 * half * (i + 1)]
        sh = (0, 0) if shift is None else (shift if i == 0 else shift[::-1])
        t = swin.windows(part, window, sh)
        q, k, vv = t[..., :half] * scale, t[..., half : 2 * half], t[..., 2 * half :]
        mask = masks[i] if shift is not None else None
        o = swin.attend(q, k, vv, p["bias"][i], mask, heads=heads, row=wp // window[1])
        outs.append(swin.unwindows(o, (b, hp, wp), window, sh))
    att = mx.concatenate(outs, axis=-1)
    if pad_b or pad_r:
        att = att[:, :h, :w]
    conv = swin.gelu(_depthwise(v, p["dw_w"], p["dw_b"]))
    y = att * mx.sigmoid(_channel_map(conv, p)) + mx.sigmoid(_spatial_map(att, p)) * conv
    return swin.dense(y, p["proj_w"], p["proj_b"])


def _normalized(x: mx.array) -> mx.array:
    """`F.normalize(x, dim=tokens)` of `[B, N, C]` (the norms in float32), in `x`'s dtype."""
    y = x.astype(mx.float32)
    n = mx.sqrt(mx.sum(y * y, axis=1, keepdims=True))
    return (y / mx.maximum(n, 1e-12)).astype(x.dtype)


def _channel(x: mx.array, p: dict[str, Any]) -> mx.array:
    """`Adaptive_Channel_Attention.forward` of a normed `[B, H, W, C]` map.

    The per-head scores `q_h^T k_h` (`d x d`, summed over the tokens) are the diagonal blocks of the `C x C` product
    `q^T k`; `p["block"]` (0 within a head, -inf across) leaves only them in the softmax, so the attention is one
    block-diagonal `C x C` matrix and `attn @ v` one product on the tokens.
    """
    b, h, w, c = x.shape
    qkv = swin.dense(x, p["qkv_w"], p["qkv_b"]).reshape(b, h * w, 3 * c)
    q, k, v = _normalized(qkv[..., :c]), _normalized(qkv[..., c : 2 * c]), qkv[..., 2 * c :]
    s = (q.swapaxes(1, 2) @ k).astype(mx.float32) * p["temperature"] + p["block"]
    a = mx.softmax(s, axis=-1, precise=True).astype(x.dtype)  # [B, C, C]
    att = (v @ a.swapaxes(1, 2)).reshape(b, h, w, c)
    conv = swin.gelu(_depthwise(v.reshape(b, h, w, c), p["dw_w"], p["dw_b"]))
    y = att * mx.sigmoid(_spatial_map(conv, p)) + conv * mx.sigmoid(_channel_map(att, p))
    return swin.dense(y, p["proj_w"], p["proj_b"])


def _ffn(x: mx.array, p: dict[str, mx.array], eps: float) -> mx.array:
    """`SGFN.forward`: `fc1` as two products (one per gate half), the gate's norm and depthwise conv, `fc2`."""
    x1 = swin.gelu(swin.dense(x, p["f1a_w"], p["f1a_b"]))
    x2 = swin.gelu(swin.dense(x, p["f1b_w"], p["f1b_b"]))
    x2 = _depthwise(swin.ln(x2, p["g_n_w"], p["g_n_b"], eps), p["g_w"], p["g_b"])
    return swin.dense(x1 * x2, p["f2_w"], p["f2_b"])


@functools.lru_cache(maxsize=32)
def _block(
    heads: int,
    scale: float,
    split: tuple[int, int] | None,
    shift: tuple[int, int] | None,
    eps: tuple[float, float, float],
) -> Callable[..., mx.array]:
    """One `DATB` as a step: `(x [B, H, W, C], arrays, *masks) -> x` (`split` None: a channel block)."""

    def f(x: mx.array, p: dict[str, Any], *masks: mx.array) -> mx.array:
        y = swin.ln(x, p["n1_w"], p["n1_b"], eps[0])
        if split is None:
            x = x + _channel(y, p)
        else:
            x = x + _spatial(y, p, masks, heads=heads, scale=scale, split=split, shift=shift)
        return x + _ffn(swin.ln(x, p["n2_w"], p["n2_b"], eps[1]), p, eps[2])

    return swin.stepped(f)


def _shifted(rg: int, b: int) -> bool:
    """Upstream's rule for a shifted spatial block (block `b` of group `rg`)."""
    return (rg % 2 == 0 and b > 0 and (b - 2) % 4 == 0) or (rg % 2 != 0 and b % 4 == 0)


def _reordered_qkv(qkv: Any, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    """The spatial attention's `qkv` rows as `[q0 k0 v0 q1 k1 v1]` (the halves of each), as `[in, out]` + bias."""
    import torch  # noqa: PLC0415 - the module tensors (torch is imported by then)

    c = qkv.in_features
    half = c // 2
    order = [j * c + i * half + r for i in range(2) for j in range(3) for r in range(half)]
    idx = torch.tensor(order, device=qkv.weight.device)
    return swin.arr(qkv.weight[idx].t(), dtype), swin.arr(qkv.bias[idx], dtype)


class Block:
    """A `DATB`: `[B, H, W, C]` -> the same (one step)."""

    def __init__(self, m: Any, dtype: mx.Dtype, *, rg: int, index: int) -> None:
        """The arrays of the block `m` in `dtype`; `rg` / `index`: its group and place (its window split and shift)."""
        a, f = m.attn, m.ffn
        half = f.fc2.in_features
        fc1 = f.fc1
        p: dict[str, Any] = {
            "n1_w": swin.arr(m.norm1.weight, dtype),
            "n1_b": swin.arr(m.norm1.bias, dtype),
            "n2_w": swin.arr(m.norm2.weight, dtype),
            "n2_b": swin.arr(m.norm2.bias, dtype),
            "f1a_w": swin.arr(fc1.weight[:half].t(), dtype),
            "f1a_b": swin.arr(fc1.bias[:half], dtype),
            "f1b_w": swin.arr(fc1.weight[half:].t(), dtype),
            "f1b_b": swin.arr(fc1.bias[half:], dtype),
            "g_n_w": swin.arr(f.sg.norm.weight, dtype),
            "g_n_b": swin.arr(f.sg.norm.bias, dtype),
            "f2_w": swin.linear(f.fc2, dtype)[0],
            "f2_b": swin.linear(f.fc2, dtype)[1],
            **_interaction(a, dtype),
        }
        p["g_w"], p["g_b"] = _dw(f.sg.conv.weight, f.sg.conv.bias, dtype)
        eps = (float(m.norm1.eps), float(m.norm2.eps), float(f.sg.norm.eps))
        self.dtype = dtype
        self.split: tuple[int, int] | None = None
        self.shift: tuple[int, int] | None = None
        if index % 2 == 0:  # DSTB
            p["qkv_w"], p["qkv_b"] = _reordered_qkv(a.qkv, dtype)
            p["bias"] = [swin.arr(_position_bias(s), dtype) for s in a.attns]
            self.split = (int(a.split_size[0]), int(a.split_size[1]))
            if _shifted(rg, index):
                self.shift = (int(a.shift_size[0]), int(a.shift_size[1]))
            s0 = a.attns[0]
            self.f = _block(int(s0.num_heads), float(s0.scale), self.split, self.shift, eps)
        else:  # DCTB
            c = a.qkv.in_features
            heads = int(a.num_heads)
            d = c // heads
            p["qkv_w"], p["qkv_b"] = swin.linear(a.qkv, dtype)
            p["temperature"] = swin.arr(a.temperature.reshape(heads, 1).repeat_interleave(d, 0), mx.float32)
            head = np.arange(c) // d
            p["block"] = mx.array(np.where(head[:, None] == head[None], 0.0, -np.inf).astype(np.float32))
            self.f = _block(heads, 1.0, None, None, eps)
        self.p = p

    def __call__(self, x: mx.array) -> mx.array:
        """The block's output."""
        if self.shift is None:
            return self.f(x, self.p)
        assert self.split is not None  # noqa: S101 - a shifted block is spatial
        s = max(self.split)
        hp, wp = x.shape[1] + -x.shape[1] % s, x.shape[2] + -x.shape[2] % s
        masks = (
            swin.border_mask((x.shape[0], hp, wp), self.split, self.shift, self.dtype),
            swin.border_mask((x.shape[0], hp, wp), self.split[::-1], self.shift[::-1], self.dtype),
        )
        return self.f(x, self.p, *masks)


def _position_bias(m: Any) -> Any:
    """`Spatial_Attention`'s dynamic relative position bias `[heads, N, N]` (a torch tensor, meta on a cache hit)."""
    import torch  # noqa: PLC0415 - the module tensors

    n = m.H_sp * m.W_sp
    if m.rpe_biases.is_meta:  # a replay reads the shape only (the MLP on meta tensors imports `torch._dynamo`)
        return torch.empty((int(m.num_heads), n, n), device="meta")
    with torch.no_grad():
        bias = m.pos(m.rpe_biases)[m.relative_position_index.view(-1)].view(n, n, -1)
    return bias.permute(2, 0, 1)


def network(net: Any, dtype: str) -> swin.Network:
    """The MLX network of a DAT (float32 or meta skeleton) in `dtype`."""
    dt = getattr(mx, dtype)
    half = dt == mx.float16
    groups = []
    for i, layer in enumerate(net.layers):
        blocks = [Block(m, dt, rg=i, index=j) for j, m in enumerate(layer.blocks)]
        last = half and i == len(net.layers) - 1
        groups.append(swin.Group(blocks, [layer.conv], dt, scale=1 / _LAST_SCALE if last else 1.0))
    return swin.Network(net, groups, dt, norm_eps_scale=1 / _LAST_SCALE**2 if half else 1.0)
