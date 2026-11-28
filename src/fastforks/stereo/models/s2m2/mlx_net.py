"""S2M2 on Apple MLX, written in `mlx.core` operations (`MlxS2M2(net, dtype)`).

`S2M2.forward` of the vendored network as one compiled function per image size, channels last, a padded image pair
`[1, 3, H, W]` (0..255) in and the three maps out (`disparity`, `occlusion`, `confidence`, `[1, H, W]` float32):

- one batch of both images through the CNN encoder (GroupNorm statistics in float32), the feature pyramid U-Net
  (conv blocks, gated fusion, global attention at 1/32 with the relative position encoding) and the stacked
  multi-resolution transformers (1-D row cross / self attention, 2-D global attention);
- the optimal-transport matching (float32 Sinkhorn on the row-wise correlation), the 5-tap soft argmax, global
  refinement and three local refinement iterations (windowed lookups into the correlation volume, ConvGRU);
- the 1x1 convolutions are matrix products, the transposed 2 x 2 convolutions a product plus a pixel shuffle, and each
  pointwise convolution before a bilinear upsampling is applied at the coarser size (both exact);
- disparity, confidence and occlusion stay float32 from the matching on; the convolutions run in the run dtype.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import attention, mx
from fastforks.shared import mlx_stereo
from fastforks.shared.mlx_seg import resize
from fastforks.shared.mlx_stereo import Weights
from fastforks.shared.mlx_vit import gelu

if TYPE_CHECKING:
    from .net.s2m2 import S2M2

__all__ = ["LAYOUT", "SOURCES", "MlxS2M2"]

LAYOUT = "s2m2-v1"
SOURCES = (Path(__file__), Path(mlx_stereo.__file__))
_RADIUS = 4  # the correlation lookup's window (2 * 4 + 1 taps)

Lin = tuple[mx.array, "mx.array | None"]  # weight [I, O], bias
Conv = tuple[mx.array, "mx.array | None", tuple[int, int], tuple[int, int]]  # weight [O, kh, kw, I], bias, stride, pad
Op = Lin | Conv


# Weights.
def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().float().numpy()


def _bias(w: Weights, c: Any) -> mx.array | None:  # noqa: ANN401 - a torch module
    return None if c.bias is None else w.a((int(c.bias.shape[0]),), lambda: _np(c.bias))


def _conv(w: Weights, c: Any) -> Op:  # noqa: ANN401 - an nn.Conv2d
    """A convolution: a matrix product for a 1 x 1 one, else weight `[O, kh, kw, I]`."""
    o, i, kh, kw = (int(n) for n in c.weight.shape)
    if (kh, kw) == (1, 1):
        return w.a((i, o), lambda: _np(c.weight)[:, :, 0, 0].T), _bias(w, c)
    return (
        w.a((o, kh, kw, i), lambda: _np(c.weight).transpose(0, 2, 3, 1)),
        _bias(w, c),
        (int(c.stride[0]), int(c.stride[1])),
        (int(c.padding[0]), int(c.padding[1])),
    )


def _linear(w: Weights, m: Any) -> Lin:  # noqa: ANN401 - an nn.Linear
    o, i = (int(n) for n in m.weight.shape)
    return w.a((i, o), lambda: _np(m.weight).T), _bias(w, m)


def _qkv(w: Weights, a: Any) -> Lin:  # noqa: ANN401 - a SelfAttn / CrossAttn: q, k, v as one product
    o, i = (int(n) for n in a.q.weight.shape)
    weight = w.a((i, 3 * o), lambda: np.concatenate([_np(m.weight) for m in (a.q, a.k, a.v)], axis=0).T)
    bias = w.a((3 * o,), lambda: np.concatenate([np.zeros(2 * o, np.float32), _np(a.v.bias)]))
    return weight, bias


def _shuffle(w: Weights, t: Any) -> Lin:  # noqa: ANN401 - an nn.ConvTranspose2d with kernel == stride == 2
    """The transposed convolution as a product to `4 * O` channels (rows, columns, channels) and a pixel shuffle."""
    i, o = int(t.weight.shape[0]), int(t.weight.shape[1])
    return (
        w.a((i, 4 * o), lambda: _np(t.weight).transpose(0, 2, 3, 1).reshape(i, 4 * o)),
        w.a((4 * o,), lambda: np.tile(_np(t.bias), 4)),
    )


def _tconv(w: Weights, t: Any) -> Op:  # noqa: ANN401 - an nn.ConvTranspose2d, stride 1: a flipped convolution
    i, o, kh, kw = (int(n) for n in t.weight.shape)
    if (kh, kw) == (1, 1):
        return w.a((i, o), lambda: _np(t.weight)[:, :, 0, 0]), _bias(w, t)
    return (
        w.a((o, kh, kw, i), lambda: _np(t.weight)[:, :, ::-1, ::-1].transpose(1, 2, 3, 0)),
        _bias(w, t),
        (1, 1),
        (kh - 1 - int(t.padding[0]), kw - 1 - int(t.padding[1])),
    )


def _fusion(w: Weights, m: Any) -> dict[str, Op]:  # noqa: ANN401 - a FeatureFusion
    return {
        "gate1": _conv(w, m.feature_gate[0]),
        "gate2": _conv(w, m.feature_gate[2]),
        "fuse1": _conv(w, m.feature_fusion[0]),
        "fuse2": _conv(w, m.feature_fusion[2]),
    }


def _ffn(w: Weights, m: Any) -> tuple[Lin, Lin]:  # noqa: ANN401 - an FFN
    return _linear(w, m.ffn[0]), _linear(w, m.ffn[2])


def _attn(w: Weights, a: Any, *, pe: bool = False) -> dict[str, Any]:  # noqa: ANN401 - a SelfAttn / CrossAttn
    out: dict[str, Any] = {"qkv": _qkv(w, a), "proj": _linear(w, a.proj), "heads": int(a.num_heads)}
    if pe:
        out["pe"] = _linear(w, a.pe_proj)
    return out


def _global(w: Weights, b: Any) -> dict[str, Any]:  # noqa: ANN401 - a GlobalAttnBlock
    out = {"self": _attn(w, b.self_attn.attn, pe=b.self_attn.attn.use_pe), "ffn": _ffn(w, b.ffn)}
    if b.cross_attn is not None:
        out |= {"cross": _attn(w, b.cross_attn.attn), "ffn_c": _ffn(w, b.ffn_c)}
    return out


def _basic(w: Weights, b: Any) -> dict[str, Any]:  # noqa: ANN401 - a BasicAttnBlock
    return {
        "cross": _attn(w, b.cross_attn.attn),
        "ffn_c": _ffn(w, b.ffn_c),
        "self": _attn(w, b.self_attn.attn),
        "ffn": _ffn(w, b.ffn),
    }


def _convblock(w: Weights, b: Any) -> dict[str, Op]:  # noqa: ANN401 - a ConvBlock2D
    return {
        "a": _conv(w, b.convs[0]),
        "b": _conv(w, b.convs[2]),
        "c": _conv(w, b.convs_1x[0]),
        "d": _conv(w, b.convs_1x[2]),
    }


def _unet(w: Weights, u: Any) -> dict[str, Any]:  # noqa: ANN401 - a Unet
    return {
        "down": [_conv(w, getattr(u, f"down_conv{i}")[1]) for i in range(3)],
        "up": [_conv(w, getattr(u, f"up_conv{i}")[1]) for i in range(3)],
        "concat": [_fusion(w, getattr(u, f"concat_conv{i}")) for i in range(3)],
        "enc": [_convblock(w, getattr(u, f"enc{i}")) for i in range(3)],
        "dec": [_convblock(w, getattr(u, f"dec{i}")) for i in range(3)],
        "enc3": [_global(w, b) for b in u.enc3s],
        "dec3": [_global(w, b) for b in u.dec3s],
    }


def _mrt(w: Weights, m: Any) -> dict[str, Any]:  # noqa: ANN401 - an MRT
    return {
        "down": [_conv(w, getattr(m, f"down_conv{i}")[1]) for i in range(3)],
        "up": [_conv(w, getattr(m, f"up_conv{i}")[1]) for i in range(3)],
        "down_cat": [_fusion(w, getattr(m, f"down_concat{i}")) for i in (1, 2, 3)],
        "up_cat": [_fusion(w, getattr(m, f"up_concat{i}")) for i in range(3)],
        "enc": [_basic(w, getattr(m, f"enc_attn{i}")) for i in range(3)],
        "dec": [_basic(w, getattr(m, f"dec_attn{i}")) for i in range(3)],
        "enc3": [_global(w, b) for b in m.enc_attn3s],
        "dec3": [_global(w, b) for b in m.dec_attn3s],
    }


_TINY = 1e-6
_CONF_MIN = 0.2  # confidence above which a match counts as reliable


# Operations (NHWC).
_WINO_CHUNK_BYTES = 128 << 20


def _conv2d(x: mx.array, w: mx.array, stride: tuple[int, int], pad: tuple[int, int]) -> mx.array:
    """`mx.conv2d`; a stacked batch whose Winograd transforms are large runs one sample after the other.

    MLX takes a dense 3 x 3 stride-1 convolution with channels in multiples of 32 (at least 256 together) and at least
    4096 positions through Winograd, which holds the transformed input and product of the whole batch (about 4x the
    input plus output); samples one after the other bound that transient (the same result).
    """
    n, h, wd, cin = x.shape
    cout = w.shape[0]
    wino = (
        tuple(w.shape[1:3]) == (3, 3)
        and stride == (1, 1)
        and cin % 32 == 0
        and cout % 32 == 0
        and cin + cout >= 256  # noqa: PLR2004
        and n * h * wd >= 4096  # noqa: PLR2004
    )
    if n > 1 and wino and n * 4 * h * wd * (cin + cout) * x.dtype.size > 2 * _WINO_CHUNK_BYTES:
        outs: list[mx.array] = []
        for i in range(n):
            xi = x[i : i + 1]
            outs.append(mx.conv2d(mx.depends(xi, outs[-1]) if outs else xi, w, stride=stride, padding=pad))
        return mx.concatenate(outs, axis=0)
    return mx.conv2d(x, w, stride=stride, padding=pad)


def _op(x: mx.array, c: Op) -> mx.array:
    """A convolution of `_conv` (a matrix product when it is 1 x 1)."""
    if len(c) == 2:  # noqa: PLR2004
        weight, bias = c
        y = x @ weight
    else:
        weight, bias, stride, pad = c
        y = _conv2d(x, weight, stride, pad)
    return y if bias is None else y + bias


def _norm(x: mx.array) -> mx.array:
    """`LayerNorm(elementwise_affine=False)`."""
    return mx.fast.layer_norm(x, None, None, 1e-5)


def _mlp(x: mx.array, ffn: tuple[Lin, Lin]) -> mx.array:
    """The pre-norm feed-forward block with its residual."""
    return x + _op(gelu(_op(_norm(x), ffn[0])), ffn[1])


def _pool2(x: mx.array) -> mx.array:
    """`AvgPool2d(2)` of an NHWC map (summed in float32)."""
    s = sum(x[:, i::2, j::2].astype(mx.float32) for i in (0, 1) for j in (0, 1))
    return (s * 0.25).astype(x.dtype)


def _fuse(a: mx.array, b: mx.array, f: dict[str, Op], *, gate: bool = True) -> mx.array:
    """`FeatureFusion(z0=a, z1=b)`."""
    z = mx.concatenate([a, b], axis=-1)
    out = _op(gelu(_op(z, f["fuse1"])), f["fuse2"])
    if not gate:
        return out
    g = mx.clip(mx.sigmoid(_op(gelu(_op(z, f["gate1"])), f["gate2"]).astype(mx.float32)), 0.01, 0.99).astype(a.dtype)
    return out + g * a + (1 - g) * b


def _convblock_apply(x: mx.array, b: dict[str, Op]) -> mx.array:
    return _op(gelu(_op(x, b["a"])), b["b"]) + _op(mx.maximum(_op(x, b["c"]), 0), b["d"])


def _tokens(qkv: mx.array, heads: int) -> tuple[mx.array, mx.array, mx.array]:
    """Packed `[B, L, 3 D]` -> q, k, v `[B, heads, L, D / heads]`."""
    b, n, d3 = qkv.shape
    q, k, v = qkv.reshape(b, n, 3, heads, d3 // (3 * heads)).transpose(2, 0, 3, 1, 4)
    return q, k, v


def _out(o: mx.array, proj: Lin) -> mx.array:
    b, h, n, d = o.shape
    return _op(o.transpose(0, 2, 1, 3).reshape(b, n, h * d), proj)


def _cross(z: mx.array, a: dict[str, Any], *, rows: bool) -> mx.array:
    """Pre-norm symmetric cross attention between the left and the right image (`rows`: along each image row)."""
    two, h, w, c = z.shape
    qkv = _op(_norm(z), a["qkv"])
    d3 = qkv.shape[-1]
    qkv = qkv.reshape(2 * h, w, d3) if rows else qkv.reshape(2, h * w, d3)
    q, k, v = _tokens(qkv, a["heads"])
    half = q.shape[0] // 2
    k, v = (mx.concatenate([t[half:], t[:half]]) for t in (k, v))  # each image attends to the other one
    o = _out(attention(q, k, v, scale=q.shape[-1] ** -0.5), a["proj"])
    return z + o.reshape(two, h, w, c)


def _selfattn(
    z: mx.array, a: dict[str, Any], *, rows: bool, pe: tuple[np.ndarray, np.ndarray] | None = None
) -> mx.array:
    """Pre-norm self attention (`rows`: along each image row), with the relative position encoding if `pe`."""
    two, h, w, c = z.shape
    qkv = _op(_norm(z), a["qkv"])
    d3 = qkv.shape[-1]
    qkv = qkv.reshape(two * h, w, d3) if rows else qkv.reshape(two, h * w, d3)
    q, k, v = _tokens(qkv, a["heads"])
    scale = q.shape[-1] ** -0.5
    if pe is None:
        o = attention(q, k, v, scale=scale)
    else:
        # sum_j attn[i, j] pe[i, j]: the encoding is a table per axis, so the attention's marginals over the key
        # columns / rows are enough (the upstream form builds the [N, N, 32] tensor)
        tx, ty = (mx.array(t).astype(z.dtype) for t in pe)
        attn = mx.softmax((scale * q) @ k.transpose(0, 1, 3, 2), axis=-1, precise=True)
        o = attn @ v
        heads = q.shape[1]
        a5 = attn.reshape(two, heads, h, w, h, w)  # query (y, x), key (y', x')
        pex = mx.einsum("bnyxj,xjc->bnyxc", a5.sum(4), tx)
        pey = mx.einsum("bnyxu,yuc->bnyxc", a5.sum(5), ty)
        pos = mx.concatenate([pex, pey], axis=-1).reshape(two, heads, h * w, -1)
        o = o + _op(pos, a["pe"])
    return z + _out(o, a["proj"]).reshape(two, h, w, c)


def _global_apply(z: mx.array, b: dict[str, Any], pe: tuple[np.ndarray, np.ndarray] | None = None) -> mx.array:
    if "cross" in b:
        z = _mlp(_cross(z, b["cross"], rows=False), b["ffn_c"])
    return _mlp(_selfattn(z, b["self"], rows=False, pe=pe), b["ffn"])


def _basic_apply(z: mx.array, b: dict[str, Any]) -> mx.array:
    z = _mlp(_cross(z, b["cross"], rows=True), b["ffn_c"])
    return _mlp(_selfattn(z, b["self"], rows=True), b["ffn"])


def _pe_tables(h: int, w: int, dim: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """The relative position encoding as one table per axis (`utils.get_pe_tables`): `[w, w, 16]` and `[h, h, 16]`."""
    sig = 5 / dim
    dim_t = np.linspace(-1, 1, dim // 2, dtype=np.float32)

    def table(n: int) -> np.ndarray:
        i = np.arange(n, dtype=np.float32)
        k = (i[:, None] - i[None, :]) + (n - 1)
        pos = np.tanh(-3 + k * np.float32(3.0 / n)).astype(np.float32)
        x = (dim_t - pos[..., None]) / np.float32(sig)
        safe = np.where(np.abs(x) < _TINY, np.float32(1.0), x)
        v = np.where(
            np.abs(x) < _TINY, np.float32(1.0), np.sin(np.float32(3.1415) * safe) / (np.float32(3.1415) * safe)
        )
        v = v.astype(np.float32)
        return (0.5 * v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)).astype(np.float32)

    return table(w), table(h)


def _unet_apply(u: dict[str, Any], x: mx.array, *, pe: bool) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """`Unet.forward`: the four levels (1/1, 1/2, 1/4, 1/8 of `x`)."""
    tables = _pe_tables(x.shape[1] // 8, x.shape[2] // 8) if pe else None
    z0 = _convblock_apply(x, u["enc"][0])
    z1 = _convblock_apply(_op(_pool2(z0), u["down"][0]), u["enc"][1])
    z2 = _convblock_apply(_op(_pool2(z1), u["down"][1]), u["enc"][2])
    z3 = _op(_pool2(z2), u["down"][2])
    for b in u["enc3"]:
        z3 = _global_apply(z3, b, tables)
    for b in u["dec3"]:
        z3 = _global_apply(z3, b)
    z2n = _convblock_apply(_fuse(z2, resize(_op(z3, u["up"][2]), z2.shape[1:3]), u["concat"][2]), u["dec"][2])
    z1n = _convblock_apply(_fuse(z1, resize(_op(z2n, u["up"][1]), z1.shape[1:3]), u["concat"][1]), u["dec"][1])
    z0n = _convblock_apply(_fuse(z0, resize(_op(z1n, u["up"][0]), z0.shape[1:3]), u["concat"][0]), u["dec"][0])
    return z0n, z1n, z2n, z3


def _mrt_apply(m: dict[str, Any], z0: mx.array, z1: mx.array, z2: mx.array, z3: mx.array) -> tuple[mx.array, ...]:
    """`MRT.forward`."""
    z0 = _basic_apply(z0, m["enc"][0])
    z1 = _fuse(z1, _op(_pool2(z0), m["down"][0]), m["down_cat"][0])
    z1 = _basic_apply(z1, m["enc"][1])
    z2 = _fuse(z2, _op(_pool2(z1), m["down"][1]), m["down_cat"][1])
    z2 = _basic_apply(z2, m["enc"][2])
    z3 = _fuse(z3, _op(_pool2(z2), m["down"][2]), m["down_cat"][2])
    for b in m["enc3"]:
        z3 = _global_apply(z3, b)
    for b in m["dec3"]:
        z3 = _global_apply(z3, b)
    z2 = _basic_apply(_fuse(z2, resize(_op(z3, m["up"][2]), z2.shape[1:3]), m["up_cat"][2]), m["dec"][2])
    z1 = _basic_apply(_fuse(z1, resize(_op(z2, m["up"][1]), z1.shape[1:3]), m["up_cat"][1]), m["dec"][1])
    z0 = _basic_apply(_fuse(z0, resize(_op(z1, m["up"][0]), z0.shape[1:3]), m["up_cat"][0]), m["dec"][0])
    return z0, z1, z2, z3


def _group_norm(x: mx.array, groups: int, weight: mx.array, bias: mx.array) -> mx.array:
    """`GroupNorm(groups, C)` of an NHWC map, float32 statistics."""
    b, h, w, c = x.shape
    g = c // groups
    n = h * w * g
    strips = 4 if h % 4 == 0 else 1
    s1 = s2 = 0.0
    for part in mx.split(x.reshape(b, strips, h * w // strips, groups, g), strips, axis=1):  # fp32 cast per strip
        p = part.astype(mx.float32)
        s1 = s1 + mx.sum(p, axis=(1, 2, 4), keepdims=True)
        s2 = s2 + mx.sum(mx.square(p), axis=(1, 2, 4), keepdims=True)
    mean = s1 / n
    scale = mx.rsqrt(s2 / n - mx.square(mean) + 1e-5)
    y = (x.reshape(b, h * w, groups, g).astype(mx.float32) - mean.reshape(b, 1, groups, 1)) * scale.reshape(
        b, 1, groups, 1
    )
    return (y.reshape(b, h, w, c) * weight + bias).astype(x.dtype)


def _logit(x: mx.array, eps: float) -> mx.array:
    x = mx.clip(x, eps, 1 - eps)
    return mx.log(x / (1 - x))


def _shuffled(x: mx.array, p: Lin) -> mx.array:
    """A transposed 2 x 2 stride-2 convolution of `_shuffle`."""
    b, h, w, _ = x.shape
    y = _op(x, p)
    return y.reshape(b, h, w, 2, 2, -1).transpose(0, 1, 3, 2, 4, 5).reshape(b, 2 * h, 2 * w, -1)


def _taps3(x: mx.array) -> mx.array:
    """`custom_unfold(x, 3, 1)` of `[B, H, W, M]` (replicate padding): `[B, H, W, M, 9]`."""
    _, h, w, _ = x.shape
    p = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], mode="edge")
    return mx.stack([p[:, i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=-1)


def _filter(maps: mx.array, logits: mx.array, factor: int) -> mx.array:
    """Softmax-weighted 3 x 3 neighbourhoods of `maps` `[B, h, w, M]` (nearest-upsampled `factor` times) at `logits`."""
    b, h, w, m = maps.shape
    weights = mx.softmax(logits.astype(mx.float32), axis=-1).reshape(b, h, factor, w, factor, 1, 9)
    out = (_taps3(maps).reshape(b, h, 1, w, 1, m, 9) * weights).sum(-1)
    return out.reshape(b, h * factor, w * factor, m)


class MlxS2M2:
    """The network `net` (eval, float32 weights) in `dtype` (the name): one compiled function per image size."""

    def __init__(self, net: S2M2, dtype: str) -> None:
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        w = Weights(self.dtype)
        self.positive = bool(net.use_positivity)
        self.iters = int(net.refine_iter)
        enc: Any = net.cnn_backbone
        self.enc: dict[str, Any] = {
            "c0": [_conv(w, enc.conv0[0]), _conv(w, enc.conv0[2])],
            "c1": [_conv(w, enc.conv1_down[0]), _conv(w, enc.conv1_down[2])],
            "c2": [_conv(w, enc.conv2[0]), _conv(w, enc.conv2[2])],
            "c3": _conv(w, enc.conv2_down[0]),
        }
        self.gn = (
            w.a((int(enc.norm1.weight.shape[0]),), lambda: _np(enc.norm1.weight)),
            w.a((int(enc.norm1.bias.shape[0]),), lambda: _np(enc.norm1.bias)),
        )
        self.groups = int(enc.norm1.num_groups)
        self.pyramid: dict[str, Any] = _unet(w, net.feat_pyramid)
        self.transformer = [_mrt(w, m) for m in net.transformer.uformer_list]
        ln = net.disp_init.layer_norm
        self.ln = (
            w.a((int(ln.weight.shape[0]),), lambda: _np(ln.weight)),
            w.a((int(ln.bias.shape[0]),), lambda: _np(ln.bias)),
            float(ln.eps),
        )
        self.ot_iter = int(net.disp_init.ot_iter)
        g: Any = net.global_refiner
        self.glob: dict[str, Any] = {
            "init": [_conv(w, g.init_feat[0]), _conv(w, g.init_feat[2])],
            "unet": _unet(w, g.refine_unet),
            "out": _conv(w, g.out_feat[0]),
        }
        self.fusion = _fusion(w, net.feat_fusion_layer)
        self.ctx: list[Op] = [_conv(w, net.ctx_feat[0]), _conv(w, net.ctx_feat[2])]
        r: Any = net.refiner
        self.ref: dict[str, Any] = {
            "disp": [_conv(w, r.disp_feat[0]), _conv(w, r.disp_feat[2])],
            "corr1": [_conv(w, r.corr_feat1[0]), _conv(w, r.corr_feat1[2])],
            "corr2": [_conv(w, r.corr_feat2[0]), _conv(w, r.corr_feat2[2])],
            "conf": [_conv(w, r.conf_occ_feat[0]), _conv(w, r.conf_occ_feat[2])],
            "cat": [_conv(w, r.disp_corr_ctx_cat[0]), _conv(w, r.disp_corr_ctx_cat[2])],
            "unet": _unet(w, r.refine_unet),
            "upd": [_conv(w, r.disp_update[0]), _conv(w, r.disp_update[2])],
            "cupd": [_conv(w, r.conf_occ_update[0]), _conv(w, r.conf_occ_update[2])],
            "gru": self._gru(w, r.gru),
        }
        m4: Any = net.upsample_mask_4x_refine
        self.up4: dict[str, Any] = {
            "x": _shuffle(w, m4.conv_x),
            "y": _conv(w, m4.conv_y),
            "cat": _conv(w, m4.conv_concat[0]),
            "out": _shuffle(w, m4.conv_concat[2]),
        }
        m1: Any = net.upsample_mask_1x
        self.up1: dict[str, Any] = {
            "disp": _tconv(w, m1.conv_disp[0]),
            "rgb": _tconv(w, m1.conv_rgb[0]),
            "ctx": _shuffle(w, m1.conv_ctx),
            "cat": _conv(w, m1.conv_concat[0]),
            "out": _tconv(w, m1.conv_concat[2]),
        }
        mx.eval(self._arrays())
        # one compiled function per stage, evaluated in turn: fused element-wise chains, but the peak is that of a stage
        self._stages = {
            n: mx.compile(getattr(self, f"_{n}"))
            for n in ("encoder", "pyramid", "transformer", "middle", "refine_step", "tail")
        }
        self._stages["front"] = self._front

    @staticmethod
    def _gru(w: Weights, g: Any) -> dict[str, Any]:  # noqa: ANN401 - a ConvGRU
        out: dict[str, Any] = {}
        for step in "12":
            z, r, q = (getattr(g, f"conv{n}{step}") for n in "zrq")
            o, i, kh, kw = (int(n) for n in z.weight.shape)
            out[f"zr{step}"] = (
                w.a(
                    (2 * o, kh, kw, i),
                    lambda z=z, r=r: np.concatenate([_np(z.weight), _np(r.weight)]).transpose(0, 2, 3, 1),
                ),
                w.a((2 * o,), lambda z=z, r=r: np.concatenate([_np(z.bias), _np(r.bias)])),
                (1, 1),
                (int(z.padding[0]), int(z.padding[1])),
            )
            out[f"q{step}"] = _conv(w, q)
        return out

    def _arrays(self) -> list[mx.array]:
        found: list[mx.array] = []

        def walk(v: object) -> None:
            if isinstance(v, mx.array):
                found.append(v)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)
            elif isinstance(v, dict):
                walk(list(v.values()))

        walk(list(vars(self).values()))
        return found

    # The stages.
    def _encoder(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """`CNNEncoder`: the 1/4 and the 1/2 resolution features of the stacked pair."""
        e = self.enc
        x = _op(gelu(_op(x, e["c0"][0])), e["c0"][1])
        x = _op(gelu(_op(x, e["c1"][0])), e["c1"][1])
        x = _group_norm(x, self.groups, *self.gn)
        x = _op(gelu(_op(x, e["c2"][0])), e["c2"][1]) + x
        return _op(x, e["c3"]), x

    def _disp_init(self, feat: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """`DispInit`: disparity, confidence, occlusion `[1, h, w, 1]` (float32) and the correlation `[h, w, w]`."""
        w = feat.shape[2]
        weight, bias, eps = self.ln
        f = mx.fast.layer_norm(feat, weight, bias, eps)
        cv = mx.matmul(f[0], f[1].transpose(0, 2, 1))  # [h, w(i), w(j)] of the left / right features
        f32 = mx.float32
        s = cv.astype(f32)
        if self.positive:
            s = mx.where(mx.triu(mx.ones((w, w), dtype=mx.bool_), k=1), -1e4, s)
        s = mx.pad(s, [(0, 0), (0, 1), (0, 1)])[None]
        marginal = np.concatenate([np.ones(w), [float(w)]]) / (2 * w)
        log_m = mx.array(np.log(marginal).reshape(1, 1, w + 1).astype(np.float32))
        lse = mx.logsumexp
        v = log_m - lse(s, axis=2)
        u = log_m - lse(s + v[:, :, None, :], axis=3)
        for _ in range(self.ot_iter - 1):
            v = log_m - lse(s + u[..., None], axis=2)
            u = log_m - lse(s + v[:, :, None, :], axis=3)
        prob = mx.exp(s[:, :, :-1, :-1] + u[:, :, :-1, None] + v[:, :, None, :-1] + float(np.log(2 * w)))
        if self.positive:
            prob = mx.where(mx.triu(mx.ones((w, w), dtype=mx.bool_), k=1), 0.0, prob)
        ind = mx.argmax(prob, axis=3, keepdims=True)  # [1, h, w, 1]
        padded = mx.pad(prob, [(0, 0), (0, 0), (0, 0), (2, 2)])
        conf = mx.zeros(ind.shape, f32)
        corr = mx.zeros(ind.shape, f32)
        for k in range(5):
            weight_k = mx.take_along_axis(padded, ind + k, axis=3)
            conf = conf + weight_k
            corr = corr + weight_k * (ind + k - 2).astype(f32)
        corr = (corr + 1e-4) / (conf + 1e-4)
        disp = mx.arange(w, dtype=f32).reshape(1, 1, w, 1) - corr
        occ = prob.sum(axis=3)[..., None]
        return disp, conf, occ, cv

    def _global_refine(self, ctx: mx.array, disp: mx.array, conf: mx.array) -> mx.array:
        g = self.glob
        mask = (conf > _CONF_MIN).astype(mx.float32)
        x = mx.concatenate([disp / 100.0 * mask, _logit(mask * conf, 0.1)], axis=-1).astype(self.dtype)
        x = mx.concatenate([x, ctx], axis=-1)
        feat = _op(gelu(_op(x, g["init"][0])), g["init"][1])
        out = _op(_unet_apply(g["unet"], feat, pe=False)[0], g["out"]).astype(mx.float32) * 100.0
        return mask * disp + (1 - mask) * out

    @staticmethod
    def _lookup(cv: mx.array, x0: mx.array, y: mx.array, width: int) -> mx.array:
        """Bilinear samples of `cv` `[h, w, W]` at columns `x * W / (W - 1) - 0.5` (zeros outside) of row `y`.

        `x0` `[1, h, w, 9]` float32: the lookup's x coordinates; `y` unused (row `i` of the pixel itself).
        """
        del y
        p = x0[0] * (width / (width - 1.0)) - 0.5
        f = mx.floor(p)
        a = p - f
        i0, i1 = f.astype(mx.int32), f.astype(mx.int32) + 1
        out = mx.zeros(p.shape, mx.float32)
        for idx, wgt in ((i0, 1.0 - a), (i1, a)):
            ok = (idx >= 0) & (idx <= width - 1)
            v = mx.take_along_axis(cv, mx.clip(idx, 0, width - 1), axis=2).astype(mx.float32)
            out = out + mx.where(ok, v, 0.0) * wgt
        return out[None]

    def _correlate(self, cv: mx.array, cv2: mx.array, disp: mx.array) -> tuple[mx.array, mx.array]:
        """`CostVolume.__call__`: the full and the 2x-pooled correlation sampled around the current disparity."""
        w = disp.shape[2]
        dx = mx.arange(-_RADIUS, _RADIUS + 1, dtype=mx.float32)
        coords = mx.arange(w, dtype=mx.float32).reshape(1, 1, w, 1)
        c1 = self._lookup(cv, coords - disp + dx, None, w)
        c2 = self._lookup(cv2, coords / 2 - disp / 2 + dx, None, w // 2)
        return c1, c2

    def _gru_apply(self, hidden: mx.array, x: mx.array) -> mx.array:
        g = self.ref["gru"]
        for step in "12":
            hx = mx.concatenate([hidden.astype(self.dtype), x], axis=-1)
            zr = mx.sigmoid(_op(hx, g[f"zr{step}"]).astype(mx.float32))
            z, r = mx.split(zr, 2, axis=-1)
            q = mx.tanh(
                _op(mx.concatenate([(r * hidden).astype(self.dtype), x], axis=-1), g[f"q{step}"]).astype(mx.float32)
            )
            hidden = (1 - z) * hidden + z * q
        return hidden

    def _refine(  # noqa: PLR0913, PLR0917 - a compiled stage takes its tensors one by one
        self,
        hidden: mx.array,
        ctx0: mx.array,
        disp: mx.array,
        conf: mx.array,
        occ: mx.array,
        cvs: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        r = self.ref
        dt = self.dtype
        conf_logit, occ_logit = _logit(conf, 1e-2), _logit(occ, 1e-2)
        corr1, corr2 = self._correlate(*cvs, disp)
        f1 = _op(gelu(_op((corr1 / 16).astype(dt), r["corr1"][0])), r["corr1"][1])
        f2 = _op(gelu(_op((corr2 / 16).astype(dt), r["corr2"][0])), r["corr2"][1])
        fd = _op(gelu(_op((disp / 100.0).astype(dt), r["disp"][0])), r["disp"][1])
        fc = _op(gelu(_op(mx.concatenate([conf_logit, occ_logit], axis=-1).astype(dt), r["conf"][0])), r["conf"][1])
        x = mx.concatenate([fd, f1, f2, ctx0, fc], axis=-1)
        x = _op(gelu(_op(x, r["cat"][0])), r["cat"][1])
        x = _unet_apply(r["unet"], x, pe=False)[0]
        hidden = self._gru_apply(hidden, x)
        h = hidden.astype(dt)
        d_up = _op(gelu(_op(h, r["upd"][0])), r["upd"][1]).astype(mx.float32)
        c_up, o_up = mx.split(_op(gelu(_op(h, r["cupd"][0])), r["cupd"][1]).astype(mx.float32), 2, axis=-1)
        return hidden, disp + d_up, mx.sigmoid(c_up + conf_logit), mx.sigmoid(o_up + occ_logit)

    def _front(self, left: mx.array, right: mx.array) -> tuple[mx.array, ...]:
        """The two images one after the other (half the transient of the stacked pair)."""
        feat4: list[mx.array] = []
        feat2 = img0 = None
        for i, im in enumerate((left, right)):
            both = (im.transpose(0, 2, 3, 1) / 255.0 - 0.5) * 2.0
            f4, f2 = self._stages["encoder"](both.astype(self.dtype))
            if i == 0:
                feat2, img0 = f2, both.astype(self.dtype)
                mx.eval(f4, f2, img0)
            else:
                mx.eval(f4)
            feat4.append(f4)
        return mx.concatenate(feat4, axis=0), feat2, img0

    def _pyramid(self, feat4: mx.array) -> tuple[mx.array, ...]:
        return _unet_apply(self.pyramid, feat4, pe=True)

    def _transformer(self, i: int, *z: mx.array) -> tuple[mx.array, ...]:
        return _mrt_apply(self.transformer[i], *z)

    def _middle(self, tr: mx.array, py0: mx.array) -> tuple[mx.array, ...]:
        """Matching and global refinement: disparity, confidence, occlusion, hidden state, context, volumes."""
        disp, conf, occ, cv = self._disp_init(tr)
        tr0 = tr[:1]
        disp = self._global_refine(tr0, disp, conf)
        if self.positive:
            disp = mx.maximum(disp, 0.0)
        ctx0 = _op(gelu(_op(_fuse(tr0, py0[:1], self.fusion), self.ctx[0])), self.ctx[1])
        h, w = disp.shape[1:3]
        cv2 = cv.reshape(h, w, w // 2, 2).astype(mx.float32).mean(axis=-1)
        return disp, conf, occ, mx.tanh(ctx0.astype(mx.float32)), ctx0, cv, cv2

    def _refine_step(  # noqa: PLR0913, PLR0917 - a compiled stage takes its tensors one by one
        self,
        hidden: mx.array,
        ctx0: mx.array,
        disp: mx.array,
        conf: mx.array,
        occ: mx.array,
        cvs: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, ...]:
        hidden, disp, conf, occ = self._refine(hidden, ctx0, disp, conf, occ, cvs)
        if self.positive:
            disp = mx.maximum(disp, 0.0)
        cols = mx.arange(disp.shape[2], dtype=mx.float32).reshape(1, 1, -1, 1)
        return hidden, disp, conf, occ * ((cols - disp) >= 0).astype(mx.float32)

    def _tail(  # noqa: PLR0913, PLR0917 - a compiled stage takes its tensors one by one
        self, hidden: mx.array, disp: mx.array, conf: mx.array, occ: mx.array, ctx2x: mx.array, img0: mx.array
    ) -> tuple[mx.array, ...]:
        """The 4x upsampling, then the edge-guided 1x sharpening."""
        dt = self.dtype
        u = self.up4
        x = mx.concatenate([_shuffled(hidden.astype(dt), u["x"]), _op(ctx2x, u["y"])], axis=-1)
        mask = _shuffled(mx.maximum(_op(x, u["cat"]), 0), u["out"])
        maps = _filter(mx.concatenate([disp * 4.0, occ, conf], axis=-1), mask, 4)  # disparity, occlusion, confidence
        u1 = self.up1
        fd = mx.maximum(_op(maps[..., :1].astype(dt), u1["disp"]), 0)
        fr = mx.maximum(_op(img0, u1["rgb"]), 0)
        fx = _shuffled(ctx2x, u1["ctx"])
        filt = _op(mx.maximum(_op(mx.concatenate([fd, fr, fx], axis=-1), u1["cat"]), 0), u1["out"])
        maps = _filter(maps, filt, 1)
        return maps[..., 0], maps[..., 1], maps[..., 2]

    def __call__(self, left: mx.array, right: mx.array) -> dict[str, mx.array]:
        """Disparity, occlusion and confidence `[1, H, W]` (float32, lazy) of a padded float32 pair `[1, 3, H, W]`."""
        st = self._stages
        feat4, ctx2x, img0 = st["front"](left, right)
        mx.eval(feat4, ctx2x, img0)
        z = st["pyramid"](feat4)
        mx.eval(*z)
        del feat4
        py0 = z[0]
        for i in range(len(self.transformer)):
            z = st["transformer"](i, *z)
            mx.eval(*z)
        disp, conf, occ, hidden, ctx0, cv, cv2 = st["middle"](z[0], py0)
        mx.eval(disp, conf, occ, hidden, ctx0, cv, cv2)
        del z, py0
        for _ in range(self.iters):
            hidden, disp, conf, occ = st["refine_step"](hidden, ctx0, disp, conf, occ, (cv, cv2))
            mx.eval(hidden, disp, conf, occ)
        out, occ, conf = st["tail"](hidden, disp, conf, occ, ctx2x, img0)
        return {"disparity": out, "occlusion": occ, "confidence": conf}
