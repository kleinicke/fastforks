"""Depth Pro on Apple MLX (`device="mlx"`): the three ViTs, the decoder and the heads, hand-written.

The maths of `model.forward` (the reorganised graph) for the normalised `[1, 3, 1536, 1536]` image, as `mx`
operations in channels-last layout: nothing is traced, so a first call is the Metal kernels' build and no
`torch.export` (which was most of a 31 s first call). The pieces:

- the pyramid: bilinear halving without `align_corners` is the mean of each 2x2 block, quartering that of the middle
  2x2 of each 4x4 block.
- the encoders: the shared ViT blocks (`fastforks.shared.mlx_vit`); the patch encoder's 35 windows are one batch,
  gathered from the per-level patch embeddings and merged by `net.tables`'s row indices.
- the projections: every 1x1 convolution and the 2x2 / stride 2 transposed convolution after it as one matmul and a
  depth-to-space; the low-resolution fusion (two such chains and a 1x1 convolution of their concatenation) as the sum
  of two matmuls.
- the decoder and the head as `model.compose_deconv_1x1` and `model.compose_head` prepare them.

It runs in stages, each queued before the previous one is waited for: the encoders, the decoder's levels 4-2 (both
compiled), then levels 1 and 0 and the head map by map, so that the 768x768 maps (300 MB each in float16) are freed
as they go.

Prepared once, at conversion (stored float32). float16: weights and activations fp16 (LayerNorm statistics and the
attention's softmax in fp32 inside MLX's kernels); the head's last 1x1 convolution and the field-of-view head run in
fp32.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastforks._core.mlx import mx, to_mx, tuning
from fastforks.shared.depth_anything_v2.mlx_net import _conv
from fastforks.shared.mlx_vit import blocks, dinov2_params, patches

from .model import compose_deconv_1x1, compose_head
from .net import GRID, HOOKS, PATCH, SIZE, tables

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["LAYOUT", "Net", "prepare"]

LAYOUT = "v1"  # version of `prepare`'s parameter layout (for the conversion cache's file name)
EPS = 1e-6  # LayerNorm (DINOv2's)
HEADS = 16  # ViT-L
DEPTH = 24
STRIPS = 6  # bands of rows a 768x768 3x3 convolution runs in (see `finish`)
HEAD_WIDTH = 160  # channels between the head's two 3x3 convolutions (see `prepare`)
VITS = ("patch", "image", "fov")  # the patch encoder, the image encoder and the field-of-view encoder
FP32 = ("head.out.", "fovnet.")  # parameters that stay float32 in a float16 run
_CAST_CACHE = 64 << 20  # bytes of cast float32 weights after which the buffer cache is emptied


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().double().numpy()


def _rows(w: np.ndarray) -> np.ndarray:
    """Transposed convolution `[Cin, Cout, 2, 2]` -> the matmul `[Cin, 4 * Cout]` to (a, b, cout)."""
    return w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1)


def _project(first: np.ndarray, deconv: np.ndarray, after: np.ndarray | None = None) -> np.ndarray:
    """`after(deconv(first(x)))` of 1x1 `first` `[C, K]`, `deconv` `[C, O, 2, 2]`, 1x1 `after` `[P, O]`: `[K, 4 P]`."""
    out = []
    for a in range(2):
        for b in range(2):
            w = first.T @ deconv[:, :, a, b]  # [K, O]
            out.append(w if after is None else w @ after.T)
    return np.concatenate(out, axis=1)


def prepare(net: Any) -> dict[str, mx.array]:  # noqa: ANN401, C901 - `net.DepthProNet`; a line per layer
    """The loaded network -> the float32 arrays `Net` uses."""
    out: dict[str, mx.array] = {}

    def put(k: str, v: np.ndarray) -> None:
        out[k] = mx.array(np.ascontiguousarray(v, dtype=np.float32))

    def conv(k: str, m: Any) -> None:  # noqa: ANN401 - nn.Conv2d
        put(k + ".weight", _np(m.weight).transpose(0, 2, 3, 1))
        if m.bias is not None:
            put(k + ".bias", _np(m.bias))

    enc, dec, fov = net.encoder, net.decoder, net.fov
    for name, vit in zip(VITS, net.vits(), strict=True):
        p = dinov2_params(dict(vit.state_dict()), PATCH)
        table = np.concatenate(
            [np.array(p.pop("pos.cls")), np.array(p.pop("pos.grid")) + np.array(p.pop("patch.bias"))]
        )
        put(f"vit.{name}.table", table)
        out.update({f"vit.{name}.{k}": v for k, v in p.items()})
    for k, seq in (
        ("lat0", enc.upsample_latent0),
        ("lat1", enc.upsample_latent1),
        ("up0", enc.upsample0),
        ("up1", enc.upsample1),
    ):
        put(f"enc.{k}.0", _project(_np(seq[0].weight)[:, :, 0, 0], _np(seq[1].weight)))
        for i, m in enumerate(seq[2:], 1):
            put(f"enc.{k}.{i}", _rows(_np(m.weight)))
    fuse = _np(enc.fuse_lowres.weight)[:, :, 0, 0]  # [C, 2 C]: the patch encoder's half, then the image encoder's
    half = fuse.shape[0]
    up2 = enc.upsample2
    put("enc.low.patch", _project(_np(up2[0].weight)[:, :, 0, 0], _np(up2[1].weight), fuse[:, :half]))
    low = _np(enc.upsample_lowres.weight)
    put("enc.low.image", _project(np.eye(low.shape[0]), low, fuse[:, half:]))
    put("enc.low.bias", np.tile(fuse[:, half:] @ _np(enc.upsample_lowres.bias) + _np(enc.fuse_lowres.bias), 4))
    for i, fu in enumerate(dec.fusions):
        if i:
            conv(f"dec.conv{i}", dec.convs[i])
            w, b = compose_deconv_1x1(fu.deconv, fu.out_conv)
            put(f"dec.fuse{i}.out.weight", _rows(_np(w)))
            put(f"dec.fuse{i}.out.bias", np.tile(_np(b), 4))
        else:
            put(f"dec.fuse{i}.out.weight", _np(fu.out_conv.weight)[:, :, 0, 0].T)
            put(f"dec.fuse{i}.out.bias", _np(fu.out_conv.bias))
        for u, rb in ((1, fu.resnet1), (2, fu.resnet2)):
            if u == 2 or i < len(dec.fusions) - 1:  # noqa: PLR2004 - the lowest level has no skip to add
                conv(f"dec.fuse{i}.unit{u}.conv1", rb.residual[1])
                conv(f"dec.fuse{i}.unit{u}.conv2", rb.residual[3])
    w0, b0, w1, b1, w2, b2 = (_np(t) for t in compose_head(net.head))
    # 136 channels between the head's 3x3 convolutions (128, the constant one and 7 of zeros): zeros up to 160 put
    # both into MLX's Winograd (768x768 float16: 256 -> 136 199 ms, 256 -> 160 73 ms; 136 -> 128 54, 160 -> 128 44)
    pad = HEAD_WIDTH - w0.shape[0]
    put("head.conv0.weight", np.pad(w0.transpose(0, 2, 3, 1), [(0, pad), (0, 0), (0, 0), (0, 0)]))
    put("head.conv0.bias", np.pad(b0, (0, pad)))
    put("head.conv1.weight", np.pad(w1.transpose(0, 2, 3, 1), [(0, 0), (0, 0), (0, 0), (0, pad)]))
    put("head.conv1.bias", b1)
    put("head.out.weight", w2.T)
    put("head.out.bias", b2)
    put("fovnet.linear.weight", _np(fov.encoder[1].weight).T)
    put("fovnet.linear.bias", _np(fov.encoder[1].bias))
    conv("fovnet.down", fov.downsample[0])
    for i, j in enumerate((0, 2, 4)):
        conv(f"fovnet.head{i}", fov.head[j])
    return out


def _shuffle(y: mx.array) -> mx.array:
    """`[1, h, w, 4 C]` (a, b, c) -> `[1, 2 h, 2 w, C]`."""
    _, h, w, c = y.shape
    return y.reshape(1, h, w, 2, 2, c // 4).transpose(0, 1, 3, 2, 4, 5).reshape(1, 2 * h, 2 * w, c // 4)


def _embedded(p: dict[str, mx.array], cols: mx.array) -> mx.array:
    """Patch embeddings without bias `[B, 576, C]` -> the ViT's input tokens `[B, 577, C]`."""
    table = p["table"]
    cls = mx.broadcast_to(table[:1], (cols.shape[0], 1, cols.shape[2]))
    return mx.concatenate([cls, cols + table[1:]], axis=1)


def _vit(p: dict[str, mx.array], cols: mx.array) -> mx.array:
    """Patch rows `[1, 576, 768]` of the 384x384 image -> normalised patch tokens `[1, 24, 24, C]`."""
    t, _ = blocks(p, DEPTH, HEADS, EPS, _embedded(p, cols @ p["patch.weight"]), ())
    t = mx.fast.layer_norm(t[:, 1:], p["norm.weight"], p["norm.bias"], EPS)
    return t.reshape(1, GRID, GRID, -1)


def encode(p: dict[str, dict[str, mx.array]], index: tuple[mx.array, ...], x: mx.array, *, with_fov: bool) -> tuple:
    """Normalised `[1, 3, 1536, 1536]` -> (latent0 tokens `[1, 96, 96, C]`, encodings 1-4, FOV tokens if `with_fov`).

    Encodings: `[1, 384, 384, 256]`, `[1, 192, 192, 512]`, `[1, 96, 96, 1024]`, `[1, 48, 48, 1024]`; FOV tokens
    `[1, 24, 24, 128]` float32.
    """
    tok, m0, m1, m2 = index
    pp, q = p["patch"], p["net"]
    dt = pp["table"].dtype
    x = x.astype(mx.float32)
    x1 = mx.mean(x.reshape(1, 3, SIZE // 2, 2, SIZE // 2, 2), axis=(3, 5))
    x2 = mx.mean(x.reshape(1, 3, SIZE // 4, 4, SIZE // 4, 4)[:, :, :, 1:3, :, 1:3], axis=(3, 5))
    cols = [patches(v.astype(dt), PATCH) for v in (x, x1, x2)]
    e = mx.concatenate([c[0] @ pp["patch.weight"] for c in cols])  # [96^2 + 48^2 + 24^2, C]
    t, lat = blocks(pp, DEPTH, HEADS, EPS, _embedded(pp, e[tok]), tuple(i + 1 for i in HOOKS))
    c = t.shape[-1]
    rows = t.reshape(-1, c)
    f0, f1, f2 = (mx.fast.layer_norm(rows[m][None], pp["norm.weight"], pp["norm.bias"], EPS) for m in (m0, m1, m2))
    lat0, lat1 = (h.reshape(-1, c)[m0][None] for h in lat)
    g = _vit(p["image"], cols[2])
    low = mx.addmm(q["enc.low.bias"], f2, q["enc.low.patch"]) + g @ q["enc.low.image"]
    out = [
        lat0,
        _shuffle(_shuffle(lat1 @ q["enc.lat1.0"]) @ q["enc.lat1.1"]),
        _shuffle(f0 @ q["enc.up0.0"]),
        _shuffle(f1 @ q["enc.up1.0"]),
        _shuffle(low),
    ]
    if with_fov:
        z = _vit(p["fov"], cols[2]).astype(mx.float32)
        out.append(mx.addmm(q["fovnet.linear.bias"], z, q["fovnet.linear.weight"]))
    return tuple(out)


def _same(a: mx.array) -> mx.array:
    return a


def _level(  # noqa: PLR0913, PLR0917 - the block's pieces
    q: dict[str, mx.array],
    i: int,
    feat: mx.array | None,
    skip: mx.array,
    conv: Callable[[dict[str, mx.array], str, mx.array], mx.array] = _conv,
    ck: Callable[[mx.array], mx.array] = _same,
) -> mx.array:
    """Fusion block `i` of the features so far and the level's (projected) encoding, before its output layer.

    `conv`, `ck`: the 3x3 convolution, and the mark of a map to evaluate on its own (see `finish`).
    """

    def unit(k: str, x: mx.array) -> mx.array:
        y = conv(q, k + ".conv1", mx.maximum(x, 0))
        return x + conv(q, k + ".conv2", mx.maximum(y, 0))

    k = f"dec.fuse{i}"
    return ck(unit(k + ".unit2", skip if feat is None else ck(feat + unit(k + ".unit1", skip))))


def _out(q: dict[str, mx.array], i: int, x: mx.array) -> mx.array:
    return mx.addmm(q[f"dec.fuse{i}.out.bias"], x, q[f"dec.fuse{i}.out.weight"])


def decode(q: dict[str, mx.array], e2: mx.array, e3: mx.array, e4: mx.array) -> tuple:
    """Encodings 2-4 -> (features `[1, 384, 384, 256]` that enter level 1, level-4 features `[1, 48, 48, 256]`)."""
    low = _conv(q, "dec.conv4", e4)
    feat = None
    for i, e in ((4, None), (3, e3), (2, e2)):
        x = _level(q, i, feat, low if e is None else _conv(q, f"dec.conv{i}", e))
        feat = _shuffle(_out(q, i, x))
    return feat, low


def finish(
    q: dict[str, mx.array], feat: mx.array, e1: mx.array, lat0: mx.array, ck: Callable[[mx.array], mx.array]
) -> mx.array:
    """Levels 1 and 0 and the head -> canonical inverse depth, float32 `[1, 1, 1536, 1536]`.

    Not compiled, and evaluated piece by piece from the 768x768 maps on (`ck`): such a map of 256 channels is 300 MB
    in float16, and inside one evaluation none of the twenty would be freed. Their 3x3 convolutions run as `STRIPS`
    bands of rows: MLX's Winograd holds about four times the input and the output while it runs (2.4 GB for 256 ->
    256), and `ck` keeps two pieces in flight.
    """

    def conv(q: dict[str, mx.array], k: str, x: mx.array) -> mx.array:
        h = x.shape[1]
        edges = [h * j // STRIPS for j in range(STRIPS + 1)]
        out = []
        for a, b in itertools.pairwise(edges):
            lo, hi = max(a - 1, 0), min(b + 1, h)  # a row of context on each inner side (the zero padding's rows cut)
            out.append(ck(_conv(q, k, x[:, lo:hi])[:, a - lo : b - lo]))
        return mx.concatenate(out, axis=1)

    feat = ck(_shuffle(_out(q, 1, _level(q, 1, feat, _conv(q, "dec.conv1", e1)))))
    v = lat0
    for i in range(3):
        v = _shuffle(v @ q[f"enc.lat0.{i}"])
    x = ck(_out(q, 0, _level(q, 0, feat, ck(v), conv, ck)))
    del feat, v
    y = ck(mx.maximum(conv(q, "head.conv1", conv(q, "head.conv0", x)), 0).astype(mx.float32))
    z = mx.maximum(mx.addmm(q["head.out.bias"], y, q["head.out.weight"]), 0)  # [1, 768, 768, 4]: the phases
    _, h, w, _ = z.shape
    return z.reshape(1, h, w, 2, 2).transpose(0, 1, 3, 2, 4).reshape(1, 1, 2 * h, 2 * w)


def field_of_view(q: dict[str, mx.array], z: mx.array, low: mx.array) -> mx.array:
    """FOV tokens and the level-4 features -> the field of view in degrees, float32 `[1, 1, 1, 1]`."""
    x = z + mx.maximum(_conv(q, "fovnet.down", low.astype(mx.float32), stride=2), 0)
    x = mx.maximum(_conv(q, "fovnet.head0", x, stride=2), 0)
    x = mx.maximum(_conv(q, "fovnet.head1", x, stride=2), 0)
    return mx.conv2d(x, q["fovnet.head2.weight"]) + q["fovnet.head2.bias"]


class Net:
    """The MLX network: converted weights and the compiled stages."""

    def __init__(self, p32: dict[str, mx.array], dtype: str) -> None:
        """Cast the float32 parameters (`p32`, emptied) to the run dtype."""
        run = getattr(mx, dtype)
        p: dict[str, dict[str, mx.array]] = {name: {} for name in (*VITS, "net")}
        freed = 0
        for k in list(p32):  # one at a time: never the float32 and the run dtype's weights of the whole net at once
            v = p32.pop(k)
            a = v if k.startswith(FP32) else v.astype(run)
            mx.eval(a)
            part, _, name = k.removeprefix("vit.").partition(".") if k.startswith("vit.") else ("net", "", k)
            p[part][name] = a
            freed += v.nbytes if a.dtype != v.dtype else 0
            del v
            if freed > _CAST_CACHE:  # the float32 originals would stay in MLX's buffer cache
                mx.clear_cache()
                freed = 0
        mx.clear_cache()
        q = p["net"]
        index = tuple(to_mx(t) for t in tables())
        wrap: Callable[[Callable], Callable] = mx.compile if tuning().compile_steps else (lambda f: f)
        self.encode = {fov: wrap(lambda x, fov=fov: encode(p, index, x, with_fov=fov)) for fov in (False, True)}
        self.decode = wrap(lambda *e: decode(q, *e))
        self.q = q
        self.fov = wrap(lambda z, low: field_of_view(q, z, low))

    def __call__(self, x: torch.Tensor, *, with_fov: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalised `[1, 3, 1536, 1536]` CPU tensor -> float32 CPU (canonical inverse depth, FOV or None)."""
        enc = list(self.encode[with_fov](to_mx(x)))
        mx.async_eval(enc)
        z = enc.pop() if with_fov else None
        lat0, e1, *rest = enc
        del enc
        feat, low = self.decode(*rest)
        del rest
        mx.async_eval(feat, low)
        fov = self.fov(z, low) if z is not None else None
        del z, low
        if fov is not None:
            mx.async_eval(fov)
        queued: list[mx.array] = []

        def ck(a: mx.array) -> mx.array:
            # queue this piece before waiting for the one before it: two in flight (one at a time: 4.24 -> 4.5 s)
            mx.async_eval(a)
            if queued:
                mx.eval(queued.pop())
            queued.append(a)
            return a

        cid = finish(self.q, feat, e1, lat0, ck)
        del feat, e1, lat0
        mx.eval(cid, *(() if fov is None else (fov,)))
        return torch.from_numpy(np.array(cid)), None if fov is None else torch.from_numpy(np.array(fov))
