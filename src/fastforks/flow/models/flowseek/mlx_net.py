"""FlowSeek-T on Apple MLX, written in `mlx.core` operations: `MlxFlowSeek(sd, dtype, iters)(image1, image2)`.

The maths of `net.FlowSeek.forward`, channels last, built from the checkpoint's state dict (`shared.mlx_flow`):

- Depth Anything V2 Small on both images resized to 518 x 518: the DINOv2 blocks of `shared.mlx_vit` (LayerScale
  folded, the patch embedding one matmul, the 37 x 37 position table with the cls token and patch bias in it) and the
  DPT head of `shared.depth_anything_v2.mlx_net` (its projections and transposed convs one matmul, each fusion
  block's 1 x 1 conv before its resize); the depth branch (`output_conv*`) runs for image 1 only, the only depth used;
- the fusion feature resized to the image size and the merge head one compiled step per image (the resized map in the
  run dtype: torch resizes in float32 and the conv casts once, the kernel accumulates in float32 and rounds once);
- the motion bases and their padding in float32;
- the ResNet18 encoders with their batch norms folded, each residual block one compiled step;
- no correlation volume: `graph_mlx.ops.corr_windows` from fmap1 and the 2 x 2 average-pooled levels of fmap2 (the
  bilinear halving at half-pixel centres is that pooling, and the pooled volume is the volume of the pooled maps);
- each update one compiled step, one in flight: the flow state and its first conv, the correlation windows and the
  convex upsampling in float32, the flow head's two flow rows only, the mask head's 0.25 folded in.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_flow as mf
from fastforks.shared.cspnext.mlx_net import stepped, steps
from fastforks.shared.depth_anything_v2 import mlx_net as da2
from fastforks.shared.mlx_vit import dinov2_layers

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fastforks.shared.mlx_flow import Conv

__all__ = ["LAYOUT", "SOURCES", "MlxFlowSeek"]

LAYOUT = "flowseek-mlx-1"
SOURCES = (Path(__file__), Path(mf.__file__), Path(da2.__file__))

FACTOR, LEVELS, RADIUS, DIM = 8, 4, 4, 128
DA_SIZE, PATCH, HEADS, LAYERS = 518, 14, 6, (3, 6, 9, 12)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_FLOW = (0, 1)  # the flow head's flow rows (the rest is the uncertainty)


def _bilinear(x: mx.array, h: int, w: int, *, align: bool) -> mx.array:
    """`F.interpolate(mode="bilinear")` of `[N, H, W, C]` to `(h, w)` (float32 taps, one rounding)."""
    _, ih, iw, _ = x.shape

    def ratio(n: int, m: int) -> float:
        return ((n - 1) / (m - 1) if m > 1 else 0.0) if align else n / m

    ratios = (ratio(ih, h), ratio(iw, w))
    return ops.bilinear_nhwc(x, (h, w), ratios, align=align)


def _linear(
    sd: Mapping[str, Any], name: str, shape: tuple[int, int], gamma: str | None, dt: mx.Dtype
) -> tuple[mx.array, mx.array]:
    """The `Linear` `name` as an `[in, out]` weight and bias, the LayerScale `gamma` folded in."""

    def scale() -> Any:  # noqa: ANN401 - an array or 1
        return mf.np64(sd[gamma]) if gamma is not None else 1.0

    w = mf.put(shape, dt, lambda: mf.np64(sd[f"{name}.weight"]).T * scale())
    return w, mf.put((shape[1],), dt, lambda: mf.np64(sd[f"{name}.bias"]) * scale())


def _vit(sd: Mapping[str, Any], p: str, dt: mx.Dtype) -> dict[str, mx.array]:
    """DINOv2 ViT-S `p` of `sd` in `mlx_vit`'s layout (`dinov2_params` and `dinov2_table` for the 37 x 37 grid)."""
    out: dict[str, mx.array] = {}
    c = int(sd[f"{p}.cls_token"].shape[-1])
    n = int(sd[f"{p}.pos_embed"].shape[1])

    def take(k: str) -> np.ndarray:
        return mf.np64(sd[f"{p}.{k}"])

    out["patch.weight"] = mf.put(
        (PATCH * PATCH * 3, c), dt, lambda: take("patch_embed.proj.weight").transpose(2, 3, 1, 0).reshape(-1, c)
    )

    def table() -> np.ndarray:
        pos = take("pos_embed")[0]
        return np.concatenate([pos[:1] + take("cls_token")[0], pos[1:] + take("patch_embed.proj.bias")])

    out["table"] = mf.put((n, c), dt, table)
    i = 0
    while f"{p}.blocks.{i}.norm1.weight" in sd:
        b, o = f"{p}.blocks.{i}.", f"blocks.{i}."
        hidden = int(sd[f"{b}mlp.fc1.weight"].shape[0])
        for name in ("norm1", "norm2"):
            for kind in ("weight", "bias"):
                out[f"{o}{name}.{kind}"] = mf.put((c,), dt, lambda k=f"{b}{name}.{kind}": mf.np64(sd[k]))
        for dst, src, shape, gamma in (
            ("qkv", "attn.qkv", (c, 3 * c), None),
            ("proj", "attn.proj", (c, c), f"{b}ls1.gamma"),
            ("fc1", "mlp.fc1", (c, hidden), None),
            ("fc2", "mlp.fc2", (hidden, c), f"{b}ls2.gamma"),
        ):
            out[f"{o}{dst}.weight"], out[f"{o}{dst}.bias"] = _linear(sd, f"{b}{src}", shape, gamma, dt)
        i += 1
    for kind in ("weight", "bias"):
        out[f"norm.{kind}"] = mf.put((c,), dt, lambda k=f"{p}.norm.{kind}": mf.np64(sd[k]))
    return out


def _projection(sd: Mapping[str, Any], p: str, i: int, dt: mx.Dtype) -> dict[str, mx.array]:
    """The head's projection `i` (with the transposed conv after it for the two finest levels) as one matmul."""
    out: dict[str, mx.array] = {}

    def take(k: str) -> np.ndarray:
        return mf.np64(sd[f"{p}.{k}"])

    def shape(k: str) -> tuple[int, ...]:
        return tuple(int(n) for n in sd[f"{p}.{k}"].shape)

    c, k_in = shape(f"projects.{i}.weight")[:2]
    if i < len(da2.SHUFFLE):
        k = da2.SHUFFLE[i]
        o = shape(f"resize_layers.{i}.weight")[1]

        def wt(i: int = i) -> np.ndarray:
            w, wd = take(f"projects.{i}.weight")[:, :, 0, 0], take(f"resize_layers.{i}.weight")
            return np.einsum("ci,coab->iabo", w, wd).reshape(w.shape[1], -1)

        def bs(i: int = i) -> np.ndarray:
            b, wd = take(f"projects.{i}.bias"), take(f"resize_layers.{i}.weight")
            return (np.einsum("c,coab->abo", b, wd) + take(f"resize_layers.{i}.bias")).reshape(-1)

        out[f"head.proj{i}.weight"] = mf.put((k_in, k * k * o), dt, wt)
        out[f"head.proj{i}.bias"] = mf.put((k * k * o,), dt, bs)
    else:
        out[f"head.proj{i}.weight"] = mf.put((k_in, c), dt, lambda i=i: take(f"projects.{i}.weight")[:, :, 0, 0].T)
        out[f"head.proj{i}.bias"] = mf.put((c,), dt, lambda i=i: take(f"projects.{i}.bias"))
    return out


def _head(sd: Mapping[str, Any], p: str, dt: mx.Dtype) -> dict[str, mx.array]:
    """The DPT head `p` of `sd` in `depth_anything_v2.mlx_net.prepare`'s layout (`head.*`)."""
    out: dict[str, mx.array] = {}

    def take(k: str) -> np.ndarray:
        return mf.np64(sd[f"{p}.{k}"])

    def shape(k: str) -> tuple[int, ...]:
        return tuple(int(n) for n in sd[f"{p}.{k}"].shape)

    def conv(dst: str, src: str, *, bias: bool = True) -> None:
        o, i, kh, kw = shape(f"{src}.weight")
        out[f"head.{dst}.weight"] = mf.put((o, kh, kw, i), dt, lambda: take(f"{src}.weight").transpose(0, 2, 3, 1))
        if bias:
            out[f"head.{dst}.bias"] = mf.put((o,), dt, lambda: take(f"{src}.bias"))

    for i in range(4):
        out.update(_projection(sd, p, i, dt))
        conv(f"rn{i}", f"scratch.layer{i + 1}_rn", bias=False)
    conv("down", "resize_layers.3")
    for i in range(4):
        src = f"scratch.refinenet{i + 1}"
        for u in (2,) if i == len(da2.SHUFFLE) + 1 else (1, 2):  # the coarsest block's unit 1: no input, unused
            for c in (1, 2):
                conv(f"fuse{i}.unit{u}.conv{c}", f"{src}.resConfUnit{u}.conv{c}")
        o, k_in = shape(f"{src}.out_conv.weight")[:2]
        out[f"head.fuse{i}.out.weight"] = mf.put(
            (k_in, o), dt, lambda s=src: take(f"{s}.out_conv.weight")[:, :, 0, 0].T
        )
        out[f"head.fuse{i}.out.bias"] = mf.put((o,), dt, lambda s=src: take(f"{s}.out_conv.bias"))
    conv("out1", "scratch.output_conv1")
    conv("out2", "scratch.output_conv2.0")
    k_in = shape("scratch.output_conv2.2.weight")[1]
    out["head.out3.weight"] = mf.put((k_in, 1), mx.float32, lambda: take("scratch.output_conv2.2.weight")[:, :, 0, 0].T)
    out["head.out3.bias"] = mf.put((1,), mx.float32, lambda: take("scratch.output_conv2.2.bias"))
    return out


class _Encoder:
    """`ResNetFPN`: 7 x 7 stride-2 stem, ResNet18 stages (strides 1, 2, 2), a 1 x 1 conv; batch norms folded."""

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype) -> None:
        """Convert the encoder `p` of `sd`."""
        stem = mf.conv(sd, f"{p}.conv1", dtype, stride=2, bn=f"{p}.bn1")
        self._steps = [stepped(lambda x: mf.relu(stem(x)))]
        for layer in (1, 2, 3):
            i = 0
            while f"{p}.layer{layer}.{i}.conv1.weight" in sd:
                b = f"{p}.layer{layer}.{i}"
                stride = 2 if i == 0 and layer > 1 else 1
                c1 = mf.conv(sd, f"{b}.conv1", dtype, stride=stride, bn=f"{b}.bn1")
                c2 = mf.conv(sd, f"{b}.conv2", dtype, bn=f"{b}.bn2")
                ds = (
                    mf.conv(sd, f"{b}.downsample.0", dtype, stride=stride, padding=0, bn=f"{b}.bn3")
                    if f"{b}.downsample.0.weight" in sd
                    else None
                )
                self._steps.append(stepped(self._block(c1, c2, ds)))
                i += 1
        self.out = mf.conv(sd, f"{p}.final_conv", dtype)

    @staticmethod
    def _block(c1: Conv, c2: Conv, ds: Conv | None) -> Callable[[mx.array], mx.array]:
        def run(x: mx.array) -> mx.array:
            y = mf.relu(c2(mf.relu(c1(x))))
            return mf.relu((ds(x) if ds is not None else x) + y)

        return run

    def __call__(self, x: mx.array) -> mx.array:
        """`[N, H, W, cin]` -> `[N, H / 8, W / 8, 256]` before the final 1 x 1 conv (`out`, the caller's step)."""
        for f in self._steps:
            x = f(x)
        return x


def _unit(x: mx.array) -> mx.array:
    """`x` `[1, H, W, 2]` divided by its norm."""
    return x * mx.rsqrt(mx.sum(mx.square(x)))


def motion_bases(disp: mx.array) -> mx.array:
    """`net.motion_bases` of float32 `[1, H, W, 1]` -> `[1, H, W, 16]` (float32)."""
    _, h, w, _ = disp.shape
    ys = (mx.arange(h, dtype=mx.float32) + 0.5) / h - 0.5
    xs = (mx.arange(w, dtype=mx.float32) + 0.5) / w - 0.5
    u = mx.broadcast_to(xs[None, None, :, None], (1, h, w, 1))
    v = mx.broadcast_to(ys[None, :, None, None], (1, h, w, 1))
    aspect = w / h
    zero, one = mx.zeros_like(disp), mx.ones_like(disp)

    def cat(a: mx.array, b: mx.array) -> mx.array:
        return mx.concatenate([a, b], axis=-1)

    trans = [cat(-one, zero), cat(zero, -one), cat(u, v)]
    rot = [cat(zero, one), cat(u * v, v * v), cat(-one, zero), cat(-u * u, -u * v), cat(-v / aspect, u * aspect)]
    return mx.concatenate([2 * disp * _unit(t) for t in trans] + [_unit(r) for r in rot], axis=-1)


class MlxFlowSeek:
    """FlowSeek-T from the state dict `sd` in `dtype` (the name), `iters` updates; one pair per call."""

    def __init__(self, sd: Mapping[str, Any], dtype: str, iters: int) -> None:
        """Convert the weights (through the conversion cache, `mlx_port.recorded`)."""
        dt = self.dtype = mf.DTYPES[dtype]
        self.iters = iters
        self.vit = _vit(sd, "dav2.pretrained", dt)
        self.p = _head(sd, "dav2.depth_head", dt)
        self.merge = [mf.conv(sd, f"merge_head.{i}", dt, stride=2) for i in (0, 2, 4)]
        self.cnet = _Encoder(sd, "cnet", dt)
        self.bnet = _Encoder(sd, "bnet", dt)
        self.fnet = _Encoder(sd, "fnet", dt)
        self.init_conv = mf.conv(sd, "init_conv", dt)
        self.head1 = mf.conv(sd, "flow_head.0", dt)
        self.head2 = mf.conv(sd, "flow_head.2", dt, rows=_FLOW)
        e = "update_block.encoder"
        self.convc1 = mf.conv(sd, f"{e}.convc1", dt)
        self.convc2 = mf.conv(sd, f"{e}.convc2", dt)
        self.convf1 = mf.conv(sd, f"{e}.convf1", mx.float32)
        self.convf2 = mf.conv(sd, f"{e}.convf2", dt)
        self.conv = mf.conv(sd, f"{e}.conv", dt)
        self.refine = mf.blocks(sd, "update_block.refine", dt)
        self.up1 = mf.conv(sd, "upsample_weight.0", dt)
        self.up2 = mf.conv(sd, "upsample_weight.2", dt, rows=mf.convex_rows(FACTOR), scale=0.25)
        self.mean = mx.array(MEAN, dtype=mx.float32)
        self.std = mx.array(STD, dtype=mx.float32)
        self._mono = stepped(self._mono_fn)
        self._merge = stepped(self._merge_fn)
        self._context = stepped(self._context_fn)
        self._features = stepped(lambda f, mono: mx.concatenate([self.fnet.out(f), mono], axis=-1))
        self._step = stepped(self._update)
        self._mask = stepped(lambda net: self.up2(mf.relu(self.up1(net))))

    # Depth Anything ---------------------------------------------------------------------------------------------------

    def _mono_fn(self, both: mx.array) -> tuple[mx.array, mx.array]:
        """`FlowSeek._mono` of `[2, H, W, 3]` in [0, 1]: the fusion feature `[2, 296, 296, 64]`, image 1's depth."""
        p = self.p
        small = _bilinear(both, DA_SIZE, DA_SIZE, align=False)
        x = (small / self.mean - self.std).transpose(0, 3, 1, 2)
        h0 = w0 = DA_SIZE // PATCH
        feats = [t[:, 1:] for t in dinov2_layers(self.vit, HEADS, x, self.vit["table"], LAYERS)]

        def up(y: mx.array, h: int, w: int) -> mx.array:
            return _bilinear(y, h, w, align=True)

        maps = []
        for i, t in enumerate(feats):
            y = mx.addmm(p[f"head.proj{i}.bias"], t, p[f"head.proj{i}.weight"])
            if i < len(da2.SHUFFLE):
                k = da2.SHUFFLE[i]
                y = y.reshape(2, h0, w0, k, k, -1).transpose(0, 1, 3, 2, 4, 5).reshape(2, h0 * k, w0 * k, -1)
            else:
                y = y.reshape(2, h0, w0, -1)
            if i == len(feats) - 1:
                y = da2._conv(p, "head.down", y, stride=2)  # noqa: SLF001 - the DPT head's pieces
            maps.append(da2._conv(p, f"head.rn{i}", y))  # noqa: SLF001
        size = [(m.shape[1], m.shape[2]) for m in maps]
        x = da2._fuse(p, "head.fuse3", maps[3], None, size[2], up)  # noqa: SLF001
        x = da2._fuse(p, "head.fuse2", x, maps[2], size[1], up)  # noqa: SLF001
        x = da2._fuse(p, "head.fuse1", x, maps[1], size[0], up)  # noqa: SLF001
        path = da2._fuse(p, "head.fuse0", x, maps[0], (2 * size[0][0], 2 * size[0][1]), up)  # noqa: SLF001
        d = up(da2._conv(p, "head.out1", path[:1]), h0 * PATCH, w0 * PATCH)  # noqa: SLF001
        d = mf.relu(da2._conv(p, "head.out2", d)).astype(mx.float32)  # noqa: SLF001
        d = mf.relu(d @ p["head.out3.weight"] + p["head.out3.bias"])
        return path, d

    def _merge_fn(self, path: mx.array, h: int, w: int) -> mx.array:
        """The merge head on one image's fusion feature resized to `(h, w)`: `[1, h / 8, w / 8, 128]`."""
        x = _bilinear(path, h, w, align=False)
        m0, m1, m2 = self.merge
        return m2(mf.relu(m1(mf.relu(m0(x)))))

    # the flow network -------------------------------------------------------------------------------------------------

    def _context_fn(self, c: mx.array, b: mx.array) -> list[mx.array]:
        """Hidden state, context and the initial flow from the context and motion-basis encoders."""
        c = self.init_conv(self.cnet.out(c))
        b = self.init_conv(self.bnet.out(b))
        net = mx.concatenate([c[..., :DIM], b[..., :DIM]], axis=-1)
        context = mx.concatenate([c[..., DIM:], b[..., DIM:]], axis=-1)
        flow = self.head2(mf.relu(self.head1(net))).astype(mx.float32)
        return [net, context, flow]

    def _update(
        self, net: mx.array, flow: mx.array, context: mx.array, f1: mx.array, *levels: mx.array
    ) -> tuple[mx.array, mx.array]:
        _, h, w, _ = net.shape
        dt = self.dtype
        coords = (mf.grid(h, w) + flow).transpose(0, 3, 1, 2)
        corr = ops.corr_windows(f1, list(levels), coords, RADIUS, 1 / math.sqrt(f1.shape[-1]))
        corr = corr[0].reshape(-1, h * w).T.astype(dt)  # [h w, L (2r + 1)^2]
        cor = mf.relu(mx.addmm(self.convc1.b, corr, self.convc1.w.T)).reshape(1, h, w, -1)
        cor = mf.relu(self.convc2(cor))
        flo = mf.relu(self.convf2(mf.relu(self.convf1(flow)).astype(dt)))
        motion = mf.relu(self.conv(mx.concatenate([cor, flo], axis=-1)))
        motion = mx.concatenate([motion, flow.astype(dt)], axis=-1)
        net = mf.refine(self.refine, net, [context, motion])
        return net, flow + self.head2(mf.relu(self.head1(net))).astype(mx.float32)

    def _run(self, image1: mx.array, image2: mx.array) -> mx.array:
        dt = self.dtype
        _, _, h, w = image1.shape
        both = mx.concatenate([image1, image2], axis=0).transpose(0, 2, 3, 1)  # float32 [2, H, W, 3]
        path, depth = self._mono(both)
        mono = mx.concatenate([self._merge(path[i : i + 1], h, w) for i in (0, 1)], axis=0)
        bases = motion_bases(_bilinear(depth, h, w, align=False))
        ph, pw = -h % FACTOR, -w % FACTOR
        pads = ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0))
        both = mx.pad(2 * both - 1, pads, mode="edge").astype(dt)
        bases = mx.pad(bases, pads, mode="edge").astype(dt)
        net, context, flow = self._context(self.cnet(mx.concatenate([both[:1], both[1:]], axis=-1)), self.bnet(bases))
        f = self._features(self.fnet(both), mono)
        f1 = f[:1]
        levels = mf.pyramid(f[1:], LEVELS)
        for _ in range(self.iters):
            net, flow = self._step(net, flow, context, f1, *levels)
        up = mf.convex_upsample(flow, self._mask(net), FACTOR)
        up = up[:, :, ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
        mx.eval(up)
        return up

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[1, 2, H, W]` (float32) of an unpadded float32 pair `[1, 3, H, W]` in [0, 1]."""
        mf.options(half_wino_f32=self.dtype == mx.float16)
        return steps(self._run, image1, image2)
