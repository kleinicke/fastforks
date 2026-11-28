"""HITNet on Apple MLX, written in `mlx.core` operations (`MlxHitnet(net, dtype, ...)`).

`net.hitnet.HITNet.forward` as one compiled function, channels last: feature U-Net on both images as one batch,
then the level-init, propagation and refinement stages. The three cost ops (`hitnet_init_cost`, `hitnet_tile_sad`,
`hitnet_refine_sad`, `fastforks.kernels.graph_mlx.ops`) are Metal kernels with no `[n, C, h, w, D]` volume. The
hypotheses (d, dx, dy and 13 descriptors), disparities, argmin decisions and warp positions stay float32; the
convolutions run in the run dtype. The images are padded and scaled here (the runner passes the raw 0..255 pair).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_stereo
from fastforks.shared.mlx_stereo import Weights, conv2d

from .net import weights as _weights

if TYPE_CHECKING:
    from collections.abc import Callable

    from .net.hitnet import HITNet

__all__ = ["LAYOUT", "SOURCES", "MlxHitnet"]

LAYOUT = "hitnet-v1"
SOURCES = (Path(__file__), Path(mlx_stereo.__file__), Path(_weights.__file__))
SLOPE = 0.2  # LeakyReLU slope of every activation of the released graphs
TILE = 4  # tile size of the hypotheses (pixels)
HYP = 16  # hypothesis channels
FIRST_REF_LEVEL = 2  # from this level on the init also sees the left features two scales up
REFINE_BLOCKS = 2  # residual blocks of a refinement stage per evaluated graph


def lrelu(x: mx.array) -> mx.array:
    """LeakyReLU(0.2)."""
    return mx.maximum(x, SLOPE * x)


def _pw(x: mx.array, c: tuple[mx.array, mx.array]) -> mx.array:
    """A 1 x 1 convolution of NHWC `x` as one matrix product (`c`: weight `[O, I]`, bias)."""
    w, b = c
    return mx.addmm(b, x, w.T)


def _upsample_hyp(hyp: mx.array, dmul: float, coef: float) -> mx.array:
    """2x tile upsampling of `[n, h, w, 16]`: `d' = dmul d + coef (+-dx +- dy)` per child, the rest nearest."""
    n, h, w, _ = hyp.shape
    d, dx, dy = (hyp[..., i].reshape(n, h, 1, w, 1) for i in range(3))
    s = mx.array([-1.0, 1.0], dtype=mx.float32)
    e = dmul * d + coef * (s.reshape(1, 1, 1, 1, 2) * dx + s.reshape(1, 1, 2, 1, 1) * dy)  # [n, h, 2 (by), w, 2 (bx)]
    rest = mx.repeat(mx.repeat(hyp[..., 1:], 2, axis=1), 2, axis=2)
    return mx.concatenate([e.reshape(n, 2 * h, 2 * w, 1), rest], axis=-1)


class MlxHitnet:
    """The network `net` (eval, float32 weights) in `dtype` (the name): one compiled function per image size."""

    def __init__(self, net: HITNet, dtype: str, *, pad_multiple: int, pad_value: float, pad_value_right: float) -> None:
        """Copy the weights of `net` to MLX; the padding is the variant's (`Params`)."""
        self.dtype = getattr(mx, dtype)
        self.in_ch = int(net.in_ch)
        self.pad_multiple = pad_multiple
        self.pads = ((pad_value + 1.0) * 127.5, (pad_value_right + 1.0) * 127.5)
        w = Weights(self.dtype)
        fe: Any = net.fe
        n = int(fe.n)
        self.n = n
        self.down1 = [self._conv(w, c) for c in fe.conv_down_1]
        self.down = [self._conv(w, c) for c in fe.conv_down]
        self.up1n, self.up2n = self._conv(w, fe.conv_up_1n), self._conv(w, fe.conv_up_2n)
        self.upsample = [self._deconv(w, c) for c in fe.conv_upsample]
        self.up0 = [self._merge(w, c) for c in fe.conv_up_0]
        self.up1 = [self._conv(w, c) for c in fe.conv_up_1]
        self.up2 = [self._conv(w, c) for c in fe.conv_up_2]
        self.levels: list[dict[str, Any]] = []
        inits: Any = net.init
        props: Any = net.prop
        for init, prop in zip(inits, props, strict=True):
            self.levels.append(
                {
                    "max_disp": int(init.max_disp),
                    "reduce": self._conv(w, init.conv_reduce),
                    "embed": self._pointwise(w, init.conv_embedding),
                    "hyp": self._pointwise(w, init.conv_hyp),
                    "neighbors": self._pointwise(w, prop.conv_neighbors),
                    "prop": self._resnet(w, prop.prop),
                    "nh": int(prop.nh),
                }
            )
        self.refine: list[dict[str, Any]] = [
            {
                "sad_scale": r.sad_scale,
                "conv1x1": self._pointwise(w, r.conv1x1),
                "net": self._resnet(w, r.refiner),
            }
            for r in net.refine
        ]
        mx.eval(self._arrays())
        self._fns: dict[str, Callable[..., mx.array]] = {}

    @staticmethod
    def _conv(w: Weights, conv: Any) -> tuple[mx.array, mx.array, int, int, int]:  # noqa: ANN401 - torch module
        return (*w.conv(conv), int(conv.dilation[0]))

    @staticmethod
    def _pointwise(w: Weights, conv: Any) -> tuple[mx.array, mx.array]:  # noqa: ANN401
        cout, cin = int(conv.weight.shape[0]), int(conv.weight.shape[1])
        return (
            w.a((cout, cin), lambda: conv.weight.detach().float().numpy().reshape(cout, cin)),
            w.a((cout,), lambda: conv.bias.detach().float().numpy()),
        )

    @staticmethod
    def _merge(w: Weights, conv: Any) -> tuple[mx.array, mx.array, mx.array]:  # noqa: ANN401
        """The 1 x 1 convolution over `[upsampled, skip]` as two matrices `[O, I / 2]` and the bias."""
        cout, cin = int(conv.weight.shape[0]), int(conv.weight.shape[1])
        half = cin // 2
        return (
            w.a((cout, half), lambda: conv.weight.detach().float().numpy().reshape(cout, cin)[:, :half]),
            w.a((cout, half), lambda: conv.weight.detach().float().numpy().reshape(cout, cin)[:, half:]),
            w.a((cout,), lambda: conv.bias.detach().float().numpy()),
        )

    @staticmethod
    def _deconv(w: Weights, conv: Any) -> tuple[mx.array, mx.array]:  # noqa: ANN401
        """A 2 x 2 stride-2 transposed convolution as a matrix `[Cin, 2 2 Cout]` (tile rows: (kh, kw, Cout))."""
        cin, cout = int(conv.weight.shape[0]), int(conv.weight.shape[1])
        return (
            w.a((cin, 4 * cout), lambda: conv.weight.detach().float().numpy().transpose(0, 2, 3, 1).reshape(cin, -1)),
            w.a((4 * cout,), lambda: np.tile(conv.bias.detach().float().numpy(), 4)),
        )

    def _resnet(self, w: Weights, m: Any) -> tuple[Any, ...]:  # noqa: ANN401
        return (
            self._conv(w, m.conv1),
            [(self._conv(w, r.conv1), self._conv(w, r.conv2)) for r in m.res],
            self._conv(w, m.convn),
        )

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

    @staticmethod
    def _run_resnet(x: mx.array, m: tuple[Any, ...]) -> mx.array:
        first, blocks, last = m
        x = lrelu(conv2d(x, first))
        for c1, c2 in blocks:
            x = lrelu(conv2d(lrelu(conv2d(x, c1)), c2) + x)
        return conv2d(x, last)

    def _input(self, left: mx.array, right: mx.array) -> mx.array:
        """Pad, scale and the first feature layer of both images (one batch)."""
        h, w = left.shape[2], left.shape[3]
        m = self.pad_multiple
        widths = [(0, 0), (0, -h % m), (0, -w % m), (0, 0)]
        imgs = [
            mx.pad(x.transpose(0, 2, 3, 1), widths, constant_values=v)
            for x, v in zip((left, right), self.pads, strict=True)
        ]
        x = mx.concatenate(imgs, axis=0)
        if self.in_ch == 1:
            x = 0.299 * x[..., 0:1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]
        return lrelu(conv2d((x * (2.0 / 255.0) - 1.0).astype(self.dtype), self.down1[0]))

    def _down(self, x: mx.array, i: int) -> mx.array:
        """Skip `i + 1` of the contracting path."""
        return lrelu(conv2d(lrelu(conv2d(x, self.down[i])), self.down1[i + 1]))

    def _bottom(self, x: mx.array) -> mx.array:
        return lrelu(conv2d(lrelu(conv2d(x, self.up1n)), self.up2n))

    def _up_merge(self, x: mx.array, s: mx.array, i: int) -> mx.array:
        """`lrelu(conv_up_0(cat(lrelu(deconv(x)), s)))`, without the concatenation.

        The upsampled half of the merge matrix is applied before the pixel shuffle, the skip's half after it.
        """
        wt, bt = self.upsample[i]
        wu, ws, b = self.up0[i]
        n, h, wd, _ = x.shape
        u = lrelu(mx.addmm(bt, x, wt)).reshape(n, h, wd, 4, -1) @ wu.T  # [n, h, w, 2 2, O]
        u = u.reshape(n, h, wd, 2, 2, -1).transpose(0, 1, 3, 2, 4, 5).reshape(n, 2 * h, 2 * wd, -1)
        u = u[:, : s.shape[1], : s.shape[2]]
        return lrelu(u + mx.addmm(b, s, ws.T))

    def _up_convs(self, x: mx.array, i: int) -> mx.array:
        return lrelu(conv2d(lrelu(conv2d(x, self.up1[i])), self.up2[i]))

    def _embed(self, lv: dict[str, Any], fl: mx.array, fr: mx.array) -> tuple[mx.array, mx.array]:
        """Left tile embedding (stride 4) and right embedding (stride (4, 1): every pixel)."""
        w, b, _, _, _ = lv["reduce"]
        lt = lrelu(_pw(lrelu(mx.conv2d(fl, w, stride=TILE) + b), lv["embed"]))
        fr = mx.pad(fr, [(0, 0), (0, 0), (1, 2), (0, 0)])
        rt = lrelu(_pw(lrelu(mx.conv2d(fr, w, stride=(TILE, 1)) + b), lv["embed"]))
        return lt, rt

    def _init(self, lv: dict[str, Any], fl: mx.array, fr: mx.array, ref: mx.array | None) -> mx.array:
        """Hypothesis `[n, h, w, 16]` float32 of the level: `[d, 0, 0, 13 descriptor channels]`."""
        lt, rt = self._embed(lv, fl, fr)
        d, cmin = ops.hitnet_init_cost(lt, rt, lv["max_disp"])  # [n, 1, h, w] float32 each (one channel: NHWC = NCHW)
        n, _, h, wd = d.shape
        d, cmin = d.reshape(n, h, wd, 1), cmin.reshape(n, h, wd, 1)
        p = lrelu(_pw(mx.concatenate([cmin.astype(lt.dtype), lt if ref is None else ref], axis=-1), lv["hyp"]))
        z = mx.zeros_like(d)
        return mx.concatenate([d, z, z, p.astype(mx.float32)], axis=-1)

    def _prop(self, lv: dict[str, Any], hyps: list[mx.array], fl: mx.array, fr: mx.array) -> mx.array:
        """Propagated hypothesis: tile costs, ResNet update, the hypothesis of the smallest predicted weight."""
        parts = []
        for hp in hyps:
            nchw = hp.transpose(0, 3, 1, 2)  # planes; all 16, as MLX passes a tiny input as `constant`
            cost = ops.hitnet_tile_sad(nchw, fl, fr).astype(fl.dtype)  # [n, h, w, 64]
            parts += [hp.astype(fl.dtype), lrelu(_pw(cost, lv["neighbors"]))]
        x = self._run_resnet(mx.concatenate(parts, axis=-1), lv["prop"]).astype(mx.float32)
        nh = lv["nh"]
        new = [hyps[i] + x[..., (HYP + 1) * i : (HYP + 1) * i + HYP] for i in range(nh)]
        if nh == 1:
            return new[0]
        sel = mx.argmin(
            mx.concatenate([x[..., (HYP + 1) * i + HYP : (HYP + 1) * (i + 1)] for i in range(nh)], axis=-1), axis=-1
        )
        out = new[0]
        for i in range(1, nh):
            out = mx.where((sel == i)[..., None], new[i], out)
        return out

    def _refine_head(self, r: int, hyp: mx.array, fl: mx.array, fr: mx.array) -> mx.array:
        """Refinement stage `r` up to its first convolution."""
        st = self.refine[r]
        if st["sad_scale"] is not None:  # the last 3 feature channels become the warp SADs at offsets 0, +1, -1
            n, h, wd, _ = hyp.shape
            d = (hyp[..., 0:1] * st["sad_scale"]).reshape(n, 1, h, wd)
            fl = mx.concatenate([fl[..., :-3], ops.hitnet_refine_sad(d, fl, fr, out=fl.dtype)], axis=-1)
        x = lrelu(_pw(mx.concatenate([fl, hyp.astype(fl.dtype)], axis=-1), st["conv1x1"]))
        return lrelu(conv2d(x, st["net"][0]))

    def _refine_blocks(self, r: int, lo: int, hi: int, x: mx.array) -> mx.array:
        for c1, c2 in self.refine[r]["net"][1][lo:hi]:
            x = lrelu(conv2d(lrelu(conv2d(x, c1)), c2) + x)
        return x

    def _refine_tail(self, r: int, x: mx.array, hyp: mx.array) -> mx.array:
        """Last convolution, the residual add and (but for the last stage) the 2x tile upsampling."""
        hyp = hyp + conv2d(x, self.refine[r]["net"][2]).astype(mx.float32)
        return _upsample_hyp(hyp, 1.0, 0.5) if r < len(self.refine) - 1 else hyp

    def _level(self, k: int, hyp: mx.array, fl: mx.array, fr: mx.array, ref: mx.array) -> mx.array:
        """Level `k`: its init (`ref`: the left features two scales up, from level 2 on) and propagation.

        The propagation takes the upsampled hypothesis `hyp` of the level before as a second candidate.
        """
        lv = self.levels[k]
        hi = self._init(lv, fl, fr, None if k < FIRST_REF_LEVEL else ref)
        return self._prop(lv, [hi] if k == 0 else [hi, _upsample_hyp(hyp, 2.0, 2.0)], fl, fr)

    def _run(self, key: str, fn: Callable[..., mx.array], *args: mx.array) -> mx.array:
        """`fn(*args)` as its own compiled graph, evaluated now (queueing raised the peak 30 %).

        MLX frees a command buffer's transients only when it completes, so one graph for the whole network
        holds every layer's activations to the end.
        """
        compiled = self._fns.get(key)
        if compiled is None:
            compiled = self._fns[key] = mx.compile(fn)
        out = compiled(*args)
        mx.eval(out)
        return out

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32, lazy) of the raw float32 pair `[1, 3, H, W]` (0..255, any size)."""
        h, w = left.shape[2], left.shape[3]
        x = self._run("in", self._input, left, right)
        skips = [x]
        for i in range(self.n):
            x = self._run(f"down{i}", functools.partial(self._down, i=i), x)
            skips.append(x)
        x = self._run("bottom", self._bottom, x)
        feats = [x]
        for i in range(self.n - 1, 0, -1):
            x = self._run(f"upm{i}", functools.partial(self._up_merge, i=i), x, skips[i])
            x = self._run(f"upc{i}", functools.partial(self._up_convs, i=i), x)
            feats.append(x)
        # the full-resolution level one image at a time: its transients are 4x the output, the pair's would be 8x
        halves = []
        for j in range(2):
            y = self._run("upm0", functools.partial(self._up_merge, i=0), x[j : j + 1], skips[0][j : j + 1])
            halves.append(self._run("upc0", functools.partial(self._up_convs, i=0), y))
        del skips, x, y
        fl, fr = [f[:1] for f in feats], [f[1:] for f in feats]
        fl.append(halves[0])
        fr.append(halves[1])
        hyp = mx.zeros(())
        for k in range(len(self.levels)):
            ref = fl[max(k - 2, 0)]
            hyp = self._run(f"level{k}", functools.partial(self._level, k), hyp, fl[k], fr[k], ref)
        base = len(self.levels) - 3
        for r in range(len(self.refine)):
            x = self._run(f"rh{r}", functools.partial(self._refine_head, r), hyp, fl[base + r], fr[base + r])
            blocks = len(self.refine[r]["net"][1])
            for lo in range(0, blocks, REFINE_BLOCKS):
                part = functools.partial(self._refine_blocks, r, lo, lo + REFINE_BLOCKS)
                x = self._run(f"rb{r}.{lo}", part, x)
            hyp = self._run(f"rt{r}", functools.partial(self._refine_tail, r), x, hyp)
        return hyp[:, :h, :w, 0]
