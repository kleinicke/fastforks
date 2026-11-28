"""CFNet on Apple MLX, written in `mlx.core` operations (`MlxCFNet(net, dtype)`): a padded pair -> disparity.

`cfnet.forward` (eval) of the vendored network, channels last, the batch norms folded into the convolutions
(`fastforks.shared.mlx_stereo`); every weight-derived array is read through the conversion cache.

- the 2-D features (Mish) are one batch of both images; the three fused cost volumes (1/8, 1/16, 1/32: group-wise
  correlation plus concatenation) are built from float32 features (Mish features reach ~2e4) and the 1/16 and 1/32
  filters (`FP32_MODULES`) stay float32 in a float16 run;
- the two refinement stages (1/4, 1/2) warp the right features at per-pixel disparity samples (integer gather plus a
  validity mask, the correlation in float32) and run in row strips with halos: one compiled function per stage,
  evaluated strip by strip so that the volume of a large input never exists whole.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import bilinear_nhwc, gwc_volume, mish
from fastforks.shared import mlx_stereo
from fastforks.shared.mlx_seg import resize
from fastforks.shared.mlx_stereo import Weights, arr, concat_volume, conv2d, deconv3d

if TYPE_CHECKING:
    from .net.cfnet import cfnet

__all__ = ["LAYOUT", "SOURCES", "MlxCFNet"]

LAYOUT = "cfnet-v1"
SOURCES = (Path(__file__), Path(mlx_stereo.__file__))
FP32_MODULES = ("dres0_5", "dres1_5", "dres0_6", "dres1_6")
STRIP, HALO = 256, 64  # stage rows per strip / halo rows (> the 3D stack's receptive field; multiples of 8)
_STAGES = {"s3": (3, 2, 14, 40), "s2": (2, 1, 10, 20)}  # feature level, scale, samples, correlation groups
_LEVELS = (2, 3, 4, 5, 6)


def _c3(x: mx.array, c: tuple[mx.array, mx.array, int, int], *, act: bool = True) -> mx.array:
    """A 3-D convolution (NDHWC, batch norm folded), Mish after it unless `act` is False."""
    w, b, stride, padding = c
    y = mx.conv3d(x, w, stride=stride, padding=padding) + b
    return mish(y) if act else y


def _pool(x: mx.array, kh: int, kw: int) -> mx.array:
    """`avg_pool2d(k, stride=k)` of NHWC `x` (partial cells at the border dropped)."""
    n, h, w, c = x.shape
    x = x[:, : h // kh * kh, : w // kw * kw]
    return x.reshape(n, h // kh, kh, w // kw, kw, c).mean(axis=(2, 4))


def _up2(x: mx.array) -> mx.array:
    """Nearest-neighbour x2."""
    return mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)


def _resize_aligned(x: mx.array, size: tuple[int, int]) -> mx.array:
    """Bilinear resize of NHWC `x` with `align_corners=True`."""
    h, w = x.shape[1:3]
    oh, ow = size
    if (h, w) == (oh, ow):
        return x
    return bilinear_nhwc(
        x, (oh, ow), ((h - 1) / (oh - 1) if oh > 1 else 0.0, (w - 1) / (ow - 1) if ow > 1 else 0.0), align=True
    )


def _walk(v: object, found: list[mx.array]) -> None:
    if isinstance(v, mx.array):
        found.append(v)
    elif isinstance(v, (list, tuple)):
        for x in v:
            _walk(x, found)
    elif isinstance(v, dict):
        _walk(list(v.values()), found)


def _stats(p: mx.array, ds: mx.array) -> tuple[mx.array, mx.array]:
    """Mean and standard deviation of `ds` under the distribution `p` over axis 1: `[B, h, W]` each."""
    pred = (p * ds).sum(axis=1)
    return pred, mx.sqrt((p * (ds - pred[:, None]) ** 2).sum(axis=1))


class MlxCFNet:
    """The network `net` (eval, float32 weights) in `dtype` (the name): compiled functions per image size."""

    def __init__(self, net: cfnet, dtype: str) -> None:
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        w, w32 = Weights(self.dtype), Weights(mx.float32)
        self.max_disp = int(net.maxdisp)
        f = net.feature_extraction
        self.first = [w.dilated(f.firstconv[i]) for i in (0, 2, 4)]
        self.layers = [w.residual(getattr(f, f"layer{i}")[0]) for i in (2, 3, 4, 5, 6)]
        self.pyramid = [w.conv(m.cbr_unit[0], m.cbr_unit[1]) for m in f.pyramid_pooling.path_module_list]  # ty: ignore[not-subscriptable]
        self.up = [w.dilated(getattr(f, f"upconv{i}")[1]) for i in (6, 5, 4, 3)]
        self.iconv = [w.dilated(getattr(f, f"iconv{i}")) for i in (5, 4, 3, 2)]
        self.heads = {
            (kind, i): (w.dilated(getattr(f, f"{kind}{i}")[0]), (*w.conv(getattr(f, f"{kind}{i}")[2]), 1))
            for kind in ("gw", "concat")
            for i in _LEVELS
        }
        self.dres0 = [w.cb(net.dres0[i]) for i in (0, 2)]
        self.dres1 = [w.cb(net.dres1[i]) for i in (0, 2)]
        self.coarse = {
            lvl: (
                [w32.cb(getattr(net, f"dres0_{lvl}")[i]) for i in (0, 2)],
                [w32.cb(getattr(net, f"dres1_{lvl}")[i]) for i in (0, 2)],
            )
            for lvl in (5, 6)
        }
        self.up_hg = self._hourglass_up(w, net.combine1)
        self.hg = self._hourglass(w, net.dres3)
        self.classif2 = (w.cb(net.classif2[0]), w.conv(net.classif2[2]))
        self.stages: dict[str, dict[str, Any]] = {}
        for s in _STAGES:
            self.stages[s] = {
                "c0": [w.cb(getattr(net, f"confidence0_{s}")[i]) for i in (0, 2)],
                "c1": [w.cb(getattr(net, f"confidence1_{s}")[i]) for i in (0, 2)],
                "c2": self._hourglass(w, getattr(net, f"confidence2_{s}")),
                "c3": self._hourglass(w, getattr(net, f"confidence3_{s}")),
                "classif": (
                    w.cb(getattr(net, f"confidence_classif1_{s}")[0]),
                    w.conv(getattr(net, f"confidence_classif1_{s}")[2]),
                ),
            }
        self.gamma = {
            s: arr((1,), mx.float32, lambda s=s: getattr(net, f"gamma_{s}").detach().float().numpy().reshape(1))
            for s in _STAGES
        }
        self.beta = {
            s: arr((1,), mx.float32, lambda s=s: getattr(net, f"beta_{s}").detach().float().numpy().reshape(1))
            for s in _STAGES
        }
        found: list[mx.array] = []
        _walk(list(vars(self).values()), found)
        mx.eval(found)
        self._front_c = mx.compile(self._front)
        self._strip_c = {s: mx.compile(partial(self._strip, s)) for s in _STAGES}
        self._range_c = {s: mx.compile(partial(self._range, s)) for s in _STAGES}

    @staticmethod
    def _deconv(w: Weights, seq: Any) -> tuple[mx.array, mx.array, bool]:  # noqa: ANN401 - torch module
        """`Sequential(ConvTranspose3d, BatchNorm3d)`: weight `[Cin, Cout, 3, 3, 3]`, bias, no activation."""
        return w.deconv3(SimpleNamespace(conv=seq[0], bn=seq[1], use_bn=True, relu=False))

    @classmethod
    def _hourglass(cls, w: Weights, h: Any) -> dict[str, Any]:  # noqa: ANN401
        out: dict[str, Any] = {k: w.cb(getattr(h, k)[0]) for k in ("conv1", "conv2", "conv3", "conv4")}
        out["conv5"], out["conv6"] = cls._deconv(w, h.conv5), cls._deconv(w, h.conv6)
        out["redir1"], out["redir2"] = w.cb(h.redir1), w.cb(h.redir2)
        return out

    @classmethod
    def _hourglass_up(cls, w: Weights, h: Any) -> dict[str, Any]:  # noqa: ANN401
        out: dict[str, Any] = {k: w.conv(getattr(h, k)) for k in ("conv1", "conv3")}
        out |= {k: w.cb(getattr(h, k)[0]) for k in ("conv2", "conv4", "combine1", "combine2")}
        out["conv8"], out["conv9"] = cls._deconv(w, h.conv8), cls._deconv(w, h.conv9)
        out["redir1"], out["redir2"] = w.cb(h.redir1), w.cb(h.redir2)
        return out

    @staticmethod
    def _hg(x: mx.array, m: dict[str, Any]) -> mx.array:
        low = _c3(_c3(x, m["conv1"]), m["conv2"])
        bottom = _c3(_c3(low, m["conv3"]), m["conv4"])
        up = mish(deconv3d(bottom, m["conv5"]) + _c3(low, m["redir2"], act=False))
        return mish(deconv3d(up, m["conv6"]) + _c3(x, m["redir1"], act=False))

    @staticmethod
    def _hg_up(x: mx.array, f4: mx.array, f5: mx.array, m: dict[str, Any]) -> mx.array:
        c1 = _c3(mx.concatenate([_c3(x, m["conv1"], act=False), f4], axis=-1), m["combine1"])
        c2 = _c3(c1, m["conv2"])
        c3 = _c3(mx.concatenate([_c3(c2, m["conv3"], act=False), f5], axis=-1), m["combine2"])
        c4 = _c3(c3, m["conv4"])
        c8 = mish(deconv3d(c4, m["conv8"]) + _c3(c2, m["redir2"], act=False))
        return mish(deconv3d(c8, m["conv9"]) + _c3(x, m["redir1"], act=False))

    def _features(self, x: mx.array) -> dict[tuple[str, int], mx.array]:
        for c in self.first:
            x = mish(conv2d(x, c))
        outs = []
        for c1, c2, down in self.layers:
            y = conv2d(mish(conv2d(x, c1)), c2)
            x = y + (x if down is None else conv2d(x, down))
            outs.append(x)
        l2, l3, l4, l5, l6 = outs
        h, w = l6.shape[1:3]
        m = min(h, w)
        sizes = [int(i * ((m - 2) / 3) + 2) for i in range(3)] + [m]
        ks = [(int(h / s), int(w / s)) for s in sizes][::-1]
        pp = l6
        for (kh, kw), c in zip(ks, self.pyramid, strict=True):
            pp = pp + 0.25 * resize(mish(conv2d(_pool(l6, kh, kw), (*c, 1))), (h, w))
        l6 = mish(pp / 2.0)
        d5 = mish(conv2d(mx.concatenate([l5, mish(conv2d(_up2(l6), self.up[0]))], -1), self.iconv[0]))
        d4 = mish(conv2d(mx.concatenate([l4, mish(conv2d(_up2(d5), self.up[1]))], -1), self.iconv[1]))
        d3 = mish(conv2d(mx.concatenate([l3, mish(conv2d(_up2(d4), self.up[2]))], -1), self.iconv[2]))
        d2 = mish(conv2d(mx.concatenate([l2, mish(conv2d(_up2(d3), self.up[3]))], -1), self.iconv[3]))
        src = {2: d2, 3: d3, 4: d4, 5: d5, 6: l6}
        feats = {}
        for lvl, s in src.items():
            for kind in ("gw", "concat"):
                a, b = self.heads[(kind, lvl)]
                feats[(kind, lvl)] = conv2d(mish(conv2d(s, a)), b)
        return feats

    def _volume(self, feats: dict[tuple[str, int], mx.array], level: int, disps: int) -> mx.array:
        """Group-wise correlation plus concatenation volume of `level` from float32 features, `[1, D, h, w, 52]`."""
        g, c = (feats[(k, level)].astype(mx.float32) for k in ("gw", "concat"))
        return mx.concatenate([gwc_volume(g[:1], g[1:], disps, 40), concat_volume(c[:1], c[1:], disps)], axis=-1)

    def _search(self, s: str, pred: mx.array, std: mx.array, size: tuple[int, int]) -> tuple[mx.array, mx.array]:
        """`pred -/+ ((gamma + 1) * std + beta)`, x2 and bilinearly resized to `size`: `[1, h, w, 1]` each."""
        g, b = self.gamma[s] + 1, self.beta[s]
        lo, hi = (pred - g * std - b) * 2, (pred + g * std + b) * 2
        return _resize_aligned(lo[..., None], size), _resize_aligned(hi[..., None], size)

    def _front(self, left: mx.array, right: mx.array) -> tuple[mx.array, ...]:
        """Features, the 1/8 cost-volume stack and the first search range: the features the stages read, `lo`, `hi`."""
        both = mx.concatenate([left, right], axis=0).transpose(0, 2, 3, 1).astype(self.dtype)
        feats = self._features(both)
        md = self.max_disp
        v4 = self._volume(feats, 4, md // 8).astype(self.dtype)
        c4 = _c3(_c3(v4, self.dres0[0]), self.dres0[1])
        c4 = _c3(_c3(c4, self.dres1[0]), self.dres1[1], act=False) + c4
        outs = []
        for lvl in (5, 6):
            d0, d1 = self.coarse[lvl]
            c = _c3(_c3(self._volume(feats, lvl, md // 2 ** (lvl - 1)), d0[0]), d0[1])
            c = _c3(_c3(c, d1[0]), d1[1], act=False) + c
            outs.append(c.astype(self.dtype))
        cost = self._hg(self._hg_up(c4, outs[0], outs[1], self.up_hg), self.hg)
        cost = _c3(_c3(cost, self.classif2[0]), self.classif2[1], act=False)[..., 0].astype(mx.float32)  # [1, D, h, w]
        p = mx.softmax(cost, axis=1)
        dv = mx.arange(md // 8, dtype=mx.float32)[None, :, None, None]
        pred, std = _stats(p, mx.broadcast_to(dv, p.shape))
        h, w = left.shape[2:]
        lo, hi = self._search("s3", pred, std, (h // 4, w // 4))
        return (*(feats[(k, lvl)] for lvl in (3, 2) for k in ("gw", "concat")), lo, hi)

    def _range(self, s: str, lo: mx.array, hi: mx.array) -> tuple[mx.array, mx.array]:
        """Upstream `generate_search_range` of stage `s` (`[1, h, w, 1]` in and out)."""
        _, scale, count, _ = _STAGES[s]
        top = self.max_disp // 2**scale - 1
        extra = mx.maximum(count + 1 - hi + lo, 0) / 2.0
        return mx.clip(lo - extra, 0, top), mx.clip(hi + extra, 0, top)

    def _strip(self, s: str, feats: tuple[mx.array, ...], lo: mx.array, hi: mx.array) -> tuple[mx.array, mx.array]:
        """One row strip of stage `s` (halo rows included): the disparity and its standard deviation, `[1, h, W]`.

        `feats`: the strip's left / right features (`g`: correlation, `c`: concatenation) as `gl, cl, gr, cr`; `lo`
        and `hi`: its search range.
        """
        fg, fc, rg, rc = feats
        _, _, count, groups = _STAGES[s]
        m = self.stages[s]
        lo, hi = lo[..., 0], hi[..., 0]
        mult = (hi - lo) / (count + 1)
        steps = mx.arange(1, count + 1, dtype=mx.float32)[None, :, None, None]
        ds = mx.concatenate([mx.floor(lo)[:, None], lo[:, None] + mult[:, None] * steps, mx.ceil(hi)[:, None]], axis=1)
        ds = ds.astype(mx.int32).astype(mx.float32)  # [1, D, h, W]
        d, h, wd = ds.shape[1:]
        x = mx.arange(wd, dtype=mx.float32) - ds
        idx = mx.clip(x, 0, wd - 1).astype(mx.int32)[..., None]
        mask = ((x >= 0) & (x <= wd - 1)).astype(mx.float32)[..., None]

        def warp(f: mx.array) -> mx.array:
            return mx.take_along_axis(f[:, None], idx, axis=3) * mask

        fg32 = fg.astype(mx.float32)[:, None]
        corr = (fg32 * warp(rg)).reshape(1, d, h, wd, groups, -1).mean(axis=-1)
        left_c = mx.broadcast_to(fc[:, None], (1, d, h, wd, fc.shape[-1]))
        vol = mx.concatenate([corr, left_c.astype(mx.float32), warp(rc), ds[..., None]], axis=-1).astype(self.dtype)
        cost = _c3(_c3(vol, m["c0"][0]), m["c0"][1])
        cost = _c3(_c3(cost, m["c1"][0]), m["c1"][1], act=False) + cost
        cost = self._hg(self._hg(cost, m["c2"]), m["c3"])
        cost = _c3(_c3(cost, m["classif"][0]), m["classif"][1], act=False)[..., 0].astype(mx.float32)
        p = mx.softmax(cost, axis=1)
        return _stats(p, ds)

    @staticmethod
    def _split(g: mx.array, c: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """The left and right maps of a level (the pair is one batch of two)."""
        return g[:1], c[:1], g[1:], c[1:]

    def _stage(self, s: str, feats: tuple[mx.array, ...], lo: mx.array, hi: mx.array) -> tuple[mx.array, mx.array]:
        lo, hi = self._range_c[s](lo, hi)
        rows = lo.shape[1]
        preds, stds = [], []
        for r0 in range(0, rows, STRIP):
            r1 = min(rows, r0 + STRIP)
            a, b = max(0, r0 - HALO), min(rows, r1 + HALO)
            pred, std = self._strip_c[s](tuple(f[:, a:b] for f in feats), lo[:, a:b], hi[:, a:b])
            pred, std = pred[:, r0 - a : r1 - a], std[:, r0 - a : r1 - a]
            mx.eval(pred, std)
            preds.append(pred)
            stds.append(std)
        return mx.concatenate(preds, axis=1), mx.concatenate(stds, axis=1)

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32, lazy) of a padded float32 pair `[1, 3, H, W]` (multiples of 64)."""
        h, w = left.shape[2:]
        out = self._front_c(left, right)
        mx.eval(out)
        g3, c3, g2, c2, lo, hi = out
        pred, std = self._stage("s3", self._split(g3, c3), lo, hi)
        lo, hi = self._search("s2", pred, std, (h // 2, w // 2))
        pred, _ = self._stage("s2", self._split(g2, c2), lo, hi)
        return _resize_aligned(pred[..., None] * 2, (h, w))[..., 0]
