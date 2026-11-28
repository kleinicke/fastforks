"""Cas-GwcNet on Apple MLX, written in `mlx.core` operations (`MlxCasGwcNet(net, dtype, vol_scale)`).

`model.forward` (eval) of the vendored network as one compiled function, channels last, the batch norms folded into the
convolutions (`fastforks.shared.mlx_stereo`; the feature trunk and the 3-D hourglass are the ones GwcNet uses):

- both images run through the 2-D features as one batch; `gc` builds the 320-channel GwcNet features plus a narrow
  concatenation head per stage, `c` the PSMNet-style pyramid pooling (64 / 32 / 16 / 8 px cells) at 1/4 and a feature
  pyramid up to 1/2;
- every stage builds its volume one hypothesis plane at a time: the right features are sampled at `x - d` (align
  corners, zeros outside) by a two-tap gather in float32, the group-wise correlation is taken in float32 (the products
  reach float16's limit) and the plane is stored times `vol_scale` (folded into the first batch norm by the runner);
- the next stage's hypotheses come from the min / max of the previous disparity in a 3 x 3 neighbourhood at 1/4
  resolution, interpolated trilinearly (taps built per axis); each stage ends in a trilinear upsample of the
  logits, a softmax over the hypotheses and their expectation, all in float32.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import trilinear
from fastforks.shared import mlx_stereo
from fastforks.shared.mlx_seg import resize
from fastforks.shared.mlx_stereo import Weights, conv2d, conv3d, deconv3d, relu, resnet

if TYPE_CHECKING:
    from .net.gwcnet import GwcNet
    from .net.psmnet import PSMNet

__all__ = ["LAYOUT", "SOURCES", "MlxCasGwcNet"]

LAYOUT = "cas_gwcnet-v1"
SOURCES = (Path(__file__), Path(mlx_stereo.__file__))
_POOLS = (("branch4", 8), ("branch3", 16), ("branch2", 32), ("branch1", 64))  # concatenation order of upstream


def _pool(x: mx.array, k: int) -> mx.array:
    """`AvgPool2d(k, stride=k)` of NHWC `x` (partial cells at the border dropped)."""
    n, h, w, c = x.shape
    x = x[:, : h // k * k, : w // k * k]
    return x.reshape(n, h // k, k, w // k, k, c).mean(axis=(2, 4))


def _up2(x: mx.array) -> mx.array:
    """Nearest-neighbour x2 of an NHWC map."""
    return mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)


def _lerp(x: mx.array, size: int, axis: int) -> mx.array:
    """Linear resize of `x` along `axis` to `size` samples (half-pixel centres, clamped, as `align_corners=False`)."""
    n = x.shape[axis]
    if n == size:
        return x
    src = np.maximum((np.arange(size) + 0.5) * (n / size) - 0.5, 0.0).astype(np.float32)
    i0 = np.minimum(src.astype(np.int32), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    shape = [1] * x.ndim
    shape[axis] = size
    frac = mx.array((src - i0).reshape(shape))
    a, b = mx.take(x, mx.array(i0), axis=axis), mx.take(x, mx.array(i1), axis=axis)
    return a + (b - a) * frac


def _pool3(x: mx.array, *, sign: float) -> mx.array:
    """`max_pool2d(sign * x, kernel_size=3, stride=1, padding=1)` of `[B, H, W, 1]` (the border padded with -inf)."""
    h, w = x.shape[1:3]
    xp = mx.pad(sign * x, [(0, 0), (1, 1), (1, 1), (0, 0)], constant_values=-float("inf"))
    taps = [xp[:, i : i + h, j : j + w] for i in range(3) for j in range(3)]
    return mx.max(mx.stack(taps), axis=0)


class MlxCasGwcNet:
    """The network `net` (eval, float32 weights) in `dtype` (the name): one compiled function per image size."""

    def __init__(self, net: GwcNet | PSMNet, dtype: str, vol_scale: float) -> None:
        """Copy the weights of `net` to MLX; the volumes are stored times `vol_scale`."""
        self.dtype = getattr(mx, dtype)
        self.vol_scale = vol_scale
        w = Weights(self.dtype)
        self.max_disp = int(net.maxdisp)
        self.ndisps = [int(n) for n in net.ndisps]
        self.intervals = [float(v) for v in net.disp_interval_pixel]
        self.ns = int(net.ns_size)
        self.scales = [int(net.stage_infos[f"stage{i + 1}"]["scale"]) for i in range(len(self.ndisps))]
        self.gc = hasattr(net, "num_groups")
        self.groups = [int(g) for g in cast("list[int]", net.num_groups)] if self.gc else [0] * len(self.ndisps)
        f: Any = net.feature_extraction
        self.first = [w.dilated(f.firstconv_a[0]), w.dilated(f.firstconv_a[2])]
        self.down = w.dilated(f.firstconv_b[0])
        self.layers = [[w.residual(b) for b in getattr(f, f"layer{i}")] for i in (1, 2, 3, 4)]
        self.inner = [(*w.conv(getattr(f, f"inner{i}")), 1) for i in range(1, len(self.ndisps))]
        self.out = [(*w.conv(getattr(f, f"out{i}")), 1) for i in range(2, len(self.ndisps) + 1)]
        if self.gc:
            self.head = (w.dilated(f.out1_cat[0]), (*w.conv(f.out1_cat[2]), 1))
            self.out_cat = [(*w.conv(getattr(f, f"out{i}_cat")), 1) for i in range(2, len(self.ndisps) + 1)]
        else:
            self.branches = [w.dilated(getattr(f, name)[1]) for name, _ in _POOLS]
            self.head = (w.dilated(f.inner0[0]), (*w.conv(f.inner0[2]), 1))
            self.last = (*w.conv(f.lastconv), 1)
        self.agg = []
        aggs: list[Any] = list(net.cost_agg)
        for agg in aggs:
            self.agg.append(
                {
                    "dres0": [w.cb(agg.dres0[i]) for i in (0, 2)],
                    "dres1": [w.cb(agg.dres1[i]) for i in (0, 2)],
                    "hg": [w.hourglass3(h) for h in (agg.dres2, agg.dres3, agg.dres4)],
                    "classif": (w.cb(agg.classif3[0]), w.conv(agg.classif3[2])),
                }
            )
        mx.eval(self._arrays())
        self._compiled = mx.compile(self._forward)

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

    def _features(self, x: mx.array) -> list[Any]:
        """Per stage the (left, right)-batched features: `(gwc, concat)` arrays for `gc`, one array for `c`."""
        s1 = relu(conv2d(relu(conv2d(x, self.first[0])), self.first[1]))
        outs = resnet(relu(conv2d(s1, self.down)), [], self.layers)
        s2 = outs[0]
        stages: list[Any] = []
        if self.gc:
            intra = mx.concatenate(outs[1:], axis=-1)
            stages.append((intra, conv2d(relu(conv2d(intra, self.head[0])), self.head[1])))
        else:
            raw, skip = outs[1], outs[3]
            h, w = skip.shape[1:3]
            pooled = [
                resize(relu(conv2d(_pool(skip, k), c)), (h, w)) for (_, k), c in zip(_POOLS, self.branches, strict=True)
            ]
            intra = conv2d(relu(conv2d(mx.concatenate([raw, skip, *pooled], axis=-1), self.head[0])), self.head[1])
            stages.append(conv2d(intra, self.last))
        skips = [s2, s1]
        for i in range(1, len(self.ndisps)):
            intra = _up2(intra) + conv2d(skips[i - 1], self.inner[i - 1])
            feat = conv2d(intra, self.out[i - 1])
            stages.append((feat, conv2d(feat, self.out_cat[i - 1])) if self.gc else feat)
        return stages

    def _first_hypotheses(self, nd: int, scale: int) -> tuple[mx.array, mx.array]:
        """Stage-1 hypotheses `[1, nd, 1, 1]` and their downsampling to the volume's `nd / scale` planes."""
        lo = max(0.0, -nd / 2 * self.intervals[0])
        hi = min(lo + (nd - 1) * self.intervals[0], float(self.max_disp))
        full = lo + mx.arange(nd, dtype=mx.float32).reshape(1, nd, 1, 1) * ((hi - lo) / (nd - 1))
        return full, _lerp(full / scale, nd // scale, 1)

    def _next_hypotheses(self, pred: mx.array, stage: int) -> tuple[mx.array, mx.array]:
        """Later-stage hypotheses `[B, nd, H, W]` from the previous disparity and their volume-resolution version."""
        _, h, w = pred.shape
        nd, scale = self.ndisps[stage], self.scales[stage]
        n = nd // 4
        s = resize(pred[..., None] / 4.0, (h // 4, w // 4))
        dmin = mx.abs(_pool3(s, sign=-1.0))
        dmax = _pool3(s, sign=1.0)
        comp = mx.maximum(n * self.intervals[stage] - mx.abs(dmax - dmin), 0.0) / 2.0
        lo = mx.clip(dmin - comp, 0.0, float(self.max_disp))
        hi = mx.clip(dmax + comp, 0.0, float(self.max_disp))
        steps = mx.arange(n, dtype=mx.float32).reshape(1, n, 1, 1)
        quarter = (lo.transpose(0, 3, 1, 2) + steps * ((hi - lo) / (n - 1)).transpose(0, 3, 1, 2)) * 4.0
        full = quarter
        for axis, size in ((1, nd), (2, h), (3, w)):
            full = _lerp(full, size, axis)
        low = full / scale
        for axis, size in ((1, nd // scale), (2, h // scale), (3, w // scale)):
            low = _lerp(low, size, axis)
        return full, low

    def _volume(self, fl: Any, fr: Any, low: mx.array, groups: int) -> mx.array:  # noqa: ANN401 - arrays or tuples
        """Upstream's cost volume times `vol_scale`, `[1, d, h, w, C]`, one hypothesis plane at a time."""
        xg = mx.zeros(())
        if self.gc:
            xg, xc = fl[0].astype(mx.float32), fl[1].astype(mx.float32)
            y = mx.concatenate([fr[0], fr[1]], axis=-1).astype(mx.float32)
        else:
            xc, y = fl.astype(mx.float32), fr.astype(mx.float32)
        b, h, wd, _ = xc.shape
        xs = mx.arange(wd, dtype=mx.float32)[None, None, :]
        planes = []
        for j in range(low.shape[1]):
            dj = mx.broadcast_to(low[:, j], (b, h, wd))
            p = xs - dj
            x0 = mx.floor(p)
            a = p - x0
            i0 = mx.clip(x0, 0, wd - 1).astype(mx.int32)[..., None]
            i1 = mx.clip(x0 + 1, 0, wd - 1).astype(mx.int32)[..., None]
            m0 = (((x0 >= 0) & (x0 <= wd - 1)).astype(mx.float32) * (1.0 - a))[..., None]
            m1 = (((x0 >= -1) & (x0 <= wd - 2)).astype(mx.float32) * a)[..., None]
            warped = mx.take_along_axis(y, i0, axis=2) * m0 + mx.take_along_axis(y, i1, axis=2) * m1
            if self.gc:
                cg = xg.shape[-1]
                valid = (xs >= dj).astype(mx.float32)[..., None]
                corr = (xg * warped[..., :cg]).reshape(b, h, wd, groups, -1).mean(axis=-1) * valid
                parts = [corr, xc * valid, warped[..., cg:]]
            else:
                parts = [xc, warped]
            planes.append((mx.concatenate(parts, axis=-1) * self.vol_scale).astype(self.dtype))
        return mx.stack(planes, axis=1)

    @staticmethod
    def _hg(x: mx.array, m: dict[str, Any]) -> mx.array:
        low = conv3d(conv3d(x, m["conv1"]), m["conv2"])
        bottom = conv3d(conv3d(low, m["conv3"]), m["conv4"])
        up = relu(deconv3d(bottom, m["conv5"]) + conv3d(low, m["redir2"], act=False))
        return relu(deconv3d(up, m["conv6"]) + conv3d(x, m["redir1"], act=False))

    def _aggregate(self, volume: mx.array, m: dict[str, Any]) -> mx.array:
        cost = conv3d(conv3d(volume, m["dres0"][0]), m["dres0"][1])
        cost = conv3d(conv3d(cost, m["dres1"][0]), m["dres1"][1], act=False) + cost
        for h in m["hg"]:
            cost = self._hg(cost, h)
        return conv3d(conv3d(cost, m["classif"][0]), m["classif"][1], act=False)[..., 0]  # [B, d, h, w]

    @staticmethod
    def _regress(cost: mx.array, values: mx.array, height: int, width: int) -> mx.array:
        """Trilinear upsample of `[B, d, h, w]` logits to the hypotheses, softmax, expectation: `[B, H, W]` float32."""
        d, h, w = cost.shape[1:]
        nd = values.shape[1]
        up = trilinear(
            cost.astype(mx.float32)[:, None], (nd, height, width), (d / nd, h / height, w / width), align=False
        )
        return mx.sum(mx.softmax(up[:, 0], axis=1) * values, axis=1)

    def _forward(self, left: mx.array, right: mx.array) -> mx.array:
        both = mx.concatenate([left, right], axis=0).transpose(0, 2, 3, 1).astype(self.dtype)
        feats = self._features(both)
        height, width = left.shape[2], left.shape[3]
        pred = None
        for s, nd in enumerate(self.ndisps):
            full, low = self._first_hypotheses(nd, self.scales[s]) if pred is None else self._next_hypotheses(pred, s)
            f = feats[s]
            fl, fr = ((f[0][:1], f[1][:1]), (f[0][1:], f[1][1:])) if self.gc else (f[:1], f[1:])
            volume = self._volume(fl, fr, low, self.groups[s])
            pred = self._regress(self._aggregate(volume, self.agg[s]), full, height, width)
        assert pred is not None  # noqa: S101 - at least one stage
        return pred

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32, lazy) of a padded float32 pair `[1, 3, H, W]` (multiples of 32)."""
        return self._compiled(left, right)
