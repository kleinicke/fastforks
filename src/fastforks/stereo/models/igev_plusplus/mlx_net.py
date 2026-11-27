"""IGEV++ on Apple MLX, written in `mlx.core` operations (`MlxIGEVPlusPlus(net, dtype, iters)`): pair -> disparity.

The maths of `portable.forward_fast`, channels last, in place of the export and graph translation the MLX runner
replayed before. The shared frame (`igev_blocks.mlx_stereo`) holds the features, the update loop and the upsampling;
this module adds the three range volumes (small / medium / large, each through its own hourglass, one evaluated at a
time), the per-pixel selective weights that blend their lookups, and the IGEV context and `ConvGRU`s.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import gwc_volume, linear_sample
from fastforks.shared.igev_blocks import mlx_blocks as bl
from fastforks.shared.igev_blocks.mlx_stereo import GROUPS, MlxIgevBase

__all__ = ["MlxIGEVPlusPlus"]

CONTEXT = 6  # constants of the three GRU levels: [cz | cr], cq each


def _flat(geo: mx.array) -> mx.array:
    """A geometry volume `[1, D, H, W, C]` as the float32 lookup rows `[H * W, C, D]`."""
    _, d, h, w, c = geo.shape
    return geo[0].transpose(1, 2, 3, 0).reshape(h * w, c, d).astype(mx.float32)


class MlxIGEVPlusPlus(MlxIgevBase):
    """`forward_fast` of `net` (a loaded IGEV++ `IGEVStereo`, float32 weights) in `dtype` (the name)."""

    def _family(self, net: Any, w: bl.Weights) -> None:  # noqa: ANN401 - `IGEVStereo`
        a = net.args
        self.ranges = (
            (a.s_disp_range, a.s_disp_interval),
            (a.m_disp_range, a.m_disp_interval),
            (a.l_disp_range, a.l_disp_interval),
        )
        self.hgs = [w.hourglass(h) for h in (net.cost_agg0, net.cost_agg1, net.cost_agg2)]
        self.patches = [self._patch(w, p) for p in (net.patch0, net.patch1)]
        self.disp_conv = [w.basic(m) for m in net.disp_conv]
        self.selective = (w.basic(net.selective_conv[0]), w.basic(net.selective_conv[1]), w.conv(net.selective_conv[2]))
        self.cnet = w.context(net.cnet, net.context_zqr_convs)
        ub = net.update_block
        self.geo_enc = [
            (w.conv(g.convg1), w.conv(g.convg2)) for g in (ub.geo_encoder0, ub.geo_encoder1, ub.geo_encoder2)
        ]
        self.grus = [w.gru(ub.gru04), w.gru(ub.gru08), w.gru(ub.gru16)]
        self._aggs = [mx.compile(lambda v, *f, i=i: self._aggregate(i, v, list(f))) for i in range(3)]

    @staticmethod
    def _patch(w: bl.Weights, m: Any) -> bl.Conv:  # noqa: ANN401 - `nn.Conv3d` (k, 1, 1), stride (k, 1, 1)
        k = int(m.kernel_size[0])
        weight = m.weight.detach().float().numpy().transpose(0, 2, 3, 4, 1)
        bias = w.a(np.zeros(weight.shape[0], np.float32))
        return cast("bl.Conv", (w.a(weight), bias, (k, 1, 1), 0))  # a tuple stride: `mx.conv3d` takes either

    def _aggregate(self, i: int, vol: mx.array, feats: list[mx.array]) -> tuple[mx.array, mx.array]:
        """Range `i` of the full volume through its hourglass: (geometry volume, float32 disparity `[1, H, W, 1]`)."""
        depth, interval = self.ranges[i]
        v = vol[:, :depth]  # the large range is the whole volume
        if i:
            v = bl.conv(v, self.patches[i - 1])  # medium / large: bins merged in pairs / fours
        hg = self.hgs[i]
        v = bl.gate(bl.basic(v, hg["conv0"]), feats[0], hg["feature_att_4"])
        geo = bl.hourglass(v, feats, hg)
        prob = mx.softmax(bl.conv(geo, self.classifier)[..., 0].astype(mx.float32), axis=1)
        values = mx.arange(0, depth, interval, dtype=mx.float32)[:, None, None]
        return geo, (prob * values).sum(axis=1)[..., None]

    def _geometry(
        self, match_l: mx.array, match_r: mx.array, feats: list[mx.array]
    ) -> tuple[list[mx.array], mx.array, list[mx.array]]:
        vol = gwc_volume(match_l, match_r, self.depth, GROUPS)
        mx.eval(vol)
        geos, aggs = [], []
        for agg in self._aggs:  # one range's temporaries at a time
            g, d = agg(vol, *feats)
            mx.eval(g, d)
            geos.append(g)
            aggs.append(d)
        del vol
        x = mx.concatenate(aggs, axis=-1).astype(self.dtype)
        for m in self.disp_conv:
            x = bl.basic(x, m)
        s0, s1, s2 = self.selective
        sw = mx.sigmoid(bl.conv(bl.basic(bl.basic(mx.concatenate([feats[0], x], axis=-1), s0), s1), s2))
        lookups = bl.GeoLookup(match_l, match_r, geos[0], self.levels, self.radius, self.dtype).state()
        lookups += [_flat(geos[1]), _flat(geos[2])]
        mx.eval(sw, lookups)
        return lookups, aggs[0], [sw]

    def _context(self, im: mx.array) -> tuple[list[mx.array], list[mx.array]]:
        context = bl.context(im, self.cnet)
        return [n for n, _ in context], [c for _, cs in context for c in cs]

    def _lookup(self, disp: mx.array, lookups: tuple[mx.array, ...]) -> list[mx.array]:
        """`FastGeoEncodingVolume(disp)`: geo0 (both levels), geo1, geo2, corr (both levels), NHWC, run dtype."""
        n = self.levels
        geo, corr, (coords, dx, g1, g2) = lookups[:n], lookups[n : 2 * n], lookups[2 * n :]
        h, w = disp.shape[1:3]
        dsp = disp.reshape(-1, 1)
        scales = [1.0 / 2**i for i in range(n)]
        outs = [
            mx.concatenate([linear_sample(geo[i], dsp * sc + dx) for i, sc in enumerate(scales)], axis=1),
            linear_sample(g1, dsp * 0.5 + dx),
            linear_sample(g2, dsp * 0.25 + dx),
            mx.concatenate(
                [linear_sample(corr[i], coords * sc - dsp * sc + dx) for i, sc in enumerate(scales)], axis=1
            ),
        ]
        return [o.reshape(1, h, w, -1).astype(self.dtype) for o in outs]

    def _iteration(self, *state: mx.array) -> tuple[mx.array, ...]:
        """One update (`BasicMultiUpdateBlock` with the blended range lookups) and the disparity step."""
        net0, net1, net2, disp = state[:4]
        ctx, sw, lookups = state[4 : 4 + CONTEXT], state[4 + CONTEXT], state[5 + CONTEXT :]
        net2 = bl.gru(net2, list(ctx[4:6]), bl.pool2x(net1), self.grus[2])
        x = mx.concatenate([bl.pool2x(net0), bl.interp(net2, net1.shape[1:3])], axis=-1)
        net1 = bl.gru(net1, list(ctx[2:4]), x, self.grus[1])
        *geo, corr = self._lookup(disp, lookups)
        feat = None
        for k, (g, (c1, c2)) in enumerate(zip(geo, self.geo_enc, strict=True)):
            f = sw[..., k : k + 1] * bl.conv(bl.relu(bl.conv(g, c1)), c2)
            feat = f if feat is None else feat + f
        motion = self._encode(mx.concatenate([feat, corr], axis=-1), disp)
        x = mx.concatenate([motion, bl.interp(net1, net0.shape[1:3])], axis=-1)
        net0 = bl.gru(net0, list(ctx[0:2]), x, self.grus[0])
        return net0, net1, net2, disp + self._delta(net0)
