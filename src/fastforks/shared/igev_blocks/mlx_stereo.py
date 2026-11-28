"""The IGEV-family stereo frame on Apple MLX (`MlxIgevBase`): features, volume, lookups, update loop, upsampling.

The maths of the families' `forward_fast`, channels last. A family supplies its context (`_context`) and one update
(`_iteration`); everything else is shared: the MobileNetV2 trunk with the 1/4 stems, the group-wise volume through its
gate and hourglass, the fp32 initial disparity, the geometry / correlation lookups and the convex upsampling.

The run dtype covers what `forward_fast` runs in fp16: the convolutions, the volumes and the GRUs. Float32:
instance-norm statistics, the softmax and regression of the initial disparity, the geometry volume and its lookups, the
disparity state and the upsampling. Stages are evaluated one at a time, so only one stage's temporaries are alive; each
iteration is one compiled function, the next one queued before the previous one is waited on.
"""

from __future__ import annotations

from typing import Any

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import gwc_volume

from . import mlx_blocks as bl

__all__ = ["MlxIgevBase", "MlxIgevUpdate"]

GROUPS = 8  # the group-wise correlation's groups (`forward_fast`)


class MlxIgevBase:
    """`forward_fast` of an IGEV-family `net` (float32 weights) in `dtype` (the name), `iters` updates."""

    mask_scale = 1.0  # `mask_feat_4` scale (Selective-IGEV: 0.25)

    def __init__(self, net: Any, dtype: str, iters: int) -> None:  # noqa: ANN401 - the family's `IGEVStereo`
        """Copy the shared weights of `net` to MLX; `_family(net, w)` copies the rest."""
        self.dtype, self.iters = getattr(mx, dtype), iters
        args = net.args
        self.depth, self.radius, self.levels = args.max_disp // 4, args.corr_radius, args.corr_levels
        w = bl.Weights(self.dtype)
        self.feature = w.feature(net.feature)
        self.stem_2 = (w.basic(net.stem_2[0]), w.conv(net.stem_2[1]))
        self.stem_4 = (w.basic(net.stem_4[0]), w.conv(net.stem_4[1]))
        if hasattr(net, "desc"):  # the group-wise families' matching head (GREAT-IGEV: its own `_match`)
            self.conv, self.desc = w.basic(net.conv), w.conv(net.desc)
            self.classifier = w.conv(net.classifier)
        if hasattr(net, "cost_agg"):  # one volume (IGEV++: three, its own `_geometry`)
            self.corr_stem, self.corr_att = w.basic(net.corr_stem), w.att(net.corr_feature_att)
            self.hg = w.hourglass(net.cost_agg)
        ub = net.update_block
        enc = ub.encoder
        self.enc = [w.conv(getattr(enc, n)) for n in ("convc1", "convc2", "convd1", "convd2", "conv")]
        self.head = (w.conv(ub.disp_head.conv1), w.conv(ub.disp_head.conv2))
        self.mask = w.conv(ub.mask_feat_4[0])
        self.spx2, self.spx = w.conv2x(net.spx_2_gru), w.deconv(net.spx_gru[0])
        self._family(net, w)
        mx.eval(bl.arrays(self.__dict__))
        self._step = mx.compile(self._iteration)
        self._volume = mx.compile(self._cost)

    def _family(self, net: Any, w: bl.Weights) -> None:  # noqa: ANN401
        """Copy the family's context encoder and GRUs."""
        raise NotImplementedError

    def _context(self, im: mx.array) -> tuple[list[mx.array], list[mx.array]]:
        """The initial hidden states (1/4 first) and the constant inputs of `_iteration` of an NHWC image."""
        raise NotImplementedError

    def _iteration(self, *state: mx.array) -> tuple[mx.array, ...]:
        """`(*nets, disp, *constants, *lookups)` -> `(*nets, disp)`: one update and the disparity step."""
        raise NotImplementedError

    # shared pieces ----------------------------------------------------------------------------------------------------

    def _stem(self, x: mx.array, m: tuple[Any, bl.Conv]) -> mx.array:
        """`BasicConv_IN -> Conv2d -> InstanceNorm2d -> ReLU`."""
        return bl.relu(bl.inorm(bl.conv(bl.basic(x, m[0]), m[1])))

    def _stem4(self, im: mx.array) -> tuple[mx.array, mx.array]:
        """`stem_2` and `stem_4` of an NHWC image."""
        s2 = self._stem(im, self.stem_2)
        return s2, self._stem(s2, self.stem_4)

    def _match(self, f4: mx.array) -> mx.array:
        return bl.conv(bl.basic(f4, self.conv), self.desc)

    def _cost(self, match_l: mx.array, match_r: mx.array, *feats: mx.array) -> tuple[mx.array, mx.array]:
        """The group-wise volume through its stem, gate and hourglass: (geometry volume, initial disparity)."""
        vol = bl.basic(gwc_volume(match_l, match_r, self.depth, GROUPS), self.corr_stem)
        vol = bl.gate(vol, feats[0], self.corr_att)
        geo = bl.hourglass(vol, list(feats), self.hg)  # [1, D, H, W, 8]
        logits = bl.conv(geo, self.classifier)[..., 0].astype(mx.float32)  # [1, D, H, W]
        prob = mx.softmax(logits, axis=1)
        disp = (prob * mx.arange(self.depth, dtype=mx.float32)[:, None, None]).sum(axis=1)
        return geo, disp[..., None]  # [1, H, W, 1]

    def _geometry(
        self, match_l: mx.array, match_r: mx.array, feats: list[mx.array]
    ) -> tuple[list[mx.array], mx.array, list[mx.array]]:
        """The lookup state, the initial disparity and the extra constants of `_iteration` (evaluated)."""
        geo, disp = self._volume(match_l, match_r, *feats)
        mx.eval(geo, disp)
        lookups = bl.GeoLookup(match_l, match_r, geo, self.levels, self.radius, self.dtype).state()
        mx.eval(lookups)
        return lookups, disp, []

    def _motion(self, disp: mx.array, lookups: list[mx.array]) -> mx.array:
        """`BasicMotionEncoder` of the float32 disparity and its lookups: `[motion (127) | disparity]`."""
        return self._encode(bl.GeoLookup.lookup(disp, lookups).astype(self.dtype), disp)

    def _encode(self, corr: mx.array, disp: mx.array) -> mx.array:
        """The motion encoder of lookup features `corr` (run dtype) and the float32 disparity."""
        dh = disp.astype(self.dtype)
        c1, c2, d1, d2, cv = self.enc
        cor = bl.relu(bl.conv(bl.relu(bl.conv(corr, c1)), c2))
        flo = bl.relu(bl.conv(bl.relu(bl.conv(dh, d1)), d2))
        return mx.concatenate([bl.relu(bl.conv(mx.concatenate([cor, flo], axis=-1), cv)), dh], axis=-1)

    def _delta(self, net0: mx.array) -> mx.array:
        """`disp_head`, float32."""
        return bl.conv(bl.relu(bl.conv(net0, self.head[0])), self.head[1]).astype(mx.float32)

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32) of a padded float32 pair `[1, 3, H, W]` (multiples of 32)."""
        im1, im2 = ((2 * (x / 255.0) - 1.0).transpose(0, 2, 3, 1).astype(self.dtype) for x in (left, right))
        # right view first: only its 1/4 matching features are kept
        match_r = self._match(mx.concatenate([bl.feature(im2, self.feature)[0], self._stem4(im2)[1]], axis=-1))
        mx.eval(match_r)
        feats = bl.feature(im1, self.feature)
        stem_2x, stem_4x = self._stem4(im1)
        feats[0] = mx.concatenate([feats[0], stem_4x], axis=-1)
        match_l = self._match(feats[0])
        mx.eval(match_l, stem_2x, feats)
        lookups, disp, extra = self._geometry(match_l, match_r, feats)
        del feats, match_l, match_r
        nets, constants = self._context(im1)
        constants = [*constants, *extra]
        mx.eval(nets, constants)
        state = (*nets, disp)
        prev = None
        for _ in range(self.iters):
            state = self._step(*state, *constants, *lookups)
            mx.async_eval(state)
            if prev is not None:
                mx.eval(prev)
            prev = state
        net0, disp = state[0], state[-1]
        mask = bl.relu(bl.conv(net0, self.mask))
        if self.mask_scale != 1.0:
            mask = mask * self.mask_scale
        x = bl.basic(mx.concatenate([bl.basic(mask, self.spx2[0], transposed=True), stem_2x], axis=-1), self.spx2[1])
        prob = mx.softmax(bl.deconv(x, self.spx).astype(mx.float32), axis=-1)  # [1, 4H, 4W, 9]
        out = bl.context_upsample(disp[0, :, :, 0] * 4.0, prob)
        mx.eval(out)
        return out


class MlxIgevUpdate(MlxIgevBase):
    """IGEV-Stereo's context encoder (`context_zqr_convs`) and 3-level `BasicMultiUpdateBlock` (GREAT-IGEV's too)."""

    def _family(self, net: Any, w: bl.Weights) -> None:  # noqa: ANN401 - `IGEVStereo`
        self.cnet = w.context(net.cnet, net.context_zqr_convs)
        ub = net.update_block
        self.grus = [w.gru(ub.gru04), w.gru(ub.gru08), w.gru(ub.gru16)]

    def _context(self, im: mx.array) -> tuple[list[mx.array], list[mx.array]]:
        context = bl.context(im, self.cnet)
        return [n for n, _ in context], [c for _, cs in context for c in cs]

    def _iteration(self, *state: mx.array) -> tuple[mx.array, ...]:
        """One update (`BasicMultiUpdateBlock`, three GRU levels) and the disparity step."""
        net0, net1, net2, disp = state[:4]
        ctx, lookups = state[4:10], list(state[10:])
        net2 = bl.gru(net2, list(ctx[4:6]), bl.pool2x(net1), self.grus[2])
        x = mx.concatenate([bl.pool2x(net0), bl.interp(net2, net1.shape[1:3])], axis=-1)
        net1 = bl.gru(net1, list(ctx[2:4]), x, self.grus[1])
        x = mx.concatenate([self._motion(disp, lookups), bl.interp(net1, net0.shape[1:3])], axis=-1)
        net0 = bl.gru(net0, list(ctx[0:2]), x, self.grus[0])
        return net0, net1, net2, disp + self._delta(net0)
