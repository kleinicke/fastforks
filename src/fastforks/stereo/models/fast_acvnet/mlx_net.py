"""Fast-ACVNet / Fast-ACVNet+ on Apple MLX, in `mlx.core` operations (`MlxFastACV(net, dtype)`): a pair -> disparity.

`model.forward` written out channels last (NHWC maps, NDHWC volumes) with the batch norms folded into the
convolutions; no exported graph, so no trace on a first call.

- the MobileNetV2 depthwise layers and Fast-ACVNet's per-slice `patch` convolution are the shared `depthwise` kernel;
- the normalized correlation volumes are the shared `gwc_volume` kernel on group-normalized features;
- Fast-ACVNet+'s 3-D stem (1 -> 8 channels) has its depth taps folded into channels (`mx.conv3d` is slow at one input
  channel) and the transposed 3-D convolutions run as one small convolution per output phase (`mlx_graph`'s);
- the five propagated disparity samples are replicate-padded shifts, their matching strength the shared `warp_dot`;
- the top-24 attention candidates are an arg-partition over the disparities (kept last), sorted by index as upstream;
  the concatenation volume gathers the right features at the integer candidates (zeros left of the image);
- everything up to the top-k (features, correlation, attention hourglass, propagation) runs in float32 also for a
  float16 run (`[modes.portable]` in the card: in half one cell of a golden picks other disparities); the stages after
  it run in `dtype`, the regressions and the superpixel upsampling in float32.

Two compiled functions (to the top-k, after it); the first is evaluated before the second is built, which bounds the
transients to one half's.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import depthwise, gwc_volume, trilinear, warp_dot
from fastforks.shared import mlx_stereo
from fastforks.shared.mlx_graph import _conv_transpose3d_phases
from fastforks.shared.mlx_stereo import Conv as _Conv
from fastforks.shared.mlx_stereo import Weights, folded
from fastforks.shared.mlx_stereo import leaky as _leaky
from fastforks.shared.mlx_stereo import relu as _relu
from fastforks.shared.mlx_stereo import relu6 as _relu6

if TYPE_CHECKING:
    from .net.fast_acv import Fast_ACVNet
    from .net.fast_acv_plus import Fast_ACVNet_plus

__all__ = ["LAYOUT", "SOURCES", "MlxFastACV"]

LAYOUT = "fastacv-v1"
SOURCES = (Path(__file__), Path(mlx_stereo.__file__))

TOP_K = 24  # upstream's attention candidates per pixel (`model.TOP_K`)
EPS = 1e-05  # upstream's L2-norm epsilon
# the five propagation taps of upstream's one-hot 3 x 3 filters, (row, column) in the replicate-padded map
_TAPS = ((0, 0), (1, 1), (2, 2), (2, 0), (0, 2))


def _shifts(x: mx.array) -> list[mx.array]:
    """Upstream `Propagation(_prob)`: the five taps of `x` `[..., H, W, C]`, replicate-padded by one pixel."""
    h, w = x.shape[-3], x.shape[-2]
    rows = mx.concatenate([x[..., :1, :, :], x, x[..., -1:, :, :]], axis=-3)
    p = mx.concatenate([rows[..., :1, :], rows, rows[..., -1:, :]], axis=-2)
    return [p[..., i : i + h, j : j + w, :] for i, j in _TAPS]


def _hourglass_weights(w: Weights, m: Any, levels: int) -> dict[str, Any]:  # noqa: ANN401 - a vendored hourglass
    """An hourglass: `levels` strided pairs, their gates, the transposed convs, the merges and their gates."""
    out: dict[str, Any] = {}
    for k in range(1, levels + 1):
        out[f"conv{k}"] = [w.basic(c) for c in getattr(m, f"conv{k}")]
        out[f"up{k}"] = w.deconv3(getattr(m, f"conv{k}_up" if hasattr(m, f"conv{k}_up") else f"conv{k}_up_"))
    for name, sub in m.named_children():
        if name.startswith("feature_att"):
            out[name] = w.att(sub)
        elif name.startswith("agg"):
            out[name] = [w.basic(c) for c in sub]
    if hasattr(m, "conv_final"):
        out["final"] = w.conv(m.conv_final)
    return out


class MlxFastACV:
    """The network `net` (eval, float32 CPU weights) in `dtype` (the name): two compiled halves per image size."""

    def __init__(self, net: Fast_ACVNet | Fast_ACVNet_plus, dtype: str) -> None:
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        self.plus = not hasattr(net, "patch")
        self.maxdisp = int(net.maxdisp)
        pre, post = Weights(mx.float32), Weights(self.dtype)
        f = net.feature
        self.stem = pre.conv(f.conv_stem, f.bn1)
        self.blocks = [[pre.block(b) for stage in getattr(f, f"block{i}") for b in stage] for i in range(5)]
        up = net.feature_up
        self.up = [pre.conv2x(m) for m in (up.deconv32_16, up.deconv16_8, up.deconv8_4)]
        self.up4 = pre.basic(up.conv4)
        self.stem_2 = pre.stem(net.stem_2)
        self.stem_4 = pre.stem(net.stem_4)
        if self.plus:
            p: Any = net
            self.match = (pre.basic(p.conv), pre.conv(p.desc))
            fold = functools.cache(lambda: folded(p.corr_stem.conv, p.corr_stem.bn))  # [8, 1, 3, 3, 3]
            self.corr_stem = (
                pre.a((8, 3, 3, 3), lambda: fold()[0][:, 0].transpose(0, 2, 3, 1)),
                pre.a((8,), lambda: fold()[1]),
            )
            self.corr_att = pre.att(p.corr_feature_att_4)
            self.hg_att = _hourglass_weights(pre, p.hourglass_att, 3)
        else:
            n: Any = net
            self.stem_8 = pre.stem(n.stem_8)
            self.patch = pre.a((3, 3, 12), lambda: n.patch.weight.detach().float().numpy()[:, 0, 0].transpose(1, 2, 0))
            self.corr_att = pre.att(n.corr_feature_att_8)
            self.hg_att = _hourglass_weights(pre, n.hourglass_att, 2)
            self.gamma = float(pre.a((1,), lambda: n.gamma.detach().float().numpy().reshape(1)).item())
            self.beta = float(pre.a((1,), lambda: n.beta.detach().float().numpy().reshape(1)).item())
        self.concat_feature = (post.basic(net.concat_feature[0]), post.conv(net.concat_feature[1]))
        self.concat_stem = post.basic(net.concat_stem)
        self.concat_att = post.att(net.concat_feature_att_4)
        self.hg = _hourglass_weights(post, net.hourglass, 2)
        self.spx_4 = post.stem(net.spx_4)
        self.spx_2 = post.conv2x(net.spx_2)
        self.spx = post.deconv(net.spx[0])
        mx.eval(self._arrays())
        self._pre = mx.compile(self._attention)
        self._post = mx.compile(self._disparity)

    def _arrays(self) -> list[mx.array]:
        found: list[mx.array] = []

        def walk(v: object) -> None:
            if isinstance(v, mx.array):
                found.append(v)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x)

        walk(list(vars(self).values()))
        return found

    # layers ----------------------------------------------------------------------------------------------------------

    @staticmethod
    def _c(x: mx.array, conv: _Conv) -> mx.array:
        w, b, stride, padding = conv
        return mx.conv2d(x, w, stride=stride, padding=padding) + b

    @staticmethod
    def _c3(x: mx.array, conv: _Conv, *, act: bool = True) -> mx.array:
        w, b, stride, padding = conv
        y = mx.conv3d(x, w, stride=stride, padding=padding) + b
        return _leaky(y) if act else y

    @staticmethod
    def _d(x: mx.array, conv: _Conv) -> mx.array:
        w, b, stride, padding = conv
        return mx.conv_transpose2d(x, w, stride=stride, padding=padding) + b

    @staticmethod
    def _d3(x: mx.array, up: tuple[mx.array, mx.array, bool]) -> mx.array:
        w, b, act = up
        y = _conv_transpose3d_phases(x, w, (2, 2, 2), (1, 1, 1), (0, 0, 0))
        if y is None:  # pragma: no cover - the pair is padded to 32s, every volume level halves exactly
            msg = f"MlxFastACV: the cost volume {tuple(x.shape[1:4])} does not halve evenly"
            raise ValueError(msg)
        y = y + b
        return _leaky(y) if act else y

    def _up(self, x: mx.array, rem: mx.array, m: tuple[_Conv, _Conv]) -> mx.array:
        """`Conv2x` with a transposed convolution (its output has the skip's size: the pair is padded to 32s)."""
        return _leaky(self._c(mx.concatenate([_leaky(self._d(x, m[0])), rem], axis=-1), m[1]))

    def _stem(self, x: mx.array, m: tuple[_Conv, _Conv]) -> mx.array:
        return _relu(self._c(_leaky(self._c(x, m[0])), m[1]))

    def _mobile(self, x: mx.array, blocks: list[tuple[Any, ...]]) -> mx.array:
        return mlx_stereo.mobile(x, blocks, self._c)

    def _gate(self, cost: mx.array, im: mx.array, att: tuple[_Conv, _Conv]) -> mx.array:
        """`channelAtt`: `cost` `[B, D, H, W, C]` gated by the left image's features `[B, H, W, K]`."""
        g = mx.sigmoid(self._c(_leaky(self._c(im, att[0])), att[1])).astype(cost.dtype)
        return cost * g[:, None]

    def _hourglass(self, x: mx.array, imgs: list[mx.array], m: dict[str, Any], first: int) -> mx.array:
        """Upstream `hourglass` / `hourglass_att`: levels `first, first + 1, ...` of the image features gate it."""
        levels = sum(k.startswith("conv") for k in m)
        skips = []
        for k in range(1, levels + 1):
            for conv in m[f"conv{k}"]:
                x = self._c3(x, conv)
            x = self._gate(x, imgs[first + k - 1], m[f"feature_att_{8 << (first + k - 2)}"])
            skips.append(x)
        merges = sorted(k for k in m if k.startswith("agg"))  # agg (2 levels) or agg_0, agg_1 (3 levels)
        for k in range(levels, 1, -1):
            x = mx.concatenate([self._d3(x, m[f"up{k}"]), skips[k - 2]], axis=-1)
            for conv in m[merges[levels - k]]:
                x = self._c3(x, conv)
            x = self._gate(x, imgs[first + k - 2], m[f"feature_att_up_{8 << (first + k - 3)}"])
        x = self._d3(x, m["up1"])
        if "final" in m:
            x = self._c3(x, m["final"], act=False)
        return x

    @staticmethod
    def _normalized(x: mx.array, groups: int) -> mx.array:
        """Per-group L2-normalized NHWC features, float32 (`model._normalized`)."""
        n, h, w, c = x.shape
        x = x.astype(mx.float32).reshape(n, h, w, groups, c // groups)
        x = x / (mx.sqrt(mx.square(x).sum(axis=-1, keepdims=True)) + EPS)
        return x.reshape(n, h, w, c)

    # the two halves --------------------------------------------------------------------------------------------------

    def _attention(self, left: mx.array, right: mx.array) -> tuple[mx.array, ...]:
        """Features, attention and its top-k; float32."""
        both = mx.concatenate([left, right], axis=0).transpose(0, 2, 3, 1).astype(mx.float32)
        x = _relu6(self._c(both, self.stem))
        x = self._mobile(x, self.blocks[0])
        x4 = self._mobile(x, self.blocks[1])
        x8 = self._mobile(x4, self.blocks[2])
        x16 = self._mobile(x8, self.blocks[3])
        x32 = self._mobile(x16, self.blocks[4])
        x16 = self._up(x32, x16, self.up[0])
        x8 = self._up(x16, x8, self.up[1])
        x4 = _leaky(self._c(self._up(x8, x4, self.up[2]), self.up4))
        stem_2 = self._stem(both, self.stem_2)
        stem_4 = self._stem(stem_2, self.stem_4)
        f4 = mx.concatenate([x4, stem_4], axis=-1)  # both images, 1/4
        f8 = x8 if self.plus else mx.concatenate([x8, self._stem(stem_4, self.stem_8)], axis=-1)
        fl = [f4[:1], f8[:1], x16[:1], x32[:1]]
        d4 = self.maxdisp // 4
        if self.plus:
            match = self._c(_leaky(self._c(f4, self.match[0])), self.match[1])
            m = self._normalized(match, 1)
            cost = gwc_volume(m[:1], m[1:], d4, 1)[0, :, :, :, 0]  # [D, H, W]
            xp = mx.pad(cost, [(1, 1), (0, 0), (0, 0)])  # the stem: depth taps as channels, one conv2d
            x = mx.stack([xp[a : a + d4] for a in range(3)], axis=-1)
            x = _leaky(mx.conv2d(x, self.corr_stem[0], padding=1) + self.corr_stem[1])[None]
            x = self._gate(x, fl[0], self.corr_att)
            att = self._hourglass(x, fl, self.hg_att, 1)[0, :, :, :, 0].transpose(1, 2, 0)  # [H, W, D]
        else:
            m = self._normalized(f8, 12)
            corr = gwc_volume(m[:1], m[1:], self.maxdisp // 8, 12)  # [1, D, H, W, 12]
            _, d8, h8, w8, g = corr.shape
            zero = mx.zeros((g,), mx.float32)
            corr = depthwise(corr.reshape(d8, h8, w8, g), self.patch, zero, (1, 1), (1, 1), (1, 1))
            x = self._gate(corr.reshape(1, d8, h8, w8, g), fl[1], self.corr_att)
            cost = self._hourglass(x, fl, self.hg_att, 2)  # [1, D8, H8, W8, 1]
            att = self._propagated(cost, stem_4, d4)
        prob = mx.softmax(att, axis=-1)
        ind = mx.sort(mx.argpartition(-prob, TOP_K - 1, axis=-1)[..., :TOP_K], axis=-1)
        top = mx.take_along_axis(prob, ind, axis=-1)  # [H, W, K]
        return f4, fl[1], fl[2], stem_2[:1], ind, top

    def _propagated(self, cost: mx.array, stem_4: mx.array, d4: int) -> mx.array:
        """Fast-ACVNet's attention `[H, W, D]` at 1/4: upsampled, re-weighted by five propagated samples' strength."""
        _, d8, h8, w8, _ = cost.shape
        h4, w4 = stem_4.shape[1:3]
        att = trilinear(cost.reshape(1, 1, d8, h8, w8), (d4, h4, w4), (d8 / d4, h8 / h4, w8 / w4), align=False)
        att = att[0, 0].transpose(1, 2, 0)  # [H, W, D]
        prob = mx.softmax(att, axis=-1)
        disp = mx.arange(d4, dtype=mx.float32)
        pred = (prob * disp).sum(axis=-1, keepdims=True)  # [H, W, 1]
        var = mx.sigmoid(self.beta + self.gamma * (prob * mx.square(disp - pred)).sum(axis=-1, keepdims=True))
        samples = mx.concatenate(_shifts(pred), axis=-1)  # [H, W, 5]
        variance = mx.concatenate(_shifts(var), axis=-1)
        xs = mx.arange(w4, dtype=mx.float32)[None, :, None]
        ys = mx.broadcast_to(mx.arange(h4, dtype=mx.float32)[:, None, None], samples.shape)
        gx = (xs - samples) / ((w4 - 1.0) / 2.0) - 1.0
        gy = ys / ((h4 - 1.0) / 2.0) - 1.0
        grid = mx.stack([gx, gy], axis=-1).transpose(2, 0, 1, 3).reshape(1, 5, h4 * w4, 2)
        strength = warp_dot(stem_4[:1], stem_4[1:], grid)[0].transpose(1, 2, 0) / stem_4.shape[-1]  # [H, W, 5]
        strength = mx.softmax(strength * variance, axis=-1)
        shifted = _shifts(att)  # 5 x [H, W, D]
        out = shifted[0] * strength[..., :1]
        for i in range(1, 5):
            out = out + shifted[i] * strength[..., i : i + 1]
        return out

    def _disparity(self, parts: tuple[mx.array, ...]) -> mx.array:
        """The concatenation volume, its hourglass, the top-2 regression and the superpixel upsampling."""
        f4, f8, f16, stem_2 = (t.astype(self.dtype) for t in parts[:4])
        ind, top = parts[4:]
        fl = [f4[:1], f8, f16]
        concat = self._c(_leaky(self._c(f4, self.concat_feature[0])), self.concat_feature[1])  # [2, H, W, 16]
        _, h, w, c = concat.shape
        k = ind.shape[-1]
        src = mx.arange(w)[:, None] - ind  # [H, W, K]
        valid = src >= 0
        right = mx.take_along_axis(concat[1], mx.maximum(src, 0).reshape(h, w * k, 1), axis=1).reshape(h, w, k, c)
        wt = top.astype(self.dtype)[..., None]
        vol = mx.concatenate(
            [wt * concat[0][:, :, None], mx.where(valid[..., None], wt * right, 0)], axis=-1
        )  # [H, W, K, 32]
        vol = vol.transpose(2, 0, 1, 3)[None]  # [1, K, H, W, 32]
        vol = self._gate(self._c3(vol, self.concat_stem), fl[0], self.concat_att)
        cost = self._hourglass(vol, fl, self.hg, 1)[0, :, :, :, 0].transpose(1, 2, 0).astype(mx.float32)

        # top-2 soft arg-max over the candidates (upstream `regression_topk`)
        samples = ind.astype(mx.float32)
        first = mx.argmax(cost, axis=-1, keepdims=True)
        best = mx.take_along_axis(cost, first, axis=-1)
        rest = mx.where(mx.arange(k) == first, -mx.inf, cost)
        second = mx.argmax(rest, axis=-1, keepdims=True)
        gap = best - mx.take_along_axis(cost, second, axis=-1)
        d1, d2 = mx.take_along_axis(samples, first, axis=-1), mx.take_along_axis(samples, second, axis=-1)
        disp = (d2 + (d1 - d2) * mx.sigmoid(gap))[:, :, 0]  # [H, W]

        # superpixel weights and the 4x upsampling: every pixel mixes the 3 x 3 neighbourhood of its coarse pixel
        s = self._stem(f4[:1], self.spx_4)
        s = self._up(s, stem_2, self.spx_2)
        prob = mx.softmax(self._d(s, self.spx).astype(mx.float32), axis=-1)  # [1, 4H, 4W, 9]
        dp = mx.pad(disp, [(1, 1), (1, 1)])
        near = mx.stack([dp[i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=-1)  # [H, W, 9]
        out = (prob.reshape(h, 4, w, 4, 9) * near[:, None, :, None]).sum(axis=-1)
        return out.reshape(1, 4 * h, 4 * w) * 4.0

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32, lazy) of a padded float32 pair `[1, 3, H, W]` (multiples of 32)."""
        parts = self._pre(left, right)
        mx.eval(parts)
        return self._post(parts)
