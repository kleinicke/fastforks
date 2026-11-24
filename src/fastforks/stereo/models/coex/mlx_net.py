"""CoEx on Apple MLX, written in `mlx.core` operations (`MlxCoEx(net, dtype)`): a padded pair -> disparity.

`CoEx.forward` (`net/coex.py`) as one compiled function, channels last, with the batch norms folded into the
convolutions. Written out instead of replayed from an exported graph: the export alone was ~5 s of a first call.

- the MobileNetV2 depthwise layers are the shared `depthwise` Metal kernel;
- the correlation volume is the shared `gwc_volume` kernel with one group;
- the 3-D stem (1 -> 8 channels) has its depth taps folded into channels (`mx.conv3d` is slow at one input channel)
  and the transposed 3-D convolutions run as one small convolution per output phase (`mlx_graph`'s);
- the top-2 regression is two arg-max passes over the disparity axis (kept last: MLX reduces the last axis fastest)
  and the superpixel upsampling a broadcast over the 4 x 4 cells, both float32 as upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import depthwise, gwc_volume
from fastforks.shared.mlx_graph import _conv_transpose3d_phases

if TYPE_CHECKING:
    from .net.coex import CoEx

__all__ = ["MlxCoEx"]

_Conv = tuple[mx.array, mx.array, int, int]  # weight [O, k.., I], bias, stride, padding


def _folded(conv: Any, norm: Any, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - torch modules
    """A convolution's weight and bias with its batch norm (or None) folded in; `axis`: the weight's output axis."""
    w = conv.weight.detach().float().numpy()
    b = np.zeros(w.shape[axis], np.float32) if conv.bias is None else conv.bias.detach().float().numpy()
    if norm is not None:
        scale = norm.weight.detach().numpy().astype(np.float64) / np.sqrt(
            norm.running_var.numpy().astype(np.float64) + norm.eps
        )
        shape = [1] * w.ndim
        shape[axis] = -1
        w = (w * scale.reshape(shape)).astype(np.float32)
        b = (norm.bias.detach().numpy() + (b - norm.running_mean.numpy()) * scale).astype(np.float32)
    return w, b


def _leaky(x: mx.array) -> mx.array:
    return mx.maximum(x, 0.01 * x)


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _relu6(x: mx.array) -> mx.array:
    return mx.clip(x, 0, 6)


class MlxCoEx:
    """The network `net` (eval, float32 weights) in `dtype` (the name): one compiled function per image size."""

    def __init__(self, net: CoEx, dtype: str) -> None:
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        self.depth = net.D
        if net.top_k != 2:  # noqa: PLR2004 - the regression below is written for upstream's top-2
            msg = f"MlxCoEx: top_k {net.top_k} is not ported (2 is)"
            raise NotImplementedError(msg)
        f = net.feature
        self.stem = self._conv(f.conv_stem, f.bn1)
        self.blocks = [[self._block(b) for stage in getattr(f, f"block{i}") for b in stage] for i in range(5)]
        up = net.up
        self.up = [self._conv2x(m) for m in (up.deconv32_16, up.deconv16_8, up.deconv8_4)]
        self.up4 = self._basic(up.conv4)
        self.stem_2 = (self._basic(net.stem_2[0]), self._conv(net.stem_2[1], net.stem_2[2]))
        self.stem_4 = (self._basic(net.stem_4[0]), self._conv(net.stem_4[1], net.stem_4[2]))
        self.spx_4 = (self._basic(net.spx_4[0]), self._conv(net.spx_4[1], net.spx_4[2]))
        self.spx_2 = self._conv2x(net.spx_2)
        self.spx = self._deconv(net.spx[0], None)
        cv = net.cost_volume
        self.cv_conv, self.cv_desc = self._basic(cv.conv), self._conv(cv.desc, None)
        agg: Any = net.cost_agg
        w, b = _folded(agg.conv_stem.conv, agg.conv_stem.bn)  # [8, 1, 3, 3, 3] -> [8, kh, kw, kd]
        self.agg_stem = (self._a(w[:, 0].transpose(0, 2, 3, 1)), self._a(b))
        self.att_stem = self._att(agg.channelAttStem)
        self.att = [self._att(m) for m in agg.channelAtt]
        self.att_down = [self._att(m) for m in agg.channelAttDown]
        self.down = [[self._basic3(m) for m in seq] for seq in agg.conv_down]
        self.agg = [[self._basic3(m) for m in seq] for seq in agg.conv_agg]
        self.skip = [self._basic3(m) for m in agg.conv_skip]
        self.up3 = []
        for m in agg.conv_up:  # the weight stays `[Cin, Cout, kd, kh, kw]`: `_conv_transpose3d_phases`'s layout
            w, b = _folded(m.conv, m.bn if m.use_bn else None, axis=1)
            self.up3.append((self._a(w), self._a(b), m.relu))
        self._compiled = mx.compile(self._forward)

    # weights ---------------------------------------------------------------------------------------------------------

    def _a(self, a: np.ndarray) -> mx.array:
        return mx.array(np.ascontiguousarray(a)).astype(self.dtype)

    def _conv(self, conv: Any, norm: Any) -> _Conv:  # noqa: ANN401 - torch modules of the vendored net
        w, b = _folded(conv, norm)
        return self._a(w.transpose(0, 2, 3, 1)), self._a(b), int(conv.stride[0]), int(conv.padding[0])

    def _basic(self, m: Any) -> _Conv:  # noqa: ANN401
        return self._conv(m.conv, m.bn if m.use_bn else None)

    def _basic3(self, m: Any) -> _Conv:  # noqa: ANN401
        w, b = _folded(m.conv, m.bn if m.use_bn else None)
        return self._a(w.transpose(0, 2, 3, 4, 1)), self._a(b), int(m.conv.stride[0]), int(m.conv.padding[0])

    def _deconv(self, conv: Any, norm: Any) -> _Conv:  # noqa: ANN401
        w, b = _folded(conv, norm, axis=1)  # [Cin, Cout, kh, kw] -> [Cout, kh, kw, Cin]
        return self._a(w.transpose(1, 2, 3, 0)), self._a(b), int(conv.stride[0]), int(conv.padding[0])

    def _conv2x(self, m: Any) -> tuple[_Conv, _Conv]:  # noqa: ANN401
        return self._deconv(m.conv1.conv, m.conv1.bn), self._basic(m.conv2)

    def _dw(self, conv: Any, norm: Any) -> tuple[mx.array, mx.array, int]:  # noqa: ANN401
        w, b = _folded(conv, norm)  # [C, 1, kh, kw] -> [kh, kw, C]
        return self._a(w[:, 0].transpose(1, 2, 0)), self._a(b), int(conv.stride[0])

    def _block(self, m: Any) -> tuple[Any, ...]:  # noqa: ANN401
        """A MobileNetV2 block: (expand or None, depthwise, project, skip)."""
        if hasattr(m, "conv_pwl"):
            return self._conv(m.conv_pw, m.bn1), self._dw(m.conv_dw, m.bn2), self._conv(m.conv_pwl, m.bn3), m.has_skip
        return None, self._dw(m.conv_dw, m.bn1), self._conv(m.conv_pw, m.bn2), m.has_skip

    def _att(self, m: Any) -> tuple[_Conv, _Conv]:  # noqa: ANN401
        return self._basic(m.im_att[0]), self._conv(m.im_att[1], None)

    # layers ----------------------------------------------------------------------------------------------------------

    @staticmethod
    def _c(x: mx.array, conv: _Conv) -> mx.array:
        w, b, stride, padding = conv
        return mx.conv2d(x, w, stride=stride, padding=padding) + b

    @staticmethod
    def _c3(x: mx.array, conv: _Conv) -> mx.array:
        w, b, stride, padding = conv
        return _leaky(mx.conv3d(x, w, stride=stride, padding=padding) + b)

    @staticmethod
    def _d(x: mx.array, conv: _Conv) -> mx.array:
        w, b, stride, padding = conv
        return mx.conv_transpose2d(x, w, stride=stride, padding=padding) + b

    def _up(self, x: mx.array, rem: mx.array, m: tuple[_Conv, _Conv]) -> mx.array:
        """`Conv2x` with a transposed convolution (its output has the skip's size: the pair is padded to 32s)."""
        return _leaky(self._c(mx.concatenate([_leaky(self._d(x, m[0])), rem], axis=-1), m[1]))

    def _mobile(self, x: mx.array, blocks: list[tuple[Any, ...]]) -> mx.array:
        for expand, (w, b, stride), project, skip in blocks:
            y = x if expand is None else _relu6(self._c(x, expand))
            y = _relu6(depthwise(y, w, b, (stride, stride), (1, 1), (1, 1)))
            y = self._c(y, project)
            x = x + y if skip else y
        return x

    def _gate(self, cost: mx.array, im: mx.array, att: tuple[_Conv, _Conv]) -> mx.array:
        """`ChannelAtt`: `cost` `[1, D, H, W, C]` gated by the left image's features `[1, H, W, K]`."""
        return cost * mx.sigmoid(self._c(_leaky(self._c(im, att[0])), att[1]))[:, None]

    def _aggregate(self, img: list[mx.array], cost: mx.array) -> mx.array:
        """`Aggregation.forward`: `cost` `[D, H, W]` -> `[H, W, D]`."""
        d = cost.shape[0]
        # the stem: the three depth taps of every slice as its channels, one 2-D convolution over the D slices
        xp = mx.pad(cost, [(1, 1), (0, 0), (0, 0)])
        x = mx.stack([xp[a : a + d] for a in range(3)], axis=-1)
        x = _leaky(mx.conv2d(x, self.agg_stem[0], padding=1) + self.agg_stem[1])[None]
        x = self._gate(x, img[0], self.att_stem)
        feats = [x]
        for i in range(3):
            for conv in self.down[i]:
                x = self._c3(x, conv)
            x = self._gate(x, img[i + 1], self.att_down[i])
            feats.append(x)
        for i in (2, 1, 0):
            wt, b, act = self.up3[i]
            y = _conv_transpose3d_phases(x, wt, (2, 2, 2), (1, 1, 1), (0, 0, 0))
            if y is None or y.shape[1:4] != feats[i].shape[1:4]:
                msg = f"MlxCoEx: the cost volume {tuple(feats[i].shape[1:4])} is not a multiple of 8"
                raise ValueError(msg)
            x = y + b
            if act:
                x = _leaky(x)
            if i == 0:
                break
            x = self._c3(mx.concatenate([x, feats[i]], axis=-1), self.skip[i])
            for conv in self.agg[i]:
                x = self._c3(x, conv)
            x = self._gate(x, img[i], self.att[i])
        return x[0, :, :, :, 0].transpose(1, 2, 0)

    def _forward(self, left: mx.array, right: mx.array) -> mx.array:
        both = mx.concatenate([left, right], axis=0).transpose(0, 2, 3, 1).astype(self.dtype)
        x = _relu6(self._c(both, self.stem))
        x = self._mobile(x, self.blocks[0])
        x4 = self._mobile(x, self.blocks[1])
        x8 = self._mobile(x4, self.blocks[2])
        x16 = self._mobile(x8, self.blocks[3])
        x32 = self._mobile(x16, self.blocks[4])
        x16 = self._up(x32, x16, self.up[0])
        x8 = self._up(x16, x8, self.up[1])
        x4 = _leaky(self._c(self._up(x8, x4, self.up[2]), self.up4))
        stem_2 = _relu(self._c(_leaky(self._c(both, self.stem_2[0])), self.stem_2[1]))
        stem_4 = _relu(self._c(_leaky(self._c(stem_2, self.stem_4[0])), self.stem_4[1]))
        x0 = mx.concatenate([x4, stem_4], axis=-1)  # both images, 1/4
        img = [x0[:1], x8[:1], x16[:1], x32[:1]]

        desc = self._c(_leaky(self._c(x0, self.cv_conv)), self.cv_desc).astype(mx.float32)
        desc = (desc * mx.rsqrt(mx.square(desc).sum(axis=-1, keepdims=True))).astype(self.dtype)
        c = desc.shape[-1]
        cost = gwc_volume(desc[:1], desc[1:], self.depth, 1)[0, :, :, :, 0] * c  # the sum over channels, [D, H, W]
        cost = self._aggregate(img, cost).astype(mx.float32)  # [H, W, D]

        # top-2 soft arg-max over the disparities
        first = mx.argmax(cost, axis=-1, keepdims=True)
        top = mx.take_along_axis(cost, first, axis=-1)
        rest = mx.where(mx.arange(self.depth) == first, -mx.inf, cost)
        second = mx.argmax(rest, axis=-1, keepdims=True)
        gap = top - mx.take_along_axis(cost, second, axis=-1)
        first, second = first.astype(mx.float32), second.astype(mx.float32)
        disp = (second + (first - second) * mx.sigmoid(gap))[:, :, 0]  # [H, W]

        # superpixel weights and the 4x upsampling: every pixel mixes the 3 x 3 neighbourhood of its coarse pixel
        s = _relu(self._c(_leaky(self._c(x0[:1], self.spx_4[0])), self.spx_4[1]))
        s = self._up(s, stem_2[:1], self.spx_2)
        prob = mx.softmax(self._d(s, self.spx).astype(mx.float32), axis=-1)  # [1, 4H, 4W, 9]
        h, w = disp.shape
        dp = mx.pad(disp, [(1, 1), (1, 1)])
        near = mx.stack([dp[i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=-1)  # [H, W, 9]
        out = (prob.reshape(h, 4, w, 4, 9) * near[:, None, :, None]).sum(axis=-1)
        return out.reshape(1, 4 * h, 4 * w) * 4.0

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32, lazy) of a padded float32 pair `[1, 3, H, W]` (multiples of 32)."""
        return self._compiled(left, right)
