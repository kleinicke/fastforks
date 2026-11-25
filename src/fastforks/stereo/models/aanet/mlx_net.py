"""AANet / AANet+ on Apple MLX, written in `mlx.core` operations (`MlxAANet(net, dtype)`): a padded pair -> disparity.

`AANet.forward` (`net/aanet.py`) as one compiled function, channels last, with the batch norms folded into the
convolutions. Written out instead of replayed from an exported graph: the export alone was 10-20 s of a first call.

- both images go through the feature extractor as one batch of two;
- the correlation pyramid is the shared `gwc_volume` kernel with one group (the mean over the channels), and the
  disparities are the channels of the aggregation, so nothing is transposed between the 2-D convolutions;
- a modulated deformable convolution is the shared `grid_sample` kernel on a `[pixels, taps]` grid per offset group
  (float32 coordinates, zeros outside as `fastforks.shared.deform_conv`), the modulation, and one matrix product
  per group; its offset convolution (grouped) is one convolution per group;
- the refinement warps with `grid_sample` and regresses in float32 as upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import bilinear_nhwc, grid_sample, gwc_volume

if TYPE_CHECKING:
    from collections.abc import Callable

    _Layer = Callable[[mx.array], mx.array]

__all__ = ["MlxAANet"]


def _folded(weight: Any, bias: Any, norm: Any, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - torch
    """A convolution's weight and bias with its batch norm (or None) folded in; `axis`: the weight's output axis."""
    w = weight.detach().float().numpy()
    b = np.zeros(w.shape[axis], np.float32) if bias is None else bias.detach().float().numpy()
    if norm is not None:
        scale = norm.weight.detach().numpy().astype(np.float64) / np.sqrt(
            norm.running_var.numpy().astype(np.float64) + norm.eps
        )
        shape = [1] * w.ndim
        shape[axis] = -1
        w = (w * scale.reshape(shape)).astype(np.float32)
        b = (norm.bias.detach().numpy() + (b - norm.running_mean.numpy()) * scale).astype(np.float32)
    return w, b


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _leaky(x: mx.array) -> mx.array:
    return mx.maximum(x, 0.2 * x)


def _same(x: mx.array) -> mx.array:
    return x


class MlxAANet:
    """The network `net` (eval, float32 weights) in `dtype` (the name): one compiled function per image size."""

    def __init__(self, net: Any, dtype: str) -> None:  # noqa: ANN401 - `AANet`, its modules read by attribute
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        self.depth = net.max_disp
        f = net.feature_extractor
        if net.feature_type == "aanet":
            stem = self._conv(f.conv1[0], f.conv1[1], _relu)
            layers = [[self._bottleneck(b) for b in layer] for layer in (f.layer1, f.layer2, f.layer3)]
            lateral = [self._conv(m, None) for m in net.fpn.lateral_convs]
            smooth = [self._conv(m[0], m[1], _relu) for m in net.fpn.fpn_convs]

            def features(x: mx.array) -> list[mx.array]:
                x = stem(x)
                maps = []
                for layer, side in zip(layers, lateral, strict=True):
                    for block in layer:
                        x = block(x)
                    maps.append(side(x))
                for i in (2, 1):  # top down: add the coarser level, nearest-neighbour upsampled
                    maps[i - 1] = maps[i - 1] + mx.repeat(mx.repeat(maps[i], 2, axis=1), 2, axis=2)
                return [conv(m) for conv, m in zip(smooth, maps, strict=True)]

        else:  # AANet+: GA-Net's hourglass and a strided pyramid on top
            start = [self._basic(f.conv_start[0]), self._basic(f.conv_start[1]), self._deform(f.conv_start[2], None)]
            hourglass = self._hourglass(f)
            p = net.fpn
            pyramid = [
                [self._conv(seq[0], seq[1], _leaky), self._conv(seq[3], seq[4], _leaky)] for seq in (p.out1, p.out2)
            ]

            def features(x: mx.array) -> list[mx.array]:
                for layer in start:
                    x = layer(x)
                maps = [hourglass(x)]
                for a, b in pyramid:
                    maps.append(b(a(maps[-1])))
                return maps

        self._features = features
        agg = net.aggregation
        self.fusions = [self._fusion(m) for m in agg.fusions]
        self.final = self._conv(agg.final_conv[0], None)
        self.refine = [self._refinement(m) for m in net.refinement]
        self._compiled = mx.compile(self._forward)

    # layers: every builder returns a function of one NHWC map ---------------------------------------------------------

    def _a(self, a: np.ndarray) -> mx.array:
        return mx.array(np.ascontiguousarray(a)).astype(self.dtype)

    def _conv(self, conv: Any, norm: Any, act: _Layer = _same) -> _Layer:  # noqa: ANN401
        """`conv` (a `Conv2d` or `ConvTranspose2d`), its batch norm or None, and an activation."""
        transposed = bool(conv.transposed)
        w, b = _folded(conv.weight, conv.bias, norm, axis=1 if transposed else 0)
        weight = self._a(w.transpose(1, 2, 3, 0) if transposed else w.transpose(0, 2, 3, 1))
        bias = self._a(b)
        stride, padding, dilation = int(conv.stride[0]), int(conv.padding[0]), int(conv.dilation[0])
        if transposed:
            return lambda x: act(mx.conv_transpose2d(x, weight, stride=stride, padding=padding) + bias)
        groups = int(conv.groups)
        if groups == 1:
            return lambda x: act(mx.conv2d(x, weight, stride, padding, dilation) + bias)
        # one convolution per group: MLX's grouped convolution is several times slower (the offsets of a deformable
        # convolution, 64 -> 54 channels in 2 groups at 336 x 496: 23 ms, 2 x 4 ms split)
        parts = mx.split(weight, groups, axis=0)
        cin = weight.shape[-1]

        def grouped(x: mx.array) -> mx.array:
            ys = [mx.conv2d(x[..., g * cin : (g + 1) * cin], w, stride, padding, dilation) for g, w in enumerate(parts)]
            return act(mx.concatenate(ys, axis=-1) + bias)

        return grouped

    def _basic(self, m: Any) -> _Layer:  # noqa: ANN401 - `BasicConv`
        return self._conv(m.conv, m.bn if m.use_bn else None, _relu if m.relu else _same)

    def _deform(self, m: Any, norm: Any, act: _Layer = _same) -> _Layer:  # noqa: ANN401
        """`DeformConv2d` (modulated, 3 x 3) with the batch norm after it (or None) and an activation."""
        dc = m.deform_conv
        if not (m.modulation and m.double_mask) or dc.groups != 1:
            msg = "MlxAANet: only the modulated deformable convolution with a doubled mask and one group is ported"
            raise NotImplementedError(msg)
        offsets = self._conv(m.offset_conv, None)
        w, b = _folded(dc.weight, dc.bias, norm)
        o, c, kh, kw = w.shape
        dg, k = int(dc.deformable_groups), kh * kw
        cg = c // dg
        # per offset group, the rows are (tap, channel of the group)
        weights = [self._a(m) for m in w.reshape(o, dg, cg, k).transpose(1, 3, 2, 0).reshape(dg, k * cg, o)]
        bias = self._a(b)
        stride, padding, dilation = int(dc.stride), int(dc.padding), int(dc.dilation)
        tap = np.arange(k)
        ty = mx.array((tap // kw * dilation - padding).astype(np.float32))
        tx = mx.array((tap % kw * dilation - padding).astype(np.float32))

        def run(x: mx.array) -> mx.array:
            n, h, wd, _ = x.shape
            om = offsets(x).astype(mx.float32)  # [n, ho, wo, dg * 3 * k]: (dy, dx) per group and tap, then the masks
            ho, wo = om.shape[1], om.shape[2]
            off = om[..., : dg * 2 * k].reshape(n, ho, wo, dg, k, 2)
            mask = (2.0 * mx.sigmoid(om[..., dg * 2 * k :])).reshape(n, ho * wo, dg, k, 1).astype(x.dtype)
            py = mx.arange(ho, dtype=mx.float32)[:, None, None] * stride + ty  # [ho, 1, k]
            px = mx.arange(wo, dtype=mx.float32)[None, :, None] * stride + tx  # [1, wo, k]
            sy, sx = 2.0 / max(h - 1, 1), 2.0 / max(wd - 1, 1)  # align_corners: the grid is in pixels
            y = bias
            for g in range(dg):
                gx = (px + off[:, :, :, g, :, 1]) * sx - 1.0
                gy = (py + off[:, :, :, g, :, 0]) * sy - 1.0
                grid = mx.stack([gx, gy], axis=-1).reshape(n, ho * wo, k, 2)
                part = grid_sample(x[..., g * cg : (g + 1) * cg], grid, align_corners=True, border=False)
                y = y + (part * mask[:, :, g]).reshape(n * ho * wo, k * cg) @ weights[g]
            return act(y.reshape(n, ho, wo, o))

        return run

    def _bottleneck(self, m: Any) -> _Layer:  # noqa: ANN401
        """The four bottlenecks (`Bottleneck`, `DeformBottleneck` and the two `Simple` ones): 1 x 1, 3 x 3, 1 x 1."""
        first = self._conv(m.conv1, m.bn1, _relu)
        deform = hasattr(m.conv2, "deform_conv")
        middle = self._deform(m.conv2, m.bn2, _relu) if deform else self._conv(m.conv2, m.bn2, _relu)
        last = self._conv(m.conv3, m.bn3)
        skip = _same if m.downsample is None else self._conv(m.downsample[0], m.downsample[1])
        return lambda x: _relu(last(middle(first(x))) + skip(x))

    def _conv2x(self, m: Any) -> Callable[[mx.array, mx.array], mx.array]:  # noqa: ANN401
        """`Conv2x`: a strided (transposed) convolution, the skip concatenated, a (deformable) convolution."""
        first = self._basic(m.conv1)
        second = self._deform(m.conv2, None) if hasattr(m.conv2, "deform_conv") else self._basic(m.conv2)
        return lambda x, rem: second(mx.concatenate([first(x), rem], axis=-1))

    def _hourglass(self, m: Any) -> _Layer:  # noqa: ANN401
        """The two stacked hourglasses of `GANetFeature` and `HourglassRefinement` (after `conv_start`)."""
        down = [
            self._basic(m.conv1a),
            self._basic(m.conv2a),
            self._deform(m.conv3a, None),
            self._deform(m.conv4a, None),
        ]
        up_a = [self._conv2x(c) for c in (m.deconv4a, m.deconv3a, m.deconv2a, m.deconv1a)]
        down_b = [self._conv2x(c) for c in (m.conv1b, m.conv2b, m.conv3b, m.conv4b)]
        up_b = [self._conv2x(c) for c in (m.deconv4b, m.deconv3b, m.deconv2b, m.deconv1b)]

        def run(x: mx.array) -> mx.array:
            rem = [x]
            for layer in down:
                x = layer(x)
                rem.append(x)
            for i, layer in enumerate(up_a):
                x = layer(x, rem[3 - i])
                rem[3 - i] = x
            for i, layer in enumerate(down_b):
                x = layer(x, rem[i + 1])
                rem[i + 1] = x
            for i, layer in enumerate(up_b):
                x = layer(x, rem[3 - i])
            return x

        return run

    def _exchange(self, m: Any, i: int, j: int) -> list[_Layer]:  # noqa: ANN401
        """The layers that bring scale `j` to scale `i` in a fusion (the bilinear upsampling is left to the caller)."""
        if i == j:
            return []
        if i < j:  # coarser: 1 x 1
            return [self._conv(m[0], m[1])]
        # finer: strided 3 x 3 convolutions, the last without an activation
        return [self._conv(s[0], s[1], _leaky if k < len(m) - 1 else _same) for k, s in enumerate(m)]

    def _fusion(self, m: Any) -> Callable[[list[mx.array]], list[mx.array]]:  # noqa: ANN401
        """`AdaptiveAggregationModule`: a bottleneck per scale, then every output scale sums all the scales."""
        branches = [[self._bottleneck(b) for b in branch] for branch in m.branches]
        fuse = [[self._exchange(layer, i, j) for j, layer in enumerate(row)] for i, row in enumerate(m.fuse_layers)]

        def run(x: list[mx.array]) -> list[mx.array]:
            x = list(x)
            for i, branch in enumerate(branches):
                for block in branch:
                    x[i] = block(x[i])
            out = []
            for i, row in enumerate(fuse):
                size = (x[i].shape[1], x[i].shape[2])
                total = None
                for j, layers in enumerate(row):
                    y = x[j]
                    for layer in layers:
                        y = layer(y)
                    if (y.shape[1], y.shape[2]) != size:
                        ratios = (y.shape[1] / size[0], y.shape[2] / size[1])
                        y = bilinear_nhwc(y, size, ratios, align=False)
                    total = y if total is None else total + y
                assert total is not None  # noqa: S101 - every scale has itself to sum
                out.append(_leaky(total))
            return out

        return run

    def _refinement(self, m: Any) -> Callable[[mx.array, mx.array, mx.array], mx.array]:  # noqa: ANN401
        """`StereoDRNetRefinement` / `HourglassRefinement`: disparity `[1, h, w, 1]` -> `[1, H, W, 1]`, float32."""
        conv1 = self._conv(m.conv1[0], m.conv1[1], _leaky)
        conv2 = self._conv(m.conv2[0], m.conv2[1], _leaky)
        final = self._conv(m.final_conv, None)
        if hasattr(m, "dilated_blocks"):
            blocks = [(self._conv(b.conv1, b.bn1, _leaky), self._conv(b.conv2, b.bn2)) for b in m.dilated_blocks]

            def body(x: mx.array) -> mx.array:
                for a, b in blocks:
                    x = _leaky(b(a(x)) + x)
                return x

        else:
            start, hourglass = self._deform(m.conv_start, None), self._hourglass(m)

            def body(x: mx.array) -> mx.array:
                return hourglass(start(x))

        def run(disp: mx.array, left: mx.array, right: mx.array) -> mx.array:
            _, height, width, _ = left.shape
            h, w = disp.shape[1], disp.shape[2]
            if w != width:
                disp = bilinear_nhwc(disp, (height, width), (h / height, w / width), align=False) * (width / w)
            # the right image warped to the left view: upstream's grid (`2 x / (W - 1) - 1`, sampled without
            # align_corners), border padding
            gx = (mx.arange(width, dtype=mx.float32)[None, :] - disp[0, :, :, 0]) * (2.0 / (width - 1)) - 1.0
            gy = mx.arange(height, dtype=mx.float32)[:, None] * (2.0 / (height - 1)) - 1.0
            grid = mx.stack([gx, mx.broadcast_to(gy, gx.shape)], axis=-1)[None]
            error = grid_sample(right, grid, align_corners=False, border=True) - left
            a = conv1(mx.concatenate([error, left], axis=-1).astype(self.dtype))
            x = body(mx.concatenate([a, conv2(disp.astype(self.dtype))], axis=-1))
            return _relu(disp + final(x).astype(mx.float32))

        return run

    # the network -----------------------------------------------------------------------------------------------------

    def _forward(self, left: mx.array, right: mx.array) -> mx.array:
        left, right = left.transpose(0, 2, 3, 1), right.transpose(0, 2, 3, 1)  # float32, for the refinement
        maps = self._features(mx.concatenate([left, right], axis=0).astype(self.dtype))
        cost = [
            gwc_volume(f[:1], f[1:], self.depth >> s, 1)[..., 0].transpose(0, 2, 3, 1) for s, f in enumerate(maps)
        ]  # [1, h, w, D] per scale: the disparities are the channels
        for fusion in self.fusions:
            cost = fusion(cost)
        prob = mx.softmax(self.final(cost[0]).astype(mx.float32), axis=-1)
        disp = (prob * mx.arange(self.depth, dtype=mx.float32)).sum(axis=-1, keepdims=True)  # [1, h, w, 1]
        for i, refine in enumerate(self.refine):
            f = 2 ** (len(self.refine) - 1 - i)  # the images at 1 / f: a bilinear downscale is the mean of f x f
            if f == 1:
                lt, rt = left, right
            else:
                _, h, w, c = left.shape
                lt = left.reshape(1, h // f, f, w // f, f, c).mean(axis=(2, 4))
                rt = right.reshape(1, h // f, f, w // f, f, c).mean(axis=(2, 4))
            disp = refine(disp, lt, rt)
        return disp[..., 0]

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32, lazy) of a padded float32 pair `[1, 3, H, W]`."""
        return self._compiled(left, right)
