"""DLNR on Apple MLX, written in `mlx.core` operations (`MlxDLNR(net, dtype)`): a padded pair -> disparity.

The maths of `net/inference.py` (`forward` on the network `prepare` returns), channels last. Written out instead of
replayed from an exported graph: the export alone was 14 s of a 17 s first call.

- Restormer blocks: the 1x1 convolutions are matrix products over the channels; the attention statistics (`q^T k`
  and the norms over the pixels) and the GDFN gate (`gelu(d1) * d2` of the depthwise conv) are one shared Metal
  kernel each (`token_gram`, `dw_gate`), the layer norms the shared row kernel. `dw_gate` reads four channels per
  thread, so the hidden widths 170 and 1361 run 172 and 1364 wide (zero filters).
- correlation: the float32 all-pairs volume is one matrix product per row; the pyramid lookup is the shared
  `linear_sample` kernel (zeros outside, as the gather lookup).
- the update step is one compiled function: each LSTM's four gates are one convolution, the motion encoder's last
  conv has 128 outputs (two zero filters, the flow added into their place) and the flow head only its x output.
- float32 in a float16 run: layer-norm and instance-norm statistics, the attention matrices, the correlation, the
  flow state and its 7x7 convolution, the upsampling and the whole normalization refinement (as upstream).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    _Layer = Callable[[mx.array], mx.array]

__all__ = ["MlxDLNR"]

RADIUS, LEVELS = 4, 4  # correlation lookup (the released architecture)
MOTION = 126  # the motion encoder's own outputs; the flow takes the channels after them
ONE = (1, 1)


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().float().numpy()


def _ohwi(conv: Any) -> np.ndarray:  # noqa: ANN401 - a `Conv2d`
    return _np(conv.weight).transpose(0, 2, 3, 1)


def _folded(conv: Any, norm: Any) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - torch modules
    """A bias-free (transposed) convolution's `[O, kh, kw, I]` weight and bias with its batch norm folded in."""
    w = _np(conv.weight).astype(np.float64)
    w = w.transpose(1, 2, 3, 0) if conv.transposed else w.transpose(0, 2, 3, 1)
    scale = _np(norm.weight).astype(np.float64) / np.sqrt(_np(norm.running_var).astype(np.float64) + norm.eps)
    bias = _np(norm.bias) - _np(norm.running_mean) * scale
    return (w * scale[:, None, None, None]).astype(np.float32), bias.astype(np.float32)


def _unshuffle(x: mx.array, r: int) -> mx.array:
    """`PixelUnshuffle(r)` of an NHWC map (output channel `c * r * r + i * r + j`)."""
    n, h, w, c = x.shape
    x = x.reshape(n, h // r, r, w // r, r, c).transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(n, h // r, w // r, c * r * r)


def _shuffle(x: mx.array, r: int) -> mx.array:
    """`PixelShuffle(r)` of an NHWC map."""
    n, h, w, c = x.shape
    x = x.reshape(n, h, w, c // (r * r), r, r).transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(n, h * r, w * r, c // (r * r))


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _instance_norm(x: mx.array) -> mx.array:
    """`InstanceNorm2d` (no affine) of an NHWC map, float32 statistics per sample."""
    out = []
    for i in range(x.shape[0]):
        mean, var = ops.channel_stats(x[i].reshape(-1, x.shape[-1]))
        out.append(((x[i].astype(mx.float32) - mean) * mx.rsqrt(var + 1e-5)).astype(x.dtype))
    return mx.stack(out)


def _interp(x: mx.array, size: Sequence[int]) -> mx.array:
    """Bilinear, align_corners=True, NHWC."""
    h, w = x.shape[1:3]
    ratios = ((h - 1) / (size[0] - 1), (w - 1) / (size[1] - 1))
    return ops.bilinear_nhwc(x, (size[0], size[1]), ratios, align=True)


class MlxDLNR:
    """The network `net` (`inference.prepare`d, float32 weights) in `dtype` (the name)."""

    def __init__(self, net: Any, dtype: str, iters: int) -> None:  # noqa: ANN401 - `DLNR`, its modules read by attribute
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        self.iters = iters
        ex = net.extractor
        self.patch = self._a(_ohwi(ex.patch_embed.proj))
        stages = ("encoder_level1", "encoder_level2", "encoder_level3", "latent", "decoder_level3", "decoder_level2")
        self.stages = {s: [self._block(b) for b in getattr(ex, s)] for s in stages}
        self.heads04 = [[self._block(b) for b in seq] for seq in ex.outputs04]
        self.resample = {
            s: self._a(_ohwi(getattr(ex, s).body[0]))
            for s in ("down1_2", "down2_3", "down3_4", "up4_3", "up3_2", "up2_1")
        }
        self.reduce3 = self._a(_np(ex.reduce_chan_level3.weight)[:, :, 0, 0].T)
        self.heads08 = [self._a(_np(m.weight)[:, :, 0, 0].T) for m in ex.outputs08]
        self.head16 = self._a(_np(ex.output16.weight)[:, :, 0, 0].T)
        res, last = net.volume_conv[0], net.volume_conv[1]
        self.volume = [self._a(_ohwi(res.conv1)), self._a(_ohwi(res.conv2))]  # their biases vanish in the norms
        self.volume_out = (self._a(_ohwi(last)), self._a(_np(last.bias)))
        self.context = [(self._a(_ohwi(c)), self._a(_np(c.bias))) for c in net.bias_convs]
        self._update(net.update_block)
        self._refinement(net.normalizationRefinement)
        self._step = mx.compile(self._iteration)

    def _a(self, a: np.ndarray) -> mx.array:
        return mx.array(np.ascontiguousarray(a)).astype(self.dtype)

    # Restormer extractor ----------------------------------------------------------------------------------------------

    def _block(self, m: Any) -> _Layer:  # noqa: ANN401 - `TransformerBlock`
        at, ff = m.attn, m.ffn
        c, heads = at.qkv.in_channels, at.num_heads
        dtype = self.dtype
        n1 = (self._a(_np(m.norm1.body.weight)), self._a(_np(m.norm1.body.bias)))
        n2 = (self._a(_np(m.norm2.body.weight)), self._a(_np(m.norm2.body.bias)))
        qkv = self._a(_np(at.qkv.weight)[:, :, 0, 0].T)  # [C, 3 C]
        dw = self._a(_np(at.qkv_dwconv.weight)[:, 0].transpose(1, 2, 0))  # [3, 3, 3 C]
        no_bias = mx.zeros((3 * c,), dtype)
        temperature = mx.array(np.repeat(_np(at.temperature).reshape(-1), c // heads)[None, :, None])
        head = np.arange(c) // (c // heads)
        other = mx.array(np.where(head[:, None] == head[None, :], 0.0, -np.inf).astype(np.float32))  # across heads
        proj = mx.array(_np(at.project_out.weight)[:, :, 0, 0])  # [C out, C in], float32
        hidden = ff.project_out.in_channels
        wide = -(-hidden // 4) * 4  # `dw_gate` reads four channels per thread
        grow = wide - hidden
        w_in = _np(ff.project_in.weight)[:, :, 0, 0].T.reshape(c, 2, hidden)
        w_in = self._a(np.pad(w_in, [(0, 0), (0, 0), (0, grow)]).reshape(c, 2 * wide))
        w_dw = _np(ff.dwconv.weight)[:, 0].transpose(1, 2, 0).reshape(3, 3, 2, hidden)
        w_dw = self._a(np.pad(w_dw, [(0, 0), (0, 0), (0, 0), (0, grow)]).reshape(3, 3, 2 * wide))
        w_out = self._a(np.pad(_np(ff.project_out.weight)[:, :, 0, 0].T, [(0, grow), (0, 0)]))  # [wide, C]

        def run(x: mx.array) -> mx.array:
            b, h, w, _ = x.shape
            t = ops.depthwise(ops.row_layer_norm(x, *n1, 1e-5) @ qkv, dw, no_bias, ONE, ONE, ONE)
            t = t.reshape(b, h * w, 3 * c)
            gram, sq = ops.token_gram(t, c)
            norm = mx.maximum(mx.sqrt(sq), 1e-12)  # F.normalize over the pixels
            g = gram * (temperature / norm[:, :c, None]) / norm[:, None, c:]
            attn = mx.softmax(g if heads == 1 else g + other, axis=-1)
            mat = mx.swapaxes(proj @ attn, 1, 2).astype(dtype)  # attention and project_out: [B, C in, C out]
            x = x + (t[..., 2 * c :] @ mat).reshape(b, h, w, c)
            return x + ops.dw_gate(ops.row_layer_norm(x, *n2, 1e-5) @ w_in, w_dw) @ w_out

        return run

    def _stage(self, name: str, x: mx.array) -> mx.array:
        for block in self.stages[name]:
            x = block(x)
        mx.eval(x)
        return x

    def _extract(self, x: mx.array) -> tuple[list[list[mx.array]], mx.array]:
        """`Channel_Attention_Transformer_Extractor.forward` of the pair `[2, H, W, 3]`.

        The context maps (two per level, of the left image) and the features `v`.
        """
        r = self.resample
        conv = lambda x, w: mx.conv2d(x, w, padding=1)  # noqa: E731
        e1 = self._stage("encoder_level1", conv(_unshuffle(x, 4), self.patch))
        e2 = self._stage("encoder_level2", _unshuffle(conv(e1, r["down1_2"]), 2))
        e3 = self._stage("encoder_level3", _unshuffle(conv(e2, r["down2_3"]), 2))
        latent = self._stage("latent", _unshuffle(conv(e3, r["down3_4"]), 2))
        d3 = mx.concatenate([_shuffle(conv(latent, r["up4_3"]), 2), e3], axis=-1) @ self.reduce3
        c16 = d3[:1] @ self.head16
        d3 = self._stage("decoder_level3", d3)
        d2 = mx.concatenate([_shuffle(conv(d3, r["up3_2"]), 2), e2], axis=-1)
        c08 = [d2[:1] @ w for w in self.heads08]
        d2 = self._stage("decoder_level2", d2 @ self.heads08[0])
        v = mx.concatenate([_shuffle(conv(d2, r["up2_1"]), 2), e1], axis=-1)
        c04 = []
        for blocks in self.heads04:
            y = v[:1]
            for block in blocks:
                y = block(y)
            c04.append(y)
        mx.eval(c04, c08, c16, v)
        return [c04, c08, [c16, c16]], v

    def _features(self, v: mx.array) -> mx.array:
        """`volume_conv`: a residual block with instance norms, then a convolution; float32."""
        y = _relu(_instance_norm(mx.conv2d(v, self.volume[0], padding=1)))
        y = _relu(_instance_norm(mx.conv2d(y, self.volume[1], padding=1)))
        w, b = self.volume_out
        return (mx.conv2d(_relu(v + y), w, padding=1) + b).astype(mx.float32)

    # update block -----------------------------------------------------------------------------------------------------

    def _update(self, ub: Any) -> None:  # noqa: ANN401 - `LSTMMultiUpdateBlock`
        e = ub.encoder
        self.gates = [self._a(_ohwi(g.gates)) for g in (ub.lstm08, ub.lstm16, ub.lstm32)]
        self.convc1 = (self._a(_np(e.convc1.weight)[:, :, 0, 0].T), self._a(_np(e.convc1.bias)))
        self.convc2 = (self._a(_ohwi(e.convc2)), self._a(_np(e.convc2.bias)))
        self.convf1 = (mx.array(_ohwi(e.convf1)[..., :1]), mx.array(_np(e.convf1.bias)))  # float32; the y flow is 0
        self.convf2 = (self._a(_ohwi(e.convf2)), self._a(_np(e.convf2.bias)))
        w, b = _ohwi(e.conv), _np(e.conv.bias)
        pad = w.shape[3] - w.shape[0]
        self.motion = (self._a(np.pad(w, [(0, pad), (0, 0), (0, 0), (0, 0)])), self._a(np.pad(b, (0, pad))))
        self.flow_slot = self._a(np.eye(w.shape[3], dtype=np.float32)[MOTION])
        f = ub.flow_head
        self.flow1 = (self._a(_ohwi(f.conv1)), self._a(_np(f.conv1.bias)))
        self.flow2 = (self._a(_ohwi(f.conv2)[:1]), self._a(_np(f.conv2.bias)[:1]))  # the x flow
        self.mask1 = (self._a(_ohwi(ub.mask[0])), self._a(_np(ub.mask[0].bias)))
        self.mask2 = (self._a(_np(ub.mask[2].weight)[:, :, 0, 0].T), self._a(_np(ub.mask[2].bias)))
        self.taps = mx.arange(-RADIUS, RADIUS + 1, dtype=mx.float32)[None]

    def _pool(self, x: mx.array) -> mx.array:
        """`avg_pool2d(x, 3, stride=2, padding=1)` (the padding counts)."""
        c = x.shape[-1]
        return ops.depthwise(x, mx.full((3, 3, c), 1 / 9, x.dtype), mx.zeros((c,), x.dtype), (2, 2), ONE, ONE)

    @staticmethod
    def _lstm(w: mx.array, h: mx.array, ctx: mx.array, *x: mx.array) -> mx.array:
        """Upstream's LSTM with c = h: `o * tanh(h * f + i * c~)`, the gates from one conv over `[h | x]`."""
        i, f, c, o = mx.split(mx.conv2d(mx.concatenate([h, *x], axis=-1), w, padding=1) + ctx, 4, axis=-1)
        return mx.sigmoid(o) * mx.tanh(h * mx.sigmoid(f) + mx.sigmoid(i) * mx.tanh(c))

    def _iteration(self, net: list[mx.array], flow: mx.array, ctx: list[mx.array], pyramid: list[mx.array]) -> Any:  # noqa: ANN401
        _, h, w, _ = flow.shape
        x = (mx.arange(w, dtype=mx.float32)[None, :, None] + flow[0]).reshape(h * w, 1)
        corr = [ops.linear_sample(vol, x / 2**i + self.taps) for i, vol in enumerate(pyramid)]
        corr = mx.concatenate(corr, axis=-1).reshape(1, h, w, -1).astype(self.dtype)
        n2 = self._lstm(self.gates[2], net[2], ctx[2], self._pool(net[1]))
        n1 = self._lstm(self.gates[1], net[1], ctx[1], self._pool(net[0]), _interp(n2, net[1].shape[1:3]))
        cor = _relu(corr @ self.convc1[0] + self.convc1[1])
        cor = _relu(mx.conv2d(cor, self.convc2[0], padding=1) + self.convc2[1])
        flo = _relu(mx.conv2d(flow, self.convf1[0], padding=3) + self.convf1[1]).astype(self.dtype)
        flo = _relu(mx.conv2d(flo, self.convf2[0], padding=1) + self.convf2[1])
        motion = _relu(mx.conv2d(mx.concatenate([cor, flo], axis=-1), self.motion[0], padding=1) + self.motion[1])
        motion = motion + flow.astype(self.dtype) * self.flow_slot
        n0 = self._lstm(self.gates[0], net[0], ctx[0], motion, _interp(n1, net[0].shape[1:3]))
        delta = _relu(mx.conv2d(n0, self.flow1[0], padding=1) + self.flow1[1])
        delta = mx.conv2d(delta, self.flow2[0], padding=1) + self.flow2[1]
        return [n0, n1, n2], flow + delta.astype(mx.float32)

    def _upsample(self, flow: mx.array, hidden: mx.array) -> mx.array:
        """The mask head and upstream's convex upsampling (factor 4) of the x flow: `[1, 4 H, 4 W, 1]` float32."""
        _, h, w, _ = flow.shape
        mask = _relu(mx.conv2d(hidden, self.mask1[0], padding=1) + self.mask1[1]) @ self.mask2[0] + self.mask2[1]
        mask = mx.softmax(0.25 * mask.astype(mx.float32).reshape(h, w, 9, 4, 4), axis=2)
        fp = mx.pad(4 * flow[0, :, :, 0], 1)
        near = mx.stack([fp[i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=-1)  # [H, W, 9]
        up = (mask * near[..., None, None]).sum(axis=2)  # [H, W, 4, 4]
        return up.transpose(0, 2, 1, 3).reshape(1, 4 * h, 4 * w, 1)

    # normalization refinement, float32 --------------------------------------------------------------------------------

    def _refinement(self, m: Any) -> None:  # noqa: ANN401 - `NormalizationRefinement`
        def conv(c: Any, norm: Any, slope: float = 0.0) -> _Layer:  # noqa: ANN401
            w, b = (mx.array(a) for a in _folded(c, norm))
            stride, padding, dilation = c.stride[0], c.padding[0], c.dilation[0]
            if c.transposed:
                return lambda x: _relu(mx.conv_transpose2d(x, w, stride=stride, padding=padding) + b)
            if slope:
                return lambda x: mx.maximum(y := mx.conv2d(x, w, stride, padding, dilation) + b, slope * y)
            return lambda x: _relu(mx.conv2d(x, w, stride, padding, dilation) + b)

        def conv2x(c: Any) -> Callable[[mx.array, mx.array], mx.array]:  # noqa: ANN401 - `Conv2x`
            first, second = conv(c.conv1.conv, c.conv1.bn), conv(c.conv2.conv, c.conv2.bn)
            return lambda x, rem: second(mx.concatenate([first(x), rem], axis=-1))

        self.ref_image = conv(m.conv1[0], m.conv1[1], 0.2)
        self.ref_disp = conv(m.conv2[0], m.conv2[1], 0.2)
        self.ref_start = conv(m.conv_start.conv, m.conv_start.bn)
        self.ref_down = [conv(c.conv, c.bn) for c in (m.conv1a, m.conv2a, m.conv3a, m.conv4a)]
        self.ref_up_a = [conv2x(c) for c in (m.deconv4a, m.deconv3a, m.deconv2a, m.deconv1a)]
        self.ref_down_b = [conv2x(c) for c in (m.conv1b, m.conv2b, m.conv3b, m.conv4b)]
        self.ref_up_b = [conv2x(c) for c in (m.deconv4b, m.deconv3b, m.deconv2b, m.deconv1b)]
        self.ref_final = (mx.array(_ohwi(m.final_conv)), mx.array(_np(m.final_conv.bias)))

    def _refine(self, flow: mx.array, left: mx.array, right: mx.array) -> mx.array:
        """`NormalizationRefinement.forward` of the full-resolution x flow `[1, H, W, 1]` (all negative)."""
        _, h, w, _ = flow.shape
        disp = -flow
        x = mx.arange(w, dtype=mx.float32)[None, None, :, None] - disp
        y = mx.broadcast_to(mx.arange(h, dtype=mx.float32)[None, :, None, None], x.shape)
        grid = mx.concatenate([2 * (x / (w - 1)) - 1, 2 * (y / (h - 1)) - 1], axis=-1)
        warped = ops.grid_sample(right, grid, align_corners=False, border=True)
        disp = disp / w * 1024
        x = self.ref_image(mx.concatenate([warped - left, left], axis=-1))
        x = self.ref_start(mx.concatenate([x, self.ref_disp(disp)], axis=-1))
        rems = [x]
        for down in self.ref_down:
            x = down(x)
            rems.append(x)
        for i, up in enumerate(self.ref_up_a):
            x = rems[3 - i] = up(x, rems[3 - i])
        for i, down2 in enumerate(self.ref_down_b):
            x = rems[i + 1] = down2(x, rems[i + 1])
        for i, up in enumerate(self.ref_up_b):
            x = up(x, rems[3 - i])
        new = disp + mx.conv2d(x, self.ref_final[0], padding=1) + self.ref_final[1]
        return -(mx.maximum(new, 0.01 * new) / 1024 * w)

    # the network ------------------------------------------------------------------------------------------------------

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32) of a padded float32 pair `[1, 3, H, W]` in 0-255."""
        left = 2 * (left.transpose(0, 2, 3, 1) / 255.0) - 1.0
        right = 2 * (right.transpose(0, 2, 3, 1) / 255.0) - 1.0
        maps, v = self._extract(mx.concatenate([left, right], axis=0).astype(self.dtype))
        fmap = self._features(v)
        net = [mx.tanh(c[0]) for c in maps]
        ctx = [mx.conv2d(_relu(c[1]), w, padding=1) + b for c, (w, b) in zip(maps, self.context, strict=True)]
        _, h, w, d = fmap.shape
        corr = (fmap[0] @ mx.swapaxes(fmap[1], 1, 2)).reshape(h * w, w) * d**-0.5  # row by row: [H W, W]
        pyramid = [corr]
        for _ in range(1, LEVELS):
            n = corr.shape[1] // 2
            corr = corr[:, : 2 * n].reshape(-1, n, 2).mean(axis=-1)
            pyramid.append(corr)
        flow = mx.zeros((1, h, w, 1), mx.float32)
        mx.eval(net, ctx, pyramid)
        del maps, v, fmap
        for _ in range(self.iters):
            net, flow = self._step(net, flow, ctx, pyramid)
            mx.eval(net, flow)
        flow = self._upsample(flow, net[0])
        mx.eval(flow)
        if not mx.any(flow >= 0).item():  # upstream refines only if the largest flow is negative
            flow = flow + self._refine(flow, left, right)
        return -flow[..., 0]
