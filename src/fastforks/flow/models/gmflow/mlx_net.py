"""GMFlow on Apple MLX, written in `mlx.core` operations (`MlxGMFlow(net, dtype, scales, refine)`): pair -> flow.

The maths of `model.forward` (upstream `UniMatch.forward`, `task="flow"`), channels last, in place of the export and
graph translation the MLX runner replayed before. The run dtype covers what autocast covers on CUDA:

- the CNN in the run dtype, one image at a time; its instance norms from float32 statistics (`ops.channel_stats`),
  so a convolution before one drops its bias. The trident convolution of `scale2` is one weight at strides 1 and 2.
- the Transformer in region order (`model._window_orders`: a shifted layer attends within the regions upstream's
  mask leaves, one attention per region size); its residual stream float32, projections, attention, merge, norms and
  MLP in the run dtype.
- float32: positions, matching, propagation, warping, the correlations and the flow. Global matching and global
  propagation are one fused attention each, the 2-channel value zero-padded to the query's width (MLX then writes no
  `[HW, HW]` scores). Local matching, local propagation and the refinement's correlation are `ops.corr_windows`
  (integer windows, zeros outside) in its x-major window order; the refinement's first convolution takes its input
  channels in that order.
- the refinement GRU convolves the context (constant over the refinements) once per direction; each refinement is
  one compiled function.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops

from .model import SCALES, _window_orders

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    _Layer = Callable[[mx.array], mx.array]
    _Wb = tuple[mx.array, mx.array | None]

__all__ = ["MlxGMFlow"]

REFINE_RADIUS = 4  # as `model.REFINE_RADIUS`
_FFN_TOKENS = 32768  # tokens per FFN chunk (1080p scale 2: 4 chunks of the 2 x 129600 tokens)


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().float().numpy().astype(np.float64)


def _ohwi(w: np.ndarray) -> np.ndarray:
    return w.transpose(0, 2, 3, 1)


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _instance_norm(x: mx.array) -> mx.array:
    """`InstanceNorm2d` (no affine, eps 1e-5) of one NHWC image: float32 statistics, a multiply-add in its dtype."""
    mean, var = ops.channel_stats(x.reshape(-1, x.shape[-1]))
    scale = mx.rsqrt(var + 1e-5)
    return x * scale.astype(x.dtype) - (mean * scale).astype(x.dtype)


@mx.compile
def _gelu(x: mx.array) -> mx.array:
    """Exact GELU (`nn.GELU()`), one kernel: unfused, its temporaries are four `[2B, hw, 8C]` arrays."""
    return x * (1 + mx.erf(x * (1 / math.sqrt(2)))) / 2


def _grid(h: int, w: int) -> mx.array:
    """Pixel coordinates `[h, w, 2]` (x, y), float32."""
    x = mx.broadcast_to(mx.arange(w, dtype=mx.float32)[None], (h, w))
    y = mx.broadcast_to(mx.arange(h, dtype=mx.float32)[:, None], (h, w))
    return mx.stack([x, y], axis=-1)


def _window_offsets(radius: int) -> np.ndarray:
    """`[(2r + 1)^2, 2]` (dx, dy) of `ops.corr_windows`' window order (x major)."""
    r = np.arange(-radius, radius + 1, dtype=np.float32)
    dx, dy = np.meshgrid(r, r, indexing="ij")
    return np.stack([dx.reshape(-1), dy.reshape(-1)], -1)


def _attend(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """`softmax(q k^T / sqrt(C)) v` of `[B, N, C]`, `[B, M, C]`, `[B, M, V]` (V <= C), fused: V padded to C."""
    c, channels = q.shape[-1], v.shape[-1]
    v = mx.pad(v, ((0, 0), (0, 0), (0, c - channels)))
    out = mx.fast.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], scale=c**-0.5)
    return out[:, 0, :, :channels]


def _position(h: int, w: int, splits: int, channels: int) -> np.ndarray:
    """`feature_add_position`'s sine positions `[h, w, C]`: each of the `splits` x `splits` windows its own."""
    hs, ws = h // splits, w // splits
    feats = channels // 2
    eps, scale = 1e-6, 2 * math.pi
    y = np.arange(1, hs + 1, dtype=np.float32)[:, None].repeat(ws, 1)
    x = np.arange(1, ws + 1, dtype=np.float32)[None].repeat(hs, 0)
    y, x = y / (hs + eps) * scale, x / (ws + eps) * scale
    dim = np.arange(feats, dtype=np.float32)
    dim = (10000 ** (2 * (dim // 2) / feats)).astype(np.float32)
    px, py = x[..., None] / dim, y[..., None] / dim
    px = np.stack((np.sin(px[..., 0::2]), np.cos(px[..., 1::2])), -1).reshape(hs, ws, -1)
    py = np.stack((np.sin(py[..., 0::2]), np.cos(py[..., 1::2])), -1).reshape(hs, ws, -1)
    return np.tile(np.concatenate((py, px), -1), (splits, splits, 1)).astype(np.float32)


class MlxGMFlow:
    """`model.forward` of `net` (a loaded `UniMatch`, float32 weights) in `dtype` (the name)."""

    def __init__(self, net: Any, dtype: str, scales: int, refine: int) -> None:  # noqa: ANN401 - `UniMatch`
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        self.scales, self.refine = SCALES[scales], refine
        self.channels, self.factor = net.feature_channels, net.upsample_factor
        self._cnn(net.backbone)
        self.layers = [(self._attn(blk.self_attn), self._attn(blk.cross_attn_ffn)) for blk in net.transformer.layers]
        fa = net.feature_flow_attn
        self.prop_q = self._lin(fa.q_proj, mx.float32)
        self.prop_k = self._lin(fa.k_proj, mx.float32)
        if refine:
            self.refine_proj = self._lin1(net.refine_proj)
            self._refine(net.refine)
            self._step = mx.compile(self._refinement)
        else:
            up = net.upsampler
            self.up1 = self._conv(up[0])
            self.up2 = self._lin1(up[2])
        mx.eval(self._arrays())
        mx.clear_cache()

    # weights ----------------------------------------------------------------------------------------------------------

    def _a(self, a: np.ndarray, dtype: mx.Dtype | None = None) -> mx.array:
        return mx.array(np.ascontiguousarray(a, dtype=np.float32)).astype(dtype or self.dtype)

    def _conv(self, conv: Any, dtype: mx.Dtype | None = None) -> _Wb:  # noqa: ANN401 - a `Conv2d`
        b = None if conv.bias is None else self._a(_np(conv.bias), dtype)
        return self._a(_ohwi(_np(conv.weight)), dtype), b

    def _lin1(self, conv: Any) -> _Wb:  # noqa: ANN401 - a 1 x 1 `Conv2d`
        return self._a(_np(conv.weight)[:, :, 0, 0].T), self._a(_np(conv.bias))

    def _lin(self, lin: Any, dtype: mx.Dtype | None = None) -> _Wb:  # noqa: ANN401 - a `Linear`
        b = None if lin.bias is None else self._a(_np(lin.bias), dtype)
        return self._a(_np(lin.weight).T, dtype), b

    def _arrays(self) -> list[mx.array]:
        out: list[mx.array] = []

        def walk(v: object) -> None:
            if isinstance(v, mx.array):
                out.append(v)
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)

        walk([v for k, v in self.__dict__.items() if k != "_step"])
        return out

    # CNN --------------------------------------------------------------------------------------------------------------

    def _cnn(self, bb: Any) -> None:  # noqa: ANN401 - `CNNEncoder`
        def conv(c: Any) -> tuple[mx.array, int, int, int]:  # noqa: ANN401 - a bias-free `Conv2d` (instance norm next)
            return self._a(_ohwi(_np(c.weight))), c.stride[0], c.padding[0], c.dilation[0]

        self.stem = conv(bb.conv1)
        self.blocks = []
        for layer in (bb.layer1, bb.layer2, bb.layer3):
            for blk in layer:
                down = None
                if blk.downsample is not None:
                    d = blk.downsample[0]
                    down = (self._a(_np(d.weight)[:, :, 0, 0].T), d.stride[0])  # its bias: cancelled by the norm
                self.blocks.append((conv(blk.conv1), conv(blk.conv2), down))
        self.cnn_out = self._lin1(bb.conv2)
        self.trident = None
        if bb.num_branch > 1:
            tc = bb.trident_conv
            self.trident = (self._a(_ohwi(_np(tc.weight))), tuple(s[0] for s in tc.strides))

    def _extract(self, x: mx.array) -> list[mx.array]:
        """`CNNEncoder.forward` of one NHWC image: features from low to high resolution."""

        def unit(y: mx.array, c: tuple[mx.array, int, int, int]) -> mx.array:
            w, s, p, d = c
            return _relu(_instance_norm(mx.conv2d(y, w, s, p, d)))

        x = unit(x, self.stem)
        for c1, c2, down in self.blocks:
            y = unit(unit(x, c1), c2)
            if down is not None:
                w, s = down
                x = _instance_norm((x[:, ::s, ::s] if s > 1 else x) @ w)
            x = _relu(x + y)
            mx.eval(x)
        x = x @ self.cnn_out[0] + self.cnn_out[1]
        if self.trident is None:
            return [x]
        w, strides = self.trident
        return [mx.conv2d(x, w, s, 1) for s in strides][::-1]

    # Transformer ------------------------------------------------------------------------------------------------------

    def _attn(self, layer: Any) -> dict[str, Any]:  # noqa: ANN401 - a vendored `TransformerLayer`
        wq, wk, wv = (_np(p.weight).T for p in (layer.q_proj, layer.k_proj, layer.v_proj))
        p: dict[str, Any] = {
            "q": self._a(wq),
            "kv": self._a(np.concatenate([wk, wv], 1)),
            "merge": self._a(_np(layer.merge.weight).T),
            "norm1": (self._a(_np(layer.norm1.weight)), self._a(_np(layer.norm1.bias)), layer.norm1.eps),
        }
        if not layer.no_ffn:
            p["fc1"] = self._a(_np(layer.mlp[0].weight).T)
            p["fc2"] = self._a(_np(layer.mlp[2].weight).T)
            p["norm2"] = (self._a(_np(layer.norm2.weight)), self._a(_np(layer.norm2.bias)), layer.norm2.eps)
        return p

    @staticmethod
    def _layer(
        p: dict[str, Any], source: mx.array, query: mx.array, target: mx.array, groups: Sequence[tuple[int, int, int]]
    ) -> mx.array:
        """`model._region_layer`: float32 `source`, run-dtype `query` / `target`, tokens in `groups` order."""
        c = query.shape[-1]
        q = query @ p["q"]
        kv = target @ p["kv"]
        k, v = kv[..., :c], kv[..., c:]
        b = q.shape[0]
        parts = []
        for start, count, size in groups:
            qa, ka, va = (x[:, start : start + count * size].reshape(b, count, size, c) for x in (q, k, v))
            parts.append(mx.fast.scaled_dot_product_attention(qa, ka, va, scale=c**-0.5).reshape(b, count * size, c))
        message = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        w, bias, eps = p["norm1"]
        message = mx.fast.layer_norm(message @ p["merge"], w, bias, eps)
        if "fc1" in p:
            w, bias, eps = p["norm2"]
            n = query.shape[1]
            outs = []
            for s in range(0, n, _FFN_TOKENS):  # per token: chunks bound the `[2B, n, 8C]` hidden temporaries
                x = mx.concatenate([query[:, s : s + _FFN_TOKENS], message[:, s : s + _FFN_TOKENS]], axis=-1)
                outs.append(mx.fast.layer_norm(_gelu(x @ p["fc1"]) @ p["fc2"], w, bias, eps))
            message = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
        return source + message.astype(mx.float32)

    def _transformer(self, f1: mx.array, f2: mx.array, splits: int) -> tuple[mx.array, mx.array]:
        """`model._region_transformer` of float32 NHWC `[B, h, w, C]` features."""
        b, h, w, c = f1.shape
        orders, groups = _window_orders(h, w, splits, "cpu")  # model's torch index tensors (cached per shape)
        to_plain, plain_to_shifted, shifted_to_plain, shifted_back, plain_back = (
            mx.array(t.numpy().astype(np.int32)) for t in orders
        )
        plain = ((0, splits**2, h * w // splits**2),)
        tokens = mx.take(mx.concatenate([f1, f2]).reshape(2 * b, h * w, c), to_plain, axis=1)
        shifted = False
        pending: mx.array | None = None
        for i, (sa, ca) in enumerate(self.layers):
            if (i % 2 == 1) != shifted:
                tokens = mx.take(tokens, shifted_to_plain if shifted else plain_to_shifted, axis=1)
                shifted = not shifted
            g = groups if shifted else plain
            low = tokens.astype(self.dtype)
            other = mx.concatenate([low[b:], low[:b]])  # the cross-attention target: the other image before the block
            tokens = self._layer(sa, tokens, low, low, g)
            tokens = self._layer(ca, tokens, tokens.astype(self.dtype), other, g)
            mx.async_eval(tokens)  # one layer in flight: a lazy stack of layers holds every layer's MLP temporaries
            if pending is not None:
                mx.eval(pending)
            pending = tokens
        tokens = mx.take(tokens, shifted_back if shifted else plain_back, axis=1).reshape(2 * b, h, w, c)
        return tokens[:b], tokens[b:]

    # refinement -------------------------------------------------------------------------------------------------------

    def _refine(self, rb: Any) -> None:  # noqa: ANN401 - `BasicUpdateBlock`
        e = rb.encoder
        k = 2 * REFINE_RADIUS + 1
        w = _np(e.convc1.weight)[:, :, 0, 0]  # input channels in upstream's y-major window order
        order = np.arange(k * k).reshape(k, k).T.reshape(-1)  # corr_windows' x-major index -> upstream's
        self.convc1 = (self._a(w[:, order].T), self._a(_np(e.convc1.bias)))
        self.convc2 = self._conv(e.convc2)
        self.convf1 = self._conv(e.convf1, mx.float32)
        self.convf2 = self._conv(e.convf2)
        self.motion = self._conv(e.conv)
        gru = rb.gru
        hidden = gru.convz1.out_channels
        self.grus = []
        for z, r, q in ((gru.convz1, gru.convr1, gru.convq1), (gru.convz2, gru.convr2, gru.convq2)):
            wz, wr, wq = (_ohwi(_np(c.weight)) for c in (z, r, q))  # inputs: hidden, context, motion
            loop = lambda w_: np.concatenate([w_[..., :hidden], w_[..., 2 * hidden :]], -1)  # noqa: E731
            ctx = np.concatenate([w_[..., hidden : 2 * hidden] for w_ in (wz, wr, wq)])
            bias = np.concatenate([_np(c.bias) for c in (z, r, q)])
            self.grus.append({
                "pad": tuple(z.padding),
                "zr": self._a(np.concatenate([loop(wz), loop(wr)])),
                "q": self._a(loop(wq)),
                "ctx": (self._a(ctx), self._a(bias)),
            })  # fmt: skip
        self.head1 = self._conv(rb.flow_head.conv1)
        self.head2 = self._conv(rb.flow_head.conv2)
        self.mask1 = self._conv(rb.mask[0])
        self.mask2 = self._lin1(rb.mask[2])

    def _refinement(
        self, hidden: mx.array, flow: mx.array, shares: list[mx.array], raw1: mx.array, raw2: mx.array
    ) -> tuple[mx.array, mx.array]:
        """One refinement: hidden state and float32 flow `[B, h, w, 2]` -> the next."""
        _, h, w, _ = flow.shape
        coords = (_grid(h, w)[None] + flow).transpose(0, 3, 1, 2)
        corr = ops.corr_windows(raw1, [raw2], coords, REFINE_RADIUS, raw1.shape[-1] ** -0.5)
        corr = corr.transpose(0, 2, 3, 1).astype(self.dtype)
        cor = _relu(corr @ self.convc1[0] + self.convc1[1])
        cor = _relu(mx.conv2d(cor, self.convc2[0], padding=1) + self.convc2[1])
        flo = _relu(mx.conv2d(flow, self.convf1[0], padding=3) + self.convf1[1]).astype(self.dtype)
        flo = _relu(mx.conv2d(flo, self.convf2[0], padding=1) + self.convf2[1])
        m = _relu(mx.conv2d(mx.concatenate([cor, flo], axis=-1), self.motion[0], padding=1) + self.motion[1])
        motion = mx.concatenate([m, flow.astype(self.dtype)], axis=-1)
        c = hidden.shape[-1]
        for g, share in zip(self.grus, shares, strict=True):
            zr = mx.conv2d(mx.concatenate([hidden, motion], axis=-1), g["zr"], padding=g["pad"]) + share[..., : 2 * c]
            zr = mx.sigmoid(zr)
            z, r = zr[..., :c], zr[..., c:]
            q = mx.conv2d(mx.concatenate([r * hidden, motion], axis=-1), g["q"], padding=g["pad"]) + share[..., 2 * c :]
            hidden = (1 - z) * hidden + z * mx.tanh(q)
        y = _relu(mx.conv2d(hidden, self.head1[0], padding=1) + self.head1[1])
        delta = mx.conv2d(y, self.head2[0], padding=1) + self.head2[1]
        return hidden, flow + delta.astype(mx.float32)

    # the network ------------------------------------------------------------------------------------------------------

    def _upsample(self, flow: mx.array, mask: mx.array) -> mx.array:
        """Convex upsampling (`upsample_flow_with_mask`) of float32 `[B, h, w, 2]` by `mask` `[B, h, w, 9 K K]`."""
        b, h, w, _ = flow.shape
        k = self.factor
        mask = mx.softmax(mask.astype(mx.float32).reshape(b, h, w, 9, k, k), axis=3)
        fp = mx.pad(k * flow, ((0, 0), (1, 1), (1, 1), (0, 0)))
        near = mx.stack([fp[:, i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=3)  # [B, h, w, 9, 2]
        up = (mask[..., None] * near[:, :, :, :, None, None]).sum(axis=3)  # [B, h, w, K, K, 2]
        return up.transpose(0, 1, 3, 2, 4, 5).reshape(b, k * h, k * w, 2)

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[B, 2, H, W]` (float32) of an ImageNet-normalised float32 pair `[B, 3, H, W]` (padded)."""
        b = image1.shape[0]
        images = mx.concatenate([image1, image2]).transpose(0, 2, 3, 1).astype(self.dtype)
        per_image = [self._extract(images[i : i + 1]) for i in range(2 * b)]
        feats = [mx.concatenate([f[s] for f in per_image]) for s in range(len(self.scales))]
        mx.eval(feats)
        del per_image
        flow = feature1 = mx.zeros((0,))
        for index, (splits, corr_radius, prop_radius) in enumerate(self.scales):
            raw = feats[index].astype(mx.float32)
            feature1, feature2 = raw[:b], raw[b:]
            _, h, w, c = feature1.shape
            if index > 0:
                fh, fw = flow.shape[1:3]
                flow = 2 * ops.bilinear_nhwc(flow, (h, w), ((fh - 1) / (h - 1), (fw - 1) / (w - 1)), align=True)
                g = _grid(h, w)[None] + flow
                norm = mx.array([2 / (w - 1), 2 / (h - 1)], mx.float32)
                feature2 = ops.grid_sample(feature2, g * norm - 1, align_corners=True, border=False)
            pos = mx.array(_position(h, w, splits, c))
            feature1, feature2 = self._transformer(feature1 + pos, feature2 + pos, splits)
            mx.eval(feature1, feature2)
            grid = _grid(h, w)[None]
            if corr_radius == -1:  # global matching: the expected position of the match
                tokens = grid.reshape(1, h * w, 2)
                match = _attend(feature1.reshape(b, h * w, c), feature2.reshape(b, h * w, c), mx.broadcast_to(
                    tokens, (b, h * w, 2)))  # fmt: skip
                predicted = match.reshape(b, h, w, 2) - grid
            else:
                predicted = self._local_match(feature1, feature2, corr_radius)
            flow = flow + predicted if index > 0 else predicted
            if prop_radius > 0:
                flow = self._local_propagate(feature1, flow, prop_radius)
            else:  # global propagation; upstream projects the key from the projected query (its own note)
                query = feature1.reshape(b, h * w, c) @ self.prop_q[0] + self.prop_q[1]
                key = query @ self.prop_k[0] + self.prop_k[1]
                flow = _attend(query, key, flow.reshape(b, h * w, 2)).reshape(b, h, w, 2)
            mx.eval(flow)
        if not self.refine:
            x = mx.concatenate([flow, feature1], axis=-1).astype(self.dtype)
            mask = _relu(mx.conv2d(x, self.up1[0], padding=1) + self.up1[1]) @ self.up2[0] + self.up2[1]
            return self._result(self._upsample(flow, mask))
        hc = feature1.astype(self.dtype) @ self.refine_proj[0] + self.refine_proj[1]
        hidden, context = mx.tanh(hc[..., : hc.shape[-1] // 2]), _relu(hc[..., hc.shape[-1] // 2 :])
        shares = [mx.conv2d(context, g["ctx"][0], padding=g["pad"]) + g["ctx"][1] for g in self.grus]
        last = len(self.scales) - 1
        raw1, raw2 = feats[last][:b], feats[last][b:]
        mx.eval(hidden, shares)
        state = hidden
        for _ in range(self.refine):  # upstream restarts every step from the projected state (`refine_proj` again)
            state, flow = self._step(hidden, flow, shares, raw1, raw2)
            mx.eval(state, flow)
        y = _relu(mx.conv2d(state, self.mask1[0], padding=1) + self.mask1[1])
        return self._result(self._upsample(flow, y @ self.mask2[0] + self.mask2[1]))

    @staticmethod
    def _result(up: mx.array) -> mx.array:
        out = up.transpose(0, 3, 1, 2)
        mx.eval(out)
        return out

    def _local_match(self, f1: mx.array, f2: mx.array, radius: int) -> mx.array:
        """`local_correlation_softmax`: the expected offset over the `(2r + 1)^2` window inside the image."""
        b, h, w, c = f1.shape
        grid = _grid(h, w)
        corr = ops.corr_windows(f1, [f2], mx.broadcast_to(grid.transpose(2, 0, 1)[None], (b, 2, h, w)), radius, c**-0.5)
        off = _window_offsets(radius)
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        tx, ty = gx[None] + off[:, 0, None, None], gy[None] + off[:, 1, None, None]
        invalid = (tx < 0) | (tx >= w) | (ty < 0) | (ty >= h)
        corr = corr + mx.array(np.where(invalid, -1e9, 0).astype(np.float32))[None]
        prob = mx.softmax(corr, axis=1)  # [B, K, h, w]
        return mx.einsum("bkhw,kd->bhwd", prob, mx.array(off))

    def _local_propagate(self, f1: mx.array, flow: mx.array, radius: int) -> mx.array:
        """`SelfAttnPropagation.forward_local_window_attn`: zero-padded windows (scores 0 there, not masked)."""
        b, h, w, c = f1.shape
        q = f1 @ self.prop_q[0] + self.prop_q[1]
        k = f1 @ self.prop_k[0] + self.prop_k[1]
        grid = mx.broadcast_to(_grid(h, w).transpose(2, 0, 1)[None], (b, 2, h, w))
        scores = ops.corr_windows(q, [k], grid, radius, c**-0.5)  # [B, K, h, w], 0 outside
        prob = mx.softmax(scores, axis=1)
        fp = mx.pad(flow, ((0, 0), (radius, radius), (radius, radius), (0, 0)))
        n = 2 * radius + 1
        win = mx.stack([fp[:, j : j + h, i : i + w] for i in range(n) for j in range(n)], axis=1)  # x major
        return (prob[..., None] * win).sum(axis=1)
