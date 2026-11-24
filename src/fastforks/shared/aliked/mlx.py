"""ALIKED on Apple MLX, written in `mlx.core` operations (`MlxAliked(net)`): image -> keypoints, descriptors.

The steps of `ALIKED.forward` (`net/aliked.py`) without the full-resolution 128-channel map, as `lazy.py` computes
them: the four feature levels stay at their own resolution, the score head's first (1 x 1, linear) convolution runs
per level before the upsampling, and the normalized descriptor map is evaluated only at the pixels SDDH reads (a
3 x 3 patch and 16 bilinear samples per keypoint). An exported graph of `detection_maps` forms that map
(`[1, 128, H, W]`, 400 MB at 768 x 1024): its concatenation, norm and division were 95 ms of the 164 ms per image.

- the dense part (encoder with its two deformable blocks, level convolutions, score head, the score NMS) is one
  compiled function per image size: batch norm folded, channels last, float32;
- the keypoints are selected on the host from the NMS map (one read per image, with the score map for the 5 x 5
  soft-argmax refinement): a few thousand candidates;
- SDDH (patches, offsets, samples, aggregation) runs on MLX; the descriptors stay there for the matcher.

Results equal the PyTorch module's up to float32 rounding (the score map's first convolution sums in another order).
"""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import bilinear_nhwc, grid_sample

if TYPE_CHECKING:
    from .net.aliked import ALIKED

__all__ = ["MlxAliked"]

_F = np.float32
_DIV_BY = 32  # fixed: `extract_dense_map`'s padding
_EPS = 1e-12  # fixed: `F.normalize`'s default eps (upstream's call)
_SELU_ALPHA = 1.6732632423543772  # fixed: SELU's constants (`torch.nn.SELU`)
_SELU_SCALE = 1.0507009873554805  # fixed: SELU's constants (`torch.nn.SELU`)
_TEMPERATURE = 0.1  # fixed: `DKD.temperature`

_Conv = tuple[mx.array, mx.array | None, int]  # weight [O, kh, kw, I], bias, padding
_Maps = tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]


def _selu(x: mx.array) -> mx.array:
    return _SELU_SCALE * mx.where(x > 0, x, _SELU_ALPHA * mx.expm1(mx.minimum(x, 0)))


def _folded(conv: Any, norm: Any) -> tuple[np.ndarray, np.ndarray | None]:  # noqa: ANN401 - torch modules of the net
    """A convolution's weight `[O, I, kh, kw]` and bias with its `BatchNorm2d` (or None) folded in."""
    w = conv.weight.detach().float().numpy()
    b = None if conv.bias is None else conv.bias.detach().float().numpy()
    if norm is not None:
        scale = norm.weight.detach().numpy().astype(np.float64) / np.sqrt(
            norm.running_var.numpy().astype(np.float64) + norm.eps
        )
        mean = norm.running_mean.numpy() - (0 if b is None else b)
        w = (w * scale[:, None, None, None]).astype(_F)
        b = (norm.bias.detach().numpy() - mean * scale).astype(_F)
    return w, b


def _conv(conv: Any, norm: Any = None) -> _Conv:  # noqa: ANN401 - torch modules of the vendored net
    w, b = _folded(conv, norm)
    return mx.array(w.transpose(0, 2, 3, 1)), None if b is None else mx.array(b), int(conv.padding[0])


def _run(x: mx.array, conv: _Conv) -> mx.array:
    w, b, padding = conv
    y = mx.conv2d(x, w, padding=padding)
    return y if b is None else y + b


class _Deform:
    """`DeformableConv2d` (3 x 3, no mask) with the batch norm after it: offsets, bilinear samples, one product."""

    def __init__(self, conv: Any, norm: Any) -> None:  # noqa: ANN401 - torch modules of the vendored net
        if conv.mask:
            msg = "MlxAliked: the masked deformable convolution is not ported"
            raise NotImplementedError(msg)
        self.offset = _conv(conv.offset_conv)
        w, b = _folded(conv.regular_conv, norm)
        o, c, kh, kw = w.shape
        self.kernel = (kh, kw)
        self.padding = int(conv.regular_conv.padding[0])
        self.weight = mx.array(w.transpose(2, 3, 1, 0).reshape(kh * kw * c, o))  # [(tap, channel), O]
        self.bias = None if b is None else mx.array(b)

    def __call__(self, x: mx.array) -> mx.array:
        """`[1, H, W, C]` -> `[1, H, W, O]`: `fastforks.shared.deform_conv`'s sampling grid, per tap."""
        _, h, w, c = x.shape
        kh, kw = self.kernel
        k = kh * kw
        limit = max(h, w) / 4.0
        off = mx.clip(_run(x, self.offset), -limit, limit)[0].reshape(h, w, k, 2)  # (dy, dx) per tap
        tap = np.arange(k)
        ty, tx = mx.array((tap // kw).astype(_F)), mx.array((tap % kw).astype(_F))
        gy = mx.array(np.arange(h, dtype=_F) - self.padding)[:, None, None]
        gx = mx.array(np.arange(w, dtype=_F) - self.padding)[None, :, None]
        sy, sx = 2.0 / max(h - 1, 1), 2.0 / max(w - 1, 1)
        grid = mx.stack(((gx + tx + off[..., 1]) * sx - 1.0, (gy + ty + off[..., 0]) * sy - 1.0), axis=-1)
        grid = grid.transpose(2, 0, 1, 3).reshape(1, k * h, w, 2)  # tap-major rows
        cols = grid_sample(x, grid, align_corners=True, border=False).reshape(k, h * w, c)
        y = cols.transpose(1, 0, 2).reshape(h * w, k * c) @ self.weight
        return (y if self.bias is None else y + self.bias).reshape(1, h, w, -1)


def _pool(x: mx.array, k: int) -> mx.array:
    """`AvgPool2d(k, k)` of `[1, H, W, C]`."""
    _, h, w, c = x.shape
    return x.reshape(1, h // k, k, w // k, k, c).mean(axis=(2, 4))


def _local_max(x: mx.array, radius: int) -> mx.array:
    """`max_pool2d(2 * radius + 1, stride=1, padding=radius)` of `[H, W]`: a running maximum per axis."""
    for axis in (0, 1):
        n = x.shape[axis]
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        p = mx.pad(x, pad, constant_values=-float("inf"))
        x = reduce(mx.maximum, [p[k : k + n] if axis == 0 else p[:, k : k + n] for k in range(2 * radius + 1)])
    return x


def _nms(scores: mx.array, radius: int) -> mx.array:
    """`simple_nms` of `[H, W]`, then `DKD`'s border removal: the score where a pixel is a local maximum, else 0."""
    h, w = scores.shape
    max_mask = scores == _local_max(scores, radius)
    for _ in range(2):
        supp_mask = _local_max(max_mask.astype(mx.float32), radius) > 0
        supp_scores = mx.where(supp_mask, 0.0, scores)
        new_max_mask = supp_scores == _local_max(supp_scores, radius)
        max_mask = max_mask | (new_max_mask & ~supp_mask)
    inner = np.zeros((h, w), bool)
    inner[radius : h - radius, radius : w - radius] = True
    return mx.where(max_mask & mx.array(inner), scores, 0.0)


class MlxAliked:
    """The extractor `net` (eval, weights loaded) on MLX: `(image)` -> keypoints in pixels and their descriptors."""

    def __init__(self, net: ALIKED) -> None:
        """Copy the weights of `net` to MLX (batch norm folded, channels last)."""
        head, dkd = net.desc_head, net.dkd
        if head.conv2D or head.mask or head.kernel_size <= 1:
            msg = "MlxAliked covers ALIKED's released SDDH (kernel_size 3, no mask, no conv2D)"
            raise NotImplementedError(msg)
        if dkd.top_k > 0 or dkd.scores_th <= 0:
            msg = "MlxAliked selects keypoints by a positive score threshold (no top-k mode)"
            raise NotImplementedError(msg)
        self.describes = bool(net.describe)
        self.radius, self.threshold, self.n_limit = int(dkd.radius), float(dkd.scores_th), int(dkd.n_limit)
        self.block1 = [_conv(net.block1.conv1, net.block1.bn1), _conv(net.block1.conv2, net.block1.bn2)]
        b2 = net.block2
        self.block2 = [_conv(b2.conv1, b2.bn1), _conv(b2.conv2, b2.bn2), _conv(b2.downsample)]
        self.deform = [
            (_Deform(b.conv1, b.bn1), _Deform(b.conv2, b.bn2), _conv(b.downsample)) for b in (net.block3, net.block4)
        ]
        self.level_convs = [_conv(c) for c in (net.conv1, net.conv2, net.conv3, net.conv4)]
        first, rest = net.score_head[0], [m for m in net.score_head[1:] if hasattr(m, "weight")]
        w = first.weight.detach().float().numpy()[:, :, 0, 0]  # [8, 128]: one slice per level
        widths = [int(c[0].shape[0]) for c in self.level_convs]
        self.score_first = [mx.array(s.T.copy()) for s in np.split(w, np.cumsum(widths)[:-1], axis=1)]
        self.score_rest = [_conv(m) for m in rest]
        # SDDH
        self.n_pos, self.patch = int(head.n_pos), int(head.kernel_size)
        o0, o2 = head.offset_conv[0], head.offset_conv[2]
        w0 = o0.weight.detach().float().numpy()  # [2 * n_pos, C, ps, ps]
        self.offset0 = (mx.array(w0.transpose(2, 3, 1, 0).reshape(-1, len(w0))), mx.array(o0.bias.detach().numpy()))
        self.offset1 = (
            mx.array(o2.weight.detach().float().numpy()[:, :, 0, 0].T.copy()),
            mx.array(o2.bias.detach().numpy()),
        )
        self.sf = mx.array(head.sf_conv.weight.detach().float().numpy()[:, :, 0, 0].T.copy())
        agg = head.agg_weights.detach().float().numpy()  # [n_pos, C, D]
        self.agg = mx.array(agg.reshape(-1, agg.shape[-1]))
        self._compiled = mx.compile(self._maps)

    # ------------------------------------------------------------------ dense maps

    def _maps(self, image: mx.array) -> _Maps:
        x = image.transpose(0, 2, 3, 1)
        h, w = x.shape[1:3]
        top, left = self._pad(h, w)
        ph, pw = -(-h // _DIV_BY) * _DIV_BY, -(-w // _DIV_BY) * _DIV_BY
        if (ph, pw) != (h, w):  # `InputPadder.pad`: replicate
            x = mx.pad(x, [(0, 0), (top, ph - h - top), (left, pw - w - left), (0, 0)], mode="edge")
        x1 = _selu(_run(_selu(_run(x, self.block1[0])), self.block1[1]))
        x2 = _pool(x1, 2)
        x2 = _selu(_run(_selu(_run(x2, self.block2[0])), self.block2[1]) + _run(x2, self.block2[2]))
        feats = [x1, x2]
        x = x2
        for conv1, conv2, down in self.deform:
            x = _pool(x, 4)
            x = _selu(conv2(_selu(conv1(x))) + _run(x, down))
            feats.append(x)
        levels = [_selu(_run(f, c)) for f, c in zip(feats, self.level_convs, strict=True)]
        # the score head's first convolution per level, then the upsampling (both linear)
        s = None
        for level, weight in zip(levels, self.score_first, strict=True):
            y = level @ weight
            if y.shape[1:3] != (ph, pw):
                ratios = ((y.shape[1] - 1) / (ph - 1), (y.shape[2] - 1) / (pw - 1))
                y = bilinear_nhwc(y, (ph, pw), ratios, align=True)
            s = y if s is None else s + y
        for conv in self.score_rest:
            s = _run(_selu(s), conv)
        score = mx.sigmoid(s)[0, top : top + h, left : left + w, 0]
        return levels[0][0], levels[1][0], levels[2][0], levels[3][0], score, _nms(score, self.radius)

    @staticmethod
    def _pad(h: int, w: int) -> tuple[int, int]:
        """`InputPadder`'s padding above and left of an `h x w` image."""
        return (-h % _DIV_BY) // 2, (-w % _DIV_BY) // 2

    def maps(self, image: mx.array) -> _Maps:
        """Of a float32 RGB image `[1, 3, H, W]`: the four feature levels, the score map and its NMS, queued.

        The levels are `[H', W', 32]` maps of the image padded to a multiple of 32; score and NMS are `[H, W]`.
        """
        out = self._compiled(image)
        mx.async_eval(*out)
        return out

    # ------------------------------------------------------------------ keypoints

    def keypoints(self, maps: _Maps) -> np.ndarray:
        """`DKD.forward`'s keypoints of `maps`, `[N, 2]` float32 `(x, y)` normalised to -1..1 (host)."""
        score, nms = np.asarray(maps[4]), np.asarray(maps[5])
        h, w = score.shape
        flat = nms.reshape(-1)
        found = np.flatnonzero(flat > self.threshold)
        if len(found) == 0:
            found = np.flatnonzero(flat > score.mean(dtype=_F))
        if len(found) > self.n_limit:
            found = found[np.argsort(-score.reshape(-1)[found], kind="stable")[: self.n_limit]]
        ys, xs = np.divmod(found, w)
        r = self.radius
        steps = np.arange(-r, r + 1)
        grid = np.stack(np.meshgrid(steps, steps, indexing="ij"), axis=-1).reshape(-1, 2)  # (dy, dx), unfold's order
        # the border removal keeps every window inside the map (the unfold's zero padding is never read)
        patch = score[ys[:, None] + grid[:, 0], xs[:, None] + grid[:, 1]]
        x_exp = np.exp((patch - patch.max(axis=1, keepdims=True)) / _F(_TEMPERATURE))
        residual = x_exp @ grid[:, ::-1].astype(_F) / x_exp.sum(axis=1, keepdims=True)
        xy = np.stack([xs, ys], axis=-1).astype(_F) + residual
        return (xy / np.array([w - 1, h - 1], _F) * _F(2) - _F(1)).astype(_F)

    @staticmethod
    def pixels(kpts: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Normalised keypoints as pixels of the `(h, w)` image (centres at integers): `ALIKED.forward`'s output."""
        h, w = size
        return (np.array([w - 1, h - 1], _F) * (kpts + _F(1)) / _F(2)).astype(_F)

    # ------------------------------------------------------------------ descriptors

    def _pixels(self, maps: _Maps, rows: mx.array, cols: mx.array) -> mx.array:
        """The normalized descriptor map at the integer pixels `(rows, cols)` of the image: `[P, C]`."""
        fine = maps[0]
        hp, wp, c = fine.shape
        h, w = maps[4].shape
        top, left = self._pad(h, w)
        rows, cols = rows + top, cols + left
        parts = [fine.reshape(-1, c)[rows * wp + cols]]
        for level in maps[1:4]:  # the upsampling's (align_corners) value at the pixel
            hl, wl, cl = level.shape
            sy, sx = rows.astype(mx.float32) * ((hl - 1) / (hp - 1)), cols.astype(mx.float32) * ((wl - 1) / (wp - 1))
            y0, x0 = mx.minimum(sy.astype(mx.int32), hl - 1), mx.minimum(sx.astype(mx.int32), wl - 1)
            y1, x1 = mx.minimum(y0 + 1, hl - 1), mx.minimum(x0 + 1, wl - 1)
            wy, wx = (sy - y0)[:, None], (sx - x0)[:, None]
            flat = level.reshape(-1, cl)
            upper = flat[y0 * wl + x0] * (1 - wx) + flat[y0 * wl + x1] * wx
            lower = flat[y1 * wl + x0] * (1 - wx) + flat[y1 * wl + x1] * wx
            parts.append(upper * (1 - wy) + lower * wy)
        x = mx.concatenate(parts, axis=-1)
        return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), _EPS)

    def describe(self, maps: _Maps, kpts: np.ndarray) -> mx.array:
        """`SDDH.forward`: unit descriptors `[N, D]` of the normalised keypoints `kpts` (lazy)."""
        h, w = maps[4].shape
        n, ps, n_pos = len(kpts), self.patch, self.n_pos
        wh = mx.array(np.array([w - 1, h - 1], _F))
        limit = max(h, w) / 4.0
        kw = (mx.array(kpts) / 2 + 0.5) * wh
        # `get_patches`: the ps x ps pixels from the corner, clamped into the map
        corner = mx.maximum(kw.astype(mx.int32) - (ps // 2), 0)
        cx, cy = mx.clip(corner[:, 0], 0, w - 1 - ps), mx.clip(corner[:, 1], 0, h - 1 - ps)
        steps = mx.arange(ps)
        py = mx.broadcast_to(cy[:, None, None] + steps[None, :, None], (n, ps, ps)).reshape(-1)
        px = mx.broadcast_to(cx[:, None, None] + steps[None, None, :], (n, ps, ps)).reshape(-1)
        patch = self._pixels(maps, py, px).reshape(n, -1)  # [N, (a, b, channel)]
        off = _selu(patch @ self.offset0[0] + self.offset0[1]) @ self.offset1[0] + self.offset1[1]
        off = mx.clip(off, -limit, limit).reshape(n, 2, n_pos).transpose(0, 2, 1)  # [N, n_pos, (x, y)]
        pos = 2.0 * (kw[:, None] + off) / wh - 1
        # `grid_sample` (align_corners, zero padding) of the normalized map: the four corners' pixels, weighted
        ixy = (pos.reshape(-1, 2) + 1) / 2 * wh
        low = mx.floor(ixy)
        x0, y0 = low[:, 0], low[:, 1]
        x1, y1 = x0 + 1, y0 + 1
        ix, iy = ixy[:, 0], ixy[:, 1]
        cxs, cys = mx.stack((x0, x1, x0, x1)), mx.stack((y0, y0, y1, y1))  # [4, P]: nw, ne, sw, se
        weights = mx.stack((x1 - ix, ix - x0, x1 - ix, ix - x0)) * mx.stack((y1 - iy, y1 - iy, iy - y0, iy - y0))
        inside = (cxs >= 0) & (cxs <= w - 1) & (cys >= 0) & (cys <= h - 1)
        weights = mx.where(inside, weights, 0.0).reshape(-1, 1)
        rows = mx.clip(cys, 0, h - 1).astype(mx.int32).reshape(-1)
        cols = mx.clip(cxs, 0, w - 1).astype(mx.int32).reshape(-1)
        values = (self._pixels(maps, rows, cols) * weights).reshape(4, n * n_pos, -1).sum(axis=0)
        feats = _selu(values @ self.sf).reshape(n, -1)  # [N, (position, channel)]
        desc = feats @ self.agg
        return desc / mx.maximum(mx.linalg.norm(desc, axis=-1, keepdims=True), _EPS)

    def __call__(self, maps: _Maps) -> tuple[np.ndarray, mx.array | None]:
        """Keypoints `[N, 2]` (float32 pixels of the image `maps` was computed of) and their descriptors `[N, D]`.

        The descriptors are lazy MLX arrays; None for a network without `describe` (FlexGlue reads keypoints only).
        """
        kpts = self.keypoints(maps)
        desc = self.describe(maps, kpts) if self.describes and len(kpts) else None
        if desc is None and self.describes:
            desc = mx.zeros((0, self.agg.shape[-1]))
        return self.pixels(kpts, maps[4].shape), desc
