"""RoMa v2 inference in MLX, with NHWC feature maps and float32 warp refinement.

The two-view transformer rounds its rotary products to bfloat16 just as its upstream position encoder does,
including in float32 inference. Only overlap (the first confidence channel) feeds the public result. The
three precision-matrix channels do not enter any subsequent warp or overlap computation.

Kernels mode, and float16, use FP16 for transformer projections, VGG convolutions and the refiner's inner blocks.
Float16 also runs the DPT head's convolutions in FP16 and keeps the VGG feature maps in FP16. The transformer
residual stream, norms, attention, the global matching, local correlation, warp and confidence updates stay in FP32.
"""
# ruff: noqa: ANN001, ANN202 - array functions mirror the upstream stages; module arguments are their torch layers

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from fastforks._core.mlx import mx, to_mx, to_torch
from fastforks.shared.depth_anything_v2.mlx_net import resize as resize_aligned
from fastforks.shared.mlx_vit import gelu, rope_tables, rotate

from .net.geometry import get_normalized_grid
from .net.normalizers import imagenet
from .net.romav2 import resize

if TYPE_CHECKING:
    from .model import Params
    from .net.romav2 import RoMaV2


def _array(t):
    return to_mx(t.detach().float())


def _linear(layer, *, half=False):
    w, b = _array(layer.weight.flatten(1).T), _array(layer.bias)
    if hasattr(layer, "bias_mask"):
        b = b * _array(layer.bias_mask)
    if half:
        w, b = w.astype(mx.float16), b.astype(mx.float16)
        return lambda x: mx.addmm(b, x.astype(mx.float16), w).astype(mx.float32)
    return lambda x: mx.addmm(b, x, w)


def _conv(layer, *, half=False):
    w = _array(layer.weight.permute(0, 2, 3, 1))
    b = _array(layer.bias) if layer.bias is not None else None
    stride, pad, groups = layer.stride, layer.padding, layer.groups
    if half:
        w = w.astype(mx.float16)
        b = b.astype(mx.float16) if b is not None else None

    def run(x):
        y = mx.conv2d(x.astype(w.dtype), w, stride=stride, padding=pad, groups=groups)
        return y + b if b is not None else y

    return run


def _norm(layer):
    w, b, eps = _array(layer.weight), _array(layer.bias), layer.eps
    return lambda x: mx.fast.layer_norm(x, w, b, eps)


def _bf(x):
    return x.astype(mx.bfloat16).astype(mx.float32)


def _rotate_bf(x, cos, sin):
    x = _bf(x)
    a, b = mx.split(x, 2, axis=-1)
    return _bf(_bf(x * cos) + _bf(mx.concatenate([-b, a], axis=-1) * sin))


def _block(layer, *, bf_rope: bool, half: bool):
    qkv, proj = _linear(layer.attn.qkv, half=half), _linear(layer.attn.proj, half=half)
    fc1, fc2 = _linear(layer.mlp.fc1, half=half), _linear(layer.mlp.fc2, half=half)
    n1, n2 = _norm(layer.norm1), _norm(layer.norm2)
    g1 = _array(layer.ls1.gamma) if hasattr(layer.ls1, "gamma") else 1
    g2 = _array(layer.ls2.gamma) if hasattr(layer.ls2, "gamma") else 1
    heads = layer.attn.num_heads
    rotary = _rotate_bf if bf_rope else rotate

    def run(x, rope):
        b, n, c = x.shape
        q, k, v = qkv(n1(x)).reshape(b, n, 3, heads, c // heads).transpose(2, 0, 3, 1, 4)
        if rope is not None:
            q, k = rotary(q, *rope), rotary(k, *rope)
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=(c // heads) ** -0.5)
        x = x + g1 * proj(h.transpose(0, 2, 1, 3).reshape(b, n, c))
        return x + g2 * fc2(gelu(fc1(n2(x))))

    return run


def _rope(vit, side, prefix):
    sin, cos = vit.rope_embed(H=side, W=side)
    return rope_tables(_array(cos), _array(sin), prefix)


def _descriptor(vit, side, *, half: bool):
    embed = _conv(vit.patch_embed.proj)
    prefix = mx.concatenate([_array(vit.cls_token), _array(vit.storage_tokens)], axis=1)
    rope = _rope(vit, side, prefix.shape[1])
    blocks = [_block(layer, bf_rope=False, half=half) for layer in vit.blocks]
    norm, taps = _norm(vit.norm), vit.layers

    def run(image):
        b = image.shape[0]
        x = mx.concatenate(
            [mx.broadcast_to(prefix, (b, *prefix.shape[1:])), embed(image).reshape(b, side * side, -1)], axis=1
        )
        out = []
        for i, block in enumerate(blocks):
            x = block(x, rope)
            if i in taps:
                out.append(norm(x[:, prefix.shape[1] :]).reshape(b, side, side, -1))
        return out

    return run


def _matcher(matcher, side, *, half: bool):
    vit = matcher.mv_vit
    proj, head, norm = _linear(vit.projector), _linear(vit.output_projector), _norm(vit.norm)
    rope = _rope(vit, side, 0)
    blocks = [_block(layer, bf_rope=True, half=half) for layer in vit.blocks]
    grid = get_normalized_grid(1, side, side)
    phases = torch.nn.functional.linear(grid, matcher.scale * matcher.omega)
    pos = _array(torch.cat([phases.sin(), phases.cos()], -1).flatten(1, 2))
    temp = float(matcher.temp)

    def run(a, b):
        x = mx.stack([mx.concatenate(a, -1), mx.concatenate(b, -1)], axis=1)
        batch, _, h, w, _ = x.shape
        x = proj(x.reshape(batch, 2 * h * w, -1))
        for i, block in enumerate(blocks):
            if i % 2:  # noqa: SIM108 - the two reshapes delimit single-view attention
                x = block(x.reshape(2 * batch, h * w, -1), rope).reshape(batch, 2 * h * w, -1)
            else:
                x = block(x, None)
        first, second = mx.split(head(norm(x)), 2, axis=1)
        q = first / mx.linalg.norm(first, axis=-1, keepdims=True)
        k = second / mx.linalg.norm(second, axis=-1, keepdims=True)
        # Chunk queries to bound the materialised cosine-score matrix at large coarse sizes.
        parts = []
        for start in range(0, h * w, 256):
            scores = (q[:, start : start + 256] @ k.transpose(0, 2, 1)) / temp
            parts.append(mx.softmax(scores, axis=-1) @ pos)
        match = mx.concatenate(parts, axis=1)
        return a[0], a[1] + first.reshape(batch, h, w, -1) + match.reshape(batch, h, w, -1)

    return run


def _resize_layer(layer, *, half: bool):
    if isinstance(layer, nn.ConvTranspose2d):
        # Nonoverlapping transposed convolution: project each pixel into a tile of output pixels.
        w = _array(layer.weight.permute(0, 2, 3, 1).flatten(1))
        bias = _array(layer.bias)
        stride = layer.stride[0]

        def run(x):
            b, h, width, _ = x.shape
            tiles = (x @ w).reshape(b, h, width, stride, stride, -1)
            return tiles.transpose(0, 1, 3, 2, 4, 5).reshape(b, h * stride, width * stride, -1) + bias

        return run
    if isinstance(layer, nn.Conv2d):
        return _conv(layer, half=half)
    if isinstance(layer, nn.Identity):
        return lambda x: x
    msg = f"unsupported RoMa DPT resize layer {type(layer).__name__}"
    raise TypeError(msg)


def _unit(layer, *, half: bool):
    first, last = _conv(layer.conv1, half=half), _conv(layer.conv2, half=half)

    def run(x):
        # Upstream's first ReLU is in-place: the skip carries the rectified input.
        residual = mx.maximum(x, 0)
        return residual + last(mx.maximum(first(residual), 0))

    return run


def _fuse(layer, up, *, half: bool):
    skip = _unit(layer.resConfUnit1, half=half) if layer.has_residual else None
    block, proj = _unit(layer.resConfUnit2, half=half), _conv(layer.out_conv, half=half)

    def run(x, other, size):
        if skip is not None:
            x = x + skip(other)
        return proj(up(block(x), *size))

    return run


def _dpt(head, up, *, half: bool):
    """`half`: the convolutions in FP16 (the output projection, which makes the warp and overlap, stays FP32)."""
    norm = _norm(head.norm)
    projects = [_linear(x) for x in head.projects]
    resizes = [_resize_layer(x, half=half) for x in head.resize_layers]
    s = head.scratch
    convs = [_conv(getattr(s, f"layer{i}_rn"), half=half) for i in range(1, 5)]
    fuses = [_fuse(getattr(s, f"refinenet{i}"), up, half=half) for i in range(1, 5)]
    reduce, hidden = _conv(s.output_conv1, half=half), _conv(s.output_conv2[0], half=half)
    out = _linear(s.output_conv2[2])
    ratio = head.patch_size // head.down_ratio

    def run(feats):
        h, w = feats[0].shape[1:3]
        a, b = map(norm, feats)
        maps = [
            conv(resize_map(proj(x)))
            for conv, resize_map, proj, x in zip(convs, resizes, projects, (a, a, b, b), strict=True)
        ]
        y = fuses[3](maps[3], None, maps[2].shape[1:3])
        for i in (2, 1, 0):
            size = maps[i - 1].shape[1:3] if i else (maps[0].shape[1] * 2, maps[0].shape[2] * 2)
            y = fuses[i](y, maps[i], size)
        y = up(reduce(y), h * ratio, w * ratio)
        pred = out(mx.maximum(hidden(y), 0).astype(mx.float32))
        return pred[..., :2], pred[..., 2:3]

    return run


def _resize_half(x, h, w):
    for axis, n in ((1, h), (2, w)):
        old = x.shape[axis]
        if old == n:
            continue
        v = mx.maximum((mx.arange(n, dtype=mx.float32) + 0.5) * (old / n) - 0.5, 0)
        i = v.astype(mx.int32)
        shape = [1, 1, 1, 1]
        shape[axis] = n
        a = mx.take(x, i, axis=axis)
        x = a + (v - i).reshape(shape) * (mx.take(x, mx.minimum(i + 1, old - 1), axis=axis) - a)
    return x


def _sample(x, warp):
    b, h, w, _ = x.shape
    xx = ((warp[..., 0] + 1) * w - 1) * 0.5
    yy = ((warp[..., 1] + 1) * h - 1) * 0.5
    x0, y0 = mx.floor(xx).astype(mx.int32), mx.floor(yy).astype(mx.int32)
    batch = mx.arange(b)[:, None, None]

    def tap(ix, iy):
        mask = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        return x[batch, mx.clip(iy, 0, h - 1), mx.clip(ix, 0, w - 1)] * mask[..., None]

    dx, dy = (xx - x0)[..., None], (yy - y0)[..., None]
    return ((1 - dx) * tap(x0, y0) + dx * tap(x0 + 1, y0)) * (1 - dy) + (
        (1 - dx) * tap(x0, y0 + 1) + dx * tap(x0 + 1, y0 + 1)
    ) * dy


def _correlation(a, b, warp, radius):
    _, h, w, c = a.shape
    dots = []
    a = a * c**-0.5
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sampled = _sample(b, warp + mx.array([2 * dx / w, 2 * dy / h]))
            dot = mx.sum(a * sampled, axis=-1)
            mx.eval(dot)  # bound peak memory to one sampled feature map
            dots.append(dot)
    return mx.stack(dots, axis=-1)


def _refine(layer, correlate, sample, *, half: bool):
    proj = _linear(layer.proj)
    disp = _linear(layer.disp_emb)
    blocks = [
        (_conv(m.conv_depthwise, half=half), _linear(m.conv_pointwise, half=half))
        for m in [layer.block1, *layer.hidden_blocks]
    ]
    delta, score = _linear(layer.warp_head), _linear(layer.confidence_head)
    radius, init = layer.cfg.local_corr_radius, layer.cfg.refine_init

    def run(a, b, warp, conf, scale):
        _, h, w, _ = a.shape
        warp, conf = _resize_half(warp, h, w), _resize_half(conf, h, w)
        a, b = proj(a), proj(b)
        grid = _array(get_normalized_grid(1, h, w))
        parts = [a, sample(b, warp), disp((warp - grid) * scale)]
        if radius is not None:
            parts.append(correlate(a, b, warp, radius))
        x = mx.concatenate(parts, axis=-1)
        for depthwise, pointwise in blocks:
            x = pointwise(mx.maximum(depthwise(x), 0))
            mx.eval(x)
        return warp + delta(x) / mx.array([w * init, h * init]), conf + score(x)[..., :1]

    return run


def _vgg(net, *, half: bool, keep_half: bool):
    """`keep_half`: the feature maps stay FP16 (the refiners' input projections read them in FP32)."""
    stages = []
    stage = []
    for layer in net.layers:
        if isinstance(layer, nn.Conv2d):
            stage.append(_conv(layer, half=half))
        if isinstance(layer, nn.MaxPool2d):
            stages.append(stage)
            stage = []

    def run(x):
        maps = {}
        for i, convs in enumerate(stages):
            if i:
                b, h, w, c = x.shape
                x = mx.max(x.reshape(b, h // 2, 2, w // 2, 2, c), axis=(2, 4))
            for conv in convs:
                x = mx.maximum(conv(x), 0)
                mx.eval(x)
            maps[2**i] = x if keep_half else x.astype(mx.float32)
        return maps

    return run


class Engine:
    """MLX descriptor, global matcher and coarse/fine refiners for one verified checkpoint."""

    def __init__(self, net: RoMaV2, params: Params, *, kernels: bool, float16: bool = False) -> None:
        """Convert the network and build RoPE tables at the selected coarse size (`float16`: see the module)."""
        self.params = params
        half = kernels or float16
        self.describe = _descriptor(net.f, params.coarse_size // net.f.patch_size, half=half)
        self.match = _matcher(net.matcher, params.coarse_size // net.f.patch_size, half=half)
        up, correlate, sample = resize_aligned, _correlation, _sample
        if kernels:
            from fastforks.kernels.dpt_mlx.ops import resize as up  # noqa: PLC0415 - runtime Metal kernels
            from fastforks.kernels.roma_mlx.ops import correlation as correlate  # noqa: PLC0415
            from fastforks.kernels.roma_mlx.ops import sample  # noqa: PLC0415
        self.dpt = _dpt(net.matcher.head, up, half=float16)
        self.features = _vgg(net.refiner_features, half=half, keep_half=float16)
        self.refiners = [
            (int(k), _refine(v, correlate, sample, half=half)) for k, v in cast("nn.ModuleDict", net.refiners).items()
        ]
        self.anchor = net.anchor_width
        self.kept: list[tuple[torch.Tensor, list[mx.array]]] = []

    def _describe(self, image: torch.Tensor, prepared: mx.array) -> list[mx.array]:
        count = self.params.keep_descriptors
        if count:
            for i, (key, features) in enumerate(self.kept):
                if key.shape == image.shape and torch.equal(key, image):
                    self.kept.append(self.kept.pop(i))
                    return features
        features = self.describe(prepared)
        if count:
            mx.eval(features)
            self.kept.append((image.clone(), features))
            if len(self.kept) > count:
                self.kept.pop(0)
        return features

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        """Dense pixel warp into image B and overlap probability on image A's grid."""

        def prepare(x, side):
            return _array(imagenet(resize(x, (side, side))).permute(0, 2, 3, 1))

        coarse = self.params.coarse_size
        x, y = prepare(a, coarse), prepare(b, coarse)
        feats = self.match(self._describe(a, x), self._describe(b, y))
        warp, confidence = self.dpt(feats)
        mx.eval(warp, confidence)
        del feats
        for side in (coarse, self.params.fine_size):
            if not side:
                continue
            if side != coarse:
                x, y = prepare(a, side), prepare(b, side)
            fa, fb = self.features(x), self.features(y)
            for stride, refine in self.refiners:
                warp, confidence = refine(fa.pop(stride), fb.pop(stride), warp, confidence, side / self.anchor)
                mx.eval(warp, confidence)
        warp = _resize_half(warp, *a.shape[-2:])
        confidence = _resize_half(mx.sigmoid(confidence), *a.shape[-2:])
        pixels = (warp + 1) * mx.array([b.shape[-1] / 2, b.shape[-2] / 2]) - 0.5
        return {"warp": to_torch(pixels.transpose(0, 3, 1, 2)), "certainty": to_torch(confidence[..., 0])}
