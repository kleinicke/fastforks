"""MoGe's trunk on Apple MLX (`device="mlx"`): DINOv2 encoder, neck, heads and the metric-scale MLP, hand-written.

The same network as `model._MlxPortable` ran through an exported graph before, as `mx` operations in channels-last
layout: nothing is traced, so a first call is the Metal kernels' build and no `torch.export` (which cost ~1.5 s of a
moge-2-vits first call: ~0.6 s of imports in a fresh process and ~0.9 s of tracing). The pieces:

- the encoder: the shared DINOv2 blocks (`fastforks.shared.mlx_vit`), the tokens of the read-out blocks projected and
  summed by one matmul each;
- the view-plane uv inputs of the neck: a 1x1 convolution of a map that is linear in the column and in the row, so
  one outer sum of the two axes (no uv map, no convolution);
- transposed convolutions (kernel = stride = 2) as one matmul and a depth-to-space;
- the heads' full-resolution level folded to half resolution (`model._FoldedTop`'s weights), in float32; the neck's
  part of it is one convolution for all heads.

Weights and activations are in the run dtype; the image resize and normalisation, the folded top level and the scale
MLP are float32 (the torch fp16 runner keeps its output convolutions and the scale MLP in float32 too).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.mlx import mx, to_mx
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD
from fastforks.shared.mlx_vit import PATCH, dinov2_layers, dinov2_params, dinov2_table

if TYPE_CHECKING:
    from .net import ConvStack, MoGeNet

__all__ = ["Trunk"]

SIZES = 4  # input sizes whose position table and resize matrices are kept (least recently used dropped)
_LEVELS = 4  # levels below the folded full-resolution one


def _f32(v: torch.Tensor | np.ndarray) -> mx.array:
    a = v.detach().double().numpy() if isinstance(v, torch.Tensor) else v
    return mx.array(np.ascontiguousarray(a, dtype=np.float32))


def _conv_params(out: dict[str, mx.array], k: str, m: nn.Module) -> None:
    """A 3x3 convolution as `[O, 3, 3, I]`."""
    conv = cast("nn.Conv2d", m)
    out[k + ".weight"] = _f32(conv.weight.permute(0, 2, 3, 1))
    out[k + ".bias"] = _f32(cast("torch.Tensor", conv.bias))


def _stack_params(out: dict[str, mx.array], k: str, cs: ConvStack) -> None:
    """Residual blocks and resamplers of the levels below the top one (the input convolutions differ per stack)."""
    for i in range(_LEVELS):
        for j, blk in enumerate(cast("nn.Sequential", cs.res_blocks[i])):
            layers = cast("nn.Sequential", blk.layers)  # (norm, relu, conv) twice
            _conv_params(out, f"{k}.res{i}.{j}.conv1", layers[2])
            _conv_params(out, f"{k}.res{i}.{j}.conv2", layers[5])
        if i < _LEVELS - 1:
            up = cast("nn.Sequential", cs.resamplers[i])
            t = cast("nn.ConvTranspose2d", up[0])  # [I, O, 2, 2] -> [I, (a, b, O)]
            out[f"{k}.up{i}.t.weight"] = _f32(t.weight.permute(0, 2, 3, 1).reshape(t.weight.shape[0], -1))
            out[f"{k}.up{i}.t.bias"] = _f32(cast("torch.Tensor", t.bias).repeat(4))
            _conv_params(out, f"{k}.up{i}.conv", up[1])


def _mat(m: nn.Module) -> torch.Tensor:
    """A 1x1 convolution's weight as `[O, I]`."""
    return cast("nn.Conv2d", m).weight.detach()[:, :, 0, 0]


def prepare(net: MoGeNet, folded: dict[str, nn.Module]) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """The float32 arrays of `forward`, `(run-dtype part, float32 part)`, from the network and its folded top levels."""
    p = dinov2_params(net.encoder.backbone.state_dict(), PATCH)
    f: dict[str, mx.array] = {}
    proj = [cast("nn.Conv2d", m) for m in net.encoder.output_projections]
    for i, m in enumerate(proj):
        p[f"proj{i}.weight"] = _f32(_mat(m).T)
    p["proj.bias"] = _f32(torch.stack([cast("torch.Tensor", m.bias).detach().double() for m in proj]).sum(0))
    width = proj[0].weight.shape[0]
    for i in range(_LEVELS):  # the neck's inputs: the features (level 0) and the view-plane uv
        m = cast("nn.Conv2d", net.neck.input_blocks[i])
        w = _mat(m)
        if i == 0:
            p["neck.feat.weight"] = _f32(w[:, :width].T)
        p[f"neck.uv{i}.weight"] = _f32(w[:, -2:].T)  # [2, C]: the rows for u and v
        p[f"neck.uv{i}.bias"] = _f32(cast("torch.Tensor", m.bias))
    _stack_params(p, "neck", net.neck)
    for k, top in folded.items():
        cs = cast("ConvStack", getattr(net, f"{k}_head"))
        for i in range(_LEVELS):
            m = cast("nn.Conv2d", cs.input_blocks[i])
            p[f"{k}.in{i}.weight"] = _f32(_mat(m).T)
            p[f"{k}.in{i}.bias"] = _f32(cast("torch.Tensor", m.bias))
        _stack_params(p, k, cs)
        f[f"{k}.top.weight"] = _f32(top.get_buffer("head").permute(0, 2, 3, 1))
        f[f"{k}.plane"] = _f32(top.get_buffer("plane").T)  # [2, O]
        f[f"{k}.bias"] = _f32(top.get_buffer("bias"))
    # the neck's part of every head's top level: one convolution, its outputs split per head
    f["top.neck.weight"] = _f32(torch.cat([top.get_buffer("neck") for top in folded.values()]).permute(0, 2, 3, 1))
    linear = [m for m in net.scale_head if isinstance(m, nn.Linear)]
    for i, m in enumerate(linear):
        f[f"scale{i}.weight"] = _f32(m.weight.T)
        f[f"scale{i}.bias"] = _f32(m.bias)
    return p, f


def _pad(x: mx.array) -> mx.array:
    """Replicate padding by one of `[B, H, W, C]`."""
    x = mx.concatenate([x[:, :1], x, x[:, -1:]], axis=1)
    return mx.concatenate([x[:, :, :1], x, x[:, :, -1:]], axis=2)


def _conv(p: dict[str, mx.array], k: str, x: mx.array) -> mx.array:
    """3x3 replicate-padded convolution."""
    return mx.conv2d(_pad(x), p[k + ".weight"]) + p[k + ".bias"]


def _res(p: dict[str, mx.array], k: str, x: mx.array) -> mx.array:
    """The residual blocks `k.<j>` of one level: `x + conv2(relu(conv1(relu(x))))` each."""
    j = 0
    while f"{k}.{j}.conv1.weight" in p:
        y = _conv(p, f"{k}.{j}.conv1", mx.maximum(x, 0))
        x = x + _conv(p, f"{k}.{j}.conv2", mx.maximum(y, 0))
        j += 1
    return x


def _up(p: dict[str, mx.array], k: str, x: mx.array) -> mx.array:
    """Resampler: transposed convolution (kernel = stride = 2: a matmul to the four phases), 3x3 convolution."""
    b, h, w, _ = x.shape
    y = mx.addmm(p[k + ".t.bias"], x, p[k + ".t.weight"])
    y = y.reshape(b, h, w, 2, 2, -1).transpose(0, 1, 3, 2, 4, 5).reshape(b, 2 * h, 2 * w, -1)
    return _conv(p, k + ".conv", y)


def _plane(w: mx.array, bias: mx.array, u: mx.array, v: mx.array) -> mx.array:
    """The 1x1 convolution `w` (`[2, C]`) of the view-plane uv map with axes `u` `[W]`, `v` `[H]`: `[H, W, C]`."""
    return u[None, :, None] * w[0] + (v[:, None, None] * w[1] + bias)


def forward(  # noqa: PLR0913, PLR0917 - the network's pieces
    p: dict[str, mx.array],
    f: dict[str, mx.array],
    heads: int,
    layers: tuple[int, ...],
    names: tuple[str, ...],
    x: mx.array,
    table: mx.array,
    axes: list[tuple[mx.array, mx.array]],
) -> tuple[list[mx.array], mx.array, mx.array]:
    """Normalised `[B, 3, 14 rows, 14 cols]` -> the heads' maps, the scale and the encoder features.

    Args:
        p: Run-dtype parameters of `prepare`.
        f: Its float32 parameters.
        heads: Attention heads of the ViT.
        layers: The ViT blocks read out (1-based).
        names: The heads to run (`"points"`, `"mask"`, `"normal"`).
        x: The resized, normalised image.
        table: `dinov2_table` of the token grid, in the run dtype.
        axes: The view-plane axes `(u, v)` of the five levels: four in the run dtype, the full-resolution one float32.

    Returns:
        Float32 `[B, 16 rows, 16 cols, C]` per head, the metric scale `[B]` and the features `[B, rows, cols, width]`
        (run dtype) the MoGe-3 refiner reads.
    """
    b, rows, cols = x.shape[0], x.shape[2] // PATCH, x.shape[3] // PATCH
    tokens = dinov2_layers(p, heads, x, table, layers)
    feat = tokens[0][:, 1:] @ p["proj0.weight"]
    for i, t in enumerate(tokens[1:], 1):
        feat = feat + t[:, 1:] @ p[f"proj{i}.weight"]
    feat = (feat + p["proj.bias"]).reshape(b, rows, cols, -1)
    cls = tokens[-1][:, 0].astype(mx.float32)
    mx.async_eval(feat, cls)

    levels = []
    y = feat @ p["neck.feat.weight"]
    for i in range(_LEVELS):
        y = _res(p, f"neck.res{i}", y + _plane(p[f"neck.uv{i}.weight"], p[f"neck.uv{i}.bias"], *axes[i]))
        levels.append(y)
        if i < _LEVELS - 1:
            y = _up(p, f"neck.up{i}", y)
    low = mx.conv2d(_pad(levels[-1].astype(mx.float32)), f["top.neck.weight"])
    mx.async_eval(*levels, low)

    outs = []
    at = 0
    for k in names:
        y = _res(p, f"{k}.res0", mx.addmm(p[f"{k}.in0.bias"], levels[0], p[f"{k}.in0.weight"]))
        for i in range(1, _LEVELS):
            y = _up(p, f"{k}.up{i - 1}", y) + mx.addmm(p[f"{k}.in{i}.bias"], levels[i], p[f"{k}.in{i}.weight"])
            y = _res(p, f"{k}.res{i}", y)
        w = f[f"{k}.top.weight"]
        y = mx.conv2d(_pad(y.astype(mx.float32)), w) + low[..., at : at + w.shape[0]]
        at += w.shape[0]
        _, h, wd, c = y.shape
        y = y.reshape(b, h, wd, c // 4, 2, 2).transpose(0, 1, 4, 2, 5, 3).reshape(b, 2 * h, 2 * wd, c // 4)
        y = y + _plane(f[f"{k}.plane"], f[f"{k}.bias"], *axes[_LEVELS])
        mx.async_eval(y)
        outs.append(y)

    s = cls
    n = sum(k.startswith("scale") for k in f) // 2
    for i in range(n):
        s = mx.addmm(f[f"scale{i}.bias"], s, f[f"scale{i}.weight"])
        s = mx.maximum(s, 0) if i < n - 1 else s
    return outs, mx.exp(s[:, 0]), feat


class Trunk:
    """The MLX trunk of one loaded model: converted weights and the per-size tables."""

    def __init__(self, net: MoGeNet, folded: dict[str, nn.Module], dtype: str) -> None:
        """Convert the float32 `net` (LayerScale unfolded or folded) and its `folded` top levels, by head name."""
        self.dtype = getattr(mx, dtype)
        self.heads = int(net.encoder.backbone.blocks[0].attn.num_heads)
        self.layers = tuple(i + 1 for i in net.encoder.layers)
        self.names = tuple(folded)
        p32, self.f = prepare(net, folded)
        host = {k: torch.from_numpy(np.array(p32.pop(k))) for k in ("pos.cls", "pos.grid", "patch.bias")}
        self.pos, self.cls, self.bias = host["pos.grid"], host["pos.cls"], host["patch.bias"]  # fp32, as upstream
        self.p = {k: v.astype(self.dtype) for k, v in p32.items()}
        del p32
        mx.eval(self.p, self.f)
        self.mean = mx.array(np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1))
        self.std = mx.array(np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1))
        self.sizes: OrderedDict[tuple[int, int, int, int], tuple[mx.array, mx.array, mx.array]] = OrderedDict()

    def _tables(self, h: int, w: int, rows: int, cols: int) -> tuple[mx.array, mx.array, mx.array]:
        """Of one input size: the antialiased resize's row and column matrices (float32) and the position table."""
        key = (h, w, rows, cols)
        if key not in self.sizes:

            def matrix(n: int, m: int) -> mx.array:  # `[m, n]`: PyTorch's own weights, read off an identity
                eye = torch.eye(n)[None, None]
                return to_mx(F.interpolate(eye, (m, n), mode="bilinear", align_corners=False, antialias=True)[0, 0])

            table = to_mx(dinov2_table(self.cls, self.pos, self.bias, rows, cols)).astype(self.dtype)
            self.sizes[key] = (matrix(h, rows * PATCH), matrix(w, cols * PATCH).T, table)
            mx.eval(self.sizes[key])
            while len(self.sizes) > SIZES:
                self.sizes.popitem(last=False)
        self.sizes.move_to_end(key)
        return self.sizes[key]

    def __call__(
        self, image: torch.Tensor, rows: int, cols: int, axes: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[list[mx.array], mx.array, mx.array]:
        """`[B, 3, H, W]` in `[0, 1]` (CPU) -> `forward`'s lazy outputs; `axes`: the view-plane axes of the levels."""
        wy, wx, table = self._tables(image.shape[2], image.shape[3], rows, cols)
        x = (wy @ to_mx(image.float()) @ wx - self.mean) / self.std
        ax = [
            (to_mx(u).astype(dt), to_mx(v).astype(dt))
            for (u, v), dt in zip(axes, [self.dtype] * _LEVELS + [mx.float32], strict=True)
        ]
        return forward(self.p, self.f, self.heads, self.layers, self.names, x, table, ax)
