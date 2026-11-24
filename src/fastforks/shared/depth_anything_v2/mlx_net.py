"""Depth Anything V2 on Apple MLX: the DINOv2 encoder (`fastforks.shared.mlx_vit`) and the DPT head, channels-last.

The maths of `net.dpt.DepthAnythingV2` for a resized, normalised `[B, 3, h, w]` image (both sides multiples of 14):
the network output `[B, h, w]` after the head's ReLU (or sigmoid, for a metric head).

Prepared once, at conversion (exact in real arithmetic, done in float64, stored float32):
- the encoder as `mlx_vit.dinov2_params` (LayerScale folded, the patch embedding as one matmul).
- the head's 1x1 projections as matmuls on the tokens; for the two finest levels the projection and the transposed
  convolution after it (stride = kernel: no overlap) are one matmul to `[k * k * C]` and a pixel shuffle.
- convolution weights as MLX's `[O, kH, kW, I]`.

float16: weights and activations fp16 (LayerNorm statistics and the attention's softmax in fp32 inside MLX's
kernels); the 1-channel output convolution and what follows run in fp32, as in the torch fp16 runner.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.mlx import mx, to_mx, tuning
from fastforks.shared.mlx_vit import dinov2_layers, dinov2_params, dinov2_table

if TYPE_CHECKING:
    from collections.abc import Callable

    Resize = Callable[[mx.array, int, int], mx.array]

LAYOUT = "v1"  # version of `prepare`'s parameter layout (for the conversion cache's file name)
PATCH = 14
HEADS = {"vits": 6, "vitb": 12, "vitl": 16, "vitg": 24}
LAYERS = {"vits": (3, 6, 9, 12), "vitb": (3, 6, 9, 12), "vitl": (5, 12, 18, 24), "vitg": (10, 20, 30, 40)}
TABLES = 8  # position tables kept (one per grid size, least recently used dropped)
SHUFFLE = (4, 2)  # stride (= kernel) of the transposed convolutions of the two finest levels


def prepare(state: dict[str, torch.Tensor]) -> dict[str, mx.array]:
    """Checkpoint tensors (upstream `DepthAnythingV2` names) -> the float32 arrays `forward` uses."""
    vit = {k.removeprefix("pretrained."): state.pop(k) for k in list(state) if k.startswith("pretrained.")}
    out = dinov2_params(vit, PATCH)

    def take(k: str) -> np.ndarray:
        return state.pop("depth_head." + k).double().numpy()

    def put(k: str, v: np.ndarray) -> None:
        out["head." + k] = mx.array(np.ascontiguousarray(v, dtype=np.float32))

    def conv(dst: str, src: str, *, bias: bool = True) -> None:
        put(dst + ".weight", take(src + ".weight").transpose(0, 2, 3, 1))
        if bias:
            put(dst + ".bias", take(src + ".bias"))

    for i in range(4):
        w, b = take(f"projects.{i}.weight")[:, :, 0, 0], take(f"projects.{i}.bias")  # [C, K], [C]
        if i < len(SHUFFLE):  # transposed convolution [C, O, k, k] after the projection: one matmul to (a, b, o)
            wd, bd = take(f"resize_layers.{i}.weight"), take(f"resize_layers.{i}.bias")
            k = SHUFFLE[i]
            put(f"proj{i}.weight", np.einsum("ci,coab->iabo", w, wd).reshape(w.shape[1], -1))
            put(f"proj{i}.bias", (np.einsum("c,coab->abo", b, wd) + bd).reshape(k * k * wd.shape[1]))
        else:
            put(f"proj{i}.weight", w.T)
            put(f"proj{i}.bias", b)
        conv(f"rn{i}", f"scratch.layer{i + 1}_rn", bias=False)
    conv("down", "resize_layers.3")
    for i in range(4):
        src = f"scratch.refinenet{i + 1}."
        for u in (1, 2):
            for c in (1, 2):  # the coarsest block's unit 1 has no input: converted, unused (as upstream)
                conv(f"fuse{i}.unit{u}.conv{c}", f"{src}resConfUnit{u}.conv{c}")
        w = take(src + "out_conv.weight")[:, :, 0, 0]
        put(f"fuse{i}.out.weight", w.T)
        put(f"fuse{i}.out.bias", take(src + "out_conv.bias"))
    conv("out1", "scratch.output_conv1")
    conv("out2", "scratch.output_conv2.0")
    put("out3.weight", take("scratch.output_conv2.2.weight")[:, :, 0, 0].T)
    put("out3.bias", take("scratch.output_conv2.2.bias"))
    return out


def _lerp_axis(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear (`align_corners=True`) taps of one axis: `(i0, i1, weight of i1)`, as torch computes them (fp32)."""
    scale = np.float32((n_in - 1) / (n_out - 1)) if n_out > 1 else np.float32(0)
    src = np.arange(n_out, dtype=np.float32) * scale
    i0 = np.minimum(np.floor(src).astype(np.int32), n_in - 1)
    return i0, np.minimum(i0 + 1, n_in - 1), (src - i0).astype(np.float32)


def resize(x: mx.array, h: int, w: int) -> mx.array:
    """`F.interpolate(mode="bilinear", align_corners=True)` of `[B, H, W, C]` to `[B, h, w, C]`."""
    for axis, n in ((1, h), (2, w)):
        if x.shape[axis] == n:
            continue
        i0, i1, t = _lerp_axis(x.shape[axis], n)
        shape = [1, 1, 1, 1]
        shape[axis] = n
        t = mx.array(t.reshape(shape)).astype(x.dtype)
        a = mx.take(x, mx.array(i0), axis=axis)
        x = a + (mx.take(x, mx.array(i1), axis=axis) - a) * t
    return x


def _conv(p: dict[str, mx.array], k: str, x: mx.array, stride: int = 1) -> mx.array:
    y = mx.conv2d(x, p[k + ".weight"], stride=stride, padding=1)
    return y + p[k + ".bias"] if k + ".bias" in p else y


def _unit(p: dict[str, mx.array], k: str, x: mx.array) -> mx.array:
    """`ResidualConvUnit`: x + conv2(relu(conv1(relu(x))))."""
    y = _conv(p, k + ".conv1", mx.maximum(x, 0))
    return x + _conv(p, k + ".conv2", mx.maximum(y, 0))


def _fuse(  # noqa: PLR0913, PLR0917 - the block's pieces
    p: dict[str, mx.array], k: str, x: mx.array, skip: mx.array | None, size: tuple[int, int], up: Resize
) -> mx.array:
    """`FeatureFusionBlock`: (x + unit1(skip)) -> unit2 -> resize -> 1x1 convolution.

    The 1x1 convolution runs before the resize, on a quarter of the pixels: both are linear and act on different
    axes, so they commute (the interpolation weights sum to 1, which carries the bias).
    """
    if skip is not None:
        x = x + _unit(p, k + ".unit1", skip)
    x = _unit(p, k + ".unit2", x)
    return up(mx.addmm(p[k + ".out.bias"], x, p[k + ".out.weight"]), *size)


def head(  # noqa: PLR0913 - the head's pieces
    p: dict[str, mx.array], feats: list[mx.array], h0: int, w0: int, *, sigmoid: bool, up: Resize = resize
) -> mx.array:
    """Normed patch tokens `[B, h0 * w0, K]` of the four layers -> float32 `[B, 14 h0, 14 w0]`.

    `up`: the bilinear resize (`resize`, or the `dpt_mlx` kernel).
    """
    b = feats[0].shape[0]
    maps = []
    for i, t in enumerate(feats):
        y = mx.addmm(p[f"head.proj{i}.bias"], t, p[f"head.proj{i}.weight"])
        if i < len(SHUFFLE):
            k = SHUFFLE[i]
            y = y.reshape(b, h0, w0, k, k, -1).transpose(0, 1, 3, 2, 4, 5).reshape(b, h0 * k, w0 * k, -1)
        else:
            y = y.reshape(b, h0, w0, -1)
        if i == len(feats) - 1:
            y = _conv(p, "head.down", y, stride=2)
        maps.append(_conv(p, f"head.rn{i}", y))
    x = _fuse(p, "head.fuse3", maps[3], None, (maps[2].shape[1], maps[2].shape[2]), up)
    x = _fuse(p, "head.fuse2", x, maps[2], (maps[1].shape[1], maps[1].shape[2]), up)
    x = _fuse(p, "head.fuse1", x, maps[1], (maps[0].shape[1], maps[0].shape[2]), up)
    x = _fuse(p, "head.fuse0", x, maps[0], (2 * maps[0].shape[1], 2 * maps[0].shape[2]), up)
    x = up(_conv(p, "head.out1", x), h0 * PATCH, w0 * PATCH)
    x = mx.maximum(_conv(p, "head.out2", x), 0).astype(mx.float32)
    x = (x @ p["head.out3.weight"] + p["head.out3.bias"])[..., 0]
    return mx.sigmoid(x) if sigmoid else mx.maximum(x, 0)


def forward(  # noqa: PLR0913 - the network's pieces
    p: dict[str, mx.array], arch: str, x: mx.array, table: mx.array, *, sigmoid: bool, up: Resize = resize
) -> mx.array:
    """Normalised `[B, 3, h, w]` -> float32 `[B, h, w]` (`table`: `Engine.table`, in the weights' dtype)."""
    h0, w0 = x.shape[2] // PATCH, x.shape[3] // PATCH
    feats = [t[:, 1:] for t in dinov2_layers(p, HEADS[arch], x, table, LAYERS[arch])]
    return head(p, feats, h0, w0, sigmoid=sigmoid, up=up)


class Engine:
    """The MLX network of one loaded model: converted weights, position tables and the compiled forward."""

    def __init__(self, p32: dict[str, mx.array], arch: str, dtype: str, *, sigmoid: bool, up: Resize = resize) -> None:
        """Cast the float32 parameters of `prepare` (`p32`, emptied) to the run dtype; `up`: the head's resize."""
        self.dtype = getattr(mx, dtype)
        host = {k: torch.from_numpy(np.array(p32.pop(k))) for k in ("pos.cls", "pos.grid", "patch.bias")}
        self.pos, self.cls, self.bias = host["pos.grid"], host["pos.cls"], host["patch.bias"]  # fp32, as upstream
        fp32 = ("head.out3.weight", "head.out3.bias")
        self.p = {k: v if k in fp32 else v.astype(self.dtype) for k, v in p32.items()}
        p32.clear()
        mx.eval(self.p)

        def run(x: mx.array, table: mx.array) -> mx.array:
            return forward(self.p, arch, x, table, sigmoid=sigmoid, up=up)

        self.run: Callable[[mx.array, mx.array], mx.array] = mx.compile(run) if tuning().compile_steps else run
        self.tables: OrderedDict[tuple[int, int], mx.array] = OrderedDict()

    def table(self, h0: int, w0: int) -> mx.array:
        """`[1 + h0 * w0, C]` in the run dtype: row 0 = cls token + its position, the rest = positions + patch bias."""
        key = (h0, w0)
        if key not in self.tables:
            table = to_mx(dinov2_table(self.cls, self.pos, self.bias, h0, w0)).astype(self.dtype)
            mx.eval(table)
            self.tables[key] = table
            while len(self.tables) > TABLES:
                self.tables.popitem(last=False)
        self.tables.move_to_end(key)
        return self.tables[key]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Normalised `[B, 3, h, w]` CPU tensor -> float32 CPU `[B, h, w]`."""
        out = self.run(to_mx(image), self.table(image.shape[2] // PATCH, image.shape[3] // PATCH))
        mx.eval(out)
        return torch.from_numpy(np.array(out))
