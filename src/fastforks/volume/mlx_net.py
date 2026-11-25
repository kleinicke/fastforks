"""nnU-Net v2's `PlainConvUNet` (3D, 3x3x3 kernels) in `mlx.core` operations (Apple MLX; `volume._mlx`).

Written out instead of replayed from an export (`shared/mlx_graph`), channels-last (NDHWC) throughout:

- the stride-1 3x3x3 convolutions are `conv_mlx.conv3d` (F(4,3) Winograd per depth slice, the three depth taps one
  GEMM) in depth bands of at most `_BAND_MB` of transforms (the band reads one slice beyond either end and drops
  the two halo outputs), in float16 only (in float32 the kernel ran a 128^3 window in 16-60 s against 0.5 s with
  `mx.conv3d`); the small deep maps and the stride-2 convolutions are `mx.conv3d`; the one-channel input
  layer is an im2col of the 27 taps and one GEMM (`mx.conv3d` is 4x slower there);
- the convolution biases in front of an instance norm are left out (the norm subtracts them again);
- every instance norm takes float32 statistics in one pass (`graph_mlx.channel_stats`); the float32 scale and shift
  are rounded to the activation dtype and applied with the LeakyReLU in one compiled elementwise kernel (`_norm_act`);
- the stride-2 transposed convolutions (kernel 2) are one GEMM to the eight output phases and an interleave;
- only the last segmentation head runs (no deep supervision), as a GEMM.

The window is one lazy graph, not compiled and not evaluated layer by layer. Its cost beyond the arithmetic is fresh
Metal allocations (zero-filled pages) for the transients; the runner asks for a 2 GB buffer cache (`volume._mlx`).
Dataset291, one 128^3 window, float16 (ms):

| buffer cache | lazy, band 512 MB | layer by layer | lazy, band 128 MB |
|---|---|---|---|
| 128 MB (the base) | 420 | 495 | |
| 2 GB | 350 | 460 | 360 |
| unlimited (about 6 GB held) | 255 | 256 | |

`mx.compile` of the whole forward ran 806 ms.

`MlxUNet.logits` is the window forward the predictor's MLX loop calls (`[1, D, H, W, C]` -> `[1, D, H, W, heads]` in the
run dtype); `__call__` is the `UNet` call on torch tensors for `Predictor`'s general path.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.errors import WeightsError
from fastforks._core.mlx import mx, to_mx
from fastforks.kernels.conv_mlx import ops as conv_ops
from fastforks.kernels.graph_mlx import ops

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fastforks.volume._nnunet import Plan

LAYOUT = "unet-v1"  # the parameter layout of `params` (the conversion cache's name)
_BAND_MB = int(os.environ.get("FF_UNET_BAND_MB", "128"))  # Winograd transforms per depth band
_WINO_MIN = 16**3  # voxels per sample below which `mx.conv3d` wins (measured: 16^3 2.5x faster, 8^3 mixed)


def params(sd: Mapping[str, torch.Tensor], plan: Plan) -> dict[str, mx.array]:
    """The float32 parameters of a `UNet` state dict in this module's layout (OIDHW -> ODHWI, the heads as GEMMs)."""

    def ohwi(t: torch.Tensor) -> mx.array:
        return to_mx(t.detach().float().permute(0, 2, 3, 4, 1).contiguous())

    def f32(t: torch.Tensor) -> mx.array:
        return to_mx(t.detach().float().contiguous())

    out: dict[str, mx.array] = {}

    def block(dst: str, src: str) -> None:
        out[f"{dst}.w"] = ohwi(sd[f"{src}.conv.weight"])
        out[f"{dst}.g"], out[f"{dst}.b"] = f32(sd[f"{src}.norm.weight"]), f32(sd[f"{src}.norm.bias"])

    n = len(plan.features)
    for i in range(n):
        for j in range(plan.n_conv_encoder[i]):
            block(f"e{i}.{j}", f"encoder.stages.{i}.0.convs.{j}")
    for s in range(n - 1):
        w = sd[f"decoder.transpconvs.{s}.weight"].detach().float()  # [Cin, Cout, 2, 2, 2]
        out[f"t{s}.w"] = to_mx(w.permute(0, 2, 3, 4, 1).reshape(w.shape[0], -1).contiguous())  # [Cin, 8 * Cout]
        if f"decoder.transpconvs.{s}.bias" in sd:
            out[f"t{s}.b"] = f32(sd[f"decoder.transpconvs.{s}.bias"])
        for j in range(plan.n_conv_decoder[s]):
            block(f"d{s}.{j}", f"decoder.stages.{s}.convs.{j}")
    head = sd[f"decoder.seg_layers.{n - 2}.weight"].detach().float()  # [heads, C, 1, 1, 1]
    out["head.w"] = to_mx(head.reshape(head.shape[0], -1).T.contiguous())
    out["head.b"] = f32(sd[f"decoder.seg_layers.{n - 2}.bias"])
    # read or left out on purpose: the biases in front of a norm, the deep-supervision heads
    unread = {k for k in sd if not k.endswith(".conv.bias") and not k.startswith("decoder.seg_layers.")}
    unread -= {k for k in unread if k.startswith(("encoder.stages.", "decoder.stages.")) and ".norm." in k}
    unread -= {k for k in unread if k.startswith(("encoder.stages.", "decoder.stages.")) and k.endswith(".conv.weight")}
    unread -= {k for k in unread if k.startswith("decoder.transpconvs.")}
    if unread:
        msg = f"nnU-Net checkpoint: parameters the MLX U-Net does not read: {sorted(unread)[:5]}"
        raise WeightsError(msg)
    return out


def supported(plan: Plan) -> bool:
    """Whether `MlxUNet` runs `plan`: a plain 3D U-Net, 3x3x3 kernels, strides 1 (first stage) or 2, LeakyReLU."""
    return (
        plan.network != "resenc"
        and plan.dims == 3  # noqa: PLR2004
        and all(tuple(k) == (3, 3, 3) for k in plan.kernel_sizes)
        and tuple(plan.strides[0]) == (1, 1, 1)
        and all(tuple(s) == (2, 2, 2) for s in plan.strides[1:])
        and all(n % 2 ** (len(plan.features) - 1) == 0 for n in plan.patch_size)
    )


def _band_conv(x: mx.array, u: mx.array) -> mx.array:
    """`conv_ops.conv3d(x, u)` in depth bands whose transforms stay under `_BAND_MB` (halo slices dropped)."""
    n, d, h, w, c = x.shape
    per_slice = 36 * n * -(-h // 4) * -(-w // 4) * (3 * c + u.shape[2]) * x.dtype.size
    band = max(2, (_BAND_MB << 20) // per_slice)
    if band >= d:
        return conv_ops.conv3d(x, u)
    outs = []
    for a in range(0, d, band):
        b = min(a + band, d)
        lo, hi = max(a - 1, 0), min(b + 1, d)
        outs.append(conv_ops.conv3d(x[:, lo:hi], u)[:, a - lo : b - lo])
    return mx.concatenate(outs, axis=1)


def _im2col_conv(x: mx.array, wk: mx.array) -> mx.array:
    """3x3x3 stride-1 convolution (padding 1) of a one-channel `x` `[N, D, H, W, 1]` with taps `wk` `[27, Cout]`."""
    d, h, w = x.shape[1:4]
    xp = mx.pad(x, [(0, 0), (1, 1), (1, 1), (1, 1), (0, 0)])
    cols = [xp[:, a : a + d, b : b + h, c : c + w] for a in range(3) for b in range(3) for c in range(3)]
    return mx.concatenate(cols, axis=-1) @ wk


class MlxUNet:
    """A `UNet` of `plan` on MLX in `dtype` (the name), from `params(...)`."""

    def __init__(self, p: Mapping[str, mx.array], plan: Plan, dtype: str) -> None:
        """Round the float32 parameters to `dtype` and transform the stride-1 kernels for Winograd."""
        self.plan = plan
        self.dtype = dt = getattr(mx, dtype)
        self.eps, self.slope = plan.norm_eps, plan.slope
        self.n = n = len(plan.features)
        self._convs: dict[str, tuple[Callable[[mx.array], mx.array], mx.array, mx.array]] = {}
        size = plan.patch_size[0] * plan.patch_size[1] * plan.patch_size[2]
        for i in range(n):
            for j in range(plan.n_conv_encoder[i]):
                stride = 2 if i and j == 0 else 1
                self._block(f"e{i}.{j}", p, stride, size // 8**i)
        for s in range(n - 1):
            for j in range(plan.n_conv_decoder[s]):
                self._block(f"d{s}.{j}", p, 1, size // 8 ** (n - 2 - s))
        self._up = [(p[f"t{s}.w"].astype(dt), p.get(f"t{s}.b")) for s in range(n - 1)]
        self._up = [(w, None if b is None else b.astype(dt)) for w, b in self._up]
        self._head = (p["head.w"].astype(dt), p["head.b"].astype(dt))
        mx.eval(self._convs, self._up, self._head)
        slope = plan.slope
        self._norm_act = mx.compile(lambda y, scale, shift: mx.maximum(v := y * scale + shift, v * slope))

    def _block(self, name: str, p: Mapping[str, mx.array], stride: int, voxels: int) -> None:
        dt = self.dtype
        w = p[f"{name}.w"]
        cout, cin = w.shape[0], w.shape[-1]
        norm = (p[f"{name}.g"], p[f"{name}.b"])
        if stride == 1 and cin == 1:
            wk = w.reshape(cout, 27).T.astype(dt)  # taps in (d, h, w) order, as the im2col below
            self._convs[name] = (lambda x, wk=wk: _im2col_conv(x, wk), *norm)
        elif dt == mx.float16 and stride == 1 and cin >= 16 and cin % 2 == 0 and cout % 2 == 0 and voxels >= _WINO_MIN:  # noqa: PLR2004
            u = conv_ops.weights3d(w.astype(dt), 4, dt)
            self._convs[name] = (lambda x, u=u: _band_conv(x, u), *norm)
        else:
            wd = w.astype(dt)
            self._convs[name] = (lambda x, wd=wd, s=stride: mx.conv3d(x, wd, stride=s, padding=1), *norm)

    def _conv_norm_act(self, name: str, x: mx.array) -> mx.array:
        conv, g, b = self._convs[name]
        y = conv(x)
        n, c = y.shape[0], y.shape[-1]
        out = []
        for i in range(n):  # instance norm: statistics per sample
            mean, var = ops.channel_stats(y[i].reshape(-1, c))
            scale = g * mx.rsqrt(var + self.eps)
            shift = b - mean * scale
            out.append(self._norm_act(y[i], scale.astype(y.dtype), shift.astype(y.dtype)))
        return out[0][None] if n == 1 else mx.stack(out)

    def _upsample(self, s: int, x: mx.array) -> mx.array:
        w, b = self._up[s]
        n, d, h, wd, c = x.shape
        co = w.shape[1] // 8
        y = (x.reshape(-1, c) @ w).reshape(n, d, h, wd, 2, 2, 2, co)
        if b is not None:
            y = y + b
        return y.transpose(0, 1, 4, 2, 5, 3, 6, 7).reshape(n, 2 * d, 2 * h, 2 * wd, co)

    def _forward(self, x: mx.array) -> mx.array:
        plan, skips = self.plan, []
        for i in range(self.n):
            for j in range(plan.n_conv_encoder[i]):
                x = self._conv_norm_act(f"e{i}.{j}", x)
            skips.append(x)
        x = skips.pop()
        for s in range(self.n - 1):
            x = mx.concatenate([self._upsample(s, x), skips.pop()], axis=-1)
            for j in range(plan.n_conv_decoder[s]):
                x = self._conv_norm_act(f"d{s}.{j}", x)
        w, b = self._head
        return (x.reshape(-1, x.shape[-1]) @ w + b).reshape(*x.shape[:-1], w.shape[1])

    def logits(self, x: mx.array) -> mx.array:
        """Logits `[N, D, H, W, heads]` (run dtype) of windows `[N, D, H, W, C]` (any float dtype)."""
        return self._forward(x.astype(self.dtype))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """`UNet.forward` on torch tensors: float32 logits `[B, heads, *patch]` of `[B, C, *patch]`."""
        y = self.logits(to_mx(x.float().permute(0, 2, 3, 4, 1).contiguous())).astype(mx.float32)
        return torch.from_numpy(np.array(y)).permute(0, 4, 1, 2, 3).to(x.device)
