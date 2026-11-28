"""Big-LaMa (a fast-Fourier-convolution ResNet inpainter) on Apple MLX, hand-written: NHWC (`mlx_conv`).

The network of `net.ffc.FFCResNetGenerator`: a 7 x 7 stem, three stride-2 reflect-padded 3 x 3 convolutions (the last
one splits its 512 channels 128 local / 384 global), 18 residual blocks of two FFC layers, three transposed
convolutions and a 7 x 7 output convolution with a sigmoid. The local and global channels live in one tensor
(`[local | global]`, as upstream's concatenation); a local output sums the local and global 3 x 3 convolutions, which is
one convolution on the concatenation. The Fourier unit (real FFT over the spatial axes, a 1 x 1 convolution on the
interleaved real / imaginary channels, inverse FFT) runs in float32 on weights rounded to the run dtype, as upstream's
fp16 network does. Every BatchNorm is folded into the convolution before it in the conversion.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fastforks._core.mlx import mx
from fastforks.shared import mlx_conv

if TYPE_CHECKING:
    from .net.ffc import FFCResNetGenerator

LAYOUT = "lama-v1"
SOURCES = (Path(__file__), Path(mlx_conv.__file__), Path(__file__).with_name("net") / "ffc.py")
LOCAL = 128  # channels of the local part of the 512 bottleneck channels


def _is(m: Any, name: str) -> bool:  # noqa: ANN401 - torch module
    return type(m).__name__ == name


class _Spectral:
    """`SpectralTransform` (1 x 1, Fourier unit, 1 x 1) of the global channels."""

    def __init__(self, st: Any, bn: Any, dt: mx.Dtype) -> None:  # noqa: ANN401 - torch modules
        self.conv1 = mlx_conv.Conv(st.conv1[0], dt, bn=st.conv1[1])
        self.fu = mlx_conv.Conv(st.fu.conv_layer, dt, bn=st.fu.bn)
        self.conv2 = mlx_conv.Conv(st.conv2, dt, bn=bn, shift=False, bias=False)  # the bias is the other term's

    def __call__(self, x: mx.array) -> mx.array:
        s = mx.maximum(self.conv1(x), 0)
        b, h, w, c = s.shape
        norm = 1 / math.sqrt(h * w)
        spec = mx.fft.rfft2(s.astype(mx.float32), axes=[1, 2]) * norm  # [B, h, w / 2 + 1, c]
        z = mx.stack([mx.real(spec), mx.imag(spec)], axis=-1).reshape(
            b, h, w // 2 + 1, 2 * c
        )  # channel c * 2 + re / im
        wt = self.fu.w.reshape(2 * c, 2 * c).T.astype(mx.float32)
        z = mx.maximum(mx.matmul(z, wt) + cast("mx.array", self.fu.b).astype(mx.float32), 0).reshape(
            b, h, w // 2 + 1, c, 2
        )
        out = mx.fft.irfft2(z[..., 0] + 1j * z[..., 1], s=(h, w), axes=[1, 2]) * (1 / norm)
        return self.conv2(s + out.astype(s.dtype))


class _Ffc:
    """`FFC_BN_ACT` of a residual block (3 x 3, reflect padding 1, global ratio 0.75 both ways) with its ReLUs."""

    def __init__(self, m: Any, dt: mx.Dtype) -> None:  # noqa: ANN401 - torch module
        f = m.ffc
        self.local = mlx_conv.Conv(f.convl2l, dt, bn=m.bn_l, more=(f.convg2l,))  # [local | global] -> local
        self.to_global = mlx_conv.Conv(f.convl2g, dt, bn=m.bn_g)
        self.spectral = _Spectral(f.convg2g, m.bn_g, dt)

    def __call__(self, x: mx.array) -> mx.array:
        xp = mlx_conv.reflect_pad(x, 1)
        out_g = self.to_global(xp[..., :LOCAL]) + self.spectral(x[..., LOCAL:])
        return mx.maximum(mx.concatenate([self.local(xp), out_g], axis=-1), 0)


class Network:
    """`FFCResNetGenerator.forward`: `[B, H, W, 4]` (masked image, mask; H, W multiples of 8) -> `[B, H, W, 3]`."""

    def __init__(self, net: FFCResNetGenerator, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`, the norms folded in."""
        dt = getattr(mx, dtype)
        layers: list[Any] = list(net.model)
        stem, *downs = (m for m in layers if _is(m, "FFC_BN_ACT"))
        self.stem = mlx_conv.Conv(stem.ffc.convl2l, dt, bn=stem.bn_l)
        self.downs = [mlx_conv.Conv(m.ffc.convl2l, dt, bn=m.bn_l) for m in downs]
        last = downs[-1]
        self.split = mlx_conv.Conv(last.ffc.convl2g, dt, bn=last.bn_g)
        self.blocks = [(_Ffc(b.conv1, dt), _Ffc(b.conv2, dt)) for b in layers if _is(b, "FFCResnetBlock")]
        ups = [m for m in layers if _is(m, "ConvTranspose2d")]
        norms = [m for m in layers if _is(m, "BatchNorm2d")]
        self.ups = [mlx_conv.ConvT(u, dt, bn=bn) for u, bn in zip(ups, norms, strict=True)]
        self.out = mlx_conv.Conv(layers[-2], dt)
        self._head = mx.compile(self._encode)
        self._blocks = [mx.compile(lambda x, f=first, g=second: x + g(f(x))) for first, second in self.blocks]
        self._tail = mx.compile(self._decode)

    def _encode(self, x: mx.array) -> mx.array:
        x = mx.maximum(self.stem(mlx_conv.reflect_pad(x, 3)), 0)
        for conv in self.downs[:-1]:
            x = mx.maximum(conv(mlx_conv.reflect_pad(x, 1)), 0)
        xp = mlx_conv.reflect_pad(x, 1)
        return mx.maximum(mx.concatenate([self.downs[-1](xp), self.split(xp)], axis=-1), 0)

    def _decode(self, x: mx.array) -> mx.array:
        for up in self.ups:
            x = mx.maximum(up(x), 0)
        return mx.sigmoid(self.out(mlx_conv.reflect_pad(x, 3)))

    def __call__(self, x: mx.array) -> mx.array:
        """The inpainted RGB (sigmoid) of the NHWC input `x`.

        Evaluated block by block: one lazy graph of all 18 blocks keeps every Fourier intermediate alive (5x the peak).
        """
        with mlx_conv.mode():
            x = self._head(x)
            mx.eval(x)
            for block in self._blocks:
                x = block(x)
                mx.eval(x)
            return self._tail(x)


def network(net: FFCResNetGenerator, dtype: str) -> Network:
    """The MLX network of a generator (float32, or the meta skeleton) in `dtype`."""
    return Network(net, dtype)
