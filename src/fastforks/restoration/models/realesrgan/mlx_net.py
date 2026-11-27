"""Real-ESRGAN's RRDBNet on Apple MLX, hand-written (`fastforks.shared.swin_sr.mlx_cnn`'s convs): NHWC, the run dtype.

The network of `model.Portable`: x2 / x1 pixel-unshuffle the (reflect-padded) input by 2 / 4, `conv_first`, the RRDB
trunk (each RRDB three residual dense blocks of five 3 x 3 convs on the growing concatenation, LeakyReLU 0.2 after
the first four, `0.2 * conv5 + x`, then `0.2 * out + x`), `conv_body` + the skip, two nearest 2x steps each with a
conv and LeakyReLU, `conv_hr` + LeakyReLU and `conv_last`. Kernels mode: one RRDB is one compiled step on a
full-resolution trunk (one per RDB measured 5 % slower at the same peak), one per RDB on a smaller one
(`RRDB_STEP_BYTES`), the Winograd convs with their LeakyReLU and the `0.2 * conv5 + x` residual
fused (conv5's weights times 0.2 at load). Portable mode: one conv per step (a whole RRDB's transients held at once
measured 1.7 times the graph translation's peak). The full-resolution tail runs in row bands (`TAIL_BAND_BYTES`),
each of its convs one step (the full map in one step held 2.6 times the graph translation's peak).
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared.swin_sr import mlx_cnn as cnn
from fastforks.shared.swin_sr import mlx_net as swin

if TYPE_CHECKING:
    from collections.abc import Callable

    from .net import RRDBNet

LAYOUT = "realesrgan-v1"
SOURCES = (Path(__file__), Path(cnn.__file__), Path(swin.__file__))

TAIL_BAND_BYTES = 64 << 20
"""Largest full-resolution map of one band of the tail (x4 at 512 x 512: 2048 x 2048 x 64 in float16, 8 bands; measured
64 MB at the graph translation's peak, 32 MB 9 % slower, 128 MB 1.3 times the peak in kernels mode)."""

END_BAND_BYTES = 128 << 20
"""Largest input of one row band of a dense block's last conv in portable mode (its 192-channel input, 4 times that in
its Winograd transform: float32 at 512 x 512 in two bands)."""

RRDB_STEP_BYTES = 16 << 20
"""Smallest trunk map run as one step per RRDB in kernels mode; smaller maps run one step per dense block (x2plus in
float16 at 512 x 512, an 8 MB map: per RDB 567 ms / +815 MB against 748 ms / +1079 MB per RRDB; x4plus, 32 MB: per
RDB 3286 ms against 3092 ms at the same peak)."""

_SLOPE = 0.2


def unshuffle(x: mx.array, r: int) -> mx.array:
    """`pixel_unshuffle(x, r)` of an NHWC map (output channel `c * r * r + i * r + j`)."""
    n, h, w, c = x.shape
    x = x.reshape(n, h // r, r, w // r, r, c).transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(n, h // r, w // r, c * r * r)


@functools.lru_cache(maxsize=4)
def _steps(dtype: mx.Dtype) -> dict[str, Callable[..., Any]]:
    """The compiled steps (in a dtype: their slope constant).

    A conv (head, last), an RRDB, a dense block and the last one with the RRDB's skip (kernels mode, small maps), a
    dense block's growing conv and its ends (portable mode), an upsampling conv and
    `conv_hr`.
    """
    slope = mx.array(_SLOPE, dtype)

    def grow(feats: mx.array, p: dict[str, mx.array]) -> mx.array:
        return mx.concatenate([feats, cnn.conv(feats, p, slope=slope)], axis=-1)

    def rdb_end(feats: mx.array, p: dict[str, mx.array], x: mx.array) -> mx.array:
        return cnn.conv(feats, p, residual=x)  # conv5 times 0.2 at load

    def rrdb_end(feats: mx.array, p: dict[str, mx.array], x: mx.array, skip: mx.array) -> mx.array:
        return rdb_end(feats, p, x) * _SLOPE + skip

    def rdb(x: mx.array, convs: list[dict[str, mx.array]]) -> mx.array:
        feats = x
        for p in convs[:-1]:
            feats = grow(feats, p)
        return rdb_end(feats, convs[-1], x)

    def rdb_last(x: mx.array, convs: list[dict[str, mx.array]], skip: mx.array) -> mx.array:
        return rdb(x, convs) * _SLOPE + skip

    def rrdb(x: mx.array, block: list[list[dict[str, mx.array]]]) -> mx.array:
        return rdb_last(functools.reduce(rdb, block[:-1], x), block[-1], x)

    def up(x: mx.array, p: dict[str, mx.array]) -> mx.array:
        return cnn.conv(swin.up2(x), p, slope=slope)

    def hr(x: mx.array, p: dict[str, mx.array]) -> mx.array:
        return cnn.conv(x, p, slope=slope)

    fs: dict[str, Callable[..., Any]] = {"conv": cnn.conv, "rrdb": rrdb, "rdb": rdb, "rdb_last": rdb_last}
    fs |= {"grow": grow, "rdb_end": rdb_end, "rrdb_end": rrdb_end, "up": up, "hr": hr}
    return {k: swin.stepped(f) for k, f in fs.items()}


class Network:
    """`RRDBNet.forward`: `[B, 3, H, W]` float32 in `[0, 1]` (padded as `Portable.pad`) -> `[B, 3, 4H', 4W']`."""

    def __init__(self, net: RRDBNet, dtype: str, *, kernels: bool) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        self.dtype, self.kernels = dt, kernels

        def c(layer: Any, scale: float = 1.0) -> dict[str, mx.array]:
            return cnn.params(layer, dt, wino=kernels, scale=scale)

        self.first = c(net.conv_first)
        self.body = [
            [[c(r.conv1), c(r.conv2), c(r.conv3), c(r.conv4), c(r.conv5, 0.2)] for r in (b.rdb1, b.rdb2, b.rdb3)]
            for b in net.body
        ]
        self.body_conv = c(net.conv_body)
        self.tail_p = {"up1": c(net.conv_up1), "up2": c(net.conv_up2), "hr": c(net.conv_hr), "last": c(net.conv_last)}
        self.unshuffle = {2: 2, 1: 4}.get(int(net.scale))
        self.feat = int(net.conv_hr.in_channels)

    @staticmethod
    def _rdbs(st: dict[str, Callable[..., Any]], x: mx.array, block: list[list[dict[str, mx.array]]]) -> mx.array:
        """An RRDB, one dense block per step (kernels mode, a trunk map under `RRDB_STEP_BYTES`)."""
        y = x
        for convs in block[:-1]:
            y = st["rdb"](y, convs)
        return st["rdb_last"](y, block[-1], x)

    @staticmethod
    def _rrdb(st: dict[str, Callable[..., Any]], x: mx.array, block: list[list[dict[str, mx.array]]]) -> mx.array:
        """An RRDB, one conv per step (portable mode), each dense block's last conv in row bands (`END_BAND_BYTES`)."""
        y = x
        for i, convs in enumerate(block):
            feats = y
            for p in convs[:-1]:
                feats = st["grow"](feats, p)
            h = feats.shape[1]
            n = min(h, -(-feats.nbytes // END_BAND_BYTES))
            outs = []
            for k in range(n):
                lo, hi = k * h // n, (k + 1) * h // n
                a, b = max(0, lo - 1), min(h, hi + 1)  # the 3 x 3 conv's halo row
                args = (feats[:, a:b], convs[-1], y[:, a:b])
                band = st["rrdb_end"](*args, x[:, a:b]) if i == len(block) - 1 else st["rdb_end"](*args)
                outs.append(band[:, lo - a : hi - a])
            y = outs[0] if n == 1 else mx.concatenate(outs, axis=1)
        return y

    def forward(self, x: mx.array) -> mx.array:
        """The network on `[B, 3, H, W]` float32 (the steps)."""
        st = _steps(self.dtype)
        conv, up, hr = st["conv"], st["up"], st["hr"]
        p = self.tail_p

        def tail(t: mx.array) -> mx.array:  # each conv its own step: their Winograd transforms are not all held
            return conv(hr(up(up(t, p["up1"]), p["up2"]), p["hr"]), p["last"])

        y = x.transpose(0, 2, 3, 1).astype(self.dtype)
        if self.unshuffle:
            y = unshuffle(y, self.unshuffle)
        f = conv(y, self.first)
        y = f
        if not self.kernels:
            rrdb = functools.partial(self._rrdb, st)
        elif f.nbytes < RRDB_STEP_BYTES:
            rrdb = functools.partial(self._rdbs, st)
        else:
            rrdb = st["rrdb"]
        for block in self.body:
            y = rrdb(y, block)
        y = cnn.conv(y, self.body_conv, residual=f)  # inside the last step's pipeline: one small conv
        b, _, w, _ = y.shape
        row = b * self.feat * 16 * w * self.dtype.size
        out = swin.bands(tail, y, scale=4, halo=2, row_bytes=row, limit=TAIL_BAND_BYTES)
        return out.transpose(0, 3, 1, 2).astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        """`forward` with its steps, evaluated."""
        cnn.set_mode(kernels=self.kernels)
        out = swin.steps(self.forward, x)
        mx.eval(out)
        return out


def network(net: RRDBNet, dtype: str, *, kernels: bool) -> Network:
    """The MLX network of an RRDBNet (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype, kernels=kernels)
