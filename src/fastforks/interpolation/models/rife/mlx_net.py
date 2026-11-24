"""RIFE on Apple MLX, written in `mlx.core` operations: two frames and a time -> the frame (`Net`).

`model._interpolate` (upstream's `IFNet.forward`, evaluated where the blocks read it) with the convolutions as MLX's
and everything between them as two Metal kernels, so that no full-resolution intermediate is written:

- A transposed convolution (4x4, stride 2, padding 1) is one 2x2 convolution with four times the outputs: output pixel
  `2j + a` reads the inputs `j + a - 1` and `j + a`, so phase `a` of the output is the 2x2 convolution's result at
  `j + a` (`mx.conv_transpose2d` is 3-4x slower). Its result stays in that phase layout: the kernels index it, the
  `PixelShuffle` after an IFBlock's included, and add the bias.
- `_block_input` writes an IFBlock's whole input at the block's resolution: per output pixel the mean over the 2x2
  full-resolution pixels a bilinear 1/s downscale reads (one pixel at scale 1) of both warped images and feature maps,
  the last block's mask and features and the flow. The flow at a pixel is the sum of the earlier blocks' upsampled
  flows, taken from their low-resolution outputs in float32: the flow field itself is never stored.
- `_blend` is the same warp of the two images with the final flow and the mask blend, written channels first.

The weights are read from the PyTorch module once: the residual scales folded into their convolutions, channels last,
ragged widths zero-padded (15 / 28 input channels to 16 / 32: a stride-2 convolution of 28 channels at 1152 x 1920
takes 7.8 ms, of 32 4.2 ms; 24 -> 32, 48 -> 64, 96 -> 128).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import metal_kernel, mx, to_mx, to_torch

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

__all__ = ["LAYOUT", "Net", "prepare"]

LAYOUT = "v1"  # version of `prepare`'s parameter layout (for the conversion cache's file name)

_OUT = 13  # an IFBlock's outputs: flow (4), mask (1), features (8)
_FEAT = 4  # the encoder's feature channels
_RES = 8  # residual convolutions of an IFBlock
_WIDTHS = (16, 32, 64, 128)  # channel counts MLX's convolution is fast at; wider layers: multiples of 64


def _width(n: int) -> int:
    """`n` channels as the layer is run (zero filters fill up)."""
    return next((w for w in _WIDTHS if n <= w), -(-n // 64) * 64)


def _conv(conv: Any, scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - torch module
    """A `Conv2d`'s weight `[O, kh, kw, I]` and bias, zero-padded to the run widths (`scale`: per output channel)."""
    w = conv.weight.detach().float().numpy()
    b = conv.bias.detach().float().numpy()
    if scale is not None:
        w, b = w * scale[:, None, None, None], b * scale
    o, i = w.shape[:2]
    full = np.zeros((_width(o), w.shape[2], w.shape[3], _width(i)), np.float32)
    full[:o, :, :, :i] = w.transpose(0, 2, 3, 1)
    bias = np.zeros(len(full), np.float32)
    bias[:o] = b
    return full, bias


def _phases(deconv: Any, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - torch module
    """A `ConvTranspose2d(4, 2, 1)` as the 2x2 convolution of its four output phases, and its bias.

    The weight is `[4 * O, 2, 2, I]` (inputs zero-padded to the run width), phase-major (`2 * ay + ax`); `order` lists
    the transposed convolution's output channels as they are stored within a phase (the bias is in that order too).
    """
    w = deconv.weight.detach().float().numpy()[:, order]  # [I, O, 4, 4]
    i, o = w.shape[:2]
    full = np.zeros((2, 2, o, 2, 2, _width(i)), np.float32)
    for ay in range(2):
        for ax in range(2):
            # output 2j + a = input j + a - 1 through tap 3 - a, input j + a through tap 1 - a
            taps = w[:, :, [3 - ay, 1 - ay]][:, :, :, [3 - ax, 1 - ax]]  # [I, O, 2, 2]
            full[ay, ax, :, :, :, :i] = taps.transpose(1, 2, 3, 0)
    return full.reshape(4 * o, 2, 2, -1), deconv.bias.detach().float().numpy()[order]


def prepare(net: Any) -> dict[str, mx.array]:  # noqa: ANN401 - `net.IFNet`, eval, float32, on the CPU
    """The float32 parameters of the MLX network, from the PyTorch one."""
    p: dict[str, np.ndarray] = {}
    enc = net.encode
    for k, conv in enumerate((enc.cnn0, enc.cnn1, enc.cnn2)):
        w, b = _conv(conv)
        p[f"enc.{k}.w"], p[f"enc.{k}.b"] = w[:, :, :, :3] if k == 0 else w, b
    p["enc.3.w"], p["enc.3.b"] = _phases(enc.cnn3, np.arange(_FEAT))
    # an IFBlock's 52 transposed-convolution outputs are `PixelShuffle(2)`'s `c * 4 + 2 * dy + dx`: sub-pixel-major
    shuffle = np.arange(4 * _OUT).reshape(_OUT, 4).T.reshape(-1)
    for n, block in enumerate((net.block0, net.block1, net.block2, net.block3, net.block4)):
        for k in range(2):
            p[f"b{n}.stem{k}.w"], p[f"b{n}.stem{k}.b"] = _conv(block.conv0[k][0])
        for k, res in enumerate(block.convblock):
            p[f"b{n}.res{k}.w"], p[f"b{n}.res{k}.b"] = _conv(res.conv, res.beta.detach().float().numpy().reshape(-1))
        p[f"b{n}.last.w"], p[f"b{n}.last.b"] = _phases(block.lastconv[0], shuffle)
    return {k: mx.array(v) for k, v in p.items()}


# The four bilinear taps of level @ (an earlier block's 13 outputs at 1 / SL of the image) at the pixel (px, py): the
# upsampling `F.interpolate(scale_factor=SL)` evaluated there. `l` is the block's 2x2-phase convolution result
# (`[HL / 4 + 1, WL / 4 + 1, 4 phases x 4 sub-pixels x 13]`), `b` the transposed convolution's bias. The sizes are
# kernel inputs (`d`), not template constants: one Metal build serves every image size and variant.
_LEVEL = """
            if (NL > @) {
                const int SL = d[4 + @], HL = H / SL, WL = W / SL, WP = WL / 4 + 1;
                float ys = max((float(py) + 0.5f) / float(SL) - 0.5f, 0.0f);
                float xs = max((float(px) + 0.5f) / float(SL) - 0.5f, 0.0f);
                int y0 = min(int(ys), HL - 1), x0 = min(int(xs), WL - 1);
                int yy[2] = {y0, min(y0 + 1, HL - 1)}, xx[2] = {x0, min(x0 + 1, WL - 1)};
                float wy = ys - float(y0), wx = xs - float(x0);
                for (int a = 0; a < 2; ++a) for (int e = 0; e < 2; ++e) {
                    int y = yy[a], x = xx[e];
                    float wgt = (a ? wy : 1.0f - wy) * (e ? wx : 1.0f - wx);
                    int ay = (y >> 1) & 1, ax = (x >> 1) & 1, sub = ((y & 1) * 2 + (x & 1)) * 13;
                    const device T* p = l@ + (size_t((y >> 2) + ay) * WP + (x >> 2) + ax) * 208
                        + (ay * 2 + ax) * 52 + sub;
                    const device float* q = b@ + sub;
                    for (int c = 0; c < 4; ++c) fl[c] += wgt * float(SL) * (float(p[c]) + q[c]);
                    if (NL == @ + 1) { REST }
                }
            }
"""

# Upstream `warp` of side @ at (px, py) by its flow: bilinear, border, `align_corners` (pixel + flow). `feat` is the
# encoder's 2x2-phase convolution result (`[H / 2 + 1, W / 2 + 1, 4 phases x 4]`).
_SIDE = """
            {
                float ix = clamp(float(px) + fl[2 * @], 0.0f, float(W - 1));
                float iy = clamp(float(py) + fl[2 * @ + 1], 0.0f, float(H - 1));
                int x0 = int(ix), y0 = int(iy);
                int yy[2] = {y0, min(y0 + 1, H - 1)}, xx[2] = {x0, min(x0 + 1, W - 1)};
                float wy = iy - float(y0), wx = ix - float(x0);
                for (int a = 0; a < 2; ++a) for (int e = 0; e < 2; ++e) {
                    int y = yy[a], x = xx[e];
                    float wgt = (a ? wy : 1.0f - wy) * (e ? wx : 1.0f - wx);
                    const device T* pi = img@ + (size_t(y) * W + x) * 3;
                    for (int c = 0; c < 3; ++c) acc[3 * @ + c] += wgt * float(pi[c]);
                    FEAT
                }
            }
"""

_SIDE_FEAT = """
                    const device T* pf = feat@ + (size_t((y >> 1) + (y & 1)) * (W / 2 + 1) + (x >> 1) + (x & 1)) * 16
                        + ((y & 1) * 2 + (x & 1)) * 4;
                    for (int c = 0; c < 4; ++c) acc[6 + 4 * @ + c] += wgt * float(pf[c]);
"""


def _levels(count: int, rest: str) -> str:
    """The flow at (px, py) from up to `count` earlier blocks; `rest` is what the last one adds per tap."""
    return "".join(_LEVEL.replace("@", str(i)).replace("REST", rest) for i in range(count))


def _sides(feat: str) -> str:
    return "".join(_SIDE.replace("FEAT", feat).replace("@", str(n)) for n in range(2))


def _names(count: int) -> list[str]:
    return [f"{n}{i}" for i in range(count) for n in "lb"]


@functools.cache
def _input_kernel(count: int) -> Callable[..., list[mx.array]]:
    """`_block_input`'s kernel for up to `count` earlier blocks."""
    rest = "for (int c = 4; c < 13; ++c) acc[11 + c] += wgt * (float(p[c]) + q[c]);"
    source = f"""
        // one thread: one pixel of the block's input, all its channels (zeros from the last one to CO)
        const int H = d[0], W = d[1], NL = d[2], S = d[3], CO = d[4 + {count}];
        uint ox = thread_position_in_grid.x, oy = thread_position_in_grid.y;
        if (ox >= W / S || oy >= H / S) return;
        const int np = S == 1 ? 1 : 2, first = S == 1 ? 0 : S / 2 - 1, last = NL ? 28 : 15;
        float acc[28];
        for (int c = 0; c < 28; ++c) acc[c] = 0.0f;
        for (int dy = 0; dy < np; ++dy) for (int dx = 0; dx < np; ++dx) {{
            int py = int(oy) * S + first + dy, px = int(ox) * S + first + dx;
            float fl[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
            {_levels(count, rest)}
            {_sides(_SIDE_FEAT)}
            for (int c = 0; c < 4; ++c) acc[24 + c] += fl[c];
        }}
        float inv = 1.0f / float(np * np);
        device T* o = out + (size_t(oy) * (W / S) + ox) * CO;
        for (int c = 0; c < 6; ++c) o[c] = T(acc[c] * inv);
        for (int c = 0; c < 8; ++c) o[6 + c] = T(acc[6 + c] * inv + fb[c & 3]);
        o[14] = T(t[0]);
        for (int c = 15; c < last; ++c) o[c] = T(c < 24 ? acc[c] * inv : acc[c] * inv / float(S));
        for (int c = last; c < CO; ++c) o[c] = T(0);
    """
    return metal_kernel(
        name=f"rife_block_input_{count}",
        input_names=["img0", "img1", "feat0", "feat1", "fb", "t", "d", *_names(count)],
        output_names=["out"],
        source=source,
    )


@functools.cache
def _blend_kernel(count: int) -> Callable[..., list[mx.array]]:
    """`_blend`'s kernel after `count` blocks."""
    source = f"""
        // one thread: one pixel of the frame, written channels first
        const int H = d[0], W = d[1], NL = d[2];
        uint px = thread_position_in_grid.x, py = thread_position_in_grid.y;
        if (px >= W || py >= H) return;
        float acc[6] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
        float fl[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
        float mask = 0.0f;
        {_levels(count, "mask += wgt * (float(p[4]) + q[4]);")}
        {_sides("")}
        float m = 1.0f / (1.0f + exp(-mask));
        for (int c = 0; c < 3; ++c) out[(size_t(c) * H + py) * W + px] = acc[3 + c] + m * (acc[c] - acc[3 + c]);
    """
    return metal_kernel(
        name=f"rife_blend_{count}",
        input_names=["img0", "img1", "d", *_names(count)],
        output_names=["out"],
        source=source,
    )


_TILE = 16  # threads per threadgroup side


def _grid(h: int, w: int) -> tuple[int, int, int]:
    return -(-w // _TILE) * _TILE, -(-h // _TILE) * _TILE, 1


Level = tuple[mx.array, mx.array, int]  # a block's phase-layout output, its bias, its scale


def _level_inputs(levels: list[Level], count: int, dtype: mx.Dtype) -> list[mx.array]:
    """The kernels' level inputs: `levels`, then placeholders up to `count` (never read)."""
    # 8 elements: MLX hands a kernel shorter arrays in Metal's `constant` address space, a different pointer type
    spare = [mx.zeros((8,), dtype), mx.zeros((8,), mx.float32)] * (count - len(levels))
    return [*(a for lv in levels for a in lv[:2]), *spare]


def _block_input(  # noqa: PLR0913, PLR0917 - the kernel's inputs
    images: list[mx.array],
    feats: list[mx.array],
    fb: mx.array,
    t: mx.array,
    levels: list[Level],
    s: int,
    width: int,
    count: int,
) -> mx.array:
    """The input `[1, H / s, W / s, width]` of the block at scale `s` after the blocks `levels` (at most `count`)."""
    _, h, w, _ = images[0].shape
    dtype = images[0].dtype
    scales = [lv[2] for lv in levels] + [1] * (count - len(levels))
    d = mx.array([h, w, len(levels), s, *scales, width], dtype=mx.int32)
    return _input_kernel(count)(
        inputs=[*images, *feats, fb, t, d, *_level_inputs(levels, count, dtype)],
        template=[("T", dtype)],
        grid=_grid(h // s, w // s),
        threadgroup=(_TILE, _TILE, 1),
        output_shapes=[(1, h // s, w // s, width)],
        output_dtypes=[dtype],
    )[0]


def _blend(images: list[mx.array], levels: list[Level]) -> mx.array:
    """The frame `[1, 3, H, W]`, float32: both images warped by the final flow, blended by the last block's mask."""
    _, h, w, _ = images[0].shape
    d = mx.array([h, w, len(levels), 1, *(lv[2] for lv in levels)], dtype=mx.int32)
    return _blend_kernel(len(levels))(
        inputs=[*images, d, *_level_inputs(levels, len(levels), images[0].dtype)],
        template=[("T", images[0].dtype)],
        grid=_grid(h, w),
        threadgroup=(_TILE, _TILE, 1),
        output_shapes=[(1, 3, h, w)],
        output_dtypes=[mx.float32],
    )[0]


def _act(x: mx.array) -> mx.array:
    return mx.maximum(x, 0.2 * x)  # LeakyReLU(0.2)


class Net:
    """The MLX network: the converted weights and one compiled function per image size."""

    def __init__(self, p32: dict[str, mx.array], scales: tuple[int, ...], dtype: str) -> None:
        """`p32` is `prepare`'s output, `scales` the blocks' downscale factors, `dtype` the convolutions' dtype."""
        self.dtype = getattr(mx, dtype)
        self.scales = scales
        # the biases the kernels add stay float32
        self.p = {k: v if k.endswith(("enc.3.b", "last.b")) else v.astype(self.dtype) for k, v in p32.items()}
        mx.eval(self.p)
        self._compiled = mx.compile(self._forward)

    def _conv(self, x: mx.array, name: str, *, stride: int = 1) -> mx.array:
        return mx.conv2d(x, self.p[f"{name}.w"], stride=stride, padding=1) + self.p[f"{name}.b"]

    def _encode(self, x: mx.array) -> mx.array:
        """Upstream `Head` of one image `[1, H, W, 3]`: the features in the 2x2-phase layout, without the bias."""
        x = _act(self._conv(x, "enc.0", stride=2))
        x = _act(self._conv(x, "enc.1"))
        x = _act(self._conv(x, "enc.2"))
        return mx.conv2d(x, self.p["enc.3.w"], padding=1)

    def _block(self, x: mx.array, n: int) -> mx.array:
        """Upstream `IFBlock` `n` of its input: the 13 outputs in the 2x2-phase layout, without the bias."""
        x = _act(self._conv(x, f"b{n}.stem0", stride=2))
        x = _act(self._conv(x, f"b{n}.stem1", stride=2))
        for k in range(_RES):
            x = _act(self._conv(x, f"b{n}.res{k}") + x)
        return mx.conv2d(x, self.p[f"b{n}.last.w"], padding=1)

    def _forward(self, image1: mx.array, image2: mx.array, t: mx.array) -> mx.array:
        images = [x.transpose(0, 2, 3, 1).astype(self.dtype) for x in (image1, image2)]
        feats = [self._encode(x) for x in images]  # per image, as upstream
        levels: list[Level] = []
        for n, s in enumerate(self.scales):
            width = self.p[f"b{n}.stem0.w"].shape[-1]
            x = _block_input(images, feats, self.p["enc.3.b"], t, levels, s, width, len(self.scales) - 1)
            levels.append((self._block(x, n), self.p[f"b{n}.last.b"], s))
        return _blend(images, levels)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> torch.Tensor:
        """The frame at `timestep` of one padded pair `[1, 3, H, W]` (float32, CPU), `[1, 3, H, W]` float32."""
        t = mx.array([timestep], dtype=mx.float32)
        return to_torch(self._compiled(to_mx(image1.float()), to_mx(image2.float()), t))
