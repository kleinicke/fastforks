"""EMA-VFI on Apple MLX, written in `mlx.core` operations: two frames and a time -> the frame (`Net`).

`model._interpolate` (upstream's `MultiScaleFlow.forward` without its unused outputs), channels last. The time goes to
the convolutions, so the port keeps MLX's and removes what an exported graph spends around them:

- A transposed convolution of the U-Net (4x4, stride 2, padding 1) is one 2x2 convolution with four times the outputs:
  output pixel `2j + a` reads the inputs `j + a - 1` and `j + a`, so phase `a` of the output is the 2x2 convolution's
  result at `j + a` (1088 x 1920 float16: 76 against 106 to 156 ms each). `_up` interleaves the phases, adds the bias,
  applies the PReLU and appends the skip connection in one pass.
- `_stack` writes a U-Net block's whole input: both frames' backbone features warped by the flow at that scale, the
  block before it and zeros up to a channel count MLX's convolution is fast at (81 -> 96: 50 -> 31 ms for the first
  one). The same kernel warps the two images.
- The window attention is plain matmuls on `[windows, heads, 49, .]` (49 tokens: not `graph_mlx`'s kernel), the mask
  and the position windows are constants of the compiled function of a shape.
- `tta` runs the two passes one after the other (one pair in memory at a time).

The flow, the mask, the warps and the blend are float32, the features and convolutions the runner's dtype (as
`model.Portable`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import metal_kernel, mx, to_mx, to_torch
from fastforks.kernels.graph_mlx.ops import bilinear_nhwc, depthwise, row_layer_norm
from fastforks.shared.mlx_vit import gelu

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    Constants = Callable[[int, int, int], tuple[np.ndarray, np.ndarray | None]]

__all__ = ["LAYOUT", "Net", "prepare"]

LAYOUT = "v1"  # version of `prepare`'s parameter layout (for the conversion cache's file name)

_HEAD_DIM = 32  # config.init_model_config: num_heads = dim // 32
_STAGES = 5  # three convolution stages, two attention stages
_CONV_STAGES = 3
_EMBED_EPS = 1e-5  # the patch embeddings' nn.LayerNorm
_EPS = 1e-6  # the blocks' and stages' norm_layer
_FLOW = 4  # the flow to both frames; the mask follows it


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().float().numpy()


def _conv(p: dict[str, np.ndarray], name: str, conv: Any, act: Any = None) -> None:  # noqa: ANN401 - torch modules
    """A `Conv2d`'s weight `[O, kh, kw, I]` and bias, and its PReLU's slopes."""
    p[f"{name}.w"], p[f"{name}.b"] = _np(conv.weight).transpose(0, 2, 3, 1), _np(conv.bias)
    if act is not None:
        p[f"{name}.a"] = _np(act.weight)


def _linear(p: dict[str, np.ndarray], name: str, lin: Any) -> None:  # noqa: ANN401 - torch module
    p[f"{name}.w"], p[f"{name}.b"] = _np(lin.weight).T, _np(lin.bias)


def _norm(p: dict[str, np.ndarray], name: str, norm: Any) -> None:  # noqa: ANN401 - torch module
    p[f"{name}.w"], p[f"{name}.b"] = _np(norm.weight), _np(norm.bias)


def _phases(deconv: Any) -> np.ndarray:  # noqa: ANN401 - torch module
    """A `ConvTranspose2d(4, 2, 1)` as the 2x2 convolution `[4 * O, 2, 2, I]` of its output phases (`2 * ay + ax`)."""
    w = _np(deconv.weight)  # [I, O, 4, 4]
    i, o = w.shape[:2]
    full = np.zeros((2, 2, o, 2, 2, i), np.float32)
    for ay in range(2):
        for ax in range(2):
            # output 2j + a = input j + a - 1 through tap 3 - a, input j + a through tap 1 - a
            taps = w[:, :, [3 - ay, 1 - ay]][:, :, :, [3 - ax, 1 - ax]]  # [I, O, 2, 2]
            full[ay, ax] = taps.transpose(1, 2, 3, 0)
    return full.reshape(4 * o, 2, 2, i)


def _warped_first(w: np.ndarray, c: int) -> np.ndarray:
    """A U-Net convolution's weight with its inputs in `_stack`'s order: both frames' `c` warped channels first."""
    return np.concatenate([w[..., -2 * c :], w[..., : -2 * c]], axis=-1)


def _backbone_params(p: dict[str, np.ndarray], bone: Any) -> None:  # noqa: ANN401 - `net.MotionFormer`
    for i in range(_STAGES):
        embed: Any = getattr(bone, f"patch_embed{i + 1}", None)  # the first stage has none
        block = getattr(bone, f"block{i + 1}")
        if i < _CONV_STAGES:
            if embed is not None:
                _conv(p, f"e{i}", embed[0], embed[1])
            for k in range(len(block.conv) // 2):
                _conv(p, f"s{i}.{k}", block.conv[2 * k], block.conv[2 * k + 1])
            continue
        if i == _CONV_STAGES:
            for k, layer in enumerate(embed.layers):
                _conv(p, f"e{i}.l{k}", layer)
            p[f"e{i}.proj.w"], p[f"e{i}.proj.b"] = _np(embed.proj.weight)[:, :, 0, 0].T, _np(embed.proj.bias)
        else:
            _conv(p, f"e{i}", embed.proj)
        _norm(p, f"e{i}.norm", embed.norm)
        for j, blk in enumerate(block):
            name = f"t{i}.{j}"
            _norm(p, f"{name}.norm1", blk.norm1)
            _norm(p, f"{name}.norm2", blk.norm2)
            for key, lin in (
                ("q", blk.attn.q),
                ("kv", blk.attn.kv),
                ("cor", blk.attn.cor_embed),
                ("proj", blk.attn.proj),
                ("mproj", blk.attn.motion_proj),
                ("fc1", blk.mlp.fc1),
                ("fc2", blk.mlp.fc2),
            ):
                _linear(p, f"{name}.{key}", lin)
            dw = blk.mlp.dwconv.dwconv
            p[f"{name}.dw.w"], p[f"{name}.dw.b"] = _np(dw.weight)[:, 0].transpose(1, 2, 0), _np(dw.bias)
        _norm(p, f"n{i}", getattr(bone, f"norm{i + 1}"))


def prepare(net: Any) -> dict[str, mx.array]:  # noqa: ANN401 - `net.MultiScaleFlow`, eval, float32, on the CPU
    """The float32 parameters of the MLX network, from the PyTorch one (channels last, linears as `[in, out]`)."""
    p: dict[str, np.ndarray] = {}
    _backbone_params(p, net.feature_bone)
    for i, head in enumerate(net.block):
        for k, layer in enumerate(head.conv):
            _conv(p, f"h{i}.{k}", layer[0], layer[1])
    unet = net.unet
    for k, down in enumerate((unet.down0, unet.down1, unet.down2, unet.down3)):
        _conv(p, f"u.down{k}.0", down.conv1[0], down.conv1[1])
        _conv(p, f"u.down{k}.1", down.conv2[0], down.conv2[1])
    for k, up in enumerate((unet.up0, unet.up1, unet.up2, unet.up3)):
        p[f"u.up{k}.w"], p[f"u.up{k}.b"], p[f"u.up{k}.a"] = _phases(up[0]), _np(up[0].bias), _np(up[1].weight)
    _conv(p, "u.conv", unet.conv)
    for k in range(_STAGES):  # upstream concatenates the block before, then the warped features
        name = f"u.down{k}.0.w" if k < _STAGES - 1 else "u.up0.w"
        p[name] = _warped_first(p[name], p[f"s{k}.0.b" if k < _CONV_STAGES else f"n{k}.b"].shape[0])
    return {k: mx.array(np.ascontiguousarray(v)) for k, v in p.items()}


# The sizes are kernel inputs (`d`), not template constants: one Metal build per dtype combination. One thread writes
# 4 channels of a pixel as one vector (a thread per pixel with a channel loop is 3 to 10 times slower).
_STACK = metal_kernel(
    name="ema_vfi_stack",
    input_names=["x", "fl", "pre", "d"],
    output_names=["out"],
    source="""
        // both frames of `x` warped (upstream `warp`: bilinear, border, `align_corners`, pixel + flow `fl`), then
        // `pre`, then zeros up to CO channels
        const int h = d[0], w = d[1], C = d[2], P = d[3], CO = d[4];
        uint ox = thread_position_in_grid.x, oy = thread_position_in_grid.y;
        int g = thread_position_in_grid.z;
        if (ox >= w || oy >= h || 4 * g >= CO) return;
        size_t at = size_t(oy) * w + ox;
        device packed_vec<TO, 4>* o = (device packed_vec<TO, 4>*)(out + at * CO) + g;
        if (2 * g < C) {
            int s = 4 * g >= C;
            int gc = g - s * (C / 4);
            float4 f = float4(((const device packed_float4*)fl)[at]);
            float ix = clamp(float(ox) + f[2 * s], 0.0f, float(w - 1));
            float iy = clamp(float(oy) + f[2 * s + 1], 0.0f, float(h - 1));
            int x0 = min(int(ix), w - 1), y0 = min(int(iy), h - 1);
            int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
            float wx = ix - float(x0), wy = iy - float(y0);
            const device T* base = x + size_t(s) * h * w * C;
            float4 p00 = float4(vec<T, 4>(((const device packed_vec<T, 4>*)(base + (size_t(y0) * w + x0) * C))[gc]));
            float4 p01 = float4(vec<T, 4>(((const device packed_vec<T, 4>*)(base + (size_t(y0) * w + x1) * C))[gc]));
            float4 p10 = float4(vec<T, 4>(((const device packed_vec<T, 4>*)(base + (size_t(y1) * w + x0) * C))[gc]));
            float4 p11 = float4(vec<T, 4>(((const device packed_vec<T, 4>*)(base + (size_t(y1) * w + x1) * C))[gc]));
            o[0] = vec<TO, 4>((1.0f - wy) * ((1.0f - wx) * p00 + wx * p01) + wy * ((1.0f - wx) * p10 + wx * p11));
        } else {
            int c0 = 4 * g - 2 * C;
            const device TP* p = pre + at * P + c0;
            float4 v = float4(0.0f);
            for (int c = 0; c < 4 && c0 + c < P; ++c) v[c] = float(p[c]);
            o[0] = vec<TO, 4>(v);
        }
    """,
)

_UP = metal_kernel(
    name="ema_vfi_up",
    input_names=["y", "b", "a", "skip", "d"],
    output_names=["out"],
    source="""
        // an output pixel's phase of the 2x2 convolution `y` ([h + 1, w + 1, 4 phases x O]) plus the bias through the
        // PReLU, then the skip connection's CS channels
        const int h2 = d[0], w2 = d[1], O = d[2], CS = d[3];
        uint ox = thread_position_in_grid.x, oy = thread_position_in_grid.y;
        int g = thread_position_in_grid.z;
        if (ox >= w2 || oy >= h2 || 4 * g >= O + CS) return;
        size_t at = size_t(oy) * w2 + ox;
        device packed_vec<T, 4>* o = (device packed_vec<T, 4>*)(out + at * (O + CS)) + g;
        if (4 * g < O) {
            int ay = oy & 1, ax = ox & 1;
            const device T* p = y + (size_t((oy >> 1) + ay) * (w2 / 2 + 1) + (ox >> 1) + ax) * (4 * O);
            float4 v = float4(vec<T, 4>(((const device packed_vec<T, 4>*)(p + (2 * ay + ax) * O))[g]));
            v += float4(((const device packed_float4*)b)[g]);
            o[0] = vec<T, 4>(select(v * float4(((const device packed_float4*)a)[g]), v, v > 0.0f));
        } else {
            o[0] = ((const device packed_vec<T, 4>*)(skip + at * CS))[g - O / 4];
        }
    """,
)

_TILE = (8, 8, 4)  # threads per threadgroup
_SPARE = 8  # elements of an unused kernel input: MLX hands shorter arrays over in Metal's `constant` address space


def _grid(h: int, w: int, channels: int) -> tuple[int, int, int]:
    tw, th, tz = _TILE
    return -(-w // tw) * tw, -(-h // th) * th, -(-(channels // 4) // tz) * tz


def _stack(x: mx.array, flow: mx.array, pre: mx.array | None, width: int, dtype: mx.Dtype) -> mx.array:
    """`[1, h, w, width]`: `x` `[2, h, w, C]` (frame 0, frame 1) warped by `flow` `[1, h, w, 4]`, `pre`, then zeros."""
    _, h, w, c = x.shape
    parts = 0 if pre is None else pre.shape[-1]
    if pre is None:
        pre = mx.zeros((_SPARE,), dtype)
    return _STACK(
        inputs=[x, flow, pre, mx.array([h, w, c, parts, width, 0, 0, 0], dtype=mx.int32)],
        template=[("T", x.dtype), ("TP", pre.dtype), ("TO", dtype)],
        grid=_grid(h, w, width),
        threadgroup=_TILE,
        output_shapes=[(1, h, w, width)],
        output_dtypes=[dtype],
    )[0]


def _up(y: mx.array, b: mx.array, a: mx.array, skip: mx.array | None) -> mx.array:
    """A transposed convolution's phases `y`, interleaved, with bias and PReLU, and `skip` appended."""
    out = b.shape[0]
    h2, w2 = 2 * (y.shape[1] - 1), 2 * (y.shape[2] - 1)
    extra = 0 if skip is None else skip.shape[-1]
    if skip is None:
        skip = mx.zeros((_SPARE,), y.dtype)
    return _UP(
        inputs=[y, b, a, skip, mx.array([h2, w2, out, extra, 0, 0, 0, 0], dtype=mx.int32)],
        template=[("T", y.dtype)],
        grid=_grid(h2, w2, out + extra),
        threadgroup=_TILE,
        output_shapes=[(1, h2, w2, out + extra)],
        output_dtypes=[y.dtype],
    )[0]


def _flow(fm: mx.array, s: int) -> mx.array:
    """The flow at 1 / s, in its own pixels: upstream halves it bilinearly per scale, which is the block's mean."""
    _, h, w, _ = fm.shape
    f = fm[..., :_FLOW]
    if s > 1:
        f = f.reshape(1, h // s, s, w // s, s, _FLOW).mean(axis=(2, 4)) / s
    return mx.contiguous(f)


def _prelu(x: mx.array, a: mx.array) -> mx.array:
    return mx.where(x > 0, x, x * a)


def _roll(x: mx.array, shift: int) -> mx.array:
    """`x` `[B, h, w, C]` rolled by `shift` rows and columns."""
    return mx.roll(x, (shift, shift), (1, 2))


def _shuffle4(x: mx.array) -> mx.array:
    """Two `PixelShuffle(2)` on `[B, h, w, C]`: channel `16 c + 4 (2 ey + ex) + 2 dy + dx` -> row `4 y + 2 dy + ey`."""
    b, h, w, c = x.shape
    x = x.reshape(b, h, w, c // 16, 2, 2, 2, 2).transpose(0, 1, 6, 4, 2, 7, 5, 3)
    return x.reshape(b, 4 * h, 4 * w, c // 16)


def _down(x: mx.array, k: int) -> mx.array:
    """`F.interpolate(scale_factor=1 / k, bilinear)` of `[1, H, W, C]`: the mean of the block's middle 2x2 pixels."""
    _, h, w, c = x.shape
    x = x.reshape(1, h // k, k, w // k, k, c)[:, :, k // 2 - 1 : k // 2 + 1, :, k // 2 - 1 : k // 2 + 1]
    return x.mean(axis=(2, 4))


def _widened(w: mx.array) -> mx.array:
    """A convolution weight with its input channels zero-padded to a multiple of 16 (the 3-channel stem as it is)."""
    i = w.shape[-1]
    return w if i % 16 == 0 or i < 16 else mx.pad(w, [(0, 0), (0, 0), (0, 0), (0, -i % 16)])  # noqa: PLR2004


def fp32(k: str) -> bool:
    """`prepare`'s arrays kept (and cached) float32: `_up` adds the bias and applies the slopes in float32."""
    return k.startswith("u.up") and not k.endswith(".w")


class Net:
    """The MLX network: the converted weights and one compiled function per image size."""

    def __init__(  # noqa: PLR0913 - the model's configuration
        self,
        p32: dict[str, mx.array],
        depths: tuple[int, ...],
        window: int,
        dtype: str,
        constants: Constants,
        *,
        tta: bool,
    ) -> None:
        """`p32` is `prepare`'s output; `constants(h, w, shift)` gives a block's position windows and mask."""
        self.dtype = getattr(mx, dtype)
        self.depths, self.window, self.tta = depths, window, tta
        self.constants = constants
        self.p = {}
        for k, v in p32.items():
            if fp32(k):
                self.p[k] = v  # `_up` adds the bias and applies the slopes in float32
            else:
                self.p[k] = (_widened(v) if v.ndim == 4 else v).astype(self.dtype)  # noqa: PLR2004
        mx.eval(self.p)
        self._compiled = mx.compile(self._forward)

    def _conv(self, x: mx.array, name: str, *, stride: int = 1) -> mx.array:
        """A 3x3 convolution with its PReLU; the input zero-padded to the weight's channels."""
        w = self.p[f"{name}.w"]
        if x.shape[-1] < w.shape[-1]:
            x = mx.concatenate([x, mx.zeros((*x.shape[:-1], w.shape[-1] - x.shape[-1]), x.dtype)], axis=-1)
        return _prelu(mx.conv2d(x, w, stride=stride, padding=1) + self.p[f"{name}.b"], self.p[f"{name}.a"])

    def _linear(self, x: mx.array, name: str) -> mx.array:
        return x @ self.p[f"{name}.w"] + self.p[f"{name}.b"]

    def _norm(self, x: mx.array, name: str, eps: float) -> mx.array:
        return row_layer_norm(x, self.p[f"{name}.w"], self.p[f"{name}.b"], eps)

    def _block(self, x: mx.array, name: str, shift: int) -> tuple[mx.array, mx.array]:
        """`MotionFormerBlock.forward` of `[2, h, w, C]` (frame 0, frame 1): the tokens and the motion features."""
        _, h, w, c = x.shape
        win = self.window
        n, heads = win * win, c // _HEAD_DIM
        ph, pw = -h % win, -w % win
        hp, wp = h + ph, w + pw
        cor_np, mask_np = self.constants(h, w, shift)
        nw = cor_np.shape[0]
        cor = self._linear(mx.array(cor_np).astype(self.dtype), f"{name}.cor")  # [nW, N, motion dim]
        md = cor.shape[-1]

        def windows(y: mx.array) -> mx.array:
            y = y.reshape(2, hp // win, win, wp // win, win, y.shape[-1])
            return y.transpose(0, 1, 3, 2, 4, 5).reshape(2 * nw, n, -1)

        def image(y: mx.array) -> mx.array:
            y = y.reshape(2, hp // win, wp // win, win, win, y.shape[-1])
            y = y.transpose(0, 1, 3, 2, 4, 5).reshape(2, hp, wp, -1)
            if shift:
                y = _roll(y, shift)
            return y[:, ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]

        if ph or pw:
            x = mx.pad(x, [(0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)])
        if shift:
            x = _roll(x, -shift)
        xn = self._norm(windows(x), f"{name}.norm1", _EPS)
        q = self._linear(xn, f"{name}.q").reshape(2 * nw, n, heads, _HEAD_DIM).transpose(0, 2, 1, 3)
        kv = self._linear(xn, f"{name}.kv").reshape(2, nw, n, 2, heads, _HEAD_DIM)
        kv = mx.concatenate([kv[1], kv[0]]).transpose(2, 0, 3, 1, 4)  # each frame attends to the other one
        ce = cor.reshape(nw, n, heads, md // heads).transpose(0, 2, 1, 3)
        attn = (q * _HEAD_DIM**-0.5) @ kv[0].transpose(0, 1, 3, 2)
        if mask_np is not None:
            attn = attn.reshape(2, nw, heads, n, n) + mx.array(mask_np).astype(self.dtype)[None, :, None]
            attn = attn.reshape(2 * nw, heads, n, n)
        attn = mx.softmax(attn, axis=-1)
        out = attn @ mx.concatenate([kv[1], mx.concatenate([ce, ce])], axis=-1)  # one product for both
        app = self._linear(out[..., :_HEAD_DIM].transpose(0, 2, 1, 3).reshape(2 * nw, n, c), f"{name}.proj")
        motion = out[..., _HEAD_DIM:].transpose(0, 2, 1, 3).reshape(2, nw, n, md) - cor
        motion = image(self._linear(motion, f"{name}.mproj"))
        x = image(xn + app)  # upstream: the residual on the normed tokens
        y = self._linear(self._norm(x, f"{name}.norm2", _EPS), f"{name}.fc1")
        y = gelu(depthwise(y, self.p[f"{name}.dw.w"], self.p[f"{name}.dw.b"], (1, 1), (1, 1), (1, 1)))
        return x + self._linear(y, f"{name}.fc2"), motion

    def _backbone(self, x: mx.array) -> tuple[list[mx.array], list[mx.array]]:
        """`MotionFormer.forward` of `[2, H, W, 3]`: the 5 stages' features, the attention stages' motion features."""
        af: list[mx.array] = []
        mf: list[mx.array] = []
        for i in range(_CONV_STAGES):
            if i:
                x = self._conv(x, f"e{i}", stride=2)
            for k in range(self.depths[i]):
                x = self._conv(x, f"s{i}.{k}")
            af.append(x)
        for i in range(_CONV_STAGES, _STAGES):
            if i == _CONV_STAGES:  # CrossScalePatchEmbed: dilated strided convolutions of all three stages
                ys = []
                for level in range(_CONV_STAGES):
                    for j in range(2**level):
                        name = f"e{i}.l{len(ys)}"
                        y = mx.conv2d(
                            af[-1 - level], self.p[f"{name}.w"], stride=2 ** (level + 1), padding=1 + j, dilation=1 + j
                        )
                        ys.append(y + self.p[f"{name}.b"])
                x = self._linear(mx.concatenate(ys, axis=-1), f"e{i}.proj")
            else:
                x = mx.conv2d(x, self.p[f"e{i}.w"], stride=2, padding=1) + self.p[f"e{i}.b"]
            x = self._norm(x, f"e{i}.norm", _EMBED_EPS)
            motions = []
            for j in range(self.depths[i]):
                x, motion = self._block(x, f"t{i}.{j}", self.window // 2 if j % 2 else 0)
                motions.append(motion)
            x = self._norm(x, f"n{i}", _EPS)
            af.append(x)
            mf.append(mx.concatenate(motions, axis=-1))
        return af, mf

    def _head(self, i: int, feat: mx.array, xs: mx.array, k: int) -> mx.array:
        """`Head.forward` at 1 / k: the flow (in full-size pixels) and mask update `[1, H, W, 5]`, float32."""
        _, h, w, _ = xs.shape
        x = mx.concatenate([_shuffle4(feat), _down(xs, k).astype(self.dtype)], axis=-1)
        for n in range(3):
            x = self._conv(x, f"h{i}.{n}")
        y = bilinear_nhwc(x.astype(mx.float32), (h, w), (1 / k, 1 / k), align=False)
        return y * mx.array([k] * _FLOW + [1], dtype=mx.float32)

    def _forward(self, image1: mx.array, image2: mx.array, t: mx.array) -> mx.array:
        imgs = mx.concatenate([image1, image2]).transpose(0, 2, 3, 1)[..., ::-1]  # BGR, as upstream's cv2 frames
        imgs = mx.contiguous(imgs)
        imgs4 = mx.pad(imgs, [(0, 0), (0, 0), (0, 0), (0, 1)])  # `_stack` works on 4 channels at a time
        pair = mx.concatenate([imgs[0:1], imgs[1:2]], axis=-1)
        af, mf = self._backbone(imgs.astype(self.dtype))
        tt = t.astype(self.dtype)
        fm = warped = pre = None
        for i in range(len(mf)):
            m, a = mf[-1 - i], af[-1 - i]
            k = 2 ** (len(mf) - i)  # the head works at 1 / k of the image, on features at 1 / (4 k)
            feat = mx.concatenate([tt * m[0:1], (1 - tt) * m[1:2], a[0:1], a[1:2]], axis=-1)
            if fm is None or pre is None:
                fm = self._head(i, feat, pair, k)
            else:  # the head reads the flow in its own pixels
                fm = fm + self._head(i, feat, pre * mx.array([1.0] * 13 + [1 / k] * _FLOW), k)
            both = _stack(imgs4, _flow(fm, 1), None, 8, mx.float32)
            warped = both[..., 0:3], both[..., 4:7]
            pre = mx.concatenate([pair, *warped, fm[..., _FLOW:], fm[..., :_FLOW]], axis=-1)
        if fm is None or warped is None:
            msg = "EMA-VFI needs at least one head"
            raise ValueError(msg)
        x = pre
        skips = []
        for n in range(4):  # Unet: every block reads the one before and that scale's warped features
            x = _stack(af[n], _flow(fm, 2**n), x, self.p[f"u.down{n}.0.w"].shape[-1], self.dtype)
            x = self._conv(self._conv(x, f"u.down{n}.0", stride=2), f"u.down{n}.1")
            skips.append(x)
        x = _stack(af[4], _flow(fm, 16), skips.pop(), self.p["u.up0.w"].shape[-1], self.dtype)
        for n in range(4):
            y = mx.conv2d(x, self.p[f"u.up{n}.w"], padding=1)
            x = _up(y, self.p[f"u.up{n}.b"], self.p[f"u.up{n}.a"], skips.pop() if skips else None)
        x = mx.conv2d(x, self.p["u.conv.w"], padding=1) + self.p["u.conv.b"]
        res = mx.sigmoid(x.astype(mx.float32)) * 2 - 1
        mask = mx.sigmoid(fm[..., _FLOW:])
        merged = warped[1] + mask * (warped[0] - warped[1])
        return mx.clip(merged + res, 0, 1)[..., ::-1].transpose(0, 3, 1, 2)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> torch.Tensor:
        """The frame at `timestep` of one padded RGB pair `[1, 3, H, W]` (float32, CPU), `[1, 3, H, W]` float32."""
        t = mx.array([timestep], dtype=mx.float32)
        a, b = to_mx(image1.float()), to_mx(image2.float())
        out = self._compiled(a, b, t)
        if self.tta:  # upstream fast_TTA: the mean with the pair turned by 180 degrees
            mx.eval(out)
            turned = self._compiled(a[:, :, ::-1, ::-1], b[:, :, ::-1, ::-1], t)
            out = (out + turned[:, :, ::-1, ::-1]) / 2
        return to_torch(out)
