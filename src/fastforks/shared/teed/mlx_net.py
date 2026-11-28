"""TED in `mlx.core` operations (Apple MLX), for the families that run it at 1280 px and above (AnyLine).

Written out instead of replayed from an export (`shared/mlx_graph`): the export is most of AnyLine's first call
(1.2 s against 0.34 s on MPS), and the network's cost is not its convolutions (58 K weights) but full-resolution
element-wise work, which three Metal kernels take:

- `smish`: `x * tanh(log(1 + sigmoid(x)))` is `x * n / (n + 2)` with `n = s * (s + 2)`, `s = sigmoid(x)`: one
  exponential per element, in float32;
- `DoubleFusion` (24 channels at the input size) in two passes: the grouped 3 -> 24 convolution writing its Smish and
  its channel sum, then the depthwise 24 -> 24 convolution reduced to the fused map in registers. The 24-channel
  maps `attn` and `attn2` are never stored.

The few-channel 3x3 convolutions run on `conv_mlx.direct`, the one-channel transposed convolutions as phase
multiply-adds (`mlx_graph._conv_transpose2d_phases`). Channels-last throughout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import metal_kernel, mx
from fastforks.kernels.conv_mlx import ops as conv_ops
from fastforks.shared.mlx_conv import swin
from fastforks.shared.mlx_graph import _conv_transpose2d_phases, conv1x1, conv1x1_ok

if TYPE_CHECKING:
    from .net.ted import TED

_SMISH_FN = """
template <typename V> inline V smish_of(V v) {
    V s = 1.0f / (1.0f + exp(-v));
    V n = s * (s + 2.0f);
    return v * (n / (n + 2.0f));
}
"""
_SMISH = metal_kernel(
    name="teed_smish",
    input_names=["x"],
    output_names=["out"],
    header=_SMISH_FN,
    source="out[thread_position_in_grid.x] = T(smish_of(float(x[thread_position_in_grid.x])));",
)
_FUSE_A = metal_kernel(
    name="teed_fusion_a",
    input_names=["x", "w", "b"],
    output_names=["act", "total"],
    header=_SMISH_FN,
    source="""
        // one thread per pixel: DWconv1 (3 -> 24 channels, 8 per input channel) of the Smish'd heads `x` [N, H, W, 4]
        // (channel 3 is padding); `act` = smish(attn) [N, H, W, 24], `total` = sum over the channels of attn
        int H = x_shape[1], W = x_shape[2];
        int px = thread_position_in_grid.x, py = thread_position_in_grid.y, n = thread_position_in_grid.z;
        if (px >= W || py >= H) return;
        float4 acc[6];
        for (int q = 0; q < 6; ++q) acc[q] = float4(b[4 * q], b[4 * q + 1], b[4 * q + 2], b[4 * q + 3]);
        const device vec<T, 4>* x4 = (const device vec<T, 4>*)x;
        const device float4* w4 = (const device float4*)w;
        for (int i = 0; i < 3; ++i) {
            int iy = py + i - 1;
            if (iy < 0 || iy >= H) continue;
            for (int j = 0; j < 3; ++j) {
                int ix = px + j - 1;
                if (ix < 0 || ix >= W) continue;
                float4 v = float4(x4[(size_t(n) * H + iy) * W + ix]);
                for (int q = 0; q < 6; ++q) acc[q] += w4[(i * 3 + j) * 6 + q] * v[q / 2];
            }
        }
        size_t p = (size_t(n) * H + py) * W + px;
        float4 sum = 0;
        for (int q = 0; q < 6; ++q) {
            sum += acc[q];
            ((device vec<T, 4>*)act)[p * 6 + q] = vec<T, 4>(smish_of(acc[q]));
        }
        total[p] = sum.x + sum.y + sum.z + sum.w;
    """,
)
_FUSE_B = metal_kernel(
    name="teed_fusion_b",
    input_names=["act", "total", "w", "b"],
    output_names=["out"],
    header=_SMISH_FN,
    source="""
        // one thread per pixel: smish(sum over the channels of (DWconv2(act) + attn)); `b` is DWconv2's bias sum
        int H = act_shape[1], W = act_shape[2];
        int px = thread_position_in_grid.x, py = thread_position_in_grid.y, n = thread_position_in_grid.z;
        if (px >= W || py >= H) return;
        const device vec<T, 4>* a4 = (const device vec<T, 4>*)act;
        const device float4* w4 = (const device float4*)w;
        float4 acc = 0;
        for (int i = 0; i < 3; ++i) {
            int iy = py + i - 1;
            if (iy < 0 || iy >= H) continue;
            for (int j = 0; j < 3; ++j) {
                int ix = px + j - 1;
                if (ix < 0 || ix >= W) continue;
                size_t o = ((size_t(n) * H + iy) * W + ix) * 6;
                for (int q = 0; q < 6; ++q) acc += w4[(i * 3 + j) * 6 + q] * float4(a4[o + q]);
            }
        }
        size_t p = (size_t(n) * H + py) * W + px;
        out[p] = smish_of(total[p] + b[0] + acc.x + acc.y + acc.z + acc.w);
    """,
)


def smish(x: mx.array) -> mx.array:
    """Smish of every element, computed in float32."""
    return _SMISH(
        inputs=[mx.contiguous(x)], template=[("T", x.dtype)], grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]  # fmt: skip


class _Conv:
    """A dense convolution with its bias: a GEMM, `conv_mlx.direct` or `mx.conv2d`, whichever is fastest."""

    def __init__(self, conv: Any, dtype: mx.Dtype) -> None:  # noqa: ANN401 - a torch module
        self.w = swin.arr(conv.weight.permute(0, 2, 3, 1), dtype)  # OHWI
        self.b = swin.arr(conv.bias, dtype)
        self.stride: tuple[int, int] = tuple(conv.stride)
        self.padding: tuple[int, int] = tuple(conv.padding)
        co, kh, kw, ci = self.w.shape
        self.wm = self.w.reshape(co, ci).T if kh * kw == 1 else None
        self.direct = conv_ops.direct_supported(ci, co, (kh, kw), self.stride)

    def __call__(self, x: mx.array) -> mx.array:
        co, _, _, ci = self.w.shape
        if self.wm is not None:
            if self.stride != (1, 1):
                x = x[:, :: self.stride[0], :: self.stride[1]]
            if conv1x1_ok(ci, co, x.shape[0] * x.shape[1] * x.shape[2], (1, 1), (1, 1), (0, 0), 1):
                return conv1x1(x, self.wm, self.b)
            return mx.conv2d(x, self.w) + self.b
        if self.direct:
            return conv_ops.direct(x, self.w, self.b, self.stride, self.padding)
        return mx.conv2d(x, self.w, stride=self.stride, padding=self.padding) + self.b


class _Up:
    """`UpConvBlock`: per step a 1x1 convolution, Smish and a stride-2 transposed convolution."""

    def __init__(self, block: Any, dtype: mx.Dtype) -> None:  # noqa: ANN401 - an `UpConvBlock`
        mods = list(block.features)
        self.steps = []
        for conv, deconv in zip(mods[0::3], mods[2::3], strict=True):
            w = swin.arr(deconv.weight, dtype)  # [Cin, Cout, kh, kw]
            wt = mx.transpose(w, (1, 2, 3, 0))  # `mx.conv_transpose2d`'s [Cout, kh, kw, Cin]
            bias = swin.arr(deconv.bias, dtype)
            self.steps.append((_Conv(conv, dtype), w, wt, bias, tuple(deconv.stride), tuple(deconv.padding)))

    def __call__(self, x: mx.array) -> mx.array:
        for conv, w, wt, bias, stride, padding in self.steps:
            x = smish(conv(x))
            y = _conv_transpose2d_phases(x, w, stride, padding, (0, 0))
            x = (mx.conv_transpose2d(x, wt, stride=stride, padding=padding) if y is None else y) + bias
        return x


class MlxTED:
    """`TED.forward` on MLX: NHWC `[B, H, W, 3]` (0..255, sides multiples of 8) -> its four maps `[B, H, W]` float32."""

    def __init__(self, ted: TED, dtype: mx.Dtype) -> None:
        """Convert the weights of the float32 `ted` to `dtype`."""
        self.dtype = dtype
        net: Any = ted
        self.block_1 = (_Conv(net.block_1.conv1, dtype), _Conv(net.block_1.conv2, dtype))
        self.block_2 = (_Conv(net.block_2.conv1, dtype), _Conv(net.block_2.conv2, dtype))
        self.side_1 = _Conv(net.side_1.conv, dtype)
        self.pre_dense_3 = _Conv(net.pre_dense_3.conv, dtype)
        dense = net.dblock_3.denselayer1
        self.dense = (_Conv(dense.conv1, dtype), _Conv(dense.conv2, dtype))
        self.ups = (_Up(net.up_block_1, dtype), _Up(net.up_block_2, dtype), _Up(net.up_block_3, dtype))
        fuse = net.block_cat
        # [tap, out channel] float32; DWconv1's output channel c reads input channel c // 8
        f32 = mx.float32
        self.fuse_w1 = swin.arr(fuse.DWconv1.weight[:, 0].permute(1, 2, 0).reshape(9, 24), f32)
        self.fuse_b1 = swin.arr(fuse.DWconv1.bias, f32)
        self.fuse_w2 = swin.arr(fuse.DWconv2.weight[:, 0].permute(1, 2, 0).reshape(9, 24), f32)
        self.fuse_b2 = swin.derived((1,), f32, lambda: fuse.DWconv2.bias.detach().double().numpy().sum().reshape(1))
        mx.eval(self.fuse_w1, self.fuse_b1, self.fuse_w2, self.fuse_b2)

    def _fuse(self, heads: list[mx.array]) -> mx.array:
        n, h, w, _ = heads[0].shape
        x = smish(mx.concatenate([*heads, mx.zeros_like(heads[0])], axis=-1))
        grid, group = (w, h, n), (16, 16, 1)
        act, total = _FUSE_A(
            inputs=[x, self.fuse_w1, self.fuse_b1], template=[("T", x.dtype)], grid=grid, threadgroup=group,
            output_shapes=[(n, h, w, 24), (n, h, w)], output_dtypes=[x.dtype, mx.float32],
        )  # fmt: skip
        return _FUSE_B(
            inputs=[act, total, self.fuse_w2, self.fuse_b2], template=[("T", x.dtype)], grid=grid, threadgroup=group,
            output_shapes=[(n, h, w)], output_dtypes=[mx.float32],
        )[0]  # fmt: skip

    def __call__(self, x: mx.array) -> list[mx.array]:
        """The four maps of `x`: the three heads and their fusion."""
        x = x.astype(self.dtype)
        block_1 = smish(self.block_1[1](smish(self.block_1[0](x))))
        side = self.side_1(block_1)
        block_2 = self.block_2[1](smish(self.block_2[0](block_1)))
        h, w = block_2.shape[1:3]
        p = mx.pad(block_2, [(0, 0), (1, 1), (1, 1), (0, 0)], constant_values=-float("inf"))
        p = mx.maximum(mx.maximum(p[:, :, 0:w:2], p[:, :, 1 : w + 1 : 2]), p[:, :, 2 : w + 2 : 2])
        down = mx.maximum(mx.maximum(p[:, 0:h:2], p[:, 1 : h + 1 : 2]), p[:, 2 : h + 2 : 2])  # MaxPool2d(3, 2, 1)
        skip = self.pre_dense_3(down)
        block_3 = 0.5 * (self.dense[1](smish(self.dense[0](smish(down + side)))) + skip)
        heads = [up(f) for up, f in zip(self.ups, (block_1, block_2, block_3), strict=True)]
        return [*(v[..., 0].astype(mx.float32) for v in heads), self._fuse(heads)]


def edge_map(outputs: list[mx.array]) -> mx.array:
    """`fastforks.shared.teed.edge_map` of `MlxTED`'s maps (all at the input size): the sigmoid of their mean."""
    return mx.sigmoid(sum(outputs[1:], outputs[0]) / len(outputs))
