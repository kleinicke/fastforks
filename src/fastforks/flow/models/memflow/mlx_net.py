"""MemFlowNet on Apple MLX, written in `mlx.core` operations: `MlxMemFlow(sd, dtype, iters, tal)(image1, image2)`.

The maths of `net.MemFlowNet.forward_pair`, channels last, built from the checkpoint's state dict (`shared.mlx_flow`):

- the `BasicEncoder`s: the context one's batch norms folded, the feature one's instance norms with float32 statistics
  (both frames as one batch); each residual block one compiled step;
- no correlation volume: `graph_mlx.ops.corr_windows` takes each window's dot products from fmap1 and the 2 x 2
  average-pooled levels of fmap2 (the pooled volume is the volume of the pooled features);
- the readout's softmax matrix computed once in the run dtype while its float32 size is at most `MATRIX_BYTES` (else
  fused attention in each update, `_core.mlx.attention`); the aggregator's `gamma` folded into `to_v`;
- the `PCBlock`s: the 1 x 1 depthwise convs a per-channel affine (`x + conv(x)` = `x (1 + w) + b`), the 15 x 15 and
  7 x 7 ones `graph_mlx.ops.depthwise`;
- each update one compiled step, one in flight; the flow state, the flow encoder's first conv, the correlation
  windows and the convex upsampling in float32; the mask head's 0.25 folded in.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import attention, mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_flow as mf
from fastforks.shared.cspnext.mlx_net import stepped, steps

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fastforks.shared.mlx_flow import Conv

__all__ = ["LAYOUT", "SOURCES", "MlxMemFlow"]

LAYOUT = "memflow-mlx-1"
SOURCES = (Path(__file__), Path(mf.__file__))

FACTOR, DIM, LEVELS, RADIUS = 8, 128, 4, 4
MATRIX_BYTES = 1 << 30  # float32 readout matrix from which fused attention recomputes it in each update


def _instance(x: mx.array) -> mx.array:
    """`InstanceNorm2d` (no affine, eps 1e-5) of `[N, H, W, C]`, float32 statistics."""
    y = x.astype(mx.float32)
    mean = y.mean(axis=(1, 2), keepdims=True)
    var = mx.square(y - mean).mean(axis=(1, 2), keepdims=True)
    return ((y - mean) * mx.rsqrt(var + 1e-5)).astype(x.dtype)


class _Encoder:
    """`BasicEncoder`: 7 x 7 stride-2 stem, residual stages of 64 / 96 / 128 channels, a 1 x 1 conv to 256."""

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype, *, batch: bool) -> None:
        """Convert the encoder `p` of `sd`: batch norms folded (`batch`) or instance norms."""

        def conv(name: str, norm: str, **kw: Any) -> Callable[[mx.array], mx.array]:  # noqa: ANN401 - conv arguments
            if batch:
                return mf.conv(sd, name, dtype, bn=norm, **kw)
            c = mf.conv(sd, name, dtype, **kw)
            return lambda x: _instance(c(x))

        stem = conv(f"{p}.conv1", f"{p}.norm1", stride=2)
        self._steps = [stepped(lambda x: mf.relu(stem(x)))]
        for layer in (1, 2, 3):
            for i in (0, 1):
                b = f"{p}.layer{layer}.{i}"
                stride = 2 if i == 0 and layer > 1 else 1
                c1 = conv(f"{b}.conv1", f"{b}.norm1", stride=stride)
                c2 = conv(f"{b}.conv2", f"{b}.norm2")
                # `norm3` is `downsample.1`: the checkpoint's two copies disagree in their running statistics, and
                # `load_state_dict` keeps the later key's, `downsample.1`
                ds = conv(f"{b}.downsample.0", f"{b}.downsample.1", stride=stride, padding=0) if stride > 1 else None
                self._steps.append(stepped(self._block(c1, c2, ds)))
        self.out = mf.conv(sd, f"{p}.conv2", dtype)

    @staticmethod
    def _block(c1: Callable, c2: Callable, ds: Callable | None) -> Callable[[mx.array], mx.array]:
        def run(x: mx.array) -> mx.array:
            y = mf.relu(c2(mf.relu(c1(x))))
            return mf.relu((ds(x) if ds is not None else x) + y)

        return run

    def __call__(self, x: mx.array) -> mx.array:
        """`[N, H, W, 3]` -> `[N, H / 8, W / 8, 256]`; the last 1 x 1 conv left to the caller's step."""
        for f in self._steps:
            x = f(x)
        return x


class _PCBlock:
    """`PCBlock`: a residual 1.5x MLP, residual depthwise convs, a residual 1 x 1, an MLP (GELU throughout)."""

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype) -> None:
        self.f1a, self.f1b = mf.conv(sd, f"{p}.ffn1.0", dtype), mf.conv(sd, f"{p}.ffn1.2", dtype)
        self.dws: list[Any] = []
        i = 0
        while f"{p}.conv_list.{i}.weight" in sd:
            name = f"{p}.conv_list.{i}"
            if int(sd[f"{name}.weight"].shape[-1]) == 1:
                c = int(sd[f"{name}.weight"].shape[0])
                self.dws.append(
                    (
                        mf.put((c,), dtype, lambda name=name: 1 + mf.np64(sd[f"{name}.weight"]).reshape(-1)),
                        mf.put((c,), dtype, lambda name=name: mf.np64(sd[f"{name}.bias"])),
                    )
                )
            else:
                self.dws.append(mf.depthwise(sd, name, dtype))
            i += 1
        self.pw = mf.conv(sd, f"{p}.pw", dtype)
        self.f2a, self.f2b = mf.conv(sd, f"{p}.ffn2.0", dtype), mf.conv(sd, f"{p}.ffn2.2", dtype)

    def __call__(self, x: mx.array) -> mx.array:
        """`[1, h, w, cin]` -> `[1, h, w, cout]`."""
        x = mf.gelu(x + self.f1b(mf.gelu(self.f1a(x))))
        for dw in self.dws:
            x = mf.gelu(x * dw[0] + dw[1]) if isinstance(dw, tuple) else mf.gelu(x + dw(x))
        return self.f2b(mf.gelu(self.f2a(mf.gelu(x + self.pw(x)))))


class MlxMemFlow:
    """MemFlowNet from the state dict `sd` in `dtype` (the name), `iters` updates, the attention scale's `tal`."""

    def __init__(self, sd: Mapping[str, Any], dtype: str, iters: int, train_avg_length: float) -> None:
        """Convert the weights (through the conversion cache, `mlx_port.recorded`)."""
        dt = self.dtype = mf.DTYPES[dtype]
        self.iters, self.tal = iters, train_avg_length
        self.cnet = _Encoder(sd, "cnet", dt, batch=True)
        self.fnet = _Encoder(sd, "fnet", dt, batch=False)
        self.qk = mf.conv(sd, "att.to_qk", dt)
        e = "update_block.encoder"
        self.convc1 = _PCBlock(sd, f"{e}.convc1", dt)
        self.convc2 = _PCBlock(sd, f"{e}.convc2", dt)
        self.convf1: Conv = mf.conv(sd, f"{e}.convf1", mx.float32)
        self.convf2 = _PCBlock(sd, f"{e}.convf2", dt)
        self.conv = _PCBlock(sd, f"{e}.conv", dt)
        a = "update_block.aggregator"
        self.to_v = mf.put(
            (DIM, DIM), dt, lambda: mf.np64(sd[f"{a}.to_v.weight"])[:, :, 0, 0] * mf.np64(sd[f"{a}.gamma"])
        )
        self.gru = _PCBlock(sd, "update_block.gru", dt)
        self.head = _PCBlock(sd, "update_block.flow_head", dt)
        self.m1 = mf.conv(sd, "update_block.mask.0", dt)
        self.m2 = mf.conv(sd, "update_block.mask.2", dt, rows=mf.convex_rows(FACTOR), scale=0.25)
        self._context = stepped(self._context_fn)
        self._features = stepped(self.fnet.out)
        self._step = stepped(self._update)
        self._mask = stepped(lambda net: self.m2(mf.relu(self.m1(net))))

    def _context_fn(self, x: mx.array) -> list[mx.array]:
        """Hidden state, context and the readout's matrix (or its scaled query and key)."""
        c = self.cnet.out(x)
        _, h, w, _ = c.shape
        net, inp = mx.tanh(c[..., :DIM]), mf.relu(c[..., DIM:])
        n = h * w
        qk = self.qk(inp).reshape(n, 2 * DIM)
        q = qk[:, :DIM] * (DIM**-0.5 * math.log(n, self.tal))
        k = qk[:, DIM:]
        if n * n * 4 <= MATRIX_BYTES:
            return [net, inp, mx.softmax(q @ k.T, axis=-1, precise=True)]
        return [net, inp, q[None, None], k[None, None]]

    def _update(
        self, net: mx.array, flow: mx.array, inp: mx.array, att: list[mx.array], f1: mx.array, *levels: mx.array
    ) -> tuple[mx.array, mx.array]:
        _, h, w, _ = net.shape
        dt = self.dtype
        coords = (mf.grid(h, w) + flow).transpose(0, 3, 1, 2)
        corr = ops.corr_windows(f1, list(levels), coords, RADIUS, 1 / math.sqrt(f1.shape[-1]))
        corr = corr.transpose(0, 2, 3, 1).astype(dt)
        cor = self.convc2(mf.gelu(self.convc1(corr)))
        flo = self.convf2(self.convf1(flow).astype(dt))
        motion = mx.concatenate([self.conv(mx.concatenate([cor, flo], axis=-1)), flow.astype(dt)], axis=-1)
        v = motion.reshape(h * w, DIM) @ self.to_v.T
        readout = att[0] @ v if len(att) == 1 else attention(att[0], att[1], v[None, None], scale=1.0)[0, 0]
        glob = motion + readout.reshape(motion.shape)
        net = self.gru(mx.concatenate([net, inp, motion, glob], axis=-1))
        return net, flow + self.head(net).astype(mx.float32)

    def _run(self, image1: mx.array, image2: mx.array) -> mx.array:
        dt = self.dtype
        x1 = image1.transpose(0, 2, 3, 1).astype(dt)
        x2 = image2.transpose(0, 2, 3, 1).astype(dt)
        net, inp, *att = self._context(self.cnet(x1))
        f = self._features(self.fnet(mx.concatenate([x1, x2], axis=0)))
        f1 = f[:1]
        levels = mf.pyramid(f[1:], LEVELS)
        _, h, w, _ = net.shape
        flow = mx.zeros((1, h, w, 2), dtype=mx.float32)
        for _ in range(self.iters):
            net, flow = self._step(net, flow, inp, att, f1, *levels)
        up = mf.convex_upsample(flow, self._mask(net), FACTOR)
        mx.eval(up)
        return up

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[1, 2, H, W]` (float32) of a padded float32 pair `[1, 3, H, W]` in [-1, 1]."""
        mf.options(half_wino_f32=self.dtype == mx.float16)
        return steps(self._run, image1, image2)
