"""MEMFOF on Apple MLX, written in `mlx.core` operations: `MlxMEMFOF(sd, dtype, iters)(image1, image2)` -> flow.

The maths of `net.MEMFOF.forward_pair` (frames `(image1, image1, image2)`, the forward flow), channels last, built
from the checkpoint's state dict (`shared.mlx_flow`):

- the ResNet34 encoders with their batch norms folded, each conv one compiled step evaluated at once (`mlx_flow.eager`:
  pipelined blocks held two blocks' Winograd transforms, 1080 x 1920 float16 3.2 -> 1.2 GB), their stem, layer1 and
  layer2 in row bands above `BAND_BYTES` (the 2x inputs of kitti and sintel); the context encoder's first conv takes
  `(image1, image2)`: its weights for the duplicated `image1` summed; the feature encoder runs the frames one after
  the other (a batch of two doubles the transforms); the final 2 x 2 stride-2 conv a GEMM over the 2 x 2 patches;
- no correlation volume: `graph_mlx.ops.corr_windows` takes each window's dot products from fmap1 and the 2 x 2
  average-pooled levels of fmap1 (backward) and fmap3 (forward);
- the GMA attention matrix computed once in the run dtype while its float32 size is at most `MATRIX_BYTES` (else
  recomputed in each update in query blocks of `SDPA_BLOCK_BYTES` scores, each block one step); the aggregation's
  `gamma` folded into `to_v`;
- each update in six compiled pieces (the correlation windows and their convs, the rest of the motion encoder, the
  aggregation's readout, each of the two ConvNeXt blocks, the flow head reduced to the 4 flow outputs it reads), one
  piece in flight in float16 and each evaluated at once in float32; the ConvNeXt `gamma` folded into `pwconv2`;
- the upsampling head only for the forward flow's 9 x 16 x 16 mask logits, its 0.25 folded in;
- float32: the flow state, the flow encoder's first conv, the correlation windows, the convex upsampling; in float16
  the convs MLX would run as half-precision Winograd run in float32 (`mlx_flow.options`), as in the graph translation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import SDPA_BLOCK_BYTES, attention, mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_flow as mf
from fastforks.shared.cspnext.mlx_net import stepped, steps

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.shared.mlx_flow import Conv

__all__ = ["LAYOUT", "SOURCES", "MlxMEMFOF"]

LAYOUT = "memfof-mlx-1"
SOURCES = (Path(__file__), Path(mf.__file__))

LEVELS, RADIUS = 4, 4
FACTOR = 16
MATRIX_BYTES = 1 << 30  # float32 attention matrix from which it is recomputed in each update (memory bound)
_FLOW = (0, 1, 6, 7)  # flow-head outputs read: the backward and the forward flow
BAND_BYTES = 128 << 20  # float32 size of the 1/4-resolution map from which the encoders' first layers run in bands
_HALO = 12  # 1/4-resolution rows a band reads beyond its own (11 come out wrong at a cut: the stem, 14 3 x 3 convs)
_BANDED = 7  # the residual blocks run in bands: layer1 (64 channels at 1/2 resolution) and layer2 (128 at 1/4)


class _Encoder:
    """`ResNetFPN16x` (ResNet34 without max pool and layer4, batch norms folded, then the 2 x 2 stride-2 conv)."""

    def __init__(self, sd: Mapping[str, Any], prefix: str, dtype: mx.Dtype, *, merge: bool = False) -> None:
        """Convert the encoder `prefix` of `sd`; `merge`: its first conv takes `(image1, image2)` for 3 frames."""
        r = f"{prefix}.resnet"

        def merged(w: np.ndarray) -> np.ndarray:
            return np.concatenate([w[:, 0:3] + w[:, 3:6], w[:, 6:9]], axis=1)

        if merge:
            self.stem = mf.conv(sd, f"{r}.conv1", dtype, stride=2, bn=f"{r}.bn1", inputs=merged, cin=6)
        else:
            self.stem = mf.conv(sd, f"{r}.conv1", dtype, stride=2, bn=f"{r}.bn1")
        self.blocks: list[tuple[Conv, Conv, Conv | None]] = []
        for layer, n in (("layer1", 3), ("layer2", 4), ("layer3", 6)):
            for i in range(n):
                p = f"{r}.{layer}.{i}"
                stride = 2 if i == 0 and layer != "layer1" else 1
                c1 = mf.conv(sd, f"{p}.conv1", dtype, stride=stride, bn=f"{p}.bn1")
                c2 = mf.conv(sd, f"{p}.conv2", dtype, bn=f"{p}.bn2")
                ds = (
                    mf.conv(sd, f"{p}.downsample.0", dtype, stride=stride, padding=0, bn=f"{p}.downsample.1")
                    if f"{p}.downsample.0.weight" in sd
                    else None
                )
                self.blocks.append((c1, c2, ds))
        self.final = mf.conv(sd, f"{prefix}.final_conv", dtype, stride=2, padding=0)
        self._stem = mf.eager(lambda x: mf.relu(self.stem(x)))
        self._steps = [(mf.eager(lambda x, c1=b[0]: mf.relu(c1(x))), mf.eager(self._second(*b))) for b in self.blocks]
        self._final = mf.eager(self.final)

    @staticmethod
    def _second(_: Conv, c2: Conv, ds: Conv | None) -> Any:  # noqa: ANN401 - a layer
        def run(y: mx.array, x: mx.array) -> mx.array:
            return mf.relu(c2(y) + (ds(x) if ds is not None else x))

        return run

    def _quarter(self, x: mx.array) -> mx.array:
        """The stem, layer1 and layer2: `[1, H, W, C]` -> `[1, H / 4, W / 4, 128]`."""
        x = self._stem(x)
        for first, second in self._steps[:_BANDED]:
            x = second(first(x), x)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        """`[1, H, W, C]` -> the 1/16 features `[1, H / 16, W / 16, 1024]`, one conv per evaluation.

        Above `BAND_BYTES` the stem, layer1 and layer2 run in row bands of the 1/4-resolution map (each with `_HALO`
        rows more on either side, cut off): a 3 x 3 conv's float32 Winograd transforms are 4.5 times its float32
        input, at 2176 x 3840 +1.9 GB in each layer2 conv.
        """
        _, h, w, _ = x.shape
        oh, ow = h // 4, w // 4  # the inputs are padded to multiples of 16
        bands = -(-oh * ow * 128 * 4 // BAND_BYTES)
        if bands <= 1:
            x = self._quarter(x)
        else:
            rows = -(-oh // bands)
            parts = []
            for r0 in range(0, oh, rows):
                r1 = min(oh, r0 + rows)
                s, e = max(0, r0 - _HALO), min(oh, r1 + _HALO)
                parts.append(self._quarter(x[:, 4 * s : 4 * e])[:, r0 - s : r1 - s])
            x = mx.concatenate(parts, axis=1)
            del parts
            mx.eval(x)
        for first, second in self._steps[_BANDED:]:
            x = second(first(x), x)
        return self._final(x)


class MlxMEMFOF:
    """MEMFOF from the state dict `sd` in `dtype` (the name), `iters` updates; one pair per call."""

    def __init__(self, sd: Mapping[str, Any], dtype: str, iters: int) -> None:
        """Convert the weights (through the conversion cache, `mlx_port.recorded`)."""
        dt = self.dtype = mf.DTYPES[dtype]
        self.iters = iters
        dim = self.dim = int(sd["init_conv.weight"].shape[0]) // 2
        self.cnet = _Encoder(sd, "cnet", dt, merge=True)
        self.fnet = _Encoder(sd, "fnet", dt)
        self.init_conv = mf.conv(sd, "init_conv", dt)
        self.head1 = mf.conv(sd, "flow_head.0", dt)
        self.head2 = mf.conv(sd, "flow_head.2", dt, rows=_FLOW)
        self.qk = mf.conv(sd, "att.to_qk", dt)
        e = "update_block.encoder"
        self.convc1 = mf.conv(sd, f"{e}.convc1", dt)
        self.convc2 = mf.conv(sd, f"{e}.convc2", dt)
        self.convf1 = mf.conv(sd, f"{e}.convf1", mx.float32)
        self.convf2 = mf.conv(sd, f"{e}.convf2", dt)
        self.conv = mf.conv(sd, f"{e}.conv", dt)
        a = "update_block.aggregator"
        self.to_v = mf.put(
            (dim, dim), dt, lambda: mf.np64(sd[f"{a}.to_v.weight"])[:, :, 0, 0] * mf.np64(sd[f"{a}.gamma"])
        )
        self.refine = mf.blocks(sd, "update_block.refine", dt)
        self.up1 = mf.conv(sd, "upsample_weight.0", dt)
        mask = 9 * FACTOR * FACTOR
        self.up2 = mf.conv(sd, "upsample_weight.2", dt, rows=mf.convex_rows(FACTOR, mask), scale=0.25)
        # float16: one piece in flight (faster at a peak below the graph's); float32: each piece evaluated at once
        # (two float32 pieces in flight cost 1x +260 MB, 2x +850 MB for the same speed)
        piece = stepped if dt == mx.float16 else mf.eager
        self._corr_step = piece(self._corr)
        self._motion_step = piece(self._motion)
        self._read = piece(lambda matrix, v: matrix @ v)
        self._sdpa = piece(lambda q, k, v: attention(q, k, v, scale=1.0)[0, 0])
        self._refine = [piece(self._block(blk)) for blk in self.refine]
        self._head = piece(self._flow_head)

    # the update ---------------------------------------------------------------------------------------------------

    def _corr(self, flow: mx.array, f1: mx.array, *levels: mx.array) -> mx.array:
        """The correlation features `[1, h, w, 768]` (the windows, `convc1`, `convc2`)."""
        _, h, w, _ = flow.shape
        dt = self.dtype
        coords = mf.grid(h, w)
        scale = 1 / math.sqrt(f1.shape[-1])
        corr = [
            ops.corr_windows(f1, list(lv), (coords + flow[..., s : s + 2]).transpose(0, 3, 1, 2), RADIUS, scale)
            for s, lv in ((0, levels[:LEVELS]), (2, levels[LEVELS:]))
        ]
        corr = mx.concatenate(corr, axis=1)[0].reshape(-1, h * w).T.astype(dt)  # [h w, 2 L (2r + 1)^2]
        cor = mf.relu(mx.addmm(self.convc1.b, corr, self.convc1.w.T)).reshape(1, h, w, -1)
        return mf.relu(self.convc2(cor))

    def _motion(self, cor: mx.array, flow: mx.array) -> tuple[mx.array, mx.array]:
        """The motion features `[1, h, w, D]` and the aggregation's values `[h w, D]`."""
        _, h, w, _ = flow.shape
        dt = self.dtype
        flo = mf.relu(self.convf1(flow)).astype(dt)
        flo = mf.relu(self.convf2(flo))
        motion = mf.relu(self.conv(mx.concatenate([cor, flo], axis=-1)))
        motion = mx.concatenate([motion, flow.astype(dt)], axis=-1)
        return motion, motion.reshape(h * w, self.dim) @ self.to_v.T

    @staticmethod
    def _block(block: mf.ConvNext) -> Any:  # noqa: ANN401 - a layer
        def run(net: mx.array, context: mx.array, motion: mx.array, readout: mx.array) -> mx.array:
            glob = motion + readout.reshape(motion.shape)
            return block(mx.concatenate([net, context, motion, glob], axis=-1))

        return run

    def _flow_head(self, net: mx.array, flow: mx.array) -> mx.array:
        return flow + self.head2(mf.relu(self.head1(net))).astype(mx.float32)

    def _update(
        self, net: mx.array, flow: mx.array, context: mx.array, att: list[mx.array], f1: mx.array, *levels: mx.array
    ) -> tuple[mx.array, mx.array]:
        """One update, each piece one evaluation.

        The pieces: the correlation features, the motion features, the aggregation (the matrix product, or each query
        block's attention), each ConvNeXt block, the flow head. One compiled update held every piece's transients
        until it completed: memfof-kitti float32 +2.4 GB in the motion encoder's two wide 3 x 3 convs alone.
        """
        motion, v = self._motion_step(self._corr_step(flow, f1, *levels), flow)
        if len(att) == 1:
            readout = self._read(att[0], v)
        else:
            readout = mx.concatenate([self._sdpa(q, att[-1], v[None, None]) for q in att[:-1]], axis=0)
        for block in self._refine:
            net = block(net, context, motion, readout)
        return net, self._head(net, flow)

    # the network --------------------------------------------------------------------------------------------------

    def _attention(self, context: mx.array) -> list[mx.array]:
        """Upstream's softmax matrix `[hw, hw]` in the run dtype, or the scaled query blocks and the keys.

        The query blocks are `[1, 1, b, D]` (each block's float32 scores at most `SDPA_BLOCK_BYTES`), the keys
        `[1, 1, hw, D]`.
        """
        _, h, w, _ = context.shape
        n = h * w
        d = self.dim
        qk = self.qk(context).reshape(n, 2 * d)
        q = qk[:, :d] * (d**-0.5 * math.log(n, 3))
        k = qk[:, d:]
        if n * n * 4 <= MATRIX_BYTES:
            return [mx.softmax(q @ k.T, axis=-1, precise=True)]
        blk = max(1, SDPA_BLOCK_BYTES // (4 * n))
        return [*(q[None, None, i : i + blk] for i in range(0, n, blk)), k[None, None]]

    def _run(self, image1: mx.array, image2: mx.array) -> mx.array:
        dt = self.dtype

        def nhwc(x: mx.array) -> mx.array:  # made where it is used: no full-resolution copy outlives its encoder
            return x.transpose(0, 2, 3, 1).astype(dt)

        cmap = self.init_conv(self.cnet(mx.concatenate([nhwc(image1), nhwc(image2)], axis=-1)))
        net, context = cmap[..., : self.dim], cmap[..., self.dim :]
        flow = self.head2(mf.relu(self.head1(net))).astype(mx.float32)
        att = self._attention(context)
        mx.eval(net, context, flow, att)
        del cmap
        f1 = self.fnet(nhwc(image1))
        levels = [*mf.pyramid(f1, LEVELS), *mf.pyramid(self.fnet(nhwc(image2)), LEVELS)]
        mx.eval(levels)
        for _ in range(self.iters):
            net, flow = self._update(net, flow, context, att, f1, *levels)
        mask = self.up2(mf.relu(self.up1(net)))
        up = mf.convex_upsample(flow[..., 2:4], mask, FACTOR)
        mx.eval(up)
        return up

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[1, 2, H, W]` (float32) of a padded float32 pair `[1, 3, H, W]` in [-1, 1]."""
        mf.options(half_wino_f32=self.dtype == mx.float16)
        return steps(self._run, image1, image2)
