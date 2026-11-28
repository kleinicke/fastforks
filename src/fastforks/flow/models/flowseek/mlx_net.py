"""FlowSeek-T on Apple MLX, written in `mlx.core` operations: `MlxFlowSeek(sd, dtype, iters)(image1, image2)`.

The maths of `net.FlowSeek.forward`, channels last, built from the checkpoint's state dict (`shared.mlx_flow`):

- Depth Anything V2 Small on both images resized to 518 x 518: the DINOv2 blocks of `shared.mlx_vit` (LayerScale
  folded, the patch embedding one matmul, the 37 x 37 position table with the cls token and patch bias in it) and the
  DPT head of `shared.depth_anything_v2.mlx_net` (its projections and transposed convs one matmul, each fusion
  block's 1 x 1 conv before its resize); the depth branch (`output_conv*`) runs for image 1 only, the only depth used;
- the fusion feature resized to the image size and the merge head one compiled step per image (the resized map in the
  run dtype: torch resizes in float32 and the conv casts once, the kernel accumulates in float32 and rounds once);
- the motion bases and their padding in float32;
- the ResNet18 encoders with their batch norms folded, each residual block one compiled step;
- no correlation volume: `graph_mlx.ops.corr_windows` from fmap1 and the 2 x 2 average-pooled levels of fmap2 (the
  bilinear halving at half-pixel centres is that pooling, and the pooled volume is the volume of the pooled maps);
- each update one compiled step, one in flight: the flow state and its first conv, the correlation windows and the
  convex upsampling in float32, the flow head's two flow rows only, the mask head's 0.25 folded in.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_flow as mf
from fastforks.shared.cspnext.mlx_net import stepped, steps
from fastforks.shared.depth_anything_v2 import mlx_net as da2
from fastforks.shared.depth_anything_v2 import mlx_sd as sdh
from fastforks.shared.mlx_vit import dinov2_layers

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fastforks.shared.mlx_flow import Conv

__all__ = ["LAYOUT", "SOURCES", "MlxFlowSeek"]

LAYOUT = "flowseek-mlx-1"
SOURCES = (Path(__file__), Path(mf.__file__), Path(da2.__file__), Path(sdh.__file__))

FACTOR, LEVELS, RADIUS, DIM = 8, 4, 4, 128
DA_SIZE, PATCH, HEADS, LAYERS = 518, 14, 6, (3, 6, 9, 12)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_FLOW = (0, 1)  # the flow head's flow rows (the rest is the uncertainty)


class _Encoder:
    """`ResNetFPN`: 7 x 7 stride-2 stem, ResNet18 stages (strides 1, 2, 2), a 1 x 1 conv; batch norms folded."""

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype) -> None:
        """Convert the encoder `p` of `sd`."""
        stem = mf.conv(sd, f"{p}.conv1", dtype, stride=2, bn=f"{p}.bn1")
        self._steps = [stepped(lambda x: mf.relu(stem(x)))]
        for layer in (1, 2, 3):
            i = 0
            while f"{p}.layer{layer}.{i}.conv1.weight" in sd:
                b = f"{p}.layer{layer}.{i}"
                stride = 2 if i == 0 and layer > 1 else 1
                c1 = mf.conv(sd, f"{b}.conv1", dtype, stride=stride, bn=f"{b}.bn1")
                c2 = mf.conv(sd, f"{b}.conv2", dtype, bn=f"{b}.bn2")
                ds = (
                    mf.conv(sd, f"{b}.downsample.0", dtype, stride=stride, padding=0, bn=f"{b}.bn3")
                    if f"{b}.downsample.0.weight" in sd
                    else None
                )
                self._steps.append(stepped(self._block(c1, c2, ds)))
                i += 1
        self.out = mf.conv(sd, f"{p}.final_conv", dtype)

    @staticmethod
    def _block(c1: Conv, c2: Conv, ds: Conv | None) -> Callable[[mx.array], mx.array]:
        def run(x: mx.array) -> mx.array:
            y = mf.relu(c2(mf.relu(c1(x))))
            return mf.relu((ds(x) if ds is not None else x) + y)

        return run

    def __call__(self, x: mx.array) -> mx.array:
        """`[N, H, W, cin]` -> `[N, H / 8, W / 8, 256]` before the final 1 x 1 conv (`out`, the caller's step)."""
        for f in self._steps:
            x = f(x)
        return x


def _unit(x: mx.array) -> mx.array:
    """`x` `[1, H, W, 2]` divided by its norm."""
    return x * mx.rsqrt(mx.sum(mx.square(x)))


def motion_bases(disp: mx.array) -> mx.array:
    """`net.motion_bases` of float32 `[1, H, W, 1]` -> `[1, H, W, 16]` (float32)."""
    _, h, w, _ = disp.shape
    ys = (mx.arange(h, dtype=mx.float32) + 0.5) / h - 0.5
    xs = (mx.arange(w, dtype=mx.float32) + 0.5) / w - 0.5
    u = mx.broadcast_to(xs[None, None, :, None], (1, h, w, 1))
    v = mx.broadcast_to(ys[None, :, None, None], (1, h, w, 1))
    aspect = w / h
    zero, one = mx.zeros_like(disp), mx.ones_like(disp)

    def cat(a: mx.array, b: mx.array) -> mx.array:
        return mx.concatenate([a, b], axis=-1)

    trans = [cat(-one, zero), cat(zero, -one), cat(u, v)]
    rot = [cat(zero, one), cat(u * v, v * v), cat(-one, zero), cat(-u * u, -u * v), cat(-v / aspect, u * aspect)]
    return mx.concatenate([2 * disp * _unit(t) for t in trans] + [_unit(r) for r in rot], axis=-1)


class MlxFlowSeek:
    """FlowSeek-T from the state dict `sd` in `dtype` (the name), `iters` updates; one pair per call."""

    def __init__(self, sd: Mapping[str, Any], dtype: str, iters: int) -> None:
        """Convert the weights (through the conversion cache, `mlx_port.recorded`)."""
        dt = self.dtype = mf.DTYPES[dtype]
        self.iters = iters
        self.vit = sdh.vit(sd, "dav2.pretrained", dt)
        self.p = sdh.head(sd, "dav2.depth_head", dt)
        self.merge = [mf.conv(sd, f"merge_head.{i}", dt, stride=2) for i in (0, 2, 4)]
        self.cnet = _Encoder(sd, "cnet", dt)
        self.bnet = _Encoder(sd, "bnet", dt)
        self.fnet = _Encoder(sd, "fnet", dt)
        self.init_conv = mf.conv(sd, "init_conv", dt)
        self.head1 = mf.conv(sd, "flow_head.0", dt)
        self.head2 = mf.conv(sd, "flow_head.2", dt, rows=_FLOW)
        e = "update_block.encoder"
        self.convc1 = mf.conv(sd, f"{e}.convc1", dt)
        self.convc2 = mf.conv(sd, f"{e}.convc2", dt)
        self.convf1 = mf.conv(sd, f"{e}.convf1", mx.float32)
        self.convf2 = mf.conv(sd, f"{e}.convf2", dt)
        self.conv = mf.conv(sd, f"{e}.conv", dt)
        self.refine = mf.blocks(sd, "update_block.refine", dt)
        self.up1 = mf.conv(sd, "upsample_weight.0", dt)
        self.up2 = mf.conv(sd, "upsample_weight.2", dt, rows=mf.convex_rows(FACTOR), scale=0.25)
        self.mean = mx.array(MEAN, dtype=mx.float32)
        self.std = mx.array(STD, dtype=mx.float32)
        self._mono = stepped(self._mono_fn)
        self._merge = stepped(self._merge_fn)
        self._context = stepped(self._context_fn)
        self._features = stepped(lambda f, mono: mx.concatenate([self.fnet.out(f), mono], axis=-1))
        self._step = stepped(self._update)
        self._mask = stepped(lambda net: self.up2(mf.relu(self.up1(net))))

    # Depth Anything ---------------------------------------------------------------------------------------------------

    def _mono_fn(self, both: mx.array) -> tuple[mx.array, mx.array]:
        """`FlowSeek._mono` of `[2, H, W, 3]` in [0, 1]: the fusion feature `[2, 296, 296, 64]`, image 1's depth."""
        p = self.p
        small = sdh.bilinear(both, DA_SIZE, DA_SIZE, align=False)
        x = (small / self.mean - self.std).transpose(0, 3, 1, 2)
        h0 = w0 = DA_SIZE // PATCH
        feats = [t[:, 1:] for t in dinov2_layers(self.vit, HEADS, x, self.vit["table"], LAYERS)]
        path = sdh.fusion(p, feats, h0, w0)
        d = sdh.bilinear(da2._conv(p, "head.out1", path[:1]), h0 * PATCH, w0 * PATCH, align=True)  # noqa: SLF001
        d = mf.relu(da2._conv(p, "head.out2", d)).astype(mx.float32)  # noqa: SLF001
        d = mf.relu(d @ p["head.out3.weight"] + p["head.out3.bias"])
        return path, d

    def _merge_fn(self, path: mx.array, h: int, w: int) -> mx.array:
        """The merge head on one image's fusion feature resized to `(h, w)`: `[1, h / 8, w / 8, 128]`."""
        x = sdh.bilinear(path, h, w, align=False)
        m0, m1, m2 = self.merge
        return m2(mf.relu(m1(mf.relu(m0(x)))))

    # the flow network -------------------------------------------------------------------------------------------------

    def _context_fn(self, c: mx.array, b: mx.array) -> list[mx.array]:
        """Hidden state, context and the initial flow from the context and motion-basis encoders."""
        c = self.init_conv(self.cnet.out(c))
        b = self.init_conv(self.bnet.out(b))
        net = mx.concatenate([c[..., :DIM], b[..., :DIM]], axis=-1)
        context = mx.concatenate([c[..., DIM:], b[..., DIM:]], axis=-1)
        flow = self.head2(mf.relu(self.head1(net))).astype(mx.float32)
        return [net, context, flow]

    def _update(
        self, net: mx.array, flow: mx.array, context: mx.array, f1: mx.array, *levels: mx.array
    ) -> tuple[mx.array, mx.array]:
        _, h, w, _ = net.shape
        dt = self.dtype
        coords = (mf.grid(h, w) + flow).transpose(0, 3, 1, 2)
        corr = ops.corr_windows(f1, list(levels), coords, RADIUS, 1 / math.sqrt(f1.shape[-1]))
        corr = corr[0].reshape(-1, h * w).T.astype(dt)  # [h w, L (2r + 1)^2]
        cor = mf.relu(mx.addmm(self.convc1.b, corr, self.convc1.w.T)).reshape(1, h, w, -1)
        cor = mf.relu(self.convc2(cor))
        flo = mf.relu(self.convf2(mf.relu(self.convf1(flow)).astype(dt)))
        motion = mf.relu(self.conv(mx.concatenate([cor, flo], axis=-1)))
        motion = mx.concatenate([motion, flow.astype(dt)], axis=-1)
        net = mf.refine(self.refine, net, [context, motion])
        return net, flow + self.head2(mf.relu(self.head1(net))).astype(mx.float32)

    def _run(self, image1: mx.array, image2: mx.array) -> mx.array:
        dt = self.dtype
        _, _, h, w = image1.shape
        both = mx.concatenate([image1, image2], axis=0).transpose(0, 2, 3, 1)  # float32 [2, H, W, 3]
        path, depth = self._mono(both)
        mono = mx.concatenate([self._merge(path[i : i + 1], h, w) for i in (0, 1)], axis=0)
        bases = motion_bases(sdh.bilinear(depth, h, w, align=False))
        ph, pw = -h % FACTOR, -w % FACTOR
        pads = ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0))
        both = mx.pad(2 * both - 1, pads, mode="edge").astype(dt)
        bases = mx.pad(bases, pads, mode="edge").astype(dt)
        net, context, flow = self._context(self.cnet(mx.concatenate([both[:1], both[1:]], axis=-1)), self.bnet(bases))
        f = self._features(self.fnet(both), mono)
        f1 = f[:1]
        levels = mf.pyramid(f[1:], LEVELS)
        for _ in range(self.iters):
            net, flow = self._step(net, flow, context, f1, *levels)
        up = mf.convex_upsample(flow, self._mask(net), FACTOR)
        up = up[:, :, ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
        mx.eval(up)
        return up

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[1, 2, H, W]` (float32) of an unpadded float32 pair `[1, 3, H, W]` in [0, 1]."""
        mf.options(half_wino_f32=self.dtype == mx.float16)
        return steps(self._run, image1, image2)
