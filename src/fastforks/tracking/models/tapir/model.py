"""TAPIR / BootsTAPIR runners: weights -> network and the offline forward pass.

TAPIR sees the whole video at once. Frames are resized to 256 x 256 (bilinear, antialiased; the TAP-Vid protocol
resizes before, so the eval feeds 256 x 256 frames), scaled to `[-1, 1]` and encoded `FRAME_CHUNK` at a time; the
queries run `QUERY_CHUNK` at a time (upstream: 64, in a random order; each query is computed on its own). A query
`(t, x, y)` in input pixels becomes the point `(t, y, x)` in the continuous 256 x 256 frame; the tracks come back the
same way. The visibility is upstream's `(1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist))` (visible where it is
> 0.5).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks.tracking._runner import Forward, NoParams, Runner

from .net.tapir import SIZE, TAPIR, PIPsConvBlock

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.tracking._runner import Context

FRAME_CHUNK = 16  # fixed: memory chunking of the feature pass, not an architecture value
QUERY_CHUNK = 64  # fixed: as upstream offline: half the peak memory of 256 at the same speed


def build(ctx: Context) -> TAPIR:
    """The variant's network: BootsTAPIR (pyramid level 1, extra convs) or TAPIR (level 0, none), from the weights."""
    sd = ctx.state_dict(strip="")
    extra = any(k.startswith("extra_convs.") for k in sd)
    net = TAPIR(pyramid_level=1 if extra else 0, extra_convs=extra)
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    return net


def track(  # noqa: PLR0913, PLR0917 - the runner call's arguments plus the autocast dtype
    net: TAPIR,
    frames: Callable[[int], torch.Tensor],
    n: int,
    size: tuple[int, int],
    queries: torch.Tensor,
    amp: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """`{"tracks": [T, Q, 2], "visibility": [T, Q]}` of frames `frames(t)` `[1, 3, H, W]` in `[0, 1]`."""
    h, w = size
    scale = torch.tensor((SIZE / w, SIZE / h), device=queries.device)  # (x, y)
    region = torch.autocast(queries.device.type, amp) if amp is not None else contextlib.nullcontext()
    hires, lowres = [], []
    with region:
        for s in range(0, n, FRAME_CHUNK):
            video = torch.cat([frames(t) for t in range(s, min(n, s + FRAME_CHUNK))])
            if video.shape[-2:] != (SIZE, SIZE):
                video = F.interpolate(video, (SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
            a, b = net.frame_features(video * 2 - 1)
            hires.append(a)
            lowres.append(b)
        q = torch.cat([queries[:, :1], ((queries[:, 1:] + 0.5) * scale).flip(-1)], 1)[None]  # (t, y, x)
        f = net.features(torch.cat(hires), torch.cat(lowres), q)
        points, vis = [], []
        for s in range(0, q.shape[1], QUERY_CHUNK):
            c = slice(s, s + QUERY_CHUNK)
            fc = f._replace(query_hires=f.query_hires[:, c], query_lowres=f.query_lowres[:, c])
            p, occ, expd = net.track(fc, q[:, c])
            points.append(p[0])
            vis.append(((1 - occ.float().sigmoid()) * (1 - expd.float().sigmoid()))[0])
    tracks = torch.cat(points).float().transpose(0, 1) / scale - 0.5
    return {"tracks": tracks, "visibility": torch.cat(vis).T}


class Portable(Runner):
    """Plain PyTorch."""

    dtypes = ("float32", "float16")
    Params = NoParams

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = build(ctx).eval().requires_grad_(requires_grad=False).to(ctx.device)
        self.amp = torch.float16 if ctx.dtype == "float16" else None

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        return track(self.net, frames, n, size, queries, self.amp)

    def module(self) -> torch.nn.Module:
        """The network with the resizing; `forward(video [T, 3, H, W], queries [Q, 3])`."""
        amp = self.amp

        def fn(net: TAPIR, video: torch.Tensor, queries: torch.Tensor) -> dict[str, torch.Tensor]:
            size = (video.shape[-2], video.shape[-1])
            return track(net, lambda t: video[t : t + 1], video.shape[0], size, queries, amp)

        return Forward(self.net, fn)


GELU_TANH = 6  # fixed: fused-epilogue activation code (fastforks._optimize.conv_ops)


def _w4(w: torch.Tensor) -> torch.Tensor:
    """A Linear weight `[out, in]` as the fused GEMM's fp16 `[out, in, 1, 1]` channels-last filter."""
    return w.detach().half().reshape(*w.shape, 1, 1).contiguous(memory_format=torch.channels_last)


def _rsc(w: torch.Tensor) -> torch.Tensor:
    """A depthwise Conv1d filter `[C, 1, R]` as the depthwise kernel's fp16 `[R, 1, C]`."""
    return w.detach()[:, 0].T.half().contiguous()[:, None]


def _block(m: PIPsConvBlock) -> Callable[[torch.Tensor], torch.Tensor]:
    """`PIPsConvBlock.forward` in fp16 on the depthwise and fused-GEMM kernels.

    Both temporal convs run over `[N, 4C, T, 1]` (the input channels repeated 4x for the channel multiplier; GELU-tanh
    epilogue on the first), the channel MLP as fused GEMMs (GELU-tanh epilogue); the residual stream stays float32.
    """
    from fastforks._optimize.conv_ops import dwconv, lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

    ln0, ln1, mix = m.layer_norm, m.layer_norm_1, m.conv_channels_mixer
    w1, b1 = _rsc(m.mlp1_up.weight), cast("torch.Tensor", m.mlp1_up.bias).detach().half()
    w2, b2 = _rsc(m.mlp1_up_1.weight), cast("torch.Tensor", m.mlp1_up_1.bias).detach().half()
    wu, bu = _w4(mix.mlp2_up.weight), mix.mlp2_up.bias.detach().half()
    wd, bd = _w4(mix.mlp2_down.weight), mix.mlp2_down.bias.detach().half()

    def forward(x: torch.Tensor) -> torch.Tensor:
        n, t, c = x.shape
        y = ln0(x).half().repeat_interleave(4, -1).view(n, t, 1, 4 * c).permute(0, 3, 1, 2)
        y = dwconv(dwconv(y, w1, b1, [1, 1], [1, 0], GELU_TANH), w2, b2, [1, 1], [1, 0], 0)
        x = x + y.permute(0, 2, 3, 1).view(n, t, c, 4).sum(-1)
        return x + lin(lin(ln1(x).half(), wu, bu, act=GELU_TANH), wd, bd)  # fp32 residual stream

    return forward


class Kernels(Portable):
    """The fp16 portable pass with the PIPs mixer blocks on depthwise / fused-GEMM kernels and fp16 accumulation."""

    def __init__(self, ctx: Context) -> None:
        """The portable network with the kernel forwards of the mixer blocks (fp16 filters captured in closures)."""
        super().__init__(ctx)
        for m in self.net.modules():
            if isinstance(m, PIPsConvBlock):
                m.forward = _block(m)  # ty: ignore[invalid-assignment]  # per-instance override

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        with fp16_accumulate():
            return super().__call__(frames, n, size, queries)

    def module(self) -> torch.nn.Module:
        """No exportable module: the mixer runs kernel launches."""
        return Runner.module(self)


RUNNERS = {"portable": Portable, "kernels": Kernels}
