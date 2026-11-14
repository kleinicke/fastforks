"""FILM runners: the film_net `Interpolator` (Style weights) and its call as the authors' PyTorch port runs it.

As dajes/frame-interpolation-pytorch `inference.py` (RGB in `[0, 1]`, zero-padded to a multiple of 64, centred, as its
`load_image`; the frame clamped to `[0, 1]`): a 7-level image pyramid of both frames, cascaded features (one shared
sub-tree extractor on every level), a residual flow pyramid in both directions (one predictor shared by the coarse
levels), both frames' images and features warped to the time on the 5 finest levels, and a U-Net-like decoder.

`_interpolate` is `Interpolator.forward` with both frames in one batch (the extractor and the flow predictors are
Siamese: the same weights for both, so batching changes nothing), the flows accumulated and the warps sampled in
float32 (upstream runs them in the network's dtype; the authors' fp16 path is `model.half()`), and the features and
convolutions in the runner's dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import base_type, override
from fastforks.interpolation._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import GraphCache

from .net import Interpolator
from .net.util import build_image_pyramid, flow_pyramid_synthesis

if TYPE_CHECKING:
    from fastforks.interpolation._runner import Context

    from .net.pyramid_flow_estimator import FlowEstimator, PyramidFlowEstimator


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`."""

    pyramid_levels: int  # image pyramid depth
    fusion_levels: int  # finest levels warped and fused
    specialized_levels: int  # decoder levels with their own widths
    sub_levels: int  # depth of the cascaded feature extractor's sub-trees
    filters: int  # feature width of the finest level
    flow_convs: tuple[int, ...]  # 3 x 3 convs of the flow predictors, finest first
    flow_filters: tuple[int, ...]  # their widths
    cuda_graphs: bool  # kernels: replay the network from a CUDA graph


def _taps(flow: torch.Tensor, *, other: bool) -> tuple[torch.Tensor, ...]:
    """`util.warp`'s bilinear taps (border, half-pixel grid) of `flow` `[N, 2, H, W]` (float32), rows of `[N*H*W, C]`.

    Returns the row indices of the four neighbours (top-left, top-right, bottom-left, bottom-right) and the weights
    `[N*H*W, 1]` along x and y. `grid_sample(align_corners=False)` of the normalized grid `linspace(1/W - 1, 1 - 1/W) +
    flow / (W / 2)` reads pixel `x + flow_x`, clipped to `[0, W - 1]` (border). `other`: sample `n` reads sample
    `(n + N / 2) % N` (the other frame of the batch).
    """
    n, _, h, w = flow.shape
    dev = flow.device
    px = (flow[:, 0] + torch.arange(w, device=dev, dtype=torch.float32)).clamp(0, w - 1)
    py = (flow[:, 1] + torch.arange(h, device=dev, dtype=torch.float32).view(h, 1)).clamp(0, h - 1)
    x0, y0 = px.floor().clamp(max=max(w - 2, 0)), py.floor().clamp(max=max(h - 2, 0))
    wx, wy = (px - x0).reshape(-1, 1), (py - y0).reshape(-1, 1)
    sample = torch.arange(n, device=dev).view(n, 1, 1)
    base = ((sample + n // 2) % n if other else sample) * (h * w)
    i00 = base + y0.long() * w + x0.long()
    dx, dy = int(w > 1), w * int(h > 1)
    return i00.view(-1), (i00 + dx).view(-1), (i00 + dy).view(-1), (i00 + dx + dy).view(-1), wx, wy


def _warp(x: torch.Tensor, taps: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """`util.warp` of `x` `[N, C, H, W]` at `taps` -> `[N, H, W, C]` in `x`'s dtype (row gathers, channels contiguous).

    The positions and weights are float32 (upstream's half path rounds the grid to float16, about half a pixel at
    1920 px); the blend is in `x`'s dtype.
    """
    _, c, h, w = x.shape
    rows = x.permute(0, 2, 3, 1).reshape(-1, c)  # a view of a channels-last tensor
    i00, i01, i10, i11, wx, wy = taps
    wx, wy = wx.to(x.dtype), wy.to(x.dtype)
    top = rows.index_select(0, i00).lerp_(rows.index_select(0, i01), wx)
    bottom = rows.index_select(0, i10).lerp_(rows.index_select(0, i11), wx)
    return top.lerp_(bottom, wy).view(-1, h, w, c)


def _nchw(x: torch.Tensor) -> torch.Tensor:
    """`[N, H, W, C]` -> `[N, C, H, W]` (channels-last memory)."""
    return x.permute(0, 3, 1, 2)


def _residual_flows(pfe: PyramidFlowEstimator, feats: list[torch.Tensor]) -> list[torch.Tensor]:
    """`PyramidFlowEstimator.forward` of both directions at once: the residual flow pyramid (float32), fine to coarse.

    `feats`: both frames' features `[2B, C, H, W]`; rows `[:B]` give the flow of frame 1 to 2, `[B:]` of 2 to 1. The
    warped features of the other frame are gathered (no swapped copy), the predictors' `cat` is built in NHWC.
    """
    shared: FlowEstimator = pfe._predictor  # noqa: SLF001 - upstream's attributes
    predictors: list[FlowEstimator] = list(pfe._predictors)  # noqa: SLF001  # ty: ignore[invalid-assignment] - a ModuleList of them

    def predict(predictor: FlowEstimator, f: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        net = _nchw(torch.cat((f.permute(0, 2, 3, 1), warped), -1))  # FlowEstimator.forward's cat
        for conv in predictor._convs:  # noqa: SLF001
            net = conv(net)
        return net.float()

    b2 = feats[0].shape[0]
    v = shared(feats[-1], torch.cat((feats[-1][b2 // 2 :], feats[-1][: b2 // 2]))).float()
    residuals = [v]
    steps: list[tuple[int, FlowEstimator]] = [(i, shared) for i in range(len(feats) - 2, len(predictors) - 1, -1)]
    steps += [(len(predictors) - 1 - k, p) for k, p in enumerate(predictors)]
    for i, predictor in steps:
        v = F.interpolate(2 * v, size=feats[i].shape[2:4], mode="bilinear")
        r = predict(predictor, feats[i], _warp(feats[i], _taps(v, other=True)))
        residuals.insert(0, r)
        v = r + v
    return residuals


def _interpolate(net: Interpolator, image1: torch.Tensor, image2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """`Interpolator.forward` of padded RGB `[B, 3, H, W]` frames at time `t` `[B, 1, 1, 1]` (the frame unclamped)."""
    b = image1.shape[0]
    dtype = net.fuse.output_conv.weight.dtype
    fmt = torch.channels_last if image1.is_cuda else torch.contiguous_format
    images = build_image_pyramid(torch.cat((image1, image2)).to(dtype, memory_format=fmt), net.pyramid_levels)
    feats = net.extract(images)
    flows = flow_pyramid_synthesis(_residual_flows(net.predict_flow, feats))[: net.fusion_pyramid_levels]
    # rows [:b]: frame 1 -> 2 (upstream's forward flow, scaled by 1 - t), [b:]: 2 -> 1 (backward, by t); frame 1 is
    # read with the backward flow and frame 2 with the forward one: sample n reads with the flow of row (n + b) % 2b
    scale = torch.cat((1 - t, t)).float()
    aligned = []
    for img, feat, flow in zip(images, feats, flows, strict=False):
        f = flow * scale
        f = torch.cat((f[b:], f[:b]))
        taps = _taps(f, other=False)
        wi, wf = _warp(img, taps), _warp(feat, taps)
        fl = f.to(dtype).permute(0, 2, 3, 1)
        level = torch.cat((wi[:b], wf[:b], wi[b:], wf[b:], fl[:b], fl[b:]), -1)  # warped 1, warped 2, flows
        aligned.append(_nchw(level) if image1.is_cuda else _nchw(level).contiguous())
    return net.fuse(aligned).float()


def _frame(net: Interpolator, image1: torch.Tensor, image2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The frame as `inference.py` keeps it: clamped to `[0, 1]`."""
    return _interpolate(net, image1, image2, t).clamp(0, 1)


def _net(ctx: Context) -> Interpolator:
    p = ctx.params
    net = Interpolator(
        pyramid_levels=p.pyramid_levels,
        fusion_pyramid_levels=p.fusion_levels,
        specialized_levels=p.specialized_levels,
        sub_levels=p.sub_levels,
        filters=p.filters,
        flow_convs=tuple(p.flow_convs),
        flow_filters=tuple(p.flow_filters),
    )
    sd = ctx.state_dict()  # the TorchScript archive's tensors
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    net = net.eval().to(ctx.device, getattr(torch, ctx.dtype))
    return net.to(memory_format=torch.channels_last) if net.fuse.output_conv.weight.is_cuda else net


def _time(image1: torch.Tensor, timestep: float) -> torch.Tensor:
    return torch.full((image1.shape[0], 1, 1, 1), timestep, device=image1.device)


class Portable(Runner):
    """Plain PyTorch in the card's dtype (the flows and the warps in float32)."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch."""
        return {"frame": _frame(self.net, image1, image2, _time(image1, timestep))}

    def module(self) -> torch.nn.Module:
        """The network; returns the middle frame (time 0.5)."""

        def run(net: Interpolator, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
            return _frame(net, image1, image2, _time(image1, 0.5))

        return Forward(self.net, run)


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    pad = tuple(k // 2 for k in conv.kernel_size) if conv.padding == "same" else conv.padding  # odd kernels only
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, pad, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convs (`cudnn_hacc`), one CUDA graph per shape."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network, route its convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if isinstance(m, torch.nn.Conv2d) and base_type(m) is torch.nn.Conv2d and m.groups == 1:
                override(m, _hacc_conv)
        after = cudnn_hacc.capture_after()
        self.graphs = GraphCache(self._eager, graphs=1, capture_after=after) if ctx.params.cuda_graphs else None

    def _eager(self, image1: torch.Tensor, image2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():
            return _frame(self.net, image1, image2, t)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch (the time is a graph input: one graph for every time)."""
        t = _time(image1, timestep)
        key = (image1.device, tuple(image1.shape))
        out = None if self.graphs is None else self.graphs.run(key, image1, image2, t)
        return {"frame": self._eager(image1, image2, t) if out is None else out}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        if self.graphs is not None:
            self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
