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
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import Interpolator
from .net.util import build_image_pyramid, flow_pyramid_synthesis

if TYPE_CHECKING:
    from fastforks.interpolation._runner import Context

    from .net.fusion import Fusion
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


def _taps(flow: torch.Tensor, dtype: torch.dtype, *, other: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """`util.warp`'s bilinear taps (border, half-pixel grid) of `flow` `[N, 2, H, W]` (float32), rows of `[N*H*W, C]`.

    Returns the row indices `[N*H*W, 4]` of the four neighbours (top-left, top-right, bottom-left, bottom-right) and
    their weights in `dtype`. `grid_sample(align_corners=False)` of the normalized grid `linspace(1/W - 1, 1 - 1/W) +
    flow / (W / 2)` reads pixel `x + flow_x`, clipped to `[0, W - 1]` (border). `other`: sample `n` reads sample
    `(n + N / 2) % N` (the other frame of the batch). The positions are float32 (upstream's half path rounds the grid
    to float16, about half a pixel at 1920 px).
    """
    n, _, h, w = flow.shape
    dev = flow.device
    px = (flow[:, 0] + torch.arange(w, device=dev, dtype=torch.float32)).clamp(0, w - 1)
    py = (flow[:, 1] + torch.arange(h, device=dev, dtype=torch.float32).view(h, 1)).clamp(0, h - 1)
    x0, y0 = px.floor().clamp(max=max(w - 2, 0)), py.floor().clamp(max=max(h - 2, 0))
    wx, wy = (px - x0).view(-1, 1, 1), (py - y0).view(-1, 1, 1)
    weights = (torch.cat((1 - wy, wy), 1) * torch.cat((1 - wx, wx), 2)).view(-1, 4).to(dtype)
    sample = torch.arange(n, device=dev).view(n, 1, 1)
    base = ((sample + n // 2) % n if other else sample) * (h * w)
    i00 = (base + y0.long() * w + x0.long()).view(-1)
    dx, dy = int(w > 1), w * int(h > 1)
    return torch.stack((i00, i00 + dx, i00 + dy, i00 + dx + dy), 1), weights


def _warp(x: torch.Tensor, taps: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """`util.warp` of `x` `[N, C, H, W]` at `taps` -> `[N, H, W, C]` in `x`'s dtype (rows, channels contiguous).

    One weighted 4-row `embedding_bag` (float32 accumulation) per pixel; a plain gather where it is not implemented.
    """
    _, c, h, w = x.shape
    rows = x.permute(0, 2, 3, 1).reshape(-1, c)  # a view of a channels-last tensor
    index, weights = taps
    if x.device.type in ("cuda", "cpu"):
        out = F.embedding_bag(index, rows, per_sample_weights=weights, mode="sum")
    else:
        out = (rows[index] * weights[..., None]).sum(1)
    return out.view(-1, h, w, c)


def _nchw(x: torch.Tensor) -> torch.Tensor:
    """`[N, H, W, C]` -> `[N, C, H, W]` (channels-last memory; contiguous off CUDA)."""
    x = x.permute(0, 3, 1, 2)
    return x if x.is_cuda else x.contiguous()


def _with_zeros(w: torch.Tensor, at: int, n: int) -> torch.Tensor:
    """`w` `[O, I, kh, kw]` with `n` zero input channels inserted at `at`."""
    return torch.cat((w[:, :at], w.new_zeros(w.shape[0], n, *w.shape[2:]), w[:, at:]), 1)


def _conv(weight: torch.Tensor, bias: torch.Tensor | None, padding: int | str) -> torch.nn.Conv2d:
    o, i, kh, kw = weight.shape
    conv = torch.nn.Conv2d(i, o, (kh, kw), padding=padding, bias=bias is not None)
    with torch.no_grad():
        conv.weight.copy_(weight)
        if bias is not None:
            conv.bias.copy_(bias)  # ty: ignore[unresolved-attribute]
    return conv


def _phases(up: torch.nn.Conv2d) -> torch.nn.Conv2d:
    """`up` (2 x 2, zero padded right and bottom) applied to a nearest 2x upsampling, as a conv of the coarse map.

    Output pixel `(2p + i, 2q + j)` reads coarse rows `p, p + 1` with the kernel rows `M_i @ W` (`M_0` sums both taps
    into row 0, `M_1` keeps them), and the same for the columns; output channel `c * 4 + i * 2 + j`, as
    `pixel_shuffle(2)` wants. Its input is the coarse map zero padded right and bottom.
    """
    w = up.weight.float()
    m = torch.tensor([[[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]], device=w.device)  # [phase, kernel row, tap]
    k = torch.einsum("ira,jsb,ocab->oijcrs", m, m, w).reshape(-1, w.shape[1], 2, 2)
    return _conv(k, None if up.bias is None else up.bias.float().repeat_interleave(4), 0)


class _Fusion(torch.nn.Module):
    """`Fusion` on levels padded to a multiple of 8 channels, the resize convolutions folded into the coarse map.

    The `nearest 2x -> zero pad -> 2 x 2 conv` of each level is one 2 x 2 conv of the coarse map with 4 phases of
    outputs and a `pixel_shuffle` (no upsampled copy); each level's `cat` takes the warped pieces and the upsampled map
    at once, and the zero channels that pad it to a multiple of 8 (cuDNN pads other widths on every call) meet zero
    weights.
    """

    def __init__(self, fuse: Fusion) -> None:
        super().__init__()
        convs = [[layer[0] for layer in level] for level in fuse.convs]  # ty: ignore[not-iterable] - coarse to fine
        coarse = convs[0][0].in_channels
        widths = [conv[1].in_channels - conv[0].out_channels for conv in convs[::-1]] + [coarse]  # fine to coarse
        self.pads = [-c % 8 for c in widths]
        first = _conv(_with_zeros(convs[0][0].weight, coarse, self.pads[-1]), convs[0][0].bias, 0)
        self.up = torch.nn.ModuleList(_phases(c[0] if k else first) for k, c in enumerate(convs))
        self.merge = torch.nn.ModuleList(
            _conv(_with_zeros(c[1].weight, widths[-2 - k], self.pads[-2 - k]), c[1].bias, "same")
            for k, c in enumerate(convs)
        )
        self.refine = torch.nn.ModuleList(c[2] for c in convs)
        self.output_conv = fuse.output_conv

    def forward(self, pyramid: list[list[torch.Tensor]]) -> torch.Tensor:
        """The frame from the NHWC pieces of every level (fine to coarse, each with its zero padding last)."""
        net = _nchw(torch.cat(pyramid[-1], -1))
        pyramid[-1].clear()
        for up, merge, refine, pieces in zip(self.up, self.merge, self.refine, pyramid[-2::-1], strict=True):
            upsampled = F.pixel_shuffle(up(F.pad(net, (0, 1, 0, 1))), 2).permute(0, 2, 3, 1)
            net = _nchw(torch.cat((*pieces, upsampled), -1))
            pieces.clear()  # the warped level is in the cat now
            net = F.leaky_relu(merge(net), 0.2, inplace=True)
            net = F.leaky_relu(refine(net), 0.2, inplace=True)
        return self.output_conv(net)


class _Head(torch.nn.Module):
    """A `FlowEstimator` whose first conv is split by input: `conv(cat(f, other)) = conv_f(f) + conv_o(other)`."""

    def __init__(self, predictor: FlowEstimator) -> None:
        super().__init__()
        first: torch.nn.Conv2d = predictor._convs[0][0]  # noqa: SLF001  # ty: ignore[not-subscriptable] - upstream's attributes
        c = first.in_channels // 2
        self.own = _conv(first.weight[:, :c], first.bias, first.padding)  # ty: ignore[invalid-argument-type]
        self.other = _conv(first.weight[:, c:], None, first.padding)  # ty: ignore[invalid-argument-type]
        self.rest = predictor._convs[1:]  # noqa: SLF001


class _Flows(torch.nn.Module):
    """`PyramidFlowEstimator`'s predictors as `_Head`s: the shared one and the finest levels' own, coarse to fine."""

    def __init__(self, pfe: PyramidFlowEstimator) -> None:
        super().__init__()
        self.shared = _Head(pfe._predictor)  # noqa: SLF001 - upstream's attributes
        self.own = torch.nn.ModuleList(_Head(p) for p in pfe._predictors)  # noqa: SLF001  # ty: ignore[invalid-argument-type]


def _predict(
    head: _Head, f: torch.Tensor, other: torch.Tensor, taps: tuple[torch.Tensor, torch.Tensor] | None
) -> torch.Tensor:
    """`FlowEstimator.forward(f, warp(other))` (float32) without the `cat`; `taps=None`: `other` is already aligned.

    The warped features live only until their half of the first conv is done (the largest buffer of the stage).
    """
    warped = other if taps is None else _warp(other, taps)
    del other
    net = head.other(_nchw(warped) if taps is not None else warped)
    del warped
    net = F.leaky_relu(head.own(f).add_(net), 0.2, inplace=True)
    for conv in head.rest:
        net = conv(net)
    return net.float()


def _residual_flows(flows: _Flows, feats: list[torch.Tensor]) -> list[torch.Tensor]:
    """`PyramidFlowEstimator.forward` of both directions at once: the residual flow pyramid (float32), fine to coarse.

    `feats`: both frames' features `[2B, C, H, W]`; rows `[:B]` give the flow of frame 1 to 2, `[B:]` of 2 to 1. The
    features of the other frame are warped by gathering its rows (no swapped copy).
    """
    b2 = feats[0].shape[0]
    v = _predict(flows.shared, feats[-1], torch.cat((feats[-1][b2 // 2 :], feats[-1][: b2 // 2])), None)
    residuals = [v]
    heads: list[_Head] = list(flows.own)  # ty: ignore[invalid-assignment] - a ModuleList of them
    steps = [(i, flows.shared) for i in range(len(feats) - 2, len(heads) - 1, -1)]
    steps += [(len(heads) - 1 - k, h) for k, h in enumerate(heads)]
    for i, head in steps:
        v = F.interpolate(2 * v, size=feats[i].shape[2:4], mode="bilinear")
        r = _predict(head, feats[i], feats[i], _taps(v, feats[i].dtype, other=True))
        residuals.insert(0, r)
        v = r + v
    return residuals


def _interpolate(net: Interpolator, image1: torch.Tensor, image2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """`Interpolator.forward` of padded RGB `[B, 3, H, W]` frames at time `t` `[B, 1, 1, 1]` (the frame unclamped)."""
    b = image1.shape[0]
    fuse = net.fuse if isinstance(net.fuse, _Fusion) else _Fusion(net.fuse)
    predict = net.predict_flow if isinstance(net.predict_flow, _Flows) else _Flows(net.predict_flow)
    dtype = fuse.output_conv.weight.dtype
    fmt = torch.channels_last if image1.is_cuda else torch.contiguous_format
    images = build_image_pyramid(torch.cat((image1, image2)).to(dtype, memory_format=fmt), net.pyramid_levels)
    feats = net.extract(images)
    flows = flow_pyramid_synthesis(_residual_flows(predict, feats))[: net.fusion_pyramid_levels]
    # rows [:b]: frame 1 -> 2 (upstream's forward flow, scaled by 1 - t), [b:]: 2 -> 1 (backward, by t); frame 1 is
    # read with the backward flow and frame 2 with the forward one: sample n reads with the flow of row (n + b) % 2b
    scale = torch.cat((1 - t, t)).float()
    pyramid = []
    for img, feat, flow, pad in zip(images, feats, flows, fuse.pads, strict=False):
        f = flow * scale
        f = torch.cat((f[b:], f[:b]))
        taps = _taps(f, dtype, other=False)
        wi, wf = _warp(img, taps), _warp(feat, taps)
        fl = f.to(dtype).permute(0, 2, 3, 1)
        zeros = fl.new_zeros(()).expand(b, *fl.shape[1:3], pad)
        pyramid.append([wi[:b], wf[:b], wi[b:], wf[b:], fl[:b], fl[b:], zeros])  # warped 1, warped 2, flows
    del images, feats, flows
    return fuse(pyramid).float()


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
    net.fuse = _Fusion(net.fuse)  # ty: ignore[invalid-assignment] - the same decoder, regrouped
    net.predict_flow = _Flows(net.predict_flow)  # ty: ignore[invalid-assignment] - the same predictors, split
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


# a graph saves 15 % at 256 x 448 and nothing from 480p on (GPU-bound), where its pool would only cost memory
_GRAPH_PIXELS = 256 * 512


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convs (`cudnn_hacc`), a CUDA graph for small inputs."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network, route its convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if isinstance(m, torch.nn.Conv2d) and base_type(m) is torch.nn.Conv2d and m.groups == 1:
                override(m, _hacc_conv)
        self.graphs = SmallShapeGraphs(
            self._eager,
            pixels=_GRAPH_PIXELS,
            graphs=1,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )

    def _eager(self, image1: torch.Tensor, image2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():
            return _frame(self.net, image1, image2, t)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch (the time is a graph input: one graph for every time)."""
        return {"frame": self.graphs(image1, image2, _time(image1, timestep))}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
