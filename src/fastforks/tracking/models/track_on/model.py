"""Track-On2 / Track-On-R runners: weights -> network and the online pass (upstream `Predictor.forward`).

The frames are resized to the card's `height x width` (bilinear) and encoded `FRAME_CHUNK` at a time. The user's
queries plus a `support_grid x support_grid` grid on frame 0 (upstream's support points, in input pixels with a
margin of `W / 64`) are tracked online: a query starts at its frame with the fused-feature sample at its position,
then every frame decodes the active queries against the frame and their memory of the last `memory` frames
(upstream inference memory 72: the trained temporal embedding interpolated to it). Query and track coordinates
follow upstream (pixel edges at integers) inside, pixel centres at integers outside. The visibility is
`sigmoid(logit - ln 4)`, so that > 0.5 is upstream's 0.8 threshold; frames before a query's frame give the query
position with visibility 0. `Kernels` runs the ViT blocks on fastbb / dpt_fused (fp16, fused RoPE, flash attention)
and the rest in float16 with fp16 accumulation.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks.tracking._runner import Forward, Runner

from .net.track_on import D, TrackOn2

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.tracking._runner import Context

FRAME_CHUNK = 8  # fixed: memory chunking of the feature pass, not an architecture value
VIS_SHIFT = math.log(0.8 / 0.2)  # fixed: maps upstream's 0.8 visibility threshold to 0.5
GRID_MARGIN = 64  # fixed: upstream get_points_on_a_grid margin W / 64


@dataclass(frozen=True)
class Params:
    """Card `[params]` and `[options]`."""

    arch: str  # DINOv3 backbone of the ViT-Adapter ("vits16plus")
    vit_scale: float  # upstream vit_upsample_factor: the ViT input is this much larger than the network frame
    height: int  # the network's frame size (upstream input_size)
    width: int
    memory: int  # inference memory in frames (upstream M_i)
    support_grid: int  # support points per side on frame 0 (upstream support_grid_size; 0 = none)


def build(ctx: Context) -> TrackOn2:
    """The variant's network from its weights: the Track-On2 checkpoint (flat) + the DINOv3 ViT (HF safetensors)."""
    p = ctx.params
    sd = ctx.state_dict(0, strip="")
    sd.update({f"backbone.vit_encoder.dinov3.{k}": v for k, v in ctx.state_dict(1, strip="").items()})
    if "t_embedding" in sd:
        sd["t_embedding"] = TrackOn2.extend_memory(sd["t_embedding"].float(), p.memory)
    net = TrackOn2(p.arch, p.vit_scale, (p.height, p.width), p.memory)
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    return net


def grid(n: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Upstream `get_points_on_a_grid(n, (h, w))`: `[n^2, 2]` (x, y), edge-aligned pixels."""
    if n == 1:
        return torch.tensor([[w / 2, h / 2]], device=device)
    m = w / GRID_MARGIN
    ys, xs = torch.meshgrid(
        torch.linspace(m, h - m, n, device=device), torch.linspace(m, w - m, n, device=device), indexing="ij"
    )
    return torch.stack([xs, ys], -1).reshape(-1, 2)


def track(  # noqa: PLR0913, PLR0917 - the runner call's arguments plus params and the autocast dtype
    net: TrackOn2,
    frames: Callable[[int], torch.Tensor],
    n: int,
    size: tuple[int, int],
    queries: torch.Tensor,
    params: Params,
    amp: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """`{"tracks": [T, Q, 2], "visibility": [T, Q]}` of frames `frames(t)` `[1, 3, H, W]` in `[0, 1]`."""
    h, w = size
    dev = queries.device
    nq = queries.shape[0]
    qt = queries[:, 0].long()
    xy = queries[:, 1:].float() + 0.5
    if params.support_grid > 0:
        g = grid(params.support_grid, h, w, dev)
        qt = torch.cat([qt, qt.new_zeros(g.shape[0])])
        xy = torch.cat([xy, g])
    order = torch.sort(qt, stable=True).indices
    qt_sorted = qt[order].tolist()
    xy_sorted = xy[order] / xy.new_tensor((w, h))
    total = len(qt_sorted)
    scale = xy.new_tensor((w / params.width, h / params.height))
    tracks = queries.new_zeros(n, total, 2)
    logits = queries.new_full((n, total), -math.inf)
    q_init = queries.new_zeros(total, D)
    memory = queries.new_zeros(total, params.memory, D)
    mask = torch.ones(total, params.memory, dtype=torch.bool, device=dev)
    region = torch.autocast(dev.type, amp) if amp is not None else contextlib.nullcontext()
    active = 0
    with region:
        for s in range(qt_sorted[0] if total else n, n, FRAME_CHUNK):
            feats = net.features(torch.cat([frames(t) for t in range(s, min(n, s + FRAME_CHUNK))]))
            for t, frame in enumerate(feats, s):
                new = active
                while new < total and qt_sorted[new] <= t:
                    new += 1
                if new > active:
                    q_init[active:new] = net.sample(frame, xy_sorted[active:new]).float()
                    active = new
                p, v, q = net.step(q_init[:active], mask[:active], memory[:active], frame)
                tracks[t, :active], logits[t, :active] = p.float() * scale, v.float()
                memory[:active] = torch.cat([memory[:active, 1:], q.float()[:, None]], 1)
                mask[:active] = torch.cat([mask[:active, 1:], mask.new_zeros(active, 1)], 1)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(total, device=dev)
    sel = inv[:nq]
    tracks, logits = tracks[:, sel] - 0.5, logits[:, sel]
    before = torch.arange(n, device=dev)[:, None] < queries[None, :, 0]
    tracks = torch.where(before[..., None], queries[None, :, 1:].float(), tracks)
    return {"tracks": tracks, "visibility": (logits - VIS_SHIFT).sigmoid()}


class Portable(Runner):
    """Plain PyTorch."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = build(ctx).eval().requires_grad_(requires_grad=False).to(ctx.device)
        self.amp = torch.float16 if ctx.dtype == "float16" else None

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        return track(self.net, frames, n, size, queries, self.ctx.params, self.amp)

    def module(self) -> torch.nn.Module:
        """The network with the online loop; `forward(video [T, 3, H, W], queries [Q, 3])`."""
        amp, params = self.amp, self.ctx.params

        def fn(net: TrackOn2, video: torch.Tensor, queries: torch.Tensor) -> dict[str, torch.Tensor]:
            size = (video.shape[-2], video.shape[-1])
            return track(net, lambda t: video[t : t + 1], video.shape[0], size, queries, params, amp)

        return Forward(self.net, fn)


class Kernels(Portable):
    """The fp16 pass with fp16 accumulation; the ViT blocks on the fastbb / dpt_fused kernels."""

    def __init__(self, ctx: Context) -> None:
        """The portable network in fp16 autocast with kernel ViT blocks (weights repacked once)."""
        super().__init__(ctx)
        self.amp = torch.float16
        from .kernels import install  # noqa: PLC0415 - loads the fastbb / dpt_fused kernels (kernels mode only)

        install(self.net)

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        with fp16_accumulate():
            return super().__call__(frames, n, size, queries)

    def module(self) -> torch.nn.Module:
        """No exportable module: the blocks run kernel launches."""
        return Runner.module(self)


RUNNERS = {"portable": Portable, "kernels": Kernels}
