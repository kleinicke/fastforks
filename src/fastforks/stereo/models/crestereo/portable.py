"""Our plain-PyTorch CREStereo forward: upstream `test.py` inference on the `net/` modules, with less work.

Against `net/crestereo.py` + `test.py`, only float reordering changes the result:

- The cross attention of the 1/16 correlation runs once per pass (upstream: in every iteration, on the same inputs).
- The upsampling mask and the convex upsampling run after each stage's last iteration only (upstream: in every
  iteration, for the training predictions).
- The second pass (from the first pass's flow) skips the 1/16 and 1/8 features, offsets and attention it never uses.
- The four group correlations sample the right features once (one `grid_sample` for all groups and offsets).
- GRU: the context part of the six GRU convolutions (input channels 128:256, constant within a stage) runs once per
  stage, with the biases; z and r share one convolution.
- The two images go through the feature encoder one at a time (instance norm is per image: same result).
- Half precision (CUDA): convolutions and correlations in fp16; flow, sampling coordinates, offsets, masks and the
  attention layers in fp32 (the features are sampled from an fp32 copy: fp16 grid coordinates are too coarse).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.sampling import grid_sample

from .net.attention import PositionEncodingSine

if TYPE_CHECKING:
    from collections.abc import Callable

    from .net.crestereo import CREStereo
    from .net.update import BasicUpdateBlock

GROUPS = 4  # fixed: AGCL's channel groups
SEARCH = 9  # fixed: AGCL's search window size (3 x 3 or 1 x 9)


def prepare(net: CREStereo, *, half: bool) -> None:
    """fp16 weights except the attention layers (once, at load)."""
    if half:
        net.half()
        net.self_att_fn.float()
        net.cross_att_fn.float()


def forward_fast(
    net: CREStereo, left: torch.Tensor, right: torch.Tensor, iters: int, ops: Ops | None = None
) -> torch.Tensor:
    """Disparity `[B, H, W]` of padded RGB `[B, 3, H, W]` pairs in 0-255 (upstream `test.py` inference)."""
    left, right = left.flip(1), right.flip(1)  # upstream reads BGR (cv2.imread)
    h, w = left.shape[-2:]
    small = [F.interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=True) for x in (left, right)]
    ops = ops or PORTABLE
    flow = run_pass(net, small[0], small[1], iters, None, ops)
    return run_pass(net, left, right, iters, flow, ops)[:, 0]


def run_pass(  # noqa: PLR0913, PLR0917 - pass inputs
    net: CREStereo,
    image1: torch.Tensor,
    image2: torch.Tensor,
    iters: int,
    flow_init: torch.Tensor | None,
    ops: Ops,
) -> torch.Tensor:
    """`CREStereo.forward` in test mode: the final full-resolution flow `[B, 2, H, W]` (fp32)."""
    dt = net.fnet.conv1.weight.dtype
    image1 = (2 * (image1 / 255.0) - 1.0).to(dt)
    image2 = (2 * (image2 / 255.0) - 1.0).to(dt)
    fmap1 = net.fnet(image1)
    fmap2 = net.fnet(image2)
    hdim = net.hidden_dim
    hidden, inp = torch.split(fmap1, [hdim, fmap1.shape[1] - hdim], dim=1)
    hidden, inp = torch.tanh(hidden), F.relu(inp)
    ub = net.update_block

    if flow_init is None:
        # 1/8: features, offsets, context
        fmap1_dw8, fmap2_dw8 = F.avg_pool2d(fmap1, 2, stride=2), F.avg_pool2d(fmap2, 2, stride=2)
        offset_dw8 = net.range_8 * (torch.sigmoid(net.conv_offset_8(fmap1_dw8).float()) - 0.5) * 2.0
        net_dw8, inp_dw8 = F.avg_pool2d(hidden, 2, stride=2), F.avg_pool2d(inp, 2, stride=2)
        # 1/16: features, offsets, context, then self and cross attention (fp32)
        fmap1_dw16, fmap2_dw16 = F.avg_pool2d(fmap1, 4, stride=4), F.avg_pool2d(fmap2, 4, stride=4)
        offset_dw16 = net.range_16 * (torch.sigmoid(net.conv_offset_16(fmap1_dw16).float()) - 0.5) * 2.0
        net_dw16, inp_dw16 = F.avg_pool2d(hidden, 4, stride=4), F.avg_pool2d(inp, 4, stride=4)
        att1, att2 = _attention(net, fmap1_dw16.float(), fmap2_dw16.float())

        flow = att1.new_zeros((att1.shape[0], 2, *att1.shape[2:]))
        corr = functools.partial(_corr_offset, att1, att2, offset_dw16)
        _, flow, mask = _stage(ub, net_dw16, inp_dw16, corr, flow, iters // 2, ops.gru)
        flow = net.convex_upsample(flow, mask, rate=4)

        flow = _resize(flow, fmap1_dw8)
        corr = functools.partial(_corr_offset, fmap1_dw8, fmap2_dw8.float(), offset_dw8)
        _, flow, mask = _stage(ub, net_dw8, inp_dw8, corr, flow, iters // 2, ops.gru)
        flow = net.convex_upsample(flow, mask, rate=4)
        flow = _resize(flow, fmap1)
    else:
        flow = _resize(flow_init, fmap1)

    corr = functools.partial(ops.corr_iter, fmap1, fmap2.float())
    _, flow, mask = _stage(ub, hidden, inp, corr, flow, iters, ops.gru)
    return -net.convex_upsample(flow, mask, rate=4)


def _resize(flow: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Upstream's stage hand-over: `-scale * interpolate(flow)` to the size of `like` (the sign flip included)."""
    scale = like.shape[2] / flow.shape[2]
    return -scale * F.interpolate(flow, size=like.shape[2:], mode="bilinear", align_corners=True)


def _attention(net: CREStereo, fmap1: torch.Tensor, fmap2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Position encoding, self attention, then the cross attention `AGCL` applies in every 1/16 iteration (once)."""
    n, c, h, w = fmap1.shape
    pe = _position_encoding(c, h, w, fmap1.device)
    seq1, seq2 = ((x + pe).flatten(2).transpose(1, 2) for x in (fmap1, fmap2))  # 'n c h w -> n (h w) c'
    seq1, seq2 = net.self_att_fn(seq1, seq2)
    seq1, seq2 = net.cross_att_fn(seq1, seq2)
    return tuple(x.transpose(1, 2).reshape(n, c, h, w) for x in (seq1, seq2))


def _position_encoding(c: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    if torch.compiler.is_compiling():  # dynamo cannot trace through lru_cache; the graph holds the encoding anyway
        return PositionEncodingSine(d_model=c, max_shape=(h, w)).pe.to(device)
    return _cached_position_encoding(c, h, w, device)


@functools.lru_cache(maxsize=8)
def _cached_position_encoding(c: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    return PositionEncodingSine(d_model=c, max_shape=(h, w)).pe.to(device)


def _coords(flow: torch.Tensor) -> torch.Tensor:
    """Pixel coordinates `coords_grid + flow` as `[N, H, W, 2]` (x, y), fp32."""
    _, _, h, w = flow.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    return (torch.stack((xs, ys)) + flow).permute(0, 2, 3, 1)


def _sample(fmap: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """`utils.bilinear_sampler`: bilinear lookup at pixel coordinates `[N, h, w, 2]`, zero outside."""
    hgt, wid = fmap.shape[-2:]
    x, y = coords.unbind(-1)
    grid = torch.stack((2 * x / (wid - 1) - 1, 2 * y / (hgt - 1) - 1), dim=-1)
    return grid_sample(fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _window(small_patch: bool) -> tuple[int, int]:  # noqa: FBT001 - upstream's flag
    """Half sizes (ry, rx) of the search window: 3 x 3 or 1 x 9."""
    return (1, 1) if small_patch else (0, 4)


def _corr_iter(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    flow: torch.Tensor,
    small_patch: bool,  # noqa: FBT001 - upstream's flag
) -> torch.Tensor:
    """`AGCL.corr_iter`: warp the right features by the flow, then correlate over a local window.

    The group correlations over a replicate-padded window; `[N, 4 * 9, H, W]`, channel `group * 9 + window index`.
    """
    n, c, h, w = fmap1.shape
    warped = _sample(fmap2, _coords(flow)).to(fmap1.dtype)
    ry, rx = _window(small_patch)
    padded = F.pad(warped, (rx, rx, ry, ry), mode="replicate")
    corrs = [
        (fmap1 * padded[:, :, dy : dy + h, dx : dx + w]).unflatten(1, (GROUPS, c // GROUPS)).mean(2)
        for dy in range(2 * ry + 1)
        for dx in range(2 * rx + 1)
    ]
    return torch.stack(corrs, dim=2).reshape(n, GROUPS * SEARCH, h, w)


def _corr_offset(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    extra_offset: torch.Tensor,
    flow: torch.Tensor,
    small_patch: bool,  # noqa: FBT001 - upstream's flag
) -> torch.Tensor:
    """`AGCL.corr_att_offset` (attention done by the caller): correlation at the learned offsets.

    Group correlations at the window positions shifted by the offsets; `[N, 4 * 9, H, W]`, channel order of
    `_corr_iter`.
    """
    n, c, h, w = fmap1.shape
    ry, rx = _window(small_patch)
    dy, dx = torch.meshgrid(
        torch.arange(-ry, ry + 1, device=flow.device, dtype=flow.dtype),
        torch.arange(-rx, rx + 1, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    window = torch.stack((dx.flatten(), dy.flatten()), dim=-1)  # [9, 2] (x, y)
    offsets = window[None, :, None, None] + extra_offset.reshape(n, SEARCH, 2, h, w).permute(0, 1, 3, 4, 2)
    coords = _coords(flow)[:, None] + offsets  # [N, 9, H, W, 2]
    warped = _sample(fmap2, coords.reshape(n, SEARCH * h, w, 2)).reshape(n, c, SEARCH, h, w).to(fmap1.dtype)
    left = fmap1.unflatten(1, (GROUPS, c // GROUPS))[:, :, :, None]
    corr = (left * warped.unflatten(1, (GROUPS, c // GROUPS))).mean(2)  # [N, 4, 9, H, W]
    return corr.reshape(n, GROUPS * SEARCH, h, w)


def _gru(  # noqa: PLR0913, PLR0917 - one GRU half step
    hidden: torch.Tensor,
    motion: torch.Tensor,
    w_zr: torch.Tensor,
    w_q: torch.Tensor,
    pad: tuple[int, int],
    c_zr: torch.Tensor,
    c_q: torch.Tensor,
) -> torch.Tensor:
    """One `SepConvGRU` half step: the new state from `hidden` and `motion`.

    The weights without the context channels; their terms are `c_zr` (z, then r) and `c_q`, with the biases.
    """
    z, r = torch.sigmoid(F.conv2d(torch.cat([hidden, motion], 1), w_zr, padding=pad) + c_zr).chunk(2, 1)
    q = torch.tanh(F.conv2d(torch.cat([r * hidden, motion], 1), w_q, padding=pad) + c_q)
    return (1 - z) * hidden + z * q


@dataclass(frozen=True)
class Ops:
    """The steps a kernels runner can swap: `_corr_iter` and `_gru` (same signatures)."""

    corr_iter: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]
    gru: Callable[..., torch.Tensor]


PORTABLE = Ops(_corr_iter, _gru)


def _stage(  # noqa: PLR0913, PLR0917 - stage inputs
    ub: BasicUpdateBlock,
    hidden: torch.Tensor,
    inp: torch.Tensor,
    corr_fn: Callable[[torch.Tensor, bool], torch.Tensor],
    flow: torch.Tensor,
    iters: int,
    gru_step: Callable[..., torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One cascade stage: `iters` updates (3 x 3 window on odd iterations), then the upsampling mask (fp32)."""
    gru = ub.gru
    hd = hidden.shape[1]
    convs = ((gru.convz1, gru.convr1, gru.convq1), (gru.convz2, gru.convr2, gru.convq2))
    weights, context = [], []
    for z, r, q in convs:
        w_zr = torch.cat([z.weight, r.weight])
        own = [slice(0, hd), slice(2 * hd, None)]  # hidden + motion input channels; 128:256 is the context
        weights.append(
            (torch.cat([w_zr[:, s] for s in own], 1), torch.cat([q.weight[:, s] for s in own], 1), z.padding)
        )
        bias = torch.cat([m.bias for m in (z, r, q) if m.bias is not None])  # all three have one
        ctx = F.conv2d(inp, torch.cat([w_zr, q.weight])[:, hd : 2 * hd], bias, padding=z.padding)
        context.append(ctx.split([2 * hd, hd], dim=1))
    for itr in range(iters):
        corr = corr_fn(flow, itr % 2 == 1).to(hidden.dtype)
        motion = ub.encoder(flow.to(hidden.dtype), corr)
        for (w_zr, w_q, pad), (c_zr, c_q) in zip(weights, context, strict=True):
            hidden = gru_step(hidden, motion, w_zr, w_q, pad, c_zr, c_q)
        flow = flow + ub.flow_head(hidden).float()
    mask = 0.25 * ub.mask(hidden).float()
    return hidden, flow, mask
