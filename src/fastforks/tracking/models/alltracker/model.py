"""AllTracker runners: weights -> network and the windowed offline pass.

The frames are resized to the card's `height x width` (bilinear, as upstream's cv2 resize), padded to multiples of 64
(centred, replicated; upstream InputPadder) and encoded `FRAME_CHUNK` at a time, once per call. Per query frame `t0`
the network tracks every pixel of `t0` forwards through frames `t0..T-1` and, if `t0 > 0`, backwards through the
reversed frames `t0..0` (upstream demo.py). A direction of `L >= 3` frames runs as upstream `Net.forward`: 16-frame
windows with stride 8 (the last frame repeated to fill the last one), each warm-started from the earlier windows'
1/8 flow, the later window overwriting the overlap; `L == 2` is upstream's two-frame flow (one window of 1 frame).
The anchor frame of a 1- or 2-frame direction is the query itself (visible). Only the last refinement step is
upsampled (8x convex), and only for the frames it keeps; the tracks are the query plus that flow, sampled bilinearly
at the query (upstream's evaluation rounds the query to a pixel). The visibility is upstream's `visible * confident`
raised to `ln 0.5 / ln 0.6`, so that > 0.5 is upstream's 0.6 threshold. `Kernels` runs the ConvNeXt blocks, the update
block and the heads in float16 through fvm_conv (fused convs / GEMMs with their epilogues, fp16 accumulation).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks.tracking._runner import Forward, Runner

from .net.alltracker import DIM, WINDOW, AllTracker, CNBlock1d, CNBlock2d, CorrPyramid, UpdateBlock, upsample

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.tracking._runner import Context

FRAME_CHUNK = 16  # fixed: memory chunking of the feature pass, not an architecture value
STRIDE = WINDOW // 2  # fixed: upstream's window stride at inference
VIS_POWER = math.log(0.5) / math.log(0.6)  # fixed: maps upstream's 0.6 visibility threshold to 0.5


@dataclass(frozen=True)
class Params:
    """Card `[params]` and `[options]`."""

    encoder: Literal["convnext", "basic"]  # upstream Net() / Net(use_basicencoder=True, no_split=True) (tiny)
    height: int  # the network's frame size (upstream evaluation image_size)
    width: int
    iters: int  # refinement steps per window (upstream inference_iters)


def build(ctx: Context) -> AllTracker:
    """The variant's network from its weights (upstream `{"model": state_dict}`)."""
    sd = ctx.state_dict(key="model", strip="")
    net = AllTracker(tiny=ctx.params.encoder == "basic")
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    return net


def _pad(h: int, w: int) -> tuple[int, int, int, int]:
    """Upstream InputPadder ('sintel'): `(left, right, top, bottom)` to the next multiples of 64, centred."""
    ph, pw = (-h) % 64, (-w) % 64
    return pw // 2, pw - pw // 2, ph // 2, ph - ph // 2


def _sample(
    flows8: torch.Tensor, visconfs8: torch.Tensor, mask: torch.Tensor, pts: torch.Tensor, offset: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tracks `[S, K, 2]` and `visible * confident` `[S, K]` of the points `pts` `[K, 2]` (unpadded network pixels)."""
    flow = upsample(flows8, mask)
    vis = upsample(visconfs8, mask).sigmoid()
    h, w = flow.shape[-2:]
    xy = pts + offset
    grid = (xy * xy.new_tensor((2 / (w - 1), 2 / (h - 1))) - 1)[None, None].expand(flow.shape[0], -1, -1, -1)
    f = F.grid_sample(torch.cat([flow, vis], 1).float(), grid, align_corners=True, padding_mode="border")[:, :, 0]
    return pts + f[:, :2].transpose(1, 2), f[:, 2] * f[:, 3]


def _direction(
    net: AllTracker, fmaps: torch.Tensor, pts: torch.Tensor, offset: torch.Tensor, iters: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tracks `[L, K, 2]` and `visible * confident` `[L, K]` of points on frame 0 of `fmaps` `[L, C, h, w]`."""
    n, k = fmaps.shape[0], pts.shape[0]
    tracks, vis = pts[None].repeat(n, 1, 1), pts.new_ones(n, k)
    if n == 1:
        return tracks, vis
    zeros = fmaps.new_zeros((1, 2, *fmaps.shape[-2:]), dtype=torch.float32)
    if n == 2:  # noqa: PLR2004 - upstream's two-frame flow
        f8, v8, m = net.window(fmaps[0], fmaps[1:], zeros, zeros, iters)
        tracks[1:], vis[1:] = _sample(f8, v8, m, pts, offset)
        return tracks, vis
    starts = list(range(0, n - WINDOW, STRIDE)) or [0]
    if starts[-1] + WINDOW < n:
        starts.append(starts[-1] + STRIDE)
    pad = starts[-1] + WINDOW - n
    if pad:
        fmaps = torch.cat([fmaps, fmaps[-1:].expand(pad, -1, -1, -1)])
    flows8 = zeros.repeat(n + pad, 1, 1, 1)
    visconfs8 = zeros.repeat(n + pad, 1, 1, 1)
    for i, s in enumerate(starts):
        win = slice(s, s + WINDOW)
        f8, v8, m = net.window(fmaps[0], fmaps[win], flows8[win], visconfs8[win], iters)
        flows8[win], visconfs8[win] = f8, v8
        if s + WINDOW < n + pad:  # the frames after the window start from its last frame (nearest visited)
            flows8[s + WINDOW :], visconfs8[s + WINDOW :] = f8[-1:], v8[-1:]
        keep = min(STRIDE if i + 1 < len(starts) else WINDOW, n - s)  # later windows overwrite the rest
        tracks[s : s + keep], vis[s : s + keep] = _sample(f8[:keep], v8[:keep], m[:keep], pts, offset)
    return tracks, vis


def track(  # noqa: PLR0913, PLR0917 - the runner call's arguments plus params and the autocast dtype
    net: AllTracker,
    frames: Callable[[int], torch.Tensor],
    n: int,
    size: tuple[int, int],
    queries: torch.Tensor,
    params: Params,
    amp: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """`{"tracks": [T, Q, 2], "visibility": [T, Q]}` of frames `frames(t)` `[1, 3, H, W]` in `[0, 1]`."""
    h, w = size
    nh, nw = params.height, params.width
    left, right, top, bottom = _pad(nh, nw)
    scale = torch.tensor((nw / w, nh / h), device=queries.device)  # (x, y)
    offset = torch.tensor((left, top), device=queries.device, dtype=torch.float32)
    region = torch.autocast(queries.device.type, amp) if amp is not None else contextlib.nullcontext()
    tracks = queries.new_zeros(n, queries.shape[0], 2)
    vis = queries.new_zeros(n, queries.shape[0])
    with region:
        chunks = []
        for s in range(0, n, FRAME_CHUNK):
            video = torch.cat([frames(t) for t in range(s, min(n, s + FRAME_CHUNK))])
            if video.shape[-2:] != (nh, nw):
                video = F.interpolate(video, (nh, nw), mode="bilinear", align_corners=False)
            chunks.append(net.encode(F.pad(video, (left, right, top, bottom), mode="replicate")))
        fmaps = torch.cat(chunks)
        pts = (queries[:, 1:] + 0.5) * scale - 0.5
        for t0 in queries[:, 0].long().unique().tolist():
            sel = (queries[:, 0] == t0).nonzero()[:, 0]
            tr, vi = _direction(net, fmaps[t0:], pts[sel], offset, params.iters)
            tracks[t0:, sel], vis[t0:, sel] = tr, vi
            if t0 > 0:
                tr, vi = _direction(net, fmaps[: t0 + 1].flip(0), pts[sel], offset, params.iters)
                tracks[:t0, sel], vis[:t0, sel] = tr[1:].flip(0), vi[1:].flip(0)
    return {"tracks": (tracks + 0.5) / scale - 0.5, "visibility": vis.clamp(0, 1) ** VIS_POWER}


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
        """The network with the resizing; `forward(video [T, 3, H, W], queries [Q, 3])`."""
        amp, params = self.amp, self.ctx.params

        def fn(net: AllTracker, video: torch.Tensor, queries: torch.Tensor) -> dict[str, torch.Tensor]:
            size = (video.shape[-2], video.shape[-1])
            return track(net, lambda t: video[t : t + 1], video.shape[0], size, queries, params, amp)

        return Forward(self.net, fn)


RELU, GELU, GELU_TANH = 1, 5, 6  # fixed: fused-epilogue activation codes (fastforks._optimize.conv_ops)
CL = torch.channels_last


def _w(w: torch.Tensor, k: int | None = None, c: int | None = None) -> torch.Tensor:
    """A conv / Linear weight as the fused conv's fp16 channels-last `[K, C, R, S]`, zero-padded to `k` x `c`."""
    w = w.detach().float()
    w = w.reshape(*w.shape, 1, 1) if w.dim() == 2 else w  # noqa: PLR2004 - a Linear weight
    w = F.pad(w, (0, 0, 0, 0, 0, (c or w.shape[1]) - w.shape[1], 0, (k or w.shape[0]) - w.shape[0]))
    return w.half().contiguous(memory_format=CL)


def _b(b: torch.Tensor | None, k: int | None = None) -> torch.Tensor:
    """A bias as fp16, zero-padded to `k`."""
    b = torch.as_tensor(b).detach().float()
    return F.pad(b, (0, (k or b.shape[0]) - b.shape[0])).half()


def _bias(conv: torch.nn.Conv2d) -> torch.Tensor:
    """The bias of a conv that has one."""
    return cast("torch.Tensor", conv.bias)


def _cnblock2d(m: CNBlock2d) -> Callable[[torch.Tensor], torch.Tensor]:
    """`CNBlock2d.forward` on the depthwise and fused-GEMM kernels (NHWC fp16; layer scale folded into the MLP)."""
    from fastforks._optimize.conv_ops import dwconv, lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

    layers = tuple(b for b in m.block if not isinstance(b, (torch.nn.Identity, torch.nn.GELU)))
    conv, ln, fc1, fc2 = cast("tuple[torch.nn.Conv2d, torch.nn.LayerNorm, torch.nn.Linear, torch.nn.Linear]", layers)
    wd, bd = conv.weight.detach()[:, 0].permute(1, 2, 0).half().contiguous(), _b(conv.bias)
    ls = m.layer_scale.detach().view(-1)
    w1, b1 = _w(fc1.weight), _b(fc1.bias)
    w2, b2 = _w(fc2.weight * ls[:, None]), _b(fc2.bias * ls)

    lnw, lnb = ln.weight.detach().half(), ln.bias.detach().half()

    @torch.autocast("cuda", enabled=False)  # fp16 LayerNorm output (float32 statistics)
    def forward(x: torch.Tensor) -> torch.Tensor:
        x = x.half().contiguous(memory_format=CL)
        y = dwconv(x, wd, bd, [1, 1], [3, 3], 0).permute(0, 2, 3, 1)
        y = F.layer_norm(y, ln.normalized_shape, lnw, lnb, ln.eps)
        return lin(lin(y, w1, b1, act=GELU), w2, b2, res=x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    return forward


def _cnblock1d(m: CNBlock1d) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """`CNBlock1d.forward` with the projections and the MLP as fused GEMMs (fp16; one q / k / v GEMM).

    The attention over the 16 frames runs as two batched GEMMs and a softmax (the attention scale folded into the
    q projection): at 16 tokens and head size 16 that is 6x faster than the flash / memory-efficient SDPA kernels.
    """
    from fastforks._optimize.conv_ops import lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

    blk = m.block
    at, mlp, heads = blk.attn, blk.mlp, blk.attn.heads
    scale = (at.to_q.out_features // heads) ** -0.5
    wqkv = _w(torch.cat([at.to_q.weight * scale, at.to_kv.weight]))
    bqkv = _b(torch.cat([at.to_q.bias * scale, at.to_kv.bias]))
    wo, bo = _w(at.to_out.weight), _b(at.to_out.bias)
    w1, b1 = _w(mlp.fc1.weight), _b(mlp.fc1.bias)
    w2, b2 = _w(mlp.fc2.weight), _b(mlp.fc2.bias)
    ls = m.layer_scale.detach().view(-1).half()

    @torch.autocast("cuda", enabled=False)  # fp16 LayerNorm / softmax outputs (float32 statistics)
    def forward(x: torch.Tensor, s: int) -> torch.Tensor:
        bs, c, h, w = x.shape
        t = x.half().reshape(bs // s, s, c, h * w).permute(0, 3, 1, 2).reshape(-1, s, c)  # (b h w) s c
        n = t.shape[0]
        qkv = lin(F.layer_norm(t, (c,), eps=blk.norm1.eps), wqkv, bqkv).view(n, s, 3, heads, -1).permute(2, 0, 3, 1, 4)
        a = ((qkv[0] @ qkv[1].transpose(-1, -2)).softmax(-1) @ qkv[2]).transpose(1, 2).reshape(n, s, c)
        y = lin(a, wo, bo, res=t)
        y = lin(lin(F.layer_norm(y, (c,), eps=blk.norm2.eps), w1, b1, act=GELU_TANH), w2, b2, res=y)
        t = torch.addcmul(t, ls, y)
        return t.view(bs // s, h * w, s, c).permute(0, 2, 3, 1).reshape(bs, c, h, w)

    return forward


class _Update:
    """`UpdateBlock` + flow / visibility heads of one window step on the fused-conv kernels (NHWC fp16).

    The motion encoder's correlation input is padded to a multiple of 8 channels, its flow input to 8; its last conv
    to 128 outputs (2 zero ones); the compressor reads `[flow features, motion, flow, visconf, 4 zeros]` and adds the
    window-constant context term (its context columns times the context features plus its bias), computed once per
    window; the two heads' first convs run as one 512-output conv, their second as one block-diagonal conv.
    """

    def __init__(self, net: AllTracker) -> None:
        """Repacked fp16 weights of `net`'s update block and heads."""
        from fastforks._optimize.conv_ops import conv_fused  # noqa: PLC0415 - loads the fvm_conv kernels

        self.conv = conv_fused
        u: UpdateBlock = net.update_block
        me = u.motion_encoder
        self.cc = me.convc1.in_channels
        self.ccp = -(-self.cc // 8) * 8
        self.c1 = _w(me.convc1.weight, c=self.ccp), _b(me.convc1.bias)
        self.c2 = _w(me.convc2.weight), _b(me.convc2.bias)
        self.f1 = _w(me.convf1.weight, c=8), _b(me.convf1.bias)
        self.f2 = _w(me.convf2.weight), _b(me.convf2.bias)
        self.mc = _w(me.conv.weight, k=DIM), _b(me.conv.bias, DIM)
        wc = u.compressor.weight.detach()  # inputs: flowfeat 128, ctxfeat 128, motion 126, flow 2, visconf 2
        d = DIM
        self.cmp = _w(torch.cat([wc[:, :d], wc[:, 2 * d :]], 1), c=-(-(wc.shape[1] - d) // 8) * 8)
        self.cmp_ctx = _w(wc[:, d : 2 * d]), _b(u.compressor.bias)
        self.attn = [_cnblock1d(b) for b in u.refine if isinstance(b, CNBlock1d)]
        self.conv2d = [_cnblock2d(b) for b in u.refine if isinstance(b, CNBlock2d)]
        self.final = _w(u.final_conv.weight), _b(u.final_conv.bias)
        fh, vh = (cast("list[torch.nn.Conv2d]", list(h)) for h in (net.flow_head, net.visconf_head))
        self.h1 = _w(torch.cat([fh[0].weight, vh[0].weight])), _b(torch.cat([_bias(fh[0]), _bias(vh[0])]))
        w2 = fh[2].weight.new_zeros(8, 4 * d, 3, 3)
        w2[:2, : 2 * d], w2[2:4, 2 * d :] = fh[2].weight, vh[2].weight
        self.h2 = _w(w2), _b(torch.cat([_bias(fh[2]), _bias(vh[2])]), 8)
        uw = cast("list[torch.nn.Conv2d]", list(net.upsample_weight))
        self.u1 = _w(uw[0].weight), _b(uw[0].bias)
        self.u2 = _w(0.25 * uw[2].weight), _b(0.25 * _bias(uw[2]))

    def c(self, x: torch.Tensor, wb: tuple[torch.Tensor, torch.Tensor], act: int = 0) -> torch.Tensor:
        p = wb[0].shape[-1] // 2
        return self.conv(x, wb[0], wb[1], None, [p, p], [1, 1], act)

    def window(  # noqa: PLR0913, PLR0917 - AllTracker.window's arguments
        self, net: AllTracker, anchor: torch.Tensor, fmaps: torch.Tensor, flows8: torch.Tensor, visconfs8: torch.Tensor,
        iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
        """`AllTracker.window` on the kernels."""
        s, _, h, w = fmaps.shape
        corr_fn = CorrPyramid(anchor.half(), fmaps.half())
        ys, xs = torch.meshgrid(
            torch.arange(h, device=anchor.device, dtype=torch.float32),
            torch.arange(w, device=anchor.device, dtype=torch.float32),
            indexing="ij",
        )
        coords1 = torch.stack((xs, ys))[None]
        f1 = anchor.half()[None].expand(s, -1, -1, -1)
        flowfeat, ctxfeat = (f1, f1) if net.tiny else f1.split(DIM, 1)
        ctxfeat = ctxfeat + net.time_embed(s).half()[:, :, None, None]
        ctx = self.c(ctxfeat.contiguous(memory_format=CL), self.cmp_ctx)  # compressor context term + bias
        flowfeat = flowfeat.contiguous(memory_format=CL)
        pad = flows8.new_zeros(s, self.cmp.shape[1] - 2 * DIM - 2, h, w, dtype=torch.float16)
        zeros6 = flows8.new_zeros(s, 6, h, w, dtype=torch.float16)
        for _ in range(iters):
            corr = corr_fn(coords1 + flows8).half()
            corr = F.pad(corr, (0, 0, 0, 0, 0, self.ccp - self.cc))
            fl = flows8.half()
            cor = self.c(self.c(corr, self.c1, RELU), self.c2, RELU)
            flo = self.c(self.c(torch.cat([fl, zeros6], 1), self.f1, RELU), self.f2, RELU)
            mot = self.c(torch.cat([cor, flo], 1), self.mc, RELU)
            x = torch.cat([flowfeat, mot[:, : DIM - 2], fl, visconfs8.half(), pad], 1)
            x = self.conv(x, self.cmp, None, ctx, [0, 0], [1, 1], 0)
            for attn, conv in zip(self.attn, self.conv2d, strict=True):
                x = conv(attn(x, s))
            flowfeat = self.c(x, self.final)
            d = self.c(self.c(flowfeat, self.h1, RELU), self.h2)
            flows8 = flows8 + d[:, :2].float()
            visconfs8 = visconfs8 + d[:, 2:4].float()
        return flows8, visconfs8, self.c(self.c(flowfeat, self.u1, RELU), self.u2)


class Kernels(Portable):
    """The fp16 pass with the ConvNeXt blocks, the update block and the heads on the fused-conv / GEMM kernels."""

    def __init__(self, ctx: Context) -> None:
        """The portable network with kernel forwards (fp16 weights repacked once)."""
        super().__init__(ctx)
        for m in self.net.modules():
            if isinstance(m, CNBlock2d):
                m.forward = _cnblock2d(m)  # ty: ignore[invalid-assignment]  # per-instance override
        update = _Update(self.net)
        net = self.net
        net.window = lambda *a: update.window(net, *a)  # ty: ignore[invalid-assignment]  # per-instance override

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
