"""CoTracker3 runners: weights -> network and the offline / online tracking loops of upstream's predictors.

Frames are resized to 384 x 512 (bilinear, align_corners=True, as `CoTrackerPredictor`) and scaled to `[-1, 1]`; a
query `(t, x, y)` in input pixels is scaled by `(511 / (W - 1), 383 / (H - 1))` and the tracks come back the same way.

- offline (`CoTrackerPredictor` with queries): a 6 x 6 support grid of frame-0 queries is added (dropped from the
  output), the whole video is one window, 6 iterations; a track is set to its query on the query frame and visible
  there. Visible where upstream's visibility > 0.9.
- online (`CoTrackerOnlinePredictor`, as upstream's TAP-Vid evaluation): windows of 16 frames, 8 new frames per step;
  a query's support features come from the window that first contains its frame, and a window starts from the
  previous window's tracks on the 8 shared frames. Visible where visibility x confidence > 0.6.

The returned visibility is upstream's score `p`, rescaled piecewise linearly so that the API's 0.5 is upstream's
threshold (`[0, thr] -> [0, 0.5]`, `[thr, 1] -> [0.5, 1]`).

Same network maths as upstream, different evaluation order (all runners, also float32):
- correlation: the 7 x 7 window of a track point shares one fractional offset, so instead of bilinearly sampling the
  128-d features at 49 sub-pixel points, the enclosing 8 x 8 integer window is gathered (border = index clamp, as
  upstream's border padding), correlated with the query's 49 support features in one batched GEMM and the
  correlations are interpolated (bilinear interpolation is linear in the features). The support features are the
  same gather at the query frame. Queries are processed in chunks (memory), the feature pyramid is kept flattened
  `[T * h * w, 128]` in the compute dtype.
- attention through `scaled_dot_product_attention`; the feature CNN in chunks of `FRAME_CHUNK` frames; online, the 8
  frames two windows share are encoded once.
- float16 (CUDA): the feature CNN (channels-last; InstanceNorm as one `var_mean` + `addcmul`), the feature pyramid,
  the correlation and its MLP and the update transformer run in float16; feature normalisation and pooling, positions,
  visibility and confidence in float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks.tracking._runner import Forward, Runner

from .net.blocks import Attention
from .net.cotracker3_offline import CoTrackerThreeOffline
from .net.cotracker3_online import CoTrackerThreeBase, CoTrackerThreeOnline, posenc
from .net.model_utils import get_points_on_a_grid

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.tracking._runner import Context

    from .net.blocks import AttnBlock
    from .net.cotracker import CrossAttnBlock, EfficientUpdateFormer

SIZE = (384, 512)  # fixed: upstream model_resolution (H, W), the predictor's interp_shape
ITERS = 6  # fixed: upstream predictors' iterations
SUPPORT_GRID = 6  # fixed: the offline predictor's support grid (6 x 6 frame-0 queries)
THRESHOLD = {False: 0.9, True: 0.6}  # fixed: upstream predictors' visibility thresholds (offline, online)
FRAME_CHUNK = 16  # fixed: memory chunking of the feature CNN, not an architecture value
CORR_ROWS = 16384  # fixed: (query, frame) rows per correlation chunk (memory)
GELU, GELU_TANH = 5, 6  # fixed: fused-epilogue activation codes (fastforks._optimize.conv_ops)


@dataclass(frozen=True)
class Params:
    """Variant `params`."""

    online: bool  # sliding windows (CoTrackerThreeOnline) or the whole video (CoTrackerThreeOffline)
    window_len: int  # upstream window_len: the time embedding's length (offline: interpolated to the video)


def build(ctx: Context) -> CoTrackerThreeBase:
    """The variant's network (upstream `build_cotracker(stride=4, corr_radius=3, window_len)`)."""
    p = cast("Params", ctx.params)
    cls = CoTrackerThreeOnline if p.online else CoTrackerThreeOffline
    net = cls(stride=4, corr_radius=3, window_len=p.window_len)
    sd = ctx.state_dict(strip="")
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    return net


def _sdpa(
    self: Attention, x: torch.Tensor, context: torch.Tensor | None = None, attn_bias: None = None
) -> torch.Tensor:
    """`Attention.forward` through `scaled_dot_product_attention` (no mask: upstream's inference passes none)."""
    del attn_bias
    b, n1, c = x.shape
    h = self.heads
    q = self.to_q(x).view(b, n1, h, c // h).transpose(1, 2)
    context = x if context is None else context
    k, v = self.to_kv(context).view(b, context.shape[1], 2, h, c // h).permute(2, 0, 3, 1, 4)
    y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
    return self.to_out(y.transpose(1, 2).reshape(b, n1, c))


def _inorm(x: torch.Tensor) -> torch.Tensor:
    """InstanceNorm2d (no affine, eps 1e-5) of a float16 channels-last map: float32 statistics, one `addcmul`."""
    v, m = torch.var_mean(x, dim=(2, 3), keepdim=True, unbiased=False)
    a = (v.float() + 1e-5).rsqrt()
    return torch.addcmul((-m.float() * a).to(x.dtype), x, a.to(x.dtype))


def _calibrate(p: torch.Tensor, thr: float) -> torch.Tensor:
    """Upstream's score rescaled so that `thr` -> 0.5 (piecewise linear, monotone, [0, 1] -> [0, 1])."""
    return torch.where(p < thr, p * (0.5 / thr), 0.5 + (p - thr) * (0.5 / (1 - thr)))


class Core:
    """The network's tracking maths on one video (batch 1) in the compute dtype; see the module docstring."""

    def __init__(self, net: CoTrackerThreeBase, *, half: bool) -> None:
        """Cast / patch the network for `half` (float16) or float32."""
        self.net, self.dt = net, torch.float16 if half else torch.float32
        self.r = net.corr_radius
        if half:
            net.fnet.half().to(memory_format=torch.channels_last)
            for m in net.fnet.modules():
                if isinstance(m, nn.InstanceNorm2d):
                    m.forward = _inorm  # ty: ignore[invalid-assignment]  # per-instance override
            net.updateformer.half()
        for m in net.updateformer.modules():
            if isinstance(m, Attention):
                m.forward = _sdpa.__get__(m)  # ty: ignore[invalid-assignment]  # per-instance override
        fc1, fc2 = net.corr_mlp.fc1, net.corr_mlp.fc2
        self.k_pad = (-fc1.in_features) % 8 if half else 0  # 2401 -> 2408 (aligned fp16 rows)
        self.c1w = F.pad(fc1.weight.detach(), (0, self.k_pad)).to(self.dt)
        self.c1b, self.c2w, self.c2b = (t.detach().to(self.dt) for t in (fc1.bias, fc2.weight, fc2.bias))
        self.in_pad = 0  # zero features appended to the update transformer's input (kernels: 1110 -> 1112)
        self.mlp: Callable[[torch.Tensor], torch.Tensor] = self._mlp
        self.updateformer: Callable[[torch.Tensor], torch.Tensor] = net.updateformer
        self.temb: dict[int, torch.Tensor] = {}

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.gelu(F.linear(x, self.c1w, self.c1b)), self.c2w, self.c2b)

    def time_embed(self, s: int) -> torch.Tensor:
        """Upstream `interpolate_time_embed` for `s` frames, `[s, 1110 + in_pad]` in the compute dtype (cached)."""
        if s not in self.temb:
            te = cast("torch.Tensor", self.net.time_emb).float()
            if te.shape[1] != s:
                te = F.interpolate(te.permute(0, 2, 1), size=s, mode="linear").permute(0, 2, 1)
            self.temb[s] = F.pad(te[0], (0, self.in_pad)).to(self.dt)
        return self.temb[s]

    def encode(self, video: torch.Tensor) -> list[torch.Tensor]:
        """Feature pyramid of `video` `[t, 3, 384, 512]` in `[0, 1]`: per level `[t, h, w, 128]` (compute dtype)."""
        x = (video * 2 - 1).to(self.dt)
        if self.dt == torch.float16:
            x = x.contiguous(memory_format=torch.channels_last)
        fm = self.net.fnet(x).float()
        fm = fm * fm.square().sum(1, keepdim=True).clamp_min(1e-12).rsqrt()
        levels = [fm]
        for _ in range(self.net.corr_levels - 1):
            levels.append(F.avg_pool2d(levels[-1], 2, stride=2))
        return [f.permute(0, 2, 3, 1).to(self.dt).contiguous() for f in levels]

    def _window(self, feat: torch.Tensor, frame: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The `8 x 8` integer window around `c` `[..., 2]` (x, y) at frames `frame` of `feat` `[t, h, w, D]`.

        Returns the window `[..., 8 (x), 8 (y), D]` (border: index clamp) and the fractional offset `[..., 2]`.
        """
        _, h, w, _ = feat.shape
        c0 = c.floor()
        k = torch.arange(-self.r, self.r + 2, device=c.device)
        xs = (c0[..., 0:1].long() + k).clamp_(0, w - 1)
        ys = (c0[..., 1:2].long() + k).clamp_(0, h - 1)
        idx = (frame * (h * w))[..., None, None] + ys[..., None, :] * w + xs[..., :, None]
        return feat.view(-1, feat.shape[-1])[idx], c - c0

    def _lerp(self, g: torch.Tensor, fr: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Bilinear interpolation of `g` `[..., 8 (x), 8 (y), C]` at offset `fr` `[..., 2]` -> `[..., 7, 7, C]`."""
        n = 2 * self.r + 1
        fr = fr.to(g.dtype)
        gy = torch.lerp(g[..., :, :n, :], g[..., :, 1:, :], fr[..., 1, None, None, None])
        if out is None:
            return torch.lerp(gy[..., :n, :, :], gy[..., 1:, :, :], fr[..., 0, None, None, None])
        return torch.lerp(gy[..., :n, :, :], gy[..., 1:, :, :], fr[..., 0, None, None, None], out=out)

    def support(self, levels: list[torch.Tensor], qf: torch.Tensor, qxy: torch.Tensor) -> list[torch.Tensor]:
        """Per level the queries' 7 x 7 support features `[N, D, 49]` at frames `qf` `[N]`, level-0 points `qxy`."""
        out = []
        for i, f in enumerate(levels):
            g, fr = self._window(f, qf, qxy / 2**i)
            s = self._lerp(g.float(), fr)  # [N, 7 (x), 7 (y), D]
            out.append(s.flatten(1, 2).transpose(1, 2).to(self.dt).contiguous())
        return out

    def _correlation(self, levels: list[torch.Tensor], sup: list[torch.Tensor], cn: torch.Tensor) -> torch.Tensor:
        """Correlation embeddings `[N, S, 4 * 256]` of the track points `cn` `[N, S, 2]` (level 0)."""
        n, s, _ = cn.shape
        rr = (2 * self.r + 1) ** 2
        frame = torch.arange(s, device=cn.device).expand(n, s)
        embs = torch.empty(n, s, len(levels) * self.c2w.shape[0], device=cn.device, dtype=self.dt)
        step = max(1, CORR_ROWS // s)
        for a in range(0, n, step):
            b = min(n, a + step)
            buf = torch.zeros((b - a) * s, rr * rr + self.k_pad, device=cn.device, dtype=self.dt)
            for i, f in enumerate(levels):
                g, fr = self._window(f, frame[a:b], cn[a:b] / 2**i)  # [n, S, 8, 8, D]
                cw = torch.matmul(g.view(b - a, s * 64, -1), sup[i][a:b]).view(b - a, s, 8, 8, rr)
                self._lerp(cw, fr, out=buf[:, : rr * rr].view(b - a, s, 7, 7, rr))
                e = self.mlp(buf)
                embs[a:b, :, i * e.shape[-1] : (i + 1) * e.shape[-1]] = e.view(b - a, s, -1)
        return embs

    def refine(  # noqa: PLR0913 - the window state (levels, support, coords, vis, conf)
        self,
        levels: list[torch.Tensor],
        sup: list[torch.Tensor],
        coords: torch.Tensor,
        vis: torch.Tensor,
        conf: torch.Tensor,
        *,
        iters: int = ITERS,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upstream's iterations on one window: level-0 `coords` `[S, N, 2]`, logits `vis` / `conf` `[S, N]`."""
        s = coords.shape[0]
        net = self.net
        scale = torch.tensor([net.model_resolution[1], net.model_resolution[0]], device=coords.device) / net.stride
        temb = self.time_embed(s)
        for _ in range(iters):
            cn = coords.transpose(0, 1)  # [N, S, 2]
            embs = self._correlation(levels, sup, cn)
            rel_f = F.pad(cn[:, :-1] - cn[:, 1:], (0, 0, 0, 1))
            rel_b = F.pad(cn[:, 1:] - cn[:, :-1], (0, 0, 1, 0))
            pe = posenc(torch.cat([rel_f, rel_b], -1) / scale.repeat(2), min_deg=0, max_deg=10)
            parts = [vis.T[..., None].to(self.dt), conf.T[..., None].to(self.dt), embs, pe.to(self.dt)]
            if self.in_pad:
                parts.append(embs.new_zeros(*embs.shape[:2], self.in_pad))
            x = torch.cat(parts, -1) + temb
            delta = self.updateformer(x[None])[0].float()  # [N, S, 4]
            coords = coords + delta[..., :2].transpose(0, 1)
            vis = vis + delta[..., 2].T
            conf = conf + delta[..., 3].T
        return coords, vis, conf


def _frames(frames: Callable[[int], torch.Tensor], a: int, b: int) -> torch.Tensor:
    """Frames `a..b-1` resized to 384 x 512 as upstream's predictor, `[b - a, 3, 384, 512]` in `[0, 1]`."""
    video = torch.cat([frames(t) for t in range(a, b)])
    if video.shape[-2:] != SIZE:
        video = F.interpolate(video, SIZE, mode="bilinear", align_corners=True)
    return video


def _scale(size: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Input pixels -> network pixels `(x, y)` (upstream: `(interp - 1) / (W - 1)`)."""
    h, w = size
    return torch.tensor([(SIZE[1] - 1) / (w - 1), (SIZE[0] - 1) / (h - 1)], device=device)


def track_offline(
    core: Core, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
) -> dict[str, torch.Tensor]:
    """`CoTrackerPredictor._compute_sparse_tracks` with the support grid (module docstring)."""
    sc = _scale(size, queries.device)
    q = torch.cat([queries[:, :1], queries[:, 1:] * sc], 1)
    grid = get_points_on_a_grid(SUPPORT_GRID, SIZE, device=queries.device)[0]
    q = torch.cat([q, torch.cat([grid.new_zeros(len(grid), 1), grid], 1)])
    chunks = [core.encode(_frames(frames, a, min(n, a + FRAME_CHUNK))) for a in range(0, n, FRAME_CHUNK)]
    levels = [torch.cat(c) for c in zip(*chunks, strict=True)]
    del chunks
    qf, qxy = q[:, 0].long(), q[:, 1:] / core.net.stride
    sup = core.support(levels, qf, qxy)
    coords = qxy[None].expand(n, -1, -1).float()
    zeros = coords.new_zeros(n, len(q))
    coords, vis, _ = core.refine(levels, sup, coords, zeros, zeros)
    nq = len(queries)
    tracks = coords[:, :nq] * core.net.stride / sc
    visibility = _calibrate(vis[:, :nq].sigmoid(), THRESHOLD[False])
    i = torch.arange(nq, device=queries.device)
    tracks[qf[:nq], i] = queries[:, 1:]
    visibility[qf[:nq], i] = 1.0
    return {"tracks": tracks, "visibility": visibility}


def track_online(
    core: Core, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
) -> dict[str, torch.Tensor]:
    """`CoTrackerOnlinePredictor` over the whole video, steps of `window_len // 2` frames (module docstring)."""
    net = core.net
    win, step = net.window_len, net.window_len // 2
    sc = _scale(size, queries.device)
    qf, qxy = queries[:, 0].long(), queries[:, 1:] * sc / net.stride
    nq = len(queries)
    coords_p = qxy.new_zeros(n, nq, 2)
    vis_p, conf_p = qxy.new_zeros(n, nq), qxy.new_zeros(n, nq)
    sup: list[torch.Tensor] = []  # per level [Q, D, 49]; zero until the query's frame was reached
    cache: list[torch.Tensor] = []  # the previous window's levels (its last `step` frames are reused)
    for ind in range(0, max(n - step, 1), step):
        end = min(n, ind + win)
        start = ind + step if ind else ind  # frames before `start` are in the previous window
        new = core.encode(_frames(frames, start, end))
        levels = [torch.cat([c[step:], f]) if ind else f for c, f in zip(cache or new, new, strict=True)]
        cache = levels
        pad = win - (end - ind)
        if pad:  # upstream repeats the last frame
            levels = [torch.cat([f, f[-1:].expand(pad, *f.shape[1:])]) for f in levels]
        # upstream: a query's support features come from the first window that contains its frame
        enter = (qf >= (ind + step if ind else 0)) & (qf < ind + win)
        if enter.any() or not sup:
            new = core.support(levels, (qf - ind).clamp(0, win - 1), qxy)
            old = sup or [torch.zeros_like(s) for s in new]
            sup = [torch.where(enter[:, None, None], s, a) for s, a in zip(new, old, strict=True)]
        coords = qxy[None].expand(win, -1, -1).float()
        vis = coords.new_zeros(win, nq)
        conf = coords.new_zeros(win, nq)
        if ind:  # the previous window's tracks on the shared frames, for queries that started there
            ov = win - step
            prev = coords_p[ind : ind + ov] / net.stride
            copy = (qf < ind + ov)[None]

            def ext(t: torch.Tensor, ov: int = ov) -> torch.Tensor:
                return torch.cat([t, t[-1:].expand(win - ov, *t.shape[1:])])

            coords = torch.where(copy[..., None], ext(prev), coords)
            vis = torch.where(copy, ext(vis_p[ind : ind + ov]), vis)
            conf = torch.where(copy, ext(conf_p[ind : ind + ov]), conf)
        coords, vis, conf = core.refine(levels, sup, coords, vis, conf)
        m = end - ind
        coords_p[ind:end], vis_p[ind:end], conf_p[ind:end] = coords[:m] * net.stride, vis[:m], conf[:m]
    visibility = _calibrate(vis_p.sigmoid() * conf_p.sigmoid(), THRESHOLD[True])
    return {"tracks": coords_p / sc, "visibility": visibility}


def track(
    core: Core, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
) -> dict[str, torch.Tensor]:
    """`{"tracks": [T, Q, 2], "visibility": [T, Q]}` of frames `frames(t)` `[1, 3, H, W]` in `[0, 1]`."""
    fn = track_online if isinstance(core.net, CoTrackerThreeOnline) else track_offline
    return fn(core, frames, n, size, queries)


class Portable(Runner):
    """Plain PyTorch."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        net = build(ctx).eval().requires_grad_(requires_grad=False).to(ctx.device)
        self.core = Core(net, half=ctx.dtype == "float16")

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        return track(self.core, frames, n, size, queries)

    def module(self) -> torch.nn.Module:
        """The network with the resizing; `forward(video [T, 3, H, W], queries [Q, 3])`."""
        core = self.core

        def fn(_: nn.Module, video: torch.Tensor, queries: torch.Tensor) -> dict[str, torch.Tensor]:
            size = (video.shape[-2], video.shape[-1])
            return track(core, lambda t: video[t : t + 1], video.shape[0], size, queries)

        return Forward(core.net, fn)


def _w4(w: torch.Tensor, pad: int = 0) -> torch.Tensor:
    """A Linear weight `[out, in]` (K zero-padded by `pad`) as the fused GEMM's fp16 `[out, in, 1, 1]` filter."""
    w = F.pad(w.detach(), (0, pad)).half()
    return w.reshape(*w.shape, 1, 1).contiguous(memory_format=torch.channels_last)


def _b(*layers: nn.Linear) -> torch.Tensor:
    """The layers' biases, concatenated, in fp16."""
    return torch.cat([cast("torch.Tensor", m.bias).detach() for m in layers]).half()


def _updateformer(uf: EfficientUpdateFormer, in_pad: int) -> Callable[[torch.Tensor], torch.Tensor]:  # noqa: PLR0915 - one closure per block type
    """`EfficientUpdateFormer.forward` (inference: space attention on, no mask) on fused GEMMs in fp16.

    One q/k/v GEMM per self-attention block, the residual adds and the GELU (tanh) in the GEMM epilogues, SDPA.
    """
    from fastforks._optimize.conv_ops import lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

    def mlp(m: nn.Module) -> tuple[torch.Tensor, ...]:
        fc1, fc2 = cast("nn.Linear", m.fc1), cast("nn.Linear", m.fc2)
        return _w4(fc1.weight), _b(fc1), _w4(fc2.weight), _b(fc2)

    def self_block(blk: AttnBlock) -> Callable[[torch.Tensor], torch.Tensor]:
        a = cast("Attention", blk.attn)
        wqkv, bqkv = _w4(torch.cat([a.to_q.weight, a.to_kv.weight])), _b(a.to_q, a.to_kv)
        wo, bo = _w4(a.to_out.weight), _b(a.to_out)
        w1, b1, w2, b2 = mlp(blk.mlp)
        heads, sc = int(a.heads), float(a.scale)

        def fwd(x: torch.Tensor) -> torch.Tensor:
            b, n, c = x.shape
            q, k, v = lin(blk.norm1(x), wqkv, bqkv).view(b, n, 3, heads, c // heads).permute(2, 0, 3, 1, 4)
            y = F.scaled_dot_product_attention(q, k, v, scale=sc).transpose(1, 2).reshape(b, n, c)
            x = lin(y, wo, bo, res=x)
            return lin(lin(blk.norm2(x), w1, b1, act=GELU_TANH), w2, b2, res=x)

        return fwd

    def cross_block(blk: CrossAttnBlock) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        a = blk.cross_attn
        wq, bq, wkv, bkv = _w4(a.to_q.weight), _b(a.to_q), _w4(a.to_kv.weight), _b(a.to_kv)
        wo, bo = _w4(a.to_out.weight), _b(a.to_out)
        w1, b1, w2, b2 = mlp(blk.mlp)
        heads, sc = int(a.heads), float(a.scale)

        def fwd(x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
            b, n, c = x.shape
            q = lin(blk.norm1(x), wq, bq).view(b, n, heads, c // heads).transpose(1, 2)
            kv = lin(blk.norm_context(ctx), wkv, bkv).view(b, ctx.shape[1], 2, heads, c // heads)
            k, v = kv.permute(2, 0, 3, 1, 4)
            y = F.scaled_dot_product_attention(q, k, v, scale=sc).transpose(1, 2).reshape(b, n, c)
            x = lin(y, wo, bo, res=x)
            return lin(lin(blk.norm2(x), w1, b1, act=GELU_TANH), w2, b2, res=x)

        return fwd

    wi, bi = _w4(uf.input_transform.weight, in_pad), _b(uf.input_transform)
    time = [self_block(cast("AttnBlock", b)) for b in uf.time_blocks]
    v2p = [cross_block(cast("CrossAttnBlock", b)) for b in uf.space_virtual2point_blocks]
    virt = [self_block(cast("AttnBlock", b)) for b in uf.space_virtual_blocks]
    p2v = [cross_block(cast("CrossAttnBlock", b)) for b in uf.space_point2virtual_blocks]
    every = len(time) // len(virt)
    vt0, nv = uf.virual_tracks.detach().half(), uf.num_virtual_tracks
    wh = torch.cat([uf.flow_head.weight, uf.vis_conf_head.weight]).detach().half()
    bh = _b(uf.flow_head, uf.vis_conf_head)

    def forward(x: torch.Tensor) -> torch.Tensor:
        b, _, t, _ = x.shape
        tokens = lin(x, wi, bi)
        tokens = torch.cat([tokens, vt0.expand(b, -1, t, -1)], dim=1)
        n = tokens.shape[1]
        j = 0
        for i, blk in enumerate(time):
            tokens = blk(tokens.reshape(b * n, t, -1)).view(b, n, t, -1)
            if i % every == 0:
                st = tokens.permute(0, 2, 1, 3).reshape(b * t, n, -1)
                pt, vt = st[:, : n - nv], st[:, n - nv :]
                vt = virt[j](v2p[j](vt, pt))
                pt = p2v[j](pt, vt)
                tokens = torch.cat([pt, vt], dim=1).view(b, t, n, -1).permute(0, 2, 1, 3)
                j += 1
        return F.linear(tokens[:, : n - nv], wh, bh)

    return forward


class Kernels(Portable):
    """The fp16 portable pass with the correlation MLP and the update transformer on fused-GEMM kernels."""

    def __init__(self, ctx: Context) -> None:
        """The portable network with kernel forwards (fp16 weights captured in closures)."""
        super().__init__(ctx)
        from fastforks._optimize.conv_ops import lin  # noqa: PLC0415 - loads the fvm_conv kernels

        core = self.core
        core.in_pad = (-core.net.input_dim) % 8  # 1110 -> 1112
        core.temb.clear()
        core.updateformer = _updateformer(core.net.updateformer, core.in_pad)
        w1, w2 = _w4(core.c1w), _w4(core.c2w)

        def mlp(x: torch.Tensor) -> torch.Tensor:
            return lin(lin(x, w1, core.c1b, act=GELU), w2, core.c2b)

        core.mlp = mlp

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        with fp16_accumulate():
            return super().__call__(frames, n, size, queries)

    def module(self) -> torch.nn.Module:
        """No exportable module: the transformer runs kernel launches."""
        return Runner.module(self)


RUNNERS = {"portable": Portable, "kernels": Kernels}
