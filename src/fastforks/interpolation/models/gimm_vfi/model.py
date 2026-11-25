"""GIMM-VFI runners: GIMM-VFI-R (`GIMMVFI_R`, RAFT flow) and its `forward` for one time.

As upstream `video_Nx.py`: the two frames in `[0, 1]` (replicate-padded to a multiple of 32, centred), RAFT (20
iterations) estimates the flow both ways at 1/8, GIMM (a SIREN MLP on (t, y, x) conditioned on the forward-splatted
flow latents) predicts the flow at time t, and the AMT-style decoder refines it with the bidirectional correlation of
the RAFT features and blends both warped frames. `ds_factor` (upstream `--ds-factor`, default 1) runs the flow part
at that scale and synthesizes at full size.

`_interpolate` is `GIMMVFI_R.forward` for one time without its host syncs (the coordinate assert, the NaN checks of
softsplat): the coordinates are built on the device from a time tensor, RAFT upsamples only its last flow (upstream
computes every iteration's and keeps the last), and the correlation lookups go through `_VolumeCorr` (portable) or
`_kernel_corr` (kernels). Times 0 and 1 return the input frames (upstream divides by t and 1 - t there).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.mps import release
from fastforks._core.patch import base_type, override
from fastforks._core.sampling import grid_sample
from fastforks._core.tracing import exporting_for_mlx, in_onnx_export
from fastforks.interpolation._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc, ema_warp, gimm_splat
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import GIMMVFI_R
from .net.modules.fi_components import ResBlock
from .net.modules.fi_utils import normalize_flow, resize, unnormalize_flow, warp
from .net.modules.layers import Sine
from .net.modules.softsplat import softsplat

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.interpolation._runner import Context

    from .net.modules.fi_components import NewMultiFlowDecoder
    from .net.modules.hyponet import HypoNet
    from .net.raft.raft import RAFT
    from .net.raft.update import BasicUpdateBlock

LEVELS, RADIUS = 4, 4  # the correlation pyramids of RAFT and of the bidirectional lookup
_REVERSE_MIN_TOKENS = 16_384  # fixed: feature size from which reusing the forward volume pays off on CUDA
_REVERSE_TILE_BYTES = 128 * 1024 * 1024
HALO = 17  # fixed: rows `convblock` reads beyond a band (17 3 x 3 convs in a chain: 1 + 3 x 5 + 1)
BAND_PIXELS = 1 << 20  # larger full-size inputs run `convblock` in row bands of about this many pixels
# fixed: kernels' bands (float16 activations, 256 channels: 512 MiB per full band). Halving them cuts 1080 x 1920's
# peak 4442 -> 3602 MiB and reserved 7110 -> 5474 MiB (each full band left a segment the next sizes do not fit), the
# frame bit-identical, for 1524 -> 1555 ms; portable's peak is elsewhere, so it keeps the larger bands (2614 ms)
KERNELS_BAND_PIXELS = 1 << 19
_FUSION_MIN_PIXELS = 500_000  # fixed: latent size from which the fused float16 path pays off on CUDA
HYPO_PIXELS = 1 << 19  # the hyponet MLP runs over pixel chunks of this size (its activations never exist at once)
# fixed: kernels inputs up to this many pixels (batch x H x W) replay a CUDA graph. They are launch-bound (Windows
# 480 x 854 308 -> 271 ms); 720 x 1280 gains under 2 % while its capture holds about three eager peaks reserved
GRAPH_PIXELS = 1 << 19
_MAX_SPLAT_PIXELS = 1 << 24  # fixed: exact range of the original FP32 flattened pixel indices


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` + `[options]`."""

    iters: int  # RAFT iterations (upstream `raft_iter`)
    ds_factor: float  # scale of the flow estimation (upstream `--ds-factor`)
    cuda_graphs: bool  # kernels: replay the whole call from a CUDA graph for small inputs (`GRAPH_PIXELS`)


def _config() -> SimpleNamespace:
    """configs/gimmvfi/gimmvfi_r_arb.yaml `arch` merged with the `GIMMVFIConfig` / `HypoNetConfig` defaults."""
    ns = SimpleNamespace
    hyponet = ns(
        type="mlp",
        n_layer=5,
        hidden_dim=[128],
        use_bias=True,
        input_dim=3,
        output_dim=2,
        output_bias=0.5,
        activation=ns(type="siren", siren_w0=1.0),
        initialization=ns(weight_init_type="siren", bias_init_type="siren"),
        normalize_weight=True,
        linear_interpo=False,
    )
    return ns(coord_range=[-1.0, 1.0], fwarp_type="linear", hyponet=hyponet)


def _sample(vol: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """`grid_sample(vol, xy)` (bilinear, zeros, align_corners, pixel coordinates `xy` `[N, K, 2]`) -> `[N, K]`.

    float32: upstream `bilinear_sampler`. float16 volumes: the same weights in float32 on gathered corners (autocast
    would promote the whole volume to float32 for `grid_sample`). The weights, bounds and indices are separable: they
    are computed per axis for both neighbours (`[N, K, 2]`) and combined as outer products, and the corners (dx, dy) =
    (0, 0), (0, 1), (1, 0), (1, 1) are summed in that order. These are the numbers of a corner at a time, in about a
    third of the kernel launches and with less memory traffic (a frame runs 168 lookups).
    """
    n, _, h, w = vol.shape
    if vol.dtype == torch.float32:
        g = torch.stack((2 * xy[..., 0] / (w - 1) - 1, 2 * xy[..., 1] / (h - 1) - 1), -1)
        return grid_sample(vol, g[:, None], align_corners=True)[:, 0, 0]
    flat = vol.reshape(n, h * w)
    x, y = xy[..., 0:1], xy[..., 1:2]
    x0, y0 = x.floor(), y.floor()
    cx, cy = torch.cat((x0, x0 + 1), -1), torch.cat((y0, y0 + 1), -1)  # [N, K, 2]: d = 0, 1
    wx, wy = 1 - (x - cx).abs(), 1 - (y - cy).abs()
    okx, oky = (cx >= 0) & (cx <= w - 1), (cy >= 0) & (cy <= h - 1)
    idx = (cy.clamp(0, h - 1)[..., None, :] * w + cx.clamp(0, w - 1)[..., :, None]).long()  # [N, K, dx, dy]
    wgt = wx[..., :, None] * wy[..., None, :]
    ok = okx[..., :, None] & oky[..., None, :]
    v = torch.where(ok, flat.gather(1, idx.view(n, -1)).view(idx.shape).float() * wgt, 0.0).flatten(-2)
    return v[..., 0] + v[..., 1] + v[..., 2] + v[..., 3]


class _VolumeCorr:
    """Upstream `CorrBlock`: the all-pairs volume of `f1` against `f2` and its 4-level pyramid, 9 x 9 lookups."""

    def __init__(self, f1: torch.Tensor, f2: torch.Tensor) -> None:
        b, c, h, w = f1.shape
        vol = torch.matmul(f1.reshape(b, c, h * w).transpose(1, 2), f2.reshape(b, c, h * w)) / math.sqrt(c)
        self.shape = (b, h, w)
        self.pyramid = [vol.reshape(b * h * w, 1, h, w)]
        for _ in range(LEVELS - 1):
            self.pyramid.append(F.avg_pool2d(self.pyramid[-1], 2, stride=2))

    def reverse(self) -> _VolumeCorr:
        """Reuse the transposed finest level for a single batch item."""
        _, h, w = self.shape
        tokens = h * w
        transposed = self.pyramid[0].view(tokens, tokens).T
        base = transposed.view(tokens, 1, h, w)
        # Pack a tile of queries at a time so pooling never copies the whole transposed volume.
        tile = max(1, _REVERSE_TILE_BYTES // (tokens * base.element_size()))
        coarse = base.new_empty((tokens, 1, h // 2, w // 2))
        for start in range(0, tokens, tile):
            block = transposed[start : start + tile].contiguous().view(-1, 1, h, w)
            coarse[start : start + tile].copy_(F.avg_pool2d(block, 2, stride=2))
        reverse = self.__class__.__new__(self.__class__)
        reverse.shape = self.shape
        reverse.pyramid = [base, coarse]
        for _ in range(LEVELS - 2):
            reverse.pyramid.append(F.avg_pool2d(reverse.pyramid[-1], 2, stride=2))
        return reverse

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        b, h, w = self.shape
        r = torch.arange(-RADIUS, RADIUS + 1, device=coords.device, dtype=torch.float32)
        delta = torch.stack(torch.meshgrid(r, r, indexing="ij"), -1).view(1, -1, 2)  # upstream's (dy, dx) as (x, y)
        centre = coords.permute(0, 2, 3, 1).reshape(b * h * w, 1, 2).float()
        out = [_sample(vol, centre / 2**i + delta).view(b, h, w, -1) for i, vol in enumerate(self.pyramid)]
        return torch.cat(out, -1).permute(0, 3, 1, 2).contiguous().float()


def _bidir(make: Callable[[torch.Tensor, torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]) -> Callable:
    """Upstream `BidirCorrBlock(f0, f1)`: its transposed pyramid is the correlation of `f1` against pooled `f0`."""

    def corr(f0: torch.Tensor, f1: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]:
        fwd = make(f0, f1)
        capture = torch.compiler.is_compiling() or torch.jit.is_tracing()
        if (
            isinstance(fwd, _VolumeCorr)
            and f0.device.type in ("cuda", "mps")  # MPS: two 1080 x 1920 pyramids do not fit beside the decoder
            and not capture
            and not torch.is_grad_enabled()
            and f0.shape[0] == 1
            and f0.shape[-2] * f0.shape[-1] >= _REVERSE_MIN_TOKENS
        ):
            bwd = fwd.reverse()
        else:
            bwd = make(f1, f0)
        if isinstance(fwd, _VolumeCorr) and not capture and not torch.is_grad_enabled():
            # Frame synthesis samples this pair once, before the full-resolution decoder.
            volumes = [bwd, fwd]
            return lambda c0, c1: (volumes.pop()(c0), volumes.pop()(c1))
        return lambda c0, c1: (fwd(c0), bwd(c1))

    return corr


def _raft(
    raft: RAFT, image1: torch.Tensor, maps: tuple[torch.Tensor, torch.Tensor], iters: int, make: Callable
) -> tuple:
    """Refine one direction using the already encoded features of both input frames."""
    corr_fn = make(*maps)
    cnet, feats = raft.cnet(image1, return_feature=True)
    net, inp = torch.split(cnet, [raft.hidden_dim, raft.context_dim], dim=1)
    net, inp = torch.tanh(net), torch.relu(inp)
    if getattr(raft.update_block, "_nhwc", False):  # kernels: a channels-last update block
        net, inp = (x.contiguous(memory_format=torch.channels_last) for x in (net, inp))
    coords0, coords1 = raft.initialize_flow(image1)
    up_mask = None
    for step in range(iters):
        corr = corr_fn(coords1)
        net, up_mask, delta = raft.update_block(net, inp, corr, coords1 - coords0, upsample=step + 1 == iters)
        del corr
        coords1 = coords1 + delta
    del corr_fn
    return raft.upsample_flow(coords1 - coords0, up_mask), feats[1:]


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """The dtype a conv would run `x` in: the autocast dtype when autocast is on for its device, else its own."""
    dev = x.device.type
    return torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype


def _update(  # noqa: PLR0913, PLR0917 - upstream's signature
    m: BasicUpdateBlock,
    net: torch.Tensor,
    inp: torch.Tensor,
    corr: torch.Tensor,
    flow: torch.Tensor,
    upsample: bool = True,  # noqa: FBT001, FBT002 - upstream's signature
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Upstream `BasicUpdateBlock.forward`, the GRU input `[h | inp | motion | flow]` in one buffer of the conv dtype.

    Upstream concatenates `inp`, the motion features and the float32 flow into a float32 `x`, then `[h, x]` and
    `[r * h, x]` twice per GRU (float32, 384 channels), which autocast casts to float16 for every one of the six
    convs. Writing the parts into one buffer of the conv dtype and only replacing its `h` slice gives the convs the
    same input values (each cast rounds once either way): bit-identical, without the six float32 copies and casts.
    """
    e, g = m.encoder, m.gru
    nhwc = getattr(m, "_nhwc", False)
    mf = torch.channels_last if nhwc else torch.contiguous_format
    if nhwc:
        flow = flow.contiguous(memory_format=mf)
    cor = F.relu(e.convc2(F.relu(e.convc1(corr))))
    flo = F.relu(e.convf2(F.relu(e.convf1(flow))))
    out = F.relu(e.conv(torch.cat([cor, flo], dim=1)))
    del cor, flo
    hd, ci, co = net.shape[1], inp.shape[1], out.shape[1]
    shape = (net.shape[0], hd + ci + co + flow.shape[1], *net.shape[2:])
    buf = torch.empty(shape, dtype=_compute_dtype(net), device=net.device, memory_format=mf)
    buf[:, hd : hd + ci] = inp
    buf[:, hd + ci : hd + ci + co] = out
    buf[:, hd + ci + co :] = flow
    del out
    h = net
    for cz, cr, cq in ((g.convz1, g.convr1, g.convq1), (g.convz2, g.convr2, g.convq2)):
        buf[:, :hd] = h
        z = torch.sigmoid(cz(buf))
        r = torch.sigmoid(cr(buf))
        buf[:, :hd] = r * h
        q = torch.tanh(cq(buf))
        h = (1 - z) * h + z * q
    del buf
    delta_flow = m.flow_head(h)
    mask = 0.25 * m.mask(h).contiguous() if upsample else None  # upsample_flow views it
    return h, mask, delta_flow


def _hyponet(
    hn: HypoNet, coord: torch.Tensor, lat: torch.Tensor, chunk: int = HYPO_PIXELS, *, bias_epilogue: bool = False
) -> torch.Tensor:
    """Upstream `HypoNet.forward(coord, None, pixel_latent=lat.permute(0, 2, 3, 1))` in pixel chunks: `[B, h, w, 2]`.

    The same ops per pixel (the latent resized to the coordinate grid, the SIREN layers as `bmm` on `[hidden, 1]`
    against the normalised `[W; b]`, the output bias), over `chunk` pixels at a time: the `[B, h w, 129]`
    activations of the whole image (1 GB at 1080p) never exist. The parameters are prepared once, not per layer
    call (upstream multiplies them by a modulation of ones), and the resize is skipped when the sizes match
    (bilinear at scale 1 is the identity).

    `bias_epilogue` (kernels) runs the layers as `_hyponet_addmm`.
    """
    b, h, w, _ = coord.shape
    if lat.shape[-2:] != (h, w):
        lat = F.interpolate(lat, size=(h, w), mode="bilinear")
    lat = lat.permute(0, 2, 3, 1).reshape(b, h * w, -1)
    coord = coord.reshape(b, h * w, -1)
    params = []
    for i in range(hn.config.n_layer):
        p = hn.params_dict[f"linear_wb{i}"]  # ty: ignore[not-subscriptable]  # set in __init__
        wgt = F.normalize(p[:-1], dim=0) if hn.normalize_weight else p[:-1]
        params.append(torch.cat([wgt, p[-1:]], 0).unsqueeze(0).expand(b, -1, -1))
    if bias_epilogue:
        return _hyponet_addmm(hn, coord, lat, params, chunk).view(b, h, w, -1)
    outs = []
    for s in range(0, h * w, chunk):
        hidden = torch.cat([lat[:, s : s + chunk], coord[:, s : s + chunk]], -1)
        for i, p in enumerate(params):
            hidden = torch.bmm(torch.cat([hidden, hidden.new_ones(b, hidden.shape[1], 1)], -1), p)
            if i < len(params) - 1:
                hidden = hn.activation(hidden)
        outs.append(hidden + hn.output_bias)
    return torch.cat(outs, 1).view(b, h, w, -1)


def _hyponet_addmm(
    hn: HypoNet, coord: torch.Tensor, lat: torch.Tensor, params: list[torch.Tensor], chunk: int
) -> torch.Tensor:
    """`_hyponet`'s layers as `addmm(b, hidden, W)` over the `[B h w, C]` pixels: `[B, h w, 2]`.

    The GEMM adds the bias row in its epilogue instead of a ones column appended to every activation (a full copy of
    it per layer); a SIREN `sin(w0 x)` takes `w0` into its layer's `W` and `b`, and the last layer takes the output
    bias into its `b`. The same products, summed and scaled in another order.
    """
    b = coord.shape[0]
    siren = base_type(hn.activation) is Sine
    w0 = hn.activation.w0 if siren else 1.0
    layers = []
    for i, p in enumerate(params):
        wgt, bias = p[0, :-1], p[0, -1]
        if i < len(params) - 1:
            wgt, bias = (wgt * w0, bias * w0) if siren else (wgt, bias)
        else:
            bias = bias + hn.output_bias  # a float or an `[out]` tensor
        layers.append((wgt, bias))
    lat, coord = lat.flatten(0, 1), coord.flatten(0, 1)
    outs = []
    for s in range(0, lat.shape[0], chunk):
        hidden = torch.cat([lat[s : s + chunk], coord[s : s + chunk]], -1)
        for i, (wgt, bias) in enumerate(layers):
            hidden = torch.addmm(bias, hidden, wgt)
            if i < len(layers) - 1:
                hidden = hidden.sin_() if siren else hn.activation(hidden)
        outs.append(hidden)
    return torch.cat(outs, 0).view(b, -1, outs[0].shape[-1])


def _predict_flow(  # noqa: PLR0913 - the network, flow and coordinate inputs, and portable fusion policy
    net: GIMMVFI_R, f: torch.Tensor, coord: torch.Tensor, t: torch.Tensor, flows: torch.Tensor, *, portable: bool
) -> torch.Tensor:
    """Upstream `predict_flow(f, [(coord, None)], [t], flows)[0]` without the coordinate assert: `[B, 2, 1, H, W]`.

    Build the residual convolution input in its compute dtype and release its buffers as each layer consumes them
    for large eager portable CUDA inputs. The hyponet runs in pixel chunks (`_hyponet`).
    """
    flow01, flow10 = flows[:, :, 0], flows[:, :, 1]
    weights1, weights2 = net.cal_splatting_weights(flow01, flow10)
    mode = net.fwarp_type + "-zeroeps"
    lat0, lat1 = net.cnn_encoder(f[:, :, 0]), net.cnn_encoder(f[:, :, 1])
    tt = t.reshape(-1, 1, 1, 1)
    if (
        not portable
        and lat0.is_cuda
        and mode == "linear-zeroeps"
        and tt.dtype == torch.float32
        and lat0.shape[-2] * lat0.shape[-1] <= _MAX_SPLAT_PIXELS
        and not (torch.is_grad_enabled() or torch.are_deterministic_algorithms_enabled())
    ):
        lat = gimm_splat.pair((lat0, lat1), (flow01, flow10), (weights1, weights2), tt)
        del weights1, weights2
    else:
        s0 = softsplat(tenIn=lat0, tenFlow=flow01 * tt, tenMetric=weights1, strMode=mode)
        s1 = softsplat(tenIn=lat1, tenFlow=flow10 * (1 - tt), tenMetric=weights2, strMode=mode)
        del weights1, weights2
        lat = torch.cat([s0, s1], dim=1)
        del s0, s1
    release(lat)
    dt = _compute_dtype(lat)
    res = torch.cat([lat0.to(dt), lat1.to(dt), lat.to(dt)], dim=1)
    if (
        portable
        and lat.is_cuda
        and dt == torch.float16
        and not (torch.compiler.is_compiling() or torch.jit.is_tracing() or in_onnx_export() or torch.is_grad_enabled())
        and lat.shape[0] * lat.shape[-2] * lat.shape[-1] > _FUSION_MIN_PIXELS
    ):
        del lat0, lat1
        for layer in net.res_conv:
            res = layer(res)
    else:
        res = net.res_conv(res)
        del lat0, lat1
    lat = lat + res
    del res
    return _hyponet(net.hyponet, coord[:, 0], lat, bias_epilogue=not portable).unsqueeze(1).permute(0, 4, 1, 2, 3)


def _coords(b: int, h: int, w: int, t: torch.Tensor) -> torch.Tensor:
    """Upstream `sample_coord_input(b, (h, w), [t])`: `[B, 1, h, w, 3]` of (t, y, x), pixel centres in [-1, 1]."""
    ys = (0.5 + torch.arange(h, device=t.device)) / h * 2 - 1
    xs = (0.5 + torch.arange(w, device=t.device)) / w * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack(
        (t.reshape(b, 1, 1, 1).expand(b, 1, h, w).float(), *(g.expand(b, 1, h, w) for g in (gy, gx))), -1
    )


def _interpolate(  # noqa: PLR0913 - the network, the pair, the time, the params and the lookup
    net: GIMMVFI_R, image1: torch.Tensor, image2: torch.Tensor, t: torch.Tensor, p: Params, *, make: Callable
) -> torch.Tensor:
    """`GIMMVFI_R.forward(xs, [coords at t], [t], ds_factor)["imgt_pred"][0]`; `t` is `[B]` in (0, 1)."""
    full = None
    if p.ds_factor != 1:
        full = torch.stack((image1, image2), 2)
        image1, image2 = resize(image1, p.ds_factor), resize(image2, p.ds_factor)
    xs = torch.stack((image1, image2), 2)
    b, _, h, w = image1.shape
    raft = net.flow_estimator
    raft_inputs = [(2 * ((255 * image) / 255.0) - 1.0).contiguous() for image in (image1, image2)]
    fnet0, fnet1 = (feature.float() for feature in raft.fnet(raft_inputs))
    f01, feats0 = _raft(raft, raft_inputs[0], (fnet0, fnet1), p.iters, make)
    f10, feats1 = _raft(raft, raft_inputs[1], (fnet1, fnet0), p.iters, make)
    del raft_inputs
    release(fnet0)  # MPS: return RAFT's cached volumes before the bidirectional pair is built
    corr_fn = _bidir(make)(net.amt_fproj(fnet0), net.amt_fproj(fnet1))
    del fnet0, fnet1
    features0 = [net.amt_second_last_cproj(feats0[0]), net.amt_last_cproj(feats0[1])]
    features1 = [net.amt_second_last_cproj(feats1[0]), net.amt_last_cproj(feats1[1])]
    del feats0, feats1
    normal, scalers = normalize_flow(torch.stack([f01, -f10], dim=2))
    flows = torch.stack([f01, f10], dim=2)
    del f01, f10
    release(flows)  # MPS: the full-resolution decoder needs the room the encoders' cache holds
    if full is not None:
        h, w = int(full.shape[-2] * p.ds_factor), int(full.shape[-1] * p.ds_factor)  # upstream's upsample_ratio
    flow_t = _predict_flow(net, normal, _coords(b, h, w, t), t, flows, portable=make is _VolumeCorr)
    flow_t = unnormalize_flow(flow_t, scalers).squeeze(2)
    return net.frame_synthesize(xs, flow_t, features0, features1, corr_fn, t.reshape(-1, 1, 1, 1), full_img=full)[0]


def _banded(block: torch.nn.Module, x: torch.Tensor, band_pixels: int = BAND_PIXELS) -> torch.Tensor:
    """`block(x)` for a chain of `HALO` same-size 3 x 3 convs, in row bands with `HALO` rows of context each.

    Each output row depends on `HALO` rows of input above and below it (zero padding only at the image edges, as in
    one pass), so the bands are the one-pass result; the full-size 256-channel activations never exist at once.
    """
    h, w = x.shape[-2:]
    band = max(band_pixels // w, HALO)
    if h <= band + 2 * HALO:
        return block(x)
    out = []
    for y0 in range(0, h, band):
        lo, hi = max(y0 - HALO, 0), min(y0 + band + HALO, h)
        out.append(block(x[:, :, lo:hi])[:, :, y0 - lo : y0 - lo + min(band, h - y0)])
    return torch.cat(out, 2)


def _resblock(m: ResBlock, x: torch.Tensor) -> torch.Tensor:
    """Upstream `ResBlock.forward` with the side convs written into their channel slice and the residual added in place.

    Same ops, without the two `cat`s of the 256-channel activation and the `x + out` temporary (three full-size
    activations fewer per block).
    """
    s = m.side_channels
    out = m.conv1(x)
    out[:, -s:] = m.conv2(out[:, -s:])
    out = m.conv3(out)
    out[:, -s:] = m.conv4(out[:, -s:])
    out = m.conv5(out)
    return m.prelu(out.add_(x))


def _decoder(  # noqa: PLR0913, PLR0917 - upstream's signature
    m: NewMultiFlowDecoder,
    ft_: torch.Tensor,
    f0: torch.Tensor,
    f1: torch.Tensor,
    flow0: torch.Tensor,
    flow1: torch.Tensor,
    mask: torch.Tensor,
    img0: torch.Tensor,
    img1: torch.Tensor,
    *,
    band_pixels: int = BAND_PIXELS,
    fused_warp: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Upstream `NewMultiFlowDecoder.forward` with its full-size input assembled part by part in the conv dtype.

    Under autocast the first conv casts the float32 `cat` to float16 anyway, so writing each part into its channel
    slice of one float16 buffer (channels-last on CUDA, the layout the convs use) is the same input at a fraction of
    the memory: the parts are made one at a time and dropped. `convblock` runs in row bands (`_banded`).
    """
    n = m.num_flows
    mf = torch.channels_last if ft_.is_cuda else torch.contiguous_format
    flow0, flow1 = 4.0 * resize(flow0, scale_factor=4.0), 4.0 * resize(flow1, scale_factor=4.0)
    # resized in the buffer's layout: the same values, and `put` copies it without a transpose (128 x 1088 x 1920 fp16:
    # 16.6 -> 10.9 ms for both)
    ft_, mask = resize(ft_.contiguous(memory_format=mf), scale_factor=4.0), resize(mask, scale_factor=4.0)
    b, _, h, w = ft_.shape
    dt = _compute_dtype(ft_)
    ch = m.convblock[0][0].in_channels  # ty: ignore[not-subscriptable]  # the first conv of the Sequential
    buf = torch.empty(b, ch, h, w, dtype=dt, device=ft_.device, memory_format=mf)
    c = 0

    def put(part: torch.Tensor) -> None:
        nonlocal c
        buf[:, c : c + part.shape[1]].copy_(part)
        c += part.shape[1]

    put(ft_)
    del ft_
    for f, flow in ((f0, flow0), (f1, flow1)):
        if fused_warp:
            features = m.upsample(f)
            channels = features.shape[1]
            ema_warp.write_rows(features, flow.float(), buf[:, c : c + channels])
            c += channels
            del features
        else:
            put(warp(m.upsample(f), flow))
    del f0, f1
    for part in (flow0, flow1, mask, img0, img1):
        put(part)
    put(warp(img0, flow0))
    put(warp(img1, flow1))
    assert c == buf.shape[1]  # noqa: S101 - the parts fill the conv input
    out = _banded(m.convblock, buf, band_pixels)
    delta_flow0, delta_flow1, delta_mask, img_res = torch.split(out, [2 * n, 2 * n, n, 3 * n], 1)
    mask = torch.sigmoid(delta_mask + mask.repeat(1, n, 1, 1))
    flow0 = delta_flow0 + flow0.repeat(1, n, 1, 1)
    flow1 = delta_flow1 + flow1.repeat(1, n, 1, 1)
    return flow0, flow1, mask, img_res


def _kernels_decoder(m: NewMultiFlowDecoder, *args: torch.Tensor, **kwargs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """`_decoder` in kernels' smaller row bands (`KERNELS_BAND_PIXELS`)."""
    return _decoder(m, *args, **kwargs, band_pixels=KERNELS_BAND_PIXELS, fused_warp=True)


def _net(ctx: Context) -> GIMMVFI_R:
    net = GIMMVFI_R(_config())
    sd = ctx.state_dict(key="state_dict")
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    override(net.amt_final_decoder, _decoder)
    override(net.flow_estimator.update_block, _update)
    for m in net.amt_final_decoder.convblock:
        if isinstance(m, ResBlock):
            override(m, _resblock)
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch in float32, or on CUDA under float16 autocast (the volumes in float16, the lookups in float32)."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.make: Callable = _VolumeCorr
        self.graphs: SmallShapeGraphs | None = None  # kernels only: portable stays the eager reference network

    def _forward(self, image1: torch.Tensor, image2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        fp16 = self.ctx.dtype == "float16" and not exporting_for_mlx()  # MLX casts the graph itself
        with (
            torch.autocast(image1.device.type, torch.float16) if fp16 else contextlib.nullcontext()
        ):  # not `enabled=False`: export wraps it
            return _interpolate(self.net, image1, image2, t, self.ctx.params, make=self.make).float()

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch (the time is a graph input: one graph for every time)."""
        if timestep in (0.0, 1.0):  # upstream's lookup scales 1 / t and 1 / (1 - t) are infinite there
            return {"frame": (image1 if timestep == 0.0 else image2).clone()}
        t = torch.full((image1.shape[0],), timestep, device=image1.device)
        # else `self._forward` as bound now (the MLX port replaces it per instance; the graphs hold the original)
        run = self.graphs if self.graphs is not None and image1.is_cuda else self._forward
        return {"frame": run(image1, image2, t)}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool (kernels)."""
        if self.graphs is not None:
            self.graphs.release()

    def module(self) -> torch.nn.Module:
        """The network; returns the middle frame (time 0.5)."""
        p, make = self.ctx.params, self.make

        def run(net: GIMMVFI_R, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
            t = torch.full((image1.shape[0],), 0.5, device=image1.device)
            return _interpolate(net, image1, image2, t, p, make=make)

        return Forward(self.net, run)


def _kernel_corr(k: CubinModule, f1: torch.Tensor, f2: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """The 4-level 9 x 9 windows of `_VolumeCorr` computed on the fly by `raft_flow`: no all-pairs volume.

    The pyramid average-pools the volume over `f2`'s pixels, which commutes with the dot product, so only the pooled
    `f2` pyramid is kept (fp16); fp32 sums and bilinear weights.
    """
    b, c, h, w = f1.shape
    a = f1.to(torch.float16).permute(0, 2, 3, 1).contiguous()
    pyr, x = [], f2.to(torch.float16)
    for _ in range(LEVELS):
        pyr.append(x.permute(0, 2, 3, 1).contiguous())
        x = F.avg_pool2d(x, 2)
    side = LEVELS * (2 * RADIUS + 1) ** 2

    def lookup(coords: torch.Tensor) -> torch.Tensor:
        xy = coords.float().permute(0, 2, 3, 1).contiguous()
        out = torch.empty(b, h, w, side, dtype=torch.float16, device=coords.device)
        for i in range(b):  # one warp per pixel, 8 warps per block
            k.launch(
                "corr_lookup_c256_r4", (-(-h * w // 8), 1, 1), (256, 1, 1),
                a[i], *(p[i] for p in pyr), h, w, xy[i], out[i], side, 1 / math.sqrt(c),
            )  # fmt: skip
        return out.permute(0, 3, 1, 2).float()

    return lookup


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """The conv in float16 with float16 accumulation (`cudnn_hacc`, the bias in its plan), on float16 copies."""
    geometry = (conv.padding, conv.stride, conv.dilation)
    return cudnn_hacc.conv_half_acc(x.half(), conv.w16, conv.b16, *geometry, fuse_bias=True)  # ty: ignore[invalid-argument-type]


class Kernels(Portable):
    """The portable float16 pass with the correlation windows computed on the fly (`raft_flow`): no volumes.

    The full-size decoder's convs (most of the time) run with float16 accumulation (`cudnn_hacc`), channels-last.
    """

    module = Runner.module  # the lookup is a kernel launch, not an exportable operator

    def __init__(self, ctx: Context) -> None:
        """Build the network and bind the lookup kernel."""
        super().__init__(ctx)
        from fastforks.kernels import raft_flow  # noqa: PLC0415 - needs the kernel library

        k = raft_flow.load(ctx.device)
        self.make = lambda f1, f2: _kernel_corr(k, f1, f2)
        cudnn_hacc.load(ctx.device)
        ema_warp.load(ctx.device)
        gimm_splat.load(ctx.device)
        for m in self.net.amt_final_decoder.modules():
            if isinstance(m, torch.nn.Conv2d) and base_type(m) is torch.nn.Conv2d:
                m.w16 = m.weight.detach().half().contiguous(memory_format=torch.channels_last)
                m.b16 = m.bias.detach().half()  # ty: ignore[unresolved-attribute]  # every decoder conv has a bias
                override(m, _hacc_conv)
        override(self.net.amt_final_decoder, _kernels_decoder)
        update = self.net.flow_estimator.update_block
        update.to(memory_format=torch.channels_last)  # its 2 x 20 steps at 1/8 then skip cuDNN's layout transposes
        update._nhwc = True  # noqa: SLF001 - read by `_raft` and `_update`
        # thousands of small launches (RAFT's 2 x 20 lookups and updates at 1/8): one graph replay for small inputs;
        # captured after the `cudnn_hacc` plans are built, so the graph records the half-compute ones
        self.graphs = SmallShapeGraphs(
            self._forward,
            pixels=GRAPH_PIXELS,
            graphs=1,
            size=_graph_size,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )


def _graph_size(image: torch.Tensor, *_: torch.Tensor) -> int:
    # A previously captured nondeterministic splat must not replay after the PyTorch flag changes.
    if torch.are_deterministic_algorithms_enabled():
        return GRAPH_PIXELS + 1
    return image.shape[0] * image.shape[-2] * image.shape[-1]


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable, attrs=(), methods=("_forward",)),
}
