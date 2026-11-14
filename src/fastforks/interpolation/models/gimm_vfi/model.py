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

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.patch import base_type, override
from fastforks.interpolation._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc

from .net import GIMMVFI_R
from .net.modules.fi_utils import normalize_flow, resize, unnormalize_flow, warp
from .net.modules.softsplat import softsplat

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.interpolation._runner import Context

    from .net.modules.fi_components import NewMultiFlowDecoder
    from .net.raft.raft import RAFT

LEVELS, RADIUS = 4, 4  # the correlation pyramids of RAFT and of the bidirectional lookup
HALO = 17  # fixed: rows `convblock` reads beyond a band (17 3 x 3 convs in a chain: 1 + 3 x 5 + 1)
BAND_PIXELS = 1 << 20  # larger full-size inputs run `convblock` in row bands of about this many pixels


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` + `[options]`."""

    iters: int  # RAFT iterations (upstream `raft_iter`)
    ds_factor: float  # scale of the flow estimation (upstream `--ds-factor`)


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
    would promote the whole volume to float32 for `grid_sample`).
    """
    n, _, h, w = vol.shape
    if vol.dtype == torch.float32:
        g = torch.stack((2 * xy[..., 0] / (w - 1) - 1, 2 * xy[..., 1] / (h - 1) - 1), -1)
        return F.grid_sample(vol, g[:, None], align_corners=True)[:, 0, 0]
    flat = vol.reshape(n, h * w)
    x0, y0 = xy[..., 0].floor(), xy[..., 1].floor()
    out = xy.new_zeros(xy.shape[:-1])
    for dx in (0, 1):
        for dy in (0, 1):
            cx, cy = x0 + dx, y0 + dy
            wgt = (1 - (xy[..., 0] - cx).abs()) * (1 - (xy[..., 1] - cy).abs())
            ok = (cx >= 0) & (cx <= w - 1) & (cy >= 0) & (cy <= h - 1)
            idx = (cy.clamp(0, h - 1) * w + cx.clamp(0, w - 1)).long()
            out += torch.where(ok, flat.gather(1, idx).float() * wgt, 0.0)
    return out


class _VolumeCorr:
    """Upstream `CorrBlock`: the all-pairs volume of `f1` against `f2` and its 4-level pyramid, 9 x 9 lookups."""

    def __init__(self, f1: torch.Tensor, f2: torch.Tensor) -> None:
        b, c, h, w = f1.shape
        vol = torch.matmul(f1.reshape(b, c, h * w).transpose(1, 2), f2.reshape(b, c, h * w)) / math.sqrt(c)
        self.shape = (b, h, w)
        self.pyramid = [vol.reshape(b * h * w, 1, h, w)]
        for _ in range(LEVELS - 1):
            self.pyramid.append(F.avg_pool2d(self.pyramid[-1], 2, stride=2))

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
        fwd, bwd = make(f0, f1), make(f1, f0)
        return lambda c0, c1: (fwd(c0), bwd(c1))

    return corr


def _raft(raft: RAFT, im1: torch.Tensor, im2: torch.Tensor, iters: int, make: Callable) -> tuple:
    """Upstream `RAFT.forward(im1, im2, iters, return_feat=True)` -> (flow_up, feats[1:], fmap1)."""
    image1, image2 = 2 * (im1 / 255.0) - 1.0, 2 * (im2 / 255.0) - 1.0
    fmap1, fmap2 = raft.fnet([image1.contiguous(), image2.contiguous()])
    corr_fn = make(fmap1.float(), fmap2.float())
    cnet, feats = raft.cnet(image1, return_feature=True)
    net, inp = torch.split(cnet, [raft.hidden_dim, raft.context_dim], dim=1)
    net, inp = torch.tanh(net), torch.relu(inp)
    coords0, coords1 = raft.initialize_flow(image1)
    up_mask = None
    for _ in range(iters):
        corr = corr_fn(coords1)
        net, up_mask, delta = raft.update_block(net, inp, corr, coords1 - coords0)
        coords1 = coords1 + delta
    return raft.upsample_flow(coords1 - coords0, up_mask), feats[1:], fmap1.float()


def _predict_flow(
    net: GIMMVFI_R, f: torch.Tensor, coord: torch.Tensor, t: torch.Tensor, flows: torch.Tensor
) -> torch.Tensor:
    """Upstream `predict_flow(f, [(coord, None)], [t], flows)[0]` without the coordinate assert: `[B, 2, 1, H, W]`."""
    flow01, flow10 = flows[:, :, 0], flows[:, :, 1]
    weights1, weights2 = net.cal_splatting_weights(flow01, flow10)
    mode = net.fwarp_type + "-zeroeps"
    lat0, lat1 = net.cnn_encoder(f[:, :, 0]), net.cnn_encoder(f[:, :, 1])
    tt = t.reshape(-1, 1, 1, 1)
    s0 = softsplat(tenIn=lat0, tenFlow=flow01 * tt, tenMetric=weights1, strMode=mode)
    s1 = softsplat(tenIn=lat1, tenFlow=flow10 * (1 - tt), tenMetric=weights2, strMode=mode)
    lat = torch.cat([s0, s1], dim=1)
    lat = lat + net.res_conv(torch.cat([lat0, lat1, lat], dim=1))
    out = net.hyponet(coord, modulation_params_dict=None, pixel_latent=lat.permute(0, 2, 3, 1))
    return out.permute(0, 4, 1, 2, 3)


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
    f01, feats0, fnet0 = _raft(net.flow_estimator, 255 * image1, 255 * image2, p.iters, make)
    f10, feats1, fnet1 = _raft(net.flow_estimator, 255 * image2, 255 * image1, p.iters, make)
    corr_fn = _bidir(make)(net.amt_fproj(fnet0), net.amt_fproj(fnet1))
    features0 = [net.amt_second_last_cproj(feats0[0]), net.amt_last_cproj(feats0[1])]
    features1 = [net.amt_second_last_cproj(feats1[0]), net.amt_last_cproj(feats1[1])]
    normal, scalers = normalize_flow(torch.stack([f01, -f10], dim=2))
    flows = torch.stack([f01, f10], dim=2)
    if full is not None:
        h, w = int(full.shape[-2] * p.ds_factor), int(full.shape[-1] * p.ds_factor)  # upstream's upsample_ratio
    flow_t = unnormalize_flow(_predict_flow(net, normal, _coords(b, h, w, t), t, flows), scalers).squeeze(2)
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
) -> tuple[torch.Tensor, ...]:
    """Upstream `NewMultiFlowDecoder.forward` with its full-size input built by one `cat` in the conv dtype.

    Under autocast the first conv casts the float32 `cat` to float16 anyway, so casting the parts first is the same
    input at half the size; `convblock` runs in row bands (`_banded`) on large inputs.
    """
    f0, f1, n = m.upsample(f0), m.upsample(f1), m.num_flows
    flow0, flow1 = 4.0 * resize(flow0, scale_factor=4.0), 4.0 * resize(flow1, scale_factor=4.0)
    ft_, mask = resize(ft_, scale_factor=4.0), resize(mask, scale_factor=4.0)
    parts = (ft_, warp(f0, flow0), warp(f1, flow1), flow0, flow1, mask, img0, img1, warp(img0, flow0))
    parts = (*parts, warp(img1, flow1))
    dev = ft_.device.type
    if torch.is_autocast_enabled(dev):
        parts = tuple(p.to(torch.get_autocast_dtype(dev)) for p in parts)
    out = _banded(m.convblock, torch.cat(parts, 1))
    delta_flow0, delta_flow1, delta_mask, img_res = torch.split(out, [2 * n, 2 * n, n, 3 * n], 1)
    mask = torch.sigmoid(delta_mask + mask.repeat(1, n, 1, 1))
    flow0 = delta_flow0 + flow0.repeat(1, n, 1, 1)
    flow1 = delta_flow1 + flow1.repeat(1, n, 1, 1)
    return flow0, flow1, mask, img_res


def _net(ctx: Context) -> GIMMVFI_R:
    net = GIMMVFI_R(_config())
    sd = ctx.state_dict(key="state_dict")
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    override(net.amt_final_decoder, _decoder)
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

    def _forward(self, image1: torch.Tensor, image2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        fp16 = self.ctx.dtype == "float16"
        with torch.autocast(image1.device.type, torch.float16, enabled=fp16):
            return _interpolate(self.net, image1, image2, t, self.ctx.params, make=self.make).float()

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch."""
        if timestep in (0.0, 1.0):  # upstream's lookup scales 1 / t and 1 / (1 - t) are infinite there
            return {"frame": (image1 if timestep == 0.0 else image2).clone()}
        t = torch.full((image1.shape[0],), timestep, device=image1.device)
        return {"frame": self._forward(image1, image2, t)}

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
    """The conv in float16 with float16 accumulation (`cudnn_hacc`), on the float16 channels-last weight copies."""
    return cudnn_hacc.conv_half_acc(x.half(), conv.w16, conv.b16, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


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
        for m in self.net.amt_final_decoder.modules():
            if isinstance(m, torch.nn.Conv2d) and base_type(m) is torch.nn.Conv2d:
                m.w16 = m.weight.detach().half().contiguous(memory_format=torch.channels_last)
                m.b16 = m.bias.detach().half()  # ty: ignore[unresolved-attribute]  # every decoder conv has a bias
                override(m, _hacc_conv)


RUNNERS = {"portable": Portable, "kernels": Kernels}
