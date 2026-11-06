"""LiteAnyStereo V2 runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch

from fastforks.kernels import cudnn_hacc, stereo_ops
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.stereo._runner import Crop, Forward, Runner, crop, pad_widths

from . import kernels as las2_kernels
from .net.liteanystereov2 import LiteAnyStereoV2
from .net.liteanystereov2_H import CombinedGeoEncodingVolume, LiteAnyStereoH, _sample_1d

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule
    from fastforks.shared.cnn_kernels import Patched
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]` + variant `params`)."""

    size: str  # "s", "m", "l" (feed-forward) or "h" (iterative)
    max_disp: int
    iters: int  # LAS2-H update iterations
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes


def _net(ctx: Context) -> LiteAnyStereoV2 | LiteAnyStereoH:
    p = ctx.params
    net = LiteAnyStereoH(max_disp=p.max_disp, valid_iters=p.iters) if p.size == "h" else LiteAnyStereoV2(p.size)
    net.load_state_dict(ctx.state_dict())
    return net.eval().to(ctx.device)


def _forward(
    net: LiteAnyStereoV2 | LiteAnyStereoH, left: torch.Tensor, right: torch.Tensor, max_disp: int
) -> torch.Tensor:
    """Upstream's `test_mode` forward (it normalizes the 0-255 input itself); disparity `[B, H, W]`."""
    return net(left, right, max_disp=max_disp, test_mode=True)[:, 0]


def _pair_features(
    net: LiteAnyStereoV2 | LiteAnyStereoH, pair: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Left images and the left / right feature pyramids of a normalized pair batch (`[2N, 3, H, W]`, left first).

    `fnet` runs per image side: the fused CUTLASS convolutions take one image.
    """
    n = pair.shape[0] // 2
    return pair[:n], net.fnet(pair[:n]), net.fnet(pair[n:])


def forward_fast(net: LiteAnyStereoV2, pair: torch.Tensor, max_disp: int, k: CubinModule) -> torch.Tensor:
    """Feed-forward LAS2 (S / M / L) in fp16 channels-last; softmax, regression and upsampling in float32.

    `pair`: normalized fp16 `[2N, 3, H, W]`, left images first; `k`: the `stereo_ops` kernels.
    """
    left, features_left, features_right = _pair_features(net, pair)
    volume = las2_kernels.corr_volume(k, features_left[0], features_right[0], max_disp // 4)
    cv = net.cost_agg(volume, features_left)
    disp = stereo_ops.soft_argmax(k, cv)  # softmax + disparity_regression
    xspx = net.refine_3(net.refine_2(net.refine_1(features_left[0]), net.stem_2(left)))
    return stereo_ops.spx_upsample(k, disp, xspx, 4.0)  # context_upsample(disp * 4, softmax(xspx))


def gwc_volume_fast(k: CubinModule, left: torch.Tensor, right: torch.Tensor, depth: int, groups: int) -> torch.Tensor:
    """`build_gwc_volume_fast` as one `stereo_ops.corr_volume` per group: no `[B, C, D, H, W]` product temporary.

    fp16 `[B, G, D, H, W]` stored as `[B, H, W, G, D]` (the layout `CombinedGeoEncodingVolume` reshapes it to).
    """
    b, c, h, w = left.shape
    cg = c // groups
    out = torch.empty((b, h, w, groups, depth), device=left.device, dtype=torch.float16).permute(0, 3, 4, 1, 2)
    for g in range(groups):
        sl = slice(g * cg, (g + 1) * cg)
        stereo_ops.corr_volume(k, left[:, sl], right[:, sl], out[:, g], 1.0 / cg)
    return out


def geo_features(
    geo_fn: CombinedGeoEncodingVolume, disp: torch.Tensor, coords: torch.Tensor, dx: torch.Tensor | torch.nn.Module
) -> torch.Tensor:
    """`geo_fn(disp, coords, dx)` as fp16 channels-last: the float32 lookups are copied once, into fp16 slices."""
    b, _, h, w = disp.shape
    dx = torch.as_tensor(dx).to(device=disp.device, dtype=disp.dtype)  # a registered buffer
    parts = []
    for level in range(geo_fn.num_levels):
        scale = 2**level
        d = disp.view(b * h * w, 1, 1, 1) / scale
        parts.append(_sample_1d(geo_fn.geo_volume_pyramid[level], dx + d).view(b, h, w, -1))
        cx = coords.view(b * h * w, 1, 1, 1) / scale - d + dx
        parts.append(_sample_1d(geo_fn.init_corr_pyramid[level], cx).view(b, h, w, -1))
    out = torch.empty((b, h, w, sum(p.shape[-1] for p in parts)), device=disp.device, dtype=torch.float16)
    i = 0
    for p in parts:
        out[..., i : i + p.shape[-1]].copy_(p)
        i += p.shape[-1]
    return out.permute(0, 3, 1, 2)  # [B, C, H, W] channels-last


def forward_fast_h(net: LiteAnyStereoH, pair: torch.Tensor, iters: int, k: CubinModule) -> torch.Tensor:
    """LAS2-H in fp16 channels-last (`pair` as in `forward_fast`; `k`: the `stereo_ops` kernels).

    Geometry / correlation lookup volumes and sampling coordinates stay float32 (fp16 positions are off by 0.1-0.2 px
    at 1/4 resolution), the disparity state is float32 and the upsampling runs once, after the last iteration.
    """
    mf = torch.channels_last
    depth = net.max_disp // 4
    left, features_left, features_right = _pair_features(net, pair)
    stem_2x = net.stem_2(left)
    gwc_volume = gwc_volume_fast(k, features_left[0], features_right[0], depth, net.cv_group)
    cv = net.cost_agg(
        las2_kernels.corr_volume(k, features_left[0], features_right[0], depth), features_left
    )  # group mean
    disp = stereo_ops.soft_argmax(k, cv)  # softmax + disparity_regression
    net_list, inp_list, att = net._prepare_context(features_left)  # noqa: SLF001 - upstream's helper
    geo_fn = CombinedGeoEncodingVolume(
        features_left[0].float(), features_right[0].float(), gwc_volume.float(), num_levels=net.corr_levels
    )
    del gwc_volume
    batch, _, height, width = features_left[0].shape
    coords = torch.arange(width, dtype=torch.float32, device=disp.device).reshape(1, 1, width, 1)
    coords = coords.repeat(batch, height, 1, 1)
    h, mask_feat_4 = net_list[0], None
    for i in range(iters):
        geo_feat = geo_features(geo_fn, disp, coords, net.dx)
        h, mask_feat_4, delta_disp = las2_kernels.update_step(
            net.update_block,
            h,
            inp_list[0],
            geo_feat,
            disp.half().contiguous(memory_format=mf),
            att[0],
            mask=i == iters - 1,
        )
        disp = disp + delta_disp.float()
    xspx = net.spx_gru(net.spx_2_gru(mask_feat_4, stem_2x))
    return stereo_ops.spx_upsample(k, disp, xspx, 4.0)


class Portable(Runner):
    """Plain PyTorch: float32, or fp16 autocast with channels-last weights (norms and softmax stay float32)."""

    deterministic_cudnn = True

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.half = ctx.dtype == "float16"
        if self.half:
            self.net.to(memory_format=torch.channels_last)

    def _run(self, net: LiteAnyStereoV2 | LiteAnyStereoH, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        max_disp = self.ctx.params.max_disp
        if not self.half:
            return _forward(net, left, right, max_disp)
        with torch.autocast(left.device.type, dtype=torch.float16):
            return _forward(
                net,
                left.contiguous(memory_format=torch.channels_last),
                right.contiguous(memory_format=torch.channels_last),
                max_disp,
            )

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


_GRAPH_PIXELS = 1_000_000  # fixed: launch-bound up to 1280x720 (batch x H x W of the raw input), measured
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Runner):
    """Our kernels: fp16 channels-last, fp16-accumulate convolutions, conv + norm + activation fused epilogues.

    A repeated input shape up to `_GRAPH_PIXELS` replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager
    pass is launch-bound there. Input normalisation and padding are one `stereo_ops` kernel inside the pass
    (`io = "raw"`).
    """

    Params = Params
    io = "raw"

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from fastforks.shared.cnn_kernels import (  # noqa: PLC0415 - needs the kernel libraries
            fuse_selective,
            pad_deconv_out,
            prepare_generic,
        )

        self.net = _net(ctx)
        if isinstance(self.net, LiteAnyStereoH):  # the 9 superpixel weights: a 16-channel transposed conv
            net: Patched = self.net
            net.spx_gru = pad_deconv_out(net.spx_gru)
            las2_kernels.prepare_update(net.update_block)
        else:
            self.net.refine_3.block[0] = pad_deconv_out(self.net.refine_3.block[0])
        self.kernels = las2_kernels.prepare(self.net, ctx.device)  # before prepare_generic: keeps fp32 line weights
        prepare_generic(self.net, ctx.device)
        fuse_selective(self.net)  # CUTLASS only for the MLP expansions (+ GELU) and strided convs
        self._graphs = SmallShapeGraphs(
            self._forward,
            pixels=_GRAPH_PIXELS,
            graphs=_GRAPHS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a raw `[0, 1]` pair (`io = "raw"`: padding and scaling run inside the pass)."""
        return {"disparity": self._graphs(left, right)}

    def _forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        h, w = left.shape[-2:]
        pads = pad_widths(h, w, self.ctx.io)
        m: Patched = self.net  # `las2_kernels.prepare` kept the image statistics in float32
        # 0-255 input divided by 255 by the network: the [0, 1] images go in unscaled
        pair = stereo_ops.norm_pad_pair(self.kernels, left, right, m._norm_stats, pads=pads)  # noqa: SLF001
        if isinstance(self.net, LiteAnyStereoH):
            disp = forward_fast_h(self.net, pair, self.ctx.params.iters, self.kernels)
        else:
            disp = forward_fast(self.net, pair, self.ctx.params.max_disp, self.kernels)
        return crop(disp, Crop(pads[2], pads[0], h, w))


class MetalKernels(Portable):
    """Apple MPS: the portable network with the correlation / group-wise correlation volume as one Metal kernel."""

    module = Runner.module  # the Metal kernel is not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network and switch its cost volume to the Metal kernel."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the kernel library
        from fastforks.kernels.stereo_metal.modules import gwc_volume  # noqa: PLC0415

        mps.adapt(self.net)
        lib = stereo_metal.load(ctx.device)
        if isinstance(self.net, LiteAnyStereoH):
            hook = "gwc_volume", partial(gwc_volume, lib)
        else:
            hook = "correlation_volume", lambda left, right, depth: gwc_volume(lib, left, right, depth, 1)[:, 0]
        object.__setattr__(self.net, *hook)  # a plain attribute, read by the forward pass


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
