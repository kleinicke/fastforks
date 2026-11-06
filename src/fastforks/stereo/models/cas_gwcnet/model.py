"""Cas-GwcNet runners: weights -> network and the portable / kernels forward passes (two-stage cascade, eval mode)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from fastforks._core.sampling import grid_sample
from fastforks.stereo._runner import Forward, Runner

from .net.gwcnet import CostAggregation, GwcNet
from .net.psmnet import PSMNet
from .regress import upsample_softmax_regress

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context

BATCHNORMS = (nn.BatchNorm2d, nn.BatchNorm3d)
# Cost volumes are stored times VOL_SCALE: Cas-GwcNet-gc's stage-2 features reach ~1e4 and their group-wise products
# ~6e4, at the edge of float16. The factor is folded exactly into the BatchNorm after each first 3D convolution.
VOL_SCALE = 1.0 / 16


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]` + variant `params`)."""

    arch: str  # "gc": group-wise correlation + concatenation volume (`gwcnet`); "c": concatenation only (`gwcnet-c`)
    max_disp: int
    ndisps: list[int]
    disp_interval: list[float]
    base_channels: list[int]
    ns_size: int


def _net(ctx: Context) -> GwcNet | PSMNet:
    p = ctx.params
    if p.arch not in {"gc", "c"}:
        msg = f"unknown Cas-GwcNet arch {p.arch!r}"
        raise ValueError(msg)
    net = (GwcNet if p.arch == "gc" else PSMNet)(
        maxdisp=p.max_disp,
        ndisps=list(p.ndisps),
        disp_interval_pixel=list(p.disp_interval),
        using_ns=True,  # neighbour search, as both checkpoints were trained (scripts/sceneflow.sh)
        ns_size=p.ns_size,
        grad_method="detach",
        cr_base_chs=list(p.base_channels),
    )
    net.load_state_dict(ctx.state_dict(key="model"))
    for agg in _aggregations(net):  # BN(conv(v * k)) == BN(conv(v)) with mean * k, var * k^2, eps * k^2 (linear conv)
        bn = cast("nn.BatchNorm3d", cast("nn.Sequential", agg.dres0[0])[1])
        mean, var = bn.running_mean, bn.running_var
        assert mean is not None  # noqa: S101 - narrows the type
        assert var is not None  # noqa: S101 - narrows the type
        mean.mul_(VOL_SCALE)
        var.mul_(VOL_SCALE**2)
        bn.eps *= VOL_SCALE**2
    return net.eval().to(ctx.device)


def _aggregations(net: GwcNet | PSMNet) -> list[CostAggregation]:
    """The per-stage 3D cost aggregations (`gc` and `c` define the same `CostAggregation`)."""
    return [cast("CostAggregation", m) for m in net.cost_agg]


def fold_batchnorms(module: nn.Module) -> int:
    """Fold every BatchNorm that directly follows a convolution in an `nn.Sequential` into the convolution's weights.

    float16 needs it: the scaled first 3D BatchNorm's eps (1e-5 / 256) underflows in float16, and deep 2D features
    nearly cancel against their running means (our kernels fold the same way). Plain PyTorch, exportable. Returns
    the number of folded layers.
    """
    n = 0
    for seq in module.modules():
        if not isinstance(seq, nn.Sequential):
            continue
        for i in range(len(seq) - 1):
            conv, bn = seq[i], seq[i + 1]
            if isinstance(conv, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)) and isinstance(bn, BATCHNORMS):
                seq[i] = fuse_conv_bn_eval(conv, bn, transpose=isinstance(conv, nn.ConvTranspose3d))
                seq[i + 1] = nn.Identity()
                n += 1
    return n


def cost_volume(
    fl: torch.Tensor | dict[str, torch.Tensor],
    fr: torch.Tensor | dict[str, torch.Tensor],
    disps: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """Upstream's `GetCostVolume` times `VOL_SCALE`, built one hypothesis plane at a time, in the features' dtype.

    `gc` (dict features): `[group-wise correlation, left concat features, warped right concat features]`, left features
    zeroed where `x < d`; `c` (tensor features): `[left, warped right]`, left not zeroed (as upstream). `disps` float32
    `[B or 1, d, h or 1, w or 1]`. Each plane warps in float32 with one 2D `grid_sample` (the same sampling as
    upstream's `[d * h, w]` grid), so no `[B, C, d, h, w]` warped-feature tensors exist.
    """
    gc = isinstance(fl, dict)
    if isinstance(fl, dict) and isinstance(fr, dict):
        xg, xc = fl["gwc_feature"].float().contiguous(), fl["concat_feature"].float().contiguous()
        y = torch.cat((fr["gwc_feature"], fr["concat_feature"]), 1).float()
        cg, channels = xg.shape[1], groups + 2 * xc.shape[1]
        dtype = fl["gwc_feature"].dtype
    else:
        assert isinstance(fl, torch.Tensor)  # noqa: S101 - narrows the type
        assert isinstance(fr, torch.Tensor)  # noqa: S101 - narrows the type
        xc, y = fl.float().contiguous(), fr.float()
        xg = xc
        cg, channels = 0, 2 * fl.shape[1]
        dtype = fl.dtype
    b, cc, h, w = xc.shape
    d = disps.shape[1]
    volume = xc.new_empty((b, channels, d, h, w), dtype=dtype)
    xs = torch.arange(w, device=xc.device, dtype=torch.float32).view(1, 1, w)
    gy = (torch.arange(h, device=xc.device, dtype=torch.float32) / ((h - 1.0) / 2.0) - 1.0).view(1, h, 1)
    gy = gy.expand(b, h, w)
    for j in range(d):
        dj = disps[:, j].expand(b, h, w)
        gx = (xs - dj) / ((w - 1.0) / 2.0) - 1.0
        warped = grid_sample(y, torch.stack((gx, gy), 3), mode="bilinear", padding_mode="zeros", align_corners=True)
        if gc:
            valid = (xs >= dj).unsqueeze(1).float()  # upstream: x_warped[:, mw < d] = 0
            corr = (xg * valid * warped[:, :cg]).view(b, groups, cg // groups, h, w).mean(2)
            volume[:, :groups, j] = corr * VOL_SCALE
            volume[:, groups : groups + cc, j] = xc * valid * VOL_SCALE
            volume[:, groups + cc :, j] = warped[:, cg:] * VOL_SCALE
        else:
            volume[:, :cc, j] = xc * VOL_SCALE
            volume[:, cc:, j] = warped * VOL_SCALE
    return volume


def _first_hypotheses(net: GwcNet | PSMNet, nd: int, scale: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Stage-1 hypotheses (upstream `get_disp_range_samples(None)`), float32.

    The same `nd` disparities at every pixel, `[1, nd, 1, 1]`, and their trilinear downsampling to the volume's
    `nd / scale` planes (spatially constant: a 1D interpolation).
    """
    interval = net.disp_interval_pixel[0]
    lo = max(0.0, -nd / 2 * interval)
    hi = min(lo + (nd - 1) * interval, float(net.maxdisp))
    full = lo + torch.arange(nd, device=device, dtype=torch.float32).view(1, nd, 1, 1) * ((hi - lo) / (nd - 1))
    low = F.interpolate((full / scale).view(1, 1, nd, 1, 1), [nd // scale, 1, 1], mode="trilinear", align_corners=False)
    return full, low.view(1, nd // scale, 1, 1)


def _next_hypotheses(
    net: GwcNet | PSMNet, pred: torch.Tensor, stage: int, scale: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Later-stage hypotheses (upstream `get_cur_disp_range_samples(using_ns=True)`), float32.

    The range spans the min / max of the previous disparity in a neighbourhood at 1/4 resolution; returns the
    hypotheses `[B, nd, H, W]` and their downsampling to the volume.
    """
    _, h, w = pred.shape
    nd, ns = net.ndisps[stage], net.ns_size
    n = nd // 4
    s = F.interpolate((pred / 4.0).unsqueeze(1), [h // 4, w // 4], mode="bilinear", align_corners=False).squeeze(1)
    dmin = torch.abs(F.max_pool2d(-s, stride=1, kernel_size=ns, padding=ns // 2))
    dmax = F.max_pool2d(s, stride=1, kernel_size=ns, padding=ns // 2)
    comp = (n * net.disp_interval_pixel[stage] - torch.abs(dmax - dmin)).clamp(min=0) / 2.0
    lo = (dmin - comp).clamp(min=0, max=net.maxdisp)
    hi = (dmax + comp).clamp(min=0, max=net.maxdisp)
    steps = torch.arange(n, device=pred.device, dtype=torch.float32).view(1, n, 1, 1)
    quarter = lo.unsqueeze(1) + steps * ((hi - lo) / (n - 1)).unsqueeze(1)
    full = F.interpolate((quarter * 4.0).unsqueeze(1), [nd, h, w], mode="trilinear", align_corners=False).squeeze(1)
    size = [nd // scale, h // scale, w // scale]
    low = F.interpolate((full / scale).unsqueeze(1), size, mode="trilinear", align_corners=False).squeeze(1)
    return full, low


def forward(net: GwcNet | PSMNet, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Upstream's eval forward (`outputs["stage2"]["pred"]`); disparity `[B, H, W]` float32.

    Inputs are cast to the network's dtype (channels-last on CUDA); hypotheses, warps and regression run in float32.
    """
    p = next(net.parameters())
    fmt = torch.channels_last if p.is_cuda else torch.contiguous_format
    left, right = (x.to(p.dtype).contiguous(memory_format=fmt) for x in (left, right))
    h, w = left.shape[2], left.shape[3]
    fl, fr = net.feature_extraction(left), net.feature_extraction(right)
    groups = getattr(net, "num_groups", [0] * net.num_stage)
    pred = left.new_empty(0)
    for stage, agg in enumerate(_aggregations(net)):
        scale = int(net.stage_infos[f"stage{stage + 1}"]["scale"])
        if stage == 0:
            full, low = _first_hypotheses(net, net.ndisps[0], scale, left.device)
        else:
            full, low = _next_hypotheses(net, pred, stage, scale)
        key = f"stage{stage + 1}"
        volume = cost_volume(fl[key], fr[key], low, groups[stage])
        cost = agg.dres0(volume)
        del volume
        cost = agg.dres1(cost) + cost
        for hourglass in (agg.dres2, agg.dres3, agg.dres4):  # rebinding frees each input volume before the next stage
            cost = hourglass(cost)
        pred = upsample_softmax_regress(agg.classif3(cost), full, h, w)
    return pred


class Portable(Runner):
    """Plain PyTorch, BatchNorms folded: float32 (CPU) or fp16 (CUDA, NHWC features), float32 warps and regression."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        fold_batchnorms(self.net)
        self.net.to(ctx.torch_dtype)
        if ctx.device.type == "cuda":  # NHWC for the 2D feature network (the 3D cost-volume convs stay NCDHW)
            self.net.feature_extraction.to(memory_format=torch.channels_last)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": forward(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, forward)


class Kernels(Runner):
    """Our kernels: fp16-accumulate convolutions, conv + BatchNorm + activation fused into one epilogue, fp16."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from fastforks.shared.cnn_kernels import prepare_generic  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        if not prepare_generic(self.net, ctx.device):
            msg = "kernels: no conv + BatchNorm run was fused (the module tree changed)"
            raise RuntimeError(msg)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": forward(self.net, left, right)}


RUNNERS = {"portable": Portable, "kernels": Kernels}
