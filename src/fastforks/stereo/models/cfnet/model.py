"""CFNet runners: weights -> network and the portable / kernels forward passes.

Inference follows our stereo-zoo fork (`zoo_fast.py`, branch cuda-kernels): the 1/16 and 1/32 cost-volume filters
(`FP32_MODULES`) stay float32 and the three fused cost volumes are built from float32 features (Mish features reach
~2e4: their products and the 3D conv sums overflow float16; the volumes are tiny); the two refinement stages run in row
strips with halos (same values, bounded memory at Middlebury-F); softmax and regression in float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from fastforks.stereo._runner import Forward, Runner

from .net.cfnet import cfnet
from .net.submodule import (
    build_concat_volume,
    build_gwc_volume,
    disparity_regression,
    disparity_variance,
    disparity_variance_confidence,
)

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context

    Features = dict[str, torch.Tensor]

BATCHNORMS = (nn.BatchNorm2d, nn.BatchNorm3d)
FP32_MODULES = ("dres0_5", "dres1_5", "dres0_6", "dres1_6")
STRIP, HALO = 256, 64  # stage rows per strip / halo rows (> the 3D stack's receptive field; multiples of 8)


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]` + variant `params`)."""

    max_disp: int


def _net(ctx: Context) -> cfnet:
    net = cfnet(ctx.params.max_disp, use_concat_volume=True)
    net.load_state_dict(ctx.state_dict(key="model"))
    return net.eval().to(ctx.device)


def fold_batchnorms(module: nn.Module) -> int:
    """Fold every BatchNorm that directly follows a convolution in an `nn.Sequential` into the convolution's weights.

    float16 needs it: the deep 2D features reach ~1e3 before their BatchNorm, whose running mean nearly cancels them;
    in float16 that subtraction loses the result (our kernels fold the same way). Plain PyTorch, exportable.
    Returns the number of folded layers.
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


def _volume(
    fl: dict[str, torch.Tensor], fr: dict[str, torch.Tensor], level: int, disps: int, groups: int
) -> torch.Tensor:
    """Upstream's group-wise correlation + concatenation volume of feature level `level`, from float32 features."""
    gl, gr, cl, cr = (f[k].float() for k in (f"gw{level}", f"concat_feature{level}") for f in (fl, fr))
    return torch.cat((build_gwc_volume(gl, gr, disps, groups), build_concat_volume(cl, cr, disps)), 1)


def _search_range(
    net: cfnet, pred: torch.Tensor, std: torch.Tensor, stage: str, size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream: `pred -/+ ((gamma + 1) * std + beta)`, x2 and bilinearly upsampled to the next stage."""
    g, b = getattr(net, f"gamma_{stage}").float() + 1, getattr(net, f"beta_{stage}").float()
    lo, hi = (pred - g * std - b) * 2, (pred + g * std + b) * 2
    return (
        F.interpolate(lo, size, mode="bilinear", align_corners=True),
        F.interpolate(hi, size, mode="bilinear", align_corners=True),
    )


class StageVolume(Protocol):
    """Builds a refinement stage's volume; `stage_volume` is upstream's."""

    def __call__(  # noqa: PLR0913 - the stage's inputs
        self, net: cfnet, feats: tuple[Features, Features], ds: torch.Tensor, *, level: int, a: int, groups: int
    ) -> torch.Tensor:
        """The volume of rows `[a, a + ds.shape[2])` (see `stage_volume`)."""
        ...


def stage_volume(  # noqa: PLR0913 - the stage's inputs
    net: cfnet, feats: tuple[Features, Features], ds: torch.Tensor, *, level: int, a: int, groups: int
) -> torch.Tensor:
    """Upstream's warped volume of feature level `level`, rows `[a, a + h)`, at disparity samples `ds`.

    `ds`: `[B, D, h, W]` float32. Channels: group-wise correlation (`groups`), concatenation, the samples; returns
    `[B, G + 2 Cc + 1, D, h, W]` in the features' dtype.
    """
    fl, fr = feats
    b = a + ds.shape[2]
    flc, frc = fl[f"concat_feature{level}"][:, :, a:b], fr[f"concat_feature{level}"][:, :, a:b]
    flg, frg = fl[f"gw{level}"][:, :, a:b], fr[f"gw{level}"][:, :, a:b]
    vc, _ = net.cost_volume_generator(flc, frc, ds, "concat")
    vg, ds = net.cost_volume_generator(flg, frg, ds, "gwc", groups)
    # the warp's float32 mask promotes the volumes (as upstream: the correlation runs in float32)
    return torch.cat((vg, vc, ds), dim=1).to(flg.dtype)


def _stage(  # noqa: PLR0913 - one refinement stage of upstream's forward
    net: cfnet,
    stage: str,
    feats: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    lo: torch.Tensor,
    hi: torch.Tensor,
    *,
    with_std: bool,
    volume: StageVolume,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refinement stage `stage` (`s3` at 1/4, `s2` at 1/2): disparity samples -> warped volume -> 3D stack -> softmax.

    All ops are row-local (horizontal warps; the 3D convs see +-HALO rows), so the stage runs in row strips with
    halos; strip starts are multiples of 8, keeping the stride-2 grids of the hourglasses. Returns the disparity and
    its standard deviation (only with `with_std`, else empty), float32 `[B, 1, h, w]`.
    """
    level, scale, count, groups = {
        "s3": (3, 2, net.sample_count_s3, net.num_groups),
        "s2": (2, 1, net.sample_count_s2, net.num_groups // 2),
    }[stage]
    c0, c1, c2, c3, classif = (
        getattr(net, f"{n}_{stage}")
        for n in ("confidence0", "confidence1", "confidence2", "confidence3", "confidence_classif1")
    )
    lo, hi = net.generate_search_range(count + 1, lo, hi, scale=scale)
    rows = lo.shape[2]
    out = torch.empty_like(lo)
    std = torch.empty_like(lo) if with_std else lo.new_empty(0)
    for r0 in range(0, rows, STRIP):
        r1 = min(rows, r0 + STRIP)
        a, b = max(0, r0 - HALO), min(rows, r1 + HALO)
        ds = net.generate_disparity_samples(lo[:, :, a:b], hi[:, :, a:b], count).float()
        cost = c0(volume(net, feats, ds, level=level, a=a, groups=groups))
        cost = c1(cost) + cost
        cost = classif(c3(c2(cost)))
        p = F.softmax(cost.squeeze(1).float(), dim=1)
        pred = torch.sum(p * ds, dim=1, keepdim=True)
        out[:, :, r0:r1] = pred[:, :, r0 - a : r1 - a]
        if with_std:
            std[:, :, r0:r1] = disparity_variance_confidence(p, ds, pred).sqrt()[:, :, r0 - a : r1 - a]
    return out, std


def forward(net: cfnet, left: torch.Tensor, right: torch.Tensor, volume: StageVolume = stage_volume) -> torch.Tensor:
    """Upstream's test forward (`[pred1_s2]`), fused volumes in float32; disparity `[B, H, W]` float32.

    Inputs are cast to the feature network's dtype (channels-last on CUDA); `volume` builds the refinement stages'
    volumes.
    """
    p = next(net.feature_extraction.parameters())
    dt = p.dtype
    fmt = torch.channels_last if p.is_cuda else torch.contiguous_format
    left, right = (x.to(dt).contiguous(memory_format=fmt) for x in (left, right))
    fl, fr = net.feature_extraction(left), net.feature_extraction(right)
    md, g = net.maxdisp, net.num_groups
    c4 = net.dres0(_volume(fl, fr, 4, md // 8, g).to(dt))
    c4 = net.dres1(c4) + c4
    c5 = net.dres0_5(_volume(fl, fr, 5, md // 16, g))
    c5 = net.dres1_5(c5) + c5
    c6 = net.dres0_6(_volume(fl, fr, 6, md // 32, g))
    c6 = net.dres1_6(c6) + c6
    cost = net.classif2(net.dres3(net.combine1(c4, c5.to(dt), c6.to(dt))))
    del c4, c5, c6
    prob = F.softmax(cost.squeeze(1).float(), dim=1)
    pred = disparity_regression(prob, md // 8).unsqueeze(1)
    std = disparity_variance(prob, md // 8, pred).sqrt()
    del prob, cost
    h, w = left.shape[2:]
    lo, hi = _search_range(net, pred, std, "s3", (h // 4, w // 4))
    pred, std = _stage(net, "s3", (fl, fr), lo, hi, with_std=True, volume=volume)
    lo, hi = _search_range(net, pred, std, "s2", (h // 2, w // 2))
    pred, _ = _stage(net, "s2", (fl, fr), lo, hi, with_std=False, volume=volume)
    return F.interpolate(pred * 2, [h, w], mode="bilinear", align_corners=True)[:, 0]


class Portable(Runner):
    """Plain PyTorch, BatchNorms folded: float32 (CPU) or fp16 (1/16 and 1/32 volume filters float32), NHWC (CUDA)."""

    deterministic_cudnn = True

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        fold_batchnorms(self.net)
        self.net.to(ctx.torch_dtype)
        for name in FP32_MODULES:
            getattr(self.net, name).float()
        if ctx.device.type == "cuda":  # NHWC for the 2D feature network (the 3D cost-volume convs stay NCDHW)
            self.net.feature_extraction.to(memory_format=torch.channels_last)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": forward(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, forward)


class Kernels(Runner):
    """Our kernels: fp16-accumulate convs, conv + BatchNorm fused into one epilogue, fp16 (`FP32_MODULES`: fp32)."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from fastforks.shared.cnn_kernels import prepare_generic  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        keep = {name: getattr(self.net, name) for name in FP32_MODULES}
        for name in keep:
            setattr(self.net, name, nn.Identity())
        if not prepare_generic(self.net, ctx.device):
            msg = "kernels: no conv + BatchNorm run was fused (the module tree changed)"
            raise RuntimeError(msg)
        for name, m in keep.items():
            setattr(self.net, name, m)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": forward(self.net, left, right)}


class MetalKernels(Portable):
    """Apple MPS: the portable network with the refinement stages' warped volumes built by one Metal kernel."""

    module = Runner.module  # the Metal kernel is not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network and the Metal stage volume."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the kernel library
        from fastforks.kernels.stereo_metal.modules import WarpVolume  # noqa: PLC0415

        self.volume = WarpVolume(stereo_metal.load(ctx.device))
        mps.adapt(self.net)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": forward(self.net, left, right, self.volume)}


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
