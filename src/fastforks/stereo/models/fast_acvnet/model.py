"""Fast-ACVNet runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.sampling import grid_sample
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.stereo._runner import Forward, Runner

from .net.fast_acv import Fast_ACVNet
from .net.fast_acv_plus import Fast_ACVNet_plus
from .net.submodule import BasicConv, context_upsample, disparity_regression, disparity_variance, regression_topk

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context

Net = Fast_ACVNet | Fast_ACVNet_plus
TOP_K = 24  # fixed: attention candidates kept per pixel; upstream hard-codes it in both nets, not a checkpoint setting
EPS = 1e-05  # fixed: upstream's L2-norm epsilon in the attention weights


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]` + variant `params`)."""

    plus: bool
    max_disp: int
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes


def _net(ctx: Context) -> Net:
    cls = Fast_ACVNet_plus if ctx.params.plus else Fast_ACVNet
    net = cls(ctx.params.max_disp, att_weights_only=False)
    net.load_state_dict(ctx.state_dict(key="model"))
    net = net.eval().to(ctx.device)
    if not ctx.params.plus:  # float32 copies: the fp16 casts below must not round the variance scale
        p = dict(net.named_parameters())
        net._gb32 = (p["gamma"].detach().float().clone(), p["beta"].detach().float().clone())  # noqa: SLF001  # ty: ignore[invalid-assignment]
    return net


def _normalized(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.float().view(b, groups, c // groups, h, w)
    return x / (torch.norm(x, 2, 2, keepdim=True) + EPS)


def norm_correlation_volume(fl: torch.Tensor, fr: torch.Tensor, d: int, groups: int) -> torch.Tensor:
    """Upstream `build_gwc_volume_norm` (`groups`) / `build_norm_correlation_volume` (1 group), float32.

    Returns `[B, G, d, H, W]` (`H, W` of the features); same values, but the features are normalized once instead of
    once per disparity.
    """
    xl, xr = _normalized(fl, groups), _normalized(fr, groups)
    b, g, _, h, w = xl.shape
    volume = xl.new_zeros((b, g, d, h, w))
    for i in range(min(d, w)):  # disparities >= width: no overlap, stays zero
        volume[:, :, i, :, i:] = (xl[..., i:] * xr[..., : w - i]).mean(dim=2)
    return volume


def _grid(samples: torch.Tensor) -> torch.Tensor:
    """Upstream `SpatialTransformer_grid`'s sampling grid `[B, N * H, W, 2]` for disparity samples `[B, N, H, W]`."""
    b, n, h, w = samples.shape
    mh, mw = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=samples.device),
        torch.arange(w, dtype=torch.float32, device=samples.device),
        indexing="ij",
    )
    x = (mw - samples) / ((w - 1.0) / 2.0) - 1.0
    y = (mh / ((h - 1.0) / 2.0) - 1.0).expand(b, n, h, w)
    return torch.stack([x, y], dim=4).view(b, n * h, w, 2)


def warp(y: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """Right features `y` `[B, C, H, W]` at the left pixels' disparity samples: float32 `[B, C, N, H, W]`."""
    b, c, h, w = y.shape
    out = grid_sample(y.float(), _grid(samples), mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.view(b, c, samples.shape[1], h, w)


def _features(net: Net, left: torch.Tensor, right: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Both images in one batch through the backbone, the top-down merge and the image stems.

    Returns the batched feature list (1/4 with the 1/4 stem, 1/8 (+ the 1/8 stem for Fast-ACVNet), 1/16, 1/32) and the
    left 1/2 stem.
    """
    b = left.shape[0]
    both = torch.cat((left, right), 0)
    x4, x8, x16, x32 = net.feature(both)
    up = net.feature_up
    x16 = up.deconv32_16(x32, x16)
    x8 = up.deconv16_8(x16, x8)
    x4 = up.conv4(up.deconv8_4(x8, x4))
    stem_2 = net.stem_2(both)
    stem_4 = net.stem_4(stem_2)
    feats = [torch.cat((x4, stem_4), 1), x8, x16, x32]
    if isinstance(net, Fast_ACVNet):
        feats[1] = torch.cat((x8, net.stem_8(stem_4)), 1)
    return feats, stem_2[:b]


def _attention(
    net: Fast_ACVNet, fl: list[torch.Tensor], fr8: torch.Tensor, s4: torch.Tensor, fmt3: torch.memory_format
) -> torch.Tensor:
    """Fast-ACVNet's attention weights, float32 `[B, 1, D, H, W]` at 1/4.

    The 1/8 group-wise attention volume, upsampled to 1/4 and re-weighted by the matching strength of five propagated
    disparity samples.
    """
    b = fr8.shape[0]
    d4 = net.maxdisp // 4
    p = next(net.parameters())
    corr = norm_correlation_volume(fl[1], fr8, net.maxdisp // 8, 12).to(p.dtype).contiguous(memory_format=fmt3)
    cost = net.hourglass_att(net.corr_feature_att_8(net.patch(corr), fl[1]), fl)
    del corr
    h4, w4 = fl[0].shape[2:]
    att = F.interpolate(cost.float(), [d4, h4, w4], mode="trilinear")
    prob = F.softmax(att.squeeze(1), dim=1)
    pred = disparity_regression(prob, d4).unsqueeze(1)
    gamma, beta = net._gb32  # noqa: SLF001  # ty: ignore[not-iterable]  # set by _net
    variance = torch.sigmoid(beta + gamma * disparity_variance(prob, d4, pred))
    del prob
    variance = net.propagation(variance)
    samples = net.propagation(pred)
    left4 = s4[:b].float()
    strength = torch.stack([(left4 * warp(s4[b:], samples[:, i : i + 1])[:, :, 0]).mean(dim=1) for i in range(5)], 1)
    strength = torch.softmax(strength * variance, dim=1)
    return torch.sum(net.propagation_prob(att) * strength.unsqueeze(2), dim=1, keepdim=True)


def forward(net: Net, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Upstream's test forward; disparity `[B, H, W]` float32.

    Inputs are cast to the network's dtype (channels-last on CUDA); correlation volumes, attention probabilities, warps,
    regression and upsampling run in float32.
    """
    p = next(net.parameters())
    cuda = p.is_cuda
    fmt = torch.channels_last if cuda else torch.contiguous_format
    fmt3 = torch.channels_last_3d if getattr(net, "_cl3d", False) else torch.contiguous_format
    left, right = (x.to(p.dtype).contiguous(memory_format=fmt) for x in (left, right))
    b = left.shape[0]
    feats, stem_2x = _features(net, left, right)
    fl = [t[:b] for t in feats]
    s4 = feats[0][:, -48:]  # the 1/4 stem (last 48 channels)
    if isinstance(net, Fast_ACVNet):
        att = _attention(net, fl, feats[1][b:], s4, fmt3)
    else:
        match = net.desc(net.conv(feats[0]))
        corr = norm_correlation_volume(match[:b], match[b:], net.maxdisp // 4, 1)
        del match
        cost = net.corr_feature_att_4(net.corr_stem(corr.to(p.dtype).contiguous(memory_format=fmt3)), fl[0])
        del corr
        att = net.hourglass_att(cost, fl).float()
        del cost
    att_prob = F.softmax(att, dim=2)
    del att
    ind = att_prob.sort(2, descending=True)[1][:, :, :TOP_K].sort(2)[0]
    att_topk = torch.gather(att_prob, 2, ind)
    del att_prob
    samples = ind.squeeze(1).float()
    concat = net.concat_feature(feats[0])
    del feats
    c = concat.shape[1]
    volume = torch.empty(
        (b, 2 * c, TOP_K, *concat.shape[2:]), dtype=p.dtype, device=p.device, memory_format=fmt3
    )  # upstream: att_topk * cat(left repeated over the samples, warped right)
    volume[:, :c] = att_topk * concat[:b].float().unsqueeze(2)
    volume[:, c:] = att_topk * warp(concat[b:], samples)
    del concat, att_topk
    volume = net.concat_feature_att_4(net.concat_stem(volume), fl[0])
    cost = net.hourglass(volume, fl)
    del volume
    spx = F.softmax(net.spx(net.spx_2(net.spx_4(fl[0]), stem_2x)).float(), 1)
    pred = regression_topk(cost.squeeze(1).float(), samples, 2)
    return context_upsample(pred, spx) * 4


class _FastACVRunner(Runner):
    Params = Params


class Portable(_FastACVRunner):
    """Plain PyTorch: float32 (CPU) or pure fp16 weights (CUDA, NHWC 2D convs); float32 volumes and regression."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx).to(ctx.torch_dtype)
        if ctx.device.type == "cuda":
            for m in self.net.modules():  # 4-D weights only: the 3-D convolutions have 5-D ones
                if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    m.to(memory_format=torch.channels_last)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": forward(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, forward)


_GRAPH_PIXELS = 500_000  # fixed: launch-bound up to KITTI 375x1242 (batch x padded H x W), measured
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(_FastACVRunner):
    """Our kernels: fp16-accumulate convolutions, conv + BatchNorm + LeakyReLU fused into one epilogue, fp16 NHWC.

    A repeated input shape up to `_GRAPH_PIXELS` replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager
    pass is launch-bound there.
    """

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from fastforks.shared import cnn_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        cnn_kernels.load(ctx.device)
        for m in self.net.modules():  # BasicConv is conv -> BN -> LeakyReLU(0.01): one fused kernel
            if isinstance(m, BasicConv) and m.use_bn:
                fused = cnn_kernels.FusedConvBN(m.conv, m.bn, cnn_kernels.ACT["leaky"] if m.relu else 0)
                m.fused = fused
                m.forward = fused.forward  # ty: ignore[invalid-assignment]  # per-instance override
        if not cnn_kernels.prepare_generic(self.net, ctx.device):
            msg = "kernels: no conv + BatchNorm run was fused (the module tree changed)"
            raise RuntimeError(msg)
        self.net._cl3d = True  # noqa: SLF001  # ty: ignore[invalid-assignment]  # channels-last 3D volumes
        self._graphs = SmallShapeGraphs(
            lambda left, right: forward(self.net, left, right),
            pixels=_GRAPH_PIXELS,
            graphs=_GRAPHS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._graphs(left, right)}


RUNNERS = {"portable": Portable, "kernels": Kernels}
