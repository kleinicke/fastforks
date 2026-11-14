"""nnU-Net v2 inference on the GPU: plans, `PlainConvUNet`, preprocessing, sliding window, argmax.

Ported from nnunetv2 2.8.1 (`nnUNetPredictor`, `DefaultPreprocessor`, Apache-2.0) and dynamic_network_architectures
0.4.4 (`PlainConvUNet`, Apache-2.0); the network keeps upstream's state-dict keys. Differences, all on purpose:

* Everything runs on the model's device (upstream: numpy / scipy on the CPU between the network calls).
* The `[classes, D, H, W]` logits are never held in full when no resampling back is needed: windows are accumulated
  per step along the first axis and every slab is argmax'ed as soon as no later window touches it (upstream: the full
  float16 logits on the device, then copied to the CPU). Same sums in the same order, so the same labels.
* Folds are summed per window (upstream: per volume); the mean over folds is the same.
* Resampling of the image to the target spacing is trilinear (upstream: cubic spline, z separately for anisotropic
  spacings) and of the logits back to the input grid trilinear (upstream: linear, the same). `use_mask_for_norm` uses
  the nonzero mask without hole filling. Both only matter when the input spacing differs from the plans'.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import WeightsError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = ["Norm", "PlainConvUNet", "Plan", "Predictor", "gaussian", "read_plan", "sliding_steps"]

_ARCH = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
_NORMS = (
    "CTNormalization",
    "ZScoreNormalization",
    "NoNormalization",
    "RescaleTo01Normalization",
    "RGBTo0_1Normalization",
)


@dataclass(frozen=True)
class Norm:
    """Intensity normalisation of one input channel (nnU-Net's `normalization_schemes` + intensity properties)."""

    scheme: str
    use_mask: bool = False
    mean: float = 0.0
    std: float = 1.0
    lower: float = 0.0  # percentile_00_5
    upper: float = 0.0  # percentile_99_5


@dataclass(frozen=True)
class Plan:
    """What inference needs from `plans.json` + `dataset.json` (one configuration, e.g. `3d_fullres`)."""

    patch_size: tuple[int, ...]
    spacing: tuple[float, ...]
    norms: tuple[Norm, ...]
    features: tuple[int, ...]
    kernel_sizes: tuple[tuple[int, ...], ...]
    strides: tuple[tuple[int, ...], ...]
    n_conv_encoder: tuple[int, ...]
    n_conv_decoder: tuple[int, ...]
    labels: Mapping[int, str]  # foreground label -> name
    transpose_forward: tuple[int, ...] = (0, 1, 2)
    transpose_backward: tuple[int, ...] = (0, 1, 2)
    conv_bias: bool = True
    norm_eps: float = 1e-5
    slope: float = 0.01
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_outputs(self) -> int:
        """Segmentation heads: background + the labels."""
        return len(self.labels) + 1


def _ints(v: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(i) for i in v)


def _network(conf: Mapping[str, Any]) -> tuple[Any, ...]:
    """features, kernels, strides, encoder / decoder convs per stage, conv bias, norm eps, LeakyReLU slope."""
    if "architecture" in conf:
        arch = conf["architecture"]
        if arch.get("network_class_name") != _ARCH:
            msg = f"nnU-Net network {arch.get('network_class_name')!r}: only {_ARCH} is supported"
            raise WeightsError(msg)
        a = arch["arch_kwargs"]
        features, kernels, strides = _ints(a["features_per_stage"]), a["kernel_sizes"], a["strides"]
        enc, dec, bias = a["n_conv_per_stage"], a["n_conv_per_stage_decoder"], bool(a.get("conv_bias", True))
        eps = float(a.get("norm_op_kwargs", {}).get("eps", 1e-5))
        slope = float((a.get("nonlin_kwargs") or {}).get("negative_slope", 0.01))
        if "InstanceNorm" not in str(a.get("norm_op")) or "LeakyReLU" not in str(a.get("nonlin")):
            msg = f"nnU-Net PlainConvUNet with {a.get('norm_op')} / {a.get('nonlin')}: only InstanceNorm + LeakyReLU"
            raise WeightsError(msg)
    elif conf.get("UNet_class_name") == "PlainConvUNet":  # legacy plans (nnU-Net < 2.3): fixed norm / nonlin
        kernels, strides = conf["conv_kernel_sizes"], conf["pool_op_kernel_sizes"]
        base, top = int(conf["UNet_base_num_features"]), int(conf["unet_max_num_features"])
        features = tuple(min(base * 2**i, top) for i in range(len(kernels)))
        enc, dec, bias, eps, slope = (
            conf["n_conv_per_stage_encoder"],
            conf["n_conv_per_stage_decoder"],
            True,
            1e-5,
            0.01,
        )
    else:
        msg = f"nnU-Net network {conf.get('UNet_class_name')!r}: only PlainConvUNet is supported"
        raise WeightsError(msg)
    return features, kernels, strides, enc, dec, bias, eps, slope


def read_plan(plans: Mapping[str, Any], dataset: Mapping[str, Any], configuration: str = "3d_fullres") -> Plan:
    """The `Plan` of one configuration of an nnU-Net v2 export (legacy and current plans formats).

    Raises:
        WeightsError: No such configuration, another network than `PlainConvUNet`, region-based labels, or an
            unknown normalisation.
    """
    confs = plans.get("configurations", {})
    if configuration not in confs:
        msg = f"plans.json has no configuration {configuration!r} (has: {', '.join(confs)})"
        raise WeightsError(msg)
    conf = dict(confs[configuration])
    while "inherits_from" in conf:  # nnU-Net's configuration inheritance: the child's keys win
        parent = dict(confs[conf.pop("inherits_from")])
        conf = {**parent, **conf}
    features, kernels, strides, enc, dec, bias, eps, slope = _network(conf)
    labels = dict(dataset["labels"])
    if any(not isinstance(v, int) for v in labels.values()):
        msg = "region-based nnU-Net labels are not supported"
        raise WeightsError(msg)
    ids = sorted(labels.values())
    if ids != list(range(len(ids))):
        msg = f"nnU-Net labels must be 0 .. n - 1, got {ids}"
        raise WeightsError(msg)
    props = plans.get("foreground_intensity_properties_per_channel", {})
    masks = conf.get("use_mask_for_norm", [False] * len(conf["normalization_schemes"]))
    norms = []
    for c, (scheme, mask) in enumerate(zip(conf["normalization_schemes"], masks, strict=True)):
        if scheme not in _NORMS:
            msg = f"nnU-Net normalisation {scheme!r} is not supported ({', '.join(_NORMS)})"
            raise WeightsError(msg)
        p = props.get(str(c), {})
        norms.append(
            Norm(
                scheme,
                bool(mask),
                float(p.get("mean", 0.0)),
                float(p.get("std", 1.0)),
                float(p.get("percentile_00_5", 0.0)),
                float(p.get("percentile_99_5", 0.0)),
            )
        )
    return Plan(
        patch_size=_ints(conf["patch_size"]),
        spacing=tuple(float(s) for s in conf["spacing"]),
        norms=tuple(norms),
        features=features,
        kernel_sizes=tuple(_ints(k) for k in kernels),
        strides=tuple(_ints(s) for s in strides),
        n_conv_encoder=_ints(enc),
        n_conv_decoder=_ints(dec),
        labels={v: k for k, v in labels.items() if v},
        transpose_forward=_ints(plans.get("transpose_forward", (0, 1, 2))),
        transpose_backward=_ints(plans.get("transpose_backward", (0, 1, 2))),
        conv_bias=bias,
        norm_eps=eps,
        slope=slope,
    )


class _ConvNormAct(nn.Module):
    """`ConvDropoutNormReLU` without dropout: conv, InstanceNorm (affine), LeakyReLU."""

    def __init__(self, cin: int, cout: int, k: tuple[int, ...], stride: tuple[int, ...], plan: Plan) -> None:
        super().__init__()
        pad = tuple((i - 1) // 2 for i in k)
        self.conv = nn.Conv3d(cin, cout, k, stride, pad, bias=plan.conv_bias)  # ty: ignore[invalid-argument-type]
        self.norm = nn.InstanceNorm3d(cout, eps=plan.norm_eps, affine=True)
        self.slope = plan.slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_act(self.conv(x))

    def norm_act(self, y: torch.Tensor) -> torch.Tensor:
        """`leaky_relu(norm(y))` as one scale / shift per channel: `InstanceNorm3d` on `channels_last_3d` is slow."""
        var, mean = torch.var_mean(y, dim=(2, 3, 4), correction=0, keepdim=True)
        w, b = (t.view(1, -1, 1, 1, 1) for t in (self.norm.weight, self.norm.bias))
        scale = w.float() * torch.rsqrt(var.float() + self.norm.eps)
        shift = b.float() - mean.float() * scale
        return F.leaky_relu_(torch.addcmul(shift.to(y.dtype), y, scale.to(y.dtype)), self.slope)


class _Stacked(nn.Module):
    """`StackedConvBlocks`: `n` conv blocks, the first with the stride."""

    def __init__(  # noqa: PLR0913, PLR0917 - StackedConvBlocks' arguments
        self, n: int, cin: int, cout: int, k: tuple[int, ...], stride: tuple[int, ...], plan: Plan
    ) -> None:
        super().__init__()
        ones = (1,) * len(k)
        self.convs = nn.Sequential(
            _ConvNormAct(cin, cout, k, stride, plan), *(_ConvNormAct(cout, cout, k, ones, plan) for _ in range(n - 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class _Encoder(nn.Module):
    def __init__(self, plan: Plan, in_channels: int) -> None:
        super().__init__()
        cins = (in_channels, *plan.features[:-1])
        self.stages = nn.ModuleList(
            nn.Sequential(_Stacked(n, ci, co, k, s, plan))
            for n, ci, co, k, s in zip(
                plan.n_conv_encoder, cins, plan.features, plan.kernel_sizes, plan.strides, strict=True
            )
        )


class _Decoder(nn.Module):
    def __init__(self, plan: Plan) -> None:
        super().__init__()
        f, n = plan.features, len(plan.features)
        self.transpconvs = nn.ModuleList(
            nn.ConvTranspose3d(
                f[-(s + 1)],
                f[-(s + 2)],
                plan.strides[-(s + 1)],  # ty: ignore[invalid-argument-type]
                plan.strides[-(s + 1)],  # ty: ignore[invalid-argument-type]
                bias=plan.conv_bias,
            )
            for s in range(n - 1)
        )
        self.stages = nn.ModuleList(
            _Stacked(plan.n_conv_decoder[s], 2 * f[-(s + 2)], f[-(s + 2)], plan.kernel_sizes[-(s + 2)], (1, 1, 1), plan)
            for s in range(n - 1)
        )
        self.seg_layers = nn.ModuleList(nn.Conv3d(f[-(s + 2)], plan.n_outputs, 1) for s in range(n - 1))


class PlainConvUNet(nn.Module):
    """dynamic_network_architectures' `PlainConvUNet` (3D, no deep supervision), with upstream's state-dict keys."""

    def __init__(self, plan: Plan, in_channels: int) -> None:
        """Build the network of `plan` for `in_channels` input channels."""
        super().__init__()
        if len(plan.patch_size) != 3:  # noqa: PLR2004
            msg = f"only 3D nnU-Net configurations are supported, got a {len(plan.patch_size)}D patch"
            raise WeightsError(msg)
        self.encoder = _Encoder(plan, in_channels)
        self.decoder = _Decoder(plan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits `[B, classes, *patch]` of `[B, C, *patch]`."""
        skips = []
        for stage in self.encoder.stages:
            x = stage(x)
            skips.append(x)
        d = self.decoder
        x = skips[-1]
        for s, (up, stage) in enumerate(zip(d.transpconvs, d.stages, strict=True)):
            x = stage(torch.cat((up(x), skips[-(s + 2)]), 1))
        return d.seg_layers[-1](x)

    @staticmethod
    def state_dict_of(weights: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """An nnU-Net checkpoint's `network_weights` without its aliases (`all_modules`, `decoder.encoder`)."""
        return {
            k.removeprefix("_orig_mod."): v
            for k, v in weights.items()
            if ".all_modules." not in k and not k.removeprefix("_orig_mod.").startswith("decoder.encoder.")
        }


def gaussian(patch: Sequence[int], device: torch.device, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """nnU-Net's importance map: a delta filtered by a Gaussian (sigma = patch / 8, truncated at 4 sigma), max 10."""
    axes = []
    for n in patch:
        sigma, c = n / 8, n // 2
        radius = int(4 * sigma + 0.5)
        x = torch.arange(n, dtype=torch.float64) - c
        w = torch.exp(-0.5 * (torch.arange(-radius, radius + 1, dtype=torch.float64) / sigma) ** 2)
        w /= w.sum()
        axes.append(torch.where(x.abs() <= radius, w[(x + radius).clamp(0, 2 * radius).long()], 0.0))
    g = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    g = (g / (g.max() / 10)).to(device, dtype)
    zero = g == 0
    if zero.any():
        g[zero] = g[~zero].min()
    return g


def sliding_steps(size: Sequence[int], patch: Sequence[int], step: float) -> list[list[int]]:
    """nnU-Net's window starts per axis (`compute_steps_for_sliding_window`; `size` >= `patch`)."""
    out = []
    for i, k in zip(size, patch, strict=True):
        n = math.ceil((i - k) / (k * step)) + 1
        actual = (i - k) / (n - 1) if n > 1 else 99999999999
        out.append([round(actual * j) for j in range(n)])
    return out


def _bbox(x: torch.Tensor) -> tuple[slice, ...]:
    """nnU-Net's `crop_to_nonzero` box: the nonzero voxels of any channel (all of the volume if there are none)."""
    nz = (x != 0).any(0)
    out = []
    for d in range(nz.dim()):
        hit = nz.any(dim=tuple(a for a in range(nz.dim()) if a != d)).nonzero()
        out.append(slice(int(hit[0]), int(hit[-1]) + 1) if hit.numel() else slice(0, nz.shape[d]))
    return tuple(out)


def _normalise(x: torch.Tensor, norms: Sequence[Norm], mask: torch.Tensor | None) -> torch.Tensor:
    """nnU-Net's normalisation schemes per channel, in float32."""
    out = torch.empty_like(x, dtype=torch.float32)
    for c, n in enumerate(norms):
        v = x[c].float()
        if n.scheme == "CTNormalization":
            v = v.clamp(n.lower, n.upper).sub_(n.mean).div_(max(n.std, 1e-8))
        elif n.scheme == "ZScoreNormalization":
            if n.use_mask and mask is not None:
                sel = v[mask].double()
                v = torch.where(mask, (v - sel.mean().float()) / max(float(sel.std(correction=0)), 1e-8), v)
            else:
                d = v.double()
                v = (v - d.mean().float()) / max(float(d.std(correction=0)), 1e-8)
        elif n.scheme == "RescaleTo01Normalization":
            v = (v - v.min()) / torch.clamp(v.max() - v.min(), min=1e-8)
        elif n.scheme == "RGBTo0_1Normalization":
            v = v / 255
        out[c] = v
    return out


def _pad(x: torch.Tensor, patch: Sequence[int]) -> tuple[torch.Tensor, tuple[slice, ...]]:
    """nnU-Net's `pad_nd_image` (constant 0, the odd voxel above) and the slices back."""
    pads, keep = [], []
    for n, k in zip(x.shape[1:], patch, strict=True):
        diff = max(k - n, 0)
        lo = diff // 2
        pads.append((lo, diff - lo))
        keep.append(slice(lo, lo + n))
    if any(p != (0, 0) for p in pads):
        x = F.pad(x, [v for p in reversed(pads) for v in p])
    return x, tuple(keep)


def _labels_dtype(n: int) -> torch.dtype:
    return torch.uint8 if n <= 256 else torch.int16  # noqa: PLR2004


def _argmax_resampled(logits: torch.Tensor, shape: Sequence[int], chunk: int = 8) -> torch.Tensor:
    """Argmax of the logits resampled (trilinear) to `shape`, a few classes at a time."""

    def part(c0: int) -> tuple[torch.Tensor, torch.Tensor]:
        v = F.interpolate(logits[None, c0 : c0 + chunk].float(), size=tuple(shape), mode="trilinear")[0]
        return v.max(0)

    best, idx = part(0)
    for c0 in range(chunk, logits.shape[0], chunk):
        m, i = part(c0)
        better = m > best
        best = torch.where(better, m, best)
        idx = torch.where(better, i + c0, idx)
    return idx


@dataclass
class Predictor:
    """nnU-Net v2's `predict_single_npy_array` for one configuration and a list of fold networks.

    Attributes:
        plan: The configuration.
        nets: The fold networks (their logits are averaged).
        step: Window step as a fraction of the patch (nnU-Net: 0.5; TotalSegmentator: 0.8 for its big tasks).
        mirror_axes: Test-time mirroring axes (0..2 of the patch; empty: none).
        batch: Windows per network call.
        amp: Autocast dtype of the network on CUDA (`None`: the networks' own dtype).
        channels_last: Windows in `channels_last_3d` memory format.
        dtype: Cast the windows to this dtype (networks of that dtype, no autocast), or `None`.
        forward: Replaces `net(x)` (kernels runners); gets `(net, x)`.
    """

    plan: Plan
    nets: Sequence[nn.Module]
    step: float = 0.5
    mirror_axes: tuple[int, ...] = ()
    batch: int = 1
    amp: torch.dtype | None = torch.float16
    channels_last: bool = False
    dtype: torch.dtype | None = None
    forward: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None
    _gauss: torch.Tensor = field(default_factory=lambda: torch.ones(()), init=False, repr=False)

    def _net(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return net(x) if self.forward is None else self.forward(net, x)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Mean over folds (and mirrorings) of the window logits `[B, classes, *patch]`."""
        if self.dtype is not None:
            x = x.to(self.dtype)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        amp = self.amp is not None and x.device.type == "cuda"
        with torch.autocast("cuda", self.amp or torch.float16, enabled=amp):
            outs = []
            for net in self.nets:
                y = self._net(net, x)
                if self.mirror_axes:
                    combos = [
                        c for i in range(len(self.mirror_axes)) for c in itertools.combinations(self.mirror_axes, i + 1)
                    ]
                    for axes in combos:
                        dims = [a + 2 for a in axes]
                        y += torch.flip(self._net(net, torch.flip(x, dims)), dims)
                    y /= len(combos) + 1
                outs.append(y)
        out = outs[0]
        for y in outs[1:]:
            out += y
        if len(self.nets) > 1:
            out /= len(self.nets)
        return out

    def preprocess(
        self, volume: torch.Tensor, spacing: Sequence[float]
    ) -> tuple[torch.Tensor, tuple[slice, ...], tuple[int, ...]]:
        """`(normalised, resampled data, crop box, cropped shape)` of a `[C, D, H, W]` volume in nnU-Net axis order."""
        p = self.plan
        x = volume.permute(0, *(t + 1 for t in p.transpose_forward))
        sp = [spacing[t] for t in p.transpose_forward]
        box = _bbox(x)
        x = x[(slice(None), *box)]
        mask = (x != 0).any(0) if any(n.use_mask for n in p.norms) else None
        x = _normalise(x, p.norms, mask)
        shape = tuple(x.shape[1:])
        new = tuple(round(s / t * n) for s, t, n in zip(sp, p.spacing, shape, strict=True))
        if new != shape:
            x = F.interpolate(x[None], size=new, mode="trilinear")[0]
        return x, box, shape

    def __call__(self, volume: torch.Tensor, spacing: Sequence[float]) -> torch.Tensor:
        """Labels `[D, H, W]` (uint8, int16 above 256 classes) of `volume` `[C, D, H, W]` with `spacing` (mm)."""
        p = self.plan
        full = volume.shape[1:]
        x, box, cropped = self.preprocess(volume, spacing)
        resample = tuple(x.shape[1:]) != cropped
        x, keep = _pad(x, p.patch_size)
        logits = self._accumulate(x, full=resample)
        if resample:
            inner = logits[(slice(None), *keep)]
            seg = _argmax_resampled(inner, cropped)
        else:
            seg = logits[keep]
        out = torch.zeros(
            tuple(full[t] for t in p.transpose_forward), dtype=_labels_dtype(p.n_outputs), device=x.device
        )
        out[box] = seg.to(out.dtype)
        return out.permute(*p.transpose_backward)

    def _windows(self, x: torch.Tensor, starts: Sequence[tuple[int, ...]]) -> torch.Tensor:
        """Gaussian-weighted logits of the windows at `starts` in the map dtype (nnU-Net: `prediction *= gaussian`)."""
        k = self.plan.patch_size
        batch = torch.stack([x[:, a : a + k[0], b : b + k[1], c : c + k[2]] for a, b, c in starts])
        y = self._logits(batch).to(self._gauss.dtype)
        return y.mul_(self._gauss)

    def _accumulate(self, x: torch.Tensor, *, full: bool) -> torch.Tensor:
        """Labels `[D, H, W]` of the padded data (or, `full`, the averaged float16 logits `[classes, D, H, W]`)."""
        p, k0 = self.plan, self.plan.patch_size[0]
        self._gauss = (
            gaussian(p.patch_size, x.device)
            if x.device.type == "cuda"
            else gaussian(p.patch_size, x.device, torch.float32)
        )
        steps = sliding_steps(x.shape[1:], p.patch_size, self.step)
        d, h, w = x.shape[1:]
        dt = self._gauss.dtype
        if full:
            acc = torch.zeros((p.n_outputs, d, h, w), dtype=dt, device=x.device)
            n = torch.zeros((d, h, w), dtype=dt, device=x.device)
            for a in steps[0]:
                self._row(x, acc, a, 0, steps, n)
            return acc.div_(n)
        acc = torch.zeros((p.n_outputs, k0, h, w), dtype=dt, device=x.device)
        labels = torch.empty((d, h, w), dtype=_labels_dtype(p.n_outputs), device=x.device)
        for i, a in enumerate(steps[0]):
            self._row(x, acc, a, a, steps)
            end = steps[0][i + 1] if i + 1 < len(steps[0]) else d
            labels[a:end] = acc[:, : end - a].argmax(0).to(labels.dtype)  # final: later rows start at >= end
            if i + 1 < len(steps[0]):
                rest = acc[:, end - a :].clone()
                acc.zero_()
                acc[:, : k0 - (end - a)] = rest
                del rest
        return labels

    def _row(  # noqa: PLR0913, PLR0917 - the sweep state
        self,
        x: torch.Tensor,
        acc: torch.Tensor,
        a: int,
        base: int,
        steps: Sequence[Sequence[int]],
        n: torch.Tensor | None = None,
    ) -> None:
        """Add the weighted logits of every window starting at depth `a` to `acc` (its depth 0 = `base`)."""
        k0, k1, k2 = self.plan.patch_size
        starts = [(a, b, c) for b in steps[1] for c in steps[2]]
        for j in range(0, len(starts), self.batch):
            chunk = starts[j : j + self.batch]
            y = self._windows(x, chunk).to(acc.dtype)
            for yi, (_, b, c) in zip(y, chunk, strict=True):
                acc[:, a - base : a - base + k0, b : b + k1, c : c + k2] += yi
                if n is not None:
                    n[a : a + k0, b : b + k1, c : c + k2] += self._gauss
