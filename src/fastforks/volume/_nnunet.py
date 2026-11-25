"""nnU-Net v2 inference on the GPU: plans, U-Nets, preprocessing, sliding window, labels.

Ported from nnunetv2 2.8.1 (`nnUNetPredictor`, `DefaultPreprocessor`, `LabelManager`, Apache-2.0) and
dynamic_network_architectures 0.4.4 (`PlainConvUNet`, `ResidualEncoderUNet`, Apache-2.0); the networks keep
upstream's state-dict keys. 2D configurations run as 3D networks with depth-1 kernels on one slice per window (the
same sums). Resampling and the nonzero mask are `fastforks.volume._resample`. Differences, all on purpose:

* Everything runs on the model's device (upstream: numpy / scipy on the CPU between the network calls).
* The `[classes, D, H, W]` logits are never held in full when no resampling back is needed (3D): windows are
  accumulated per step along the first axis and every slab is decided as soon as no later window touches it
  (upstream: the full float16 logits on the device, then copied to the CPU). Same sums in the same order.
* Folds are summed per window (upstream: per volume); the mean over folds is the same.
"""

from __future__ import annotations

import itertools
import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import InputError, WeightsError
from fastforks._core.tracing import exporting_for_mlx, in_onnx_export
from fastforks._core.upload import to_device
from fastforks.volume import _resample

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

__all__ = ["Encoder", "Norm", "Plan", "Predictor", "UNet", "gaussian", "read_plan", "sliding_steps"]

_MAX_RESAMPLED_VOXELS = 10**9  # 4 GB as float32: above it a wrong spacing is far likelier than a real scan
_DECODER_TILE_BYTES = 144 << 20  # FP16 concatenated input to the first decoder convolution
_HEAD_TILE_BYTES = 128 << 20  # FP16 output budget per segmentation projection
# fixed: `_ConvNormAct.forward_` convolves depth slabs of about this size, for maps of at least four slabs (224^3, 32
# channels: 20 slices; nnFoundation-CNN at 200^3: 2 % slower, peak 2325 -> 2092 MiB)
_SLAB_BYTES = 64 << 20
_DNA = "dynamic_network_architectures.architectures.unet."
_ARCHS = {_DNA + "PlainConvUNet": "plain", _DNA + "ResidualEncoderUNet": "resenc"}
_NORMS = (
    "CTNormalization",
    "ZScoreNormalization",
    "NoNormalization",
    "RescaleTo01Normalization",
    "RGBTo0_1Normalization",
)
_2D = 2


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
    """What inference needs from `plans.json` + `dataset.json` (one configuration, e.g. `3d_fullres`).

    2D configurations are stored lifted to 3D: patch `(1, y, x)`, kernels and strides `(1, k, k)`, spacing
    `(None, y, x)` (the first axis keeps the input's spacing).
    """

    patch_size: tuple[int, int, int]
    spacing: tuple[float | None, float, float]
    norms: tuple[Norm, ...]
    features: tuple[int, ...]
    kernel_sizes: tuple[tuple[int, ...], ...]
    strides: tuple[tuple[int, ...], ...]
    n_conv_encoder: tuple[int, ...]  # convs per stage (plain) or residual blocks per stage (resenc)
    n_conv_decoder: tuple[int, ...]
    labels: Mapping[int, str]  # output label -> name
    network: str = "plain"  # "plain" (PlainConvUNet) or "resenc" (ResidualEncoderUNet)
    dims: int = 3
    stem: int | None = None  # resenc stem channels (None: features[0])
    regions: tuple[int, ...] | None = None  # region-based training: the label of each sigmoid head, in order
    transpose_forward: tuple[int, ...] = (0, 1, 2)
    transpose_backward: tuple[int, ...] = (0, 1, 2)
    conv_bias: bool = True
    norm_eps: float = 1e-5
    slope: float = 0.01
    resample_data: tuple[str, Mapping[str, Any]] = ("resample_data_or_seg_to_shape", {"order": 3, "order_z": 0})
    resample_logits: tuple[str, Mapping[str, Any]] = ("resample_data_or_seg_to_shape", {"order": 1, "order_z": 0})
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_outputs(self) -> int:
        """Segmentation heads: one per region, or background + the labels."""
        return len(self.regions) if self.regions is not None else len(self.labels) + 1

    @property
    def n_labels(self) -> int:
        """Largest output label + 1."""
        return max(self.labels, default=0) + 1


def _ints(v: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(i) for i in v)


def _per_stage(v: Any, n: int) -> list[Any]:  # noqa: ANN401 - an int or a list in the plans
    return list(v) if isinstance(v, (list, tuple)) else [v] * n


def _network(conf: Mapping[str, Any]) -> dict[str, Any]:
    """The `Plan` fields of the network: type, features, kernels, strides, convs / blocks, bias, eps, slope."""
    if "architecture" in conf:
        arch = conf["architecture"]
        name = arch.get("network_class_name")
        if name not in _ARCHS:
            msg = f"nnU-Net network {name!r}: only {', '.join(_ARCHS)} are supported"
            raise WeightsError(msg)
        a = arch["arch_kwargs"]
        n = int(a.get("n_stages", len(a["features_per_stage"])))
        kind = _ARCHS[name]
        enc = a["n_conv_per_stage"] if kind == "plain" else a["n_blocks_per_stage"]
        if kind == "resenc" and "BottleneckD" in str(a.get("block", "")):
            msg = "nnU-Net ResidualEncoderUNet with BottleneckD blocks is not supported"
            raise WeightsError(msg)
        nonlin = str(a.get("nonlin"))
        if "InstanceNorm" not in str(a.get("norm_op")) or not ("LeakyReLU" in nonlin or nonlin.endswith(".ReLU")):
            msg = f"nnU-Net {name} with {a.get('norm_op')} / {nonlin}: only InstanceNorm + (Leaky)ReLU"
            raise WeightsError(msg)
        slope = float((a.get("nonlin_kwargs") or {}).get("negative_slope", 0.01)) if "Leaky" in nonlin else 0.0
        return {
            "network": kind,
            "features": _ints(_per_stage(a["features_per_stage"], n)),
            "kernel_sizes": _per_stage(a["kernel_sizes"], n),
            "strides": _per_stage(a["strides"], n),
            "n_conv_encoder": _ints(_per_stage(enc, n)),
            "n_conv_decoder": _ints(_per_stage(a["n_conv_per_stage_decoder"], n - 1)),
            "conv_bias": bool(a.get("conv_bias", kind == "plain")),
            "norm_eps": float((a.get("norm_op_kwargs") or {}).get("eps", 1e-5)),
            "slope": slope,
            "stem": a.get("stem_channels"),
        }
    if conf.get("UNet_class_name") == "PlainConvUNet":  # legacy plans (nnU-Net < 2.3): fixed norm / nonlin
        kernels = conf["conv_kernel_sizes"]
        base, top = int(conf["UNet_base_num_features"]), int(conf["unet_max_num_features"])
        return {
            "network": "plain",
            "features": tuple(min(base * 2**i, top) for i in range(len(kernels))),
            "kernel_sizes": kernels,
            "strides": conf["pool_op_kernel_sizes"],
            "n_conv_encoder": _ints(conf["n_conv_per_stage_encoder"]),
            "n_conv_decoder": _ints(conf["n_conv_per_stage_decoder"]),
        }
    msg = f"nnU-Net network {conf.get('UNet_class_name')!r}: only PlainConvUNet / ResidualEncoderUNet are supported"
    raise WeightsError(msg)


def configuration_of(plans: Mapping[str, Any], configuration: str) -> dict[str, Any]:
    """One configuration of `plans.json` with nnU-Net's inheritance resolved (the child's keys win).

    Raises:
        WeightsError: No such configuration.
    """
    confs = plans.get("configurations", {})
    if configuration not in confs:
        msg = f"plans.json has no configuration {configuration!r} (has: {', '.join(confs)})"
        raise WeightsError(msg)
    conf = dict(confs[configuration])
    while "inherits_from" in conf:
        parent = dict(confs[conf.pop("inherits_from")])
        conf = {**parent, **conf}
    return conf


def _labels(dataset: Mapping[str, Any]) -> tuple[dict[int, str], tuple[int, ...] | None, int]:
    """Output label -> name, the region class order (or `None`), and the number of heads."""
    labels = {k: v for k, v in dict(dataset["labels"]).items() if k != "ignore"}
    if any(isinstance(v, (list, tuple)) and len(v) > 1 for v in labels.values()):
        order = dataset.get("regions_class_order")
        regions = [k for k, v in labels.items() if v not in (0, [0])]
        if order is None or len(order) != len(regions):
            msg = f"region-based nnU-Net labels need one regions_class_order entry per region ({len(regions)})"
            raise WeightsError(msg)
        return {int(c): r for c, r in zip(order, regions, strict=True)}, _ints(order), len(regions)
    ids = sorted(int(v[0] if isinstance(v, (list, tuple)) else v) for v in labels.values())
    if ids != list(range(len(ids))):
        msg = f"nnU-Net labels must be 0 .. n - 1, got {ids}"
        raise WeightsError(msg)
    named = {int(v[0] if isinstance(v, (list, tuple)) else v): k for k, v in labels.items()}
    return {k: n for k, n in named.items() if k}, None, len(ids)


def _lift(v: Sequence[Any], dims: int) -> tuple[int, ...]:
    return (1, *_ints(v)) if dims == _2D else _ints(v)


def read_plan(plans: Mapping[str, Any], dataset: Mapping[str, Any], configuration: str = "3d_fullres") -> Plan:
    """The `Plan` of one configuration of an nnU-Net v2 export (legacy and current plans formats).

    Raises:
        WeightsError: No such configuration, a cascade stage, an unsupported network, normalisation or resampling.
    """
    conf = configuration_of(plans, configuration)
    if conf.get("previous_stage"):
        msg = f"{configuration!r} is a cascade stage (needs the {conf['previous_stage']!r} prediction): not supported"
        raise WeightsError(msg)
    net = _network(conf)
    labels, regions, heads = _labels(dataset)
    dims = len(conf["patch_size"])
    if dims not in {2, 3}:
        msg = f"nnU-Net {dims}D configurations are not supported"
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
    resampling = []
    for key in ("resampling_fn_data", "resampling_fn_probabilities"):
        fn = conf.get(key, "resample_data_or_seg_to_shape")
        if fn not in _resample.RESAMPLERS:
            msg = f"nnU-Net {key} {fn!r} is not supported ({', '.join(_resample.RESAMPLERS)})"
            raise WeightsError(msg)
        resampling.append((fn, dict(conf.get(key + "_kwargs") or {})))
    spacing = tuple(float(s) for s in conf["spacing"])
    plan = Plan(
        patch_size=_lift(conf["patch_size"], dims),  # ty: ignore[invalid-argument-type]
        spacing=(None, *spacing) if dims == _2D else spacing,  # ty: ignore[invalid-argument-type]
        norms=tuple(norms),
        features=net.pop("features"),
        kernel_sizes=tuple(_lift(k, dims) for k in net.pop("kernel_sizes")),
        strides=tuple(_lift(s, dims) for s in net.pop("strides")),
        n_conv_encoder=net.pop("n_conv_encoder"),
        n_conv_decoder=net.pop("n_conv_decoder"),
        labels=labels,
        dims=dims,
        regions=regions,
        transpose_forward=_ints(plans.get("transpose_forward", (0, 1, 2))),
        transpose_backward=_ints(plans.get("transpose_backward", (0, 1, 2))),
        resample_data=resampling[0],
        resample_logits=resampling[1],
        **net,
    )
    if plan.n_outputs != heads:  # pragma: no cover - _labels guarantees it
        msg = f"nnU-Net heads: {plan.n_outputs} != {heads}"
        raise WeightsError(msg)
    return plan


class _ConvNormAct(nn.Module):
    """`ConvDropoutNormReLU` without dropout: conv, InstanceNorm (affine), LeakyReLU (slope 1: none)."""

    def __init__(  # noqa: PLR0913 - the block's arguments
        self,
        cin: int,
        cout: int,
        k: tuple[int, ...],
        stride: tuple[int, ...],
        plan: Plan,
        *,
        act: bool = True,
        bias: bool | None = None,
    ) -> None:
        super().__init__()
        pad = tuple((i - 1) // 2 for i in k)
        b = plan.conv_bias if bias is None else bias
        self.conv = nn.Conv3d(cin, cout, k, stride, pad, bias=b)  # ty: ignore[invalid-argument-type]
        self.norm = nn.InstanceNorm3d(cout, eps=plan.norm_eps, affine=True)
        self.slope = plan.slope if act else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv
        if conv.bias is not None and exporting_for_mlx():
            # The instance norm removes any per-channel constant: on MLX the bias is one more pass over the map.
            return self.norm_act(F.conv3d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups))
        return self.norm_act(conv(x))

    def norm_act(self, y: torch.Tensor) -> torch.Tensor:
        """`leaky_relu(norm(y))` as one scale / shift per channel: `InstanceNorm3d` on `channels_last_3d` is slow."""
        var, mean = torch.var_mean(y, dim=(2, 3, 4), correction=0, keepdim=True)
        w, b = (t.view(1, -1, 1, 1, 1) for t in (self.norm.weight, self.norm.bias))
        scale = w.float() * torch.rsqrt(var.float() + self.norm.eps)
        shift = b.float() - mean.float() * scale
        scale, shift = scale.to(y.dtype), shift.to(y.dtype)
        if y.is_cuda and not (torch.is_grad_enabled() or torch.compiler.is_compiling() or torch.jit.is_tracing()):
            # This input is the fresh convolution output. The broadcast affine reads each voxel once.
            torch.addcmul(shift, y, scale, out=y)
        else:
            y = torch.addcmul(shift, y, scale)
        return y if self.slope == 1.0 else F.leaky_relu_(y, self.slope)

    def forward_(self, y: torch.Tensor, depth: int) -> torch.Tensor:
        """`forward(y)` written over `y`, the convolution `depth` slices at a time (`slab_depth`)."""
        return self.norm_act(_conv_slabs_(self.conv, y, self.conv.bias, depth))


def _bias_norm_act(block: _ConvNormAct, y: torch.Tensor) -> torch.Tensor:
    """CUDA epilogue for a convolution whose bias has yet to be added."""
    from fastforks.kernels import vol_norm  # noqa: PLC0415 - kernels only

    bias = block.conv.bias
    if vol_norm.supported(y):
        return vol_norm.norm_act_(y, block.norm.weight, block.norm.bias, block.norm.eps, block.slope, conv_bias=bias)
    if bias is not None:
        y.add_(bias.view(1, -1, 1, 1, 1))
    return block.norm_act(y)


def slab_depth(conv: nn.Conv3d, y: torch.Tensor) -> int:
    """Depth slices per slab for `_ConvNormAct.forward_(y)`; 0 where it does not apply or `y` is under four slabs.

    It applies to an inference-mode CUDA map and a same-width, stride-1 convolution with a depth kernel of 3.
    """
    if (
        y.dim() != len(conv.kernel_size) + 2
        or not y.is_cuda
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
    ):
        return 0
    if torch.jit.is_tracing() or in_onnx_export() or exporting_for_mlx():
        return 0
    same = conv.in_channels == conv.out_channels and conv.groups == 1 and conv.padding_mode == "zeros"
    if (
        not same
        or any(s != 1 for s in (*conv.stride, *conv.dilation))
        or (conv.kernel_size[0], conv.padding[0]) != (3, 1)
    ):
        return 0
    depth = max(2, _SLAB_BYTES // (y[:, :, 0].numel() * y.element_size()))
    return depth if depth * 4 <= y.shape[2] else 0


def _conv_slabs_(conv: nn.Conv3d, y: torch.Tensor, bias: torch.Tensor | None, depth: int) -> torch.Tensor:
    """`F.conv3d(y, conv.weight, bias)` (`conv`'s padding) written over `y`, `depth` (`slab_depth`) slices at a time.

    A slab reads one slice beyond either end. Its output overwrites the slices no later slab reads; its last slice,
    which the next slab still reads, is written once that slab has run. Peak: `y` and about two slabs, not two maps.
    """
    d, w, pad = y.shape[2], conv.weight, tuple(int(p) for p in conv.padding)  # `slab_depth`: no string padding
    inner = (0, *pad[1:])
    held: torch.Tensor | None = None
    for a in range(0, d, depth):
        b = min(a + depth, d)
        if a == 0:  # the volume's ends: zero padding, the slice past the slab dropped
            out = F.conv3d(y[:, :, : b + 1], w, bias, 1, pad)[:, :, :b]
        elif b == d:
            out = F.conv3d(y[:, :, a - 1 :], w, bias, 1, pad)[:, :, 1:]
        else:
            out = F.conv3d(y[:, :, a - 1 : b + 1], w, bias, 1, inner)
        if held is not None:
            y[:, :, a - 1].copy_(held)
        y[:, :, a : b - 1].copy_(out[:, :, :-1])
        held = out[:, :, -1].clone()
        del out
    if held is not None:
        y[:, :, d - 1].copy_(held)
    return y


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


class _AvgPool(nn.Module):
    """`nn.AvgPool3d(stride, stride)` as a float32 sum of strided views that keeps `channels_last_3d`.

    avg_pool3d copies such inputs to contiguous and back (192^3, 32 channels: 20 ms vs 7 ms).
    """

    def __init__(self, stride: tuple[int, ...]) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = x.shape[2:]
        views = (
            x[(..., *(slice(o, o + s * (n // s), s) for o, s, n in zip(offs, self.stride, dims, strict=True)))]
            for offs in itertools.product(*(range(s) for s in self.stride))
        )
        out = next(views).to(torch.float32, copy=True)
        for v in views:
            out.add_(v)
        return out.mul_(1.0 / math.prod(self.stride)).to(x.dtype)


class _BasicBlock(nn.Module):
    """`BasicBlockD`: two conv blocks (the second without nonlinearity) + the (avg-pooled, projected) input."""

    def __init__(self, cin: int, cout: int, k: tuple[int, ...], stride: tuple[int, ...], plan: Plan) -> None:
        super().__init__()
        self.conv1 = _ConvNormAct(cin, cout, k, stride, plan)
        self.conv2 = _ConvNormAct(cout, cout, k, (1,) * len(k), plan, act=False)
        ops: list[nn.Module] = []
        if any(s != 1 for s in stride):
            ops.append(_AvgPool(stride))
        if cin != cout:
            ones = (1,) * len(k)
            ops.append(_ConvNormAct(cin, cout, ones, ones, plan, act=False, bias=False))
        self.skip = nn.Sequential(*ops) if ops else None
        self.slope = plan.slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.skip is None else self.skip(x)
        y = self.conv1(x)
        depth = slab_depth(self.conv2.conv, y)  # conv2 over conv1's map: one map less at the peak
        y = self.conv2.forward_(y, depth) if depth else self.conv2(y)
        return F.leaky_relu_(y.add_(res), self.slope)


class _ResStage(nn.Module):
    """`StackedResidualBlocks`: `n` basic blocks, the first with the stride."""

    def __init__(  # noqa: PLR0913, PLR0917 - StackedResidualBlocks' arguments
        self, n: int, cin: int, cout: int, k: tuple[int, ...], stride: tuple[int, ...], plan: Plan
    ) -> None:
        super().__init__()
        ones = (1,) * len(k)
        self.blocks = nn.Sequential(
            _BasicBlock(cin, cout, k, stride, plan), *(_BasicBlock(cout, cout, k, ones, plan) for _ in range(n - 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Encoder(nn.Module):
    """dynamic_network_architectures' `PlainConvEncoder` / `ResidualEncoder` (stem + stages), upstream's keys."""

    def __init__(self, plan: Plan, in_channels: int) -> None:
        super().__init__()
        if plan.network == "resenc":
            stem = plan.stem or plan.features[0]
            self.stem = _Stacked(1, in_channels, stem, plan.kernel_sizes[0], (1, 1, 1), plan)
            cins = (stem, *plan.features[:-1])
            self.stages = nn.ModuleList(
                _ResStage(n, ci, co, k, s, plan)
                for n, ci, co, k, s in zip(
                    plan.n_conv_encoder, cins, plan.features, plan.kernel_sizes, plan.strides, strict=True
                )
            )
        else:
            self.stem = None
            cins = (in_channels, *plan.features[:-1])
            self.stages = nn.ModuleList(
                nn.Sequential(_Stacked(n, ci, co, k, s, plan))
                for n, ci, co, k, s in zip(
                    plan.n_conv_encoder, cins, plan.features, plan.kernel_sizes, plan.strides, strict=True
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """The output of every stage (the skips; the last is the bottleneck)."""
        if self.stem is not None:
            x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


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
        self.tile_skip = False

    def features(self, skips: Sequence[torch.Tensor], *, consume: bool = False) -> torch.Tensor:
        owned = consume and isinstance(skips, list)
        pending = skips if owned else list(skips)
        x = pending.pop()
        for up, layer in zip(self.transpconvs, self.stages, strict=True):
            stage = cast("_Stacked", layer)
            depth = _decoder_tile_depth(stage, pending[-1]) if owned and self.tile_skip else 0
            if depth:
                x = up(x)
                x = _decode_into_skip(stage, x, pending.pop(), depth)
            else:
                x = stage(_cat_consuming(up, x, pending.pop()))
        return x

    def forward(self, skips: Sequence[torch.Tensor], *, consume: bool = False) -> torch.Tensor:
        return self.seg_layers[-1](self.features(skips, consume=consume))


def _decoder_tile_depth(stage: _Stacked, skip: torch.Tensor) -> int:
    """Number of full-resolution planes processed at once, where a skip can be safely reused."""
    conv = cast("_ConvNormAct", stage.convs[0]).conv
    if (
        not skip.is_cuda
        or skip.dtype != torch.float16
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or (conv.kernel_size[0], conv.padding[0]) != (3, 1)
        or not skip.is_contiguous(memory_format=torch.channels_last_3d)
    ):
        return 0
    plane_bytes = 2 * skip[:, :, 0].numel() * skip.element_size()
    depth = max(2, _DECODER_TILE_BYTES // plane_bytes)
    return depth if depth * 4 <= skip.shape[2] else 0


def _decode_into_skip(stage: _Stacked, upsampled: torch.Tensor, skip: torch.Tensor, tile: int) -> torch.Tensor:
    """Convolve depth tiles from the upsampled features and write the output over the consumed skip."""
    first = cast("_ConvNormAct", stage.convs[0])
    conv = first.conv
    depth = skip.shape[2]
    padding = cast("tuple[int, int, int]", conv.padding)
    inner = (0, *padding[1:])
    held = None
    for start in range(0, depth, tile):
        stop = min(start + tile, depth)
        lo, hi = max(0, start - 1), min(depth, stop + 1)
        joined = torch.cat((upsampled[:, :, lo:hi], skip[:, :, lo:hi]), dim=1).contiguous(
            memory_format=torch.channels_last_3d
        )
        if start > 0 and stop < depth:
            part = F.conv3d(joined, conv.weight, None, padding=inner)
        else:
            part = F.conv3d(joined, conv.weight, None, padding=padding)[:, :, start - lo : stop - lo]
        if held is not None:
            skip[:, :, start - 1].copy_(held)
        if stop < depth:
            # The next tile still needs this plane of the original skip as its low halo.
            skip[:, :, start : stop - 1].copy_(part[:, :, :-1])
            held = part[:, :, -1].clone()
        else:
            skip[:, :, start:stop].copy_(part)
        del joined, part
    x = _bias_norm_act(first, skip)
    for conv_block in stage.convs[1:]:
        x = conv_block(x)
    return x


def _cat_consuming(up: nn.Module, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """`torch.cat((up(x), skip), 1)`, the skip copied in (and, when consumed, freed) before `up(x)` is made.

    The skip, the upsampled map and their concatenation are never alive together: at the full-resolution stage of
    nnInteractive (192^3 patch, 32 + 32 channels, float16) peak 2429 -> 1997 MiB.
    """
    c = skip.shape[1]
    fmt = (
        torch.channels_last_3d if skip.is_contiguous(memory_format=torch.channels_last_3d) else torch.contiguous_format
    )
    out = torch.empty((skip.shape[0], 2 * c, *skip.shape[2:]), dtype=skip.dtype, device=skip.device, memory_format=fmt)
    out[:, c:].copy_(skip)
    del skip
    out[:, :c].copy_(up(x))  # the transposed conv makes the skip's channel count
    return out


class UNet(nn.Module):
    """dynamic_network_architectures' `PlainConvUNet` / `ResidualEncoderUNet` (no deep supervision), upstream's keys.

    3D throughout; a 2D plan has depth-1 kernels and strides (`state_dict_of` lifts the 2D weights).
    """

    def __init__(self, plan: Plan, in_channels: int) -> None:
        """Build the network of `plan` for `in_channels` input channels."""
        super().__init__()
        self.plan = plan
        self.encoder = Encoder(plan, in_channels)
        self.decoder = _Decoder(plan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits `[B, heads, *patch]` of `[B, C, *patch]`."""
        return self.decoder(self.encoder(x), consume=True)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Decoded patch features before the pointwise segmentation projection."""
        return self.decoder.features(self.encoder(x), consume=True)

    @staticmethod
    def state_dict_of(weights: Mapping[str, torch.Tensor], dims: int = 3) -> dict[str, torch.Tensor]:
        """An nnU-Net checkpoint's `network_weights` without its aliases (`all_modules`, `decoder.encoder`).

        `dims=2`: the 2D conv weights `[O, I, k, k]` as depth-1 3D weights `[O, I, 1, k, k]`.
        """
        out = {}
        for k, v in weights.items():
            key = k.removeprefix("_orig_mod.")
            if ".all_modules." in key or key.startswith("decoder.encoder."):
                continue
            out[key] = v.unsqueeze(2) if dims == _2D and v.dim() == 4 else v  # noqa: PLR2004
        return out


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
    g = (g / (g.max() / 10)).to(dtype)  # cast and zero fix on the CPU: on the device each was a host sync
    zero = g == 0
    if zero.any():
        g[zero] = g[~zero].min()
    return to_device(g, device)


def sliding_steps(size: Sequence[int], patch: Sequence[int], step: float) -> list[list[int]]:
    """nnU-Net's window starts per axis (`compute_steps_for_sliding_window`, `size` >= `patch`; patch 1: all slices)."""
    out = []
    for i, k in zip(size, patch, strict=True):
        if k == 1:  # the slice axis of a 2D configuration
            out.append(list(range(i)))
            continue
        n = math.ceil((i - k) / (k * step)) + 1
        actual = (i - k) / (n - 1) if n > 1 else 99999999999
        out.append([round(actual * j) for j in range(n)])
    return out


def nonzero_extent(nz: torch.Tensor) -> list[tuple[int, int] | None]:
    """Per axis of a mask, `(first, last + 1)` of its nonzero entries (`None`: none), with one host sync."""
    rows = []
    for d in range(nz.dim()):
        hit = nz.any(dim=tuple(a for a in range(nz.dim()) if a != d)).to(torch.uint8)
        # whether there is a nonzero entry, the first and the last one (argmax gives the first maximum)
        rows.append(torch.stack((hit.amax().long(), hit.argmax(), hit.numel() - hit.flip(0).argmax())))
    return [(i, j) if any_ else None for any_, i, j in torch.stack(rows).tolist()]


def _bbox(nz: torch.Tensor) -> tuple[slice, ...]:
    """nnU-Net's `crop_to_nonzero` box of a nonzero mask (all of the volume if it is empty)."""
    return tuple(slice(*e) if e else slice(0, n) for e, n in zip(nonzero_extent(nz), nz.shape, strict=True))


def _normalise(
    x: torch.Tensor, norms: Sequence[Norm], mask: torch.Tensor | None, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """nnU-Net's normalisation schemes per channel, in float32 (`mask`: the hole-filled nonzero mask)."""
    out = torch.empty_like(x, dtype=dtype)
    for c, n in enumerate(norms):
        v = x[c].float()
        if n.scheme == "CTNormalization":
            v = v.clamp(n.lower, n.upper).sub_(n.mean).div_(max(n.std, 1e-8))
        elif n.scheme == "ZScoreNormalization":
            if n.use_mask and mask is not None:
                sel = v[mask].cpu().double()  # float64 on the CPU: MPS has none
                v = torch.where(mask, (v - sel.mean().float()) / max(float(sel.std(correction=0)), 1e-8), v)
            else:
                d = v.cpu().double()
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


def _depth_slices(start: int, length: int, capacity: int) -> tuple[tuple[slice, slice], ...]:
    """Pair each physical ring-buffer span with its corresponding span in the incoming window."""
    at = start % capacity
    lead = min(length, capacity - at)
    first = (slice(at, at + lead), slice(0, lead))
    if lead == length:
        return (first,)
    return first, (slice(0, length - lead), slice(lead, length))


@dataclass
class Predictor:
    """nnU-Net v2's `predict_single_npy_array` for one configuration and a list of fold networks.

    Attributes:
        plan: The configuration.
        nets: The fold networks (their logits are averaged).
        step: Window step as a fraction of the patch (nnU-Net: 0.5; TotalSegmentator: 0.8 for its big tasks).
        mirror_axes: Test-time mirroring axes (0..2 of the 3D window; a 2D plan's axes + 1; empty: none).
        batch: Windows per network call.
        amp: Autocast dtype of the network on CUDA (`None`: the networks' own dtype).
        channels_last: Windows in `channels_last_3d` memory format.
        dtype: Cast the windows to this dtype (networks of that dtype, no autocast), or `None`.
        forward: Replaces `net(x)` (kernels runners); gets `(net, x)`.
        cuda_kernels: Use the fused accumulation and label-resampling kernels for FP16 CUDA inference.
    """

    plan: Plan
    nets: Sequence[nn.Module]
    step: float = 0.5
    mirror_axes: tuple[int, ...] = ()
    batch: int = 1
    amp: torch.dtype | None = torch.float16
    channels_last: bool = False
    dtype: torch.dtype | None = None
    forward: Callable[[Callable[[torch.Tensor], torch.Tensor], torch.Tensor], torch.Tensor] | None = None
    cuda_kernels: bool = False
    _gauss: torch.Tensor = field(default_factory=lambda: torch.ones(()), init=False, repr=False)

    def _net(self, net: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        return net(x) if self.forward is None else self.forward(net, x)

    def _logits(self, x: torch.Tensor, nets: Sequence[nn.Module]) -> torch.Tensor:
        """Window logits `[folds, B, heads, *patch]` of `nets`, each averaged over the mirrorings."""
        amp = self.amp is not None and x.device.type == "cuda"
        combos = [c for i in range(len(self.mirror_axes)) for c in itertools.combinations(self.mirror_axes, i + 1)]
        with torch.autocast("cuda", self.amp or torch.float16, enabled=amp):
            outs = []
            for net in nets:
                y = self._net(net, x)
                for axes in combos:
                    dims = [a + 2 for a in axes]
                    y += torch.flip(self._net(net, torch.flip(x, dims)), dims)
                if combos:
                    y /= len(combos) + 1
                outs.append(y)
        return outs[0][None] if len(outs) == 1 else torch.stack(outs)  # one network: no copy

    def target_spacing(self, spacing: Sequence[float]) -> tuple[float, ...]:
        """The plan's spacing in nnU-Net axis order (2D: the first axis keeps `spacing`'s)."""
        return tuple(s if t is None else t for s, t in zip(spacing, self.plan.spacing, strict=True))

    def preprocess(
        self, volume: torch.Tensor, spacing: Sequence[float]
    ) -> tuple[torch.Tensor, tuple[slice, ...], tuple[int, ...]]:
        """`(normalised, resampled data, crop box, cropped shape)` of a `[C, D, H, W]` volume in nnU-Net axis order."""
        p = self.plan
        x = volume.permute(0, *(t + 1 for t in p.transpose_forward))
        sp = [spacing[t] for t in p.transpose_forward]
        nz = (x != 0).any(0)
        box = _bbox(nz)
        x = x[(slice(None), *box)]
        mask = _resample.fill_holes(nz[box]) if any(n.use_mask for n in p.norms) else None
        shape = tuple(x.shape[1:])
        target = self.target_spacing(sp)
        new = _resample.new_shape(shape, sp, target)
        voxels = math.prod(new) * x.shape[0]
        if min(new) < 1 or voxels > _MAX_RESAMPLED_VOXELS:
            msg = (
                f"resampling the {tuple(volume.shape[1:])} volume from {tuple(spacing)} mm to the model's "
                f"{tuple(self.target_spacing(sp))} mm gives {tuple(new)} voxels"
                f"{'' if min(new) < 1 else f' ({voxels * 4 / 2**30:.0f} GiB as float32)'}: is the spacing in mm "
                "and in (D, H, W) order? Crop the volume or pass a coarser grid if it is right"
            )
            raise InputError(msg)
        fn, kw = p.resample_data
        dtype = (self.dtype or torch.float32) if self.cuda_kernels else torch.float32
        same_grid = tuple(new) == shape or fn == "no_resampling_hack"
        x = _normalise(x, p.norms, mask, dtype=dtype if same_grid else torch.float32)
        x = _resample.resample(x, new, sp, target, fn, kw)
        return x.to(dtype), box, shape

    def __call__(self, volume: torch.Tensor, spacing: Sequence[float]) -> torch.Tensor:
        """Labels `[D, H, W]` (uint8, int16 above 256 labels) of `volume` `[C, D, H, W]` with `spacing` (mm)."""
        p = self.plan
        full = volume.shape[1:]
        x, box, cropped = self.preprocess(volume, spacing)
        resample = tuple(x.shape[1:]) != cropped
        x, keep = _pad(x, p.patch_size)
        sp = [spacing[t] for t in p.transpose_forward]
        current = self.target_spacing(sp)
        fn, kwargs = p.resample_logits
        if (
            resample
            and self.cuda_kernels
            and x.is_cuda
            and self.channels_last
            and p.dims != _2D
            and p.regions is None
            and p.n_labels <= 256  # noqa: PLR2004 - the resampling kernel emits uint8 labels
            and len(self.nets) == 1
            and fn == "resample_torch_fornnunet"
            and not kwargs.get("is_seg")
            and _resample.lowres_axis(current, sp, kwargs.get("force_separate_z")) is None
        ):
            seg = self._stream_resampled(x, keep, cropped)
        else:
            logits = self._accumulate(x, full=resample or p.dims == _2D)
            if logits.dim() == 3:  # noqa: PLR2004 - already labels
                seg = logits[keep]
            else:
                inner = logits[(slice(None), *keep)]
                del logits
                seg = self._decide_resampled(inner, cropped, current, sp)
        out = torch.zeros(tuple(full[t] for t in p.transpose_forward), dtype=_labels_dtype(p.n_labels), device=x.device)
        out[box] = seg.to(out.dtype)
        return out.permute(*p.transpose_backward)

    def decide(self, logits: torch.Tensor, c0: int = 0, seg: torch.Tensor | None = None) -> torch.Tensor:
        """Labels of logits `[heads, ...]` (argmax; regions: sigmoid > 0.5 in class order, later wins)."""
        regions = self.plan.regions
        if regions is None:
            return logits.argmax(0).to(_labels_dtype(self.plan.n_labels))
        if seg is None:
            seg = torch.zeros(logits.shape[1:], dtype=_labels_dtype(self.plan.n_labels), device=logits.device)
        for i in range(logits.shape[0]):
            seg[logits[i] > 0] = regions[c0 + i]
        return seg

    def _decide_resampled(
        self,
        logits: torch.Tensor,
        shape: Sequence[int],
        current: Sequence[float],
        new: Sequence[float],
        chunk: int = 8,
    ) -> torch.Tensor:
        """Labels of the logits resampled (`resampling_fn_probabilities`) to `shape`, a few heads at a time."""
        fn, kw = self.plan.resample_logits

        def part(c0: int) -> torch.Tensor:
            return _resample.resample(logits[c0 : c0 + chunk].float(), shape, current, new, fn, kw)

        if self.plan.regions is not None:
            seg = None
            for c0 in range(0, logits.shape[0], chunk):
                seg = self.decide(part(c0), c0, seg)
            assert seg is not None  # noqa: S101 - at least one head
            return seg
        best, idx = part(0).max(0)
        labels = idx.to(_labels_dtype(self.plan.n_labels))
        del idx
        for c0 in range(chunk, logits.shape[0], chunk):
            values, idx = part(c0).max(0)
            better = values > best
            torch.where(better, values, best, out=best)
            torch.where(better, idx.add_(c0).to(labels.dtype), labels, out=labels)
            del values, idx, better
        return labels

    def _stream_resampled(self, x: torch.Tensor, keep: Sequence[slice], shape: Sequence[int]) -> torch.Tensor:
        """Finalize each target slice once both of its contributing score planes are complete."""
        from fastforks.kernels import vol_norm  # noqa: PLC0415 - kernels mode only

        depth = shape[0]
        src_depth = keep[0].stop - keep[0].start
        # Compute the higher source index for every target slice with the kernel's FP32 fused multiply-add.
        scale = torch.tensor(src_depth / depth, dtype=torch.float32, device="cpu").item()
        z = torch.arange(depth, dtype=torch.float64, device="cpu")
        coords = ((z + 0.5) * scale - 0.5).float().clamp_min_(0)
        right = (coords.long() + 1).clamp_max_(src_depth - 1).tolist()
        self._gauss = gaussian(self.plan.patch_size, x.device)
        steps = sliding_steps(x.shape[1:], self.plan.patch_size, self.step)
        out = torch.empty(tuple(shape), dtype=torch.uint8, device=x.device)
        previous = None
        next_row = 0
        for padded_start, scores in self._completed_slabs(x, steps, normalize=True):
            start = max(padded_start, keep[0].start)
            end = min(padded_start + scores.shape[1], keep[0].stop)
            if end <= start:
                continue
            current = scores[:, start - padded_start : end - padded_start, keep[1], keep[2]]
            source_start, source_end = start - keep[0].start, end - keep[0].start
            available = bisect_right(right, source_end - 1, lo=next_row)
            if available != next_row:
                vol_norm.resize_labels_(
                    out[next_row:available],
                    current,
                    previous,
                    first_src_z=source_start,
                    first_out_z=next_row,
                    src_depth=src_depth,
                    dst_depth=depth,
                )
            next_row = available
            if next_row == depth:
                break
            # The next target slab may still interpolate from the last plane before the ring buffer reuses it.
            previous = current[:, -1].permute(1, 2, 0).clone(memory_format=torch.contiguous_format)
        if next_row != depth:
            msg = f"Volume resampling finished {next_row} of {depth} target slices"
            raise RuntimeError(msg)
        return out

    def _window_batch(self, x: torch.Tensor, starts: Sequence[tuple[int, int, int]]) -> torch.Tensor:
        """Stack windows and prepare the batch in the network's layout and dtype."""
        d, h, w = self.plan.patch_size
        if self.cuda_kernels and self.channels_last and x.is_cuda:
            batch = torch.empty(
                (len(starts), x.shape[0], d, h, w),
                dtype=self.dtype or x.dtype,
                device=x.device,
                memory_format=torch.channels_last_3d,
            )
            for i, (a, b, c) in enumerate(starts):
                batch[i].copy_(x[:, a : a + d, b : b + h, c : c + w])
            return batch
        batch = torch.stack([x[:, a : a + d, b : b + h, c : c + w] for a, b, c in starts])
        return batch.to(
            dtype=self.dtype or x.dtype,
            memory_format=torch.channels_last_3d if self.channels_last else torch.preserve_format,
        )

    def _average(self, acc: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """nnU-Net's fold average of float16 sums `[folds, heads, ...]`: each fold / `n`, summed, / folds (float16)."""
        out = acc[0].div_(n)
        for f in range(1, acc.shape[0]):
            out += acc[f].div_(n)
        if acc.shape[0] > 1:
            out /= acc.shape[0]
        return out

    def _accumulate(self, x: torch.Tensor, *, full: bool) -> torch.Tensor:
        """Labels `[D, H, W]` of the padded data (or, `full`, the averaged float16 logits `[heads, D, H, W]`).

        The rounding is nnU-Net's: float16 maps and weights (also on the CPU), every fold's map divided by the weight
        sum before the folds are averaged (the windows of a volume corner have weights near float16's smallest).
        `full` runs the folds one after another (two maps, as upstream); otherwise all folds see each window and the
        sums are kept for one window depth (`[folds, heads, patch depth, H, W]`), decided as soon as they are complete
        (one network: on the sums, which differs from upstream only where the float16 division rounds two logits equal).
        """
        p = self.plan
        self._gauss = gaussian(p.patch_size, x.device)
        steps = sliding_steps(x.shape[1:], p.patch_size, self.step)
        d, h, w = x.shape[1:]
        dt = self._gauss.dtype
        # Match the window logits so accumulation reads and writes adjacent channels together on CUDA.
        layout = torch.channels_last_3d if self.channels_last else torch.contiguous_format
        if full:
            starts = [(a, b, c) for a in steps[0] for b in steps[1] for c in steps[2]]
            n = torch.zeros((d, h, w), dtype=dt, device=x.device)
            out = None
            for f, net in enumerate(self.nets):
                acc = torch.empty((1, p.n_outputs, d, h, w), dtype=dt, device=x.device, memory_format=layout).zero_()
                self._sweep(x, acc, n if f == 0 else None, starts, [net])
                y = acc[0].div_(n)
                out = y if out is None else out.add_(y)
                del acc, y
            assert out is not None  # noqa: S101 - at least one network
            if len(self.nets) > 1:
                out /= len(self.nets)
            return out
        labels = torch.empty((d, h, w), dtype=_labels_dtype(p.n_labels), device=x.device)
        # For hard labels on the model grid, argmax and sign can use the sums directly for one fold.
        for start, values in self._completed_slabs(x, steps, normalize=len(self.nets) > 1):
            labels[start : start + values.shape[1]] = self.decide(values)
        return labels

    def _completed_slabs(
        self, x: torch.Tensor, steps: Sequence[Sequence[int]], *, normalize: bool
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield complete slabs before reusing their space in the sliding-window ring buffer."""
        p = self.plan
        k0, k1, k2 = p.patch_size
        d, h, w = x.shape[1:]
        if tiling := self._head_tiling(x, self.nets):
            _, tile = tiling
            step = min((b - a for a, b in itertools.pairwise(steps[0])), default=k0)
            capacity = max(tile, k0 - step)
            windows = len(steps[1]) * len(steps[2])
            retained = (windows - 1) // self.batch * self.batch
            saved_scores = (k0 - capacity) * h * w * p.n_outputs
            extra_features = retained * k0 * k1 * k2 * p.features[0]
            if saved_scores > extra_features:
                yield from self._completed_head_tiles(x, steps, tiling, capacity, normalize=normalize)
                return
        dt = self._gauss.dtype
        layout = torch.channels_last_3d if self.channels_last else torch.contiguous_format
        acc = torch.empty(
            (len(self.nets), p.n_outputs, k0, h, w), dtype=dt, device=x.device, memory_format=layout
        ).zero_()
        n = torch.zeros((k0, h, w), dtype=dt, device=x.device) if normalize else None
        for i, a in enumerate(steps[0]):
            self._sweep(x, acc, n, [(a, b, c) for b in steps[1] for c in steps[2]], self.nets)
            end = steps[0][i + 1] if i + 1 < len(steps[0]) else d
            for physical, local in _depth_slices(a, end - a, k0):
                done = acc[:, :, physical]
                values = self._average(done, n[physical]) if n is not None else done[0]
                yield a + local.start, values
                if i + 1 < len(steps[0]):
                    done.zero_()
                    if n is not None:
                        n[physical].zero_()

    def _completed_head_tiles(
        self,
        x: torch.Tensor,
        steps: Sequence[Sequence[int]],
        tiling: tuple[UNet, int],
        capacity: int,
        *,
        normalize: bool,
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Hold the output features for one row of windows and finish one score tile at a time.

        A plane remains live only until all the windows that cover it have been merged. The ring holds at most the
        largest depth overlap or one output tile. Every voxel sees its contributing windows in the usual order.
        """
        net, tile = tiling
        head = net.decoder.seg_layers[-1]
        k0 = self.plan.patch_size[0]
        d, h, w = x.shape[1:]
        acc = torch.empty(
            (1, self.plan.n_outputs, capacity, h, w),
            device=x.device,
            dtype=self._gauss.dtype,
            memory_format=torch.channels_last_3d,
        ).zero_()
        n = torch.zeros((capacity, h, w), device=x.device, dtype=self._gauss.dtype) if normalize else None
        for i, a in enumerate(steps[0]):
            end = steps[0][i + 1] if i + 1 < len(steps[0]) else d
            starts = [(a, b, c) for b in steps[1] for c in steps[2]]
            batches = list(self._decoded_windows(x, starts, net))
            for k in range(0, k0, tile):
                stop = min(k + tile, k0)
                for chunk, features in batches:
                    for j, position in enumerate(chunk):
                        self._merge_head_tile(acc, n, head(features[j : j + 1, :, k:stop])[0], position, k)
                    del features
                ready = min(a + stop, end) - (a + k)
                if ready <= 0:
                    continue
                for physical, local in _depth_slices(a + k, ready, capacity):
                    values = acc[0, :, physical]
                    if n is not None:
                        values.div_(n[physical])
                    yield a + k + local.start, values
                    values.zero_()
                    if n is not None:
                        n[physical].zero_()
            del batches

    def _sweep(
        self,
        x: torch.Tensor,
        acc: torch.Tensor,
        n: torch.Tensor | None,
        starts: Sequence[tuple[int, int, int]],
        nets: Sequence[nn.Module],
    ) -> None:
        """Add the weighted logits of `nets` at `starts` to `acc` `[folds, heads, ...]` and the weights to `n`.

        Depth indices wrap at the capacity of `acc` and `n` for slab prediction.
        """
        k0, k1, k2 = self.plan.patch_size
        if tiling := self._head_tiling(x, nets):
            self._sweep_tiled_head(x, acc, n, starts, tiling)
            return
        weighted_add = None
        if self.cuda_kernels and self.channels_last and len(nets) == 1:
            from fastforks.kernels.vol_norm import accumulate_  # noqa: PLC0415 - kernels mode only

            weighted_add = accumulate_
        for j in range(0, len(starts), self.batch):
            chunk = starts[j : j + self.batch]
            y = self._logits(self._window_batch(x, chunk), nets)
            fuse = weighted_add is not None and y[0, 0].permute(1, 2, 3, 0).is_contiguous()
            if not fuse:
                y.mul_(self._gauss)  # keep the product rounded before adding to the FP16 maps
            for i, (a, b, c) in enumerate(chunk):
                ys, xs = slice(b, b + k1), slice(c, c + k2)
                for physical, local in _depth_slices(a, k0, acc.shape[2]):
                    if fuse and weighted_add is not None:
                        weighted_add(acc[0, :, physical, ys, xs], y[0, i, :, local], self._gauss[local])
                    else:
                        acc[:, :, physical, ys, xs] += y[:, i, :, local]
                    if n is not None:
                        n[physical, ys, xs] += self._gauss[local]
            del y  # free this batch of window logits before running the next network forward

    def _head_tiling(self, x: torch.Tensor, nets: Sequence[nn.Module]) -> tuple[UNet, int] | None:
        k0, k1, k2 = self.plan.patch_size
        plane_bytes = 2 * self.plan.n_outputs * k1 * k2
        if (
            self.cuda_kernels
            and x.is_cuda
            and self.channels_last
            and self.amp is None
            and not self.mirror_axes
            and len(nets) == 1
            and isinstance(nets[0], UNet)
            and k0 > 1
            and plane_bytes * k0 > 4 * _HEAD_TILE_BYTES
        ):
            return nets[0], math.ceil(_HEAD_TILE_BYTES / plane_bytes)
        return None

    def _decoded_windows(
        self, x: torch.Tensor, starts: Sequence[tuple[int, int, int]], net: UNet
    ) -> Iterator[tuple[Sequence[tuple[int, int, int]], torch.Tensor]]:
        for j in range(0, len(starts), self.batch):
            chunk = starts[j : j + self.batch]
            yield chunk, self._net(net.features, self._window_batch(x, chunk))

    def _sweep_tiled_head(
        self,
        x: torch.Tensor,
        acc: torch.Tensor,
        n: torch.Tensor | None,
        starts: Sequence[tuple[int, int, int]],
        tiling: tuple[UNet, int],
    ) -> None:
        """Project and merge depth tiles of the decoded features instead of materializing the full logits."""
        kd = self.plan.patch_size[0]
        net, depth = tiling
        head = net.decoder.seg_layers[-1]
        for chunk, features in self._decoded_windows(x, starts, net):
            for i, position in enumerate(chunk):
                for k in range(0, kd, depth):
                    self._merge_head_tile(acc, n, head(features[i : i + 1, :, k : k + depth])[0], position, k)
            del features

    def _merge_head_tile(
        self, acc: torch.Tensor, n: torch.Tensor | None, scores: torch.Tensor, position: tuple[int, int, int], k: int
    ) -> None:
        from fastforks.kernels.vol_norm import accumulate_  # noqa: PLC0415 - kernels mode only

        a, b, c = position
        depth, height, width = scores.shape[1:]
        ys, xs = slice(b, b + height), slice(c, c + width)
        fused = scores.permute(1, 2, 3, 0).is_contiguous()
        if not fused:
            scores.mul_(self._gauss[k : k + depth])
        for physical, local in _depth_slices(a + k, depth, acc.shape[2]):
            target = acc[0, :, physical, ys, xs]
            weights = self._gauss[k + local.start : k + local.stop]
            if fused:
                accumulate_(target, scores[:, local], weights)
            else:
                target.add_(scores[:, local])
            if n is not None:
                n[physical, ys, xs].add_(weights)
