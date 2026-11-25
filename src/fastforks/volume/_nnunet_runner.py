"""Runners of nnU-Net v2 model folders and checkpoints (`load(path)`, TotalSegmentator, the DKFZ families).

A variant lists one or more parts (nnU-Net model folders `<dataset>/<trainer>__<plans>__<configuration>`): every
part is nnU-Net v2's `predict_single_npy_array` (`fastforks.volume._nnunet`), the fold networks of a part are averaged,
and the parts' label maps are merged in order (later parts win). The weight files per part are `plans.json`,
`dataset.json` and `fold_<k>/<checkpoint>`, or (`embedded`) one bare checkpoint whose `init_args` hold the plans, the
dataset json and the configuration (nnU-Net v2 writes them into every checkpoint).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import NUMPY_ARRAYS, Inert, latin1_encode, load_strict, restricted_load
from fastforks._core.errors import OptionsError, WeightsError
from fastforks._core.patch import base_type, override
from fastforks._core.runner import host
from fastforks.volume._nnunet import Plan, Predictor, UNet, _conv_slabs_, _ConvNormAct, read_plan
from fastforks.volume._runner import Runner

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from fastforks.volume._runner import Context

__all__ = ["CHECKPOINT_GLOBALS", "Kernels", "Params", "Portable", "fused_norm_act", "weight_names"]

_FILES = ("plans.json", "dataset.json")
# besides the tensors: training state (numpy scalars and arrays in `logging` / `_best_ema`, the device in `init_args`)
CHECKPOINT_GLOBALS = {
    **NUMPY_ARRAYS,
    **{(m, "scalar"): Inert for m in ("numpy._core.multiarray", "numpy.core.multiarray")},
    **{("numpy.dtypes", f"{t}DType"): Inert for t in ("Float64", "Float32", "Int64", "Int32", "Bool")},
    ("_codecs", "encode"): latin1_encode,
    ("torch", "device"): Inert,
}


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`.

    Attributes:
        parts: nnU-Net model folders (`<dataset>/<trainer>__<plans>__<configuration>`), merged in this order;
            `embedded`: the checkpoint files.
        folds: Folds to ensemble (`fold_<k>/<checkpoint>` in every part; ignored when `embedded`).
        step: Sliding-window step (fraction of the patch).
        mirror: Test-time mirroring along the axes the checkpoints allow.
        configuration: The plans configuration, or empty: the part name's last `__` field (nnU-Net's layout).
        checkpoint: Checkpoint file name in every fold folder.
        stream: Keep fold networks on the CPU between parts and stage one part per volume on CUDA.
        embedded: The parts are bare checkpoints; plans, dataset json and configuration come from their `init_args`.
    """

    parts: tuple[str, ...]
    folds: tuple[int, ...] = (0,)
    step: float = 0.5
    mirror: bool = True
    configuration: str = ""
    checkpoint: str = "checkpoint_final.pth"
    embedded: bool = False
    stream: bool = False


def weight_names(
    parts: Sequence[str], folds: Sequence[int], checkpoint: str = "checkpoint_final.pth", *, embedded: bool = False
) -> list[str]:
    """The weight file names of a variant, in the order the runner reads them."""
    if embedded:
        return list(parts)
    return [f"{p}/{f}" for p in parts for f in (*_FILES, *(f"fold_{k}/{checkpoint}" for k in folds))]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _init_args(path: Path) -> dict[str, Any]:
    ckpt = restricted_load(path, CHECKPOINT_GLOBALS)
    init = ckpt.get("init_args") if isinstance(ckpt, dict) else None
    if not isinstance(init, dict) or not {"plans", "dataset_json", "configuration"} <= init.keys():
        msg = f"{path}: not an nnU-Net v2 checkpoint (no init_args with plans, dataset_json and configuration)"
        raise WeightsError(msg)
    return init


def _mirror_axes(path: Path, plan: Plan) -> tuple[int, ...]:
    """The checkpoint's `inference_allowed_mirroring_axes` as axes of the 3D window (2D: + 1)."""
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open  # noqa: PLC0415 - only for nnU-Net's safetensors checkpoints

        with safe_open(path, framework="pt", device="cpu") as file:
            axes = json.loads((file.metadata() or {}).get("inference_allowed_mirroring_axes", "null")) or ()
    else:
        ckpt = restricted_load(path, CHECKPOINT_GLOBALS)
        axes = ckpt.get("inference_allowed_mirroring_axes") or () if isinstance(ckpt, dict) else ()
    return tuple(int(a) + (3 - plan.dims) for a in axes)


class Portable(Runner):
    """Plain PyTorch: float16 networks in `channels_last_3d` on CUDA, two windows per network call.

    cuDNN's 3D convolutions are channels-last (no layout transposes); InstanceNorm + LeakyReLU is one per-channel
    scale / shift (`InstanceNorm3d` is slow on channels-last).
    """

    dtypes = ("float32", "float16")
    Params = Params
    compiled_methods = ("_forward_patch",)
    windows: ClassVar[int] = 2

    def __init__(self, ctx: Context) -> None:
        """Read the plans and the fold checkpoints of every part."""
        super().__init__(ctx)
        p: Params = ctx.params
        if p.stream and ctx.device.type != "cuda":
            msg = "stream=True requires CUDA for nnU-Net part transfers"
            raise OptionsError(msg)
        files = ctx.weights()
        self.parts: list[Predictor] = []
        labels: list[Mapping[int, str]] = []
        i = 0
        for part in p.parts:
            if p.embedded:
                init = _init_args(files[i])
                plans, ds, conf = init["plans"], init["dataset_json"], str(init["configuration"])
                ckpts = [i]
            else:
                plans, ds = _json(files[i]), _json(files[i + 1])
                conf = p.configuration or part.rsplit("__", 1)[-1]
                ckpts = list(range(i + len(_FILES), i + len(_FILES) + len(p.folds)))
            i = ckpts[-1] + 1
            plan = read_plan(plans, ds, conf)
            channels = len(ds.get("channel_names", ds.get("modality", {"0": ""})))
            nets = [self._net(k, plan, channels) for k in ckpts]
            mirror = _mirror_axes(files[ckpts[0]], plan) if p.mirror else ()
            self.parts.append(self._predictor(plan, nets, mirror))
            self.channels = channels
            labels.append(plan.labels)
        self.class_names = self._names(labels)
        index = {n: k for k, n in self.class_names.items()}
        self._dtype = torch.uint8 if max(self.class_names, default=0) <= 255 else torch.int16  # noqa: PLR2004
        self._luts: list[torch.Tensor | None] = []
        for part, lab in zip(self.parts, labels, strict=True):
            lut = torch.zeros(part.plan.n_labels, dtype=torch.int16)
            for k, n in lab.items():
                if n not in index:
                    msg = f"{ctx.variant.ref}: class {n!r} is not in the class list"
                    raise WeightsError(msg)
                lut[k] = index[n]
            identity = torch.equal(lut, torch.arange(lut.numel(), dtype=lut.dtype))
            self._luts.append(None if identity else lut.to(device=ctx.device, dtype=self._dtype))

    def _names(self, labels: Sequence[Mapping[int, str]]) -> dict[int, str]:
        """The merged map's label -> name: every part's names, numbered in order of appearance (one part: its own)."""
        if len(labels) == 1:
            return dict(labels[0])
        out: dict[int, str] = {}
        for lab in labels:
            for _, n in sorted(lab.items()):
                if n not in out.values():
                    out[len(out) + 1] = n
        return out

    def _net(self, index: int, plan: Plan, channels: int) -> torch.nn.Module:
        """The network of weight file `index` (a fold checkpoint)."""
        net = UNet(plan, channels)
        if self.ctx.params.checkpoint.lower().endswith(".safetensors"):
            sd = self.ctx.state_dict(index, strip="")
            # nnU-Net 2.9 stores training state separately from the flat network tensors.
            sd.pop("__nnunet_checkpoint_state__", None)
        else:
            sd = self.ctx.state_dict(index, key="network_weights", strip="", extra=CHECKPOINT_GLOBALS)
        load_strict(net, UNet.state_dict_of(sd, plan.dims), f"{self.ctx.variant.ref} checkpoint")
        device = torch.device("cpu") if self.ctx.params.stream else self.ctx.device
        net.eval().requires_grad_(requires_grad=False).to(device, self._dtype_of())
        if self.ctx.device.type == "cuda":
            net = net.to(memory_format=torch.channels_last_3d)
        return net

    def _dtype_of(self) -> torch.dtype:
        return torch.float16 if self.ctx.dtype == "float16" else torch.float32

    def _predictor(self, plan: Plan, nets: list[torch.nn.Module], mirror: tuple[int, ...]) -> Predictor:
        return Predictor(
            plan,
            nets,
            step=self.ctx.params.step,
            mirror_axes=mirror,
            batch=self.windows,
            amp=None,
            channels_last=self.ctx.device.type == "cuda",
            dtype=self._dtype_of(),
        )

    def _forward_patch(self, net: Callable[[torch.Tensor], torch.Tensor], patch: torch.Tensor) -> torch.Tensor:
        """Compile the fold network on fixed-size patches. Resampling and merging stay eager."""
        return net(patch)

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """Every fold network of every part."""
        return tuple(n for part in self.parts for n in part.nets)

    def segment(self, volume: torch.Tensor, spacing: Sequence[float]) -> torch.Tensor:
        """The merged labels of every part on the grid of `volume`."""

        def predict(part: Predictor, lut: torch.Tensor | None) -> torch.Tensor:
            nets = part.nets
            if self.ctx.params.stream:
                # Keep each fold on device for every window of this part. The host masters stay
                # immutable, including when two callers run inference on different CUDA streams.
                nets = [copy.deepcopy(net).to(volume.device) for net in part.nets]
            predictor = replace(part, nets=nets, forward=self._forward_patch)
            labels = predictor(volume, spacing)
            if lut is None:
                return labels.to(self._dtype)
            # index_select accepts 32-bit indices, halving the full-volume indexing temporary.
            indices = labels.to(torch.int32, memory_format=torch.contiguous_format).view(-1)
            return lut.index_select(0, indices).view(labels.shape)

        out = predict(self.parts[0], self._luts[0])
        for part, lut in zip(self.parts[1:], self._luts[1:], strict=True):
            labels = predict(part, lut)
            torch.where(labels != 0, labels, out, out=out)
        return out

    @host
    def __call__(self, volume: torch.Tensor, spacing: Sequence[float]) -> dict[str, torch.Tensor]:
        """Labels of one volume on its grid."""
        return {"labels": self.segment(volume, spacing)}


def fused_norm_act(self: Any, y: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - a `_ConvNormAct`
    """`norm_act` of a `_ConvNormAct` with `vol_norm` (in place), the portable one where the kernel does not apply."""
    from fastforks.kernels import vol_norm  # noqa: PLC0415 - kernels only

    if vol_norm.supported(y):
        return vol_norm.norm_act_(y, self.norm.weight, self.norm.bias, self.norm.eps, self.slope)
    return base_type(self).norm_act(self, y)


def _biased_conv_norm(self: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - nnU-Net conv block
    # cuDNN otherwise adds the channel bias in a separate pass over the complete feature map.
    conv = self.conv
    return _bias_norm_act(self, F.conv3d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups))


def _biased_conv_norm_(self: Any, y: torch.Tensor, depth: int) -> torch.Tensor:  # noqa: ANN401 - nnU-Net conv block
    """`_biased_conv_norm` written over `y` (`_ConvNormAct.forward_`)."""
    return _bias_norm_act(self, _conv_slabs_(self.conv, y, None, depth))


def _bias_norm_act(self: Any, y: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - nnU-Net conv block
    from fastforks.kernels import vol_norm  # noqa: PLC0415 - kernels only

    conv = self.conv
    if vol_norm.supported(y):
        return vol_norm.norm_act_(y, self.norm.weight, self.norm.bias, self.norm.eps, self.slope, conv_bias=conv.bias)
    return self.norm_act(y.add_(conv.bias.view(1, -1, 1, 1, 1)))


class Kernels(Portable):
    """Portable + InstanceNorm + (Leaky)ReLU as one in-place two-pass kernel (`vol_norm`) on the fp16 feature maps."""

    def __init__(self, ctx: Context) -> None:
        """Load `vol_norm` first, then the networks."""
        from fastforks.kernels import vol_norm  # noqa: PLC0415 - kernels only

        vol_norm.load(ctx.device)
        super().__init__(ctx)

    def _predictor(self, plan: Plan, nets: list[torch.nn.Module], mirror: tuple[int, ...]) -> Predictor:
        predictor = super()._predictor(plan, nets, mirror)
        predictor.cuda_kernels = self.ctx.dtype == "float16"
        return predictor

    def _net(self, index: int, plan: Plan, channels: int) -> torch.nn.Module:
        net = super()._net(index, plan, channels)
        for m in net.modules():
            if isinstance(m, _ConvNormAct):
                override(m, fused_norm_act, "norm_act")
                if m.conv.bias is not None:
                    override(m, _biased_conv_norm)
                    override(m, _biased_conv_norm_, "forward_")
        return net
