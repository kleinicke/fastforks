"""TotalSegmentator runners: TotalSegmentator 2.18.0's `nnUNet_predict_image` on the GPU (also any nnU-Net v2 export).

The pipeline of `totalsegmentator.nnunet.nnUNet_predict_image` for the open tasks: the RAS-canonical image (the caller's
`[D, H, W]` from `fastforks.volume.read_nifti`) resampled to the task spacing as `change_spacing` does it (scipy
`zoom`, order 1, corner-aligned, truncated to int32), nnU-Net v2 (`fastforks.volume._nnunet`, fold 0, step 0.8, no
mirroring) per part, the parts merged into one label map (later parts win), and an order-0 corner-aligned zoom back to
the input grid. Not reproduced: the split of > 512 x 512 x 900 voxel volumes into three z chunks (it bounds upstream's
CPU memory for the full logits, which are never held here), and the optional crops / statistics / extra outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import NUMPY_ARRAYS, Inert, latin1_encode, load_strict, restricted_load
from fastforks._core.errors import WeightsError
from fastforks._core.patch import base_type, override
from fastforks.volume._nnunet import PlainConvUNet, Predictor, read_plan
from fastforks.volume._runner import Runner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks.volume._runner import Context

_FILES = ("plans.json", "dataset.json")
# besides the tensors: training state (numpy scalars and arrays in `logging` / `_best_ema`, not used)
_CHECKPOINT = {
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
        parts: nnU-Net model folders (`<dataset>/<trainer>__<plans>__<configuration>`), merged in this order.
        folds: Folds to ensemble (`fold_<k>/checkpoint_final.pth` in every part).
        spacing: TotalSegmentator's resampling to this isotropic spacing (mm) first, or `None` (nnU-Net's own).
        step: Sliding-window step (fraction of the patch).
        mirror: Test-time mirroring along the axes the checkpoints allow.
        classes: Label file (one name per line, line k = label k) of the merged map, or `None`: the single part's.
    """

    parts: tuple[str, ...]
    folds: tuple[int, ...] = (0,)
    spacing: float | None = None
    step: float = 0.5
    mirror: bool = True
    classes: str | None = None


def weight_names(parts: Sequence[str], folds: Sequence[int]) -> list[str]:
    """The weight file names of a variant, in the order the runner reads them."""
    return [f"{p}/{f}" for p in parts for f in (*_FILES, *(f"fold_{k}/checkpoint_final.pth" for k in folds))]


def zoom_linear(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """`scipy.ndimage.zoom(x, order=1)` to `shape` (corner-aligned) in float64, truncated to integers like `int32`."""
    y = F.interpolate(x[None, None].double(), size=tuple(shape), mode="trilinear", align_corners=True)[0, 0]
    return y.trunc_().float()


def zoom_nearest(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """`scipy.ndimage.zoom(x, order=0)` to `shape`: index `floor(o * (n - 1) / (m - 1) + 0.5)` per axis."""
    for axis, m in enumerate(shape):
        n = x.shape[axis]
        if n == m:
            continue
        scale = (n - 1) / (m - 1) if m > 1 else 0.0
        idx = torch.floor(torch.arange(m, dtype=torch.float64, device=x.device) * scale + 0.5).long().clamp_(0, n - 1)
        x = x.index_select(axis, idx)
    return x


class Portable(Runner):
    """Plain PyTorch: float16 networks in `channels_last_3d` on CUDA, two windows per network call.

    cuDNN's 3D convolutions are channels-last (no layout transposes); InstanceNorm + LeakyReLU is one per-channel
    scale / shift (`InstanceNorm3d` is slow on channels-last).
    """

    dtypes = ("float32", "float16")
    Params = Params
    windows: ClassVar[int] = 2

    def __init__(self, ctx: Context) -> None:
        """Read the plans and the fold checkpoints of every part."""
        super().__init__(ctx)
        p: Params = ctx.params
        files = ctx.weights()
        per = len(_FILES) + len(p.folds)
        self.parts: list[Predictor] = []
        luts, names = [], self._names(files, p)
        index = {n: k for k, n in names.items()}
        for i in range(len(p.parts)):
            plans, dataset = files[i * per : i * per + len(_FILES)]
            plan = read_plan(_json(plans), ds := _json(dataset))
            channels = len(ds.get("channel_names", ds.get("modality", {"0": ""})))
            first = i * per + len(_FILES)
            nets = [self._net(k, plan, channels) for k in range(first, first + len(p.folds))]
            mirror = _mirror_axes(files[first]) if p.mirror else ()
            self.parts.append(self._predictor(plan, nets, mirror))
            self.channels = channels
            lut = torch.zeros(plan.n_outputs, dtype=torch.int16)
            for k, n in plan.labels.items():
                if n not in index:
                    msg = f"{ctx.variant.ref}: class {n!r} of {p.parts[i]} is not in the class list"
                    raise WeightsError(msg)
                lut[k] = index[n]
            luts.append(lut.to(ctx.device))
        self.class_names = names
        self._luts = luts
        self._dtype = torch.uint8 if max(names, default=0) <= 255 else torch.int16  # noqa: PLR2004

    @staticmethod
    def _names(files: Sequence[Path], p: Params) -> dict[int, str]:
        if p.classes is None:
            labels = _json(files[1])["labels"]
            return {int(v): k for k, v in labels.items() if int(v)}
        lines = (Path(__file__).parent / p.classes).read_text().splitlines()
        return {k: n for k, n in enumerate(lines, start=1) if n}

    def _net(self, index: int, plan: Any, channels: int) -> torch.nn.Module:  # noqa: ANN401
        """The network of weight file `index` (a `checkpoint_final.pth`)."""
        net = PlainConvUNet(plan, channels)
        sd = self.ctx.state_dict(index, key="network_weights", strip="", extra=_CHECKPOINT)
        load_strict(net, PlainConvUNet.state_dict_of(sd), f"{self.ctx.variant.ref} checkpoint")
        net.eval().requires_grad_(requires_grad=False).to(self.ctx.device, self._dtype_of())
        if self.ctx.device.type == "cuda":
            net = net.to(memory_format=torch.channels_last_3d)
        return net

    def _dtype_of(self) -> torch.dtype:
        return torch.float16 if self.ctx.dtype == "float16" else torch.float32

    def _predictor(self, plan: Any, nets: list[torch.nn.Module], mirror: tuple[int, ...]) -> Predictor:  # noqa: ANN401
        cl = self.ctx.device.type == "cuda"
        return Predictor(
            plan,
            nets,
            step=self.ctx.params.step,
            mirror_axes=mirror,
            batch=self.windows,
            amp=None,
            channels_last=cl,
            dtype=self._dtype_of(),
        )

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """Every fold network of every part."""
        return tuple(n for part in self.parts for n in part.nets)

    def __call__(self, volume: torch.Tensor, spacing: Sequence[float]) -> dict[str, torch.Tensor]:
        """Labels of one volume on its grid."""
        p: Params = self.ctx.params
        shape = tuple(volume.shape[1:])
        x, sp = volume, tuple(spacing)
        if p.spacing is not None:
            sp32 = np.asarray(spacing, dtype=np.float32)
            if not np.array_equal(sp32, np.full(3, p.spacing)):
                zoom = sp32.astype(np.float64) / p.spacing
                new = tuple(round(n * float(z)) for n, z in zip(shape, zoom, strict=True))
                x = torch.stack([zoom_linear(c, new) for c in volume])
                sp = (p.spacing,) * 3
        out = self._luts[0][self.parts[0](x, sp).long()]
        for part, lut in zip(self.parts[1:], self._luts[1:], strict=True):
            m = lut[part(x, sp).long()]
            out = torch.where(m != 0, m, out)
        out = out.to(self._dtype)
        if tuple(out.shape) != shape:
            out = zoom_nearest(out, shape)
        return {"labels": out}


def _fused_norm_act(self: Any, y: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - a `_ConvNormAct`
    from fastforks.kernels import vol_norm  # noqa: PLC0415 - kernels only

    if vol_norm.supported(y):
        return vol_norm.norm_act_(y, self.norm.weight, self.norm.bias, self.norm.eps, self.slope)
    return base_type(self).norm_act(self, y)


class Kernels(Portable):
    """Portable + InstanceNorm + LeakyReLU as one in-place two-pass kernel (`vol_norm`) on the fp16 feature maps."""

    def __init__(self, ctx: Context) -> None:
        """Load `vol_norm` first, then the networks."""
        from fastforks.kernels import vol_norm  # noqa: PLC0415 - kernels only

        vol_norm.load(ctx.device)
        super().__init__(ctx)

    def _net(self, index: int, plan: Any, channels: int) -> torch.nn.Module:  # noqa: ANN401
        net = super()._net(index, plan, channels)
        for m in net.modules():
            if hasattr(m, "norm_act"):
                override(m, _fused_norm_act, "norm_act")
        return net


RUNNERS = {"portable": Portable, "kernels": Kernels}


def _mirror_axes(path: Path) -> tuple[int, ...]:
    """The checkpoint's `inference_allowed_mirroring_axes` (nnU-Net's test-time mirroring)."""
    ckpt = restricted_load(path, _CHECKPOINT)
    return tuple(ckpt.get("inference_allowed_mirroring_axes") or ()) if isinstance(ckpt, dict) else ()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
