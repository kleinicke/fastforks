"""3D volume segmentation API: `list_models`, `model_card`, `load` (also nnU-Net v2 folders), model / result types."""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from fastforks._core import model as core
from fastforks._core import registry
from fastforks._core.api import is_path, task_api
from fastforks._core.arguments import typo_checked_load
from fastforks._core.errors import InputError, OptionsError, WeightsError
from fastforks._core.files import FileSpec
from fastforks._core.io import atomic_write, check_suffix
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs
from fastforks.volume._nifti import VolumeMeta, write_nifti

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fastforks._core.bench import BenchInputs, BenchSize
    from fastforks._core.cards import Dtype
    from fastforks._core.model import ModeArg
    from fastforks.volume._runner import Runner

__all__ = ["Backend", "VolumeModel", "VolumeResult", "list_models", "load", "model_card"]

TASK = "volume"
_NDIM = (3, 4)


@dataclass(frozen=True)
class VolumeResult(NumpyOutputs):
    """Output of a volume segmentation model, on the model's device, on the input grid.

    Attributes:
        labels: Label per voxel `[D, H, W]`, uint8 (int16 for more than 255 classes); 0 is background.
        class_names: Label -> class name (background not listed).
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
        meta: The `meta=` of the call (the input file's grid), else `None`.
    """

    labels: torch.Tensor
    class_names: Mapping[int, str]
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)
    meta: VolumeMeta | None = None

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the labels: `.nii.gz` / `.nii` on the input file's grid (needs the call's `meta=`), or `.npy` as is.

        Raises:
            InputError: Unknown suffix, or a NIfTI path and a call without `meta=`.
        """
        name = os.fspath(path).lower()
        if name.endswith((".nii.gz", ".nii")):
            if self.meta is None:
                msg = "VolumeResult.save writes NIfTI on the input file's grid: call the model with meta= (read_nifti)"
                raise InputError(msg)
            write_nifti(path, self.labels, self.meta)
            return
        check_suffix(path, (".nii.gz", ".nii", ".npy"), type(self).__name__)
        atomic_write(path, lambda tmp: np.save(tmp, self.labels.detach().cpu().numpy()))


class VolumeModel(core.Model):
    """A loaded volume segmentation model. Call it with a volume and its voxel spacing.

    Besides the attributes of `fastforks.Model`:

    Attributes:
        class_names: Label -> class name (background 0 not listed).
        channels: Input channels (modalities) the model takes.
    """

    _runner: Runner

    def __init__(self, runner: Runner, backend: Backend, options: Mapping[str, Any]) -> None:
        """Wrap a runner (built by `load`)."""
        super().__init__(runner, backend, options)
        self.class_names = dict(runner.class_names)
        self.channels = runner.channels

    def module(self) -> torch.nn.Module:
        """Not available: the sliding-window pipeline is not one module (`model.networks()` are the networks).

        Raises:
            BackendUnavailableError: Always.
        """
        return super().module()

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The networks (every fold of every part), e.g. to inspect or export one patch-sized U-Net."""
        return self._runner.networks()

    def __call__(
        self,
        volume: np.ndarray | torch.Tensor,
        *,
        spacing: Sequence[float] | None = None,
        meta: VolumeMeta | None = None,
    ) -> VolumeResult:
        """Label every voxel of a volume.

        Args:
            volume: `[D, H, W]` or `[C, D, H, W]` (numpy or torch; CT in Hounsfield units), e.g. from
                `fastforks.volume.read_nifti`, whose `[D, H, W]` is the RAS-canonical `(z, y, x)` order.
            spacing: Voxel size in mm along `D`, `H`, `W`.
            meta: The `VolumeMeta` of `read_nifti` instead of `spacing`: its spacing, and `result.save("x.nii.gz")`
                writes the labels on that file's grid. Give one of the two.

        Returns:
            The labels on the input grid.

        Raises:
            InputError: Wrong number of dimensions or channels, neither or both of `spacing` and `meta`, a spacing that
                is not three positive numbers, or a volume not on the grid of `meta`.
        """
        x = torch.as_tensor(np.ascontiguousarray(volume) if isinstance(volume, np.ndarray) else volume)
        if x.ndim not in _NDIM or x.is_complex() or x.dtype == torch.bool:
            msg = f"volume must be a real [D, H, W] or [C, D, H, W] array, got {x.dtype} {tuple(x.shape)}"
            raise InputError(msg)
        if x.ndim == 3:  # noqa: PLR2004
            x = x[None]
        if x.shape[0] != self.channels:
            msg = f"{self.variant.ref} takes {self.channels} channel(s), got {x.shape[0]} ({tuple(x.shape)})"
            raise InputError(msg)
        if (spacing is None) == (meta is None):
            msg = "give the voxel size as spacing=(d, h, w) in mm or meta= (from read_nifti), one of the two"
            raise InputError(msg)
        if meta is not None:
            if tuple(x.shape[1:]) != tuple(meta.shape):
                msg = f"volume {tuple(x.shape[1:])} is not on the grid of meta {tuple(meta.shape)}"
                raise InputError(msg)
            spacing = meta.spacing
        sp = tuple(float(s) for s in cast("Sequence[float]", spacing))
        if len(sp) != 3 or not all(math.isfinite(s) and s > 0 for s in sp):  # noqa: PLR2004
            msg = f"spacing must be three positive voxel sizes in mm (D, H, W), got {spacing!r}"
            raise InputError(msg)
        with self._session():
            x = x.to(self.device, torch.float32, non_blocking=True)
            out = self._runner(x, sp)
            labels = self._check_outputs(out, {"labels": tuple(x.shape[1:])}, cast=False)["labels"]
        return VolumeResult(labels, self.class_names, self.variant.ref, self.backend, self.options, meta)


def bench_inputs(model: VolumeModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: a synthetic CT-like volume `[H, W, W]` at 1.5 mm (H slices of W x W voxels)."""
    g = torch.Generator().manual_seed(0)
    vol = torch.randn((model.channels, size.height, size.width, size.width), generator=g) * 300.0
    return (vol.to(model.device),), {"spacing": (1.5, 1.5, 1.5)}


list_models, model_card, _load = task_api(TASK, VolumeModel)


def load(  # noqa: PLR0913 - the arguments of every task's `load` + the nnU-Net options
    model: str | os.PathLike[str],
    *,
    mode: ModeArg = "auto",
    device: str | torch.device | None = None,
    dtype: Dtype | torch.dtype | None = None,
    weights: str | os.PathLike[str] | None = None,
    folds: Sequence[int] | None = None,
    step: float | None = None,
    mirror: bool | None = None,
    checkpoint: str | None = None,
    **options: object,
) -> VolumeModel:
    """Load a volume model: a model id, or a trained nnU-Net v2 model folder or one of its checkpoints.

    A path (`PathLike`, a string with a slash or backslash, or ending in `.pth .pt .safetensors .ckpt .bin .onnx`) is an
    nnU-Net v2 model folder (`<dataset>/<trainer>__<plans>__<configuration>`: `plans.json`, `dataset.json` and
    `fold_<k>/<checkpoint>`) or a checkpoint `.pth` (it carries its plans, dataset json and configuration). nnU-Net
    v2's own inference runs on the GPU: its cropping, normalisation and resampling, the Gaussian sliding window,
    mirroring, the fold ensemble, and the resampling of the logits back. 2D and 3D configurations of `PlainConvUNet`
    and `ResidualEncoderUNet` (basic blocks), label- and region-based training; no cascade stages. Not hash-checked.
    Anything else is a model id (`list_models()`), loaded as by every task's `load` (`help(fastforks.stereo.load)`).

    Args:
        model: Model id (optionally `@revision`), or the path of an nnU-Net model folder or checkpoint file.
        mode: `"auto"`, `"portable"` or `"kernels"`, as every task's `load`.
        device: As every task's `load`.
        dtype: As every task's `load`.
        **options: Load options of a model id (card `[options]`); not for a path.
        weights: Your own copy of a model id's weights; not for a path.
        folds: nnU-Net folder: folds to ensemble; `None`: every `fold_<k>` with the checkpoint.
        step: nnU-Net path: sliding-window step (fraction of the patch; `None`: nnU-Net's default 0.5).
        mirror: nnU-Net path: test-time mirroring along the axes the checkpoint allows (`None`: on, nnU-Net's default).
        checkpoint: nnU-Net folder: checkpoint file name in the fold folders (`None`: `checkpoint_final.pth`;
            `checkpoint_best.pth`: nnU-Net's best epoch).

    Returns:
        The loaded model.

    Raises:
        ModelNotFoundError: Unknown id or revision.
        OptionsError: As every task's `load`; options / `weights` with a path, the nnU-Net arguments with a model id,
            or `folds` / `checkpoint` with a checkpoint file.
        WeightsError: Missing files, or an architecture / configuration the engine does not implement.
    """
    nnunet = {"folds": folds, "step": step, "mirror": mirror, "checkpoint": checkpoint}
    if not is_path(model):
        if given := [k for k, v in nnunet.items() if v is not None]:
            msg = f"{', '.join(f'{k}=' for k in given)}: for nnU-Net folders and checkpoints, not model ids"
            raise OptionsError(msg)
        return _load(str(model), mode=mode, device=device, dtype=dtype, weights=weights, **options)
    if options or weights is not None:
        msg = f"{str(model)!r} is an nnU-Net path: options and weights= are for model ids"
        raise OptionsError(msg)
    return _load_nnunet(
        model,
        mode=mode,
        device=device,
        dtype=dtype,
        folds=folds,
        step=0.5 if step is None else step,
        mirror=True if mirror is None else mirror,
        checkpoint=checkpoint,
    )


load.__module__ = f"fastforks.{TASK}"
load.__annotations__["return"] = VolumeModel  # resolved, as task_api's load
load = typo_checked_load(load)  # ty: ignore[invalid-assignment]


def _load_nnunet(  # noqa: PLR0913 - the arguments of `load` + the nnU-Net options
    path: str | os.PathLike[str],
    *,
    mode: ModeArg,
    device: str | torch.device | None,
    dtype: Dtype | torch.dtype | None,
    folds: Sequence[int] | None,
    step: float,
    mirror: bool,
    checkpoint: str | None,
) -> VolumeModel:
    """`load` of an nnU-Net v2 model folder or checkpoint file."""
    from fastforks.volume._nnunet_runner import weight_names  # noqa: PLC0415 - the runner's layout

    p = Path(os.fspath(path)).expanduser().resolve()
    if p.is_file() and (folds is not None or checkpoint is not None):
        msg = f"{str(path)!r} is a checkpoint file: folds= and checkpoint= are for model folders"
        raise OptionsError(msg)
    checkpoint = checkpoint or "checkpoint_final.pth"
    if p.is_file():
        parts, root, embedded, folds = (p.name,), p.parent, True, ()
    else:
        if folds is None:
            folds = sorted(int(d.name[5:]) for d in p.glob("fold_*") if (d / checkpoint).is_file())
            if not folds:
                msg = f"{str(path)!r}: no fold_<k>/{checkpoint}"
                raise WeightsError(msg)
        parts, root, embedded = (f"{p.parent.name}/{p.name}",), p.parent.parent, False
    card, variant = registry.resolve(TASK, "totalsegmentator-total-3mm")  # the nnU-Net runners, other params
    params = {
        "parts": parts,
        "folds": tuple(folds),
        "spacing": None,
        "step": step,
        "mirror": mirror,
        "classes": None,
        "configuration": "",
        "checkpoint": checkpoint,
        "embedded": embedded,
    }
    names = weight_names(parts, tuple(folds), checkpoint, embedded=embedded)
    weights = tuple(FileSpec(n, "0" * 64) for n in names)
    variant = dataclasses.replace(variant, weights=weights, params=params)
    return core.load_variant(TASK, card, variant, VolumeModel, mode=mode, device=device, dtype=dtype, weights=root)
