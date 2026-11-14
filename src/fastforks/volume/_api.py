"""3D volume segmentation API: `list_models`, `model_card`, `load`, `load_nnunet`, and the model / result types."""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastforks._core import model as core
from fastforks._core import registry
from fastforks._core.api import task_api
from fastforks._core.errors import InputError, WeightsError
from fastforks._core.files import FileSpec
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fastforks._core.bench import BenchInputs, BenchSize
    from fastforks._core.cards import Dtype
    from fastforks._core.model import ModeArg
    from fastforks.volume._runner import Runner

__all__ = ["Backend", "VolumeModel", "VolumeResult", "list_models", "load", "load_nnunet", "model_card"]

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
    """

    labels: torch.Tensor
    class_names: Mapping[int, str]
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)


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

    def __call__(self, volume: np.ndarray | torch.Tensor, spacing: Sequence[float]) -> VolumeResult:
        """Label every voxel of a volume.

        Args:
            volume: `[D, H, W]` or `[C, D, H, W]` (numpy or torch; CT in Hounsfield units), e.g. from
                `fastforks.volume.read_nifti`, whose `[D, H, W]` is the RAS-canonical `(z, y, x)` order.
            spacing: Voxel size in mm along `D`, `H`, `W` (`meta.spacing` of `read_nifti`).

        Returns:
            The labels on the input grid.

        Raises:
            InputError: Wrong number of dimensions or channels, or a spacing that is not three positive numbers.
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
        sp = tuple(float(s) for s in spacing)
        if len(sp) != 3 or not all(math.isfinite(s) and s > 0 for s in sp):  # noqa: PLR2004
            msg = f"spacing must be three positive voxel sizes in mm (D, H, W), got {spacing!r}"
            raise InputError(msg)
        with self._session():
            x = x.to(self.device, torch.float32, non_blocking=True)
            out = self._runner(x, sp)
            labels = self._check_outputs(out, {"labels": tuple(x.shape[1:])}, cast=False)["labels"]
        return VolumeResult(labels, self.class_names, self.variant.ref, self.backend, self.options)


def bench_inputs(model: VolumeModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: a synthetic CT-like volume `[H, W, W]` at 1.5 mm (H slices of W x W voxels)."""
    g = torch.Generator().manual_seed(0)
    vol = torch.randn((model.channels, size.height, size.width, size.width), generator=g) * 300.0
    return (vol.to(model.device), (1.5, 1.5, 1.5)), {}


def load_nnunet(  # noqa: PLR0913 - the arguments of `load` + the folds
    folder: str | os.PathLike[str],
    *,
    folds: Sequence[int] | None = None,
    mode: ModeArg = "auto",
    device: str | torch.device | None = None,
    dtype: Dtype | torch.dtype | None = None,
    step: float = 0.5,
    mirror: bool = True,
) -> VolumeModel:
    """Load a trained nnU-Net v2 model folder (`<dataset>/<trainer>__<plans>__<configuration>`).

    The folder holds `plans.json`, `dataset.json` and `fold_<k>/checkpoint_final.pth`; nnU-Net v2's own inference
    (its resampling, normalisation, Gaussian sliding window, mirroring, fold ensemble) runs on the TotalSegmentator
    engine. Only `PlainConvUNet` 3D configurations with label (not region) training are supported. Not hash-checked.

    Args:
        folder: The model folder.
        folds: Folds to ensemble; `None`: every `fold_<k>` with a `checkpoint_final.pth`.
        mode: As `load`.
        device: As `load`.
        dtype: As `load`.
        step: Sliding-window step (fraction of the patch; nnU-Net's default 0.5).
        mirror: Test-time mirroring along the axes the checkpoint allows (nnU-Net's default).

    Raises:
        WeightsError: Missing files, or an architecture / configuration the engine does not implement.
    """
    p = Path(os.fspath(folder)).expanduser().resolve()
    if folds is None:
        folds = sorted(int(d.name[5:]) for d in p.glob("fold_*") if (d / "checkpoint_final.pth").is_file())
        if not folds:
            msg = f"{str(folder)!r}: no fold_<k>/checkpoint_final.pth"
            raise WeightsError(msg)
    from fastforks.volume.models.totalsegmentator.model import weight_names  # noqa: PLC0415 - the family's layout

    part = f"{p.parent.name}/{p.name}"
    card, variant = registry.resolve(TASK, "totalsegmentator-total-3mm")
    params = {"parts": (part,), "folds": tuple(folds), "spacing": None, "step": step, "mirror": mirror, "classes": None}
    weights = tuple(FileSpec(n, "0" * 64) for n in weight_names((part,), tuple(folds)))
    variant = dataclasses.replace(variant, weights=weights, params=params)
    return core.load_variant(
        TASK, card, variant, VolumeModel, mode=mode, device=device, dtype=dtype, weights=p.parent.parent
    )


list_models, model_card, load = task_api(TASK, VolumeModel)
