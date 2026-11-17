"""3D volume API: `list_models`, `model_card`, `load` (also nnU-Net v2 folders), model / session / result types."""

from __future__ import annotations

import dataclasses
import logging
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
from fastforks._core.errors import CardError, InputError, ModelNotFoundError, OptionsError, WeightsError
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
    from fastforks.volume._runner import EncoderRunner, InteractiveRunner, Runner

__all__ = [
    "Backend",
    "VolumeEncoder",
    "VolumeFeatures",
    "VolumeInteractive",
    "VolumeMask",
    "VolumeModel",
    "VolumeResult",
    "VolumeSession",
    "list_models",
    "load",
    "model_card",
]

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


@dataclass(frozen=True)
class VolumeFeatures(NumpyOutputs):
    """Output of a volume encoder, on the model's device.

    Attributes:
        features: One float32 map `[C_i, ceil(D / s_i), ceil(H / s_i), ceil(W / s_i)]` per stride `s_i`, finest first.
        strides: The stride `s_i` of every map.
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    features: tuple[torch.Tensor, ...]
    strides: tuple[int, ...]
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VolumeMask(NumpyOutputs):
    """The object mask of a promptable model after a prompt, on the model's device, on the input grid.

    Attributes:
        mask: `[D, H, W]` bool.
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    mask: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)


_MIN_HU_RANGE = 255.0  # a CT of a patient spans air to bone, more than 1000 HU; 8-bit images and z-scores are below


def _volume(volume: np.ndarray | torch.Tensor, channels: int, ref: str) -> torch.Tensor:
    """`[C, D, H, W]` of a `[D, H, W]` / `[C, D, H, W]` numpy or torch volume (not yet on the device)."""
    if not isinstance(volume, (np.ndarray, torch.Tensor)):
        hint = (
            "; read a file with `vol, meta = fastforks.volume.read_nifti(path)` and pass `model(vol, meta=meta)`"
            if isinstance(volume, (str, os.PathLike))
            else ""
        )
        msg = f"volume must be a numpy array or torch tensor, got {type(volume).__name__}{hint}"
        raise InputError(msg)
    x = torch.as_tensor(np.ascontiguousarray(volume) if isinstance(volume, np.ndarray) else volume)
    if x.ndim not in _NDIM or x.is_complex() or x.dtype == torch.bool:
        msg = f"volume must be a real [D, H, W] or [C, D, H, W] array, got {x.dtype} {tuple(x.shape)}"
        raise InputError(msg)
    if x.ndim == 3:  # noqa: PLR2004
        x = x[None]
    if x.shape[0] != channels:
        msg = f"{ref} takes {channels} channel(s), got {x.shape[0]} ({tuple(x.shape)})"
        raise InputError(msg)
    return x


def _check_intensities(x: torch.Tensor, ref: str) -> None:
    """NaN / inf are an error; a value range that cannot be Hounsfield units (CT) is logged once per call."""
    lo, hi = (float(v) for v in torch.aminmax(x))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        msg = f"the volume has NaN or infinite voxels ({int((~torch.isfinite(x)).sum())}): fill or mask them first"
        raise InputError(msg)
    if "totalsegmentator" in ref and hi - lo <= _MIN_HU_RANGE:
        logging.getLogger("fastforks.volume").warning(
            "%s expects CT in Hounsfield units (air about -1000, bone above 300) but the volume's range is %.3g to "
            "%.3g: normalised or 8-bit data gives meaningless labels; pass the HU values (read_nifti scales them)",
            ref, lo, hi,
        )  # fmt: skip


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
        x = _volume(volume, self.channels, self.variant.ref)
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
            _check_intensities(x, self.variant.ref)
            out = self._runner(x, sp)
            labels = self._check_outputs(out, {"labels": tuple(x.shape[1:])}, cast=False)["labels"]
        return VolumeResult(labels, self.class_names, self.variant.ref, self.backend, self.options, meta)


class VolumeEncoder(core.Model):
    """A loaded volume encoder (pretrained 3D backbone). Call it with a volume; it returns multi-scale feature maps.

    Besides the attributes of `fastforks.Model`:

    Attributes:
        channels: Input channels (modalities) the model takes.
        strides: Stride of every feature map (input voxels per feature voxel along each axis), finest first.
        widths: Channels of every feature map.
        max_size: Largest side in voxels a call takes, or `None` (any size; memory grows with the volume).
    """

    _runner: EncoderRunner

    def __init__(self, runner: EncoderRunner, backend: Backend, options: Mapping[str, Any]) -> None:
        """Wrap a runner (built by `load`)."""
        super().__init__(runner, backend, options)
        self.channels = runner.channels
        self.strides = tuple(runner.strides)
        self.widths = tuple(runner.widths)
        self.max_size = runner.max_size

    def example_inputs(
        self, height: int | None = None, width: int | None = None, *, batch: int = 1
    ) -> tuple[torch.Tensor, ...]:
        """Return a normalised, padded 3D example for the encoder's `module()`.

        Args:
            height: Volume depth; default: the task's export size.
            width: Volume height and width.
            batch: Number of volumes.

        Returns:
            One `[batch, channels, depth, height, width]` tensor on the model's device, padded as the encoder needs.

        Raises:
            InputError: A requested side exceeds the encoder's maximum size.
        """
        from fastforks._core.export import resolve_size  # noqa: PLC0415

        height, width = resolve_size(self, height, width)
        original_height, original_width = height, width
        if self.max_size is not None and max(height, width) > self.max_size:
            msg = f"{self.variant.ref} takes sides up to {self.max_size} voxels, got {height}x{width}x{width}"
            raise InputError(msg)
        if self.max_size is not None and getattr(self._runner, "arch", None) == "vit":
            height = width = self.max_size
        else:
            multiple = getattr(self._runner, "multiple", 1)
            height = -(-height // multiple) * multiple
            width = -(-width // multiple) * multiple
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(batch, self.channels, original_height, original_width, original_width, generator=generator)
        padding = (0, width - original_width, 0, width - original_width, 0, height - original_height)
        return (torch.nn.functional.pad(x, padding).to(self.device),)

    def __call__(self, volume: np.ndarray | torch.Tensor) -> VolumeFeatures:
        """Encode a volume (no resampling: pass it at the spacing the model card names).

        Args:
            volume: `[D, H, W]` or `[C, D, H, W]` raw intensities (numpy or torch), e.g. from
                `fastforks.volume.read_nifti`; the model normalises them (z-score over the volume).

        Returns:
            The feature maps, finest first.

        Raises:
            InputError: Wrong number of dimensions or channels, or a side above `max_size`.
        """
        x = _volume(volume, self.channels, self.variant.ref)
        if self.max_size is not None and max(x.shape[1:]) > self.max_size:
            msg = f"{self.variant.ref} takes sides up to {self.max_size} voxels, got {tuple(x.shape[1:])}"
            raise InputError(msg)
        with self._session():
            x = x.to(self.device, torch.float32, non_blocking=True)
            maps = self._runner(x)["features"]
            if len(maps) != len(self.strides):
                msg = f"{self.variant.ref}: runner returned {len(maps)} feature maps for strides {self.strides}"
                raise CardError(msg)
            out = []
            for t, s, c in zip(maps, self.strides, self.widths, strict=True):
                shape = (c, *(-(-n // s) for n in x.shape[1:]))
                out.append(self._check_outputs({"features": t}, {"features": shape})["features"])
        return VolumeFeatures(tuple(out), self.strides, self.variant.ref, self.backend, self.options)


class VolumeSession:
    """One volume in a promptable model: every prompt refines the same object mask. Made by `VolumeInteractive.session`.

    Prompts are in voxels of the input volume, `(z, y, x)` = `[D, H, W]` order; `positive=False` marks background.
    """

    def __init__(self, model: VolumeInteractive, volume: torch.Tensor) -> None:
        """Start a session (use `VolumeInteractive.session`)."""
        self._model = model
        self.shape = tuple(int(n) for n in volume.shape[1:])
        with model._session():  # noqa: SLF001 - the model's locks and device guard
            self._state = model._runner.start(volume.to(model.device, torch.float32))  # noqa: SLF001

    def _run(self, prompt: str, *args: Any) -> VolumeMask:  # noqa: ANN401
        m = self._model
        with m._session():  # noqa: SLF001
            getattr(m._runner, prompt)(self._state, *args)  # noqa: SLF001
            return self._mask()

    def _mask(self) -> VolumeMask:
        m = self._model
        out = m._check_outputs({"mask": m._runner.mask(self._state)}, {"mask": self.shape}, cast=False)  # noqa: SLF001
        return VolumeMask(out["mask"], m.variant.ref, m.backend, m.options)

    @property
    def mask(self) -> VolumeMask:
        """The current mask (all False before the first prompt)."""
        with self._model._session():  # noqa: SLF001
            return self._mask()

    def add_point(self, zyx: Sequence[float], *, positive: bool = True) -> VolumeMask:
        """Click a voxel `(z, y, x)` (fractions are rounded) inside the volume; returns the updated mask.

        Raises:
            InputError: Not three finite coordinates inside the volume.
        """
        p = tuple(float(c) for c in zyx)
        n_ax = len(self.shape)
        if len(p) != n_ax or not all(
            math.isfinite(c) and 0 <= round(c) < n for c, n in zip(p, self.shape, strict=True)
        ):
            msg = f"point must be (z, y, x) inside the volume {self.shape}, got {zyx!r}"
            raise InputError(msg)
        return self._run("point", p, positive)

    def add_box(self, box: Sequence[Sequence[float]], *, positive: bool = True) -> VolumeMask:
        """Draw a box `[[z0, z1], [y0, y1], [x0, x1]]` (half-open; a 2D box spans one slice, e.g. `[z, z + 1]`).

        Raises:
            InputError: Not three `[start, end]` pairs with `0 <= start < end <= size`.
        """
        b = [tuple(float(c) for c in ax) for ax in box]
        if len(b) != len(self.shape) or not all(
            len(ax) == 2 and all(math.isfinite(c) for c in ax) and 0 <= ax[0] < ax[1] <= n  # noqa: PLR2004
            for ax, n in zip(b, self.shape, strict=True)
        ):
            msg = f"box must be [[z0, z1], [y0, y1], [x0, x1]] with 0 <= start < end <= {self.shape}, got {box!r}"
            raise InputError(msg)
        return self._run("box", b, positive)

    def add_scribble(self, mask: np.ndarray | torch.Tensor, *, positive: bool = True) -> VolumeMask:
        """Paint voxels: a `[D, H, W]` bool / 0-1 mask on the volume's grid.

        Raises:
            InputError: A different shape, or no voxel set.
        """
        s = torch.as_tensor(np.ascontiguousarray(mask) if isinstance(mask, np.ndarray) else mask)
        if tuple(s.shape) != self.shape or s.is_complex():
            msg = f"scribble must be a [D, H, W] mask of the volume's shape {self.shape}, got {tuple(s.shape)}"
            raise InputError(msg)
        s = s.to(self._model.device) != 0
        if not s.any():
            msg = "scribble has no voxels set"
            raise InputError(msg)
        return self._run("scribble", s, positive)

    def reset(self) -> None:
        """Forget every prompt and the mask."""
        with self._model._session():  # noqa: SLF001
            self._model._runner.reset(self._state)  # noqa: SLF001


class VolumeInteractive(core.Model):
    """A loaded promptable volume model: `model.session(volume)` takes point, box and scribble prompts.

    `model(volume, point=...)` (or `box=` / `scribble=`) is the one-prompt shortcut.

    Besides the attributes of `fastforks.Model`:

    Attributes:
        channels: Input channels (modalities) the model takes.
    """

    _runner: InteractiveRunner

    def __init__(self, runner: InteractiveRunner, backend: Backend, options: Mapping[str, Any]) -> None:
        """Wrap a runner (built by `load`)."""
        super().__init__(runner, backend, options)
        self.channels = runner.channels

    def example_inputs(
        self, height: int | None = None, width: int | None = None, *, batch: int = 1
    ) -> tuple[torch.Tensor, ...]:
        """A prepared network patch: image, previous mask and six prompt channels."""
        from fastforks._core.export import resolve_size  # noqa: PLC0415

        height, width = resolve_size(self, height, width)
        if (height, width) != (192, 192):
            msg = "nnInteractive exports its fixed 192x192x192 network patch; use size 192x192"
            raise InputError(msg)
        runner = cast("Any", self._runner)
        x = torch.zeros((batch, 8, 192, 192, 192), device=self.device, dtype=runner._dtype)  # noqa: SLF001
        x[:, 0] = torch.linspace(-1, 1, 192, device=self.device, dtype=x.dtype).view(1, 1, 1, 192)
        x[:, 4, 96, 96, 96] = 1
        return (x,)

    def session(self, volume: np.ndarray | torch.Tensor) -> VolumeSession:
        """Start a session on a volume (raw intensities, any spacing; the model z-scores its nonzero region).

        Args:
            volume: `[D, H, W]` or `[C, D, H, W]` (numpy or torch), e.g. from `fastforks.volume.read_nifti`.

        Raises:
            InputError: Wrong number of dimensions or channels, or no nonzero voxel.
        """
        return VolumeSession(self, _volume(volume, self.channels, self.variant.ref))

    def __call__(
        self,
        volume: np.ndarray | torch.Tensor,
        *,
        point: Sequence[float] | None = None,
        box: Sequence[Sequence[float]] | None = None,
        scribble: np.ndarray | torch.Tensor | None = None,
        positive: bool = True,
    ) -> VolumeMask:
        """Segment one object from one prompt (a new session each call).

        Args:
            volume: As `session`.
            point: `(z, y, x)` voxel, as `VolumeSession.add_point`.
            box: `[[z0, z1], [y0, y1], [x0, x1]]`, as `VolumeSession.add_box`.
            scribble: `[D, H, W]` mask, as `VolumeSession.add_scribble`.
            positive: Foreground (True) or background prompt.

        Raises:
            InputError: Not exactly one prompt, or an invalid volume or prompt.
        """
        given = [k for k, v in (("point", point), ("box", box), ("scribble", scribble)) if v is not None]
        if len(given) != 1:
            msg = f"pass exactly one of point=, box=, scribble= (got {given or 'none'})"
            raise InputError(msg)
        s = self.session(volume)
        if point is not None:
            return s.add_point(point, positive=positive)
        if box is not None:
            return s.add_box(box, positive=positive)
        return s.add_scribble(scribble, positive=positive)  # ty: ignore[invalid-argument-type]


def bench_inputs(model: VolumeModel | VolumeEncoder | VolumeInteractive, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: a synthetic CT-like volume `[H, W, W]` at 1.5 mm (H slices of W x W voxels).

    Promptable models: one positive point at the centre.
    """
    g = torch.Generator().manual_seed(0)
    vol = torch.randn((model.channels, size.height, size.width, size.width), generator=g) * 300.0
    if isinstance(model, VolumeEncoder):
        return (vol.to(model.device),), {}
    if isinstance(model, VolumeInteractive):
        return (vol.to(model.device),), {"point": tuple(n // 2 for n in vol.shape[1:])}
    return (vol.to(model.device),), {"spacing": (1.5, 1.5, 1.5)}


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
        The loaded model: a `VolumeModel`, or a `VolumeEncoder` / `VolumeInteractive` for those cards.

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
        return cast("VolumeModel", _load(str(model), mode=mode, device=device, dtype=dtype, weights=weights, **options))
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


def _model_class(model: str) -> type[VolumeModel | VolumeEncoder | VolumeInteractive]:
    """`VolumeEncoder` for cards whose outputs are feature maps, `VolumeInteractive` for masks, else `VolumeModel`."""
    try:
        _, variant = registry.resolve(TASK, model)
    except ModelNotFoundError:
        return VolumeModel  # `load` raises the error
    if "mask" in variant.outputs:
        return VolumeInteractive
    return VolumeEncoder if "features" in variant.outputs else VolumeModel


list_models, model_card, _load = task_api(TASK, _model_class)
