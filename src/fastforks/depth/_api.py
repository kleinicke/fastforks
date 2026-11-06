"""Monocular depth API: `list_models`, `model_card`, `load`, and the `DepthModel` / `DepthResult` types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch

from fastforks._core import cameras
from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, is_batch, to_rgb_batch, unbatch
from fastforks._core.io import MAP_SUFFIXES, check_suffix, write_png_x256
from fastforks._core.model import Backend
from fastforks._core.outputs import MapOutputs
from fastforks._core.save import CLOUD_SUFFIXES
from fastforks._core.tasks import DepthKind, spec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from fastforks.depth._runner import Runner

__all__ = ["Backend", "DepthKind", "DepthModel", "DepthResult", "list_models", "load", "model_card"]

TASK = "depth"
_STANDARD = spec(TASK).standard
_VECTOR_MAPS = ("points", "normals")
ArrayLike: TypeAlias = "torch.Tensor | np.ndarray | Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]]"


@dataclass(frozen=True)
class DepthResult(MapOutputs):
    """Output of a monocular depth model; maps are on the model's device and aligned with the input image.

    Shapes are for one image; a batch adds a leading `B`. Outputs the model does not produce are `None`. Vector maps
    are channel-last (`[H, W, 3]`, as MoGe / Open3D).

    Attributes:
        depth: Depth along the optical axis (z), `[H, W]` float32: metres if `kind == "metric"`, else up to an
            unknown scale (`"scale_invariant"`) or scale and shift (`"affine_invariant"`).
        inverse_depth: Inverse depth `[H, W]` float32; for `kind == "affine_invariant_inverse"` (e.g. Depth Anything
            V2 relative, MiDaS) only up to an unknown scale and shift, so it is not `1 / depth`.
        points: Camera-frame 3D points `[H, W, 3]` float32 (x right, y down, z forward), same units as `depth`.
        intrinsics: Pinhole matrix `[3, 3]` float32 in pixels of the input: the one passed in, or the model's
            estimate (MoGe: always the camera its `points` use; it follows only `fx` of given intrinsics). OpenCV
            pixel centres (pixel `i` centred at `i`, image centre `((w - 1) / 2, (h - 1) / 2)`), as
            `fastforks.stereo.geometry` and the multi-view task.
        confidence: Per-pixel confidence `[H, W]` float32 (higher is better; range model-specific).
        mask: Valid pixels `[H, W]` bool (e.g. not sky).
        normals: Unit surface normals `[H, W, 3]` float32 in the camera frame.
        extras: Further model outputs by name (e.g. rays, sky logits); leading dim as the image (none or `B`).
        kind: What `depth` / `inverse_depth` / `points` mean (card `[output] kind`; one kind per variant).
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    depth: torch.Tensor | None
    inverse_depth: torch.Tensor | None
    points: torch.Tensor | None
    intrinsics: torch.Tensor | None
    confidence: torch.Tensor | None
    mask: torch.Tensor | None
    normals: torch.Tensor | None
    extras: Mapping[str, torch.Tensor]
    kind: DepthKind
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    _save_fields = ("depth", "inverse_depth")
    _save_suffixes = MAP_SUFFIXES + CLOUD_SUFFIXES

    def _save_float_png(self, path: str | Path, key: str | None) -> bool:
        """Write a float map as the x256 uint16 PNG; `False` (nothing written) for an integer or bool map."""
        m = self._map(key)
        if isinstance(m, np.ndarray):
            is_float = m.dtype.kind == "f"
        else:
            is_float, m = m.is_floating_point(), m.detach().float().cpu().numpy()
        if is_float:
            write_png_x256(path, m)
        return is_float

    def save(
        self,
        path: str | Path,
        key: str | None = None,
        bits: int | None = None,
        *,
        max_depth: float | None = None,
        axes: str = "opencv",
    ) -> None:
        """Write one `[H, W]` map (`MapOutputs.save`; default `depth`, else `inverse_depth`) or the point cloud.

        `.ply` / `.las` / `.laz` write the finite `points` inside `mask`, coloured by the input image (`key` and `bits`
        are for maps); a metric result without `points` but with `depth` and `intrinsics` writes the unprojected depth.
        `max_depth=` drops points with z beyond it (metres; default: all). A `.png` of a float map (`depth`,
        `inverse_depth`) is a uint16 PNG of value x 256 with 0 = invalid, as the stereo disparity PNG (clipped at
        255.996, non-finite and negative values 0): metric depth in metres keeps 1/256 m resolution. `key="mask"`
        writes the bool mask as a 1-bit PNG.

        Args:
            path: Output file; the suffix selects the format.
            key: Output to write: a field name or an `extras` name.
            bits: Mantissa bits kept in a TIFF, 1..23, or `None` (lossless).
            max_depth: Point clouds only: drop points whose z is larger than this (`None`: keep all).
            axes: Point clouds only: `"opencv"` (default: x right, y down, z forward), `"opengl"` / `"blender"` (RUB),
                `"ros"` (FLU), `"pytorch3d"` (LUF) or a 3-letter code (one of R/L, U/D, F/B each).

        Raises:
            InputError: A cloud without `points` (and without metric `depth` + `intrinsics`), of a batch or with
                `key` / `bits`, `max_depth=` for a map, or as `MapOutputs.save`.
            OptionsError: Unknown `axes`.
        """
        suffix = check_suffix(path, self._save_suffixes, type(self).__name__)
        if suffix not in CLOUD_SUFFIXES:
            if max_depth is not None:
                msg = "max_depth= is for point clouds (.ply / .las / .laz); a map keeps every pixel"
                raise InputError(msg)
            if suffix == ".png" and self._save_float_png(path, key):
                return
            super().save(path, key, bits)
            return
        if key is not None or bits is not None:
            msg = "a point cloud writes points: key= and bits= are for maps (.tif / .npy / .pfm / .png)"
            raise InputError(msg)
        from fastforks.multiview import unproject  # noqa: PLC0415 - only for a result without `points`
        from fastforks.stereo.geometry.ply import write_cloud  # noqa: PLC0415 - as `fastforks.save`

        points, mask, image = self.points, self.mask, self.__dict__.get("_image")
        if points is None:
            if self.kind != "metric" or self.depth is None or self.intrinsics is None:
                msg = (
                    f"{self.model} returned no points and no metric depth with intrinsics to write as a point cloud:"
                    " call it with intrinsics=K (metric models), or save a map (.tif / .npy / .pfm)"
                )
                raise InputError(msg)
            depth = self.depth.reshape(-1, *self.depth.shape[-2:])
            k = self.intrinsics.reshape(-1, 3, 3)
            points = unproject(depth, k).reshape(*self.depth.shape, 3)
        if points.ndim == 4:  # noqa: PLR2004 - [B, H, W, 3]
            if points.shape[0] != 1:
                msg = f"the result holds {points.shape[0]} images: save the cloud of one image per call"
                raise InputError(msg)
            points, mask = points[0], None if mask is None else mask[0]
            image = None if image is None else image[0]
        if mask is not None:
            points = points.masked_fill(~mask[..., None], float("nan"))
        colors = None if image is None else image.movedim(-3, -1)
        units = "metres" if self.kind == "metric" else "arbitrary (not metric)"
        write_cloud(str(path), points, colors, max_depth=max_depth, axes=axes, frame="camera", units=units)


class DepthModel(core.Model):
    """A loaded monocular depth model. Call it with an image (or a batch) and optionally its intrinsics.

    Has the attributes of `fastforks.Model`.
    """

    _runner: Runner

    def __call__(self, image: ImageLike, *, intrinsics: ArrayLike | None = None) -> DepthResult:
        """Estimate depth (and whatever else the model produces) for an image.

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).
            intrinsics: Pinhole matrix `[3, 3]` (or `[B, 3, 3]`; tensor, numpy array or nested lists) in pixels of
                `image`, for models that use it (see the card); `None` lets the model estimate it. OpenCV pixel
                centres (pixel `i` centred at `i`), as the estimates.

        Returns:
            The result; batched inputs give batched maps.

        Raises:
            InputError: Unsupported or too small image, intrinsics of the wrong shape, intrinsics passed to a model
                that does not use them, or missing for one that needs them.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)
        k = self._intrinsics(intrinsics, x.shape[0])
        b, (h, w) = x.shape[0], x.shape[-2:]
        maps = self._batched(self._runner, x, k)
        shapes = {
            name: (b, h, w, 3)
            if name in _VECTOR_MAPS
            else (b, 3, 3)
            if name == "intrinsics"
            else (b, h, w)
            if name in _STANDARD
            else (b, *v.shape[1:])
            for name, v in maps.items()
        }
        out = self._check_outputs(maps, shapes)
        if k is not None and "intrinsics" not in out:
            out["intrinsics"] = k
        out = unbatch(out, image)
        result = DepthResult(
            **{name: out.pop(name, None) for name in _STANDARD},
            extras=out,
            kind=self.variant.output_info["kind"],
            model=self.variant.ref,
            backend=self.backend,
            options=self.options,
        )
        object.__setattr__(result, "_image", x if is_batch(image) else x[0])  # not a field: colours of `save`
        return result

    def _intrinsics(self, k: ArrayLike | None, batch: int) -> torch.Tensor | None:
        use = self._runner.intrinsics
        if k is None:
            if use == "required":
                msg = f"{self.variant.ref} needs the camera intrinsics (intrinsics=[3, 3] pixel matrix)"
                raise InputError(msg)
            return None
        if use == "unused":
            msg = f"{self.variant.ref} does not use camera intrinsics; call it without them"
            raise InputError(msg)
        try:  # where the input lives (numpy / lists: checked on the CPU, no GPU sync); may share the caller's memory
            given = k = torch.as_tensor(k, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as e:
            msg = f"intrinsics must be a [3, 3] or [B, 3, 3] array: {e}"
            raise InputError(msg) from e
        if tuple(k.shape[-2:]) != (3, 3) or k.ndim not in {2, 3}:
            msg = f"intrinsics must be a [3, 3] or [B, 3, 3] array, got shape {tuple(k.shape)}"
            raise InputError(msg)
        k = k.reshape(-1, 3, 3)
        if k.shape[0] == 1:
            k = k.expand(batch, 3, 3)
        if k.shape[0] != batch:
            msg = f"intrinsics: expected 1 or {batch} matrices, got shape {tuple(k.shape)}"
            raise InputError(msg)
        if not bool(cameras.valid(k, "intrinsics")):
            msg = f"intrinsics: {cameras.problem(k, 'intrinsics')}"
            raise InputError(msg)
        k = k.to(self.device).contiguous()
        return k.clone() if k.data_ptr() == given.data_ptr() else k  # the result owns its matrices (9 floats each)


def bench_inputs(model: DepthModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]`, plus pinhole intrinsics if the model needs them."""
    h, w = size.height, size.width
    image = synthetic_image(h, w, model.device)
    if model._runner.intrinsics != "required":  # noqa: SLF001 - the task's own runner contract
        return (image,), {}
    k = torch.tensor([[0.9 * w, 0.0, (w - 1) / 2], [0.0, 0.9 * w, (h - 1) / 2], [0.0, 0.0, 1.0]])  # OpenCV pixels
    return (image,), {"intrinsics": k}


list_models, model_card, load = task_api(TASK, DepthModel)
