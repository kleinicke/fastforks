"""Multi-view depth API: `list_models`, `model_card`, `load`, `unproject` and `MultiViewModel` / `MultiViewResult`."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch

from fastforks._core import cameras
from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import CardError, InputError
from fastforks._core.image import ImageLike, to_rgb_batch, to_rgb_batches
from fastforks._core.io import MAP_SUFFIXES, check_suffix, write_map
from fastforks._core.model import Backend
from fastforks._core.outputs import MapOutputs, _array
from fastforks._core.save import CLOUD_SUFFIXES
from fastforks._core.tasks import spec
from fastforks._core.tasks.multiview import MultiViewKind
from fastforks.multiview._gaussians import Gaussians, pop_gaussians
from fastforks.multiview._runner import Runner
from fastforks.multiview._scene import SceneFormat, save_scene

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


__all__ = [
    "Backend",
    "MultiViewKind",
    "MultiViewModel",
    "MultiViewResult",
    "bench_inputs",
    "list_models",
    "load",
    "model_card",
    "unproject",
]

TASK = "multiview"
_STANDARD = spec(TASK).standard
_BATCH_DIMS = 4
_CAMERAS = ("intrinsics", "cam_to_world", "depth_range")
ArrayLike: TypeAlias = "torch.Tensor | np.ndarray | Sequence[Any]"
Images: TypeAlias = "Sequence[ImageLike] | np.ndarray | torch.Tensor"


@dataclass(frozen=True)
class MultiViewResult(MapOutputs):
    """Output of a multi-view depth model for one scene; maps are on the model's device, aligned with the inputs.

    Every map has a leading `V`: one entry per predicted view, the input view `views[i]` (all `N` views for models
    that predict every view, the reference view(s) for calibrated MVS). Outputs the model does not produce (and
    cameras that were not passed in) are `None`. Cameras: `intrinsics` are OpenCV pixel matrices of the input images,
    `cam_to_world` camera-to-world matrices with OpenCV axes (x right, y down, z forward); estimated cameras are in
    the frame of the reference view (or of the given `cam_to_world`).

    Attributes:
        depth: Depth along each view's optical axis (z), `[V, H, W]` float32 (units: see `kind`).
        world_points: 3D points `[V, H, W, 3]` float32 in the world frame (the frame of `cam_to_world`: the reference
            view's camera for estimated cameras, OpenCV axes x right, y down, z forward), same units as `depth`. Unlike
            `depth.DepthResult.points` (camera frame of the one view).
        confidence: Per-pixel confidence `[V, H, W]` float32 (higher is better; range model-specific).
        mask: Valid pixels `[V, H, W]` bool (e.g. not sky).
        intrinsics: Pinhole matrices `[V, 3, 3]` float32 in pixels of the input: estimated or passed in.
        cam_to_world: Camera-to-world matrices `[V, 4, 4]` float32 (float64 if passed in as float64, on the CPU for
            MPS): estimated or passed in.
        extras: Further model outputs by name (e.g. sky probability, depth range used), leading dim `V`.
        views: Input view index of each of the `V` entries.
        kind: `"metric"` (absolute units: metres for models that estimate the cameras, the units of the given
            `cam_to_world` for calibrated MVS) or `"scale_invariant"` (one unknown scale shared by every view, depth
            and translation alike).
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
        gaussians: Feed-forward splatting models (card output `"gaussians"`): the scene as 3D Gaussians in the world
            frame of `cam_to_world` (`gaussians.save("scene.ply")`); otherwise `None`.
    """

    depth: torch.Tensor | None
    world_points: torch.Tensor | None
    confidence: torch.Tensor | None
    mask: torch.Tensor | None
    intrinsics: torch.Tensor | None
    cam_to_world: torch.Tensor | None
    extras: Mapping[str, torch.Tensor]
    views: tuple[int, ...]
    kind: MultiViewKind
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)
    gaussians: Gaussians | None = None

    _save_fields = ("depth",)
    _save_suffixes = MAP_SUFFIXES + CLOUD_SUFFIXES

    def save(  # noqa: PLR0913
        self,
        path: str | Path,
        key: str | None = None,
        bits: int | None = None,
        *,
        view: int | None = None,
        max_depth: float | None = None,
        axes: str = "opencv",
    ) -> None:
        """Write one `[H, W]` map of one view, as `MapOutputs.save`, or the point cloud; the suffix selects the format.

        `.ply` / `.las` / `.laz` write the finite `world_points` inside `mask` (the valid pixels), coloured by the input
        images: the merged cloud of all views in the world frame, or only view `view` (`key` and `bits` are for maps);
        a result without `world_points` (calibrated MVS, e.g. PatchmatchNet) writes `unproject(depth, intrinsics,
        cam_to_world)`;
        `max_depth=` drops the points whose `depth` is larger.

        Args:
            path: Output file.
            key: Output to write: a field name or an `extras` name (default: `depth`).
            bits: Mantissa bits kept in a TIFF, 1..23, or `None` (lossless).
            view: Input view index to write (one of `views`); needed for a map when the result holds more than one
                view; a cloud of several views is merged without it.
            max_depth: Point clouds only: drop points whose `depth` is larger than this (`None`: keep all).
            axes: Point clouds only: `"opencv"` (default: x right, y down, z forward), `"opengl"` / `"blender"` (RUB),
                `"ros"` (FLU), `"pytorch3d"` (LUF) or a 3-letter code (one of R/L, U/D, F/B each).

        Raises:
            InputError: Several views and no `view=` for a map, a view the result does not hold (or a bool), a cloud
                without `world_points` (and without depth + intrinsics + cam_to_world) or with `key` / `bits`, or as
                `MapOutputs.save`.
            OptionsError: Unknown `axes`.
        """
        cloud = check_suffix(path, self._save_suffixes, type(self).__name__) in CLOUD_SUFFIXES
        if isinstance(view, bool):
            msg = f"view must be a view index (one of {self.views}), got {view!r}"
            raise InputError(msg)
        if view is None and len(self.views) > 1 and not cloud:
            msg = f"the result holds {len(self.views)} views {self.views}: pass view= (one of them) to save one view"
            raise InputError(msg)
        if view is not None and view not in self.views:
            msg = f"view={view!r} is not in this result (views: {self.views})"
            raise InputError(msg)
        if cloud:
            i = slice(None) if view is None else slice(self.views.index(view), self.views.index(view) + 1)
            self._save_cloud(path, i, key, bits, max_depth, axes)
        elif max_depth is not None:
            msg = "max_depth= is for point clouds (.ply / .las / .laz); a map keeps every pixel"
            raise InputError(msg)
        elif view is None:
            super().save(path, key, bits)
        else:
            tensors = self._tensors()
            t = tensors.get(key or "depth")
            if not isinstance(t, torch.Tensor):
                msg = f"no output {key or 'depth'!r} to save (available: {sorted(tensors)})"
                raise InputError(msg)
            write_map(os.fspath(path), _array(t[self.views.index(view)]), bits, self._description(key or "depth"))

    def _save_cloud(  # noqa: PLR0913, PLR0917
        self, path: str | Path, i: slice, key: str | None, bits: int | None, max_depth: float | None, axes: str
    ) -> None:
        """Write the finite `world_points[i]` inside `mask[i]` (the valid pixels), coloured by the images."""
        if key is not None or bits is not None:
            msg = "a point cloud writes world_points: key= and bits= are for maps (.tif / .npy / .pfm)"
            raise InputError(msg)
        from fastforks.stereo.geometry.ply import check_max_depth, write_cloud  # noqa: PLC0415 - as `fastforks.save`

        check_max_depth(max_depth)
        points, units = self._cloud_points(i)
        if max_depth is not None:
            if self.depth is None:
                msg = "max_depth= needs the result's depth"
                raise InputError(msg)
            points = points.masked_fill((self.depth[i] > max_depth).to(points.device)[..., None], float("nan"))
        image = self.__dict__.get("_image")
        colors = None if image is None else image[i].movedim(-3, -1)
        write_cloud(os.fspath(path), points, colors, axes=axes, frame="world", units=units)

    def _cloud_points(self, i: slice) -> tuple[torch.Tensor, str]:
        """`world_points[i]` (or the unprojected depth) with NaN outside `mask`, and their units."""
        units = "metres" if self.kind == "metric" else "arbitrary (not metric)"
        if self.world_points is not None:
            points = self.world_points[i]
        elif self.depth is not None and self.intrinsics is not None and self.cam_to_world is not None:
            points = unproject(self.depth[i], self.intrinsics[i], self.cam_to_world[i])  # e.g. PatchmatchNet
            units = "units of the input cam_to_world"  # calibrated: the scale is the user's, not metres
        else:
            msg = (
                f"{self.model} returned no world_points and no depth with intrinsics and cam_to_world to write as a"
                " point cloud: save a map (.tif / .npy / .pfm)"
            )
            raise InputError(msg)
        if self.mask is not None:
            points = points.masked_fill(~self.mask[i, ..., None].to(points.device), float("nan"))  # float64: CPU points
        return points, units

    def save_scene(
        self,
        folder: str | Path,
        format: SceneFormat,  # noqa: A002 - as `read_calib`
        *,
        image_names: Sequence[str | os.PathLike[str]] | None = None,
        max_points: int | None = 100_000,
    ) -> None:
        """Write the views as a scene for 3D Gaussian splatting / NeRF training or COLMAP MVS.

        `"colmap"`: a text model `sparse/0/{cameras,images,points3D}.txt` (PINHOLE cameras, one per distinct
        intrinsics; world-to-camera poses; points without tracks), the images in `images/` (gsplat, COLMAP, `ns-train
        ... colmap`). `"nerfstudio"`: `transforms.json` (per-frame PINHOLE intrinsics, camera-to-world with OpenGL
        camera axes) and its `ply_file_path` `sparse_pc.ply` (splatfacto's initial points). Both store pixel
        centres at +0.5 (`cx = K[0, 2] + 0.5`) and use the result's world frame for poses and points; both formats
        can share one folder. Only the `views` the result holds are written.

        Args:
            folder: Output folder (created; existing files of the same names are replaced).
            format: `"colmap"` or `"nerfstudio"` (the `read_calib` names of these formats).
            image_names: One image path per view, relative to `folder` (e.g. the original files,
                `images/IMG_0001.JPG`; COLMAP stores them relative to `images/`): referenced, not written. `None`
                writes the input images as lossless `images/frame_00001.png` ... (numbered by input view).
            max_points: Points kept of the finite `world_points` inside `mask` (or the depth unprojected with
                `intrinsics` and `cam_to_world`), coloured by the images, drawn at random with a fixed seed;
                `None` keeps all.

        Raises:
            InputError: No `intrinsics` or `cam_to_world`, no input images and no `image_names`, `image_names` not one
                relative path per view, `max_points` not a positive int, or no points (as `save` of a cloud).
            OptionsError: Unknown `format`.
        """
        save_scene(self, folder, format, image_names, max_points)

    @property
    def extrinsics(self) -> torch.Tensor | None:
        """World-to-camera matrices `[V, 3, 4]` (`[R | t]`, the VGGT / COLMAP convention), from `cam_to_world`."""
        if self.cam_to_world is None:
            return None
        rt = self.cam_to_world[:, :3, :3].transpose(1, 2)  # rigid: inverse rotation = transpose
        return torch.cat([rt, -rt @ self.cam_to_world[:, :3, 3:]], dim=2)


def unproject(depth: torch.Tensor, intrinsics: torch.Tensor, cam_to_world: torch.Tensor | None = None) -> torch.Tensor:
    """World points `[V, H, W, 3]` of depth maps `[V, H, W]` (camera-frame points if `cam_to_world` is `None`).

    Args:
        depth: z-depth per view.
        intrinsics: `[V, 3, 3]` (or `[3, 3]`) OpenCV pixel matrices (pixel centres at integer coordinates).
        cam_to_world: `[V, 4, 4]` camera-to-world matrices, or `None`.
    """
    v, h, w = depth.shape
    k = intrinsics.to(depth).expand(v, 3, 3)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(1, -1, 3)  # [1, HW, 3]
    rays = pix @ torch.linalg.inv(k).transpose(1, 2)  # [V, HW, 3], z = 1
    pts = rays * depth.reshape(v, -1, 1)
    if cam_to_world is not None:
        c2w = cam_to_world if cam_to_world.dtype == torch.float64 else cam_to_world.to(depth)  # float64: UTM-sized
        # to the CPU before float64 (MPS has no float64)
        pts = pts.to(c2w.device).to(c2w.dtype) @ c2w[:, :3, :3].transpose(1, 2) + c2w[:, None, :3, 3]
    return pts.reshape(v, h, w, 3)


class MultiViewModel(core.Model):
    """A loaded multi-view depth model. Call it with the views of one scene (and the cameras the model uses).

    Has the attributes of `fastforks.Model`.
    """

    _runner: Runner

    def __call__(
        self,
        images: Images,
        *,
        intrinsics: ArrayLike | None = None,
        cam_to_world: ArrayLike | None = None,
        reference: int | Sequence[int] = 0,
        depth_range: tuple[float, float] | None = None,
    ) -> MultiViewResult:
        """Estimate depth (and whatever else the model produces) for the views of one scene.

        Args:
            images: The `N` views, all the same size: a sequence of images (path, PIL, numpy HWC, torch CHW), or one
                `[N, H, W, 3]` numpy / `[N, 3, H, W]` torch batch.
            intrinsics: `[3, 3]` (shared) or `[N, 3, 3]` OpenCV pixel matrices of the views.
            cam_to_world: `[N, 4, 4]` camera-to-world matrices, OpenCV camera axes, any world frame and unit.
            reference: Calibrated MVS (reference-only models): the view(s) to predict, each from all other views.
                Models that predict every view: the view whose camera frame is the world frame of estimated cameras.
            depth_range: `(near, far)` depth search range in the units of `cam_to_world`, for models that use one.

        Returns:
            The result; `result.views` lists the input view of each entry.

        Raises:
            InputError: Unsupported images, sizes or view count; cameras or a depth range of the wrong shape, passed
                to a model that does not use them, or missing for one that needs them; a bad reference.
        """
        r = self._runner
        x = self._images(images)
        n = x.shape[0]
        if n < r.min_views or (r.max_views is not None and n > r.max_views):
            most = "any number of" if r.max_views is None else f"at most {r.max_views}"
            msg = f"{self.variant.ref} takes at least {r.min_views} and {most} views, got {n}"
            raise InputError(msg)
        k = self._camera("intrinsics", r.intrinsics, intrinsics, n, (3, 3))
        c2w = self._camera("cam_to_world", r.cam_to_world, cam_to_world, n, (4, 4))
        run_c2w = c2w
        if c2w is not None and c2w.device.type != self.device.type:  # float64 on MPS (kept on the CPU): float32
            run_c2w = c2w.clone()  # relative to the first camera, so float32 keeps the precision
            run_c2w[:, :3, 3] -= c2w[0, :3, 3]
            run_c2w = run_c2w.to(self.device, torch.float32)
        rng = self._depth_range(depth_range)
        refs = self._references(reference, n)
        with self._session():
            # per-scene work (e.g. the features of every view) once for several references; for one, a plain call
            predict = (r.scene if len(refs) > 1 else partial(Runner.scene, r))(x, k, run_c2w, rng)
            outs = []
            gaussians = None
            for ref in refs:
                order = None if ref == 0 else torch.tensor([ref, *(i for i in range(n) if i != ref)], device=x.device)
                maps = predict(order)
                gaussians = self._gaussians(maps)
                if order is None:  # input order: no gather
                    outs.append(maps)
                    continue
                if r.per_view:
                    maps = {name: v[torch.argsort(order).to(v.device)] for name, v in maps.items()}
                outs.append(maps)
        maps = outs[0] if len(outs) == 1 else {name: torch.cat([o[name] for o in outs]) for name in outs[0]}
        views = tuple(range(n)) if r.per_view else refs
        out = self._check(maps, len(views), (x.shape[-2], x.shape[-1]))
        idx = torch.tensor(views, device=x.device)
        for name, given in (("intrinsics", k), ("cam_to_world", c2w)):
            if given is not None and name not in out:
                out[name] = given[idx.to(given.device)]
        result = MultiViewResult(
            **{name: out.pop(name, None) for name in _STANDARD if name != "gaussians"},
            extras=out,
            views=views,
            kind=self.variant.output_info["kind"],
            model=self.variant.ref,
            backend=self.backend,
            options=self.options,
            gaussians=gaussians,
        )
        object.__setattr__(result, "_image", x if views == tuple(range(n)) else x[idx])  # not a field: save colours
        return result

    def example_inputs(
        self, height: int | None = None, width: int | None = None, *, batch: int = 1
    ) -> tuple[torch.Tensor, ...]:
        """Example inputs of `module()` (as the exports trace them): views and the cameras the model requires.

        `batch` is the number of views of the one scene (raised to the model's minimum, 2 for calibrated MVS): shifted
        copies of the synthetic image `[views, 3, height, width]`, then `intrinsics` `[views, 3, 3]`, `cam_to_world`
        `[views, 4, 4]` (a sideways rig) and `depth_range` `[2]` where the model requires them (optional ones are left
        out; the required ones are a prefix of `module()`'s arguments).
        """
        from fastforks._core.export import resolve_size  # noqa: PLC0415

        height, width = resolve_size(self, height, width)
        r = self._runner
        rig = _rig(height, width, max(batch, r.min_views), self.device)
        need = [getattr(r, name) == "required" for name in _CAMERAS]
        if need != sorted(need, reverse=True):
            msg = f"{self.variant.ref}: required inputs {_CAMERAS} are not a prefix of module()'s arguments"
            raise NotImplementedError(msg)
        return (rig["images"], *(rig[name] for name, req in zip(_CAMERAS, need, strict=True) if req))

    def _images(self, images: Images) -> torch.Tensor:
        if isinstance(images, (np.ndarray, torch.Tensor)):
            if images.ndim != _BATCH_DIMS:
                msg = f"a scene is a sequence of images or a [N, H, W, 3] / [N, 3, H, W] batch, got {images.ndim} dims"
                raise InputError(msg)
            x = to_rgb_batch(images, self.device)
        elif isinstance(images, (str, os.PathLike, bytes, bytearray)) or not isinstance(images, Sequence):
            msg = f"a scene is a sequence of images or one batch, got {type(images).__name__}"
            raise InputError(msg)
        else:
            if not images:
                msg = "a scene needs at least one image"
                raise InputError(msg)
            views = to_rgb_batches(images, self.device)  # one host sync for all range checks
            sizes = {tuple(v.shape[1:]) for v in views}
            if any(v.shape[0] != 1 for v in views) or len(sizes) != 1:
                msg = f"the views of a scene must be single images of one size, got {[tuple(v.shape) for v in views]}"
                raise InputError(msg)
            x = torch.cat(views)
        self._check_size(x)
        return x

    def _camera(
        self, name: str, use: str, value: ArrayLike | None, n: int, shape: tuple[int, int]
    ) -> torch.Tensor | None:
        if value is None:
            if use == "required":
                msg = f"{self.variant.ref} needs {name} ([N, {shape[0]}, {shape[1]}])"
                raise InputError(msg)
            return None
        if use == "unused":
            msg = f"{self.variant.ref} does not use {name}; call it without them"
            raise InputError(msg)
        try:  # where the input lives (numpy / lists: checked on the CPU, no GPU sync)
            t = torch.as_tensor(value)
            # float64 poses stay float64 (large world offsets); the models form relative poses before any float32 cast
            t = t.to(torch.float64 if name == "cam_to_world" and t.dtype == torch.float64 else torch.float32)
        except (TypeError, ValueError, RuntimeError) as e:
            msg = f"{name} must be a [N, {shape[0]}, {shape[1]}] array: {e}"
            raise InputError(msg) from e
        if tuple(t.shape[-2:]) != shape or t.ndim not in {2, 3} or (t.ndim == 2 and name != "intrinsics"):  # noqa: PLR2004
            msg = f"{name} must be a [N, {shape[0]}, {shape[1]}] array, got shape {tuple(t.shape)}"
            raise InputError(msg)
        t = t.reshape(-1, *shape)
        if t.shape[0] == 1 and name == "intrinsics":
            t = t.expand(n, *shape)
        if t.shape[0] != n:
            msg = f"{name}: expected {n} matrices, got shape {tuple(t.shape)}"
            raise InputError(msg)
        if not bool(cameras.valid(t, name)):
            msg = f"{name}: {cameras.problem(t, name)}"
            raise InputError(msg)
        if t.dtype == torch.float64 and self.device.type == "mps":  # no float64 on MPS: kept on the CPU
            return t.cpu().contiguous()
        return t.to(self.device).contiguous()

    def _depth_range(self, value: tuple[float, float] | None) -> torch.Tensor | None:
        use = self._runner.depth_range
        if value is None:
            if use == "required":
                msg = f"{self.variant.ref} needs depth_range=(near, far)"
                raise InputError(msg)
            return None
        if use == "unused":
            msg = f"{self.variant.ref} does not use a depth range; call it without one"
            raise InputError(msg)
        try:
            t = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        except (TypeError, ValueError, RuntimeError) as e:
            msg = f"depth_range must be (near, far): {e}"
            raise InputError(msg) from e
        if t.numel() != 2 or not torch.isfinite(t).all() or not 0 < t[0] < t[1]:  # noqa: PLR2004
            msg = f"depth_range must be finite (near, far) with 0 < near < far, got {value}"
            raise InputError(msg)
        return t.to(self.device)

    def _references(self, reference: int | Sequence[int], n: int) -> tuple[int, ...]:
        if isinstance(reference, (int, np.integer)) and not isinstance(reference, bool):
            refs = (int(reference),)
        elif self._runner.per_view:
            msg = f"{self.variant.ref} predicts every view: reference must be one int (the world-frame view)"
            raise InputError(msg)
        elif isinstance(reference, Sequence) and reference and all(isinstance(i, (int, np.integer)) for i in reference):
            refs = tuple(int(i) for i in reference)
        else:
            msg = f"reference must be an int or a non-empty sequence of ints, got {reference!r}"
            raise InputError(msg)
        if not all(0 <= i < n for i in refs) or len(set(refs)) != len(refs):
            msg = f"reference views {refs} must be distinct indices in [0, {n})"
            raise InputError(msg)
        return refs

    def _gaussians(self, maps: dict[str, torch.Tensor]) -> Gaussians | None:
        """Pop the runner's `gaussians/*` entries into `Gaussians` (models that predict every view: one reference)."""
        g = pop_gaussians(maps)
        if g is None:
            return None
        if "gaussians" not in self.variant.outputs:
            msg = f"{self.variant.ref}: runner returned gaussians; card outputs are {list(self.variant.outputs)}"
            raise CardError(msg)
        return g

    def _check(self, maps: Mapping[str, torch.Tensor], v: int, hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        """The runner's maps checked (names, shapes with leading dim `v`, bool masks) and cast to the result dtypes."""
        h, w = hw
        fixed = {
            "depth": (v, h, w),
            "confidence": (v, h, w),
            "mask": (v, h, w),
            "world_points": (v, h, w, 3),
            "intrinsics": (v, 3, 3),
            "cam_to_world": (v, 4, 4),
        }
        return self._check_outputs(maps, {name: fixed.get(name, (v, *t.shape[1:])) for name, t in maps.items()})


def _rig(h: int, w: int, n: int, device: torch.device) -> dict[str, torch.Tensor]:
    """`n` shifted copies of the synthetic image, cameras of a sideways rig (0.05 units apart), depth range 1-10."""
    image = synthetic_image(h, w, device)
    k = torch.tensor([[0.9 * w, 0, (w - 1) / 2], [0, 0.9 * w, (h - 1) / 2], [0, 0, 1]], device=device)
    c2w = torch.eye(4, device=device).repeat(n, 1, 1)
    c2w[:, 0, 3] = 0.05 * torch.arange(n, device=device)
    return {
        "images": torch.cat([torch.roll(image, 4 * i, dims=-1) for i in range(n)]),
        "intrinsics": k.expand(n, 3, 3).contiguous(),
        "cam_to_world": c2w,
        "depth_range": torch.tensor([1.0, 10.0], device=device),
    }


def bench_inputs(model: MultiViewModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: `size.views` shifted copies of the synthetic image, plus what the model requires.

    Cameras of a sideways rig (0.05 units apart) and a depth range of 1-10, passed only where the runner requires them.
    """
    rig = _rig(size.height, size.width, size.views, model.device)
    runner = model._runner  # noqa: SLF001 - the task's own runner contract
    given: dict[str, Any] = {name: rig[name] for name in _CAMERAS if getattr(runner, name) == "required"}
    if "depth_range" in given:
        given["depth_range"] = tuple(given["depth_range"].tolist())
    return (rig["images"],), given


list_models, model_card, load = task_api(TASK, MultiViewModel)
