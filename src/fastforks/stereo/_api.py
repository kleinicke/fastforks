"""Stereo API: `list_models`, `model_card`, `load`, and the `StereoModel` / `StereoResult` types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_pair
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, _as_tensor, is_batch, to_rgb_pair, unbatch
from fastforks._core.model import Backend
from fastforks.stereo._runner import crop, scale
from fastforks.stereo.calibrated import (
    StereoResult,
    build_result,
    plain_result,
    resolve_geometry,
    run_views,
    view_names,
)
from fastforks.stereo.pipeline import run_many

if TYPE_CHECKING:
    import os

    import torch

    from fastforks._core.pipeline import OnError, Pipeline
    from fastforks.stereo._runner import Runner
    from fastforks.stereo.calibrated import Rig, Views
    from fastforks.stereo.pipeline import Images

__all__ = ["Backend", "StereoModel", "StereoResult", "list_models", "load", "model_card"]

TASK = "stereo"


class StereoModel(core.Model):
    """A loaded stereo model. Call it with a left and a right image (or batches of them).

    It is deliberately not an `nn.Module`: calls take images of any supported type and size and do the scaling,
    padding and cropping. `module()` is the same model as an `nn.Module` for `torch.compile` / `torch.export`, and
    `export_onnx` writes the portable one to ONNX (for TensorRT).

    An instance is not reentrant: calls from several threads are serialised (use one instance per thread to run them
    in parallel).

    Has the attributes of `fastforks.Model`.
    """

    _list_images = 2

    _runner: Runner

    def module(self) -> torch.nn.Module:
        """The whole model as an `nn.Module`, for `torch.compile` and `torch.export`.

        `forward(left, right)` takes RGB `[B, 3, H, W]` float32 tensors in `[0, 1]` on the model's device and returns
        the disparity `[B, H, W]` float32, as `StereoModel(left, right).disparity` does for a batch (scaling, padding
        and cropping included). Portable: plain PyTorch ops (also for ONNX, `export_onnx`). Kernels: one opaque custom
        op that runs the kernels; it compiles without graph breaks, and an exported program runs in this process only.
        Call it under `torch.inference_mode()` (or `no_grad`) like any inference module.
        """
        from fastforks.stereo._module import KernelsModule, PortableModule  # noqa: PLC0415 - registers the custom op

        return KernelsModule(self) if self.backend.mode == "kernels" else PortableModule(self._runner)

    def __call__(  # noqa: PLR0913
        self,
        left: ImageLike,
        right: ImageLike,
        *,
        rig: Rig | None = None,
        views: Views = "left",
        size: tuple[int, int] | None = None,
        focal: float | None = None,
    ) -> StereoResult:
        """Estimate the disparity of a stereo pair, or depth and point clouds of a calibrated one.

        ```python
        result = model("left.png", "right.png")  # rectified pair -> StereoResult.disparity
        result = model("left.png", "right.png", rig="calib.yaml")  # unrectified + calibration -> depth, point cloud
        result = model(left, right, rig=rig, views="both")  # result.right: the right camera's disparity, depth, cloud
        ```

        With `rig` the pair is rectified (maps cached per calibration, size and device), the model runs, and the
        disparity becomes depth in the original left image: the same numbers as `rectify` -> `remap` -> model ->
        `depth` -> `unrectify` by hand. Holes are NaN. Each view's maps are in `result.left` / `result.right`
        (`StereoView`: `disparity`, `confidence`, `extras`, `depth(euclidean=)`, `points()`, `save()`);
        `result.disparity`, `result.depth()`, ... are those of the main view `result.view` (the one that was run,
        left for `"both"`).

        Args:
            left: Left image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`); a
                batch gives `[B, H, W]` maps, also with `rig` and `views`.
            right: Right image, same size and type.
            rig: The calibration (default: none, the pair is taken as rectified): a calibration file
                (`stereo.geometry.read_calib`), a `StereoRig`, its `Rectification`, or a `RectifiedRig`.
            views: `"left"` (default), `"right"` (the right camera's maps, from the mirrored pair) or `"both"`: one
                model run after the other, sharing the decoded images, the rectification and the padding. A view
                that was not run cannot be asked for afterwards.
            size: Rectified `(height, width)`; default: the image size.
            focal: Rectified focal length in pixels of the rectified size; default: the left camera's `fx` (as
                `stereo.geometry.rectify`).

        Returns:
            A `StereoResult`: `disparity` (and with `rig` `depth()`, `points()`) of the main view, `left` / `right`.

        Raises:
            InputError: Unsupported, mismatched or too small inputs, or arguments that need a `rig`.
        """
        if rig is None and views == "left":
            if size is not None or focal is not None:
                msg = "size= and focal= need a calibration: pass rig="
                raise InputError(msg)
            lt, rt = to_rgb_pair(left, right, self._device)
            self._check_size(lt)
            maps = unbatch(self._batched(self._run, lt, rt), left)
            disparity, confidence = maps.pop("disparity"), maps.pop("confidence", None)
            return plain_result(
                disparity, confidence, maps, self.variant.ref, self.backend, self.options, device=self._device
            )
        view_names(views)
        lt, rt = _as_tensor(left).to(self._device), _as_tensor(right).to(self._device)
        if lt.shape != rt.shape:
            msg = f"left and right images differ: {tuple(lt.shape)} vs {tuple(rt.shape)}"
            raise InputError(msg)
        geometry = None
        if rig is not None:
            geometry = resolve_geometry(self, rig, (lt.shape[-2], lt.shape[-1]), size=size, focal=focal)
        elif size is not None or focal is not None:
            msg = "size= and focal= need a calibration: pass rig="
            raise InputError(msg)
        lt, rt, maps = run_views(self, lt, rt, geometry, views)
        if not is_batch(left):  # one pair: `[H, W]` maps, as the plain call
            lt, rt = lt[0], rt[0]
            maps = {v: {k: None if t is None else t[0] for k, t in m.items()} for v, m in maps.items()}
        return build_result(self, geometry, (lt, rt), maps, views)

    def run_many(  # noqa: PLR0913 - keyword options with defaults
        self,
        lefts: Images,
        rights: Images,
        *,
        rig: Rig | None = None,
        views: Views = "left",
        size: tuple[int, int] | None = None,
        focal: float | None = None,
        save_to: str | os.PathLike[str] | None = None,
        save_format: str = ".tif",
        bits: int | None = None,
        max_depth: float | None = None,
        skip_existing: bool = False,
        on_error: OnError = "raise",
        prefetch: int | None = None,
        readers: int | None = None,
        writers: int | None = None,
        warmup: bool = False,
    ) -> Pipeline[StereoResult]:
        """Run many pairs: decoding, the GPU and saving overlap; results come in input order.

        ```python
        # depth TIFFs named after the lefts; work starts at construction, the loop only drains the pipeline
        for _ in model.run_many("left/", "right/", rig="calib.yaml", save_to="depth/"):
            pass
        for r in model.run_many(lefts, rights):  # 10 numpy pairs in memory
            r.disparity  # numpy [H, W]
        for r in model.run_many(lefts, rights, rig="calib.yaml", views="both"):
            r.left.depth(), r.right.depth()  # numpy [h, w] at the original pixels of each camera
        for r in model.run_many(["l0.png", "l1.png"], ["r0.png", "r1.png"]): ...
        ```

        The results equal those of `model(left, right)` (and, with `rig`, of `remap` -> model -> `depth` ->
        `unrectify`) bit for bit. Workers start in this call and run up to `prefetch` pairs ahead (with `save_to`, files
        may be written before the first iteration); iterating drains the pipeline (with `save_to`: `for _ in ...` or
        `list(...)`). To stop early use `with model.run_many(...) as p: for r in p: ...` or `p.close()`: it finishes
        the writes in progress and drops the other queued pairs; a plain `break` does not close a pipeline that is
        still referenced. `prefetch`, `readers`, `writers` and `on_error` tune it.

        Args:
            lefts: Left images: a directory (its image files, `IMAGE_SUFFIXES`), a glob pattern (`*`, `?`, `[...]`,
                `**` for subfolders), or a list / iterable of paths, numpy arrays and tensors (mixed is fine; a
                generator is consumed in a worker thread, a few pairs ahead). Directories and patterns are sorted
                naturally (`frame_2` before `frame_10`). Pairs match by position.
            rights: Right images, the same way and the same number.
            rig: The calibration of all pairs: a calibration file (`stereo.geometry.read_calib`), a `StereoRig`, its
                `Rectification` (reuses its maps), or a `RectifiedRig` for rectified images. Each pair is then
                rectified as `model(left, right, rig=rig)` does, and `StereoResult.depth()` is the depth at the
                original pixels; with `save_to` the depth is written, else the disparity.
            views: `"left"` (default), `"right"` or `"both"`: which camera's maps to produce (`StereoResult.left` /
                `.right`, as for a single call). The right view runs the mirrored pair; `"both"` runs the two views
                one after the other on the same decoded, uploaded and rectified pair. With `save_to` the right map
                is written as `<name>_right<save_format>`.
            size: Rectified `(height, width)` with a `StereoRig` (as the single call); default: the image size.
            focal: Rectified focal length in pixels with a `StereoRig`; default: the left camera's `fx`.
            save_to: Write each map to this directory instead of returning it: named after the left image (path
                relative to the directory or pattern root, the stem for lists, the 6-digit index for arrays) with
                `save_format`; two names that differ only in case raise (one file on macOS and Windows). Files appear
                atomically (no partial file after a crash).
            save_format: `.tif` (float32, deflate), `.npy`, `.pfm`, or `.png` (KITTI disparity x 256, no `rig`).
            bits: Mantissa bits kept in a TIFF (see `StereoResult.save`); None is lossless.
            max_depth: With `rig=` and `save_to=`: depth beyond this many metres is written as NaN (default: kept).
            skip_existing: Skip pairs whose output files (every view) all exist as files (resume an interrupted run).
            on_error: `"raise"` stops at the first failing pair (the error names it); `"skip"` yields a
                `StereoResult` with `error` for unreadable or invalid inputs and failed writes, and goes on. CUDA errors
                always raise.
            prefetch: Pairs in flight (decoded, on the GPU, being written); default: what fits in 1 GiB, 2-8.
            readers: Decoding threads (default: half the cores, at most 4; 0 decodes inline in the feeding thread).
            writers: Saving threads (default: as `readers`; 0 saves in the collecting thread).
            warmup: Run the model once on zeros of the first pair's size while the first pair is decoded (hides the
                first call's setup work in the kernels builds; the model's internal call counters then count one more).

        Returns:
            An iterator of `StereoResult` in input order, also a context manager (`close()` stops it); its `stats`
            attribute holds timings (setup, first result, time the GPU waited for decoding).

        Raises:
            InputError: Bad arguments, empty directories, patterns matching nothing, or different counts (raised at
                once when the counts are known, else when one side ends).
        """
        return run_many(
            self,
            lefts,
            rights,
            rig=rig,
            views=views,
            size=size,
            focal=focal,
            save_to=save_to,
            save_format=save_format,
            bits=bits,
            max_depth=max_depth,
            skip_existing=skip_existing,
            on_error=on_error,
            prefetch=prefetch,
            readers=readers,
            writers=writers,
            warmup=warmup,
        )

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Scale, pad, run, crop and check one chunk."""
        r = self._runner
        if r.io == "raw":
            out = r(left, right)
        else:
            io = self.variant.io
            (lp, region), (rp, _) = r.pad(scale(left, io)), r.pad(scale(right, io))
            out = {k: crop(v, region) for k, v in r(lp, rp).items()}
        want = (left.shape[0], *left.shape[-2:])
        return self._check_outputs(out, dict.fromkeys(out, want))


def bench_inputs(model: StereoModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: a synthetic textured pair `[1, 3, H, W]` (the right view shifted by 8 px)."""
    return synthetic_pair(size.height, size.width, model._device), {}  # noqa: SLF001


list_models, model_card, load = task_api(TASK, StereoModel)
