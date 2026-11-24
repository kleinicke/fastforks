"""Tracking API: `list_models`, `model_card`, `load`, `TrackingModel`, `TrackingResult`."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, deferred_ranges, to_rgb_batch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs
from fastforks._core.upload import to_device

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fastforks.tracking._runner import Runner

__all__ = ["Backend", "TrackingModel", "TrackingResult", "list_models", "load", "model_card"]

TASK = "tracking"
VideoLike = "Sequence[ImageLike] | np.ndarray | torch.Tensor"
BENCH_FRAMES = 24  # fixed: frames of the `fastforks bench` video
BENCH_GRID = 16  # fixed: its queries, a 16 x 16 grid on frame 0


@dataclass(frozen=True)
class TrackingResult(NumpyOutputs):
    """Tracks of the query points through a video, float32 on the model's device.

    Attributes:
        tracks: `[T, Q, 2]` `(x, y)` position of every query in every frame, in input pixels (pixel centres at
            integer coordinates). Where a point is occluded, the model's guess.
        visibility: `[T, Q]` in `[0, 1]`: the model's probability that the point is visible in the frame, `None` if
            the card does not list it.
        model: Versioned model id, e.g. `"tapnext@1"`.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults (they change the numerics).
    """

    tracks: torch.Tensor
    visibility: torch.Tensor | None
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def visible(self) -> torch.Tensor | None:
        """`[T, Q]` bool: `visibility > 0.5` (the original models' visible flag), `None` without a visibility."""
        return None if self.visibility is None else self.visibility > 0.5  # noqa: PLR2004 - a probability


class TrackingModel(core.Model):
    """A loaded point-tracking model. Call it with a video and query points.

    It is deliberately not an `nn.Module`: calls take videos of any supported type and size and do the resizing.
    `module()` is the same model as an `nn.Module` for `torch.compile` / `torch.export`, and `export_onnx` writes it
    to ONNX (for TensorRT).

    An instance is not reentrant: calls from several threads are serialised (use one instance per thread to run them
    in parallel).

    Has the attributes of `fastforks.Model`.
    """

    _runner: Runner

    def module(self) -> torch.nn.Module:
        """The whole model as an `nn.Module`, for `torch.compile` and `torch.export`.

        `forward(video, queries)` takes RGB `[T, 3, H, W]` float32 frames in `[0, 1]` and `[Q, 3]` float32 `(t, x, y)`
        queries on the model's device and returns `{"tracks": [T, Q, 2], "visibility": [T, Q]}` float32, as a
        `TrackingModel` call does. Call it under `torch.inference_mode()` (or `no_grad`).

        Raises:
            BackendUnavailableError: The backend has no exportable module.
        """
        return _Module(super().module())

    def example_inputs(
        self, height: int | None = None, width: int | None = None, *, batch: int = 1
    ) -> tuple[torch.Tensor, ...]:
        """`module()` inputs: a synthetic video and a 4 x 4 query grid on frame 0.

        The video has `8 * batch` frames of a texture moving 2 px per frame.

        Args:
            height: Frame height in pixels; default (with `width`): `size_range().export_size`.
            width: Frame width in pixels.
            batch: Multiplies the frame count (a video is one call).

        Returns:
            `(video [T, 3, H, W], queries [16, 3])`.
        """
        from fastforks._core.export import resolve_size  # noqa: PLC0415 - not needed for inference

        height, width = resolve_size(self, height, width)
        return synthetic_video(8 * batch, height, width, self.device), grid_queries(height, width, 4, self.device)

    def __call__(self, video: Sequence[ImageLike] | np.ndarray | torch.Tensor, queries: Any) -> TrackingResult:  # noqa: ANN401 - array-like
        """Track query points through a video from each query's frame on.

        Online models (see the card's limitations) track forward only: before a query's frame its track holds no
        meaningful position.

        ```python
        result = model(frames, [[0, 120.0, 80.0], [10, 300.0, 200.0]])  # (t, x, y) per query
        result.tracks[:, 0]  # [T, 2] (x, y) of the first query in every frame
        result.visible[:, 0]  # [T] bool
        ```

        Args:
            video: The frames in order: a sequence of images (paths, PIL, numpy HWC, torch CHW; one size), or a
                `[T, H, W, 3]` numpy / `[T, 3, H, W]` torch batch.
            queries: `[Q, 3]` (or `[3]`) `(t, x, y)`: the frame index of each query point and its position in pixels
                of that frame (pixel centres at integer coordinates).

        Returns:
            The `TrackingResult`, `[T, Q, ...]` (`[T, ...]` for a single `[3]` query).

        Raises:
            InputError: An empty or mixed-size video, too small frames, or queries that are not `[Q, 3]`, have a
                frame index outside the video or a non-integer one, or a position outside the frames (x and y swapped).
        """
        if isinstance(video, str | bytes) or not hasattr(video, "__len__") or not hasattr(video, "__getitem__"):
            msg = f"video must be a sequence of frames or a [T, ...] frame batch, got {type(video).__name__}"
            raise InputError(msg)
        n = len(video)
        if n == 0:
            msg = "video has no frames"
            raise InputError(msg)
        first = self._frame(video, 0)
        size = (int(first.shape[-2]), int(first.shape[-1]))

        def frame(t: int) -> torch.Tensor:
            x = first if t == 0 else self._frame(video, t)
            if tuple(x.shape[-2:]) != size:
                msg = f"video frames must have one size: frame 0 is {size}, frame {t} is {tuple(x.shape[-2:])}"
                raise InputError(msg)
            return x

        q, single = _queries(queries, n)
        _check_in_image(q, size)
        with self._session(), deferred_ranges():  # one range check for all frames, not one sync per frame
            out = self._runner(frame, n, size, q.to(self.device))
        maps = self._check_outputs(out, {"tracks": (n, q.shape[0], 2), "visibility": (n, q.shape[0])})
        if single:
            maps = {k: v[:, 0] for k, v in maps.items()}
        return TrackingResult(maps["tracks"], maps.get("visibility"), self.variant.ref, self.backend, self.options)

    def _frame(self, video: Sequence[ImageLike] | np.ndarray | torch.Tensor, t: int) -> torch.Tensor:
        """Frame `t` as a checked `[1, 3, H, W]` batch on the device."""
        v = video[t]
        if isinstance(v, torch.Tensor):  # a host frame: pinned and async, not a copy that waits for the GPU
            v = to_device(v, self.device)
        x = to_rgb_batch(cast("ImageLike", v), self.device)
        if x.shape[0] != 1:
            msg = f"video frame {t} is a batch of {x.shape[0]} images; pass one image per frame"
            raise InputError(msg)
        self._check_size(x)
        return x


def _queries(queries: Any, n: int) -> tuple[torch.Tensor, bool]:  # noqa: ANN401 - array-like
    """`[Q, 3]` float32 CPU queries and whether a single `[3]` query was given.

    Raises:
        InputError: Not a finite real `[Q, 3]` / `[3]` array, or a frame index that is not an integer in `[0, n)`.
    """
    try:
        q = torch.as_tensor(np.asarray(queries) if isinstance(queries, list | tuple) else queries)
    except (TypeError, ValueError, RuntimeError) as e:
        msg = f"queries: not a numeric array ({e})"
        raise InputError(msg) from e
    if q.is_complex() or q.dtype == torch.bool:
        msg = f"queries must be real numbers, got {q.dtype}"
        raise InputError(msg)
    q = q.detach().cpu().double()  # not .to("cpu", float64): from MPS (no float64) that gives wrong values (torch 2.14)
    single = q.ndim == 1
    if single:
        q = q[None]
    if q.ndim != 2 or q.shape[1] != 3 or q.shape[0] == 0:  # noqa: PLR2004 - [Q, (t, x, y)]
        msg = f"queries must be [Q, 3] (t, x, y) with Q >= 1, got shape {tuple(q.shape)}"
        raise InputError(msg)
    if not bool(torch.isfinite(q).all()):
        msg = "queries have non-finite values"
        raise InputError(msg)
    t = q[:, 0]
    if bool((t != t.round()).any()) or bool((t < 0).any()) or bool((t >= n).any()):
        bad = t[(t != t.round()) | (t < 0) | (t >= n)][:4].tolist()
        msg = f"query frame indices must be integers in [0, {n}), got {bad}"
        raise InputError(msg)
    return q.float(), single


def synthetic_video(frames: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """`[T, 3, H, W]` in `[0, 1]`: the synthetic image moving left by 2 px per frame."""
    wide = synthetic_image(height, width + 2 * frames, device)[0]
    return torch.stack([wide[:, :, 2 * t : 2 * t + width] for t in range(frames)])


def grid_queries(height: int, width: int, n: int, device: torch.device) -> torch.Tensor:
    """`[n * n, 3]` queries on frame 0: an `n x n` grid of pixel centres inside the frame."""
    ys = (torch.arange(n, dtype=torch.float32) + 0.5) * height / n - 0.5
    xs = (torch.arange(n, dtype=torch.float32) + 0.5) * width / n - 0.5
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((torch.zeros(n * n), x.flatten(), y.flatten()), -1).to(device)


class _Module(torch.nn.Module):
    """The runner's module with float32 outputs."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net.eval().requires_grad_(requires_grad=False)  # inference only
        self.eval()

    def forward(self, video: torch.Tensor, queries: torch.Tensor) -> dict[str, torch.Tensor]:
        """`{"tracks": [T, Q, 2], "visibility": [T, Q]}` float32 of `[T, 3, H, W]` frames and `[Q, 3]` queries."""
        return {k: v.float() for k, v in self.net(video, queries).items()}


def bench_inputs(model: TrackingModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: a synthetic video of 24 frames at the size and a 16 x 16 query grid on frame 0.

    The texture moves 2 px per frame.
    """
    video = synthetic_video(BENCH_FRAMES, size.height, size.width, model.device)
    return (video, grid_queries(size.height, size.width, BENCH_GRID, model.device)), {}


list_models, model_card, load = task_api(TASK, TrackingModel)


def _check_in_image(q: torch.Tensor, size: tuple[int, int]) -> None:
    """Queries are `(t, x, y)` in pixels: raise for a point outside the frame (x, y swapped), warn for 0..1 fractions.

    Raises:
        InputError: A query position outside the `H x W` frame.
    """
    h, w = size
    x, y = q[:, 1], q[:, 2]
    if bool((x < -0.5).any() or (x > w - 0.5).any() or (y < -0.5).any() or (y > h - 0.5).any()):  # noqa: PLR2004
        msg = (
            f"queries are (t, x, y) with x, y in pixels of the {w}x{h} (WxH) frames; "
            "got a point outside them (x and y swapped?)"
        )
        raise InputError(msg)
    if min(h, w) >= 32 and bool((x <= 1).all() and (y <= 1).all()):  # noqa: PLR2004 - not tiny test frames
        warnings.warn(
            "all query positions are <= 1 pixel: queries are (t, x, y) in pixels, not fractions", stacklevel=3
        )
