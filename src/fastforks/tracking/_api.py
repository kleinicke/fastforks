"""Tracking API: `list_models`, `model_card`, `load`, `TrackingModel`, `TrackingResult`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, to_rgb_batch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs

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
            height: Frame height in pixels; default (with `width`): `export_size()`.
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
                frame index outside the video or a non-integer one.
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
        with self._session():
            out = self._runner(frame, n, size, q.to(self.device))
        maps = self._check_outputs(out, {"tracks": (n, q.shape[0], 2), "visibility": (n, q.shape[0])})
        if single:
            maps = {k: v[:, 0] for k, v in maps.items()}
        return TrackingResult(maps["tracks"], maps.get("visibility"), self.variant.ref, self.backend, self.options)

    def _frame(self, video: Sequence[ImageLike] | np.ndarray | torch.Tensor, t: int) -> torch.Tensor:
        """Frame `t` as a checked `[1, 3, H, W]` batch on the device."""
        x = to_rgb_batch(cast("ImageLike", video[t]), self.device)
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
    q = q.detach().to("cpu", torch.float64)
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
