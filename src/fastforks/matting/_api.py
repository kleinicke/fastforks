"""Matting API: `list_models`, `model_card`, `load`, and the model / result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, to_rgb_batch, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import MapOutputs, NumpyOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np

    from fastforks.matting._runner import Runner

__all__ = ["Backend", "MattingModel", "MattingResult", "VideoMattingResult", "list_models", "load", "model_card"]

TASK = "matting"


@dataclass(frozen=True)
class MattingResult(MapOutputs):
    """Output of a matting model, on the model's device, aligned with the input image.

    `save(path)` writes `alpha` (`MapOutputs.save`): `.tif`, `.npy`, `.pfm`; for an 8-bit PNG mask write
    `fastforks.save("mask.png", (result.alpha * 255).round().byte())`.

    Attributes:
        alpha: Foreground alpha per pixel `[H, W]` float32 in `[0, 1]` (`[B, H, W]` for a batch); 1 = foreground.
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    alpha: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    _save_fields = ("alpha",)


@dataclass(frozen=True)
class VideoMattingResult(NumpyOutputs):
    """Alpha of the object masked on frame 0, for every frame of a video (`track`), on the CPU.

    The CPU, not the model's device: the result grows with the video length, the device memory does not.

    Attributes:
        alpha: `[T, H, W]` float32 foreground alpha in `[0, 1]` per frame at the video size.
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    alpha: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)


class MattingModel(core.Model):
    """A loaded matting model. Call it with an image (or a batch).

    Has the attributes of `fastforks.Model`.
    """

    _runner: Runner

    def __call__(
        self, image: ImageLike, *, trimap: ImageLike | None = None, mask: ImageLike | None = None
    ) -> MattingResult:
        """Foreground alpha of an image.

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).
            trimap: Only for trimap-guided models (`contract().inputs["trimap"] == "required"`): a gray image of the
                same size (and batch, or one for all), 0 = background, 255 (1.0 for float) = foreground, 128 (0.5) =
                unknown; read like `image` (colour: the first channel).
            mask: Only for mask-guided models (`contract().inputs["mask"] == "required"`, video matting): a gray image
                of the same size (and batch, or one for all), nonzero = the object; read like `image`.

        Returns:
            The result; batched inputs give batched maps.

        Raises:
            InputError: Unsupported or too small image; a trimap / mask missing for a model that needs one, given to
                one that does not use it, or of another size or batch.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)
        t = self._guide(trimap, mask, x)

        def run(chunk: torch.Tensor, tri: torch.Tensor | None) -> dict[str, torch.Tensor]:
            out = self._runner(chunk, tri)
            return self._check_outputs(out, {"alpha": (chunk.shape[0], *chunk.shape[-2:])})

        maps = unbatch(self._batched(run, x, t), image)
        return MattingResult(maps["alpha"], self.variant.ref, self.backend, self.options)

    def track(self, video: Sequence[ImageLike] | np.ndarray | torch.Tensor, *, mask: ImageLike) -> VideoMattingResult:
        """Matte the object masked on the first frame through a video (models with `video`, MatAnyone).

        Frames are read and processed one at a time; the alpha of each frame goes to the CPU.

        Args:
            video: The frames in order: a sequence of images (paths, PIL, numpy HWC, torch CHW; one size), or a
                `[T, H, W, 3]` numpy / `[T, 3, H, W]` torch batch.
            mask: The object on frame 0: a gray image of the frame size, nonzero = the object; read like `image`.

        Returns:
            The alpha per frame.

        Raises:
            InputError: The model does not matte video, an empty or mixed-size video, or a mask of another size.
        """
        if not self._runner.video:
            msg = f"{self.variant.ref} does not matte video"
            raise InputError(msg)
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

        m = self._guide(None, mask, first)
        if m is None or m.shape[0] != 1:
            msg = "track takes one mask (the object on frame 0)"
            raise InputError(msg)
        with self._session():
            out = self._runner.track(frame, n, size, m[0, 0])
        return VideoMattingResult(out["alpha"], self.variant.ref, self.backend, self.options)

    def _frame(self, video: Sequence[ImageLike] | np.ndarray | torch.Tensor, t: int) -> torch.Tensor:
        """Frame `t` as a checked `[1, 3, H, W]` batch on the device."""
        x = to_rgb_batch(cast("ImageLike", video[t]), self.device)
        if x.shape[0] != 1:
            msg = f"video frame {t} is a batch of {x.shape[0]} images; pass one image per frame"
            raise InputError(msg)
        self._check_size(x)
        return x

    def _guide(self, trimap: ImageLike | None, mask: ImageLike | None, x: torch.Tensor) -> torch.Tensor | None:
        """The trimap or mask the runner requires as `[B, 1, H, W]` float32 (mask: binary), checked against `x`."""
        for name, given, need in (("trimap", trimap, self._runner.trimap), ("mask", mask, self._runner.mask)):
            if given is None and need == "required":
                what = "0 background, 128 unknown, 255 foreground" if name == "trimap" else "nonzero = the object"
                msg = f"{self.variant.ref} needs a {name} (gray image: {what})"
                raise InputError(msg)
            if given is not None and need != "required":
                msg = f"{self.variant.ref} does not use a {name}; call it without one"
                raise InputError(msg)
        given = trimap if trimap is not None else mask
        if given is None:
            return None
        name = "trimap" if trimap is not None else "mask"
        t = to_rgb_batch(given, self.device)[:, :1]
        if t.shape[-2:] != x.shape[-2:] or t.shape[0] not in {1, x.shape[0]}:
            msg = (
                f"{name}: {t.shape[0]} map(s) of {tuple(t.shape[-2:])} px do not match the image(s): {x.shape[0]} of "
                f"{tuple(x.shape[-2:])} px"
            )
            raise InputError(msg)
        if mask is not None:
            t = t.ne(0).float()
        return t.expand(x.shape[0], -1, -1, -1).contiguous()

    def example_inputs(
        self, height: int | None = None, width: int | None = None, *, batch: int = 1
    ) -> tuple[torch.Tensor, ...]:
        """Example inputs of `module()` (as the exports trace them): the synthetic image, plus a required trimap / mask.

        The trimap `[batch, 1, height, width]` marks a centred box as unknown (0.5), a ring around it as foreground (1)
        and the border as background (0); the mask is its nonzero part.
        """
        (image,) = super().example_inputs(height, width, batch=batch)[:1]
        guide = _synthetic_guide(self._runner, *image.shape[-2:], self.device)
        if guide is None:
            return (image,)
        return image, guide.expand(image.shape[0], -1, -1, -1).contiguous()


def synthetic_trimap(height: int, width: int, device: torch.device) -> torch.Tensor:
    """`[1, 1, H, W]` float32 trimap: background border, foreground ring, unknown centre (`example_inputs`, bench)."""
    t = torch.zeros(1, 1, height, width, device=device)
    t[..., height // 8 : height - height // 8, width // 8 : width - width // 8] = 1.0
    t[..., height // 4 : height - height // 4, width // 4 : width - width // 4] = 0.5
    return t


def _synthetic_guide(runner: Runner, height: int, width: int, device: torch.device) -> torch.Tensor | None:
    """The synthetic trimap, its nonzero part as the mask, or `None`, as `runner` requires."""
    if runner.trimap == "required":
        return synthetic_trimap(height, width, device)
    if runner.mask == "required":
        return synthetic_trimap(height, width, device).ne(0).float()
    return None


def bench_inputs(model: MattingModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]`, plus a synthetic trimap or mask if required."""
    image = synthetic_image(size.height, size.width, model.device)
    runner = model._runner  # noqa: SLF001 - the task's own runner contract
    guide = _synthetic_guide(runner, size.height, size.width, model.device)
    if guide is None:
        return (image,), {}
    return (image,), {"trimap" if runner.trimap == "required" else "mask": guide}


list_models, model_card, load = task_api(TASK, MattingModel)
