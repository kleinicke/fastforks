"""Frame-interpolation API: `list_models`, `model_card`, `load`, `InterpolationModel` and `InterpolationResult`."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_pair
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, to_rgb_pair, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs
from fastforks.interpolation._runner import crop, scale

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.interpolation._runner import Runner

__all__ = ["Backend", "InterpolationModel", "InterpolationResult", "list_models", "load", "model_card"]

TASK = "interpolation"


@dataclass(frozen=True)
class InterpolationResult(NumpyOutputs):
    """Output of a frame-interpolation model: float32 on the model's device, the size of the inputs.

    Attributes:
        frame: The interpolated RGB frame `[3, H, W]` (`[B, 3, H, W]` for batches) in `[0, 1]`.
        model: Versioned model id, e.g. `"rife-4.26@1"`.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults (they change the numerics).
    """

    frame: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)


class InterpolationModel(core.Model):
    """A loaded frame-interpolation model. Call it with two images (or two batches) of the same size.

    It is deliberately not an `nn.Module`: calls take images of any supported type and size and do the scaling,
    padding and cropping. `module()` is the same model as an `nn.Module` for `torch.compile` / `torch.export`, and
    `export_onnx` writes it to ONNX (for TensorRT).

    An instance is not reentrant: calls from several threads are serialised (use one instance per thread to run them
    in parallel).

    Has the attributes of `fastforks.Model`.
    """

    _runner: Runner

    def module(self) -> torch.nn.Module:
        """The whole model as an `nn.Module`, for `torch.compile` and `torch.export`.

        `forward(image1, image2)` takes RGB `[B, 3, H, W]` float32 tensors in `[0, 1]` on the model's device and
        returns the middle frame `[B, 3, H, W]` float32, as `InterpolationModel(image1, image2).frame` does for a batch
        (time 0.5; scaling, padding and cropping included). Call it under `torch.inference_mode()` (or `no_grad`).

        Raises:
            BackendUnavailableError: The backend has no exportable module.
        """
        return _PortableModule(self._runner, super().module())

    def __call__(self, image1: ImageLike, image2: ImageLike, *, timestep: float = 0.5) -> InterpolationResult:
        """Interpolate the frame between `image1` and `image2`.

        ```python
        result = model("frame_0001.png", "frame_0002.png")
        result.frame  # [3, H, W] in [0, 1]
        ```

        Args:
            image1: First frame (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`); a
                batch gives `[B, 3, H, W]` frames.
            image2: Second frame, same size and type.
            timestep: Time of the output frame in `[0, 1]`: 0 is `image1`, 1 is `image2`.

        Returns:
            The `InterpolationResult`.

        Raises:
            InputError: Unsupported, mismatched or too small inputs, or a `timestep` outside `[0, 1]`.
        """
        if not (isinstance(timestep, (int, float)) and math.isfinite(timestep) and 0 <= timestep <= 1):
            msg = f"timestep must be a number in [0, 1], got {timestep!r}"
            raise InputError(msg)
        t1, t2 = to_rgb_pair(image1, image2, self.device, ("image1", "image2"))
        self._check_size(t1)
        maps = unbatch(self._batched(functools.partial(self._run, timestep=float(timestep)), t1, t2), image1)
        return InterpolationResult(maps["frame"], self.variant.ref, self.backend, self.options)

    def _run(self, image1: torch.Tensor, image2: torch.Tensor, *, timestep: float) -> dict[str, torch.Tensor]:
        """Scale, pad, run, crop and check one chunk."""
        r = self._runner
        io = self.variant.io
        (p1, region), (p2, _) = r.pad(scale(image1, io)), r.pad(scale(image2, io))
        out = {k: crop(v, region) for k, v in r(p1, p2, timestep).items()}
        return self._check_outputs(out, {"frame": tuple(image1.shape)})


class _PortableModule(torch.nn.Module):
    """Scaling, padding, the portable runner (its network and precision) and cropping, at time 0.5."""

    def __init__(self, runner: Runner, net: torch.nn.Module) -> None:
        """Wrap a portable runner; `net` (its `Runner.module()`) holds the parameters."""
        super().__init__()
        self.net = net.eval().requires_grad_(requires_grad=False)  # inference only
        self._runner = runner
        self.eval()

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Middle frame `[B, 3, H, W]` float32 of RGB `[B, 3, H, W]` pairs in `[0, 1]`."""
        r = self._runner
        io = r.ctx.io
        (p1, region), (p2, _) = r.pad(scale(image1, io)), r.pad(scale(image2, io))
        return crop(r(p1, p2, 0.5)["frame"], region).float()


def bench_inputs(model: InterpolationModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: a synthetic textured pair `[1, 3, H, W]` (the second image shifted by 8 px)."""
    return synthetic_pair(size.height, size.width, model.device), {}


list_models, model_card, load = task_api(TASK, InterpolationModel)
