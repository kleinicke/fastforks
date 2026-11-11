"""Dense-matching API: `list_models`, `model_card`, `load`, and the `MatchingModel` / `MatchingResult` types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_pair
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, is_batch, to_rgb_batches, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs
from fastforks.matching._runner import scale

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.matching._runner import Runner

__all__ = ["Backend", "MatchingModel", "MatchingResult", "list_models", "load", "model_card"]

TASK = "matching"


@dataclass(frozen=True)
class MatchingResult(NumpyOutputs):
    """Output of a dense-matching model: float32 on the model's device, aligned with the first image.

    Attributes:
        warp: Where every pixel of the first image lies in the second, `[2, H1, W1]` (`[B, 2, H1, W1]` for batches):
            channel 0 is `x`, channel 1 is `y`, in pixels of the second image at its input resolution (pixel centres
            at integer coordinates). Meaningful only where `certainty` is high.
        certainty: `[H1, W1]` in `[0, 1]`: the model's probability that the pixel is visible in the second image
            (unthresholded), `None` if the card does not list it.
        model: Versioned model id, e.g. `"romav2@1"`.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults (they change the numerics).
    """

    warp: torch.Tensor
    certainty: torch.Tensor | None
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)


class MatchingModel(core.Model):
    """A loaded dense-matching model. Call it with two images (or two batches of the same length) of any sizes.

    It is deliberately not an `nn.Module`: calls take images of any supported type and size and do the resizing.
    `module()` is the same model as an `nn.Module` for `torch.compile` / `torch.export`, and `export_onnx` writes it
    to ONNX (for TensorRT).

    An instance is not reentrant: calls from several threads are serialised (use one instance per thread to run them
    in parallel).

    Has the attributes of `fastforks.Model`.
    """

    _runner: Runner

    def module(self) -> torch.nn.Module:
        """The whole model as an `nn.Module`, for `torch.compile` and `torch.export`.

        `forward(image1, image2)` takes RGB `[B, 3, H, W]` float32 tensors in `[0, 1]` on the model's device and
        returns `{"warp": [B, 2, H, W], "certainty": [B, H, W]}` float32, as a `MatchingModel` call does for a batch.
        Call it under `torch.inference_mode()` (or `no_grad`).

        Raises:
            BackendUnavailableError: The backend has no exportable module.
        """
        return _PortableModule(self._runner, super().module())

    def __call__(self, image1: ImageLike, image2: ImageLike) -> MatchingResult:
        """Match every pixel of `image1` into `image2`.

        ```python
        result = model("view_a.jpg", "view_b.jpg")
        x, y = result.warp  # [H1, W1] each, pixels of view_b
        ```

        Args:
            image1: First image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`); a
                batch gives `[B, 2, H1, W1]` warps.
            image2: Second image, any size; a batch of the same length if `image1` is one.

        Returns:
            The `MatchingResult`.

        Raises:
            InputError: Unsupported or too small inputs, or batches of different lengths.
        """
        t1, t2 = to_rgb_batches((image1, image2), self.device)
        if is_batch(image1) != is_batch(image2) or t1.shape[0] != t2.shape[0]:
            msg = f"image1 and image2 must be two images or equal-length batches, got {t1.shape[0]}, {t2.shape[0]}"
            raise InputError(msg)
        self._check_size(t1)
        self._check_size(t2)
        maps = unbatch(self._batched(self._run, t1, t2), image1)
        return MatchingResult(maps["warp"], maps.get("certainty"), self.variant.ref, self.backend, self.options)

    def _run(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Scale, run and check one chunk."""
        io = self.variant.io
        out = self._runner(scale(image1, io), scale(image2, io))
        b, (h, w) = image1.shape[0], image1.shape[-2:]
        return self._check_outputs(out, {"warp": (b, 2, h, w), "certainty": (b, h, w)})


class _PortableModule(torch.nn.Module):
    """Scaling and the portable runner (its network, resizing and precision)."""

    def __init__(self, runner: Runner, net: torch.nn.Module) -> None:
        """Wrap a portable runner; `net` (its `Runner.module()`) holds the parameters."""
        super().__init__()
        self.net = net.eval().requires_grad_(requires_grad=False)  # inference only
        self._runner = runner
        self.eval()

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """`{"warp": [B, 2, H, W], "certainty": [B, H, W]}` float32 of RGB `[B, 3, H, W]` pairs in `[0, 1]`."""
        io = self._runner.ctx.io
        return {k: v.float() for k, v in self.net(scale(image1, io), scale(image2, io)).items()}


def bench_inputs(model: MatchingModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: a synthetic textured pair `[1, 3, H, W]` (the second image shifted by 8 px)."""
    return synthetic_pair(size.height, size.width, model.device), {}


list_models, model_card, load = task_api(TASK, MatchingModel)
