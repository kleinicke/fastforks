"""Optical-flow API: `list_models`, `model_card`, `load`, and the `FlowModel` / `FlowResult` types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_pair
from fastforks._core.image import ImageLike, to_rgb_pair, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs
from fastforks.flow._runner import crop, scale

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.flow._runner import Runner

__all__ = ["Backend", "FlowModel", "FlowResult", "list_models", "load", "model_card"]

TASK = "flow"


@dataclass(frozen=True)
class FlowResult(NumpyOutputs):
    """Output of an optical-flow model: float32 on the model's device, aligned with the first image.

    Attributes:
        flow: Forward flow from the first image to the second, `[2, H, W]` (`[B, 2, H, W]` for batches): channel 0 is
            `dx`, channel 1 is `dy`, in pixels of the input images at their resolution. The pixel `(x, y)` of the
            first image moves to `(x + dx, y + dy)` in the second.
        model: Versioned model id, e.g. `"raft-large@1"`.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults (they change the numerics).
    """

    flow: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)


class FlowModel(core.Model):
    """A loaded optical-flow model. Call it with two images (or two batches) of the same size.

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
        returns the flow `[B, 2, H, W]` float32, as `FlowModel(image1, image2).flow` does for a batch (scaling,
        padding and cropping included). Call it under `torch.inference_mode()` (or `no_grad`).

        Raises:
            BackendUnavailableError: The backend has no exportable module.
        """
        return _PortableModule(self._runner, super().module())

    def __call__(self, image1: ImageLike, image2: ImageLike) -> FlowResult:
        """Estimate the optical flow from `image1` to `image2`.

        ```python
        result = model("frame_0001.png", "frame_0002.png")
        dx, dy = result.flow  # [H, W] each, input pixels
        ```

        Args:
            image1: First image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`); a
                batch gives `[B, 2, H, W]` flow.
            image2: Second image, same size and type.

        Returns:
            The `FlowResult`.

        Raises:
            InputError: Unsupported, mismatched or too small inputs.
        """
        t1, t2 = to_rgb_pair(image1, image2, self.device, ("image1", "image2"))
        self._check_size(t1)
        maps = unbatch(self._batched(self._run, t1, t2), image1)
        return FlowResult(maps["flow"], self.variant.ref, self.backend, self.options)

    def _run(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Scale, resize (the runner's), pad, run, crop, restore and check one chunk."""
        r = self._runner
        io = self.variant.io
        (p1, region), (p2, _) = r.pad(r.resize(scale(image1, io))), r.pad(r.resize(scale(image2, io)))
        size = (image1.shape[-2], image1.shape[-1])
        out = {k: r.restore(crop(v, region), size) for k, v in r(p1, p2).items()}
        return self._check_outputs(out, {"flow": (image1.shape[0], 2, *image1.shape[-2:])})


class _PortableModule(torch.nn.Module):
    """Scaling, resizing, padding, the portable runner (its network, precision and autocast), cropping and restoring."""

    def __init__(self, runner: Runner, net: torch.nn.Module) -> None:
        """Wrap a portable runner; `net` (its `Runner.module()`) holds the parameters."""
        super().__init__()
        self.net = net.eval().requires_grad_(requires_grad=False)  # inference only
        self._runner = runner
        self.eval()

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Flow `[B, 2, H, W]` float32 of RGB `[B, 3, H, W]` pairs in `[0, 1]`."""
        r = self._runner
        io = r.ctx.io
        (p1, region), (p2, _) = r.pad(r.resize(scale(image1, io))), r.pad(r.resize(scale(image2, io)))
        size = (image1.shape[-2], image1.shape[-1])
        return r.restore(crop(r(p1, p2)["flow"], region), size).float()


def bench_inputs(model: FlowModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: a synthetic textured pair `[1, 3, H, W]` (the second image shifted by 8 px)."""
    return synthetic_pair(size.height, size.width, model.device), {}


list_models, model_card, load = task_api(TASK, FlowModel)
