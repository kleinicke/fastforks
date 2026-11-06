"""`StereoModel.module()`: the whole model as an `nn.Module` for `torch.compile` and `torch.export`.

Both backends take RGB `[B, 3, H, W]` float32 in `[0, 1]` and return the disparity `[B, H, W]` float32, exactly what
`StereoModel(left, right).disparity` holds for a batch:

- portable: plain PyTorch, scaling, padding, network and cropping traced as ordinary ops (compile, export, ONNX).
- kernels: one opaque custom op, `fastforks::stereo_disparity`, that runs the kernels eagerly. `torch.compile` traces it
  without graph breaks and `torch.export` records it, but the op refers to the model loaded in this process, so an
  exported max program only runs here (no AOTInductor, ONNX or TensorRT).
"""

from __future__ import annotations

import itertools
import weakref
from typing import TYPE_CHECKING

import torch

from fastforks.stereo._runner import crop, scale

if TYPE_CHECKING:
    from fastforks.stereo._api import StereoModel
    from fastforks.stereo._runner import Runner

_KERNELS_MODELS: weakref.WeakValueDictionary[int, StereoModel] = weakref.WeakValueDictionary()
_KEYS = itertools.count()


class PortableModule(torch.nn.Module):
    """Scaling, padding, the portable runner (its network, precision and autocast) and cropping."""

    def __init__(self, runner: Runner) -> None:
        """Wrap a portable runner; `net` holds its parameters (`Runner.module()`)."""
        super().__init__()
        self.net = (
            runner.module().eval().requires_grad_(requires_grad=False)
        )  # inference only: no autograd graph when compiled
        self._runner = runner
        self.eval()

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Disparity `[B, H, W]` float32 of RGB `[B, 3, H, W]` pairs in `[0, 1]`."""
        r = self._runner
        if r.io == "raw":
            return r(left, right)["disparity"].float()
        io = r.ctx.io
        (lp, region), (rp, _) = r.pad(scale(left, io)), r.pad(scale(right, io))
        return crop(r(lp, rp)["disparity"], region).float()


class KernelsModule(torch.nn.Module):
    """The `kernels` model behind the opaque op `fastforks::stereo_disparity`."""

    def __init__(self, model: StereoModel) -> None:
        """Register `model` for the op; the module keeps it alive."""
        super().__init__()
        self._model = model
        self._key = next(_KEYS)
        _KERNELS_MODELS[self._key] = model

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Disparity `[B, H, W]` float32 of RGB `[B, 3, H, W]` pairs in `[0, 1]`."""
        return torch.ops.fastforks.stereo_disparity(self._key, left, right)


@torch.library.custom_op("fastforks::stereo_disparity", mutates_args=())
def _stereo_disparity(key: int, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Run the registered `kernels` model `key` eagerly (inputs as `StereoModel.__call__` takes batches)."""
    model = _KERNELS_MODELS.get(key)
    if model is None:
        msg = f"fastforks::stereo_disparity: no kernels model {key} here (exported programs run in-process only)"
        raise RuntimeError(msg)
    disparity = model(left, right).disparity
    return disparity.clone()  # a fresh tensor: custom ops must not return internal buffers


@_stereo_disparity.register_fake
def _(key: int, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:  # noqa: ARG001 - same schema as the op
    return left.new_empty((left.shape[0], *left.shape[-2:]), dtype=torch.float32)
