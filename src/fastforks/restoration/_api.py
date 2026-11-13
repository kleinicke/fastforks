"""Restoration API: `list_models`, `model_card`, `load`, `RestorationModel` and `RestorationResult`."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, to_rgb_batch, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs
from fastforks.restoration._runner import Crop, crop, scale

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Runner

__all__ = ["Backend", "RestorationModel", "RestorationResult", "list_models", "load", "model_card"]

TASK = "restoration"
OVERLAP = 32  # tile overlap in input pixels, as ComfyUI's "Upscale Image (using Model)" node


@dataclass(frozen=True)
class RestorationResult(NumpyOutputs):
    """Output of a restoration model: float32 on the model's device, `model.scale` x the input size.

    Attributes:
        image: The restored RGB image `[3, sH, sW]` (`[B, 3, sH, sW]` for batches) in `[0, 1]`, s = `model.scale`.
        model: Versioned model id, e.g. `"realesrgan-x4plus@1"`.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults (they change the numerics).
    """

    image: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)


class RestorationModel(core.Model):
    """A loaded restoration (super-resolution) model. Call it with an image (or a batch).

    It is deliberately not an `nn.Module`: calls take images of any supported type and size and do the scaling,
    padding, tiling and cropping. `module()` is the same model as an `nn.Module` for `torch.compile` / `torch.export`,
    and `export_onnx` writes it to ONNX (for TensorRT).

    An instance is not reentrant: calls from several threads are serialised (use one instance per thread to run them
    in parallel).

    Has the attributes of `fastforks.Model`.
    """

    _runner: Runner

    @property
    def scale(self) -> int:
        """Output size / input size (4 for a x4 upscaler)."""
        return self._runner.scale

    def module(self) -> torch.nn.Module:
        """The whole model as an `nn.Module`, for `torch.compile` and `torch.export`.

        `forward(image)` takes RGB `[B, 3, H, W]` float32 tensors in `[0, 1]` on the model's device and returns
        `[B, 3, sH, sW]` float32 in `[0, 1]`, as `RestorationModel(image).image` does for a batch without tiling
        (scaling, padding, cropping and clamping included). Call it under `torch.inference_mode()` (or `no_grad`).

        Raises:
            BackendUnavailableError: The backend has no exportable module.
        """
        return _PortableModule(self._runner, super().module())

    def __call__(self, image: ImageLike, *, tile: int | None = None) -> RestorationResult:
        """Restore (upscale) an image.

        ```python
        result = model("photo.png")  # the whole image in one pass
        result = model("photo.png", tile=512)  # tiles of 512 px overlapping by 32, blended: less memory
        result.image  # [3, sH, sW] in [0, 1]
        ```

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`); a batch gives
                `[B, 3, sH, sW]` images.
            tile: Tile size in input pixels, or `None` for one pass. As ComfyUI's "Upscale Image (using Model)": tiles
                overlap by 32 px and are blended with linear ramps over the overlap; an image that fits in one tile
                is one pass.

        Returns:
            The `RestorationResult`.

        Raises:
            InputError: Unsupported or too small image, or a `tile` that is not an integer above 32 px and at least the
                model's smallest input.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)
        if tile is not None:
            self._check_tile(tile)
        maps = unbatch(self._batched(functools.partial(self._run, tile=tile), x), image)
        return RestorationResult(maps["image"], self.variant.ref, self.backend, self.options)

    def _check_tile(self, tile: object) -> None:
        low = max(OVERLAP + 1, self.size_range().reject_below or self.size_range().min_size)
        if not (isinstance(tile, int) and not isinstance(tile, bool) and tile >= low):
            msg = f"tile must be an integer of at least {low} px (or None for one pass), got {tile!r}"
            raise InputError(msg)

    def _pass(self, x: torch.Tensor) -> torch.Tensor:
        """Scale, pad, run, crop, check and clamp one image or tile."""
        r, s = self._runner, self._runner.scale
        b, _, h, w = x.shape
        padded, region = r.pad(scale(x, self.variant.io))
        out = crop(r(padded)["image"], Crop(region.top * s, region.left * s, h * s, w * s))
        return self._check_outputs({"image": out}, {"image": (b, 3, h * s, w * s)})["image"].clamp(0, 1)

    def _run(self, x: torch.Tensor, *, tile: int | None) -> dict[str, torch.Tensor]:
        """One chunk, in one pass or in tiles (ComfyUI `comfy.utils.tiled_scale`, overlap 32)."""
        b, _, h, w = x.shape
        if tile is None or (h <= tile and w <= tile):
            return {"image": self._pass(x)}
        s = self._runner.scale
        out = torch.zeros(b, 3, h * s, w * s, device=x.device)
        weight = torch.zeros(1, 1, h * s, w * s, device=x.device)
        for top in _starts(h, tile):
            for left in _starts(w, tile):
                th, tw = min(tile, h - top), min(tile, w - left)
                y = self._pass(x[..., top : top + th, left : left + tw])
                m = _ramp(th * s, OVERLAP * s, x.device)[:, None] * _ramp(tw * s, OVERLAP * s, x.device)[None, :]
                out[..., top * s : (top + th) * s, left * s : (left + tw) * s] += y * m
                weight[..., top * s : (top + th) * s, left * s : (left + tw) * s] += m
        return {"image": (out / weight).clamp_(0, 1)}


def _starts(n: int, tile: int) -> list[int]:
    """Tile starts along a side of `n` px: every `tile - OVERLAP` px, the last tile shorter (as ComfyUI)."""
    if n <= tile:
        return [0]
    return [max(0, min(n - OVERLAP, p)) for p in range(0, n - OVERLAP, tile - OVERLAP)]


def _ramp(n: int, feather: int, device: torch.device) -> torch.Tensor:
    """Blend weights along one side of a tile: linear ramps (1 / f, ..., f / f) over `feather` px at both ends."""
    v = torch.ones(n, device=device)
    if feather < n:
        a = torch.arange(1, feather + 1, device=device) / feather
        v[:feather] *= a
        v[n - feather :] *= a.flip(0)
    return v


class _PortableModule(torch.nn.Module):
    """Scaling, padding, the portable runner (its network and precision), cropping and clamping, without tiling."""

    def __init__(self, runner: Runner, net: torch.nn.Module) -> None:
        """Wrap a portable runner; `net` (its `Runner.module()`) holds the parameters."""
        super().__init__()
        self.net = net.eval().requires_grad_(requires_grad=False)  # inference only
        self._runner = runner
        self.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """`[B, 3, sH, sW]` float32 in `[0, 1]` of RGB `[B, 3, H, W]` in `[0, 1]`."""
        r, s = self._runner, self._runner.scale
        h, w = image.shape[-2:]
        padded, region = r.pad(scale(image, r.ctx.io))
        out = crop(r(padded)["image"], Crop(region.top * s, region.left * s, h * s, w * s))
        return out.float().clamp(0, 1)


def bench_inputs(model: RestorationModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]` in one pass."""
    return (synthetic_image(size.height, size.width, model.device),), {}


list_models, model_card, load = task_api(TASK, RestorationModel)
