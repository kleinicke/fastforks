"""Restoration API: `list_models`, `model_card`, `load`, `RestorationModel` and `RestorationResult`."""

from __future__ import annotations

import dataclasses
import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core import model as core
from fastforks._core import registry
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.checkpoint import load_state_dict
from fastforks._core.errors import InputError, WeightsError
from fastforks._core.image import ImageLike, to_rgb_batch, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs
from fastforks.restoration._runner import Crop, crop, scale

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks._core.cards import Dtype, Io, ModelCard
    from fastforks._core.model import ModeArg
    from fastforks.restoration._runner import Runner

__all__ = [
    "Backend",
    "RestorationModel",
    "RestorationResult",
    "checkpoint_family",
    "list_models",
    "load",
    "load_file",
    "model_card",
]

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
        (scaling, padding, cropping and clamping included). Inpainting models take `forward(image, mask)`, the mask
        `[B, 1, H, W]` float32 (1 = fill, else 0). Call it under `torch.inference_mode()` (or `no_grad`).

        Raises:
            BackendUnavailableError: The backend has no exportable module.
        """
        return _PortableModule(self._runner, super().module())

    def __call__(
        self, image: ImageLike, *, mask: ImageLike | None = None, tile: int | None = None
    ) -> RestorationResult:
        """Restore (upscale) an image.

        ```python
        result = model("photo.png")  # the whole image in one pass
        result = model("photo.png", tile=512)  # tiles of 512 px overlapping by 32, blended: less memory
        result.image  # [3, sH, sW] in [0, 1]
        model("photo.png", mask="holes.png")  # inpainting models: fill the mask's nonzero pixels
        ```

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`); a batch gives
                `[B, 3, sH, sW]` images.
            mask: Only for inpainting models (`contract().inputs["mask"] == "required"`): a gray image of the same size
                (and batch, or one for all) whose nonzero pixels are filled; read like `image` (colour: the first
                channel).
            tile: Tile size in input pixels, or `None` for one pass. As ComfyUI's "Upscale Image (using Model)": tiles
                overlap by 32 px and are blended with linear ramps over the overlap; an image that fits in one tile
                is one pass.

        Returns:
            The `RestorationResult`.

        Raises:
            InputError: Unsupported or too small image; a `tile` that is not an integer above 32 px and at least the
                model's smallest input; a mask missing for a model that needs one, given to one that does not use
                it, or of another size or batch.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)
        if tile is not None:
            self._check_tile(tile)
        m = self._mask(mask, x)
        if m is not None:
            x = torch.cat([x, m], 1)
        maps = unbatch(self._batched(functools.partial(self._run, tile=tile), x), image)
        return RestorationResult(maps["image"], self.variant.ref, self.backend, self.options)

    def _mask(self, mask: ImageLike | None, x: torch.Tensor) -> torch.Tensor | None:
        """The binary `[B, 1, H, W]` float32 mask (1 = fill) of an inpainting model's call, else `None`."""
        need = self._runner.mask == "required"
        if mask is None:
            if need:
                msg = f"{self.variant.ref} needs a mask (gray image: nonzero = the region to fill)"
                raise InputError(msg)
            return None
        if not need:
            msg = f"{self.variant.ref} does not use a mask; call it without one"
            raise InputError(msg)
        m = to_rgb_batch(mask, self.device)[:, :1]
        if m.shape[-2:] != x.shape[-2:] or m.shape[0] not in {1, x.shape[0]}:
            msg = (
                f"mask: {m.shape[0]} map(s) of {tuple(m.shape[-2:])} px do not match the image(s): {x.shape[0]} of "
                f"{tuple(x.shape[-2:])} px"
            )
            raise InputError(msg)
        return (m > 0).float().expand(x.shape[0], -1, -1, -1).contiguous()

    def example_inputs(
        self, height: int | None = None, width: int | None = None, *, batch: int = 1
    ) -> tuple[torch.Tensor, ...]:
        """Example inputs of `module()` (as the exports trace them): the synthetic image, plus a mask if required.

        The mask `[batch, 1, height, width]` marks a centred box (half the sides) to fill.
        """
        (image,) = super().example_inputs(height, width, batch=batch)[:1]
        if self._runner.mask != "required":
            return (image,)
        return image, synthetic_mask(*image.shape[-2:], self.device).expand(image.shape[0], -1, -1, -1).contiguous()

    def _check_tile(self, tile: object) -> None:
        low = max(OVERLAP + 1, self.size_range().reject_below or self.size_range().min_size)
        if not (isinstance(tile, int) and not isinstance(tile, bool) and tile >= low):
            msg = f"tile must be an integer of at least {low} px (or None for one pass), got {tile!r}"
            raise InputError(msg)

    def _pass(self, x: torch.Tensor) -> torch.Tensor:
        """Scale, pad, run, crop, check and clamp one image or tile."""
        r, s = self._runner, self._runner.scale
        b, _, h, w = x.shape
        padded, region = r.pad(_scale(x, self.variant.io))
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


def _scale(x: torch.Tensor, io: Io) -> torch.Tensor:
    """`scale` of the RGB channels; a mask channel (inpainting) stays 0 / 1."""
    if x.shape[1] == 3:  # noqa: PLR2004 - RGB
        return scale(x, io)
    return torch.cat([scale(x[:, :3], io), x[:, 3:]], 1)


def synthetic_mask(height: int, width: int, device: torch.device) -> torch.Tensor:
    """`[1, 1, H, W]` float32 inpainting mask: a centred box of half the sides to fill (`example_inputs`, bench)."""
    m = torch.zeros(1, 1, height, width, device=device)
    m[..., height // 4 : height - height // 4, width // 4 : width - width // 4] = 1.0
    return m


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

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """`[B, 3, sH, sW]` float32 in `[0, 1]` of RGB `[B, 3, H, W]` in `[0, 1]` (+ inpainting: the mask)."""
        r, s = self._runner, self._runner.scale
        h, w = image.shape[-2:]
        x = image if mask is None else torch.cat([image, (mask > 0).to(image.dtype)], 1)
        padded, region = r.pad(_scale(x, r.ctx.io))
        out = crop(r(padded)["image"], Crop(region.top * s, region.left * s, h * s, w * s))
        return out.float().clamp(0, 1)


def bench_inputs(model: RestorationModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]` in one pass (+ inpainting: a synthetic mask)."""
    image = synthetic_image(size.height, size.width, model.device)
    if model._runner.mask != "required":  # noqa: SLF001 - the task's own runner contract
        return (image,), {}
    return (image,), {"mask": synthetic_mask(size.height, size.width, model.device)}


list_models, model_card, load = task_api(TASK, RestorationModel)

_WRAPPERS = ("params_ema", "params", "state_dict", "model")  # BasicSR / training-framework checkpoint entries


def _read(path: Path) -> dict[str, torch.Tensor]:
    """The state dict of a checkpoint file: unwrapped from BasicSR's `params_ema` / `params` and similar entries."""
    sd: Mapping[str, Any] = load_state_dict(path, strip="")
    for k in _WRAPPERS:
        inner = sd.get(k)
        if isinstance(inner, dict):
            sd = inner
            break
    return {k.removeprefix("module."): v for k, v in sd.items() if isinstance(v, torch.Tensor)}


def _detect(sd: Mapping[str, torch.Tensor], name: str) -> tuple[ModelCard, str, dict[str, Any], dict[str, Any]]:
    """`(card, variant id, params, mapped state dict)` of the family whose `from_checkpoint` recognises `sd`."""
    families = {card.family: card for card, _ in registry.cards(TASK).values()}
    readers = [(c, f) for c in families.values() if (f := getattr(registry.import_model(c), "from_checkpoint", None))]
    for card, read in readers:
        found = read(sd)
        if found is not None:
            return card, *found
    msg = (
        f"{name}: unsupported architecture; fastforks.restoration.load_file reads the networks of"
        f" {', '.join(c.name for c, _ in readers)} (keys of the original code or of spandrel's formats)"
    )
    raise WeightsError(msg)


def checkpoint_family(path: str | os.PathLike[str]) -> str | None:
    """The family `load_file` builds for a checkpoint file (`"realesrgan"`, `"span"`, `"swinir"`), or `None`.

    `None` for a file `load_file` cannot load: an unsupported architecture or configuration, or not a checkpoint.
    """
    try:
        return _detect(_read(Path(os.fspath(path))), Path(path).name)[0].family
    except (WeightsError, OSError, RuntimeError, ValueError):
        return None


def load_file(
    path: str | os.PathLike[str],
    *,
    mode: ModeArg = "auto",
    device: str | torch.device | None = None,
    dtype: Dtype | torch.dtype | None = None,
) -> RestorationModel:
    """Load an upscaler checkpoint of a supported architecture (e.g. a file of ComfyUI's `models/upscale_models`).

    The architecture and its parameters (scale, blocks, channels, ...) are read from the state dict, and keys of
    other layouts of the same network (ESRGAN's old `model.N` keys) are mapped; the model is built as the nearest
    variant of the family (`model.variant`) with these parameters and loaded strictly. Not hash-checked: the
    golden parity does not apply. `mode`, `device` and `dtype` are those of `load`.

    Raises:
        WeightsError: Missing or unreadable file, or an architecture or configuration fastforks does not implement
            (the message says which).
    """
    p = Path(os.fspath(path)).expanduser()
    if not p.is_file():
        msg = f"{str(path)!r}: no such file"
        raise WeightsError(msg)
    card, variant_id, params, sd = _detect(_read(p), p.name)
    variant = next(v for v in card.variants if v.id == variant_id)
    variant = dataclasses.replace(variant, params={**variant.params, **params})
    return core.load_variant(
        TASK, card, variant, RestorationModel, mode=mode, device=device, dtype=dtype, weights=p, state=sd
    )
