"""Backbone feature API: `list_models`, `model_card`, `load`, and the `FeaturesModel` / `FeaturesResult` types."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, is_batch, to_rgb_batch, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs
from fastforks._core.preprocess import scale
from fastforks._core.runner import Forward

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks._core.cards import Io
    from fastforks.features._runner import Runner

__all__ = ["Backend", "FeaturesModel", "FeaturesResult", "list_models", "load", "model_card", "network_size"]

TASK = "features"
Size = int | tuple[int, int] | None
Layers = int | Sequence[int] | Literal["all"]


@dataclass(frozen=True)
class FeaturesResult(NumpyOutputs):
    """Backbone features of an image, on the model's device, float32, named and laid out as Hugging Face's output.

    Shapes are for one image; a batch adds a leading `B`. Tokens are `[N, C]` with `N = num_prefix_tokens + h * w`:
    the cls token, then the register tokens, then the patch grid `grid = (h, w)` row-major (ConvNeXt: the pooled
    token, then the final feature map).

    Attributes:
        last_hidden_state: All tokens after the final norm `[N, C]`.
        pooler_output: The model's image embedding `[C]` (the cls token after the final LayerNorm, not unit length;
            ConvNeXt: the pooled mean after its LayerNorm; the family's pool where its card says, e.g. C-RADIO's
            summary `[3 * C]`; `Model.pooler_size` channels).
        hidden_states: As HF `output_hidden_states=True`: a tuple of `num_layers + 1` entries, entry 0 the embedding
            output, entry i the output of block i, before the final norm, each `[N, C]`; entries not requested with
            `layers=` are `None`. `None` if no layers were requested.
        layers: The `hidden_states` indices that were computed (sorted, non-negative).
        grid: Patch grid `(h, w)` of the network input.
        num_prefix_tokens: Tokens in front of the patch grid (cls + registers).
        num_register_tokens: Register tokens among them.
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor | None
    hidden_states: tuple[torch.Tensor | None, ...] | None
    layers: tuple[int, ...]
    grid: tuple[int, int]
    num_prefix_tokens: int
    num_register_tokens: int
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def cls_token(self) -> torch.Tensor:
        """The first token of `last_hidden_state` `[C]` (ConvNeXt: the pooled token)."""
        return self.last_hidden_state[..., 0, :]

    @property
    def register_tokens(self) -> torch.Tensor:
        """Register tokens `[R, C]` (`R = num_register_tokens`, may be 0)."""
        return self.last_hidden_state[..., 1 : 1 + self.num_register_tokens, :]

    @property
    def patch_tokens(self) -> torch.Tensor:
        """Patch tokens `[h * w, C]` of `last_hidden_state`, row-major."""
        return self.last_hidden_state[..., self.num_prefix_tokens :, :]

    def feature_map(self, layer: int | None = None) -> torch.Tensor:
        """Patch tokens as a channel-first grid `[C, h, w]`.

        Args:
            layer: `None` for `last_hidden_state`, else a computed `hidden_states` index (negative counts from the end).

        Raises:
            InputError: `layer` was not computed.
        """
        if layer is None:
            t = self.last_hidden_state
        else:
            hs = self.hidden_states or ()
            t = hs[layer] if -len(hs) <= layer < len(hs) else None
            if t is None:
                msg = f"hidden state {layer} was not computed (computed: {list(self.layers)}); pass layers= to the call"
                raise InputError(msg)
        p = t[..., self.num_prefix_tokens :, :]
        return p.transpose(-1, -2).reshape(*p.shape[:-2], p.shape[-1], *self.grid)


def network_size(h: int, w: int, size: Size, multiple: int) -> tuple[int, int]:
    """The network input size for an `h x w` image: each side a multiple of `multiple` (the patch stride).

    `size=None` keeps the image size, `int` scales the shorter side to `size` (aspect ratio kept); both then round
    every side to the nearest multiple (at least one). `(height, width)` is used as given and must be multiples.

    Raises:
        InputError: An explicit `(height, width)` that is not a multiple of `multiple`, or a size < 1.
    """

    def snap(x: float) -> int:
        return max(multiple, round(x / multiple) * multiple)

    if size is None:
        return snap(h), snap(w)
    if _is_int(size):
        size = int(size)  # ty: ignore[invalid-argument-type]
        if size < 1:
            msg = f"size must be >= 1, got {size}"
            raise InputError(msg)
        s = size / min(h, w)
        return snap(h * s), snap(w * s)
    if not isinstance(size, (tuple, list)) or len(size) != 2 or not all(map(_is_int, size)):  # noqa: PLR2004
        msg = f"size must be None, an int (shorter side) or (height, width) ints, got {size!r}"
        raise InputError(msg)
    sh, sw = int(size[0]), int(size[1])
    if sh < 1 or sw < 1 or sh % multiple or sw % multiple:
        msg = f"size {tuple(size)}: height and width must be positive multiples of {multiple}"
        raise InputError(msg)
    return sh, sw


def _is_int(v: object) -> bool:
    """An integer (numpy integers too), not a bool."""
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def prepare(x: torch.Tensor, hw: tuple[int, int], io: Io) -> torch.Tensor:
    """`[B, 3, H, W]` RGB in `[0, 1]` -> resized to `hw` (`io.interpolation`, antialiased) and normalised."""
    if tuple(x.shape[-2:]) != hw:
        x = F.interpolate(x, size=hw, mode=io.interpolation, antialias=io.interpolation != "nearest")
        if io.interpolation == "bicubic":  # bicubic overshoots; PIL on uint8 clamps to the range
            x = x.clamp(0, 1)
    return scale(x, io)


class FeaturesModel(core.Model):
    """A loaded backbone. Call it with an image (or a batch) to get its features.

    `module()` is the plain network for export: `[B, 3, H, W]` RGB in `[0, 1]` -> `last_hidden_state` `[B, N, C]`,
    with the default input handling of a call (`size=None`: resized to the nearest stride multiples, normalised).

    Besides the attributes of `fastforks.Model`:

    Attributes:
        stride: Patch size: the network input's sides are multiples of it (card `[io] pad.multiple`).
        hidden_size: Channels per token.
        num_layers: Blocks (HF `hidden_states` has `num_layers + 1` entries).
        num_prefix_tokens: Tokens in front of the patch grid (cls + registers).
        num_register_tokens: Register tokens among them.
        pooler_size: Channels of `pooler_output` (`hidden_size`, except C-RADIO: 3 x `hidden_size`).
    """

    _runner: Runner

    def __init__(self, runner: Runner, backend: Backend, options: Mapping[str, Any]) -> None:
        """Wrap a runner (built by `load`)."""
        super().__init__(runner, backend, options)
        self.stride = self.variant.io.pad_multiple
        self.hidden_size = runner.hidden_size
        self.num_layers = runner.num_layers
        self.num_prefix_tokens = runner.num_prefix_tokens
        self.num_register_tokens = runner.num_register_tokens
        self.pooler_size = runner.pooler_size or runner.hidden_size

    def __call__(self, image: ImageLike, *, size: Size = None, layers: Layers = ()) -> FeaturesResult:
        """Extract features of an image.

        The image is resized (never cropped or padded) to the network size, then normalised as the card's `[io]`
        says. The resize uses `[io] interpolation` (antialiased; as the HF image processors: bicubic for DINOv2,
        bilinear for DINOv3) and makes each side a multiple of `stride`.

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).
            size: `None`: keep the image size, each side rounded to the nearest multiple of `stride`. `int`: scale the
                shorter side to `size` (aspect ratio kept), then round each side. `(height, width)`: exactly this
                (multiples of `stride`).
            layers: `hidden_states` entries to compute, HF indexing (0 = embeddings, i = block i, -1 = last block),
                or `"all"`; default none.

        Returns:
            The result; a batch keeps the leading `B`.

        Raises:
            InputError: Unsupported or too small image, bad `size`, `layers` out of range or not supported.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)
        hw = network_size(*x.shape[-2:], size, self.stride)
        idx = self._layers(layers)
        if idx and "hidden_states" not in self.variant.outputs:
            msg = f"{self.variant.ref} has no per-block hidden states; call it without layers="
            raise InputError(msg)
        out = self._batched(lambda xb: self._runner(prepare(xb, hw, self.variant.io), idx), x)
        b, c = x.shape[0], self.hidden_size
        n = self.num_prefix_tokens + (hw[0] // self.stride) * (hw[1] // self.stride)
        shapes = {
            "last_hidden_state": (b, n, c),
            "pooler_output": (b, self.pooler_size),
            "hidden_states": (b, len(idx), n, c),
        }
        maps = unbatch(self._check_outputs(out, shapes), image)
        hidden = None
        if idx:
            slots: list[torch.Tensor | None] = [None] * (self.num_layers + 1)
            stacked = maps["hidden_states"]
            for j, i in enumerate(idx):
                slots[i] = stacked[:, j] if is_batch(image) else stacked[j]
            hidden = tuple(slots)
        return FeaturesResult(
            last_hidden_state=maps["last_hidden_state"],
            pooler_output=maps.get("pooler_output"),
            hidden_states=hidden,
            layers=idx,
            grid=(hw[0] // self.stride, hw[1] // self.stride),
            num_prefix_tokens=self.num_prefix_tokens,
            num_register_tokens=self.num_register_tokens,
            model=self.variant.ref,
            backend=self.backend,
            options=self.options,
        )

    def _layers(self, layers: Layers) -> tuple[int, ...]:
        n = self.num_layers + 1
        if isinstance(layers, str):
            if layers == "all":
                return tuple(range(n))
            msg = f"layers={layers!r}: pass 'all', an int or a sequence of ints"
            raise InputError(msg)
        raw: Any = layers  # checked here: ints (numpy too) or an iterable of them
        try:
            seq = (raw,) if _is_int(raw) else tuple(raw)
        except TypeError:
            seq = (raw,)  # not iterable: reported below
        bad = [i for i in seq if not _is_int(i) or not -n <= int(i) < n]
        if bad:
            msg = f"layers {bad}: hidden_states indices are -{n}..{n - 1} (0 = embeddings, i = block i) or 'all'"
            raise InputError(msg)
        return tuple(sorted({int(i) % n for i in seq}))

    def module(self) -> torch.nn.Module:
        """`[B, 3, H, W]` RGB in `[0, 1]` -> `last_hidden_state` `[B, N, C]` (default input handling, plain PyTorch).

        Raises:
            BackendUnavailableError: The backend has no exportable module (e.g. `kernels`).
        """
        net = super().module()
        io, stride = self.variant.io, self.stride

        def fwd(_net: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
            return net(prepare(image, network_size(*image.shape[-2:], None, stride), io))

        return Forward(net, fwd)


def bench_inputs(model: FeaturesModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]` (each side rounded to a multiple of the stride)."""
    return (synthetic_image(size.height, size.width, model.device),), {}


list_models, model_card, load = task_api(TASK, FeaturesModel)
