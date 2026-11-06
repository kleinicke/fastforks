"""Image classification API: `list_models`, `model_card`, `load`, and the model / result types."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.arguments import typo_checked_load
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError, OptionsError
from fastforks._core.image import ImageLike, is_batch, to_rgb_batch
from fastforks._core.labels import label_set
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD, resize_crop, scale

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping, Sequence

    import torch

    from fastforks._core.cards import Dtype
    from fastforks._core.model import ModeArg
    from fastforks.classification._runner import Runner

__all__ = ["Backend", "ClassificationModel", "ClassificationResult", "list_models", "load", "model_card"]

TASK = "classification"


@dataclass(frozen=True)
class ClassificationResult(NumpyOutputs):
    """Output of a classifier, on the model's device.

    Attributes:
        logits: Class scores `[classes]` float32 (`[B, classes]` for a batch).
        class_names: Name of every class, in logit order (the card's label set, e.g. ImageNet-1k, or
            `load(labels=)`).
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    logits: torch.Tensor
    class_names: tuple[str, ...]
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def probs(self) -> torch.Tensor:
        """Softmax of `logits` over the classes."""
        return self.logits.softmax(-1)

    def top(self, k: int = 5) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """The `k` most likely classes as `(name, probability)`, best first.

        Returns:
            For one image a list of `k` pairs; for a batch one such list per image.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            msg = f"k must be an integer >= 1, got {k!r}"
            raise InputError(msg)
        p, i = self.probs.topk(min(k, self.logits.shape[-1]), dim=-1)
        rows = [
            [(self.class_names[c], v) for c, v in zip(ci, pi, strict=True)]
            for ci, pi in zip(i.reshape(-1, i.shape[-1]).tolist(), p.reshape(-1, p.shape[-1]).tolist(), strict=True)
        ]
        return rows if self.logits.ndim > 1 else rows[0]


class ClassificationModel(core.Model):
    """A loaded image classifier. Call it with an image (or a batch).

    `module()` is the network as `timm.create_model` gives it: transformed (resized, cropped, normalised) images
    `[B, 3, h, w]` -> logits `[B, classes]`; `data_config` describes that transform in timm's terms.

    Besides the attributes of `fastforks.Model`:

    Attributes:
        class_names: Name of every class, in logit order (the card's label set, or `load(labels=)`).
        num_classes: Number of classes (timm's `model.num_classes`).
        id2label: Class index -> name (transformers' `config.id2label`).
    """

    _runner: Runner

    def __init__(self, runner: Runner, backend: Backend, options: Mapping[str, Any]) -> None:
        """Wrap a runner (built by `load`)."""
        super().__init__(runner, backend, options)
        self._class_names = label_set(self.variant.output_info["label_set"])

    @property
    def class_names(self) -> tuple[str, ...]:
        """Name of every class, in logit order (read-only: `load(labels=)` sets them for your own weights)."""
        return self._class_names

    @property
    def num_classes(self) -> int:
        """Number of classes, as timm's `model.num_classes`."""
        return len(self.class_names)

    @property
    def id2label(self) -> dict[int, str]:
        """Class index -> name, as transformers' `model.config.id2label` (a new dict on every access)."""
        return dict(enumerate(self.class_names))

    @property
    def data_config(self) -> dict[str, Any]:
        """The eval transform as timm's data config.

        `create_transform` on a PIL image resizes uint8 pixels, ours a float tensor: logits can move by ~0.5 and
        close top-5 ranks can swap.

        Keys: `input_size` (`(3, h, w)`), `interpolation`, `mean`, `std`, `crop_pct`, `crop_mode`.
        """
        io = self.variant.io
        mean, std = {
            "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
            "mean_std": (io.mean, io.std),
            "0-1": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "0-255": ((0.0, 0.0, 0.0), (1 / 255, 1 / 255, 1 / 255)),
        }[io.input]
        side = io.crop or io.resize
        return {
            "input_size": (3, side, side),
            "interpolation": io.interpolation,
            "mean": mean,
            "std": std,
            "crop_pct": io.crop / io.resize if io.crop and io.resize else 1.0,
            "crop_mode": "center",
        }

    def export_size(self) -> tuple[int, int] | None:
        """The transformed crop `module()` takes (`data_config["input_size"]`), the default export size."""
        side = self.variant.io.crop or self.variant.io.resize
        return (side, side) if side else super().export_size()

    def example_inputs(
        self, height: int | None = None, width: int | None = None, *, batch: int = 1
    ) -> tuple[torch.Tensor, ...]:
        """A synthetic image of this size (default: the card's crop), normalised as the card's `[io] input` says."""
        return tuple(scale(x, self.variant.io) for x in super().example_inputs(height, width, batch=batch))

    def __call__(self, image: ImageLike) -> ClassificationResult:
        """Classify an image (resized and centre-cropped as the model was evaluated; see the card's `[io]`).

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).

        Returns:
            The result; a batch gives `[B, classes]` logits.

        Raises:
            InputError: Unsupported or too small image.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)
        out = self._batched(self._run, x)
        logits = self._check_outputs(out, {"logits": (x.shape[0], len(self.class_names))})["logits"]
        return ClassificationResult(
            logits if is_batch(image) else logits[0], self.class_names, self.variant.ref, self.backend, self.options
        )

    def _run(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        io = self.variant.io
        return self._runner(scale(resize_crop(x, io), io))


def bench_inputs(model: ClassificationModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]` (resized and cropped by the model as usual)."""
    return (synthetic_image(size.height, size.width, model.device),), {}


list_models, model_card, _load = task_api(TASK, ClassificationModel)


@typo_checked_load
def load(  # noqa: PLR0913 - the arguments of every task plus `labels`
    model: str,
    *,
    mode: ModeArg = "auto",
    device: str | torch.device | None = None,
    dtype: Dtype | torch.dtype | None = None,
    options: Mapping[str, Any] | None = None,
    weights: str | os.PathLike[str] | None = None,
    labels: Sequence[str] | None = None,
) -> ClassificationModel:
    """Load a classifier; the arguments and errors are those of `fastforks.stereo.load`, plus `labels`.

    An untagged timm name (`"resnet50"`) resolves to the installed tag of that architecture (`"resnet50.tv2_in1k"`).

    Args:
        model: Model id, optionally pinned to a revision.
        mode: `"auto"`, `"portable"` or `"kernels"`.
        device: Default: `"cuda"` if available, else `"mps"` for a family validated on it, else `"cpu"`.
        dtype: `None` (the validated one), `"float32"` (portable only), `"float16"` or `"bfloat16"`; see
            `fastforks.stereo.load`.
        options: Overrides of the model's load options (card `[options]`).
        weights: Your own weights file instead of the recorded one, e.g. a fine-tune; not hash-checked, loaded
            strictly (see `fastforks.stereo.load`), so its head has the card's number of classes.
        labels: Class names in logit order for such weights (`class_names`, `id2label` and the results use them);
            exactly one per logit of the head (`num_classes` of the card's label set). Only the names change: the
            input handling and the meaning of the logits stay those of the family.

    Raises:
        ModelNotFoundError: Unknown id or revision.
        OptionsError: Invalid option, mode, dtype or device, or `labels` that are not one string per class.
        BackendUnavailableError: `mode="kernels"` cannot run here. Never raised for `"auto"`.
        WeightsError: Weights cannot be found, downloaded or verified, or `weights` does not fit.
    """
    names = None if labels is None else _labels(labels, len(label_set(model_card(model).output_info["label_set"])))
    m = _load(model, mode=mode, device=device, dtype=dtype, options=options, weights=weights)
    if names is not None:
        m._class_names = names  # noqa: SLF001 - the documented way to set them
    return m


def _labels(labels: Sequence[str], n: int) -> tuple[str, ...]:
    """`labels` as a tuple, checked before anything is loaded: one string per logit of the head."""
    names = tuple(labels) if isinstance(labels, Iterable) and not isinstance(labels, str | bytes) else ()
    if not names or not all(isinstance(x, str) and x for x in names):
        msg = f"labels must be a sequence of class-name strings (non-empty), got {type(labels).__name__}"
        raise OptionsError(msg)
    if len(names) != n:
        msg = f"labels has {len(names)} names, the classifier head has {n} classes"
        raise OptionsError(msg)
    return names
