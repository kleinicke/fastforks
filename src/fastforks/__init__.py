"""fastforks: fast, typed inference for published vision models, one subpackage per task.

Tasks: `fastforks.stereo`, `flow`, `matching`, `depth`, `multiview`, `classification`, `segmentation`, `detection`,
`promptable_segmentation`, `features`, `pose`, `matting`, `tracking`; `fastforks.list_models()` lists the installed
models of every task and `fastforks.load(id)` loads any of them.
The shared types (`Model`, `Backend`, `ModelCard`, `SizeRange`, `TensorRTEngine`, `ImageLike`) are exported here too.
`fastforks.patch(model)` speeds up your own timm / torchvision / HF transformers classifier in place (imported on
first use: `import fastforks` stays torch-free).
"""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import TYPE_CHECKING, Any

from fastforks._core import registry
from fastforks._core.arguments import typo_checked_load
from fastforks._core.errors import (
    ArgumentError,
    BackendUnavailableError,
    CardError,
    CheckpointMismatchError,
    FastforksError,
    InputError,
    KernelUnavailableError,
    LicenceError,
    MissingDependencyError,
    ModelNotFoundError,
    OptionsError,
    WeightsDownloadError,
    WeightsError,
    WeightsHashError,
)
from fastforks._core.patch import base_type
from fastforks._core.save import save

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping, Sequence

    import torch

    from fastforks._core.cards import Commercial, Dtype, ModelCard, Tier
    from fastforks._core.export import TensorRTEngine
    from fastforks._core.image import ImageLike, read_images
    from fastforks._core.model import Backend, ModeArg, Model
    from fastforks._core.sizes import SizeRange
    from fastforks._optimize import FastModel, patch

__version__ = importlib.metadata.version("fastforks-core")
__all__ = [
    "ArgumentError",
    "Backend",
    "BackendUnavailableError",
    "CardError",
    "CheckpointMismatchError",
    "FastModel",
    "FastforksError",
    "ImageLike",
    "InputError",
    "KernelUnavailableError",
    "LicenceError",
    "MissingDependencyError",
    "Model",
    "ModelCard",
    "ModelNotFoundError",
    "OptionsError",
    "SizeRange",
    "TensorRTEngine",
    "WeightsDownloadError",
    "WeightsError",
    "WeightsHashError",
    "__version__",
    "base_type",
    "describe_models",
    "list_models",
    "load",
    "model_card",
    "patch",
    "read_images",
    "save",
]


def list_models(  # noqa: PLR0913 - the filters of the task-level `list_models` plus `task`
    filter: str | Sequence[str] = "*",  # noqa: A002 - mirrors timm.list_models
    *,
    task: str | None = None,
    exclude_filters: str | Sequence[str] = (),
    commercial: Commercial | None = None,
    tier: Tier | None = None,
    family: str | None = None,
    runnable: bool | None = None,
) -> list[str]:
    """Ids of the installed models of every task (or of one), optionally filtered; like `timm.list_models`.

    Reads model cards (`runnable=` also looks up weight files); no model code is imported. Ids are unique across
    tasks; `fastforks.<task>.load(id)` loads a model.

    Args:
        filter: Shell-style pattern matched against the model id (`"*anything*"`), or a list of patterns (any may
            match); case-insensitive.
        task: Only this task (a subpackage name: `"stereo"`, `"depth"`, ...).
        exclude_filters: Drop ids matching this pattern (or any of these patterns).
        commercial: Keep variants with this effective status (code and weights licence).
        tier: Keep models of this support tier.
        family: Keep variants of this family (the card's `family`).
        runnable: `True`: only models that `load` can build: their optional libraries (`model_card(id).requires`,
            e.g. timm) are present in a supported version, and their weights are local or may be downloaded
            (non-commercial weights download only with `FASTFORKS_NONCOMMERCIAL=1`; local copies always load);
            `False`: only the others; `None`: all. Libraries are looked up, not imported; weight files by name and
            size, not hashed.

    Returns:
        Sorted model ids.

    Raises:
        OptionsError: An unknown task, or `commercial` / `tier` is not one of its allowed values.
    """
    return registry.list_models(
        registry.TASKS if task is None else (task,),
        filter,
        exclude_filters=exclude_filters,
        commercial=commercial,
        tier=tier,
        family=family,
        runnable=runnable,
    )


def model_card(model: str) -> ModelCard:
    """Card of a model id of any task (`"s2m2-s"` or `"s2m2-s@1"`): the same as `fastforks.<task>.model_card(model)`.

    Reads cards only (no torch import). `card.task` names the task, `card.requires` the optional libraries the model
    needs, and `card.contract()` its resolved inputs, outputs, options and modes.

    Args:
        model: Model id, optionally with `@revision`.

    Raises:
        ModelNotFoundError: No task has `model` (the message suggests similar ids), or the revision differs.
    """
    return registry.variant_card(*registry.resolve_any(model))


@typo_checked_load
def load(  # noqa: PLR0913 - the arguments of every task's `load`, plus classification's `labels`
    model: str,
    *,
    mode: ModeArg = "auto",
    device: str | torch.device | None = None,
    dtype: Dtype | torch.dtype | None = None,
    options: Mapping[str, Any] | None = None,
    weights: str | os.PathLike[str] | None = None,
    labels: Sequence[str] | None = None,
) -> Model:
    """Load any model by id: the same as `fastforks.<task>.load(model, ...)` for the task that has the model.

    `fastforks.load("s2m2-s")` is `fastforks.stereo.load("s2m2-s")`; the result is that task's model class (call it
    as documented for the task). Use the task's own `load` when you want its precise result type for type checking.
    The arguments are those of the task loaders, see e.g. `fastforks.stereo.load`.

    Args:
        model: Model id, optionally pinned to a revision (`"s2m2-s@1"`).
        mode: `"auto"` (default), `"portable"` or `"kernels"`, as in the task loaders.
        device: Default: `"cuda"` if available, else `"mps"` for a family validated on it, else `"cpu"`.
        dtype: Compute dtype (`None`: the validated one; a string or the torch dtype).
        options: Model options (see `fastforks info MODEL`).
        weights: Your own weights file or directory instead of the downloaded ones.
        labels: Classification only: class names for your own weights (see `fastforks.classification.load`).

    Returns:
        The model of the task that has `model` (`StereoModel`, `DepthModel`, ...).

    Raises:
        ModelNotFoundError: No task has `model`; the message suggests similar ids.
        ArgumentError: An unknown keyword (the message suggests the closest one, e.g. `mdoe=` -> `mode`).
        OptionsError: Several tasks have this id (the message names the task loaders), or `labels` for a model that
            is not a classifier.
        FastforksError: Anything the task's `load` raises.
    """
    tasks = [t for t in registry.TASKS if _has(t, model)]
    if len(tasks) > 1:
        loaders = ", ".join(f"fastforks.{t}.load({model!r})" for t in tasks)
        msg = f"{model!r} is a model of several tasks ({', '.join(tasks)}); use the task's loader: {loaders}"
        raise OptionsError(msg)
    card, _ = registry.resolve_any(model)
    task = importlib.import_module(f"fastforks.{card.task}")
    extra: dict[str, Any] = {}
    if labels is not None:
        if card.task != "classification":
            msg = f"labels= is for classifiers; {model!r} is a {card.task} model"
            raise OptionsError(msg)
        extra["labels"] = labels
    return task.load(model, mode=mode, device=device, dtype=dtype, options=options, weights=weights, **extra)


def _has(task: str, model: str) -> bool:
    try:
        registry.resolve(task, model)
    except FastforksError:
        return False
    return True


def describe_models(models: Sequence[str] | None = None, **filters: Any) -> str:  # noqa: ANN401 - `list_models` filters
    """A readable table of models: id, task, tier, code / weights licence, commercial status and what blocks a `yes`.

    `print(fastforks.describe_models(stereo.list_models("hitnet*")))`; `list_models` itself keeps returning ids.

    Args:
        models: A list of model ids (any task); `None` takes `list_models(**filters)`. Not a string: a pattern goes to
            `filter=`.
        **filters: The arguments of `list_models` (`filter`, `task`, `commercial`, ...), used if `models` is `None`.

    Returns:
        The table, one row per id, sorted by task and id.

    Raises:
        ModelNotFoundError: An id that no installed model has.
        TypeError: `models` is a string; pass a list of ids, or a pattern as `filter=`
            (`describe_models(filter="igev*")`).
    """
    if isinstance(models, str):  # a str is a Sequence[str]: its characters would be looked up as ids
        msg = (
            f"describe_models: models= takes a list of ids, got the string {models!r}; for one id pass "
            f"[{models!r}], for a pattern describe_models(filter={models!r})"
        )
        raise TypeError(msg)
    ids = list_models(**filters) if models is None else models
    rows = []
    for mid in ids:
        card, v = registry.resolve_any(mid)
        st = card.status(v)
        rows.append((card.task, v.ref, card.tier, card.licence.code, st.weights, st.commercial, "; ".join(st.reasons)))
    rows.sort()
    head = ("task", "model", "tier", "code licence", "weights licence", "commercial", "why not yes")
    table = [head, *rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(head) - 1)]
    return "\n".join(
        "  ".join(f"{c:{w}}" for c, w in zip(r[:-1], widths, strict=True)) + f"  {r[-1]}".rstrip() for r in table
    )


_LAZY = {  # public name -> module that defines it; these import torch
    "Backend": "fastforks._core.model",
    "FastModel": "fastforks._optimize",
    "ImageLike": "fastforks._core.image",
    "Model": "fastforks._core.model",
    "ModelCard": "fastforks._core.cards",
    "SizeRange": "fastforks._core.sizes",
    "TensorRTEngine": "fastforks._core.export",
    "patch": "fastforks._optimize",
    "read_images": "fastforks._core.image",
}


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy re-export
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    if name in registry.TASKS:  # `import fastforks; fastforks.stereo`
        return importlib.import_module(f"fastforks.{name}")
    msg = f"module 'fastforks' has no attribute {name!r}"
    raise AttributeError(msg)
