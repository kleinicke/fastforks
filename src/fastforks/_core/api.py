"""The `list_models`, `model_card` and `load` functions every task package exposes, built once per task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from fastforks._core import model as core
from fastforks._core.arguments import typo_checked_load
from fastforks._core.model import M, ModeArg, Model

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Mapping, Sequence

    import torch

    from fastforks._core.cards import Commercial, Dtype, ModelCard, Tier

__all__ = ["ListModelsFn", "LoadFn", "ModelCardFn", "task_api"]


class ListModelsFn(Protocol):
    """Signature of every task's `list_models`."""

    def __call__(  # noqa: PLR0913 - the filters of every task
        self,
        filter: str | Sequence[str] = "*",  # noqa: A002 - mirrors timm.list_models
        *,
        exclude_filters: str | Sequence[str] = (),
        commercial: Commercial | None = None,
        tier: Tier | None = None,
        family: str | None = None,
        runnable: bool | None = None,
    ) -> list[str]:
        """Ids of the installed models of this task, optionally filtered. Reads model cards only; imports no model code.

        Args:
            filter: Shell-style pattern matched against the model id (`"s2m2*"`, `"*-vit-*"`), or a list of
                patterns (any may match); case-insensitive.
            exclude_filters: Drop ids matching this pattern (or any of these patterns), as in `timm.list_models`.
            commercial: Keep variants with this effective status (code and weights licence; see `ModelCard.status`).
            tier: Keep models of this support tier.
            family: Keep variants of this family (the card's `family`, e.g. `"s2m2"`).
            runnable: `True`: only models whose optional libraries (`model_card(id).requires`, e.g. timm) are
                present in a supported version, so `load` can build them (every listed model is installed as a card);
                `False`: only the others; `None`: all. Libraries are looked up, not imported.

        Returns:
            Sorted model ids; pass one to `load` or `model_card`.

        Raises:
            OptionsError: `commercial` or `tier` is not one of its allowed values.
        """
        ...


class ModelCardFn(Protocol):
    """Signature of every task's `model_card`."""

    def __call__(self, model: str) -> ModelCard:
        """Card of a model id of this task (`"s2m2-s"` or `"s2m2-s@1"`).

        Args:
            model: Model id, optionally pinned to a revision.

        Returns:
            The model family's card with the id's own `outputs` / `output_info` (a variant may override the
            family's, e.g. the depth `kind`); `card.requires` lists the optional libraries it needs (`card.extras`:
            the `fastforks-core[...]` extras that install them).

        Raises:
            ModelNotFoundError: Unknown id or revision.
        """
        ...


M_co = TypeVar("M_co", bound=Model, covariant=True)


class LoadFn(Protocol[M_co]):
    """Signature of every task's `load`; it returns the task's model class."""

    def __call__(  # noqa: PLR0913 - the same arguments in every task
        self,
        model: str,
        *,
        mode: ModeArg = "auto",
        device: str | torch.device | None = None,
        dtype: Dtype | torch.dtype | None = None,
        options: Mapping[str, Any] | None = None,
        weights: str | os.PathLike[str] | None = None,
    ) -> M_co:
        """Load a model of this task.

        Args:
            model: Model id, optionally pinned to a revision (`"s2m2-s@1"`; weights are pinned by URL and SHA-256
                per revision).
            mode: `"auto"`: the kernels backend if it can run here and was measured faster, else `portable`; the
                choice is logged and recorded in the result's `backend`. `"portable"`: plain PyTorch, runs everywhere.
                `"kernels"`: our custom kernels; raises if they cannot run here.
            device: Default: `"cuda"` if available, else `"mps"` (Apple GPU) for a family validated on it (card
                `[modes.portable] mps_dtypes`), else `"cpu"`.
            dtype: Compute dtype. `None`: the validated one (the card's fp16 / bf16 on CUDA, float32 on MPS and the CPU,
                always reduced precision for `kernels`). `"float32"`: portable only, the path closest to the original
                code (slower). `"float16"` / `"bfloat16"`: portable on CUDA where the model code implements
                it (logged if it is not the validated one), portable on MPS where the card lists it, or `kernels` when
                it is the kernels' dtype (else `"auto"` falls back to portable). `torch.float32` / `torch.float16` /
                `torch.bfloat16` are the same as the strings.
            options: Overrides of the model's load options (card `[options]`, e.g. `{"iters": 16}`); the defaults
                are the validated values, and non-default ones are recorded in every result.
            weights: Your own copy of the weights instead of the recorded ones, e.g. a fine-tuned checkpoint with
                the same keys, shapes and file format: the file, or a directory holding the variant's files as
                `<dir>/<name>` (several files, e.g. a diffusers layout; weights shipped in an archive: the
                extracted file). Not hash-checked, so the golden parity does not apply (logged once); loaded
                strictly.

        Returns:
            The loaded model.

        Raises:
            ModelNotFoundError: Unknown id or revision.
            OptionsError: Unknown option, a value of the wrong type, an invalid `mode` or `dtype`, a `dtype` the
                backend does not implement, `dtype="float32"` with `mode="kernels"`, or an unknown or unusable
                `device`.
            BackendUnavailableError: `mode="kernels"` cannot run here (the message says why), e.g. no kernel build for
                this GPU, or `kernels` already runs on another CUDA device in this process. Never raised for `"auto"`.
            WeightsError: Weights cannot be found, downloaded or verified, or `weights` is missing or does not
                fit the model (the message lists the missing, unexpected and mismatched keys).
        """
        ...


def task_api(
    task: str, cls: type[M] | Callable[[str], type[M]], note: str = ""
) -> tuple[ListModelsFn, ModelCardFn, LoadFn[M]]:
    """`(list_models, model_card, load)` of `task`; `load` builds a `cls`, or the class `cls(model id)` returns.

    `note` is a paragraph added to the runtime docstring of `load` (what is specific to this task).
    """

    def list_models(  # noqa: PLR0913 - the filters of every task
        filter: str | Sequence[str] = "*",  # noqa: A002 - mirrors timm.list_models
        *,
        exclude_filters: str | Sequence[str] = (),
        commercial: Commercial | None = None,
        tier: Tier | None = None,
        family: str | None = None,
        runnable: bool | None = None,
    ) -> list[str]:
        return core.list_models(
            task,
            filter,
            exclude_filters=exclude_filters,
            commercial=commercial,
            tier=tier,
            family=family,
            runnable=runnable,
        )

    def model_card(model: str) -> ModelCard:
        return core.model_card(task, model)

    def load(  # noqa: PLR0913 - the same arguments in every task
        model: str,
        *,
        mode: ModeArg = "auto",
        device: str | torch.device | None = None,
        dtype: Dtype | torch.dtype | None = None,
        options: Mapping[str, Any] | None = None,
        weights: str | os.PathLike[str] | None = None,
    ) -> M:
        model_cls = cast("type[M]", cls if isinstance(cls, type) else cls(model))
        return core.load_model(
            task, model, model_cls, mode=mode, device=device, dtype=dtype, options=options, weights=weights
        )

    for fn, proto in ((list_models, ListModelsFn), (model_card, ModelCardFn), (load, LoadFn)):
        fn.__module__ = f"fastforks.{task}"
        fn.__qualname__ = fn.__name__
        fn.__doc__ = proto.__call__.__doc__
    if note:
        summary, rest = (load.__doc__ or "").split("\n\n", 1)
        load.__doc__ = f"{summary}\n\n{note}\n\n{rest}"
    checked = typo_checked_load(load)
    if isinstance(cls, type):  # `help(load)` / `inspect.signature` show the class, not the TypeVar
        for fn in (load, checked):  # set on both: Python 3.14 `functools.wraps` copies `__annotate__`, not the dict
            fn.__annotations__ = {**load.__annotations__, "return": cls}
    return list_models, model_card, checked
