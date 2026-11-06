"""Model registry: discovers the `card.toml` files of a task package without importing model code."""

from __future__ import annotations

import dataclasses
import difflib
import fnmatch
import functools
import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import re
import tomllib
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, get_args

from fastforks._core import licences
from fastforks._core.cards import LIBRARY_EXTRA, LIBRARY_SPEC, Commercial, ModelCard, Tier, Variant, read_card
from fastforks._core.errors import (
    BackendUnavailableError,
    CardError,
    LicenceError,
    MissingDependencyError,
    ModelNotFoundError,
    OptionsError,
    WeightsDownloadError,
    install_hint,
)
from fastforks._core.files import env_flag, locate, present
from fastforks._core.tasks import TASK_SPECS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType


TASKS = tuple(TASK_SPECS)
"""Every task (subpackage of `fastforks`) that has model cards, in listing order (from `fastforks._core.tasks`)."""


@functools.cache
def cards(task: str) -> Mapping[str, tuple[ModelCard, Variant]]:
    """Return `{model id: (card, variant)}` of all installed models of `task`, sorted by id.

    Raises:
        CardError: Two cards declare the same model id.
    """
    root = importlib.resources.files(f"fastforks.{task}.models")
    found: dict[str, tuple[ModelCard, Variant]] = {}
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        card_file = entry / "card.toml"
        if not card_file.is_file():
            continue
        card = read_card(Path(str(card_file)), task=task)
        for v in card.variants:
            if v.id in found:
                msg = f"model id {v.id!r} declared by {found[v.id][0].family} and {card.family}"
                raise CardError(msg)
            found[v.id] = (card, v)
    return MappingProxyType(dict(sorted(found.items())))  # cached: read-only for callers


def list_models(  # noqa: PLR0913 - the filters of every `list_models`
    tasks: Sequence[str],
    filter: str | Sequence[str] = "*",  # noqa: A002 - mirrors timm.list_models
    *,
    exclude_filters: str | Sequence[str] = (),
    commercial: Commercial | None = None,
    tier: Tier | None = None,
    family: str | None = None,
    runnable: bool | None = None,
) -> list[str]:
    """Sorted ids of the installed models of `tasks`; see `fastforks.list_models` for the filters.

    Raises:
        OptionsError: An unknown task, or `commercial` / `tier` is not one of its allowed values.
    """
    only = commercial_only()
    for name, value, allowed in (("commercial", commercial, get_args(Commercial)), ("tier", tier, get_args(Tier))):
        if value is not None and value not in allowed:
            msg = f"{name} must be one of {allowed}, got {value!r}"
            raise OptionsError(msg)
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        msg = f"unknown task(s) {unknown}; tasks: {', '.join(TASKS)}"
        raise OptionsError(msg)
    if only and commercial not in (None, "yes"):
        return []  # FASTFORKS_COMMERCIAL_ONLY hides every model that is not cleared
    commercial = "yes" if only else commercial
    return sorted(
        model_id
        for task in tasks
        for model_id, (card, v) in cards(task).items()
        if matches(model_id, filter)
        and not (exclude_filters and matches(model_id, exclude_filters))
        and (commercial is None or card.status(v).commercial == commercial)
        and (tier is None or card.tier == tier)
        and (family is None or card.family == family)
        and (runnable is None or (not missing(card) and not needs_nc_download(card, v)) == runnable)
    )


def _version(lib: str) -> tuple[int, ...] | None:
    """Installed version of `lib` from its metadata (no import); `None` if it has none."""
    try:
        text = importlib.metadata.version(lib)
    except importlib.metadata.PackageNotFoundError:
        return None
    return tuple(int(x) for x in re.findall(r"\d+", text.split("+")[0])[:3])


def _in_bounds(version: tuple[int, ...], spec: str) -> bool:
    """Whether `version` satisfies a `>=a.b,<c` spec of `LIBRARY_SPEC`."""
    for clause in spec.split(","):
        bound = tuple(int(x) for x in re.findall(r"\d+", clause))
        ok = version[: len(bound)] >= bound if clause.startswith(">=") else version[: len(bound)] < bound
        if not ok:
            return False
    return True


def problems(card: ModelCard) -> dict[str, str]:
    """`{library: why}` for the libraries of `card.requires` that cannot be used: absent, or an unsupported version.

    Looks libraries up with `find_spec` and `importlib.metadata`; nothing is imported (a library that is present but
    fails to import is reported by the helper that imports it).
    """
    found = {}
    for lib in card.requires:
        if importlib.util.find_spec(lib) is None:
            found[lib] = "absent"
            continue
        v = _version(lib)
        if v is not None and not _in_bounds(v, LIBRARY_SPEC[lib]):
            found[lib] = f"{lib}{LIBRARY_SPEC[lib]}, found {'.'.join(map(str, v))}"
    return found


def missing(card: ModelCard) -> tuple[str, ...]:
    """The libraries of `card.requires` that cannot be used here (absent or too old / new); see `problems`."""
    return tuple(problems(card))


def require(card: ModelCard, model: str) -> None:
    """Raise unless every library the model needs is installed in a supported version.

    Runs before any weight is read or fetched.

    Raises:
        BackendUnavailableError: A library is absent or its version is outside the bounds of the core extra; the
            message names the extra to install.
    """
    bad = problems(card)
    if bad:
        extras = sorted({LIBRARY_EXTRA[x] for x in bad})
        absent = [x for x, why in bad.items() if why == "absent"]
        parts = [f"{', '.join(absent)}, which is not installed"] if absent else []
        parts += [why for why in bad.values() if why != "absent"]
        upgrade = len(absent) < len(bad)
        msg = f"{model} needs {'; '.join(parts)}: {install_hint(*extras, upgrade=upgrade)}"
        if len(absent) == len(bad):  # only absent libraries: an import error as the cause (tests skip on it)
            cause = ModuleNotFoundError(f"No module named {absent[0]!r}", name=absent[0])
            raise MissingDependencyError(msg) from cause
        raise BackendUnavailableError(msg)


NONCOMMERCIAL_ENV = "FASTFORKS_NONCOMMERCIAL"


def noncommercial_enabled() -> bool:
    """Whether non-commercial weights may be downloaded: only with `FASTFORKS_NONCOMMERCIAL=1` (installing does not)."""
    return env_flag(NONCOMMERCIAL_ENV)


COMMERCIAL_ONLY_ENV = "FASTFORKS_COMMERCIAL_ONLY"


def commercial_only() -> bool:
    """Whether `FASTFORKS_COMMERCIAL_ONLY=1`: `load` refuses models not cleared for commercial use, `list` too."""
    return env_flag(COMMERCIAL_ONLY_ENV)


def require_commercial(card: ModelCard, variant: Variant) -> None:
    """With `FASTFORKS_COMMERCIAL_ONLY=1`: raise `LicenceError` unless the variant's code and weights are cleared.

    Also covers `weights=PATH` and weights already in the cache, which `nc_weights_blocked` lets through.
    """
    if not commercial_only():
        return
    st = card.status(variant)
    if st.commercial != "yes":
        why = f" ({'; '.join(st.reasons)})" if st.reasons else ""
        msg = (
            f"{variant.ref}: commercial use is {st.commercial!r}{why}, and {COMMERCIAL_ONLY_ENV}=1 refuses models not "
            f"cleared for it; unset the variable, or pick one of `fastforks list --commercial yes`"
        )
        raise LicenceError(msg)


def nc_weights_blocked(card: ModelCard, variant: Variant) -> bool:
    """Whether `weights` must not download the variant: its weights are non-commercial and not enabled.

    Local copies are still used (cache, `FASTFORKS_WEIGHTS_PATH`, `load(weights=...)`).
    """
    nc = licences.weights_commercial(card.variant_licence(variant).weights_licence) == "no"
    return nc and not noncommercial_enabled()


def needs_nc_download(card: ModelCard, variant: Variant) -> bool:
    """Whether the variant cannot load yet for its weights: they are blocked (`nc_weights_blocked`) and not local.

    "Local" is a copy of the right size in the cache, `FASTFORKS_WEIGHTS_PATH` or another tool's cache (no hashing).
    """
    return nc_weights_blocked(card, variant) and not all(
        present(spec, card.family, WEIGHTS_PATH_ENV) for spec in variant.weights
    )


def _nc_weights_error(variant: Variant, card: ModelCard) -> LicenceError:
    return LicenceError(
        f"{variant.ref}: the published weights are non-commercial ({card.status(variant).weights}), so fastforks "
        f"downloads them only on request: set {NONCOMMERCIAL_ENV}=1, or pass weights=PATH (or put the files on "
        f"{WEIGHTS_PATH_ENV})"
    )


def matches(model_id: str, patterns: str | Sequence[str]) -> bool:
    """Whether `model_id` matches a shell-style pattern (or any of several; case-insensitive, ids are lower case).

    An empty pattern (or list) matches everything, as in `timm.list_models`; callers test exclude filters for
    emptiness first.
    """
    if not patterns:
        return True
    pats = [patterns] if isinstance(patterns, str) else patterns
    return any(fnmatch.fnmatchcase(model_id, p.lower()) for p in pats)


def resolve(task: str, model: str) -> tuple[ModelCard, Variant]:
    """Look up `"id"` or `"id@revision"` among the models of `task`.

    Raises:
        ModelNotFoundError: Unknown id or not a string (the message names the right task if another task has the
            id), or the installed revision differs from the requested one.
    """
    model_id, at, rev = _split(model)
    entry = _lookup(task, model_id)
    if entry is None:
        raise _not_found(model_id, task)
    have = entry[1].revision
    if at and not (rev.isascii() and rev.isdigit() and int(rev) == int(have)):  # "@01" is "@1"; "@" alone is no pin
        msg = f"{model_id}: no revision {rev!r} (have {have}); pin a matching fastforks version"
        raise ModelNotFoundError(msg)
    return entry


def _split(model: object) -> tuple[str, str, str]:
    """`(id, "@", revision)` of an id string (`model.partition("@")`); a non-string is a `ModelNotFoundError`."""
    if not isinstance(model, str):
        msg = f"model must be an id string such as 'las2-s' or 'las2-s@1', got {type(model).__name__}"
        raise ModelNotFoundError(msg)
    return model.partition("@")


def resolve_any(model: str) -> tuple[ModelCard, Variant]:
    """`resolve` in whichever task has the model (ids are unique across tasks); the card says which task.

    Raises:
        ModelNotFoundError: No task has the id (with suggestions across all tasks), or the revision differs.
    """
    model_id = _split(model)[0]
    task = task_of(model)
    if task is None:
        raise _not_found(model_id, None)
    return resolve(task, model)


def variant_card(card: ModelCard, variant: Variant) -> ModelCard:
    """`card` with the variant's own `io`, `outputs` and `output_info`, and `variant_ref` set (`model_card(id)`)."""
    return dataclasses.replace(
        card, io=variant.io, outputs=variant.outputs, output_info=variant.output_info, variant_ref=variant.ref
    )


def task_of(model: str) -> str | None:
    """The task whose installed models include `model` (`"id"` or `"id@revision"`); `None` if none (or not a str)."""
    if not isinstance(model, str):
        return None
    model_id = model.partition("@")[0]
    return next((t for t in TASKS if _lookup(t, model_id) is not None), None)


def _lookup(task: str, model_id: str) -> tuple[ModelCard, Variant] | None:
    found = cards(task)
    return found.get(model_id) or found.get(model_id.lower()) or default_tags(task).get(model_id.lower())


@functools.cache
def default_tags(task: str) -> Mapping[str, tuple[ModelCard, Variant]]:
    """`{untagged name: (card, variant)}` of the variants marked `default_tag` (`"resnet50"` -> `resnet50.tv2_in1k`).

    Raises:
        CardError: Two variants are the default of the same name, or a name is also a model id.
    """
    found: dict[str, tuple[ModelCard, Variant]] = {}
    for model_id, entry in cards(task).items():
        if not entry[1].default_tag:
            continue
        name = model_id.split(".", 1)[0]
        if name in found or name in cards(task):
            other = found[name][1].id if name in found else name
            msg = f"{task} model {model_id!r} is marked default_tag, but {name!r} already means {other!r}"
            raise CardError(msg)
        found[name] = entry
    return MappingProxyType(found)


def _not_found(model_id: str, task: str | None) -> ModelNotFoundError:
    """The error for an id that `task` (`None`: every task) does not have, with the most useful hint."""
    other = task_of(model_id) if task is not None else None
    if other is not None:
        msg = f"{model_id!r} is a {other} model, not a {task} model: fastforks.{other}.load({model_id!r})"
        return ModelNotFoundError(msg)
    searched = (task,) if task is not None else TASKS
    for t in dict.fromkeys((*searched, *TASKS)):  # own task first; another task's package names that task
        dist = _known_models(t).get(model_id.lower())
        if dist is not None:
            msg = (
                f"{t} model {model_id!r} is in the package {dist}, which is not installed: "
                f'pip install "fastforks-core[{dist.removeprefix("fastforks-")}]" '
                "(or fastforks; check its licence first)"
            )
            return ModelNotFoundError(msg)
    pool = [mid for t in searched for mid in cards(t)]
    tagged = sorted(mid for mid in pool if mid.startswith(f"{model_id.lower()}."))
    if tagged:
        msg = (
            f"{model_id!r} has no default pretrained tag here; pass one of: {', '.join(tagged)}"
            " (timm-style `<name>.<tag>` ids)"
        )
        return ModelNotFoundError(msg)
    family = sorted(
        mid
        for t in searched
        for mid, (card, _) in cards(t).items()
        if card.family == model_id.lower().replace("-", "_")
    )
    if family:
        return ModelNotFoundError(f"{model_id!r} is a model family, not an id; its models: {', '.join(family)}")
    close = difflib.get_close_matches(model_id.lower(), pool, n=5)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    stem = model_id.lower().split("-")[0]
    if task is not None:
        msg = (
            f"unknown {task} model {model_id!r}{hint} ({len(pool)} installed models; "
            f"fastforks.{task}.list_models('*{stem}*'))"
        )
    else:
        msg = (
            f"unknown model {model_id!r} (no task has it; tasks: {', '.join(TASKS)}){hint} "
            f"({len(pool)} installed models; fastforks list '*{stem}*')"
        )
    return ModelNotFoundError(msg)


@functools.cache
def _known_models(task: str) -> dict[str, str]:
    """Every published model id of `task` -> its distribution (`<task>/known_models.toml`, from tools/gen_docs.py)."""
    path = Path(__file__).parents[1] / task / "known_models.toml"
    return tomllib.loads(path.read_text()) if path.is_file() else {}


def import_model(card: ModelCard) -> ModuleType:
    """Import the implementation module of a card (the only place model code is imported)."""
    return importlib.import_module(card.module)


WEIGHTS_PATH_ENV = "FASTFORKS_WEIGHTS_PATH"


def weights(card: ModelCard, variant: Variant, *, allow_download: bool = True) -> list[Path]:
    """Verified local paths of a variant's weight files (downloaded into the cache if needed).

    Raises:
        LicenceError: A file is missing and the weights are non-commercial but not enabled (`nc_weights_blocked`).
        WeightsError: A file is missing and cannot be downloaded, or has the wrong SHA-256.
    """
    blocked = nc_weights_blocked(card, variant)
    siblings = [s for v in card.variants for s in v.weights]  # one fetch per shared zip archive
    try:
        return [
            locate(
                spec,
                group=card.family,
                env_var=WEIGHTS_PATH_ENV,
                allow_download=allow_download and not blocked,
                siblings=siblings,
                hint=f"fastforks download {variant.ref}",
                model=variant.ref,
            )
            for spec in variant.weights
        ]
    except WeightsDownloadError as e:
        if blocked and allow_download:
            raise _nc_weights_error(variant, card) from e
        raise
