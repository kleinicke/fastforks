"""Keyword typos in calls reported as `ArgumentError` with a did-you-mean (no torch import: `load` uses it)."""

from __future__ import annotations

import difflib
import functools
import inspect
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from fastforks._core.errors import ArgumentError, OptionsError

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")


def bind_error(
    fn: Callable[..., Any], label: str, args: tuple[Any, ...], kwargs: dict[str, Any], *, skip: int = 0
) -> ArgumentError | None:
    """An `ArgumentError` (with a did-you-mean) if the arguments do not bind to `fn`'s signature, else `None`.

    Only called after a `TypeError`, so a successful call pays nothing for it. `label` names the call in the message;
    `skip` leading parameters (the model, `self`) are not listed as valid keywords.
    """
    sig = inspect.signature(fn)
    try:
        sig.bind(*args, **kwargs)
    except TypeError as e:
        names = [n for n, p in sig.parameters.items() if p.kind is not inspect.Parameter.VAR_KEYWORD][skip:]
        unknown = [k for k in kwargs if k not in sig.parameters]
        if not unknown:
            return ArgumentError(f"{label}: {e}")
        hints = []
        for k in unknown:
            close = difflib.get_close_matches(k, names, n=1)
            hints.append(f"{k!r} (did you mean {close[0]!r}?)" if close else repr(k))
        return ArgumentError(
            f"{label} got an unexpected keyword argument {', '.join(hints)}; valid: {', '.join(names)}"
        )
    return None


REMOVAL = "2026-12-15"  # renamed 0.1.1 names keep working, with a FutureWarning, until this date


def warn_renamed(old: str, new: str, *, stacklevel: int = 3) -> None:
    """A `FutureWarning` that `old` is renamed to `new` and goes away after `REMOVAL`."""
    warnings.warn(f"{old} is renamed to {new}; the old name is removed after {REMOVAL}", FutureWarning, stacklevel)


def _option_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """`kwargs` of a `load(..., **options)` call with a legacy `options={...}` merged in (warned)."""
    legacy = kwargs.pop("options", None)
    if legacy is None:
        return kwargs
    warn_renamed("load(..., options={...})", "load(..., **options)", stacklevel=4)
    if not isinstance(legacy, Mapping):
        msg = f"options must be a dict such as {{'iters': 8}}, got {type(legacy).__name__}"
        raise OptionsError(msg)
    params = inspect.signature(fn).parameters
    if clash := sorted(k for k in legacy if k in params or k in kwargs):
        msg = f"options={{...}} has {clash}: each is a load() argument or is also given as a keyword"
        raise OptionsError(msg)
    return {**kwargs, **legacy}


def _typo(fn: Callable[..., Any], kwargs: dict[str, Any]) -> ArgumentError | None:
    """An `ArgumentError` for an option keyword close to a `load` argument (`mdoe=`) that no card has as an option."""
    params = [n for n, p in inspect.signature(fn).parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY]
    for k in kwargs:
        if k in params or not (close := difflib.get_close_matches(k, params, n=1, cutoff=0.7)):
            continue
        from fastforks._core import registry  # noqa: PLC0415 - only on a near miss

        if k not in registry.option_names():
            return ArgumentError(
                f"{fn.__module__}.load(...) got an unexpected keyword argument {k!r} (did you mean {close[0]!r}?); "
                f"valid: {', '.join(params)}, or the model's options"
            )
    return None


def typo_checked_load(fn: Callable[P, R]) -> Callable[P, R]:
    """`fn` (a `load(model, *, ..., **options)` function) with keyword typos reported as `ArgumentError`.

    A keyword close to one of `fn`'s arguments that is no card's option (`mdoe=`) is a typo; the legacy
    `options={...}` is merged into the keywords with a `FutureWarning`. Free on the success path.
    """

    @functools.wraps(fn)
    def load(*args: P.args, **kwargs: P.kwargs) -> R:
        kwargs = _option_kwargs(fn, kwargs)  # ty: ignore[invalid-assignment]
        if err := _typo(fn, kwargs):
            raise err
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            if isinstance(e, ArgumentError):
                raise
            err = bind_error(fn, f"{fn.__module__}.load(...)", args, kwargs, skip=1)
            if err is None:
                raise
            raise err from e

    return load
