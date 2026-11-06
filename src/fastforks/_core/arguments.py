"""Keyword typos in calls reported as `ArgumentError` with a did-you-mean (no torch import: `load` uses it)."""

from __future__ import annotations

import difflib
import functools
import inspect
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from fastforks._core.errors import ArgumentError

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


def typo_checked_load(fn: Callable[P, R]) -> Callable[P, R]:
    """`fn` (a `load` function) with a keyword typo (`mdoe=`) reported as `ArgumentError`; free on the success path."""

    @functools.wraps(fn)
    def load(*args: P.args, **kwargs: P.kwargs) -> R:
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
