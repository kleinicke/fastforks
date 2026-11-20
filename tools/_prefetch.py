"""Bound CPU read-ahead during dataset evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def prefetch(load: Callable[[_Input], _Output], items: Iterable[_Input]) -> Generator[_Output, None, None]:
    """Load the next item on a CPU thread while the caller processes the current item, preserving item order."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for item in items:
            previous = pending
            pending = pool.submit(load, item)
            if previous is not None:
                yield previous.result()
        if pending is not None:
            yield pending.result()
