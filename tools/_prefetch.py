"""Bound CPU read-ahead during dataset evaluation."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from concurrent.futures import Future

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def prefetch(
    load: Callable[[_Input], _Output], items: Iterable[_Input], *, workers: int = 1
) -> Generator[_Output, None, None]:
    """Load at most `workers` items ahead of the current item and yield them in input order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque[Future[_Output]] = deque()
        for item in items:
            pending.append(pool.submit(load, item))
            if len(pending) > workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
