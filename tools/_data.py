"""Where the evaluation datasets live: `$FASTFORKS_DATA` (default `~/.cache/fastforks/data`)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

DATA = Path(os.environ.get("FASTFORKS_DATA", "~/.cache/fastforks/data")).expanduser()

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def prefetch(load: Callable[[_Input], _Output], items: Iterable[_Input]) -> Generator[_Output, None, None]:
    """Load the next dataset item on a CPU thread while the caller processes the current item."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for item in items:
            previous = pending
            pending = pool.submit(load, item)
            if previous is not None:
                yield previous.result()
        if pending is not None:
            yield pending.result()
