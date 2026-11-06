"""Class-name lists of the label sets that classification and segmentation cards name in `[output] label_set`.

One name per line in `<label set>.txt`, in class-index order.
"""

from __future__ import annotations

import functools
from pathlib import Path

from fastforks._core.errors import CardError


@functools.cache
def label_set(name: str) -> tuple[str, ...]:
    """Class names of `name` (e.g. `"imagenet-1k"`), index i = class i.

    Raises:
        CardError: No such label set.
    """
    path = Path(__file__).parent / f"{name}.txt"
    if "/" in name or not path.is_file():
        known = sorted(p.stem for p in Path(__file__).parent.glob("*.txt"))
        msg = f"unknown label set {name!r} (known: {known})"
        raise CardError(msg)
    return tuple(path.read_text(encoding="utf-8").splitlines())
