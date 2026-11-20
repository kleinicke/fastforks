"""Bound CPU work ahead of the current evaluation item and keep input order."""

from __future__ import annotations

from contextlib import closing

import pytest

from _prefetch import prefetch


@pytest.mark.parametrize("workers", [1, 3])
def test_prefetch_limits_items_and_preserves_order(workers: int) -> None:
    requested: list[int] = []

    def items():
        for item in range(9):
            requested.append(item)
            yield item

    with closing(prefetch(str, items(), workers=workers)) as output:
        assert next(output) == "0"
        assert requested == list(range(workers + 1))
        assert list(output) == [str(item) for item in range(1, 9)]

    requested.clear()
    with closing(prefetch(str, items(), workers=workers)) as output:
        assert next(output) == "0"
    assert requested == list(range(workers + 1))
