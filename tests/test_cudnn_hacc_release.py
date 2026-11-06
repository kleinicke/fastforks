"""The cudnn_hacc plan workspace is freed with the last live model (CPU-safe: a stand-in instance)."""

from __future__ import annotations

import gc
from types import SimpleNamespace
from typing import TYPE_CHECKING

from fastforks.kernels import cudnn_hacc

if TYPE_CHECKING:
    import pytest


class _Owner:
    pass


def test_release_after_last_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    freed: list[int] = []

    class Fake:
        def free_workspaces(self) -> None:
            freed.append(1)

    class Cached:
        cache_info = staticmethod(lambda: SimpleNamespace(currsize=1))

        def __call__(self) -> Fake:
            return Fake()

    monkeypatch.setattr(cudnn_hacc, "_instance", Cached())
    monkeypatch.setattr(cudnn_hacc, "_users", 0)
    a, b = _Owner(), _Owner()
    cudnn_hacc.attach(a)
    cudnn_hacc.attach(b)
    del a
    gc.collect()
    assert not freed
    del b
    gc.collect()
    assert freed == [1]
    assert cudnn_hacc._users == 0
