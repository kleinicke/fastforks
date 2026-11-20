"""Bound CPU work ahead of the current evaluation item and keep input order."""

from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING

import pytest
import torch
from PIL import Image

from _prefetch import prefetch
from eval_classification import _loader

if TYPE_CHECKING:
    from pathlib import Path


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


def test_classification_prefetch_keeps_batches_and_labels(tmp_path: Path) -> None:
    files = []
    for label, height in enumerate((3, 4, 5), start=2):
        path = tmp_path / f"{label}.png"
        Image.new("RGB", (6, height), (label, 0, 0)).save(path)
        files.append((path, label))

    raw = list(_loader(files, None, 2))
    assert [labels.tolist() for _, labels in raw] == [[2, 3], [4]]
    assert [tuple(x.shape) for batch, _ in raw for x in batch] == [(3, 3, 6), (3, 4, 6), (3, 5, 6)]

    def transform(image: Image.Image) -> torch.Tensor:
        return torch.full((3, 2, 2), image.getpixel((0, 0))[0], dtype=torch.float32)

    transformed = list(_loader(files, transform, 2))
    assert [labels.tolist() for _, labels in transformed] == [[2, 3], [4]]
    for batch, labels in transformed:
        assert isinstance(batch, torch.Tensor)
        assert torch.equal(batch, labels[:, None, None, None].float().expand(-1, 3, 2, 2))
