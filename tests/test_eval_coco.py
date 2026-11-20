"""COCO point prompts retain the annotation id of each nonempty mask."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

import eval_coco

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_point_inputs_keep_annotation_ids_when_a_mask_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "000000000004.jpg"
    Image.new("L", (6, 4), color=17).save(path)
    empty, valid, zero_area = {"id": 10, "area": 3}, {"id": 11, "area": 4}, {"id": 12, "area": 0}
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 1:3] = 1

    def decode(ann: dict[str, int]) -> np.ndarray:
        assert ann != zero_area
        return mask if ann == valid else np.zeros_like(mask)

    coco = SimpleNamespace(
        getAnnIds=lambda **_: [10, 11, 12],
        loadAnns=lambda _: [empty, valid, zero_area],
        annToMask=decode,
    )
    monkeypatch.setattr(eval_coco, "_point", lambda _: (1.0, 1.0))

    image_id, rgb, ids, masks, points = eval_coco._point_inputs(coco, path)

    assert image_id == 4
    assert rgb.shape == (4, 6, 3)
    assert ids == [11]
    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0], mask.astype(bool))
    assert points == [(1.0, 1.0)]
