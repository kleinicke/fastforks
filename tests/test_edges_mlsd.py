"""M-LSD: the card, the segment decoding and the `cv2.line` rasterizer vs OpenCV."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from fastforks._core import registry
from fastforks.edges.models.mlsd.model import decode, draw


def test_card() -> None:
    cards = {v.id: c for c, v in registry.cards("edges").values() if c.family == "mlsd"}
    assert sorted(cards) == ["mlsd"]
    assert cards["mlsd"].licence.code == "Apache-2.0"
    assert cards["mlsd"].options["thr_v"] == 0.1


def test_draw_matches_cv2_line() -> None:
    """Exact `cv2.line(thickness=1)` incl. segments crossing or outside the border, batched with a keep mask."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        h, w = (int(v) for v in rng.integers(5, 50, 2))
        seg = rng.integers(-40, 90, (2, 6, 4))
        keep = rng.random((2, 6)) < 0.8
        got = draw(torch.from_numpy(seg), torch.from_numpy(keep), h, w).numpy()
        for b in range(2):
            want = np.zeros((h, w), np.uint8)
            for (x0, y0, x1, y1), k in zip(seg[b].tolist(), keep[b], strict=True):
                if k:
                    cv2.line(want, (x0, y0), (x1, y1), 255, 1)
            np.testing.assert_array_equal(got[b], want / 255)


def test_decode() -> None:
    """One center: `2 * (center + displacement)`, kept only above both thresholds."""
    maps = torch.full((1, 9, 8, 8), -10.0)
    maps[0, 0, 3, 5] = 2.0  # sigmoid 0.88
    maps[0, 1:5, 3, 5] = torch.tensor([-1.0, 0.5, 2.0, -0.5])
    seg, keep = decode(maps, 0.1, 0.1)
    assert keep.sum() == 1
    assert seg[keep].tolist() == [[8.0, 7.0, 14.0, 5.0]]
    _, keep = decode(maps, 0.9, 0.1)
    assert not keep.any()
    _, keep = decode(maps, 0.1, 4.0)
    assert not keep.any()
