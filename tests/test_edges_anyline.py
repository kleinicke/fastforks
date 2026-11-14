"""AnyLine: the card and the line-art helpers (Gaussian blur, median, small-object removal) vs references."""

from __future__ import annotations

import math

import numpy as np
import torch

from fastforks._core import registry
from fastforks.edges.models.anyline.model import gaussian_blur, median_above, remove_small


def test_card() -> None:
    cards = {v.id: c for c, v in registry.cards("edges").values() if c.family == "anyline"}
    assert sorted(cards) == ["anyline"]
    assert cards["anyline"].options["resolution"] == 1280


def _components(fg: np.ndarray) -> np.ndarray:
    """4-connected component sizes per pixel of a 2D bool mask (breadth-first search)."""
    size = np.zeros(fg.shape, int)
    seen = np.zeros(fg.shape, bool)
    h, w = fg.shape
    for i, j in zip(*np.nonzero(fg), strict=True):
        if seen[i, j]:
            continue
        todo, comp = [(i, j)], []
        seen[i, j] = True
        while todo:
            a, b = todo.pop()
            comp.append((a, b))
            for c, d in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                if 0 <= c < h and 0 <= d < w and fg[c, d] and not seen[c, d]:
                    seen[c, d] = True
                    todo.append((c, d))
        for a, b in comp:
            size[a, b] = len(comp)
    return size


def test_remove_small_is_exact() -> None:
    g = torch.Generator().manual_seed(0)
    for p in (0.3, 0.45, 0.55, 0.7):  # around the percolation threshold: components of every size and shape
        fg = torch.rand(3, 40, 56, generator=g) < p
        want = np.stack([_components(m) >= 12 for m in fg.numpy()]) & fg.numpy()
        assert (remove_small(fg).numpy() == want).all()


def test_gaussian_blur_and_median() -> None:
    cv2 = __import__("cv2")
    x = torch.rand(1, 3, 37, 50, generator=torch.Generator().manual_seed(1)) * 255
    want = cv2.GaussianBlur(x[0].permute(1, 2, 0).numpy(), (0, 0), 2.0)
    torch.testing.assert_close(gaussian_blur(x)[0].permute(1, 2, 0), torch.from_numpy(want), rtol=0, atol=1e-3)
    v = torch.tensor([[1.0, 5, 4, 9, 2, 7], [0, 1, 2, 3, 3, 3], [4, 6, 0, 0, 0, 0]])
    got = median_above(v, 3).tolist()
    assert got[0] == np.median([5, 4, 9, 7])
    assert math.isnan(got[1])
    assert got[2] == 5
