"""Track-On2: the support grid, the memory extension and the deformable sampling (no weights)."""

import torch
import torch.nn.functional as F

from fastforks.tracking.models.track_on.model import grid
from fastforks.tracking.models.track_on.net.track_on import TrackOn2, msda_sample


def test_support_grid_matches_upstream() -> None:
    g = grid(3, 100, 128, torch.device("cpu"))  # upstream get_points_on_a_grid(3, (100, 128)): margin 128 / 64
    want = torch.tensor([[2.0, 2.0], [64.0, 2.0], [126.0, 2.0], [2.0, 50.0], [64.0, 50.0], [126.0, 50.0]])
    torch.testing.assert_close(g[:6], want)
    assert g.shape == (9, 2)
    torch.testing.assert_close(grid(1, 100, 128, torch.device("cpu")), torch.tensor([[64.0, 50.0]]))


def test_extend_memory_keeps_the_current_slot() -> None:
    t = torch.arange(2 * 5 * 3, dtype=torch.float32).view(1, 10, 3)
    out = TrackOn2.extend_memory(t, 18)
    assert out.shape == (1, 19, 3)
    torch.testing.assert_close(out[:, -1], t[:, -1])
    torch.testing.assert_close(out[:, [0, -2]], t[:, [0, -2]])  # align_corners: the ends stay


def test_msda_sample_is_weighted_grid_sample() -> None:
    torch.manual_seed(0)
    shapes = [(4, 6), (2, 3)]
    value = torch.randn(1, 4 * 6 + 2 * 3, 2, 5)
    loc = torch.rand(1, 3, 2, 2, 1, 2)
    w = torch.rand(1, 3, 2, 2, 1)
    out = msda_sample(value, shapes, loc, w)
    assert out.shape == (1, 3, 10)
    want = torch.zeros(1, 3, 2, 5)
    start = 0
    for lvl, (h, wl) in enumerate(shapes):
        v = value[0, start : start + h * wl].permute(1, 2, 0).reshape(2, 5, h, wl)
        start += h * wl
        for head in range(2):
            g = 2 * loc[0, :, head, lvl, 0] - 1  # [Q, 2]
            s = F.grid_sample(v[head : head + 1], g[None, :, None], align_corners=False)[0, :, :, 0].T  # [Q, 5]
            want[0, :, head] += s * w[0, :, head, lvl, 0, None]
    torch.testing.assert_close(out, want.flatten(2))
