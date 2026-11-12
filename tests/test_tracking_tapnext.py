"""Tracking: the query checks, the video checks and the bench inputs (no weights)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastforks._core.errors import InputError
from fastforks.tracking._api import TrackingModel, _queries, grid_queries, synthetic_video


def test_queries_shapes_and_frame_indices() -> None:
    q, single = _queries([[0, 1.5, 2.0], [3, 4, 5]], 4)
    assert not single
    assert q.dtype == torch.float32
    assert q.shape == (2, 3)
    q, single = _queries(np.array([2, 1.0, 1.0]), 4)
    assert single
    assert q.shape == (1, 3)
    for bad, match in (
        ([[0, 1]], r"\[Q, 3\]"),
        (np.zeros((0, 3)), r"\[Q, 3\]"),
        ([[0.5, 1, 1]], "integers"),
        ([[4, 1, 1]], "integers"),
        ([[-1, 1, 1]], "integers"),
        ([[0, float("nan"), 1]], "non-finite"),
        (torch.ones(1, 3, dtype=torch.complex64), "real"),
        ([["a", 1, 1]], "numeric"),
    ):
        with pytest.raises(InputError, match=match):
            _queries(bad, 4)


def _model() -> TrackingModel:
    """A `TrackingModel` without weights: size checks pass, the runner returns frame 1 (checking its size)."""
    m = object.__new__(TrackingModel)
    m.device = torch.device("cpu")
    m._check_size = lambda _: None
    m._session = torch.no_grad
    m._runner = SimpleNamespace()
    return m


def test_video_checks() -> None:
    m = _model()
    for bad, match in (("clip.mp4", "sequence"), ([], "no frames"), (3, "sequence")):
        with pytest.raises(InputError, match=match):
            m(bad, [[0, 1, 1]])
    m._runner = lambda frame, *_: frame(1)
    with pytest.raises(InputError, match="one size"):
        m([np.zeros((8, 8, 3), np.uint8), np.zeros((8, 9, 3), np.uint8)], [[0, 1, 1]])
    with pytest.raises(InputError, match="one image per frame"):
        m._frame([np.zeros((2, 8, 8, 3), np.uint8)], 0)


def test_bench_inputs_move_and_grid() -> None:
    v = synthetic_video(3, 16, 20, torch.device("cpu"))
    assert v.shape == (3, 3, 16, 20)
    torch.testing.assert_close(v[1, :, :, :-2], v[0, :, :, 2:])  # moves left by 2 px per frame
    g = grid_queries(16, 20, 4, torch.device("cpu"))
    assert g.shape == (16, 3)
    assert (g[:, 0] == 0).all()
    assert (g[:, 1].min(), g[:, 1].max(), g[:, 2].min(), g[:, 2].max()) == (2.0, 17.0, 1.5, 13.5)
