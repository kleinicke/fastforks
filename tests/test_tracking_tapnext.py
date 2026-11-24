"""Tracking: the query checks, the video checks and the bench inputs (no weights).

Also the TAPNext kernels per-frame CUDA graphs against the eager runner (GPU, weights).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastforks import tracking
from fastforks._core.errors import InputError, WeightsError
from fastforks.tracking._api import TrackingModel, _queries, grid_queries, synthetic_video


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_queries_on_mps_keep_their_values() -> None:
    """MPS queries reach the CPU unchanged (`.to("cpu", torch.float64)` from MPS gave wrong values in torch 2.14)."""
    q = grid_queries(480, 854, 16, torch.device("mps"))
    assert torch.equal(_queries(q, 24)[0], q.cpu())


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


@pytest.mark.model
@pytest.mark.gpu
def test_tapnext_kernels_frame_graphs_match_eager() -> None:
    """The graphed frame steps: bit-equal to eager across videos and query counts; release frees them."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        graphed = tracking.load("tapnext@1", mode="kernels", device="cuda")
        eager = tracking.load("tapnext@1", mode="kernels", device="cuda", cuda_graphs=False)
    except WeightsError as e:
        pytest.skip(f"tapnext weights not available: {e}")
    assert eager._runner.steps is None
    gen = torch.Generator("cuda").manual_seed(0)
    a, b = (torch.rand(5, 3, 120, 160, device="cuda", generator=gen) for _ in range(2))
    few = grid_queries(120, 160, 2, torch.device("cuda"))
    many = torch.cat([few, few + torch.tensor([1.0, 3.0, 2.0], device="cuda")])
    for video, queries in ((a, few), (a, few), (b, few), (a, many), (b, many), (b, few)):
        got, want = graphed(video, queries), eager(video, queries)
        torch.testing.assert_close(got.tracks, want.tracks, rtol=0, atol=0)
        torch.testing.assert_close(got.visibility, want.visibility, rtol=0, atol=0)
    steps = graphed._runner.steps
    assert steps is not None
    assert len(steps.graphs) == 2, f"want one recurrent graph per query count (failed: {list(steps._failed)})"
    graphed.release_graphs()
    assert not steps.graphs
