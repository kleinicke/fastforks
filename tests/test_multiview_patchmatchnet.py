"""PatchmatchNet: the kernels path's CUDA graphs."""

from __future__ import annotations

import pytest
import torch

import fastforks
from fastforks._core.bench import BenchSize, bench_inputs
from fastforks._core.errors import WeightsError


@pytest.mark.gpu
def test_kernels_graphs_match_eager() -> None:
    """The graphed kernels forward: bit-equal to eager per input size and depth range; release frees the graphs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        graphed = fastforks.load("patchmatchnet", mode="kernels", device="cuda")
        eager = fastforks.load("patchmatchnet", mode="kernels", device="cuda", cuda_graphs=False)
    except WeightsError as e:
        pytest.skip(f"patchmatchnet weights not available: {e}")
    assert eager._runner.graphs is None
    small, wide = (bench_inputs(graphed, BenchSize(h, w, 3)) for h, w in ((128, 160), (96, 224)))
    near = small[1] | {"depth_range": tuple(0.5 * float(v) for v in small[1]["depth_range"])}
    calls = [small] * 3 + [(small[0], near)] * 3 + [wide] * 3 + [small]
    with torch.inference_mode():
        for args, kw in calls:
            got, want = graphed(*args, **kw), eager(*args, **kw)
            torch.testing.assert_close(got.depth, want.depth, rtol=0, atol=0)
            torch.testing.assert_close(got.confidence, want.confidence, rtol=0, atol=0)
    graphs = graphed._runner.graphs
    assert len(graphs.graphs) == 2, (
        f"want one graph per depth range (the cache holds 2) (failed: {list(graphs._failed)})"
    )
    graphed.release_graphs()
    assert not graphs.graphs
