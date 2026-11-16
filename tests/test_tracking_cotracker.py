"""CoTracker3 kernels backend: freeing the replaced fp16 weights must leave what the call still reads (LayerNorms)."""

from __future__ import annotations

import pytest
import torch

from fastforks import tracking


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize("model", ["cotracker3-offline", "cotracker3-online"])
def test_kernels_track_like_portable(model: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    g = torch.Generator().manual_seed(0)
    base = torch.nn.functional.avg_pool2d(torch.rand(1, 3, 112, 160, generator=g), 5, 1, 2)[0]
    frames = torch.stack([base[:, :96, 2 * i : 2 * i + 128] for i in range(8)]).mul(255 / base.max()).to(torch.uint8)
    queries = [[0, 40.0, 30.0], [2, 80.0, 50.0]]
    out = {}
    for mode in ("portable", "kernels"):
        try:
            m = tracking.load(model, mode=mode)
        except Exception as e:  # noqa: BLE001 - no kernels in this build
            pytest.skip(str(e))
        out[mode] = m(frames, queries).tracks.float().cpu()
        del m
    assert (out["kernels"] - out["portable"]).abs().max() < 3.0
