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


@pytest.mark.gpu
def test_correlation_windows_and_padded_rows_match_torch() -> None:
    from fastforks.tracking.models.cotracker.model import Core, _KernelCore  # noqa: PLC0415

    portable = object.__new__(Core)
    fused = object.__new__(_KernelCore)
    portable.r, portable.dt = 3, torch.float16
    frames = torch.arange(3, device="cuda").expand(5, 3)
    coords = torch.randn(3, 5, 2, device="cuda").transpose(0, 1) * 5
    coords[0] = coords.new_tensor([1.5, -2.25])
    grid = torch.randn(3, 7, 11, 13, dtype=torch.float16, device="cuda")
    win, frac = portable._window(grid, frames, coords)
    g, f = fused._window(grid, frames, coords)
    torch.testing.assert_close(g, win, atol=0, rtol=0)
    torch.testing.assert_close(f, frac, atol=0, rtol=0)
    storage = g.new_full((15, 49 * 13 + 7), 17)
    dst = storage[:, : 49 * 13].view(5, 3, 7, 7, 13)
    ref = portable._lerp(win, frac)
    assert fused._lerp(g, f, dst) is dst
    torch.testing.assert_close(dst, ref, atol=0, rtol=0)
    assert torch.all(storage[:, -7:] == 17)
    frame, point = frames[:, 0], coords[:, 0]
    got = fused.support([grid], frame, point)[0]
    torch.testing.assert_close(got, portable.support([grid], frame, point)[0], atol=0, rtol=0)
