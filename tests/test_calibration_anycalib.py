"""AnyCalib: the card, the input size and crop, the batched camera fit."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastforks import calibration
from fastforks._core import registry
from fastforks._core.errors import WeightsError
from fastforks.calibration.models.anycalib.model import Params, _AnyCalib, fit, preprocess, target_size
from fastforks.calibration.models.anycalib.net.utils import solve_small_export
from fastforks.kernels import cudnn_hacc

P = Params(resolution=102400, edge=14, ar_min=0.5, ar_max=2.0, gn_iters=10, cuda_graphs=False)


def test_card() -> None:
    card, variant = registry.cards("calibration")["anycalib-pinhole"]
    assert variant.outputs == ("intrinsics", "vfov")
    assert card.licence.code == card.licence.weights == "Apache-2.0"
    assert "anycalib-pinhole" in calibration.list_models()


def test_target_size_and_crop() -> None:
    """Upstream `compute_target_size` / `set_im_size`: LaMAR portrait, a wide crop, an upsampled small input."""
    assert target_size(1920, 1440, P) == (364, 280)
    assert target_size(100, 1000, P) == target_size(200, 400, P)  # aspect ratio clamped to 0.5
    x, scale, shift = preprocess(torch.rand(1, 3, 400, 600), target_size(400, 600, P))
    assert x.shape == (1, 3, 266, 392)
    c = round(600 - 400 * 392 / 266)  # width cropped to the target aspect ratio
    torch.testing.assert_close(scale, torch.tensor([392 / (600 - c), 266 / 400]))
    torch.testing.assert_close(shift, torch.tensor([-(c // 2) * 392 / (600 - c), 0.0]))
    x, scale, _ = preprocess(torch.rand(2, 3, 100, 140), (266, 392))  # upsampled first
    assert x.shape == (2, 3, 266, 392)
    assert (scale > 2).all()
    x = torch.rand(1, 3, 266, 392)
    y, scale, shift = preprocess(x, (266, 392))
    assert y is x
    assert scale.tolist() == [1.0, 1.0]
    assert shift.tolist() == [0.0, 0.0]


def test_fit_recovers_pinhole_batched() -> None:
    """Rays of known pinholes give their intrinsics back; a degenerate ray field gives NaN, not its neighbour's."""
    h, w = 20, 30
    k = torch.tensor([[40.0, 42.0, 15.5, 9.0], [25.0, 25.0, 14.0, 11.0], [1.0, 1.0, 0.0, 0.0]])
    xs, ys = torch.meshgrid(torch.arange(w) + 0.5, torch.arange(h) + 0.5, indexing="xy")
    uv = torch.stack([xs, ys], -1).view(-1, 2)
    rays = F.normalize(torch.cat([(uv - k[:, None, 2:]) / k[:, None, :2], torch.ones(3, h * w, 1)], -1), dim=-1)
    rays[2] = torch.tensor([0.0, 0.0, 1.0])  # every pixel the same ray: singular linear fit
    got = fit(rays, h, w, 10)
    torch.testing.assert_close(got[:2], k[:2], atol=1e-3, rtol=1e-4)
    assert got[2].isnan().all()


def test_export_solver_matches_and_detects_singular() -> None:
    """The export-only elimination handles pivoting and reports a singular system."""
    torch.manual_seed(13)
    for n in (2, 4):
        a = torch.randn(100, n, n)
        a = a.transpose(-1, -2) @ a + 0.1 * torch.eye(n)
        b = torch.randn(100, n)
        actual, info = solve_small_export(a, b)
        expected, expected_info = torch.linalg.solve_ex(a, b)
        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
        assert torch.equal(info, expected_info)
        _, singular = solve_small_export(torch.zeros(1, n, n), torch.ones(1, n))
        assert singular.item() != 0
    a = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    b = torch.tensor([[1.0, 5.0]])
    actual, info = solve_small_export(a, b)
    torch.testing.assert_close(actual, torch.tensor([[1.0, 1.0]]))
    assert info.item() == 0


def test_export_fit_preserves_degenerate_rays() -> None:
    """Exported camera fitting keeps finite solutions and the eager singular-fit behavior."""

    class Fitter(torch.nn.Module):
        def forward(self, rays: torch.Tensor) -> torch.Tensor:
            return fit(rays, 20, 30, 2)

    h, w = 20, 30
    k = torch.tensor([[40.0, 42.0, 15.5, 9.0], [25.0, 25.0, 14.0, 11.0], [1.0, 1.0, 0.0, 0.0]])
    xs, ys = torch.meshgrid(torch.arange(w) + 0.5, torch.arange(h) + 0.5, indexing="xy")
    uv = torch.stack([xs, ys], -1).view(-1, 2)
    rays = F.normalize(torch.cat([(uv - k[:, None, 2:]) / k[:, None, :2], torch.ones(3, h * w, 1)], -1), dim=-1)
    rays[2] = torch.tensor([0.0, 0.0, 1.0])
    eager = Fitter()(rays)
    exported = torch.export.export(Fitter(), (rays,), strict=False).module()(rays)
    torch.testing.assert_close(exported[:2], eager[:2], atol=1e-3, rtol=1e-4)
    assert exported[2].isnan().all()


@pytest.mark.gpu
@pytest.mark.model
def test_kernels_graph_replays_network_and_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kernels graph holds the network and the camera fit: replay equals eager bit for bit, with fresh outputs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    monkeypatch.setattr(cudnn_hacc, "BUILD_AFTER", 0)  # plans on the first call, the graph on the second
    monkeypatch.setattr(cudnn_hacc, "CAPTURE_AFTER", 1)
    try:
        model = calibration.load("anycalib-pinhole", mode="kernels", device="cuda")
    except WeightsError as e:
        pytest.skip(f"anycalib-pinhole weights not available: {e}")
    r = model._runner
    cache = r.graphs
    assert cache is not None
    g = torch.Generator().manual_seed(0)
    a, b = (torch.rand(1, 3, 266, 392, generator=g).cuda() for _ in range(2))
    with torch.inference_mode():
        want_a = _AnyCalib._camera(r, a)  # the eager network + fit
        want_b = _AnyCalib._camera(r, b)
        r._camera(a)  # first call: eager
        assert not cache.graphs
        first = r._camera(a)  # captures
        assert len(cache.graphs) == 1
        second = r._camera(b)  # replay with new data; `first` must not change
    torch.testing.assert_close(first, want_a, rtol=0, atol=0)
    torch.testing.assert_close(second, want_b, rtol=0, atol=0)
    assert first.data_ptr() != second.data_ptr()
    model.release_graphs()
    assert not cache.graphs
