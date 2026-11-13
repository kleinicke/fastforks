"""AnyCalib: the card, the input size and crop, the batched camera fit."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastforks import calibration
from fastforks._core import registry
from fastforks.calibration.models.anycalib.model import Params, fit, preprocess, target_size

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
