"""GeoCalib: the card, the pre-processing (resize / crop) and the device LM step."""

from __future__ import annotations

import torch

from fastforks import calibration
from fastforks._core import registry
from fastforks.calibration.models.geocalib.model import preprocess
from fastforks.calibration.models.geocalib.net.lm_optimizer import optimizer_step
from fastforks.calibration.models.geocalib.net.misc import J_vecnorm, SphericalManifold


def test_card() -> None:
    card, variant = registry.cards("calibration")["geocalib-pinhole"]
    assert variant.outputs == ("intrinsics", "gravity", "roll", "pitch", "vfov")
    assert card.params["camera_model"] == "pinhole"
    assert card.licence.code == "Apache-2.0"
    assert "geocalib-pinhole" in calibration.list_models()


def test_vecnorm_jacobian_exports_batch_wide_zero_guard() -> None:
    """One zero vector applies the upstream epsilon to the whole batch, including nonzero vectors."""

    class Jacobian(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return J_vecnorm(x)

    normal = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    zero = normal.clone()
    zero[0] = 0
    exported = torch.export.export(Jacobian(), (normal,)).module()
    for x in (normal, zero):
        norm = torch.norm(x, dim=-1, keepdim=True).unsqueeze(-1)
        if (norm == 0).any():
            norm = norm + 1e-6
        expected = torch.eye(3) / norm - torch.einsum("...i,...j->...ij", x, x) / norm**3
        torch.testing.assert_close(J_vecnorm(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(exported(x), expected, rtol=0, atol=0)


def test_householder_exports_small_sigma_guard() -> None:
    """The exported transform applies the same small-sigma adjustment as eager."""

    class Householder(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return SphericalManifold.householder_vector(x)

    normal = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, -3.0]])
    small = normal.clone()
    small[0] = torch.tensor([0.0, 0.0, 1.0])
    exported = torch.export.export(Householder(), (normal,)).module()
    for x in (normal, small):
        expected = Householder()(x)
        actual = exported(x)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_preprocess_shapes() -> None:
    """Shorter side to 320, crop to multiples of 32; scales and crop map back to the input."""
    x, scales, crop = preprocess(torch.rand(1, 3, 1920, 1440), 320, 32)
    assert x.shape == (1, 3, 416, 320)  # 426 -> 416
    torch.testing.assert_close(scales, torch.tensor([320 / 1440, 426 / 1920]))
    torch.testing.assert_close(crop, torch.tensor([0.0, -10.0]))
    x, _, crop = preprocess(torch.rand(2, 3, 300, 400), 320, 32)  # upscaled: no blur
    assert x.shape == (2, 3, 320, 416)
    assert crop.tolist() == [-10.0, 0.0]


def test_optimizer_step_solves_and_fails_to_zero() -> None:
    torch.manual_seed(0)
    a = torch.randn(2, 3, 3)
    h = a @ a.transpose(1, 2) + torch.eye(3)
    g = torch.randn(2, 3)
    lamb = torch.zeros(2)
    d = optimizer_step(g, h, lamb)
    h_damped = h + h.diagonal(dim1=-2, dim2=-1).mul(0).clamp(min=1e-6).diag_embed()
    torch.testing.assert_close(d, torch.linalg.solve(h_damped, g), atol=1e-5, rtol=1e-5)
    bad = h.clone()
    bad[1] = -torch.eye(3)  # not positive definite: the whole batch takes a zero step (as upstream)
    assert not optimizer_step(g, bad, lamb).any()
