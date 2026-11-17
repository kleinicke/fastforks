"""Lotus normals: the card, the input sizes, the posterior sample and the output frame."""

from __future__ import annotations

import pytest
import torch

from fastforks import depth
from fastforks._core import registry
from fastforks.depth.models.lotus.model import finish, network_size, prepare
from fastforks.shared.sd2.net.vae import sample


def test_card() -> None:
    for name, generative in (("lotus-normal-g-v1-1", True), ("lotus-normal-d-v1-1", False)):
        card, variant = registry.cards("depth")[name]
        assert card.outputs == ("normals",)
        assert variant.params["generative"] is generative
        assert name in depth.list_models()


def test_sizes() -> None:
    """Upstream `resize_max_res` truncates in float arithmetic; no padding (the VAE floors odd sizes)."""
    assert network_size(100, 150, 768) == (512, 768)
    assert network_size(480, 641, 768) == (575, 768)
    assert tuple(prepare(torch.rand(1, 3, 480, 641), 768).shape[-2:]) == (575, 768)
    assert tuple(prepare(torch.rand(1, 3, 30, 40), 0).shape[-2:]) == (30, 40)
    with pytest.raises(ValueError, match="aspect ratio"):
        network_size(1, 2000, 768)


def test_sample() -> None:
    """diffusers `DiagonalGaussianDistribution.sample`: mean + exp(logvar / 2) x noise, logvar clamped to [-30, 20]."""
    mean, logvar, eps = (
        torch.randn(1, 4, 3, 3),
        torch.tensor([-40.0, 0.0, 2.0, 30.0]).view(1, 4, 1, 1),
        torch.randn(1, 4, 3, 3),
    )
    want = mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * eps
    torch.testing.assert_close(sample(torch.cat([mean, logvar.expand_as(mean)], 1), eps), want, rtol=0, atol=0)


def test_finish_nearest_exact_and_camera_frame() -> None:
    """(x / 2 + 0.5) clamped, nearest-exact resize, 2 x - 1, unit, Lotus (x left, y up, z out) -> (-x, -y, -z)."""
    n = torch.tensor([0.3, 0.4, 0.5]).view(1, 3, 1, 1).expand(1, 3, 4, 4).clone()
    n[:, :, :2, :2] = 5.0  # clamped to 1 before normalising
    out = finish(n, (8, 8))
    assert out.shape == (1, 8, 8, 3)
    torch.testing.assert_close(out[0, 7, 7], torch.tensor([-0.3, -0.4, -0.5]) / 0.5**0.5)
    torch.testing.assert_close(out[0, 0, 0], torch.tensor([-1.0, -1.0, -1.0]) / 3**0.5)
