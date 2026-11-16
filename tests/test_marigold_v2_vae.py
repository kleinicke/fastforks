"""Marigold V2 VAE: the low-memory RMS norm and the row-banded convolutions against their full-size forms (CPU)."""

import pytest
import torch
import torch.nn.functional as F

from fastforks.depth.models.marigold_v2.net import vae as V


def _vae(c: int, *, bands: bool) -> V.VAE2D:
    t = {
        "vae.latents_mean": torch.zeros(16),
        "vae.latents_std": torch.ones(16),
        "vae.n.gamma": torch.rand(c, 1, 1) + 0.5,
        "vae.c.weight": torch.randn(c, c, 3, 3) / (3 * c**0.5),
        "vae.c.bias": torch.randn(c),
    }
    return V.VAE2D(t, bands=bands)


def test_norm_is_upstreams():
    torch.manual_seed(0)
    v = _vae(24, bands=False)
    x = torch.randn(1, 24, 9, 13).bfloat16()
    gamma = v.t("n.gamma")
    ref = F.normalize(x.float(), dim=1).to(x.dtype) * (24**0.5) * gamma
    assert torch.equal(v.norm(x, "n"), ref)
    assert torch.equal(v.norm(x, "n", silu=True), F.silu(ref))


@pytest.mark.parametrize("up", [False, True])
def test_conv_bands(monkeypatch, up):
    torch.manual_seed(0)
    x = torch.randn(1, 8, 21, 11).bfloat16()
    full = _vae(8, bands=False)
    ref = full.conv(F.interpolate(x, scale_factor=2.0, mode="nearest-exact") if up else x, "c")
    monkeypatch.setattr(V, "_BAND", 1)  # the minimum band: 8 rows
    got = _vae(8, bands=True)
    got.load_state_dict(full.state_dict())
    torch.testing.assert_close(got.conv(x, "c", up=up), ref, rtol=1e-2, atol=1e-2)
