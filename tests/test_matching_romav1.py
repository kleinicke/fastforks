"""RoMa v1: the card, the conversion of the normalised warp to image2 pixels, the local correlation, the soft-argmax."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastforks import matching
from fastforks._core import registry
from fastforks.matching.models.romav1 import model as family
from fastforks.matching.models.romav1.net.local_correlation import local_correlation
from fastforks.matching.models.romav1.net.roma import cls_to_flow_refine, grid


def test_card() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "romav1"}
    assert sorted(cards) == ["romav1"]
    card, variant = cards["romav1"]
    assert dict(card.options) == {"coarse_size": 560, "fine_size": 864}  # upstream roma_outdoor defaults
    assert [w.name for w in variant.weights] == ["roma_outdoor.pth", "dinov2_vitl14_pretrain.pth"]
    assert "romav1" in matching.list_models()


class _Identity(torch.nn.Module):
    """The identity warp (every pixel to the same normalised place) at the last size, certainty 0.25."""

    def forward(self, a: torch.Tensor, _b: torch.Tensor, a_hi: torch.Tensor | None, _bh: object) -> tuple:
        b, _, h, w = (a if a_hi is None else a_hi).shape
        return grid(b, h, w, a.device), torch.full((b, h, w), 0.25)


@pytest.mark.parametrize(("coarse", "fine"), [(28, 0), (14, 32)])
def test_warp_in_image2_pixels(coarse: int, fine: int) -> None:
    """Pixel centres of image1 (32 x 32) land on the same relative place of image2 (48 x 64)."""
    out = family.forward(_Identity(), torch.rand(2, 3, 32, 32), torch.rand(2, 3, 48, 64), family.Params(coarse, fine))
    assert out["warp"].shape == (2, 2, 32, 32)
    torch.testing.assert_close(out["certainty"], torch.full((2, 32, 32), 0.25))
    ys, xs = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
    want = torch.stack(((xs + 0.5) * 64 / 32 - 0.5, (ys + 0.5) * 48 / 32 - 0.5))
    inner = (slice(None), slice(2, -2), slice(2, -2))  # 28 -> 32: bilinear of a linear map, exact off the border
    torch.testing.assert_close(out["warp"][0][inner], want[inner], rtol=0, atol=1e-4)


def test_sizes_must_be_multiples() -> None:
    with pytest.raises(ValueError, match="multiple of 14"):
        matching.load("romav1", device="cpu", options={"coarse_size": 100})


def _upstream_local_correlation(f0: torch.Tensor, f1: torch.Tensor, r: int, warp: torch.Tensor) -> torch.Tensor:
    """Upstream `shitty_native_torch_local_corr`: one grid_sample of all window coordinates."""
    b, c, h, w = f0.shape
    k = (2 * r + 1) ** 2
    wy, wx = torch.meshgrid(
        torch.linspace(-2 * r / h, 2 * r / h, 2 * r + 1),
        torch.linspace(-2 * r / w, 2 * r / w, 2 * r + 1),
        indexing="ij",
    )
    window = torch.stack((wx, wy), dim=-1).reshape(1, k, 2)
    coords = (warp[:, :, :, None] + window[:, None, None]).reshape(b, h, w * k, 2)
    feat = F.grid_sample(f1, coords, padding_mode="zeros", align_corners=False).reshape(b, c, h, w, k)
    return (f0[..., None] / c**0.5 * feat).sum(dim=1).permute(0, 3, 1, 2)


def test_local_correlation_matches_upstream() -> None:
    torch.manual_seed(0)
    f0, f1 = torch.randn(2, 8, 9, 11), torch.randn(2, 8, 9, 11)
    warp = grid(2, 9, 11, torch.device("cpu")).permute(0, 2, 3, 1) + torch.randn(2, 9, 11, 2) / 3
    got = local_correlation(f0, f1, 2, warp)
    torch.testing.assert_close(got, _upstream_local_correlation(f0, f1, 2, warp), rtol=0, atol=1e-5)


def test_soft_argmax_of_a_peak_is_its_anchor() -> None:
    """A single dominant class gives (almost) its anchor; the neighbour probabilities pull it slightly."""
    cls = torch.full((1, 16, 1, 1), -30.0)
    cls[0, 5] = 30.0  # anchor (x 1, y 1) of the 4 x 4 grid: (-0.25, -0.25)
    flow = cls_to_flow_refine(cls)
    torch.testing.assert_close(flow[0, 0, 0], torch.tensor([-0.25, -0.25]), rtol=0, atol=1e-6)
