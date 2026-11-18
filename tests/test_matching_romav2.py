"""RoMa v2: the card, the conversion of upstream's normalised warp to image2 pixels, the bf16 regions."""

from __future__ import annotations

import pytest
import torch

from fastforks import matching
from fastforks._core import registry
from fastforks.kernels import romav2 as lib
from fastforks.matching.models.romav2 import model as family
from fastforks.matching.models.romav2.net.geometry import get_normalized_grid
from fastforks.matching.models.romav2.net.local_correlation import local_correlation


def test_card() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "romav2"}
    assert sorted(cards) == ["romav2"]
    card, _ = cards["romav2"]
    assert dict(card.options) == {
        "coarse_size": 800,
        "fine_size": 1280,
        "keep_descriptors": 0,
    }  # upstream "precise"
    assert "romav2" in matching.list_models()


class _Identity(torch.nn.Module):
    """Upstream's output layout with the identity warp (every pixel to the same normalised place) at the last size."""

    def forward(self, a_lr: torch.Tensor, _b: torch.Tensor, a_hr: torch.Tensor | None, _bh: object) -> dict:
        b, _, h, w = (a_lr if a_hr is None else a_hr).shape
        return {"warp_AB": get_normalized_grid(b, h, w), "confidence_AB": torch.zeros(b, h, w, 4)}


@pytest.mark.parametrize(("coarse", "fine"), [(32, 0), (16, 32)])
def test_warp_in_image2_pixels(coarse: int, fine: int) -> None:
    """Pixel centres of image1 (32 x 32) land on the same relative place of image2 (48 x 64)."""
    out = family.forward(_Identity(), torch.rand(2, 3, 32, 32), torch.rand(2, 3, 48, 64), family.Params(coarse, fine))
    assert out["warp"].shape == (2, 2, 32, 32)
    assert out["certainty"].shape == (2, 32, 32)
    torch.testing.assert_close(out["certainty"], torch.full((2, 32, 32), 0.5))  # sigmoid of the overlap logit
    ys, xs = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
    want = torch.stack(((xs + 0.5) * 64 / 32 - 0.5, (ys + 0.5) * 48 / 32 - 0.5))
    torch.testing.assert_close(out["warp"][0], want, rtol=0, atol=1e-4)


class _Net(_Identity):
    """`describe` = the image + 1, recorded; `forward` checks the descriptors it gets and spoils its lists."""

    def __init__(self) -> None:
        super().__init__()
        self.described: list[float] = []

    def describe(self, img: torch.Tensor) -> list[torch.Tensor]:
        self.described.append(round(float(img.flatten()[0]), 3))
        return [img + 1]

    def forward(self, a_lr, b_lr, *hr, f_A, f_B):  # noqa: N803 - upstream's names
        assert torch.equal(f_A[0], a_lr + 1)
        assert torch.equal(f_B[0], b_lr + 1)
        f_A[-1] = f_A[-1] * 0  # upstream's matcher replaces the last entry of its list
        return super().forward(a_lr, b_lr, *hr)


def test_keep_descriptors_reuses_the_last_images() -> None:
    """`Descriptors`: a kept image is not described again, in either role; the least recently used one is dropped."""
    a, b, c, d = (torch.full((1, 3, 16, 16), v) for v in (0.1, 0.2, 0.3, 0.4))
    net, params = _Net(), family.Params(16, 0, keep_descriptors=2)
    kept = family.Descriptors(net, 2)
    for x, y in ((a, b), (b, c), (b, a), (a.clone(), a)):
        family.forward(net, x, y, params, kept)
    assert net.described == [0.1, 0.2, 0.3, 0.1]
    assert len(kept.kept) == 2
    net, kept = _Net(), family.Descriptors(_Net(), 3)
    kept.net = net
    for x, y in ((a, b), (c, a), (d, a), (b, c)):  # d drops b (a was used after it), then b drops c
        family.forward(net, x, y, params, kept)
    assert net.described == [0.1, 0.2, 0.3, 0.4, 0.2, 0.3]
    assert len(kept.kept) == 3


def test_sizes_must_be_multiples() -> None:
    with pytest.raises(ValueError, match="multiples of 16"):
        matching.load("romav2", device="cpu", coarse_size=100)


@pytest.mark.gpu
@pytest.mark.parametrize(("c", "radius", "height", "width", "batch"), [(192, 3, 17, 21, 2), (48, 1, 80, 96, 1)])
def test_kernel_local_correlation_matches_portable(c: int, radius: int, height: int, width: int, batch: int) -> None:
    """The one-pass kernel equals upstream's grid_sample window (fp32), also for windows over and beyond the border."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    f_a, f_b = (torch.randn(batch, height, width, c, device="cuda") for _ in range(2))
    shift = torch.randn(batch, height, width, 2, device="cuda") / 2  # many windows over the border
    warp = get_normalized_grid(batch, height, width, device="cuda") + shift
    warp[0, 0, 0] = torch.tensor([1e9, -1e9])  # far outside: all zero
    want = local_correlation(f_a.permute(0, 3, 1, 2), f_b.permute(0, 3, 1, 2), radius, warp)
    got = family._kernel_correlate(lib.load(torch.device("cuda")), f_a, f_b, warp, radius)
    assert got.shape == want.shape
    # the reference rounds warp + offset in normalised units (about 1e-7 * width / 2 px): up to 5e-5 here at width 96
    torch.testing.assert_close(got, want, rtol=0, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize(("c", "height", "width", "batch"), [(512, 13, 22, 2), (128, 9, 7, 1), (32, 6, 33, 1)])
def test_kernel_depthwise_matches_cudnn(c: int, height: int, width: int, batch: int) -> None:
    """The fused depthwise 5x5 + bias + ReLU equals cuDNN's bf16 conv, bias add and ReLU (borders, ragged widths)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    x = torch.randn(batch, c, height, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(c, 1, 5, 5, device="cuda", dtype=torch.bfloat16) / 5
    bias = torch.randn(c, device="cuda", dtype=torch.bfloat16)
    want = torch.nn.functional.conv2d(x.to(memory_format=torch.channels_last), weight, bias, padding=2, groups=c).relu()
    wt = weight.reshape(c, 25).t().contiguous()
    for layout in (torch.contiguous_format, torch.channels_last):  # the first block's input is NCHW
        got = family._kernel_depthwise(lib.load(torch.device("cuda")), wt, bias, x.contiguous(memory_format=layout))
        assert got.is_contiguous(memory_format=torch.channels_last)
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize(
    "backend", [pytest.param("mlx", marks=pytest.mark.mlx), pytest.param("mps", marks=pytest.mark.mps)]
)
@pytest.mark.parametrize("radius", [1, 3])
def test_metal_correlation_at_image_boundaries(backend: str, radius: int) -> None:
    """Fractional and off-image windows on two independent rectangular images match the reference."""
    gen = torch.Generator().manual_seed(7)
    a, b = (torch.randn(2, 5, 7, 37, generator=gen) for _ in range(2))
    warp = torch.rand(2, 5, 7, 2, generator=gen) * 3 - 1.5
    warp[0, 0, 0] = torch.tensor([1.0e9, -1.0e9])
    want = local_correlation(a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2), radius, warp)
    if backend == "mlx":
        if not pytest.importorskip("mlx.core").metal.is_available():
            pytest.skip("needs Apple Metal")
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - optional MLX
        from fastforks.kernels.roma_mlx.ops import correlation  # noqa: PLC0415

        got = to_torch(correlation(to_mx(a), to_mx(b), to_mx(warp), radius))
    else:
        if not torch.backends.mps.is_available():
            pytest.skip("needs PyTorch MPS")
        from fastforks.kernels import roma_metal  # noqa: PLC0415 - device-specific library

        lib = roma_metal.load(torch.device("mps"))
        got = roma_metal.correlate(lib, a.to("mps"), b.to("mps"), warp.to("mps"), radius).permute(0, 2, 3, 1).cpu()
    torch.testing.assert_close(got.permute(0, 3, 1, 2), want, atol=1e-5, rtol=1e-5)
