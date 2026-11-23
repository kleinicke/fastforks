"""MemFlow: the card's variants, the network on random weights (batch, odd sizes, export), the kernel lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import flow
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.flow.models.memflow.model import (
    _fft_depthwise,
    _fft_fits,
    _fft_size,
    _kernel_lookup,
    _pcblock,
    _pointwise,
    _pointwise_fits,
)
from fastforks.flow.models.memflow.net import CorrBlock, MemFlowNet, PCBlock

if TYPE_CHECKING:
    from pathlib import Path

CT = ["flyingchairs", "sceneflow"]
CTSKH = [*CT, "sintel-flow", "kitti", "hd1k"]
VARIANTS = {"memflow": (6750.0, CT), "memflow-sintel": (6624.0, CTSKH), "memflow-kitti": (6480.0, CTSKH)}


def _pair(height: int, width: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """A textured pair in [0, 1] on the CPU; item `i` of a batch moves by `4 * (i + 1)` px."""
    pairs = [synthetic_pair(height, width, torch.device("cpu"), shift=4 * (i + 1)) for i in range(batch)]
    return torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("flow").values() if c.family == "memflow"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, (length, data) in VARIANTS.items():
        card, variant = cards[vid]
        assert list(variant.licence.get("training_data", card.licence.training_data)) == data, vid
        assert card.options["iters"] == 15, vid
        assert {**card.params, **variant.params}["train_avg_length"] == length, vid
    assert set(flow.list_models()) >= set(VARIANTS)


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint with random weights in the authors' layout (`{"model": ...}` with the unused position embedding)."""
    torch.manual_seed(0)
    sd = {k: v.contiguous() for k, v in MemFlowNet(6750.0).state_dict().items()}
    sd["att.pos_emb.rel_height.weight"] = torch.zeros(1)  # dropped by `convert`
    path = tmp_path_factory.mktemp("memflow") / "random.pth"
    torch.save({"model": sd}, path)
    return path


@pytest.mark.slow
def test_batch_odd_size_and_export(weights: Path) -> None:
    model = flow.load("memflow", weights=weights, device="cpu", iters=2)
    a, b = _pair(131, 141, batch=2)  # above the 128 px minimum, not multiples of 8: replicate padding, cropped again
    out = model(a, b).flow
    assert out.shape == (2, 2, 131, 141)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[1:], model(a[1:], b[1:]).flow, rtol=0, atol=1e-4)  # the items do not mix
    a, b = _pair(128, 160)
    with torch.inference_mode():
        exported = model.module()(a, b)  # export does what the call does
    torch.testing.assert_close(exported, model(a, b).flow, rtol=0, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize(("height", "width", "batch"), [(17, 21, 2), (135, 240, 1)])  # odd cells; the 1080p cells
def test_kernel_lookup_matches_portable(height: int, width: int, batch: int) -> None:
    """The on-the-fly windows equal upstream's volume lookup (fp32) up to fp16 operands and output rounding."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from fastforks.kernels import raft_flow  # noqa: PLC0415

    torch.manual_seed(0)
    f1, f2 = (torch.randn(batch, 256, height, width, device="cuda").half() for _ in range(2))
    coords = torch.randn(batch, 2, height, width, device="cuda") * 20  # windows far off and over the border
    coords += torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")).cuda()
    want = CorrBlock(f1.float(), f2.float())(coords)
    got = _kernel_lookup(raft_flow.load(torch.device("cuda")), f1, f2)(coords)
    assert got.shape == want.shape
    torch.testing.assert_close(got.float(), want, rtol=0, atol=4e-3)  # fp16 pooling and output of |corr| < 8


@pytest.mark.parametrize(("channels", "size"), [(3, (7, 9)), (70, (12, 5))])  # 70: two channel chunks
def test_fft_depthwise_is_the_conv(channels: int, size: tuple[int, int]) -> None:
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(channels, channels, 15, padding=7, groups=channels).double()
    x = torch.randn(2, channels, *size, dtype=torch.float64)
    assert _fft_fits(conv)
    got = _fft_depthwise(conv, x)
    torch.testing.assert_close(got, conv(x), rtol=0, atol=1e-12)
    assert _fft_depthwise(conv, x[:, :, 1:]).shape == x[:, :, 1:].shape  # another size: new spectra
    assert not _fft_fits(torch.nn.Conv2d(4, 4, 15, padding=7))  # dense
    assert not _fft_fits(torch.nn.Conv2d(4, 4, 7, padding=3, groups=4))  # small


def test_pointwise_is_the_conv() -> None:
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(6, 5, 1).double()
    x = torch.randn(2, 6, 7, 9, dtype=torch.float64)
    assert _pointwise_fits(conv)
    torch.testing.assert_close(_pointwise(conv, x), conv(x), rtol=0, atol=1e-12)
    torch.testing.assert_close(_pointwise(conv, x[..., ::2]), conv(x[..., ::2]), rtol=0, atol=1e-12)  # strided input
    assert not _pointwise_fits(torch.nn.Conv2d(4, 4, 1, stride=2))
    assert not _pointwise_fits(torch.nn.Conv2d(4, 4, 1, groups=4))
    assert not _pointwise_fits(torch.nn.Conv2d(4, 4, 3, padding=1))


@pytest.mark.parametrize("kernels", [(1, 7), (1, 15)])
def test_pcblock_fused_is_the_block(kernels: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    block = PCBlock(6, 5, kernels).double()
    x = torch.randn(2, 6, 9, 11, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(_pcblock(block, x), block(x), rtol=0, atol=1e-12)


def test_fft_size_has_only_small_factors() -> None:
    assert [_fft_size(n) for n in (149, 254, 256, 7)] == [150, 256, 256, 8]
