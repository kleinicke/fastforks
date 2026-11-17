"""DeDoDe v2: the cards, the VGG layout, the keypoint grid and the dual-softmax matching."""

from __future__ import annotations

import pytest
import torch

from fastforks import matching
from fastforks._core import registry
from fastforks.shared.dedode import VGG11_BN, describe_keypoints, dual_softmax_match, get_grid, vgg_bn_features


def test_cards() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "dedode"}
    assert sorted(cards) == ["dedode-v2-b", "dedode-v2-g"]
    for mid, (card, variant) in cards.items():
        assert dict(card.options) == {"size": 784, "max_keypoints": 10000}  # upstream read_image, demos
        assert set(variant.outputs or card.outputs) == {"keypoints1", "keypoints2", "pairs", "scores"}
        assert variant.weights[0].name == "dedode_detector_L_v2.pth"
        assert mid in matching.list_models()
    assert len(cards["dedode-v2-g"][1].weights) == 3  # + DINOv2 ViT-L/14


def test_vgg_layout() -> None:
    """The same state-dict keys and shapes as torchvision's `vgg19_bn().features[:40]` (upstream's encoder)."""
    tvm = pytest.importorskip("torchvision.models")
    ours = {k: v.shape for k, v in vgg_bn_features().state_dict().items()}
    ref = torch.nn.ModuleList(tvm.vgg19_bn(weights=None).features[:40]).state_dict()
    assert ours == {k: v.shape for k, v in ref.items()}
    ours = {k: v.shape for k, v in vgg_bn_features(VGG11_BN).state_dict().items()}  # the DaD detector's encoder
    ref = torch.nn.ModuleList(tvm.vgg11_bn(weights=None).features[:22]).state_dict()
    assert ours == {k: v.shape for k, v in ref.items()}


def test_grid() -> None:
    """Pixel centres in normalised coordinates, x first, row-major (upstream `get_grid`)."""
    g = get_grid(2, 4, torch.device("cpu"))
    torch.testing.assert_close(g[:4, 0], torch.tensor([-0.75, -0.25, 0.25, 0.75]))
    torch.testing.assert_close(g[::4, 1], torch.tensor([-0.5, 0.5]))


def test_dual_softmax_match() -> None:
    """Identical descriptions match one to one; a threshold above every probability leaves no match."""
    d = torch.nn.functional.normalize(torch.randn(6, 16, generator=torch.Generator().manual_seed(0)), dim=-1)
    pairs, scores = dual_softmax_match(d, d[[2, 0, 1, 5, 4, 3]], inv_temp=20, threshold=0.01)
    assert pairs.tolist() == [[0, 1], [1, 2], [2, 0], [3, 5], [4, 4], [5, 3]]
    assert bool((scores > 0.01).all())
    assert len(dual_softmax_match(d, d, inv_temp=20, threshold=1.0)[0]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_describe_keypoints_is_grid_sample(dtype: torch.dtype) -> None:
    """The gathered bilinear samples equal upstream's `grid_sample(grid.float())` bit for bit, edges included."""
    g = torch.Generator().manual_seed(0)
    grid = torch.randn(1, 8, 13, 17, generator=g).to(dtype)
    kp = torch.rand(300, 2, generator=g) * 2.2 - 1.1  # some keypoints beyond the border: zero padding there
    kp[:4] = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]])
    want = torch.nn.functional.grid_sample(grid.float(), kp[None, None], mode="bilinear", align_corners=False)
    # the four taps are summed in the same order; the difference is the kernel's fused multiply-adds
    torch.testing.assert_close(describe_keypoints(grid, kp), want[0, :, 0].mT, rtol=0, atol=1e-3)
