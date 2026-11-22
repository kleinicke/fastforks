"""LightGlue sparse matchers: cards, the sparse result's matched points, the long-side resize, the packed forward."""

from __future__ import annotations

import pytest
import torch

import fastforks
from fastforks import matching
from fastforks._core import registry
from fastforks._core.errors import WeightsError
from fastforks.matching import SparseMatchingResult
from fastforks.matching.models.lightglue.model import Matcher, Params
from fastforks.shared.lightglue import LightGlue, match_many, resize_long


def test_cards() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "lightglue"}
    assert sorted(cards) == ["aliked-lightglue", "disk-lightglue"]
    for mid, (card, variant) in cards.items():
        assert dict(card.options) == {
            "max_keypoints": 2048,
            "long_side": 1024,
            "cuda_graphs": True,
        }  # upstream's match_pair
        assert set(variant.outputs or card.outputs) == {"keypoints1", "keypoints2", "pairs", "scores"}
        assert mid in matching.list_models()


def test_matches_index_the_keypoints() -> None:
    k1, k2 = torch.arange(10.0).reshape(5, 2), -torch.arange(6.0).reshape(3, 2)
    r = SparseMatchingResult(k1, k2, torch.tensor([[4, 0], [1, 2]]), torch.tensor([0.9, 0.2]), "m", None)  # type: ignore[arg-type]
    p1, p2 = r.matches()
    assert torch.equal(p1, k1[[4, 1]])
    assert torch.equal(p2, k2[[0, 2]])
    p1, p2 = r.matches(score_threshold=0.5)
    assert torch.equal(p1, k1[[4]])
    assert torch.equal(p2, k2[[0]])


def test_resize_long_keeps_the_aspect() -> None:
    x = torch.rand(1, 3, 48, 64)
    assert resize_long(x, 32).shape[-2:] == (24, 32)
    assert resize_long(x.transpose(-1, -2), 32).shape[-2:] == (32, 24)


def _features(gen: torch.Generator, n: int, size: tuple[int, int]) -> dict[str, torch.Tensor]:
    w, h = size
    return {
        "keypoints": torch.rand(1, n, 2, generator=gen) * torch.tensor([w, h], dtype=torch.float32),
        "descriptors": torch.nn.functional.normalize(torch.randn(1, n, 128, generator=gen), dim=-1),
        "image_size": torch.tensor([[w, h]], dtype=torch.float32),
    }


def test_match_many_equals_the_per_pair_forward() -> None:
    """The packed forward of many pairs gives every pair the matches of upstream's forward, with its own adaptive
    depth (stop layer) and width (pruning)."""
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    net = LightGlue(features="aliked").eval()
    # random weights: make the stop and pruning rules fire at different layers for different pairs
    net.confidence_thresholds = torch.linspace(0.42, 0.56, net.conf.n_layers)
    net.conf.width_confidence = 0.5
    net.conf.filter_threshold = 0.0  # random weights give tiny scores: every mutual nearest neighbour counts
    net.pruning_keypoint_thresholds = {**net.pruning_keypoint_thresholds, "cpu": 8}
    sizes = [(64, 48), (80, 40), (32, 32), (48, 64), (64, 64)]
    feats = [(_features(gen, 10 + 7 * k, sizes[k]), _features(gen, 30 - 5 * k, sizes[-1 - k])) for k in range(5)]
    feats.append((_features(gen, 0, sizes[0]), feats[0][1]))  # no keypoints on one side
    with torch.inference_mode():
        outs = match_many(net, feats)
        singles = [net({"image0": f0, "image1": f1}) for f0, f1 in feats]
    stops = {s["stop"] for s in singles}
    assert len(stops) > 1, stops  # several stop layers in the batch
    assert any(int((s["prune0"] > 1).sum()) for s in singles)  # pruning happened
    assert sum(len(s["matches"][0]) for s in singles) >= 10  # enough matches for the comparison to mean something
    for out, single in zip(outs, singles, strict=True):
        assert torch.equal(out["matches"], single["matches"][0])
        assert torch.allclose(out["scores"], single["scores"][0], atol=1e-5)
    assert outs[-1]["matches"].shape == (0, 2)


@pytest.mark.parametrize("extractor", ["aliked", "disk"])
def test_float16_extractor_keeps_float32_outputs(extractor) -> None:
    """`dtype="float16"` autocasts only the dense network: keypoints and descriptors come out float32 (the matcher's
    and the API's dtype) and are the float32 run's up to half precision."""
    torch.manual_seed(0)
    net = Matcher(Params(extractor, 0.0, 64, 0, cuda_graphs=False)).eval()
    image = torch.rand(1, 3, 64, 96)
    with torch.inference_mode():
        ref = net.extract(image)
        net.extractor.fp16 = True
        out = net.extract(image)  # CPU autocast to float16: the same code path as CUDA
    assert out["keypoints"].dtype == out["descriptors"].dtype == torch.float32
    assert out["keypoints"].shape[1] > 0
    assert ref["descriptors"].shape == out["descriptors"].shape
    # the same detections up to the half-precision maps (random weights: near-equal scores change the score order)
    d = torch.cdist(out["keypoints"][0], ref["keypoints"][0]).min(1).values
    assert (d < 1.0).float().mean() > 0.9, d


@pytest.mark.gpu
def test_aliked_kernels_graphs_match_eager() -> None:
    """ALIKED's graphed dense maps: matches and `describe` bit-equal to eager at two sizes; release frees them."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        graphed = fastforks.load("aliked-lightglue", mode="kernels", device="cuda")
        eager = fastforks.load("aliked-lightglue", mode="kernels", device="cuda", cuda_graphs=False)
    except WeightsError as e:
        pytest.skip(f"aliked-lightglue weights not available: {e}")
    assert eager._runner.graphs is None
    gen = torch.Generator().manual_seed(0)
    images = [torch.rand(1, 3, h, w, generator=gen).cuda() for h, w in ((240, 320), (320, 240), (300, 400))]
    keys = ("keypoints1", "keypoints2", "pairs", "scores")
    with torch.inference_mode():
        for i, j in [(0, 1)] * 4 + [(2, 0)] * 3 + [(1, 0)]:
            got, want = graphed._runner(images[i], images[j]), eager._runner(images[i], images[j])
            for k in keys:
                torch.testing.assert_close(got[k], want[k], rtol=0, atol=0, msg=k)
        got, want = graphed._runner.describe(images[2]), eager._runner.describe(images[2])
        for k, v in want.items():
            torch.testing.assert_close(got[k], v, rtol=0, atol=0, msg=k)
    graphs = graphed._runner.graphs
    assert len(graphs.graphs) == 2, f"want one graph per long-side size (failed: {list(graphs._failed)})"
    graphed.release_graphs()
    assert not graphs.graphs
