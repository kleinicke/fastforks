"""XFeat sparse matchers: card metadata, empty descriptor sets and the MLX runner against the PyTorch module."""

from __future__ import annotations

import pytest
import torch

from fastforks import matching
from fastforks._core import registry
from fastforks.matching.models.xfeat.model import LIGHTERGLUE, Matcher, Params, _MlxPortable
from fastforks.matching.models.xfeat.net.xfeat import XFeat
from fastforks.shared.lightglue.net.lightglue import LightGlue


def test_cards() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "xfeat"}
    assert sorted(cards) == ["xfeat", "xfeat-lighterglue"]
    for mid, (card, variant) in cards.items():
        assert dict(card.options) == {"max_keypoints": 4096, "long_side": 0}  # upstream's top_k, no resize
        assert set(variant.outputs or card.outputs) == {"keypoints1", "keypoints2", "pairs", "scores"}
        assert mid in matching.list_models()


def test_match_without_keypoints() -> None:
    model = XFeat()
    empty = torch.empty(0, 64)
    populated = torch.ones(2, 64)
    for left, right in ((empty, empty), (empty, populated), (populated, empty)):
        idx0, idx1 = model.match(left, right)
        for indices in (idx0, idx1):
            assert indices.shape == (0,)
            assert indices.dtype == torch.long
            assert indices.device == left.device


def _random_matcher(matcher: str, seed: int) -> Matcher:
    """A `Matcher` of random weights and batch norm statistics that finds a few hundred keypoints."""
    torch.manual_seed(seed)
    net = Matcher(Params(matcher, detection_threshold=0.002, max_keypoints=300, long_side=0)).eval()
    gen = torch.Generator().manual_seed(seed)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            assert m.running_mean is not None
            assert m.running_var is not None
            m.running_mean.copy_(torch.randn(m.num_features, generator=gen) * 0.1)
            m.running_var.copy_(torch.rand(m.num_features, generator=gen) + 0.5)
    return net


def _mlx_runner(net: Matcher) -> _MlxPortable:
    """`_MlxPortable` around `net` (no weight files)."""
    from fastforks.matching.models.xfeat import mlx_net  # noqa: PLC0415 - optional MLX
    from fastforks.shared.lightglue.mlx import MlxLightGlue  # noqa: PLC0415

    runner = _MlxPortable.__new__(_MlxPortable)
    runner.net = net
    runner._dense = mlx_net.MlxDense(net.xfeat.net, net.params.detection_threshold)
    runner._glue = None if net.matcher is None else MlxLightGlue(net.matcher)
    return runner


def _by_position(kpts: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """`rows` in the order of their keypoints sorted by `(x, y)` (equal scores may be ranked differently)."""
    order = sorted(range(len(kpts)), key=lambda i: tuple(kpts[i].tolist()))
    return rows[order]


@pytest.mark.mlx
@pytest.mark.parametrize("size", [(96, 128), (110, 150)])  # the second is resized to multiples of 32
def test_mlx_features_match_module(size: tuple[int, int]) -> None:
    """The hand-written MLX network and keypoint selection: `detectAndCompute`'s keypoints and descriptors."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_torch  # noqa: PLC0415 - optional MLX

    net = _random_matcher("mnn", seed=3)
    runner = _mlx_runner(net)
    image = torch.rand(1, 3, *size, generator=torch.Generator().manual_seed(5))
    with torch.inference_mode():
        want = net.extract(image)
        kpts, desc = runner._keypoints(*runner._dense_maps(image))
    kpts, desc = torch.from_numpy(kpts), to_torch(desc).cpu()
    assert len(want["keypoints"]) > 100
    torch.testing.assert_close(_by_position(kpts, kpts), _by_position(want["keypoints"], want["keypoints"]))
    # random weights give nearly parallel descriptors (cosine > 0.95): compare them with their mean removed
    got_desc, want_desc = _by_position(kpts, desc), _by_position(want["keypoints"], want["descriptors"])
    mean = want_desc.mean(dim=0)
    assert float((want_desc - mean).abs().max()) > 0.01
    torch.testing.assert_close(got_desc - mean, want_desc - mean, atol=2e-5, rtol=0)


@pytest.mark.mlx
def test_mlx_mutual_nearest_matches_module() -> None:
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx  # noqa: PLC0415 - optional MLX
    from fastforks.matching.models.xfeat import mlx_post  # noqa: PLC0415

    gen = torch.Generator().manual_seed(11)
    desc1, desc2 = (torch.nn.functional.normalize(torch.randn(n, 64, generator=gen), dim=-1) for n in (300, 280))
    idx0, idx1 = XFeat().match(desc1, desc2, min_cossim=-1)
    pairs, scores = mlx_post.mutual_nearest(to_mx(desc1), to_mx(desc2))
    assert len(idx0) > 50
    torch.testing.assert_close(torch.from_numpy(pairs), torch.stack([idx0, idx1], dim=-1))
    torch.testing.assert_close(torch.from_numpy(scores), (desc1[idx0] * desc2[idx1]).sum(-1), atol=1e-5, rtol=0)


@pytest.mark.mlx
@pytest.mark.parametrize(("seed", "width", "depth"), [(0, 0.95, -1), (0, 0.5, -1), (2, 0.5, 0.2)])
def test_mlx_lighterglue_matches_module(seed: int, width: float, depth: float) -> None:
    """`MlxLightGlue` against the vendored LightGlue (CPU): upstream's setting, point pruning, adaptive depth."""
    if not pytest.importorskip("mlx.core").metal.is_available():
        pytest.skip("needs Apple Metal")
    from fastforks._core.mlx import to_mx  # noqa: PLC0415 - optional MLX
    from fastforks.shared.lightglue.mlx import MlxLightGlue  # noqa: PLC0415

    torch.manual_seed(seed)
    net = LightGlue(features=None, **(LIGHTERGLUE | {"width_confidence": width, "depth_confidence": depth})).eval()
    net.conf.filter_threshold = 0.0  # random weights: every mutual pair counts
    # 300 keypoints with random descriptors and 260 of them moved by a pixel, with noisy descriptors
    gen = torch.Generator().manual_seed(seed)
    size = torch.tensor([640, 480])
    k1 = torch.rand(300, 2, generator=gen) * size
    d1 = torch.nn.functional.normalize(torch.randn(300, 64, generator=gen), dim=-1)
    kept = torch.randperm(300, generator=gen)[:260]
    k2 = k1[kept] + torch.randn(260, 2, generator=gen)
    d2 = torch.nn.functional.normalize(d1[kept] + 0.05 * torch.randn(260, 64, generator=gen), dim=-1)
    with torch.inference_mode():
        want = net(
            {
                "image0": {"keypoints": k1[None], "descriptors": d1[None], "image_size": size[None]},
                "image1": {"keypoints": k2[None], "descriptors": d2[None], "image_size": size[None]},
            }
        )
    pairs, scores = MlxLightGlue(net)((k1.numpy(), to_mx(d1), size.numpy()), (k2.numpy(), to_mx(d2), size.numpy()))
    assert len(want["matches"][0]) >= 30
    assert (width == 0.95) == (int(want["prune0"].min()) == net.conf.n_layers)  # the smaller width prunes points
    torch.testing.assert_close(torch.from_numpy(pairs), want["matches"][0])
    torch.testing.assert_close(torch.from_numpy(scores), want["scores"][0], rtol=1e-3, atol=0)
