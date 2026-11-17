"""FlexGlue: the card, the entmax window readout, the mutual NN, the checkpoint split and the net's shapes."""

from __future__ import annotations

import torch

from fastforks import matching
from fastforks._core import registry
from fastforks.matching.models.flexglue.model import HEAD, split_checkpoint
from fastforks.matching.models.flexglue.net.flexglue import (
    FlexGlueNet,
    cell_centers,
    entmax15_window_readout,
    mutual_nn,
    sq_dist_gemm,
)


def test_card() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "flexglue"}
    assert sorted(cards) == ["aliked-flexglue"]
    card, variant = cards["aliked-flexglue"]
    assert dict(card.options) == {"max_keypoints": 2048, "long_side": 1024, "size": 512}  # upstream README, conf
    assert dict(card.params) == {"threshold": 0.1, "max_distance_px": 3.0}  # upstream default_conf
    assert set(card.outputs) == {"keypoints1", "keypoints2", "pairs", "scores"}
    assert [w.name for w in variant.weights] == ["flexglue.pth", "aliked-n16.pth"]
    assert "aliked-flexglue" in matching.list_models()


def test_entmax_window_readout() -> None:
    """A dominant cell: its centre with mass 1; two equal neighbours: their midpoint, the mass split evenly."""
    h, w = 4, 5
    grid = cell_centers(h, w, scale=(w, h), device=torch.device("cpu"))
    logits = torch.full((1, 2, h * w), -10.0)
    logits[0, 0, 7] = 10.0  # (y, x) = (1, 2)
    logits[0, 1, 12] = logits[0, 1, 13] = 10.0  # (2, 2) and (2, 3)
    for k in (4, 512):  # the top-k and the full-sort branch
        pos, mass = entmax15_window_readout(logits, grid, (h, w), k=k)
        torch.testing.assert_close(pos[0, 0], grid[7])
        torch.testing.assert_close(pos[0, 1], (grid[12] + grid[13]) / 2)
        torch.testing.assert_close(mass, torch.ones(1, 2))


def test_mutual_nn() -> None:
    """GEMM squared distances match cdist; non-mutual nearest neighbours are -1."""
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(1, 6, 4, generator=g), torch.randn(1, 9, 4, generator=g)
    torch.testing.assert_close(sq_dist_gemm(x, y), torch.cdist(x, y).square(), atol=1e-4, rtol=1e-4)
    d = torch.tensor([[[0.0, 5.0], [1.0, 6.0], [9.0, 2.0]]])  # rows 0 and 1 both nearest col 0
    m0, m1 = mutual_nn(d)
    assert m0.tolist() == [[0, -1, 1]]
    assert m1.tolist() == [[0, 2]]


def test_split_checkpoint_and_shapes() -> None:
    """Upstream's layout loads strictly (last head as `head.`, unused heads skipped); the head gives points in j."""
    net = FlexGlueNet(input_dim=16, dim=32, search_dim=8, n_layers=2, heads=2)
    sd = {f"matcher.{HEAD.removeprefix('matcher.')}{k[5:]}" if k.startswith("head.") else f"matcher.{k}": v
          for k, v in net.state_dict().items()}  # fmt: skip
    sd["matcher.heads.0.to_search.weight"] = torch.zeros(1)
    sd[f"{HEAD}disambiguate.weight"] = torch.zeros(1)
    sd["extractor.backbone.stages.0.0.gamma"] = torch.zeros(1)
    sd["extractor.descriptor_head.weight"] = torch.zeros(1)
    parts = split_checkpoint(sd)
    assert list(parts["descriptor"]) == ["backbone.stages.0.0.gamma"]
    assert set(parts["matcher"]) == set(net.state_dict())
    net.load_state_dict(parts["matcher"], strict=True)
    g = torch.Generator().manual_seed(0)
    f0, f1 = torch.randn(1, 16, 3, 4, generator=g), torch.randn(1, 16, 5, 2, generator=g)
    cf0, cf1 = net.decode(f0, f1, None)
    assert cf0.shape == (1, 32, 3, 4)
    assert cf1.shape == (1, 32, 5, 2)
    p2d, conf = net.head(torch.rand(1, 7, 2, generator=g) * 2 - 1, cf0, cf1)
    assert p2d.shape == (1, 7, 2)
    assert conf.shape == (1, 7)
    assert ((conf >= 0) & (conf <= 1)).all()
