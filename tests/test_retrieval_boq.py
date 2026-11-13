"""BoQ: the card, and the precomputed queries equal upstream's per-call query branch."""

from __future__ import annotations

import torch

from fastforks._core import registry
from fastforks.retrieval.models.boq.model import _aggregate, _queries
from fastforks.retrieval.models.boq.net.boq import BoQ


def test_card() -> None:
    card, variant = registry.cards("retrieval")["boq-dinov2"]
    assert variant.outputs == ("descriptor",)
    assert card.params["dim"] == 12288
    assert card.licence.code == card.licence.weights == "MIT"


def test_aggregate_matches_upstream() -> None:
    """`_aggregate` with the queries computed once = upstream `BoQ.forward` (small random aggregator)."""
    torch.manual_seed(0)
    agg = BoQ(in_channels=32, proj_channels=64, num_queries=8, num_layers=2, row_dim=4).eval()
    x = torch.randn(2, 32, 5, 6)
    with torch.no_grad():
        want = agg(x)[0]
        got = _aggregate(agg, _queries(agg), x)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=0)
