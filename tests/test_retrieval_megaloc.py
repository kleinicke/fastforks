"""MegaLoc: the card, and the task's exact search helpers (`top_k`, `pairs`)."""

from __future__ import annotations

import pytest
import torch

from fastforks import retrieval
from fastforks._core import registry
from fastforks._core.errors import InputError
from fastforks.retrieval.models.megaloc.net.megaloc import get_matching_probs


def test_card() -> None:
    card, variant = registry.cards("retrieval")["megaloc"]
    assert variant.outputs == ("descriptor",)
    assert card.params["dim"] == 8448
    assert card.licence.code == card.licence.weights == "MIT"
    assert "megaloc" in retrieval.list_models()


def test_top_k() -> None:
    torch.manual_seed(0)
    db = torch.nn.functional.normalize(torch.randn(50, 16), dim=1)
    q = db[[3, 7]] + 0.01 * torch.randn(2, 16)
    scores, index = retrieval.top_k(q, db, k=4)
    assert index.dtype == torch.int64
    assert scores.shape == index.shape == (2, 4)
    assert index[:, 0].tolist() == [3, 7]
    want = (q @ db.T).topk(4, dim=1)
    torch.testing.assert_close(scores, want.values)
    torch.testing.assert_close(retrieval.top_k(q[0], db, k=1)[1], torch.tensor([[3]]))
    _, own = retrieval.top_k(db, None, k=49)  # self-retrieval: never the query itself
    assert not (own == torch.arange(50)[:, None]).any()
    with pytest.raises(InputError, match="k must be"):
        retrieval.top_k(db, None, k=50)
    with pytest.raises(InputError, match="do not match"):
        retrieval.top_k(q, db[:, :8])
    with pytest.raises(InputError, match="float tensor"):
        retrieval.top_k(torch.zeros(2, 3, 4), db)


def test_pairs() -> None:
    d = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]]), dim=1)
    assert retrieval.pairs(d, k=1).tolist() == [[0, 1], [2, 3]]
    p = retrieval.pairs(d, k=3)
    assert p.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]


def test_sinkhorn_rows_sum_to_one() -> None:
    """The vendored optimal transport: each feature's assignment over the clusters + dustbin sums to one."""
    torch.manual_seed(0)
    p = get_matching_probs(torch.randn(2, 8, 30), torch.tensor(1.0), 3).exp()
    torch.testing.assert_close(p.sum(1), torch.ones(2, 30), atol=1e-4, rtol=0)
