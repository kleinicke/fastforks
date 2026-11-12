"""BEN2: the card and the vendored network's fastforks changes (einops-free rearranges, attention without weights)."""

from __future__ import annotations

import pytest
import torch

from fastforks import matting
from fastforks._core import registry
from fastforks.matting.models.ben2.net import ben2


def test_card() -> None:
    card, variant = registry.cards("matting")["ben2"]
    assert variant.outputs == ("alpha",)
    assert card.licence.code == card.licence.weights == "MIT"
    assert "ben2" in matting.list_models()


def test_patches() -> None:
    """`b c (hg h) (wg w) <-> (hg wg b) c h w` (hg = wg = 2) as einops: grid row, grid column, then batch."""
    x = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
    got = ben2.image2patches(x)
    want = torch.cat([x[:, :, i * 2 : i * 2 + 2, j * 3 : j * 3 + 3] for i in range(2) for j in range(2)])
    assert torch.equal(got, want)
    assert torch.equal(ben2.patches2image(got), x)


@pytest.mark.slow
def test_attention_without_weights() -> None:
    """MCLM / MCRM with `need_weights=False` = upstream's default (weights computed, then dropped)."""
    torch.manual_seed(0)
    for m, args in ((ben2.MCLM(16, 1, [1, 4, 8]), (torch.randn(4, 16, 8, 8), torch.randn(1, 16, 8, 8))),
                    (ben2.MCRM(16, 1, [2, 4, 8]), (torch.randn(5, 16, 8, 8),))):  # fmt: skip
        m.eval()
        got = m(*args)
        calls: list[int] = []
        for a in m.attention:
            a.forward = lambda *x, _o=a.forward, _c=calls, **k: _c.append(1) or _o(*x, **{**k, "need_weights": True})
        want = m(*args)
        assert len(calls) == len(m.attention)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
