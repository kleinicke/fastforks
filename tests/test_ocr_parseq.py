"""PARSeq: the card, the input resize, early stopping vs the fixed-length (export) decoding."""

from __future__ import annotations

import pytest
import torch

from fastforks._core import registry
from fastforks.ocr.models.parseq.model import to_input
from fastforks.ocr.models.parseq.net import IMAGE, PARSeq


def test_card() -> None:
    cards = {v.id: c for c, v in registry.cards("ocr").values() if c.family == "parseq"}
    assert sorted(cards) == ["parseq", "parseq-tiny"]
    assert cards["parseq"].licence.code == "Apache-2.0"


def test_to_input_range_and_float_images() -> None:
    img = torch.randint(0, 256, (1, 3, 40, 90), dtype=torch.uint8)
    x = to_input(img)
    assert x.shape == (1, 3, *IMAGE)
    assert float(x.min()) >= -1
    assert float(x.max()) <= 1
    torch.testing.assert_close(to_input(img.float() / 255), x, rtol=0, atol=0)  # float input: the same 8-bit levels


@pytest.mark.slow
def test_stop_early_is_the_full_decoding() -> None:
    """`stop=False` (all 26 steps, export) gives the same refined logits as the early exit."""
    torch.manual_seed(0)
    net = PARSeq(192, 3, 6).eval()
    with torch.no_grad():
        net.pos_queries.normal_(0, 0.02)
        net.head.bias[0] = 4  # the end token wins early, so the early exit triggers
        x = torch.rand(2, 3, *IMAGE) * 2 - 1
        a, b = net(x), net(x, stop=False)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-5)


def _runner(device: str, *, cuda_graphs: bool) -> tuple[object, torch.Tensor]:
    from types import SimpleNamespace  # noqa: PLC0415

    from fastforks.ocr.models.parseq.model import Params, Portable  # noqa: PLC0415

    torch.manual_seed(0)
    ref = PARSeq(192, 3, 6)
    with torch.no_grad():
        ref.pos_queries.normal_(0, 0.02)
    ctx = SimpleNamespace(
        params=Params(dim=192, enc_heads=3, dec_heads=6, cuda_graphs=cuda_graphs),
        torch_dtype=torch.float32,
        device=torch.device(device),
        state_dict=ref.state_dict,
    )
    return Portable(ctx), torch.randint(0, 256, (1, 3, 40, 90), dtype=torch.uint8, device=device)  # type: ignore[arg-type]


def test_no_graph_on_cpu() -> None:
    runner, _ = _runner("cpu", cuda_graphs=True)
    assert runner._graphs is None
    runner.release_graphs()


@pytest.mark.gpu
@torch.inference_mode()
def test_graph_replay_matches_eager() -> None:
    """The CUDA graph replays the AR loop, then the eager-length refinement matches the early-exit result."""
    graphed, img = _runner("cuda", cuda_graphs=True)
    eager, _ = _runner("cuda", cuda_graphs=False)
    ref = eager(img)
    for _ in range(4):  # call 1 eager, call 2 captures, the rest replay
        out = graphed(img)
        assert out["texts"] == ref["texts"]
        torch.testing.assert_close(out["scores"], ref["scores"], rtol=0, atol=0)
    assert len(graphed._graphs.graphs) == 1
    x = to_input(img)
    torch.testing.assert_close(graphed._logits(img), eager.net(x), rtol=0, atol=0)
    graphed.release_graphs()
    assert not graphed._graphs.graphs
