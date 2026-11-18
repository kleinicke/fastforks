"""MapAnything stages its two transformer stacks with their original, distinct LayerScale precisions."""

from types import SimpleNamespace

import pytest
import torch

from fastforks._core.errors import OptionsError
from fastforks.multiview.models.mapanything import model


def test_stream_requires_cuda_before_checkpoint_load() -> None:
    ctx = SimpleNamespace(params=model.Params(long_side=518, stream=True), device=torch.device("cpu"))
    with pytest.raises(OptionsError, match="stream=True requires CUDA"):
        model.Portable(ctx)  # ty: ignore[invalid-argument-type]


@pytest.mark.gpu
def test_staged_block_order_and_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model, "DIM", 8)
    enc = torch.nn.ModuleList([model._Block(), model._Block()])
    info = torch.nn.ModuleList([model._Block()])
    dev = torch.device("cuda")
    for i, block in enumerate([*enc, *info]):
        # Values distinguish blocks after buffer reuse; retain sub-half precision in info-sharing LayerScale.
        block.ls1.gamma.data.fill_(i + 0.1234567)
    host = model._host_blocks(enc, info, torch.float16)
    resident = model._resident(enc, torch.float16, dev, info=False) + model._resident(
        info, torch.float16, dev, info=True
    )
    for _ in range(2):
        for expected, state in zip(resident, host.run(dev), strict=True):
            actual = model._Blk({k.partition(".")[2]: v for k, v in state.items()})
            for got, want in ((actual.g1, expected.g1), (actual.w12, expected.w12), (actual.n2[0], expected.n2[0])):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert resident[0].g1.dtype == torch.float16
    assert resident[-1].g1.dtype == torch.float32
    assert len(host.dev) == 2
