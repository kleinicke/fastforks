"""Sapiens uses the selected backend for both resident and streamed transformer blocks."""

import copy
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from fastforks._core.errors import OptionsError
from fastforks.segmentation.models.sapiens import model


def test_stream_requires_cuda() -> None:
    ctx = SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.float32, params=SimpleNamespace(stream=True))
    with pytest.raises(OptionsError, match="requires CUDA"):
        model.Portable(ctx)  # ty: ignore[invalid-argument-type]


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["portable", "kernels"])
def test_streamed_logits_match_resident(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(58)
    arch = model.Params(embed_dim=128, depth=3, num_heads=2, grid=(4, 5), channels=32, num_classes=8)
    net = model.Sapiens(arch).eval().requires_grad_(requires_grad=False)
    monkeypatch.setattr(model, "_load", lambda _ctx: copy.deepcopy(net))
    runners = [
        model.RUNNERS[mode](
            SimpleNamespace(  # ty: ignore[invalid-argument-type]
                device=torch.device("cuda"),
                torch_dtype=torch.float16,
                params=dataclasses.replace(arch, stream=enabled),
                io=SimpleNamespace(input="0-1"),
            )
        )
        for enabled in (False, True)
    ]
    with torch.inference_mode():
        for batch in (1, 2, 1):
            image = torch.rand(batch, 3, 85, 92, device="cuda")
            expected, streamed = (runner(image)["logits"] for runner in runners)
            torch.testing.assert_close(streamed, expected, atol=0, rtol=0)
