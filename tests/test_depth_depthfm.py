"""DepthFM ensemble decoding preserves member order when CUDA splits the batch."""

from __future__ import annotations

import pytest
import torch

from fastforks.depth.models.depthfm import model as family


@pytest.mark.gpu
def test_decode_ensemble_keeps_all_members(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    latent = torch.arange(5 * 4 * 3 * 7, device="cuda", dtype=torch.float32).reshape(5, 4, 3, 7)
    monkeypatch.setattr(family, "_DECODE_PIXELS", 2 * 3 * 7 * 64)
    batches: list[int] = []

    def decode(part: torch.Tensor) -> torch.Tensor:
        batches.append(part.shape[0])
        return part[:, :3] + 1

    with torch.inference_mode():
        result = family._decode_ensemble(decode, latent)
    assert batches == [2, 2, 1]
    torch.testing.assert_close(result, latent[:, :3] + 1, atol=0, rtol=0)
