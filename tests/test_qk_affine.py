"""Captured decoder blocks must use their own query and key norm parameters."""

import importlib

import pytest
import torch
from torch import nn

from fastforks.shared.qk_affine import cached_qk_affine


class _NormPair(nn.Module):
    def __init__(self, value: int) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(8)
        self.k_norm = nn.LayerNorm(8)
        with torch.no_grad():
            self.q_norm.weight.fill_(value)
            self.k_norm.weight.fill_(value + 1)
            self.q_norm.bias.fill_(-value)
            self.k_norm.bias.fill_(1 - value)


class _Decoder(nn.Module):
    def __init__(self, family: str) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_NormPair(2 * i) for i in range(1, 13)])
        module = importlib.import_module(f"fastforks.{family}.model")
        self.runner = object.__new__(module._Fast)
        self.runner._aff = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x * weight + bias for weight, bias in map(self.runner.qk_affine, self.blocks)])


@pytest.mark.parametrize(
    "family", ["flow.models.megaflow", "multiview.models.anysplat", "multiview.models.pi3", "multiview.models.vggt"]
)
def test_export_uses_each_blocks_qk_norm(family: str) -> None:
    model = _Decoder(family).eval()
    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    cache = model.runner._aff
    with torch.no_grad():
        expected = model(x)
        assert len(cache) == len(model.blocks)
        # A compiler trace must read the parameters from each block, independently of the eager cache.
        wrong_affine = cached_qk_affine(model.blocks[0], cache)
        for key in cache:
            cache[key] = wrong_affine
        got = torch.export.export(model, (x,), strict=False).module()(x)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
