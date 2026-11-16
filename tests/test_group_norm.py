"""`shared.group_norm`: GroupNorm from per-channel statistics equals `nn.GroupNorm` (every family that routes it)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fastforks.shared import group_norm as gn

CASES = [  # (batch, channels, groups, h, w): RealPLKSR, MVSplat, Track-On, one channel per group
    (1, 64, 4, 24, 20),
    (2, 32, 8, 16, 16),
    (2, 256, 32, 6, 8),
    (1, 8, 8, 5, 7),
]


def _norm(c: int, g: int, *, affine: bool) -> nn.GroupNorm:
    norm = nn.GroupNorm(g, c, eps=1e-5, affine=affine)
    if affine:
        with torch.no_grad():
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
    return norm


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stats_path_matches_group_norm(monkeypatch, case, affine, dtype):
    monkeypatch.setattr(gn, "MIN_NUMEL", 0)
    b, c, g, h, w = case
    norm = _norm(c, g, affine=affine).to(dtype)
    x = (torch.randn(b, c, h, w) * 3 + 1).to(dtype).contiguous(memory_format=torch.channels_last)
    want = nn.functional.group_norm(
        x.float(), g, *(t.float() if t is not None else None for t in (norm.weight, norm.bias))
    )
    got = gn.group_norm(norm, x)
    assert got.dtype == dtype
    tol = 1e-5 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(got.float(), want, rtol=tol, atol=tol)


def test_small_or_nchw_inputs_keep_group_norm(monkeypatch):
    calls = []
    real = torch.var_mean
    monkeypatch.setattr(torch, "var_mean", lambda *a, **k: calls.append(1) or real(*a, **k))
    norm = _norm(32, 8, affine=True)
    x = torch.randn(1, 32, 8, 8)
    torch.testing.assert_close(gn.group_norm(norm, x.contiguous(memory_format=torch.channels_last)), norm(x))
    monkeypatch.setattr(gn, "MIN_NUMEL", 0)
    torch.testing.assert_close(gn.group_norm(norm, x), norm(x))  # NCHW: F.group_norm
    assert not calls


def test_fast_group_norms_routes_every_group_norm(monkeypatch):
    monkeypatch.setattr(gn, "MIN_NUMEL", 0)
    net = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), _norm(16, 4, affine=True), nn.ReLU(), _norm(16, 16, affine=False)
    )
    x = torch.randn(2, 3, 9, 11).contiguous(memory_format=torch.channels_last)
    want = net(x)
    assert gn.fast_group_norms(net) == 2
    torch.testing.assert_close(net(x), want, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast")
def test_autocast_output_is_float32_like_group_norm(monkeypatch):
    monkeypatch.setattr(gn, "MIN_NUMEL", 0)
    norm = _norm(64, 4, affine=True).cuda()
    x = torch.randn(1, 64, 16, 16, device="cuda").half().contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.float16):
        want, got = norm(x), gn.group_norm(norm, x)
    assert got.dtype == want.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


def test_always_routes_small_channels_last_inputs_through_stats(monkeypatch):
    calls = []
    real = torch.var_mean
    monkeypatch.setattr(torch, "var_mean", lambda *a, **k: calls.append(1) or real(*a, **k))
    net = nn.Sequential(_norm(32, 8, affine=True))
    x = torch.randn(1, 32, 8, 8).contiguous(memory_format=torch.channels_last)
    want = net(x)
    assert gn.fast_group_norms(net, always=True) == 1
    torch.testing.assert_close(net(x), want, rtol=1e-5, atol=1e-5)
    assert calls
