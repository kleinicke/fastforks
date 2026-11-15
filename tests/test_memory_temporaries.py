"""Peak-memory fixes that keep no temporaries alive (the CPU-checkable half; GPU peaks are guarded by the speed floor).

Each fix removed a tensor that lived next to its successor at the peak. On the CPU the structural property is
checked: the freed object is gone / the copy is not made, and the result is unchanged. The GPU goldens
(`tests/test_golden.py -m gpu`) cover the numbers of the attention chunk (c-foundationstereo-s) and patchmatchnet;
a live-bytes assertion needs their weights and a CUDA allocator, which a CPU test does not have.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastforks.restoration.models.dat import model as dat_model
from fastforks.restoration.models.dat.net.dat_arch import Adaptive_Spatial_Attention
from fastforks.tracking.models.locotrack.model import _join


@pytest.mark.parametrize("n", [1, 3])
def test_locotrack_join_releases_the_chunks(n: int) -> None:
    chunks = [torch.randn(2, 4) for _ in range(n)]
    want = torch.cat(chunks)
    out = _join(chunks)
    assert chunks == []  # the per-chunk lists do not stay alive next to the joined tensor
    torch.testing.assert_close(out, want)


@pytest.mark.filterwarnings("ignore:Initializing zero-element")  # upstream network, built here directly
@pytest.mark.filterwarnings("ignore:torch.meshgrid")
@pytest.mark.parametrize(("h", "w", "pads"), [(16, 16, False), (13, 16, True)])
def test_dat_spatial_block_pads_only_when_needed(monkeypatch: pytest.MonkeyPatch, h: int, w: int, pads: bool) -> None:
    torch.manual_seed(0)
    m = Adaptive_Spatial_Attention(dim=16, num_heads=2, split_size=[8, 8], shift_size=[1, 2], qkv_bias=True).eval()
    x = torch.randn(1, h * w, 16)
    calls = []
    real = F.pad
    monkeypatch.setattr(F, "pad", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    with torch.no_grad():
        got = dat_model._spatial_block(m, x, h, w)  # noqa: SLF001 - the helper under test
        made = bool(calls)
        want = m(x, h, w)
    assert made is pads  # a divisible size makes no padded copy of q, k, v in the helper
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
