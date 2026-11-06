"""`fastbb.Flash64`, the one flash64 launcher of every family: fp16 overflow falls back to a wider variant.

CPU: the guard logic on a fake kernel (`Flash64._run` patched). GPU: q k^T beyond the fp16 range, where `"hh"` gives
NaN and `Flash64("hh")` returns the fp32 result. Source: no family launches `flash64_*` itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fastforks.kernels import fastbb
from fastforks.kernels.fastbb import FLASH_VARIANTS, Flash64

SRC = Path(__file__).parents[1] / "src" / "fastforks"


@pytest.fixture
def fake(monkeypatch):
    """`_run` returns NaN for the variants in `fake.bad`, else the variant's index; records the launched variants."""

    class Fake:
        bad: set[str] = set()  # noqa: RUF012 - set per test
        runs: list[str] = []  # noqa: RUF012

    def run(variant, q, k, v, scale):
        Fake.runs.append(variant)
        return torch.full_like(q, float("nan") if variant in Fake.bad else FLASH_VARIANTS.index(variant))

    monkeypatch.setattr(Flash64, "_run", staticmethod(run))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return Fake


def _call(attn):
    q = torch.zeros(1, 4, 2, 64)
    return attn(q, q, q, 0.125)


def test_finite_output_runs_once(fake):
    attn = Flash64("hh", probe=2)
    assert _call(attn).eq(0).all()
    assert fake.runs == ["hh"]
    assert attn.variant == "hh"
    assert attn.probe == 1


@pytest.mark.parametrize(("bad", "kept"), [({"hh"}, "fh"), ({"hh", "fh"}, "ff")])
def test_overflow_falls_back_and_pins(fake, bad, kept):
    fake.bad = bad
    attn = Flash64("hh")
    assert _call(attn).eq(FLASH_VARIANTS.index(kept)).all()
    assert attn.variant == kept
    fake.runs.clear()
    _call(attn)
    assert fake.runs == [kept]  # pinned: no retry of the overflowing variant


def test_nan_inputs_keep_the_variant(fake):
    fake.bad = set(FLASH_VARIANTS)
    attn = Flash64("fh")
    assert _call(attn).isnan().all()
    assert fake.runs == ["fh", "ff"]
    assert attn.variant == "fh"


def test_no_checks_after_probe_or_during_capture(fake, monkeypatch):
    fake.bad = {"hh"}
    attn = Flash64("hh", probe=0)
    assert _call(attn).isnan().all()  # unchecked: no sync, no extra launch
    assert fake.runs == ["hh"]
    attn = Flash64("hh")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    fake.runs.clear()
    _call(attn)
    assert fake.runs == ["hh"]
    assert attn.probe == fastbb.FLASH_PROBE


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="flash64 variant"):
        Flash64("hf")


def test_no_family_launches_flash64_directly():
    """Every caller gets the guard: the kernel names appear in `kernels/fastbb` only."""
    own = SRC / "kernels" / "fastbb" / "__init__.py"
    bad = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if p != own and re.search(r"flash64_r|[\"']flash64_", p.read_text(encoding="utf-8"))
    ]
    assert not bad, f"flash64 launched outside fastbb.Flash64: {bad}"


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["hh", "fh"])
def test_overflowing_logits_give_the_fp32_result(variant):
    """ "hh": |q . k| ~ 1.6e5 (2.5x the fp16 range) from one outlier channel, as DINOv3 ViT-S / B. "fh": equal logits
    and v ~ 4.5e3, so a 64-key tile of p v sums to ~2.9e5."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    g = torch.Generator().manual_seed(0)
    b, n, h = 1, 300, 2
    q, k, v = (torch.randn(b, n, h, 64, generator=g) for _ in range(3))
    if variant == "hh":
        q[..., 0], k[..., 0] = 400.0, 400.0
    else:
        q, k = torch.zeros_like(q), torch.zeros_like(k)
        q[..., 0], k[..., 0] = 4.0, 4.0
        v = 3e3 * (1 + torch.rand(b, n, h, 64, generator=g))
    q, k, v = (t.cuda().half() for t in (q, k, v))
    scale = 64**-0.5
    raw = Flash64._run(variant, q, k, v, scale)
    assert not torch.isfinite(raw).all(), "the test input must overflow the variant"
    attn = Flash64(variant)
    out = attn(q, k, v, scale)
    assert attn.variant != variant
    ref = F.scaled_dot_product_attention(*(t.float().transpose(1, 2) for t in (q, k, v)), scale=scale).transpose(1, 2)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item())
