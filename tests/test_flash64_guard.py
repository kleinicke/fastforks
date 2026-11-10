"""`fastbb.Flash64`, the one flash64 launcher of every family: fp16 overflow falls back to a wider variant.

CPU: the guard logic on a fake kernel (`Flash64._run` patched): a single launch, and a model call (`kernels.checked`,
the kernels runners' entry points) that reruns after a non-finite launch. GPU: q k^T beyond the fp16 range, where
`"hh"` gives NaN and `Flash64("hh")` returns the fp32 result; every family on fastbb, through its public call with the
narrow variants forced to overflow. Source: no family launches `flash64_*` itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fastforks._core import kernels, registry
from fastforks._core.errors import WeightsError
from fastforks._core.model import _kernels_entry_points
from fastforks._core.runner import Runner
from fastforks.kernels import fastbb
from fastforks.kernels.fastbb import FLASH_VARIANTS, Flash64
from test_golden import API, INPUTS, SMALLEST, TASK, _check, forward, golden, options

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


class _TwoAttentions(Runner):
    """A model call with two `Flash64` launchers, the second fed by the first (its NaN propagates)."""

    def __init__(self) -> None:
        super().__init__(None)  # no card needed
        self.a, self.b, self.calls = Flash64("hh"), Flash64("hh"), 0

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        o = self.a(q, q, q, 0.125)
        return self.b(o, o, o, 0.125) + o


def test_model_call_reruns_after_the_first_bad_launch(fake, monkeypatch):
    """One sync per run; only the first non-finite launcher widens (the second saw its NaN); the call reruns."""

    def run(variant, q, k, v, scale):  # "hh" overflows on the first launcher's input (zeros); NaN propagates
        return torch.full_like(q, float("nan")) if q.isnan().any() or (variant == "hh" and q.eq(0).all()) else q + 1

    monkeypatch.setattr(Flash64, "_run", staticmethod(run))
    syncs = []  # launches read per sync
    finite = kernels._Checks.finite
    monkeypatch.setattr(kernels._Checks, "finite", lambda self: syncs.append(len(self.launches)) or finite(self))
    model = _TwoAttentions()
    _kernels_entry_points(model)
    q = torch.zeros(1, 4, 2, 64)
    assert model(q).eq(3).all()
    assert (model.a.variant, model.b.variant, model.calls, syncs) == ("fh", "hh", 2, [2, 2])
    assert model.a.probe == fastbb.FLASH_PROBE - 1  # a new probe window for "fh"
    assert model(q).eq(3).all()
    assert (model.calls, syncs) == (3, [2, 2, 2])
    assert not kernels.checking()


def test_model_call_with_nan_inputs_keeps_the_variants(fake):
    fake.bad = set(FLASH_VARIANTS)
    run = _TwoAttentions()
    _kernels_entry_points(run)
    assert run(torch.zeros(1, 4, 2, 64)).isnan().all()
    assert (run.a.variant, run.b.variant, run.calls) == ("hh", "hh", 3)  # hh, fh, ff: all non-finite
    assert run.a.probe == run.b.probe == fastbb.FLASH_PROBE - 1  # one run's launch counts, no new window


class _Launchers(Runner):
    """A model call with `n` `Flash64` launchers, each launched `repeat` times."""

    def __init__(self, n: int, repeat: int) -> None:
        super().__init__(None)  # no card needed
        self.attns, self.repeat = [Flash64("hh") for _ in range(n)], repeat

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        for a in self.attns:
            for _ in range(self.repeat):
                q = a(q, q, q, 0.125)
        return q


@pytest.mark.parametrize("slots", [kernels._SLOTS, 7])  # 7: the checks span several buffers
def test_reruns_do_not_use_up_the_other_launchers_probes(fake, monkeypatch, slots):
    """3 launchers x 40 launches, "hh" / "fh" overflow: 6 reruns, yet every launcher is checked until it is at "ff"."""
    monkeypatch.setattr(kernels, "_SLOTS", slots)
    fake.bad = {"hh", "fh"}
    run = _Launchers(3, 40)
    _kernels_entry_points(run)
    assert run(torch.zeros(1, 4, 2, 64)).isfinite().all()
    assert [(a.variant, a.probe) for a in run.attns] == [("ff", fastbb.FLASH_PROBE - 40)] * 3


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


def _on_fastbb(ref: str) -> bool:
    card = API[TASK[ref]].model_card(ref)
    return "kernels" in card.modes and "fastbb" in card.kernel_libs


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("ref", [r for r in SMALLEST if _on_fastbb(r)])
def test_forced_overflow_reruns_the_model_call(ref, monkeypatch):
    """Every family on fastbb, public call: "hh" / "fh" outputs set to inf, so each launcher ends at "ff"; the rerun
    call and the next one (pinned; a graph capture where the family has one) are within the golden's tolerances."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    want = golden(ref)
    real, call = Flash64._run, Flash64.__call__
    attns: set[Flash64] = set()

    def run(variant, q, k, v, scale):
        o = real(variant, q, k, v, scale)
        return o if variant == "ff" else o.fill_(float("inf"))

    def spy(self, *a):
        attns.add(self)
        return call(self, *a)

    card, variant = registry.resolve(TASK[ref], ref)
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"weights of {ref} not available locally: {e}")
    model = API[TASK[ref]].load(ref, mode="kernels", device="cuda", options=options(ref, want))
    monkeypatch.setattr(Flash64, "_run", staticmethod(run))
    monkeypatch.setattr(Flash64, "__call__", spy)
    outputs = {k: v for k, v in want.items() if k not in INPUTS}
    for i in (1, 2):
        got, _ = forward(model, ref, want)
        if not attns:
            pytest.skip(f"{ref}: no flash64 attention on this input")
        assert {a.variant for a in attns} == {"ff"}
        _check(ref, "kernels", outputs, got, f"call {i}: ")
