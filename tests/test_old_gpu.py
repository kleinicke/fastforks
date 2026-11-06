"""An RTX 20 card (Turing, sm_75) has no kernel binary and no bfloat16 hardware: `portable` must still work."""

from __future__ import annotations

import dataclasses
import logging
import types

import pytest
import torch

from fastforks import KernelUnavailableError, stereo
from fastforks._core import kernels, registry, warmup
from fastforks._core import model as core_model
from fastforks.stereo._runner import Runner


@dataclasses.dataclass(frozen=True)
class Params:
    feature_channels: int
    num_transformer: int
    dim_expansion: int
    use_positivity: bool
    refine_iter: int


class Dummy(Runner):
    Params = Params
    dtypes = ("float16", "bfloat16", "float32")

    def __call__(self, left, right):  # noqa: ARG002 - never called, only built
        raise AssertionError


def _gpu(monkeypatch, capability):
    """Pretend one CUDA device of this compute capability; nothing touches a real GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _d=None: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _d=None: "NVIDIA GeForce RTX 2080")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(kernels.sys, "platform", "linux")
    monkeypatch.setattr(kernels.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(kernels, "driver_version", lambda: (13, 0))
    monkeypatch.setattr(warmup, "start", lambda _d: None)
    runners = {"portable": Dummy, "kernels": Dummy}
    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS=runners))


def test_auto_falls_back_to_portable_with_one_log_line(monkeypatch, caplog):
    _gpu(monkeypatch, (7, 5))
    with caplog.at_level(logging.INFO, logger="fastforks"):
        m = stereo.load("s2m2-s", device="cuda")
    assert m.backend.mode == "portable"
    lines = [r for r in caplog.records if "using portable" in r.getMessage()]
    assert len(lines) == 1
    assert "no binary for sm_75" in lines[0].getMessage()


def test_kernels_mode_raises_a_clear_error(monkeypatch):
    _gpu(monkeypatch, (7, 5))
    with pytest.raises(KernelUnavailableError, match=r"no binary for sm_75.*mode='auto'"):
        stereo.load("s2m2-s", mode="kernels", device="cuda")


@pytest.mark.parametrize(("capability", "dtype"), [((7, 5), "float32"), ((8, 6), "bfloat16"), ((12, 0), "bfloat16")])
def test_bfloat16_models_compute_in_float32_before_ampere(monkeypatch, capability, dtype):
    """DEFOM's card validates bfloat16; float16 overflows for it, so Turing gets float32."""
    _gpu(monkeypatch, capability)
    card, _ = registry.resolve("stereo", "defom-stereo-vitl-sceneflow")
    assert card.portable_cuda_dtype == "bfloat16"
    got = core_model.compute_dtype(card, Dummy, "portable", torch.device("cuda", 0))
    assert got == dtype


def test_float16_models_keep_float16_on_turing(monkeypatch):
    _gpu(monkeypatch, (7, 5))
    card, _ = registry.resolve("stereo", "s2m2-s")
    assert card.portable_cuda_dtype == "float16"
    assert core_model.compute_dtype(card, Dummy, "portable", torch.device("cuda", 0)) == "float16"
