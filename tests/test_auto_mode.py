"""`mode="auto"` stays on portable for models whose kernels measured no faster (auto_portable.json)."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from fastforks import KernelUnavailableError, stereo
from fastforks._core import model as core_model
from fastforks._core import registry
from test_old_gpu import _gpu

ROOT = Path(__file__).resolve().parents[1]


def test_list_is_current():
    subprocess.run([sys.executable, ROOT / "tools" / "gen_auto_mode.py", "--check"], check=True)


def test_listed_models_exist_and_have_kernels():
    listed = json.loads((ROOT / "src/fastforks/_core/auto_portable.json").read_text())["portable"]
    assert listed
    for model_id in listed:
        card, _ = registry.resolve_any(model_id)
        assert "kernels" in card.modes, f"{model_id}: listed, but has no kernels backend"


def test_auto_skips_kernels_for_listed_models(monkeypatch, caplog):
    _gpu(monkeypatch, (12, 0))
    monkeypatch.setattr(core_model, "_auto_portable", lambda: frozenset({"s2m2-s"}))
    with caplog.at_level(logging.INFO, logger="fastforks"):
        m = stereo.load("s2m2-s", device="cuda")
    assert m.backend.mode == "portable"
    assert [r.levelno for r in caplog.records if "using portable" in r.getMessage()] == [logging.INFO]
    assert "not faster than portable" in caplog.text


def test_explicit_kernels_mode_ignores_the_list(monkeypatch):
    def boom():
        msg = "explicit mode must not consult the list"
        raise AssertionError(msg)

    monkeypatch.setattr(core_model, "_auto_portable", boom)
    with pytest.raises(KernelUnavailableError, match="needs a CUDA device"):
        stereo.load("s2m2-s", mode="kernels", device="cpu")
