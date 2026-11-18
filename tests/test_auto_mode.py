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


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_auto_dtype_error_names_the_modes():
    from fastforks import matching  # noqa: PLC0415
    from fastforks._core.errors import OptionsError  # noqa: PLC0415

    with pytest.raises(OptionsError, match="mode='auto' found no backend for float16"):
        matching.load("romav2", device="cuda", dtype="float16")


def test_apple_routing_uses_passing_comparable_rows(tmp_path, monkeypatch):
    import gen_auto_mode  # noqa: PLC0415 - sibling tool under test

    monkeypatch.setattr(gen_auto_mode, "ROOT", tmp_path)
    speed = tmp_path / "docs" / "data" / "speed"
    speed.mkdir(parents=True)
    row = {
        "ref": "example@2",
        "input": "224x224",
        "scene": "synthetic",
        "protocol_id": "warm-median",
        "mode": "portable",
        "dtype": "float32",
        "status": "ok",
        "golden": "pass",
        "ms": 10,
    }
    cases = {
        gen_auto_mode.APPLE_DEVICE: [row, row | {"mode": "kernels", "ms": 7}],
        gen_auto_mode.APPLE_MLX_DEVICE: [
            row | {"ms": 1, "golden": "skip"},
            row | {"ms": 2, "input": "96x96"},
            row | {"ms": 5, "ref": "example@1"},
            row | {"ms": 8},
        ],
    }
    for device, rows in cases.items():
        (speed / f"{device}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    measured = json.loads(gen_auto_mode.render_apple())["models"]
    assert "auto" not in measured["example@2"]
    assert measured["example@2"]["mps_auto"] == ["", "float32"]
    assert "kernels_default" not in measured["example@2"]
    assert "auto" not in measured["example@1"]
