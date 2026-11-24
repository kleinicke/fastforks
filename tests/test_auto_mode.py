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
    listed = json.loads((ROOT / "src/fastforks/_core/auto_portable.json").read_text(encoding="utf-8"))["portable"]
    assert listed
    for model_id in listed:
        card, _ = registry.resolve_any(model_id)
        assert "kernels" in card.modes, f"{model_id}: listed, but has no kernels backend"


def test_windows_reads_its_own_list(monkeypatch):
    data = json.loads((ROOT / "src/fastforks/_core/auto_portable.json").read_text(encoding="utf-8"))
    for platform, listed in (("win32", data["windows"]["portable"]), ("linux", data["portable"])):
        monkeypatch.setattr(core_model.sys, "platform", platform)
        core_model._auto_portable.cache_clear()
        assert core_model._auto_portable() == frozenset(listed), platform
    core_model._auto_portable.cache_clear()


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
        "peak_footprint_mb": 100,
    }
    cases = {
        gen_auto_mode.APPLE_DEVICE: [row, row | {"mode": "kernels", "ms": 7}, row | {"ref": "example@3"}],
        gen_auto_mode.APPLE_MLX_DEVICE: [
            row | {"ms": 1, "golden": "skip"},
            row | {"ms": 2, "input": "96x96"},
            row | {"ms": 5, "ref": "example@1"},
            row | {"ms": 8},
            row | {"ms": 8, "ref": "example@3", "peak_footprint_mb": 150},  # faster, but more memory than MPS
            row | {"ms": 5, "ref": "example@4"},
        ],
    }
    for device, rows in cases.items():
        (speed / f"{device}.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    measured = json.loads(gen_auto_mode.render_apple())["models"]
    assert "auto" not in measured["example@2"]
    assert measured["example@2"]["mps_auto"] == ["", "float32"]
    assert "kernels_default" not in measured["example@2"]
    assert "auto" not in measured["example@1"]
    assert "auto" not in measured["example@3"]  # 1.25x faster than MPS does not repay 50 % more memory


def test_mlx_route_where_mps_runs_out_of_memory():
    sys.path.insert(0, str(ROOT / "tools"))
    import gen_auto_mode  # noqa: PLC0415

    w = ("994x1482", "synthetic", "warm-median")
    mlx = {("portable", "float32"): {w: (13000.0, 9000.0, 40.0)}, ("portable", "float16"): {w: (10000.0, 6000.0, 30.0)}}
    mps = {("portable", "float16"): {w: (13000.0, 9000.0, 20.0)}}
    assert gen_auto_mode._oom_routes(mlx, mps, {"float32": {w}}) == {"float32": ("portable", "float32")}
    # MPS ran this dtype fine on the workload too: the usual comparison decides, not the OOM row
    assert gen_auto_mode._oom_routes(mlx, mps, {"float16": {w}}) == {}
    # MLX measured on a workload MPS never failed: no OOM route
    assert gen_auto_mode._oom_routes(mlx, {}, {"float32": {("1x1", "", "")}}) == {}


def test_apple_route_rule_weighs_speed_against_memory():
    sys.path.insert(0, str(ROOT / "tools"))
    import gen_auto_mode  # noqa: PLC0415

    score = gen_auto_mode._route_score
    assert score([0.94], [1.0], memory=True) == 0.94  # at least 5 % faster, no more memory
    assert score([0.97], [1.0], memory=True) is None  # within noise, the same memory
    assert score([1.04], [0.9], memory=True) == 1.04  # the same speed within noise, 10 % less memory
    assert score([1.04], [0.95], memory=True) is None
    assert score([1.06], [0.5], memory=True) is None  # slower than noise: memory alone does not decide
    assert score([0.94], [1.1], memory=True) is None  # 6 % faster does not repay 10 % more memory
    assert score([0.5], [1.1], memory=True) == 0.5  # twice as fast does
    assert score([0.5], [None], memory=True) is None  # unknown memory
    assert score([0.5], [None], memory=False) == 0.5


def test_apple_auto_takes_validated_float16_on_mps():
    sys.path.insert(0, str(ROOT / "tools"))
    import gen_auto_mode  # noqa: PLC0415

    w = ("224x224", "synthetic", "warm-median")
    full, half, metal = ("portable", "float32"), ("portable", "float16"), ("kernels", "float16")
    choice = gen_auto_mode._mps_choice
    assert choice({full: {w: (10.0, 100.0, 0.1)}, half: {w: (9.0, 100.0, 0.1)}}, {}) == (half, {})
    assert choice({full: {w: (10.0, 100.0, 0.1)}, half: {w: (9.8, 100.0, 0.1)}}, {}) == (full, {})
    assert choice({full: {w: (10.0, 100.0, 0.1)}, half: {w: (10.2, 80.0, 0.1)}}, {}) == (half, {})  # leaner
    assert choice({full: {w: (10.0, 100.0, 0.1)}, half: {w: (9.0, 120.0, 0.1)}}, {}) == (full, {})  # more memory
    # float32 ran out of memory there: float16 is the only portable path that runs
    assert choice({half: {w: (9.0, 100.0, 0.1)}}, {"float32": {w}}) == (half, {})
    assert choice({half: {w: (9.0, 100.0, 0.1)}}, {}) == (None, {})
    # the Metal kernels must beat the portable dtype `auto` would take, not only float32
    native = {full: {w: (10.0, 100.0, 0.1)}, half: {w: (8.0, 100.0, 0.1)}, metal: {w: (7.9, 100.0, 0.1)}}
    assert choice(native, {}) == (half, {})
    native[metal] = {w: (7.0, 100.0, 0.1)}
    assert choice(native, {}) == (half, {"": metal, "float16": metal})


def test_auto_mps_dtype_needs_the_card_and_the_table(monkeypatch):
    card, variant = registry.resolve("interpolation", "ema-vfi")
    assert "float16" in card.portable_mps_dtypes
    assert core_model._auto_mps_dtype(card, variant) == "float16"  # 1981 ms against 19632 ms in float32 (M1 Pro)
    monkeypatch.setattr(core_model, "_apple_mlx", dict)
    assert core_model._auto_mps_dtype(card, variant) is None
    monkeypatch.setattr(core_model, "_apple_mlx", lambda: {variant.ref: {"mps_dtype": "float16"}})
    unvalidated = core_model.dataclasses.replace(card, portable_mps_dtypes=("float32",))
    assert core_model._auto_mps_dtype(unvalidated, variant) is None


def test_apple_table_routes_only_to_card_mlx_dtypes():
    """Every MLX route and passing run in apple_mlx.json is a mode/dtype the card's MLX port declares."""
    models = json.loads((ROOT / "src/fastforks/_core/apple_mlx.json").read_text(encoding="utf-8"))["models"]
    assert models
    bad = []
    for key, row in models.items():
        card, _ = registry.resolve_any(key.split("@", 1)[0])
        used = {(mode, dtype) for dtype, mode in row.get("mlx", {}).items()}
        used |= {(mode, dtype) for mode, dtypes in row.get("passed", {}).items() for dtype in dtypes}
        if row.get("mlx_default"):
            used.add(tuple(row["mlx_default"]))
        bad += [f"{key}: {mode} {dtype}" for mode, dtype in sorted(used) if dtype not in card.mlx_dtypes(mode)]
    assert not bad, "routes to dtypes the card's MLX port does not declare:\n" + "\n".join(bad)
