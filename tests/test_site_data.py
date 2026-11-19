"""Site measurements keep their evaluated precision and input size."""

import json
from types import SimpleNamespace

import site_data


def test_compile_and_tensorrt_keep_their_input_sizes(monkeypatch):
    ref = "nninteractive"
    portable = {"mode": "portable", "input": "200x200", "ms": 215, "dtype": "float16"}
    compiled = {"mode": "compile", "input": "200x200", "ms": 162}
    engine = {"ref": ref, "mode": "tensorrt", "input": "192x192", "ms": 262, "export_and_build_s": 32}
    failed = {"ref": ref, "mode": "tensorrt", "input": "200x200", "fails": "OutOfMemoryError"}
    monkeypatch.setattr(
        site_data.g,
        "_compile_rows",
        lambda: {
            (ref, "compile", "200x200"): compiled,
            (ref, "tensorrt", "192x192"): engine,
            (ref, "tensorrt", "200x200"): failed,
        },
    )
    monkeypatch.setattr(site_data.g, "_speed_rows", lambda: {ref: [portable]})

    data = site_data._speed(SimpleNamespace(id=ref), "volume", {ref: [portable]})
    sizes = {r["input"]: r for r in data["inputs"]}
    assert sizes["192x192"]["tensorrt"]["ms"] == 262
    assert sizes["192x192"]["tensorrt"]["setup_s"] == 32
    assert "compile" not in sizes["192x192"]
    assert sizes["200x200"]["compile"]["ms"] == 162
    assert "fails" in sizes["200x200"]["tensorrt"]

    lines = site_data.g._compile_block().splitlines()
    rows = [line.strip("|").split(" | ") for line in lines if line.startswith(f"| {ref} |")]
    assert len(rows) == 2
    for row in rows:
        assert any(cell.startswith("262 ms") for cell in row) == (row[1] == "192x192")
        assert ("162 ms" in row) == (row[1] == "200x200")


def test_video_measurements_keep_distinct_frame_counts(monkeypatch):
    ref = "cotracker3-online"
    engines = {
        (ref, "tensorrt", shape): {"ref": ref, "mode": "tensorrt", "input": shape, "ms": ms}
        for shape, ms in (("256x256", 50), ("24x256x256", 141))
    }
    monkeypatch.setattr(site_data.g, "_compile_rows", lambda: engines)
    monkeypatch.setattr(site_data.g, "_oom", set)

    speed = site_data._speed(SimpleNamespace(id=ref), "tracking", {})
    assert [(row["input"], row["tensorrt"]["ms"]) for row in speed["inputs"]] == [
        ("256x256", 50),
        ("24x256x256", 141),
    ]


def test_standalone_engine_measurement_remains_visible_when_export_is_planned(tmp_path, monkeypatch):
    g = site_data.g
    variant = SimpleNamespace(id="model", planned={"tensorrt": "export needs validation"})
    monkeypatch.setattr(g, "ROOT", tmp_path)
    monkeypatch.setattr(g.registry, "TASKS", ("depth",))
    monkeypatch.setattr(g.registry, "cards", lambda _: {"model": (SimpleNamespace(family="model"), variant)})
    data = tmp_path / "docs" / "data"
    data.mkdir(parents=True)
    rows = [
        {"ref": "model", "mode": "tensorrt", "input": "32x32", "ms": 5},
        {
            "ref": "model",
            "mode": "tensorrt",
            "input": "64x64",
            "ms": 10,
            "standalone_build": True,
            "device_mb": 20,
            "peak_alloc_mb": 30,
            "peak_reserved_mb": 40,
        },
        {"ref": "model", "mode": "compile", "input": "64x64", "ms": 12, "backend_part": "VAE only"},
    ]
    (data / "speed_compile.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    compiled = g._compile_rows.__wrapped__()
    assert compiled["model", "tensorrt", "32x32"]["fails"] == g.PLANNED
    assert compiled["model", "tensorrt", "64x64"]["ms"] == 10
    monkeypatch.setattr(g, "_compile_rows", lambda: compiled)
    monkeypatch.setattr(g, "_engine_part", g._engine_part.__wrapped__)
    monkeypatch.setattr(g, "_oom", set)

    sizes = {row["input"]: row for row in site_data._speed(variant, "depth", {})["inputs"]}
    engine = sizes["64x64"]["tensorrt"]
    assert engine["standalone"]
    assert engine["part"] == "standalone build"
    assert (engine["peak_alloc_mb"], engine["peak_reserved_mb"]) == (50, 60)
    assert sizes["64x64"]["compile"]["part"] == "VAE only"
    assert "32x32" not in sizes
    with_portable = {"model": [{"input": "32x32", "mode": "portable", "ms": 20}]}
    failed = site_data._speed(variant, "depth", with_portable)["inputs"][0]
    assert failed["tensorrt"] == {"fails": "export needs validation", "planned": True}
    assert g._compile_ms(compiled["model", "tensorrt", "64x64"]) == "10.0 ms (standalone build)"
    assert g._compile_ms(compiled["model", "tensorrt", "32x32"]) == "public API unavailable"
    assert g._compile_ms(compiled["model", "compile", "64x64"]) == "12.0 ms (VAE only)"


def test_quality_keeps_precision_specific_rows(monkeypatch):
    monkeypatch.setattr(site_data.g, "_overlap", lambda *_: None)
    monkeypatch.setattr(site_data, "eval_version", lambda *_: "test")
    card = SimpleNamespace(task="depth")
    variant = SimpleNamespace(id="model")
    rows = [
        {"dataset": "nyu", "mode": "upstream", "tier": "upstream", "dtype": "float32", "metrics": {"rel_absrel": 0.1}},
        {
            "dataset": "nyu",
            "mode": "upstream",
            "tier": "upstream16",
            "dtype": "float16",
            "metrics": {"rel_absrel": 0.2},
        },
        {
            "dataset": "nyu",
            "mode": "portable",
            "tier": "portable32",
            "dtype": "float32",
            "metrics": {"rel_absrel": 0.3},
        },
        {"dataset": "nyu", "mode": "portable", "tier": "portable", "dtype": "float16", "metrics": {"rel_absrel": 0.4}},
    ]
    modes = site_data._quality(card, variant, {"model": rows})[0]["modes"]
    assert {key: value["metrics"]["rel_absrel"] for key, value in modes.items()} == {
        "original32": 0.1,
        "original16": 0.2,
        "portable32": 0.3,
        "portable": 0.4,
    }


def test_bfloat16_original_does_not_claim_float32(monkeypatch):
    monkeypatch.setattr(site_data.g, "_overlap", lambda *_: None)
    monkeypatch.setattr(site_data, "eval_version", lambda *_: "test")
    row = {
        "dataset": "nyu",
        "mode": "upstream",
        "tier": "upstream",
        "dtype": "bfloat16",
        "metrics": {"rel_absrel": 0.1},
    }
    modes = site_data._quality(SimpleNamespace(task="depth"), SimpleNamespace(id="model"), {"model": [row]})[0]["modes"]
    assert set(modes) == {"original16"}


def test_explicit_bf16_original_keeps_autocast_row(monkeypatch):
    monkeypatch.setattr(site_data.g, "_overlap", lambda *_: None)
    monkeypatch.setattr(site_data, "eval_version", lambda *_: "test")
    rows = [
        {
            "dataset": "nyu",
            "mode": "upstream",
            "tier": "upstream16",
            "dtype": "float16",
            "metrics": {"rel_absrel": 0.1},
        },
        {
            "dataset": "nyu",
            "mode": "upstream",
            "tier": "upstreambf16",
            "dtype": "bfloat16",
            "metrics": {"rel_absrel": 0.2},
        },
    ]
    modes = site_data._quality(SimpleNamespace(task="depth"), SimpleNamespace(id="model"), {"model": rows})[0]["modes"]
    assert modes["original16"]["metrics"]["rel_absrel"] == 0.1
    assert modes["originalhalf"]["metrics"]["rel_absrel"] == 0.2
    assert modes["originalhalf"]["dtype"] == "bfloat16"


def test_task_without_accuracy_protocol_has_no_quality_rows():
    assert site_data._quality(SimpleNamespace(task="edges"), SimpleNamespace(id="model"), {}) == []


def test_default_float32_portable_is_also_labelled_as_fp32(monkeypatch):
    monkeypatch.setattr(site_data.g, "_overlap", lambda *_: None)
    monkeypatch.setattr(site_data, "eval_version", lambda *_: "test")
    row = {"dataset": "nyu", "mode": "portable", "dtype": "float32", "metrics": {"rel_absrel": 0.2}}
    modes = site_data._quality(SimpleNamespace(task="depth"), SimpleNamespace(id="model"), {"model": [row]})[0]["modes"]
    assert modes["portable32"]["metrics"] == modes["portable"]["metrics"]
    assert "same measurement" in modes["portable32"]["note"]


def test_stream_rows_preserve_explicit_settings_and_oom(tmp_path, monkeypatch):
    data = tmp_path / "docs" / "data"
    data.mkdir(parents=True)
    rows = [
        {"ref": "large", "options": {"stream": False}, "status": "oom"},
        {"ref": "large", "options": {"stream": True}, "ms": 18.7, "peak_alloc_mb": 400},
        {"ref": "large", "options": {"quant": "nf4", "stream": True}, "ms": 11.2},
        {"ref": "other", "ms": 5.6},
    ]
    (data / "speed_options.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(site_data.g, "ROOT", tmp_path)
    assert site_data.g.stream_rows.__wrapped__() == {"large": rows[:2]}
