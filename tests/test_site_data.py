"""Site measurements keep their evaluated precision and input size."""

import json
from types import SimpleNamespace

import gen_docs as g
import site_data
from fastforks._core import registry


def test_parseq_prefers_word_crop_without_relabeling_page_ocr(monkeypatch):
    monkeypatch.setattr(site_data.g, "_compile_rows", dict)
    monkeypatch.setattr(site_data.g, "_oom", set)
    card, word = registry.resolve_any("parseq")
    rows = {
        word.id: [{"input": "32x128", "mode": "portable", "ms": 2}],
        "ppocrv5-server": [{"input": "1080x1920", "mode": "portable", "ms": 20}],
    }
    data = site_data._speed(word, card.task, rows, native=site_data._native_speed_input(card, word))
    assert data["standard"] == "1080x1920"
    assert data["preferred"] == "32x128"
    assert data["inputs"][0]["native"]
    assert not data["inputs"][0]["standard"]

    card, page = registry.resolve_any("ppocrv5-server")
    data = site_data._speed(page, card.task, rows, native=site_data._native_speed_input(card, page))
    assert data["standard"] == data["preferred"] == "1080x1920"
    assert data["inputs"][0]["standard"]
    assert not data["inputs"][0]["native"]


def test_classification_prefers_measured_native_crop_to_task_standard(monkeypatch):
    monkeypatch.setattr(site_data.g, "_compile_rows", dict)
    monkeypatch.setattr(site_data.g, "_oom", set)
    card, model = registry.resolve_any("coatnet_rmlp_2_rw_384")
    native = site_data._native_speed_input(card, model)
    row = {"mode": "portable", "ms": 2}
    rows = {model.id: [row | {"input": "224x224"}, row | {"input": "384x384"}]}
    data = site_data._speed(model, card.task, rows, native=native)
    assert data["preferred"] == "384x384"
    assert data["standard"] == "224x224"
    assert [(r["input"], r["standard"], r["native"]) for r in data["inputs"]] == [
        ("224x224", True, False),
        ("384x384", False, True),
    ]

    data = site_data._speed(model, card.task, {model.id: rows[model.id][:1]}, native=native)
    assert data["preferred"] == "224x224"
    assert [r["input"] for r in data["inputs"]] == ["224x224"]


def test_preferred_speed_falls_back_to_an_existing_input(monkeypatch):
    monkeypatch.setattr(site_data.g, "_compile_rows", dict)
    monkeypatch.setattr(site_data.g, "_oom", set)
    model = SimpleNamespace(id="model")
    row = {"input": "256x512", "mode": "portable", "ms": 2}
    assert site_data._speed(model, "segmentation", {model.id: [row]})["preferred"] == "256x512"
    assert site_data._speed(model, "segmentation", {})["preferred"] is None


def test_preferred_speed_keeps_a_measured_call_over_a_standalone_engine(monkeypatch):
    monkeypatch.setattr(site_data.g, "_oom", set)
    engine = {"ref": "pi3", "mode": "tensorrt", "input": "8x512x640", "ms": 8, "compare_to_original": False}
    monkeypatch.setattr(site_data.g, "_compile_rows", lambda: {("pi3", "tensorrt", "8x512x640"): engine})
    card, variant = registry.resolve_any("pi3")
    rows = {variant.id: [{"input": "336x518", "mode": "portable", "ms": 822}]}
    speed = site_data._speed(variant, card.task, rows)
    profiles = {p["input"]: p for p in speed["inputs"]}
    assert speed["preferred"] == "8x336x518"
    assert profiles["8x512x640"]["tensorrt"]["compare_to_original"] is False


def test_pi3_eight_view_legacy_measurements_share_the_export_profile(monkeypatch):
    monkeypatch.setattr(site_data.g, "_oom", set)
    card, variant = registry.resolve_any("pi3")
    legacy, explicit = "336x518", "8x336x518"
    three_view = {"mode": "portable", "input": "3x336x518", "views": 3, "ms": 50}
    rows = {
        variant.id: [
            {"mode": "portable", "input": explicit, "ms": 826},
            {"mode": "portable", "input": legacy, "ms": 822},
            {"mode": "kernels", "input": legacy, "ms": 560},
            {"mode": "upstream", "input": legacy, "dtype": "bfloat16", "ms": 1057},
            three_view,
        ]
    }
    engine = {"ref": variant.id, "mode": "tensorrt", "input": explicit, "ms": 691, "views": 8}
    old_engine = engine | {"input": legacy, "fails": "old public export failed"}
    compile_row = {"ref": variant.id, "mode": "compile", "input": legacy, "ms": 752}
    monkeypatch.setattr(
        site_data.g,
        "_compile_rows",
        lambda: {
            (variant.id, "tensorrt", explicit): engine,
            (variant.id, "tensorrt", legacy): old_engine,
            (variant.id, "compile", legacy): compile_row,
        },
    )
    speed = site_data._speed(variant, card.task, rows)
    by_size = {r["input"]: r for r in speed["inputs"]}
    assert speed["preferred"] == explicit
    assert set(by_size) == {explicit, three_view["input"]}
    profile = by_size[explicit]
    assert profile["workload"] == {"views": 8}
    assert {k: r["ms"] for k, r in profile["modes"].items()} == {
        "kernels": 560,
        "original16": 1057,
        "portable": 826,
    }
    assert profile["compile"]["ms"] == 752
    assert profile["tensorrt"]["ms"] == 691
    assert by_size[three_view["input"]]["workload"] == {"views": 3}


def test_unmeasured_planned_backend_reports_card_reason(monkeypatch):
    monkeypatch.setattr(site_data.g, "_compile_rows", dict)
    monkeypatch.setattr(site_data.g, "_oom", set)
    card, variant = registry.resolve_any("mlsd")
    speed = {variant.id: [{"input": "1080x1920", "mode": "kernels", "ms": 8}]}
    (result,) = site_data._speed(variant, card.task, speed)["inputs"]
    assert result["tensorrt"] == {"fails": variant.planned["tensorrt"], "planned": True}
    assert "compile" not in result


def test_legacy_original_speed_rows_use_their_recorded_precision(monkeypatch):
    monkeypatch.setattr(site_data.g, "_compile_rows", dict)
    monkeypatch.setattr(site_data.g, "_oom", set)
    model = SimpleNamespace(id="model")
    rows = {
        model.id: [
            {"input": "480x854", "mode": "upstream", "dtype": "float32", "ms": 273},
            {"input": "480x854", "mode": "upstream", "dtype": "bfloat16", "ms": 203},
            {"input": "256x256", "mode": "upstream", "ms": 23},
        ]
    }
    data = site_data._speed(model, "tracking", rows)
    speed = {row["input"]: row["modes"] for row in data["inputs"]}
    assert speed["480x854"]["original32"] == {"dtype": "float32", "ms": 273}
    assert speed["480x854"]["original16"] == {"dtype": "bfloat16", "ms": 203}
    assert speed["256x256"] == {"original16": {"dtype": "float16", "ms": 23}}


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


def test_video_measurements_keep_distinct_frame_counts(tmp_path, monkeypatch):
    g = site_data.g
    ref = "cotracker3-online"
    data = tmp_path / "docs" / "data"
    data.mkdir(parents=True)
    engine = {
        "ref": ref,
        "mode": "tensorrt",
        "input": "256x256",
        "ms": 50,
        "comparison_input": "8x256x256",
        "standalone_build": True,
    }
    separate = {"ref": ref, "mode": "tensorrt", "input": "24x256x256", "ms": 141, "standalone_build": True}
    (data / "speed_compile.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in (engine, separate)), encoding="utf-8"
    )
    monkeypatch.setattr(g, "ROOT", tmp_path)
    engines = g._compile_rows.__wrapped__()
    monkeypatch.setattr(site_data.g, "_compile_rows", lambda: engines)
    monkeypatch.setattr(site_data.g, "_oom", set)

    native = {"mode": "portable", "input": "256x256", "ms": 100, "frames": 24, "queries": 256}
    speed = site_data._speed(SimpleNamespace(id=ref), "tracking", {ref: [native]})
    rows = {row["input"]: row for row in speed["inputs"]}
    assert rows["256x256"]["modes"]["portable"]["ms"] == 100
    assert "tensorrt" not in rows["256x256"]
    assert rows["8x256x256"]["tensorrt"]["ms"] == 50
    assert rows["24x256x256"]["tensorrt"]["ms"] == 141


def test_published_clip_timings_match_exported_workloads():
    speed = site_data.g._speed_rows()
    profiles = (
        ("video-depth-anything-s", "depth", "480x640", "88x480x640"),
        ("locotrack-b", "tracking", "480x854", "8x480x854"),
        ("locotrack-s", "tracking", "480x854", "8x480x854"),
    )
    for ref, task, standard_size, clip_size in profiles:
        card = site_data._speed(SimpleNamespace(id=ref), task, speed)
        by_size = {row["input"]: row for row in card["inputs"]}
        standard = by_size[standard_size]
        clip = by_size[clip_size]
        assert "portable" in standard["modes"]
        assert "portable" in clip["modes"]
        assert "ms" in clip["tensorrt"]
        if task == "depth":
            assert "original16" not in standard["modes"]
            assert "original16" in clip["modes"]
            assert standard["tensorrt"]["part"] == "single image"
        else:
            assert standard["workload"] == {"frames": 24, "queries": 256}
            assert clip["workload"] == {"frames": 8, "queries": 16}
            assert set(clip["modes"]) >= {"original32", "original16", "portable", "kernels"}
            assert clip["compile"]["part"] == "fixed 8-frame, 16-query tensor module"
            assert clip["tensorrt"]["part"] == "fixed 8-frame, 16-query tensor module"
            assert "ms" in standard["tensorrt"]
            assert standard["tensorrt"]["part"] == "24 frames with 256 queries"


def test_volume_workload_keeps_source_key_and_reports_patch_engine_shape(monkeypatch):
    _, model = registry.resolve_any("totalsegmentator-total-3mm")
    engine = {
        "ref": model.id,
        "mode": "tensorrt",
        "input": "200x200",
        "ms": 22,
        "volume_shape": [200, 200, 200],
        "net_input": "2x1x112x112x128",
        "backend_part": "patch network, standalone build",
        "engine_load_s": 0.1,
        "engine_first_call_s": 0.02,
        "first_call_s": 0.03,
        "torch": "2.14.1+cu130",
    }
    monkeypatch.setattr(site_data.g, "_compile_rows", lambda: {(model.id, "tensorrt", "200x200"): engine})
    monkeypatch.setattr(site_data.g, "_oom", set)
    rows = {
        model.id: [
            {"input": "200x200", "mode": "portable", "ms": 60, "scene": "synthetic"},
            {
                "input": "422x255x255",
                "mode": "portable",
                "ms": 100,
                "scene": "CT scan",
                "volume_shape": [422, 255, 255],
                "spacing_mm": [1.0, 0.8, 0.8],
            },
        ]
    }
    site = site_data._speed(model, "volume", rows)
    profiles = {profile["input"]: profile for profile in site["inputs"]}
    assert site["standard"] == site["preferred"] == "200x200"
    assert profiles["200x200"]["workload"] == {"volume_shape": [200, 200, 200], "spacing_mm": [1.5, 1.5, 1.5]}
    assert profiles["200x200"]["tensorrt"]["engine_input"] == "2x1x112x112x128"
    assert profiles["200x200"]["tensorrt"]["part"] == "patch network, standalone build"
    assert profiles["200x200"]["tensorrt"]["engine_load_s"] == 0.1
    assert profiles["200x200"]["tensorrt"]["first_call_s"] == 0.02
    assert profiles["200x200"]["tensorrt"]["torch"] == "2.14.1+cu130"
    assert profiles["422x255x255"]["workload"] == {"volume_shape": [422, 255, 255], "spacing_mm": [1.0, 0.8, 0.8]}


def test_depth_video_profile_reports_recorded_frame_count():
    rows = [{"frames": 32, "scene": "Synthetic 32-frame clip, decoded CPU RGB"}]
    assert site_data._speed_workload("video-depth-anything-s", "depth", "32x994x1482", rows) == {"frames": 32}
    assert site_data._speed_workload("video-depth-anything-s", "depth", "994x1482", [{}]) == {}


def test_workload_inference_is_limited_to_known_legacy_inputs():
    assert site_data._speed_workload("other-tracker", "tracking", "480x854", [{}]) == {}
    assert site_data._speed_workload("locotrack-b", "tracking", "480x854", [{"scene": "TAP-Vid DAVIS"}]) == {}
    assert site_data._speed_workload("locotrack-b", "tracking", "480x854", [{"frames": 16}]) == {"frames": 16}
    assert site_data._speed_workload("locotrack-s", "tracking", "8x480x854", [{}]) == {"frames": 8, "queries": 16}


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
        {
            "ref": "model",
            "mode": "tensorrt",
            "input": "128x128",
            "ms": 25,
            "standalone_build": True,
            "nvml_peak_mb": 140,
        },
        {
            "ref": "model",
            "mode": "compile",
            "input": "128x128",
            "fails": "run: OutOfMemoryError: CUDA out of memory. Tried to allocate 1.96 GiB.",
        },
        {
            "ref": "model",
            "mode": "tensorrt",
            "input": "96x96",
            "standalone_build": True,
            "fails": "TensorRT: label validation failed",
        },
    ]
    (data / "speed_compile.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
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
    process = sizes["128x128"]["tensorrt"]
    assert process["process_peak_mb"] == 140
    assert process["peak_alloc_mb"] is None
    assert sizes["128x128"]["compile"] == {"fails": "CUDA out of memory (requested 1.96 GiB)", "oom": True}
    assert g._trt_peak(compiled["model", "tensorrt", "128x128"]) == "140 process"
    assert "32x32" not in sizes
    with_portable = {"model": [{"input": "32x32", "mode": "portable", "ms": 20}]}
    failed = site_data._speed(variant, "depth", with_portable)["inputs"][0]
    assert failed["tensorrt"] == {"fails": "export needs validation", "planned": True}
    with_portable["model"].append({"input": "96x96", "mode": "portable", "ms": 30})
    profiles = site_data._speed(variant, "depth", with_portable)["inputs"]
    failure = next(row for row in profiles if row["input"] == "96x96")
    assert failure["tensorrt"] == {"fails": "TensorRT: label validation failed"}
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
            "mode": "upstream-fp32-strict",
            "tier": "upstream32strict",
            "dtype": "float32",
            "metrics": {"rel_absrel": 0.11},
        },
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
        "original32strict": 0.11,
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


def test_quality_labels_error_direction(monkeypatch):
    monkeypatch.setattr(site_data.g, "_overlap", lambda *_: None)
    monkeypatch.setattr(site_data, "eval_version", lambda *_: "test")
    cases = {
        ("depth", "nyu-prompt-every4"): {"met_absrel": True, "met_d1": False},
        ("depth", "nyu-normals"): {"ang_mean": True, "ang_median": True, "a11.25": False, "a30": False},
        ("matting", "dis-vd"): {"mae": True, "max_fbeta": False},
        ("matting", "aim500"): {"sad": True, "mse": True},
        ("matting", "youtubematte-512x288-every2"): {"mad": True, "mse": True, "dtssd": True},
        ("pose", "lapa-test-landmarks-gtbox"): {"nme_pct": True},
        ("volume", "totalseg-featparity-8"): {"cos_min": False, "rel_l2": True},
    }
    for (task, dataset), expected in cases.items():
        row = {"dataset": dataset, "mode": "portable", "dtype": "float16", "metrics": dict.fromkeys(expected, 0.25)}
        (quality,) = site_data._quality(SimpleNamespace(task=task), SimpleNamespace(id="model"), {"model": [row]})
        assert {metric["key"]: metric["lower"] for metric in quality["metrics"]} == expected


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
    assert "same measurement" in modes["portable32"]["note"].casefold()


def test_stream_rows_preserve_explicit_settings_and_oom(tmp_path, monkeypatch):
    data = tmp_path / "docs" / "data"
    data.mkdir(parents=True)
    rows = [
        {"ref": "large", "options": {"stream": False}, "status": "oom"},
        {"ref": "large", "options": {"stream": True}, "ms": 18.7, "peak_alloc_mb": 400},
        {"ref": "large", "options": {"quant": "nf4", "stream": True}, "ms": 11.2},
        {"ref": "other", "ms": 5.6},
    ]
    (data / "speed_options.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(site_data.g, "ROOT", tmp_path)
    assert site_data.g.stream_rows.__wrapped__() == {"large": rows[:2]}


def test_speed_scene_normalizes_only_known_synthetic_fixtures():
    def key(scene, **kw):
        return g._speed_scene({"scene": scene, **kw}).get("scene_key")

    for alias in (
        " synthetic ",
        "Synthetic (fastforks bench)",
        "synthetic (fastforks bench), one point prompt",
        "synthetic (fastforks bench), two frames",
        "synthetic (fastforks bench), 8 views",
    ):
        assert key(alias) == "synthetic"
    assert key("synthetic (fastforks bench), 9 views", ref="mapanything-apache") == "synthetic"
    assert key("synthetic (fastforks bench), 9 views", ref="pi3") != "synthetic"
    assert key("synthetic (fastforks bench), 8 views", ref="pi3", views=4) != "synthetic"
    assert key("synthetic randn, seed 0") != "synthetic"
    assert key(None) is None
    slow = {"scene": "synthetic (fastforks bench)", "ms": 10}
    fast = {"scene": "synthetic", "ms": 2}
    assert g._ratio(slow, fast) == "5.0x"
    assert g._ratio(slow, fast | {"scene": "Middlebury 2014 trainingH Motorcycle"}) == g.DASH
    assert g._ratio(slow, {"ms": 2}) == "5.0x"


def test_ratio_uses_known_timing_boundaries_and_preserves_legacy_rows():
    original = {
        "ref": "totalsegmentator-total-3mm",
        "input": "200x200",
        "mode": "upstream",
        "ms": 20,
        "scene": "synthetic (fastforks bench)",
        "note": "in-memory NIfTI (CPU numpy in, labels out; weights load per call as upstream)",
    }
    resident = {
        "ref": original["ref"],
        "input": original["input"],
        "mode": "kernels",
        "ms": 2,
        "scene": "synthetic",
        "note": "GPU-resident input and output.",
    }
    legacy_original_scope = "cpu-ct-to-cpu-labels-per-call-load"
    assert g._speed_scope(original)["comparison_scope"] == legacy_original_scope
    assert g._ratio(original, resident) == g.DASH
    assert g._ratio(original, resident | {"note": ""}) == "10.0x"
    host = "cpu-ct-to-cpu-labels"
    assert g._ratio(original, resident | {"comparison_scope": host}) == g.DASH
    assert g._ratio(original | {"comparison_scope": host}, resident | {"comparison_scope": host}) == "10.0x"
    assert g._ratio(original | {"comparison_scope": host}, resident) == g.DASH


def test_selected_backend_timings_and_notes_survive_scope_mismatch(monkeypatch):
    card, variant = registry.resolve_any("totalsegmentator-v3-total-3mm")
    data = {
        "ref": variant.id,
        "task": card.task,
        "input": "200x200",
        "scene": "synthetic",
        "volume_shape": [200, 200, 200],
        "spacing_mm": [1.5, 1.5, 1.5],
    }
    original = data | {
        "mode": "upstream",
        "ms": 20,
        "dtype": "float16",
        "comparison_scope": "cpu-ct-to-cpu-labels",
    }
    portable = data | {
        "mode": "portable",
        "ms": 8,
        "note": "GPU-resident input and output.",
        "peak_alloc_mb": 100,
        "peak_reserved_mb": 120,
    }
    kernels = portable | {"mode": "kernels", "ms": 4}
    compiled = data | {
        "mode": "compile",
        "ms": 5,
        "backend_part": "compiled patch network inside the full sliding-window call",
    }
    engine = data | {
        "mode": "tensorrt",
        "ms": 6,
        "comparison_scope": "fixed-patch-logits",
        "standalone_build": True,
        "compare_to_original": False,
        "note": "Strict FP32. FP16 exceeded the validation limit.",
    }
    monkeypatch.setattr(site_data.g, "_oom", set)
    monkeypatch.setattr(
        site_data.g,
        "_compile_rows",
        lambda: {(variant.id, mode, data["input"]): row for mode, row in (("compile", compiled), ("tensorrt", engine))},
    )
    source = {variant.id: [original, portable, kernels]}
    result = site_data._speed(variant, card.task, source)["inputs"][0]
    assert result["mixed_scopes"]
    assert result["modes"]["original16"]["ms"] == 20
    assert result["compile"]["comparison_scope"] == result["modes"]["kernels"]["comparison_scope"]
    assert result["tensorrt"]["ms"] == 6
    assert result["tensorrt"]["compare_to_original"] is False
    assert result["tensorrt"]["note"] == engine["note"]
    assert g._ratio(original, compiled) == g.DASH
    assert g._ratio(portable, kernels) == "2.0x"
    overview = {row["id"]: row for row in g._overview_rows(card.task, {}, source, {})}[variant.id]
    assert overview["original"] == 20
    assert overview["speedup"] == (2.0, "portable")


def test_legacy_native_edges_and_parseq_output_boundaries():
    rgb = {"task": "edges", "input": "1080x1920", "scene": "synthetic"}
    original = rgb | {
        "mode": "upstream",
        "ms": 30,
        "note": "huggingface/controlnet_aux@b1aac318b15c hed, host OpenCV resize, float post-processing",
    }
    trt = rgb | {"mode": "tensorrt", "ms": 5}
    assert g._ratio(original, trt) == g.DASH
    assert g._ratio(original, trt | {"comparison_scope": "cuda-rgb-to-cpu-edge-map"}) == "6.0x"
    assert g._ratio(original, trt | {"input": "720x1280"}) == "6.0x"

    word = {"ref": "parseq", "input": "32x128", "scene": "synthetic"}
    original = word | {"mode": "upstream", "ms": 15}
    assert g._ratio(original, word | {"mode": "kernels", "ms": 3}) == "5.0x"
    assert g._ratio(original, word | {"mode": "tensorrt", "ms": 1.5}) == g.DASH


def test_speed_scene_uses_visible_rows_and_retains_conflicting_timings(monkeypatch):
    card, variant = registry.resolve_any("pi3")
    size = "8x336x518"
    bench = "synthetic (fastforks bench)"
    scene = "Middlebury 2014 trainingH Motorcycle"
    compiled = {"ref": variant.id, "mode": "compile", "input": size, "scene": bench, "ms": 752}
    engine = {"ref": variant.id, "mode": "tensorrt", "input": size, "scene": scene, "ms": 691}
    monkeypatch.setattr(site_data.g, "_oom", set)
    monkeypatch.setattr(
        site_data.g,
        "_compile_rows",
        lambda: {
            (variant.id, "compile", size): compiled,
            (variant.id, "tensorrt", size): engine,
        },
    )
    source = {
        variant.id: [
            {"input": size, "mode": "portable", "scene": scene, "ms": 820},
            {"input": size, "mode": "portable", "scene": "synthetic", "ms": 826},
            {"input": size, "mode": "upstream", "scene": bench, "dtype": "bfloat16", "ms": 1057},
        ]
    }
    (profile,) = site_data._speed(variant, card.task, source)["inputs"]
    assert profile["scene"] == "Mixed scenes"
    assert profile["mixed_scenes"]
    assert profile["modes"]["original16"]["scene_key"] == profile["modes"]["portable"]["scene_key"] == "synthetic"
    assert profile["modes"]["portable"]["ms"] == 826
    assert profile["compile"]["scene_key"] == "synthetic"
    assert profile["tensorrt"]["scene"] == scene
    assert profile["tensorrt"]["ms"] == engine["ms"]

    monkeypatch.setattr(site_data.g, "_compile_rows", lambda: {(variant.id, "compile", size): compiled})
    (profile,) = site_data._speed(variant, card.task, source)["inputs"]
    assert "mixed_scenes" not in profile
    assert profile["scene"] == "synthetic"
