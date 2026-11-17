"""Accuracy rows keep their measured precision on the website."""

from types import SimpleNamespace

import site_data


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


def test_task_without_accuracy_protocol_has_no_quality_rows():
    assert site_data._quality(SimpleNamespace(task="edges"), SimpleNamespace(id="model"), {}) == []


def test_default_float32_portable_is_also_labelled_as_fp32(monkeypatch):
    monkeypatch.setattr(site_data.g, "_overlap", lambda *_: None)
    monkeypatch.setattr(site_data, "eval_version", lambda *_: "test")
    row = {"dataset": "nyu", "mode": "portable", "dtype": "float32", "metrics": {"rel_absrel": 0.2}}
    modes = site_data._quality(SimpleNamespace(task="depth"), SimpleNamespace(id="model"), {"model": [row]})[0]["modes"]
    assert modes["portable32"]["metrics"] == modes["portable"]["metrics"]
    assert "same measurement" in modes["portable32"]["note"]
