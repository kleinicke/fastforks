"""`tools/stage_hf.py cards --check-live` diffing and `tools/coverage_report.py` (offline, nothing is fetched)."""

import types
from collections import Counter

import coverage_report
import stage_hf
from fastforks._core import registry

SRC = "<!-- staged from fastforks abc1234 (2026-11-05) -->"


def test_staged_cards_carry_the_source_commit():
    cards = stage_hf.staged_cards()
    assert cards
    assert all(t.rstrip().endswith("-->") and "staged from fastforks " in t.splitlines()[-1] for t in cards.values())


def test_strip_source_ignores_commit_and_whitespace():
    assert stage_hf.strip_source(f"# a\n\ntext\n\n{SRC}\n") == "# a\n\ntext"
    assert stage_hf.strip_source("# a\r\n\r\ntext\r\n") == "# a\n\ntext"


def test_differing_lists_changed_and_missing_cards():
    staged = {"fastforks/a": f"# A\n\n{SRC}\n", "fastforks/b": f"# B new\n\n{SRC}\n", "fastforks/c": f"# C\n\n{SRC}\n"}
    live = {"fastforks/a": "# A\n\n<!-- staged from fastforks 0000000 (2026-01-01) -->\n", "fastforks/b": "# B old\n"}
    assert stage_hf.differing(staged, live.get) == ["fastforks/b", "fastforks/c"]
    assert stage_hf.differing({"fastforks/a": staged["fastforks/a"]}, live.get) == []


def test_kind_and_ref_parsing():
    assert coverage_report.kind_of("test_gpu", "kernels-s2m2-s@1") == "gpu-kernels"
    assert coverage_report.kind_of("test_gpu", "portable-s2m2-s@1") == "gpu-portable"
    assert coverage_report.kind_of("test_onnx_depth", "x@1") == "onnx"
    assert coverage_report.kind_of("test_unrelated", "") is None
    assert coverage_report.ref_of("kernels-s2m2-s@1", {"s2m2-s@1", "s2m2"}) == "s2m2-s@1"


def test_summarize_and_render():
    fams = {"s2m2": ("s2m2", {"s2m2-s@1"}, False), "dino": ("dino", {"dino-b@1"}, True)}
    rows = [
        ("test_cpu_portable", "s2m2-s@1", "passed"),
        ("test_onnx", "s2m2-s@1", "skipped"),
        ("test_cpu_portable", "dino-b@1", "failed"),
    ]
    summary = coverage_report.summarize(rows, fams)
    assert summary["s2m2"]["cpu-portable"] == Counter(passed=1)
    assert summary["dino"]["cpu-portable"] == Counter(failed=1)
    text = coverage_report.render(summary, fams)
    assert "| dino | yes | 1f |" in text
    assert "- onnx: 2 of 2" in text


def test_stage_hf_maps_cover_every_task():
    for name in ("_PIPELINE", "_TASK_TAGS", "_SNIPPET"):
        assert set(registry.TASKS) <= set(getattr(stage_hf, name)), name


def test_fetch_live_reports_unauthorized_and_missing_repos_as_none(monkeypatch):
    import urllib.error  # noqa: PLC0415

    for code in (401, 404):

        def boom(*_a, _c=code, **_k):
            raise urllib.error.HTTPError("u", _c, "x", None, None)  # type: ignore[arg-type]  # noqa: EM101

        monkeypatch.setattr(stage_hf.urllib.request, "urlopen", boom)
        assert stage_hf.fetch_live("fastforks/nope") is None


def test_find_golden_takes_the_first_source_with_the_recorded_hash(tmp_path):
    import hashlib  # noqa: PLC0415

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "m@1.npz").write_bytes(b"stale")
    (b / "m@1.npz").write_bytes(b"good")
    good = hashlib.sha256(b"good").hexdigest()
    assert stage_hf._find_golden([a, b], "m@1.npz", good) == b / "m@1.npz"
    assert stage_hf._find_golden([a], "m@1.npz", good) is None
    assert stage_hf._find_golden([a, b], "x@1.npz", good) is None


def test_stage_goldens_copies_a_per_os_golden(tmp_path, monkeypatch):
    """A `[refs."<ref>".windows]` golden is staged next to the ref's own file, each under its published name."""
    import hashlib  # noqa: PLC0415

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    (src / "m@1.npz").write_bytes(b"linux")
    (src / "m@1.windows.npz").write_bytes(b"windows")
    entry = {
        "sha256": hashlib.sha256(b"linux").hexdigest(),
        "upstream": "u",
        "windows": {"sha256": hashlib.sha256(b"windows").hexdigest(), "upstream": "u"},
    }
    card = types.SimpleNamespace(
        family="fam",
        licence=types.SimpleNamespace(code="MIT"),
        status=lambda _v: types.SimpleNamespace(weights="ok"),
    )
    monkeypatch.setattr(stage_hf.registry, "cards", lambda _t: {"m": (card, types.SimpleNamespace(ref="m@1"))})
    monkeypatch.setattr(stage_hf.store, "load_all", lambda _t: {"fam": {"refs": {"m@1": entry}}})
    monkeypatch.setattr(stage_hf.registry, "TASKS", ("stereo",))
    rows = stage_hf.stage_goldens([src], out)
    assert (out / "m@1.npz").read_bytes() == b"linux"
    assert (out / "m@1.windows.npz").read_bytes() == b"windows"
    assert len(rows) == 1
    (src / "m@1.windows.npz").write_bytes(b"stale")
    assert stage_hf.stage_goldens([src], tmp_path / "out2", skip_missing=True) == []
