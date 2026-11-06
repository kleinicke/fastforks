"""`tools/stage_hf.py cards --check-live` diffing and `tools/coverage_report.py` (offline, nothing is fetched)."""

from collections import Counter

import coverage_report
import stage_hf

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
