"""Per model family: which kinds of test ran, skipped or failed in a pytest run (CPU only, reads a junit XML).

    pytest -m "not gpu" --junitxml=junit.xml; uv run python tools/coverage_report.py junit.xml

Kinds: golden parity per backend (`cpu-portable`, `gpu-portable`, `gpu-kernels`, `mps`, `gpu-module`),
`kernels-validated`, `onnx`, `compile`. A cell is `<passed>p <skipped>s <failed>f`; `-` means no such test was
collected or it was deselected (e.g. `-m "not gpu"` drops every GPU kind).
Families that need timm or transformers are marked `vision`.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

KINDS = ("cpu-portable", "gpu-portable", "gpu-kernels", "mps", "gpu-module", "kernels-validated", "onnx", "compile")
MAX_LISTED = 12
VISION_LIBS = frozenset({"timm", "transformers"})


def kind_of(test: str, param: str) -> str | None:
    """The kind of a test from its function name (and parameter id for the GPU backend), or None if not per-model."""
    if test == "test_cpu_portable":
        return "cpu-portable"
    if test == "test_gpu":
        return "gpu-kernels" if "kernels" in param.split("-") else "gpu-portable"
    if test in ("test_mps", "test_gpu_module", "test_kernels_validated"):
        return {"test_mps": "mps", "test_gpu_module": "gpu-module", "test_kernels_validated": "kernels-validated"}[test]
    if test.startswith("test_onnx"):
        return "onnx"
    if test in ("test_module_compile_export", "test_model_compile_equals_eager"):
        return "compile"
    return None


def ref_of(param: str, refs: set[str]) -> str | None:
    """The model ref in a parameter id such as `s2m2-s@1` or `portable-s2m2-s@1`."""
    parts = param.split("-")
    found = ["-".join(parts[i:j]) for i in range(len(parts)) for j in range(i + 1, len(parts) + 1)]
    return max((r for r in found if r in refs), key=len, default=None)


def read_junit(path: Path) -> list[tuple[str, str, str]]:
    """`(test function, parameter id, outcome)`; outcome is passed, skipped or failed (an xfail counts as skipped)."""
    rows = []
    for case in ET.parse(path).getroot().iter("testcase"):  # noqa: S314
        m = re.fullmatch(r"([^\[]+)(?:\[(.*)\])?", case.get("name", ""))
        if m is None:
            continue
        bad = case.find("failure") is not None or case.find("error") is not None
        rows.append(
            (m[1], m[2] or "", "failed" if bad else "skipped" if case.find("skipped") is not None else "passed")
        )
    return rows


def summarize(
    rows: list[tuple[str, str, str]], families: dict[str, tuple[str, set[str], bool]]
) -> dict[str, dict[str, Counter]]:
    """`family -> kind -> Counter(outcome)`; `families`: see `registry_families`."""
    owner = {r: fam for fam, (_, refs, _) in families.items() for r in refs}
    out: dict[str, dict[str, Counter]] = {fam: defaultdict(Counter) for fam in families}
    for test, param, outcome in rows:
        kind = kind_of(test, param)
        ref = ref_of(param, set(owner)) if kind else None
        if ref is not None and kind is not None:
            out[owner[ref]][kind][outcome] += 1
    return out


def render(summary: dict[str, dict[str, Counter]], families: dict[str, tuple[str, set[str], bool]]) -> str:
    def cell(c: Counter) -> str:
        return " ".join(f"{c[o]}{o[0]}" for o in ("passed", "skipped", "failed") if c[o]) or "-"

    lines = ["| family | vision | " + " | ".join(KINDS) + " |", "|---|---|" + "---|" * len(KINDS)]
    gaps: dict[str, list[str]] = {k: [] for k in KINDS}
    for fam in sorted(summary):
        per = summary[fam]
        lines.append(
            f"| {fam} | {'yes' if families[fam][2] else ''} | " + " | ".join(cell(per[k]) for k in KINDS) + " |"
        )
        for k in KINDS:
            if not per[k]["passed"]:
                gaps[k].append(fam)
    lines += ["", "Families with no passing test per kind:"]
    lines += [
        f"- {k}: {len(g)} of {len(summary)}" + (f" ({', '.join(g)})" if 0 < len(g) <= MAX_LISTED else "")
        for k, g in gaps.items()
    ]
    return "\n".join(lines) + "\n"


def registry_families() -> dict[str, tuple[str, set[str], bool]]:
    from fastforks._core import registry  # noqa: PLC0415 - keeps `--help` and the tests of the parser import-light

    fams: dict[str, tuple[str, set[str], bool]] = {}
    for task in registry.TASKS:
        for card, v in registry.cards(task).values():
            _, refs, _ = fams.setdefault(card.family, (card.family, set(), bool(VISION_LIBS & set(card.requires))))
            refs.add(v.ref)
    return fams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("junit", type=Path, help="pytest --junitxml output")
    a = ap.parse_args()
    if not a.junit.is_file():
        sys.exit(f"missing: {a.junit}")
    fams = registry_families()
    sys.stdout.write(render(summarize(read_junit(a.junit), fams), fams))


if __name__ == "__main__":
    main()
