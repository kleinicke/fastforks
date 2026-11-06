import inspect
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import fastforks
import gen_docs
from fastforks import ModelNotFoundError, stereo
from fastforks._core import licences, registry
from fastforks.stereo import geometry

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_model_pages_current():
    subprocess.run([sys.executable, ROOT / "tools" / "gen_docs.py", "--check"], check=True)


def test_public_functions_document_their_args():
    """Every public function and method of `fastforks` / `.stereo` / `.stereo.geometry` with parameters has `Args:`."""
    missing = []
    for mod in (fastforks, stereo, geometry):
        for name in mod.__all__:
            obj = getattr(mod, name)
            members = [(name, obj)] if inspect.isfunction(obj) else []
            if inspect.isclass(obj) and not issubclass(obj, BaseException):
                members += [
                    (f"{name}.{n}", f)
                    for n, f in vars(obj).items()
                    if inspect.isfunction(f) and (not n.startswith("_") or n == "__call__")
                ]
            missing += [
                label
                for label, f in members
                if set(inspect.signature(f).parameters) - {"self", "cls"} and "Args:" not in (inspect.getdoc(f) or "")
            ]
    assert not missing, f"no `Args:` section: {sorted(set(missing))}"


_DOC_FILES = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]
_ID = r"([a-z0-9][\w.\-]*(?:@\d+)?)"
_QUOTED = re.compile(rf"""\b(\w+)\.(?:load|model_card)\(["']{_ID}["']""")
_CLI = re.compile(rf"fastforks (?:bench|info|download|export) {_ID}")
_TABLE = re.compile(rf"^\| .*`fastforks\.(\w+)` \|.*\| `{_ID}` \|$", re.MULTILINE)


def _quoted_model_ids() -> list[tuple[str, str | None, str]]:
    """(where, task or None, id) of every model id quoted in the docs and in the package's docstrings and messages."""
    texts = {p.relative_to(ROOT).as_posix(): p.read_text() for p in _DOC_FILES}
    texts |= {
        p.relative_to(ROOT).as_posix(): p.read_text()
        for p in sorted((ROOT / "src" / "fastforks").rglob("*.py"))
        if "models" not in p.relative_to(ROOT / "src" / "fastforks").parts
    }
    found = []
    for where, text in texts.items():
        found += [(where, task, model_id) for task, model_id in _QUOTED.findall(text)]
        found += [(where, None, model_id) for model_id in _CLI.findall(text)]
        found += [(where, task, model_id) for task, model_id in _TABLE.findall(text)]
    return [(where, task if task in registry.TASKS else None, model_id) for where, task, model_id in found]


# Ids of task docstrings whose models are on a branch that is not merged yet: allowed only while the id is not installed
# (remove the entry when the models land; the id must then resolve). Per id, not per task: a task can get other
# families first (segmentation: Sapiens before SegFormer).
_PENDING: dict[str, str] = {}


def test_quoted_model_ids_resolve():
    """Every model id the README, the docs or a docstring uses exists (or is pending, see `_PENDING`)."""
    quoted = _quoted_model_ids()
    assert len({model_id for _, _, model_id in quoted}) >= 5, "the id patterns no longer match the docs"
    bad = []
    for where, task, model_id in quoted:
        if _PENDING.get(model_id) == task and task is not None and registry.task_of(model_id) != task:
            continue
        try:
            registry.resolve_any(model_id)
        except ModelNotFoundError:
            bad.append(f"{where}: {model_id}")
            continue
        if task is not None and registry.task_of(model_id) != task:
            bad.append(f"{where}: {model_id} is a {registry.task_of(model_id)} model, quoted as {task}")
    assert not bad, f"quoted model ids that do not resolve: {bad}"
    landed = [model_id for model_id, task in _PENDING.items() if registry.task_of(model_id) == task]
    assert not landed, f"{landed}: installed now; drop them from _PENDING"


_FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)
_NEEDS = re.compile(r"^# needs: (.+)$", re.MULTILINE)
_EXTRA = re.compile(r"fastforks-core\[([\w,-]+)\]")


def _extras_of(model_id: str) -> set[str]:
    """The extras a model needs beyond `pip install fastforks-core`: its distribution's (`nc`) and its libraries'."""
    model_id = model_id.split("@", maxsplit=1)[0]
    task = registry.task_of(model_id)
    if task is None:  # not in this tree (unknown ids fail test_quoted_model_ids_resolve)
        return set()
    card, _ = registry.resolve(task, model_id)
    dist = tomllib.loads((ROOT / "src" / "fastforks" / task / "known_models.toml").read_text()).get(model_id)
    return set(card.extras) | ({dist.removeprefix("fastforks-")} if dist else set())


def _core_extras() -> set[str]:
    """Extras of the released core wheel: the pyproject ones and one per dists.toml entry."""
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    dists = tomllib.loads((ROOT / "dists.toml").read_text())
    return {*extras, *(n.removeprefix("fastforks-") for n in dists)}


def test_documented_examples_run_with_the_extras_named_next_to_them():
    """A block that loads a model outside the plain install (`[nc]` code, or timm / torchvision / transformers) names
    those extras on a `# needs: pip install "fastforks-core[nc]"` line; every extra named anywhere in the docs
    exists."""
    problems = []
    known = _core_extras()
    for path in [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = path.read_text()
        where = path.relative_to(ROOT).as_posix()
        unknown = {e for group in _EXTRA.findall(text) for e in group.split(",")} - known
        if unknown:
            problems.append(f"{where}: names extras that do not exist: {sorted(unknown)}")
        for _, block in _FENCE.findall(text):
            ids = {m for _, m in _QUOTED.findall(block)} | set(_CLI.findall(block))
            needs = _NEEDS.search(block)
            named = {e for group in _EXTRA.findall(needs[1]) for e in group.split(",")} if needs else set()
            wanted = set().union(*map(_extras_of, ids))
            if wanted - named:
                problems.append(f"{where}: a block loads {sorted(ids)}, needs {sorted(wanted)}, names {sorted(named)}")
            elif needs and not (named - {"onnx", "tensorrt", "libs"}) <= wanted:
                problems.append(f"{where}: a block names {sorted(named)}, its models need only {sorted(wanted)}")
    assert not problems, "\n".join(problems)


_LOADS = re.compile(rf"""\b\w+\.load\(["']{_ID}["']|fastforks (?:bench|download|export) {_ID}""")


def _nc_weights(model_id: str) -> bool:
    """Whether the model's weights download only with FASTFORKS_NONCOMMERCIAL=1."""
    model_id = model_id.split("@", maxsplit=1)[0]
    task = registry.task_of(model_id)
    if task is None:
        return False
    card, v = registry.resolve(task, model_id)
    return licences.weights_commercial(card.variant_licence(v).weights_licence) == "no"


def test_documented_examples_with_noncommercial_weights_say_so():
    """A block that loads a model with non-commercial weights names FASTFORKS_NONCOMMERCIAL=1: without it a new
    user's first run raises LicenceError."""
    problems = []
    for path in _DOC_FILES:
        for _, block in _FENCE.findall(path.read_text()):
            nc = sorted({m for pair in _LOADS.findall(block) for m in pair if m and _nc_weights(m)})
            if nc and "FASTFORKS_NONCOMMERCIAL=1" not in block:
                where = path.relative_to(ROOT).as_posix()
                problems.append(f"{where}: a block loads {nc} without FASTFORKS_NONCOMMERCIAL=1")
    assert not problems, "\n".join(problems)


def test_every_task_is_in_the_readme_table_and_the_install_extras_are_listed():
    readme = (ROOT / "README.md").read_text()
    for task in registry.TASKS:
        assert f"`fastforks.{task}`" in readme, f"task {task} missing in the README task table"
        assert f"fastforks.{task}" in (ROOT / "docs" / "usage.md").read_text() or task in ("stereo", "depth")
    for extra in _core_extras() - {"kernels"}:
        assert f'fastforks-core[{extra}]"' in readme, f"fastforks-core[{extra}] missing in the README install list"
    assert "pip install fastforks " in readme, "the full install (the fastforks metapackage) missing in the README"


def test_eval_overlap_marked_in_measured_columns():
    """Variants trained on the eval images get a note on that row, unstated ones "possibly ..."."""
    page = (ROOT / "docs" / "models" / "stereo.md").read_text()
    rows: dict[tuple[str, str], str] = {}
    for m in re.finditer(r"^\| `([\w.\-]+)` \| ((?:ETH3D|Middlebury)[^|]*)\|.*\| ([^|]*) \|$", page, re.MULTILINE):
        rows[m.group(1), m.group(2).split()[0]] = m.group(3).strip()
    assert "trained on" in rows["raft-stereo-middlebury", "Middlebury"]
    assert rows["raft-stereo-middlebury", "ETH3D"] == ""
    assert "trained on" in rows["hitnet-eth3d", "ETH3D"]
    assert "trained on" in rows["selective-igev-middlebury-finetune", "Middlebury"]
    assert rows["igev-stereo-eth3d", "ETH3D"] == "possibly in the training data"
    assert rows["aanet-sceneflow", "Middlebury"] == ""


def _headings(task: str) -> set[str]:
    text = (ROOT / "docs" / "models" / f"{task}.md").read_text()
    return {gen_docs._slug(m[2]) for m in re.finditer(r"^(#{1,6}) (.+)$", text, re.MULTILINE)}


def test_overview_lists_every_variant_once_and_links_resolve():
    """Each task of the overview has one row per variant and every row links to a heading of its model page."""
    page = gen_docs.overview()
    for task in registry.TASKS:
        section = page.split(f"## {gen_docs._title(task)}\n", 1)[1].split("\n## ", 1)[0]
        rows = re.findall(r"^\| \[`([^`]+)`\]\(models/(\w+)\.md#([^)]+)\) \|", section, re.MULTILINE)
        ids = [r[0] for r in rows]
        expected = sorted({gen_docs._name(v) for card, _ in registry.cards(task).values() for v in card.variants})
        assert sorted(ids) == expected, task
        assert all(r[1] == task and r[2] in _headings(task) for r in rows), task


def test_overview_numbers_come_from_the_data(monkeypatch):
    """No hand-typed numbers: a changed measurement changes the table, a missing one shows as a dash."""
    base = gen_docs.overview()
    rows = gen_docs._speed_rows()
    orig = [
        r
        for r in rows["raft-stereo-sceneflow"]
        if r["mode"] in ("upstream", "upstream-default") and r["input"] == "994x1482"
    ]
    assert f"{min(r['ms'] for r in orig):.1f} (994x1482)" in base  # the faster original
    changed = {k: [{**r, "ms": 12345.0} if any(r is o for o in orig) else r for r in v] for k, v in rows.items()}
    monkeypatch.setattr(gen_docs, "_speed_rows", lambda: changed)
    assert "12345.0 (994x1482)" in gen_docs.overview()
    monkeypatch.setattr(gen_docs, "_speed_rows", dict)
    empty = gen_docs.overview()
    assert "(994x1482)" not in empty
    assert gen_docs._DASH in empty


def test_overview_picks_follow_the_rules():
    """Fastest = lowest ms of the table; the accuracy picks never name a row marked as trained on / unstated data."""
    fixed, speed = gen_docs._fixed_sizes(), gen_docs._speed_rows()
    for task in ("stereo", "depth"):
        rows = gen_docs._overview_rows(task, fixed, speed, gen_docs._eval_rows(task))
        timed = [r for r in rows if r["ms"] is not None]
        picks = gen_docs._picks(rows)
        assert f"`{min(timed, key=lambda r: r['ms'])['id']}`" in picks[0]
        marked = {r["id"] for r in rows if r["mark"]}
        accuracy = [p for p in picks if p.startswith(("- Most accurate", "- Best trade-off"))]
        assert accuracy
        assert not any(f"`{m}`" in p for p in accuracy for m in marked)
    assert gen_docs._slug("Segment Anything 2.1 (SAM 2.1), image") == "segment-anything-21-sam-21-image"


def test_fixed_size_table_parser():
    fixed = gen_docs._fixed_sizes()
    cell = fixed["raft-stereo-sceneflow", "kernels"]["1080x1920"]
    assert isinstance(cell, tuple)
    assert cell[0] > 0
    assert set(next(iter(fixed.values()))) >= {"480x640", "1080x1920"}
    assert "224x224" in fixed["resnet50.tv2_in1k", "kernels"]


def test_every_family_has_a_plain_description():
    """docs/data/plain.toml holds one entry per family (the model pages fall back to the technical summary)."""
    plain = gen_docs._plain()
    families = {c.family for t in registry.TASKS for c, _ in registry.cards(t).values()}
    assert sorted(families - plain.keys()) == []
    assert sorted(plain.keys() - families) == []
    assert all(0 < len(text) < 300 for text in plain.values())


def test_summary_table_matches_the_speed_tables():
    """The summary's kernels ms is the kernels row of the speed table at the task's one summary input."""
    speed = gen_docs._speed_rows()
    row = next(r for r in speed["crestereo-eth3d"] if r["mode"] == "kernels" and r["input"] == "994x1482")
    page = (ROOT / "docs" / "models" / "stereo.md").read_text()
    assert f"| [`crestereo-eth3d`](#crestereo) | yes | 1.39 \u2020 | {row['ms']}, " in page


def test_summary_memory_ratio_uses_the_smaller_original():
    """Kernels peak reserved over the smaller original peak; nothing without a measured original."""
    fp32, fp16 = {"peak_reserved_mb": 624}, {"peak_reserved_mb": 846}
    assert gen_docs._mem_ratio(388, fp16, fp32) == ", 0.62x"
    assert gen_docs._mem_ratio(900, None, fp32) == ", 1.4x"
    assert gen_docs._mem_ratio(388, None, {"ms": 1.0}) == ""
    assert gen_docs._mem_ratio(gen_docs._DASH, fp16, fp32) == ""


def test_plain_licence_words_match_the_cards():
    """'commercially usable' needs all variants commercial yes, 'non-commercial' at least one no."""
    status: dict[str, set[str]] = {}
    for t in registry.TASKS:
        for card, v in registry.cards(t).values():
            status.setdefault(card.family, set()).add(card.status(v).commercial)
    for family, text in gen_docs._plain().items():
        low = text.lower()
        if "commercially usable" in low:
            assert status[family] == {"yes"}, family
        if "non-commercial" in low:
            assert "no" in status[family], family


def test_mps_section_names_every_metal_kernel_family():
    """R2-4: the Apple silicon section lists every family whose card has `mps_kernels`."""
    text = (ROOT / "docs" / "usage.md").read_text()
    section = text.split("## Apple silicon (MPS)", 1)[1].split("\n## ", 1)[0]
    families = {
        card.name for task in registry.TASKS for card, _ in registry.cards(task).values() if card.mps_kernel_libs
    }
    assert len(families) >= 19
    missing = sorted(f for f in families if f not in section)
    assert not missing, f"docs/usage.md Apple silicon section does not name {missing}"


def test_measured_date_span():
    """A campaign that crosses midnight is one row set; the card shows the date span."""
    assert gen_docs._span({"2026-11-05"}) == "2026-11-05"
    assert gen_docs._span({"2026-11-06", "2026-11-05"}) == "2026-11-05 to 2026-11-06"


def test_measured_cells_read_nested_metrics_and_optional_ones():
    """Rows of the COCO / ADE20K evals nest their metrics; an optional metric shows only where the row has it."""
    box = {"ref": "x", "mode": "portable", "dataset": "coco-val2017-every10", "n": 500, "metrics": {"box_map": 41.5}}
    mask = box | {"metrics": {"box_map": 42.13, "mask_map": 38.67}}
    assert gen_docs._accuracy("detection", {("portable", box["dataset"]): box}, "portable", box["dataset"]) == "41.50"
    assert gen_docs._accuracy("detection", {("portable", box["dataset"]): mask}, "portable", box["dataset"]) == (
        "42.13 / 38.67"
    )
    assert gen_docs._suffixes("detection", box["dataset"], [box]) == []
    assert gen_docs._suffixes("detection", box["dataset"], [mask]) == [" / mask mAP"]
    for (task, dataset), metrics in [(k, v) for k, v in gen_docs._METRICS.items() if isinstance(k, tuple)]:
        assert dataset in gen_docs._DATASETS[task], (task, dataset, metrics)
