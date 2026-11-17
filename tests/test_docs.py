import ast
import importlib
import inspect
import json
import re
import subprocess
import sys
import textwrap
import tomllib
import types
import typing
from pathlib import Path

import pytest

import fastforks
import gen_docs
from fastforks import ModelNotFoundError, stereo
from fastforks._core import licences, registry
from fastforks._core.api import is_path
from fastforks.stereo import geometry

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_model_pages_current():
    subprocess.run([sys.executable, ROOT / "tools" / "gen_docs.py", "--check"], check=True)


def test_measured_table_keeps_portable_precisions_separate():
    card, variant = registry.resolve("detection", "grounding-dino-tiny")
    common = {
        "ref": variant.id,
        "dataset": "coco-val2017-every10",
        "mode": "portable",
        "gpu": "test GPU",
        "torch": "test torch",
        "commit": "test",
        "date": "2026-11-17",
        "n": 500,
    }
    rows = [
        {**common, "tier": "portable", "dtype": "float16", "metrics": {"box_map": 51.02}},
        {
            **common,
            "tier": "portable32",
            "dtype": "float32",
            "metrics": {"box_map": 51.12},
            "date": "2026-11-18",
            "commit": "fp32",
        },
    ]
    table = "\n".join(gen_docs._measured("detection", card, {variant.id: rows}))
    assert "| portable | portable fp32 |" in table
    assert "| 51.02 | 51.12 |" in table
    assert "portable32: test GPU, torch test torch, fastforks fp32, 2026-11-18" in table
    assert gen_docs._eval_mode({"mode": "upstream", "tier": "upstreamhalf"}) == "upstreamhalf"
    assert gen_docs._eval_mode({"mode": "upstream", "dtype": "float16"}) == "upstream-fp16"


def test_overview_prefers_default_portable_to_portable32():
    rows = [
        {"mode": "portable", "dataset": "kitti2015-train", "epe": 1.0},
        {"mode": "portable", "tier": "portable32", "dataset": "kitti2015-train", "epe": 2.0},
    ]
    overview = gen_docs._overview_rows("flow", {}, {}, {"gmflow-scale1-things": rows})
    assert next(r["error"] for r in overview if r["id"] == "gmflow-scale1-things") == 1.0


def test_nnfoundation_engine_labels_preprocessing_scope():
    assert (
        gen_docs._engine_part("nnfoundation-cnn") == " (encoder on a normalized, padded volume only; memory estimated)"
    )


def test_anyline_engine_labels_memory_estimate():
    assert gen_docs._engine_part("anyline") == " (full line map only; memory estimated)"


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
        if is_path(model_id):  # `restoration.load("4x-UltraSharp.pth")`: a checkpoint file, not an id
            continue
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
    assert "trained on" in rows["igev-stereo-eth3d", "ETH3D"]  # the card lists ETH3D (submission model)
    assert rows["bridgedepth-rvc-pretrain", "ETH3D"] == "possibly in the training data"
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
    assert ("resnet50.tv2_in1k", "kernels") not in fixed  # native-size models are in the speed tables


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
    assert f"| [`crestereo-eth3d`](#crestereo) | yes | 1.39 \u2020 | {gen_docs._ms_str(row['ms'])}, " in page


def test_summary_times_are_rounded_on_every_page():
    """Every model page's summary prints its kernels ms to 3 significant digits, like the speed tables."""
    checked = 0
    for page in sorted((ROOT / "docs" / "models").glob("*.md")):
        table = page.read_text().split("kernels time (ms)", 1)[-1].split("\n\n", 1)[0]
        for line in table.splitlines()[2:]:
            ms = line.split("|")[-3].strip().split(",")[0].split(" ")[0]
            if ms in (gen_docs._DASH, "OOM", "no"):  # "no kernels": the family has no kernels backend
                continue
            checked += 1
            assert ms == gen_docs._ms_str(float(ms)), (page.name, ms)
    assert checked > 50


def test_one_speed_table_per_task_and_scene():
    """benchmarks.md: one speed table per (task, scene); models at different inputs get an input column."""
    text = (ROOT / "docs" / "benchmarks.md").read_text()
    block = text.split(gen_docs._TABLE_BEGIN, 1)[1].split("### Startup and first call")[0]
    titles = re.findall(r"^### (.+)$", block, re.MULTILINE)
    keys = [tuple(t.split(", ")[:2]) for t in titles]
    assert len(keys) == len(set(keys)), titles
    multiview = block.split("### Multiview, synthetic\n", 1)[1].split("\n### ", 1)[0]
    assert "| model | input |" in multiview
    assert "| patchmatchnet | 1536x1024 |" in multiview


def test_flow_benchmark_keeps_standard_size_when_one_model_ooms(monkeypatch):
    """An OOM model's smaller timed input must not shrink the other models' comparison size."""

    def row(ref, size):
        return {"ref": ref, "input": size, "mode": "portable", "ms": 1.0, "scene": "synthetic"}

    monkeypatch.setattr(
        gen_docs,
        "_speed_rows",
        lambda: {
            "gmflow-scale1-things": [
                row("gmflow-scale1-things", "1080x1920"),
                row("gmflow-scale1-things", "375x1242"),
            ],
            "flowformer-small-chairs": [row("flowformer-small-chairs", "375x1242")],
        },
    )
    block = gen_docs._bench_block()
    flow = block.split("### Flow, synthetic\n", 1)[1].split("\n### ", 1)[0]
    assert "| gmflow-scale1-things | 1080x1920 |" in flow
    assert "| flowformer-small-chairs | 375x1242 |" in flow


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


def test_a_failing_original_is_a_cell_and_a_failing_fastforks_tier_is_not():
    """An original that raises on the eval data (PatchMatchNet fp16 on DTU) is a result; our own tiers must not fail."""
    d = "dtu-scans1-9-24-33-75-view24"
    err = {"ref": "x", "dataset": d, "error": "call: _LinAlgError: linalg.inv: (Batch element 0): singular"}
    assert gen_docs._accuracy("multiview", {("upstream-fp16", d): err}, "upstream-fp16", d) == (
        "fails (LinAlgError: linalg.inv)"
    )
    with pytest.raises(ValueError, match="no cell"):
        gen_docs._accuracy("multiview", {("portable", d): err}, "portable", d)


def test_original_eval_cells_come_from_their_tier():
    """The row a docs cell shows (the last per model, mode and dataset): original fp32 from the fp32 tier, original
    fp16 from an fp16 tier (an fp16 row labelled `upstream` overwrote the fp32 cell of moge-2-vitb-normal)."""
    want = {
        "upstream": ("upstream",),
        "upstream-fp16": ("upstream16", "upstreamac"),
        "upstreamhalf": ("upstreamhalf", "upstreambf16"),
    }
    for path in (ROOT / "docs" / "data").glob("*_eval.jsonl"):
        last = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            mode = gen_docs._eval_mode(r) if "mode" in r else None
            if mode in want:
                last[r["ref"], mode, r.get("dataset")] = r
        bad = []
        for key, row in last.items():
            tiers = want[key[1]]
            if key[1] == "upstreamhalf" and row.get("dtype") == "bfloat16":
                tiers = (*tiers, "upstream")  # some originals use BF16 by default
            if "tier" in row and row["tier"] not in tiers:
                bad.append((*key, row["tier"]))
        assert not bad, f"{path.name}: {bad}"


# Python examples: model ids, load keywords, call / method keywords and result attributes, read with `ast` (nothing
# runs). Docstring examples count too (the fenced blocks in the package's module docstrings).
_PY_FENCE = re.compile(r"^( *)```(?:python|py)\n(.*?)^\1```", re.MULTILINE | re.DOTALL)
_PATH_SUFFIXES = (".pth", ".pt", ".safetensors", ".ckpt", ".bin", ".onnx")


def _python_blocks() -> list[tuple[str, int, str]]:
    """(file, first line, code) of every ```python block of the docs and of the public modules' docstrings."""
    src = sorted((ROOT / "src" / "fastforks").rglob("*.py"))
    public = [p for p in src if "models" not in p.parts and (p.stem in {"__init__", "_api"} or p.stem[0] != "_")]
    blocks = []
    for path in [*_DOC_FILES, *sorted((ROOT / "docs" / "models").glob("*.md")), *public]:
        text = path.read_text()
        blocks += [
            (path.relative_to(ROOT).as_posix(), text.count("\n", 0, m.start()) + 2, textwrap.dedent(m[2]))
            for m in _PY_FENCE.finditer(text)
        ]
    return blocks


def _model_classes(task: str) -> list[type]:
    mod = importlib.import_module(f"fastforks.{task}")
    found = [getattr(mod, n) for n in mod.__all__]
    return [c for c in found if inspect.isclass(c) and issubclass(c, fastforks.Model)] or [fastforks.Model]


def _result_classes(fn: object, owner: object) -> list[type]:
    """The fastforks classes `fn` returns: a class itself, else the names in its return annotation (resolved in the
    modules of `fn` and `owner`); empty if unresolvable."""
    if inspect.isclass(fn):
        return [fn]
    try:
        ret = inspect.signature(fn).return_annotation
    except (TypeError, ValueError):
        return []
    names = re.findall(r"[A-Za-z_]\w*", ret) if isinstance(ret, str) else [getattr(ret, "__name__", "")]
    ns = {**vars(sys.modules[owner.__module__]), **getattr(fn, "__globals__", {})}
    return [ns[n] for n in names if inspect.isclass(ns.get(n)) and ns[n].__module__.startswith("fastforks")]


def _resolve(path: str) -> object:
    """The object a dotted `fastforks...` path names; `None` if there is none."""
    module, _, name = path.rpartition(".")
    try:
        return getattr(importlib.import_module(module), name, None)
    except ImportError:
        return None


def _has(classes: list[type], attr: str) -> bool:
    return any(hasattr(c, attr) or attr in getattr(c, "__dataclass_fields__", {}) for c in classes)


def _bind(fn: object, call: ast.Call, *, bound: bool) -> str | None:
    """Why `call`'s arguments do not fit `fn`'s signature; `None` if they do (or cannot be checked)."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    try:
        sig.bind(*([None] * (len(call.args) + (not bound))), **{k.arg: None for k in call.keywords if k.arg})
    except TypeError as e:
        return str(e)
    return _bad_literal(fn, call)


def _literals(tp: object) -> set[object] | None:
    """The values of a `Literal[...]` (also inside a union); `None` if any member is another type."""
    if typing.get_origin(tp) is typing.Literal:
        return set(typing.get_args(tp))
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        parts = [_literals(a) if a is not type(None) else set() for a in typing.get_args(tp)]
        return None if any(p is None for p in parts) else set().union(*parts)
    return None


def _bad_literal(fn: object, call: ast.Call) -> str | None:
    """A literal keyword value that the parameter's `Literal[...]` annotation does not allow (placeholders pass)."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - unresolvable forward references: nothing to check
        return None
    for k in call.keywords:
        if k.arg in hints and isinstance(k.value, ast.Constant) and k.value.value is not None:
            allowed = _literals(hints[k.arg])
            if allowed is not None and k.value.value not in allowed:
                return f"{k.arg}={k.value.value!r} is not one of {sorted(map(repr, allowed))}"
    return None


class _Examples(ast.NodeVisitor):
    """Checks one block; `env` maps names to ("module", path) / ("model", task, id) / ("result", classes)."""

    def __init__(self, where: str, line: int, env: dict[str, tuple]) -> None:
        self.where, self.line, self.env, self.problems = where, line, env, []

    def bad(self, node: ast.AST, msg: str) -> None:
        self.problems.append(f"{self.where}:{self.line + getattr(node, 'lineno', 1) - 1}: {msg}")

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.name.startswith("fastforks"):
                self.env[a.asname or a.name.split(".")[0]] = ("module", a.name if a.asname else "fastforks")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.startswith("fastforks"):
            for a in node.names:
                path = f"{node.module}.{a.name}"
                self.env[a.asname or a.name] = ("module", path) if path.count(".") < 3 else ("other",)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        kind = self.infer(node.value)
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.env[t.id] = kind

    def infer(self, node: ast.AST) -> tuple:  # noqa: PLR0911 - one case per expression kind
        if isinstance(node, ast.Name):
            return self.env.get(node.id, ("other",))
        if isinstance(node, ast.Attribute):
            base = self.infer(node.value)
            if base[0] == "module":
                path = f"{base[1]}.{node.attr}"
                return ("module", path) if node.attr in (*registry.TASKS, "geometry") else ("func", path)
            return ("other",)
        if not isinstance(node, ast.Call):
            return ("other",)
        func = self.infer(node.func)
        if func[0] == "func" and func[1].rsplit(".", 1)[-1] == "load":
            return self.load(node, func[1].rsplit(".", 1)[0])
        if func[0] == "func":
            obj = _resolve(func[1])
            return ("result", _result_classes(obj, obj)) if obj is not None else ("other",)
        if func[0] == "model":
            return ("result", [r for c in _model_classes(func[1]) for r in _result_classes(c.__call__, c)])
        if isinstance(node.func, ast.Attribute):
            owner = self.infer(node.func.value)
            classes = _model_classes(owner[1]) if owner[0] == "model" else owner[1] if owner[0] == "result" else []
            fns = [(getattr(c, node.func.attr), c) for c in classes if callable(getattr(c, node.func.attr, None))]
            if found := [r for f, c in fns for r in _result_classes(f, c)]:
                return ("result", found)
        return ("other",)

    def load(self, node: ast.Call, module: str) -> tuple:
        arg = node.args[0] if node.args else None
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            return ("other",)
        model_id = arg.value
        if "/" in model_id or "\\" in model_id or model_id.endswith(_PATH_SUFFIXES):
            return ("model", module.removeprefix("fastforks."), None) if module != "fastforks" else ("other",)
        task = registry.task_of(model_id.split("@")[0])
        if task is None or module not in {"fastforks", f"fastforks.{task}"}:
            self.bad(node, f"{module}.load({model_id!r}): no such model in that task")
            return ("other",)
        card, _ = registry.resolve(task, model_id)
        loader = importlib.import_module(module).load
        params = set(inspect.signature(loader).parameters)
        names = set(card.options) | set(card.option_aliases)
        for k in node.keywords:
            if k.arg == "options":
                self.bad(node, f"load({model_id!r}, options=...): the 0.1.1 form; pass options as keywords")
            elif k.arg and k.arg not in params and k.arg not in names:
                self.bad(node, f"load({model_id!r}, {k.arg}=...): not a load argument or an option of the card")
        return ("model", task, model_id)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        func = self.infer(node.func)
        if func[0] == "func" and not func[1].endswith(".load"):
            obj = _resolve(func[1])
            if obj is None:
                self.bad(node, f"{func[1]}: no such name")
            elif why := _bind(obj, node, bound=True):
                self.bad(node, f"{func[1]}(...): {why}")
            return
        if func[0] == "model":
            classes, name, bound = _model_classes(func[1]), "__call__", False
        elif isinstance(node.func, ast.Attribute) and (owner := self.infer(node.func.value))[0] in {"model", "result"}:
            classes = _model_classes(owner[1]) if owner[0] == "model" else owner[1]
            name, bound = node.func.attr, False
        else:
            return
        fns = [getattr(c, name) for c in classes if callable(getattr(c, name, None))]
        if classes and not fns:
            self.bad(node, f"{name}: no such method of {[c.__name__ for c in classes]}")
        elif fns and all(why := [_bind(f, node, bound=bound) for f in fns]):
            self.bad(node, f"{name}(...): {why[0]}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.generic_visit(node)
        owner = self.infer(node.value)
        if owner[0] == "result" and owner[1] and not _has(owner[1], node.attr):
            self.bad(node, f".{node.attr}: not a field of {[c.__name__ for c in owner[1]]}")


def test_python_examples_use_real_ids_options_and_signatures():
    """Every ```python block of the docs and docstrings loads ids that exist, with load arguments / card options that
    exist, and calls models and results with keywords their signatures take (`ast` only; nothing runs)."""
    problems = []
    blocks = _python_blocks()
    assert len(blocks) > 50, "the block pattern no longer matches the docs"
    env: dict[str, tuple] = {}
    for i, (where, line, code) in enumerate(blocks):
        # the blocks of one file share their imports; other names (`model`, ...) are per block
        env = {k: v for k, v in env.items() if v[0] in {"module", "func"}} if i and blocks[i - 1][0] == where else {}
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            problems.append(f"{where}:{line + (e.lineno or 1) - 1}: not valid Python ({e.msg})")
            continue
        v = _Examples(where, line, env)
        v.visit(tree)
        problems += v.problems
    assert not problems, "\n".join(sorted(set(problems)))


def test_example_check_rejects_a_literal_outside_the_annotation():
    def fn(*, mode: typing.Literal["a", "b"] | None = None, n: int = 1) -> None: ...

    bad = ast.parse("fn(mode='c', n=3)").body[0].value
    good = ast.parse("fn(mode='a', n=x)").body[0].value
    assert "mode='c'" in (_bind(fn, bad, bound=True) or "")
    assert _bind(fn, good, bound=True) is None


def test_oom_row_older_than_a_timed_row_is_stale(tmp_path, monkeypatch):
    """An OOM row only marks a cell while no measured row of the same key is as new (seedvr2-7b before streaming)."""
    data = tmp_path / "docs" / "data"
    data.mkdir(parents=True)
    rows = [
        {"ref": "a", "mode": "portable", "input": "1x1", "status": "oom", "date": "2026-11-14"},
        {"ref": "a", "mode": "portable", "input": "1x1", "ms": 5.0, "side": "lib", "date": "2026-11-14"},
        {"ref": "b", "mode": "portable", "input": "1x1", "status": "oom", "date": "2026-11-15"},
        {"ref": "b", "mode": "portable", "input": "1x1", "ms": 5.0, "side": "lib", "date": "2026-11-14"},
    ]
    (data / "speed.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(gen_docs, "ROOT", tmp_path)
    gen_docs._oom.cache_clear()
    try:
        assert gen_docs._oom() == {("b", "portable", "1x1")}
    finally:
        gen_docs._oom.cache_clear()


def test_no_original_reason_only_without_an_original_row():
    """A card states why its original was not timed only while docs/data has no timed upstream row of its variants."""
    from fastforks._core import registry  # noqa: PLC0415

    timed = {
        json.loads(line)["ref"]
        for line in (ROOT / "docs" / "data" / "speed.jsonl").read_text().splitlines()
        if '"upstream' in line and '"ms"' in line and json.loads(line).get("ms") is not None
    }
    for task in registry.TASKS:
        for card, _ in registry.cards(task).values():
            if card.no_original_reason:
                assert not timed & {v.id for v in card.variants}, card.name
