"""The tests a change needs (the per-merge gate): the families it affects and their pytest selection.

    python tools/select_tests.py BASE [HEAD] [--json]      # the files of `git diff --name-only BASE..HEAD`
    python tools/select_tests.py --files PATH ... [--json]
    python tools/select_tests.py BASE [HEAD] --golden-nodeids {cpu,gpu}

A changed file selects every family whose static import closure holds it. The closure of a family starts from its own
modules (`src/fastforks/<task>/models/<family>/**/*.py`), its card's kernel libraries (`[modes.kernels] kernels`) and
`uses` (shared modules), its task package (the `load` path) and the GPU harness (`HARNESS`: the direct imports of
tests/test_golden.py and tools/speed_check.py, without other tasks' modules), and follows every import read with `ast`
(not under `if TYPE_CHECKING:`) plus the parent packages of each module. Imports inside functions count in family,
shared and kernels modules only: the framework (_core, _optimize, the task modules) imports lazily what a card or a call
asks for (`kernel_libs`, a task's save path), and the harness seeds drop kernel libraries for the same reason. A package
that loads its submodules with `pkgutil.iter_modules` (the task specs) imports all of them. A file that is not a module
(kernel source, manifest, card.toml, golden.toml, a JSON table) belongs to its package; a deleted one to its nearest
existing package, except the ref tables of _core (`REF_TABLES`: auto_portable.json, sizes.json): with a git range they
select the families whose refs occur in their changed lines. A git range limited to existing golden reference records
selects those revisions when the manifest's shared settings are unchanged. Changing tests/test_golden.py selects every
family. Changing a `GATE_TOOLS` file selects the fast tier. pyproject.toml / uv.lock select every family
and flag `release` (dependencies changed: use the broad gate). When both git snapshots are available, an edit limited
to the local package version in either file leaves the dependency selection unchanged. Edits limited to a card's
summary and limitations select card and documentation checks. Everything else (docs, tools, other tests) selects
no family.

An empty change set selects no tests. Fast tier (`-m "not gpu and not model"`): every test file whose closure reaches
no family module, the family-specific files of the selected families and every test file whose closure holds a changed
file. Changes limited to existing evaluation tools and their tests omit unrelated fast-tier tests. All of tests/ run
when conftest.py,
non-Python test data or the dependencies changed. Documentation-only changes use the docs checks, plus the site data
and speed-floor checks for generated model or benchmark pages and their data. The GPU selection gives each family's
golden refs as a `-k` expression for tests/test_golden.py. Use `--golden-nodeids` for exact item IDs that preserve
revisions and distinguish similarly named families. These IDs can be passed to pytest using an argument file.

For edits limited to existing model implementation files, `--affected-fast` keeps import-linked tests and tests
that name the affected models. The exact golden selection is the same as in the standard selection.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ("tests/test_golden.py", "tools/speed_check.py")  # their direct imports seed every family's closure
GATE_TOOLS = frozenset({"tools/select_tests.py", "tools/speed_check.py", "tools/merge_check.py"})
DEPENDENCIES = frozenset({"pyproject.toml", "uv.lock"})
TOP_LEVEL_DOCS = frozenset({"README.md", "CONTRIBUTING.md"})
BENCHMARK_DOCS = frozenset({"docs/benchmarks.md", "docs/overview.md"})
GOLDEN_MANIFEST = re.compile(r"src/fastforks/[^/]+/models/[^/]+/golden[.]toml")
MODEL_CARD = re.compile(r"src/fastforks/[^/]+/models/[^/]+/card[.]toml")
CARD_TEXT_FIELDS = frozenset({"summary", "limitations"})
CARD_TEXT_CHECKS = frozenset({"tests/test_cards.py", "tests/test_docs.py", "tests/test_site_data.py"})
FRAMEWORK = "src/fastforks/"  # outside LAZY_OK and the family dirs: imports inside functions do not count
LAZY_OK = ("src/fastforks/shared/", "src/fastforks/kernels/", "src/fastforks/_optimize/")
REF_TABLES = re.compile(r"src/fastforks/_core/[^/]+\.json")  # tables keyed by ref (auto_portable.json, sizes.json)
EVALUATION_FILES = re.compile(
    r"(?:(?:tools/(?:_prefetch|eval_[^/]+)|tests/test_eval_[^/]+)\.py|tools/bsds_correspond\.cc)"
)
PATH_ORDER = ("tests", "tools")  # pytest `pythonpath`: top-level modules of tests/ shadow tools/


@dataclass
class Selection:
    """What a change needs: per family its task, golden refs and the changed files that selected it."""

    changed: list[str]
    families: dict[str, dict] = field(default_factory=dict)  # family -> {"task", "refs", "why"}
    fast: list[str] = field(default_factory=list)  # test files; ["tests"] = the whole fast tier
    release: bool = False

    def k(self, family: str) -> str:
        """The family's `-k` expression for tests/test_golden.py."""
        refs = sorted({r.split("@")[0] for r in self.families[family]["refs"]})
        return "test_gpu and (" + " or ".join(f"[{r}" for r in refs) + ")"

    def golden_nodeids(self, device: Literal["cpu", "gpu"]) -> list[str]:
        """Exact test_golden.py item IDs for the selected families and revisions."""
        items = set()
        for family in self.families.values():
            for ref in family["refs"]:
                if device == "cpu":
                    cases = [f"test_cpu_portable[{ref}]", f"test_kernels_validated[{ref}]"]
                else:
                    cases = [f"test_gpu[{ref}-{mode}]" for mode in ("portable", "kernels")]
                    if family["task"] == "classification":
                        cases.append(f"test_gpu_module[{ref}]")
                items.update(f"tests/test_golden.py::{case}" for case in cases)
        return sorted(items)

    def as_dict(self) -> dict:
        """JSON form (tools/merge_check.py reads it)."""
        fams = {f: v | {"k": self.k(f)} for f, v in sorted(self.families.items())}
        return {"changed": self.changed, "release": self.release, "fast": self.fast, "families": fams}


class Graph:
    """Python files of src/fastforks, tests/ and tools/ with the files each one imports."""

    def __init__(self, root: Path) -> None:
        """Index the module names of `root`'s src/, tests/ and tools/ (imports are read on demand)."""
        self.root = root
        self.mod: dict[str, str] = {}  # module name -> file (relative to root)
        for top in ("src", *reversed(PATH_ORDER)):  # tests/ wins over tools/
            for p in sorted((root / top).rglob("*.py")):
                parts = p.relative_to(root / top).with_suffix("").parts
                self.mod[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = self.rel(p)
        self.name = {f: m for m, f in self.mod.items()}
        self.edges: dict[str, set[str]] = {}

    def rel(self, p: Path) -> str:
        """`p` relative to the root."""
        return p.relative_to(self.root).as_posix()

    def imports(self, file: str) -> set[str]:
        """Files `file` imports (with their parent packages)."""
        if file not in self.edges:
            self.edges[file] = self._read(file)
        return self.edges[file]

    def _read(self, file: str) -> set[str]:
        path = self.root / file
        if not path.is_file():
            return set()
        text = path.read_text(encoding="utf-8")
        me = self.name.get(file, "")
        pkg = me if file.endswith("__init__.py") else me.rpartition(".")[0]
        lazy = not file.startswith(FRAMEWORK) or file.startswith(LAZY_OK) or "/models/" in file
        out = set()
        for name in _imported_names(ast.parse(text, filename=file), pkg, lazy=lazy):
            mod = name
            while mod and mod not in self.mod:  # `from pkg import function`: the module is pkg
                mod = mod.rpartition(".")[0]
            while mod:  # the module and its parent packages
                if mod in self.mod:
                    out.add(self.mod[mod])
                mod = mod.rpartition(".")[0]
        if file.endswith("__init__.py") and "pkgutil.iter_modules" in text:  # loads every submodule
            out |= {self.rel(p) for p in [*path.parent.glob("*.py"), *path.parent.glob("*/__init__.py")]}
        out.discard(file)
        return out

    def closure(self, seeds: set[str]) -> set[str]:
        """`seeds` and every file they import, transitively."""
        seen, todo = set(), list(seeds)
        while todo:
            f = todo.pop()
            if f not in seen:
                seen.add(f)
                todo += self.imports(f) - seen
        return seen

    def owner(self, file: str) -> str:
        """The module a changed file belongs to: itself, else the package of its (nearest existing) directory."""
        if file in self.name:
            return file
        d = (self.root / file).parent
        while d != self.root and d.is_relative_to(self.root / "src"):
            init = d / "__init__.py"
            if init.is_file():
                return self.rel(init)
            d = d.parent
        return file


def _imported_names(tree: ast.AST, pkg: str, *, lazy: bool) -> set[str]:
    """Absolute module names `tree` imports (`from a import b`: `a` and `a.b`); `pkg`: the file's package."""
    names: set[str] = set()
    for node in _runtime_imports(tree, lazy=lazy):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        else:
            base = node.module or ""
            if node.level:
                up = pkg.split(".")[: len(pkg.split(".")) - node.level + 1]
                base = ".".join([*up, base] if base else up)
            names |= {f"{base}.{a.name}" for a in node.names} | {base}
    return names


def _runtime_imports(tree: ast.AST, *, lazy: bool) -> Iterator[ast.Import | ast.ImportFrom]:
    """Walk statement bodies that can contain imports, skipping type-checking branches."""
    containers = (ast.stmt, ast.ExceptHandler, ast.match_case)
    todo = [tree]
    while todo:
        node = todo.pop()
        if isinstance(node, ast.Import | ast.ImportFrom):
            yield node
        elif isinstance(node, ast.If) and _is_type_checking(node.test):
            todo.extend(node.orelse)
        elif lazy or not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            todo.extend(child for child in ast.iter_child_nodes(node) if isinstance(child, containers))


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def families(root: Path) -> dict[str, dict]:
    """Per family {"task", "dir", "refs", "kernel_libs", "uses"} from the card.toml / golden.toml files."""
    out = {}
    for card in sorted((root / "src" / "fastforks").glob("*/models/*/card.toml")):
        c = tomllib.loads(card.read_text(encoding="utf-8"))
        g = card.with_name("golden.toml")
        refs = sorted(tomllib.loads(g.read_text(encoding="utf-8")).get("refs", {})) if g.is_file() else []
        out[card.parent.name] = {
            "task": card.parents[2].name,
            "dir": card.parent.relative_to(root).as_posix(),
            "refs": refs,
            "kernel_libs": c.get("modes", {}).get("kernels", {}).get("kernels", []),
            "uses": c.get("uses", []),
        }
    return out


def _kernel_lib(file: str) -> bool:
    return file.startswith(LAZY_OK[1]) and file.count("/") > LAZY_OK[1].count("/")


def _module_file(graph: Graph, name: str) -> set[str]:
    return {graph.mod[name]} if name in graph.mod else set()


def family_closures(graph: Graph, fams: dict[str, dict]) -> dict[str, set[str]]:
    harness = {h: graph.imports(h) for h in HARNESS if (graph.root / h).is_file()}
    tasks = {f["task"] for f in fams.values()}
    out = {}
    for name, f in fams.items():
        others = tuple(f"src/fastforks/{t}/" for t in tasks - {f["task"]})
        seeds = {graph.rel(p) for p in (graph.root / f["dir"]).rglob("*.py")}
        seeds |= _module_file(graph, f"fastforks.{f['task']}")
        for lib in f["kernel_libs"]:
            seeds |= _module_file(graph, f"fastforks.kernels.{lib}")
        for mod in f["uses"]:
            seeds |= _module_file(graph, f"fastforks.shared.{mod}")
        for imported in harness.values():  # not other tasks' modules, not kernel libraries (the card names them)
            seeds |= {i for i in imported if not i.startswith(others) and not _kernel_lib(i)}
        out[name] = graph.closure(seeds) | {"tests/test_golden.py"}
    return out


def _table_refs(fams: dict[str, dict], text: str) -> dict[str, bool]:
    """Families whose refs (`"ref"` / `"ref@rev"`) occur in the changed lines of a ref table's diff."""
    lines = "\n".join(line for line in text.splitlines() if line[:1] in "+-" and line[:3] not in {"+++", "---"})
    found = set(re.findall(r'"([^"@]+)(?:@\d+)?"', lines))
    return {name: bool(found & {r.split("@")[0] for r in f["refs"]}) for name, f in fams.items()}


def _fast_checks(
    graph: Graph, sel: Selection, fams: dict[str, dict], owned: set[str], *, affected_fast: bool
) -> list[str]:
    """Find tests linked to the selected files and families, optionally skipping generic tests."""
    root, changed = graph.root, set(sel.changed)
    if "tools/bsds_correspond.cc" in changed:
        owned = owned | {"tools/eval_edges.py"}
    evaluation_only = all(EVALUATION_FILES.fullmatch(path) and (root / path).is_file() for path in changed)
    family_dirs = tuple(f["dir"] + "/" for f in fams.values())
    selected_dirs = tuple(fams[name]["dir"] + "/" for name in sel.families)
    affected_refs = None
    if (
        affected_fast
        and sel.families
        and all(c in graph.name and c.startswith(family_dirs) and not c.endswith("/__init__.py") for c in changed)
    ):
        names = set(sel.families) | {ref.split("@")[0] for f in sel.families.values() for ref in f["refs"]}
        alternatives = "|".join(re.escape(name) for name in sorted(names))
        affected_refs = re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w-])")
    include_generic = not evaluation_only and affected_refs is None
    checks = []
    for test in sorted(f for f in graph.name if f.startswith("tests/test_")):
        cl = graph.closure({test})
        if (
            test in changed
            or cl & owned
            or any(f.startswith(selected_dirs) for f in cl)
            or (include_generic and not any(f.startswith(family_dirs) for f in cl))
            or (affected_refs and affected_refs.search((root / test).read_text(encoding="utf-8")))
        ):
            checks.append(test)
    return checks


def select(  # noqa: PLR0913 - the source snapshots and local selection mode are independent
    changed: list[str],
    root: Path = ROOT,
    diff: Callable[[str], str] | None = None,
    *,
    metadata_only: frozenset[str] = frozenset(),
    fixture_refs: dict[str, tuple[str, ...]] | None = None,
    affected_fast: bool = False,
) -> Selection:
    """Select affected variants, using tracked metadata where the caller has both git snapshots."""
    card_text = {path for path in metadata_only if MODEL_CARD.fullmatch(path)}
    card_checks = CARD_TEXT_CHECKS if card_text else frozenset()
    if all(path in TOP_LEVEL_DOCS or path.startswith("docs/") or path in card_text for path in changed):
        checks = set(card_checks)
        if changed:
            checks.add("tests/test_docs.py")
        if any(path in BENCHMARK_DOCS or path.startswith(("docs/data/", "docs/models/")) for path in changed):
            checks.update(("tests/test_site_data.py", "tests/test_speed_floor.py"))
        if "docs/stereo_geometry.md" in changed:
            checks.add("tests/test_geometry_example.py")
        return Selection(changed=sorted(changed), fast=sorted(path for path in checks if (root / path).is_file()))
    graph = Graph(root)
    fams = families(root)
    closures = family_closures(graph, fams)
    dependencies = DEPENDENCIES.intersection(changed) - metadata_only
    sel = Selection(changed=sorted(changed), release=bool(dependencies))
    fixtures = fixture_refs or {}
    tables = {c: _table_refs(fams, diff(c)) for c in changed if diff and REF_TABLES.fullmatch(c)}
    owners = {
        c: graph.owner(c)
        for c in changed
        if c not in GATE_TOOLS and c not in tables and c not in fixtures and c not in card_text
    }
    for name, cl in closures.items():
        family = fams[name]
        why = sorted(c for c, o in owners.items() if o in cl or (sel.release and c in dependencies))
        why += sorted(c for c, hit in tables.items() if hit[name])
        manifest = f"{family['dir']}/golden.toml"
        refs = family["refs"] if why else list(fixtures.get(manifest, ()))
        if fixtures.get(manifest):
            why.append(manifest)
        if why:
            sel.families[name] = {"task": family["task"], "refs": refs, "why": why}
    full = sel.release or any(
        c == "tests/conftest.py" or (c.startswith("tests/") and not c.endswith(".py")) for c in changed
    )
    if full:
        sel.fast = ["tests"]
        return sel
    checks = set(_fast_checks(graph, sel, fams, set(owners.values()), affected_fast=affected_fast))
    sel.fast = sorted(checks | {path for path in card_checks if (root / path).is_file()})
    return sel


def git_changed(base: str, head: str, root: Path = ROOT, *, from_merge_base: bool = False) -> list[str]:
    """Return both paths for a move. Use the merge base for pull-request fixture selection."""
    # A cross-family move affects both the deleted module and its new owner.
    dots = "..." if from_merge_base else ".."
    r = subprocess.run(
        ["git", "-C", str(root), "diff", "--no-renames", "--name-only", "-z", f"{base}{dots}{head}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [path for path in r.stdout.split("\0") if path]


def git_diff(base: str, head: str, root: Path = ROOT) -> Callable[[str], str]:
    """`file -> its unified diff base..head`."""

    def diff(file: str) -> str:
        cmd = ["git", "-C", str(root), "diff", "-U0", f"{base}..{head}", "--", file]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    return diff


def configuration_without_local_version(path: str, text: str) -> dict[str, object] | None:
    """Parse package configuration and drop the version of the editable FastForks project."""
    try:
        doc = tomllib.loads(text)
        if path == "uv.lock":
            packages = [
                p for p in doc["package"] if p.get("name") == "fastforks-core" and p.get("source") == {"editable": "."}
            ]
            if len(packages) != 1:
                return None
            project = packages[0]
        else:
            project = doc["project"]
            if project.get("name") != "fastforks-core":
                return None
        if not isinstance(project.get("version"), str):
            return None
        del project["version"]
    except (AttributeError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return None
    return doc


def same_lock_dependencies(old: str, new: str) -> bool:
    """Whether two lockfiles match without the editable project's version."""
    reference = configuration_without_local_version("uv.lock", old)
    return reference is not None and reference == configuration_without_local_version("uv.lock", new)


def same_project_configuration(old: str, new: str) -> bool:
    """Whether two project configurations match without the project's version."""
    reference = configuration_without_local_version("pyproject.toml", old)
    return reference is not None and reference == configuration_without_local_version("pyproject.toml", new)


def _git_snapshots(base: str, head: str, path: str, root: Path) -> list[str] | None:
    snapshots = [
        subprocess.run(["git", "-C", str(root), "show", f"{rev}:{path}"], capture_output=True, text=True, check=False)
        for rev in (base, head)
    ]
    return [s.stdout for s in snapshots] if all(s.returncode == 0 for s in snapshots) else None


def split_golden_manifest(text: str) -> tuple[dict[str, object], dict[str, object]] | None:
    """Split a golden manifest into its shared settings and its per-revision fixture records."""
    try:
        common = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None
    refs = common.pop("refs", None)
    if not isinstance(refs, dict) or any(not isinstance(record, dict) for record in refs.values()):
        return None
    return common, refs


def _changed_fixture_refs(old: str, new: str) -> tuple[str, ...] | None:
    """Isolate changes to existing reference records while the rest of the manifest matches."""
    before, after = split_golden_manifest(old), split_golden_manifest(new)
    if before is None or after is None:
        return None
    shared_before, previous = before
    shared_after, current = after
    if shared_before != shared_after or previous.keys() != current.keys():
        return None
    return tuple(sorted(ref for ref, data in current.items() if data != previous[ref]))


def git_fixture_refs(base: str, head: str, paths: list[str], root: Path = ROOT) -> dict[str, tuple[str, ...]]:
    """Return the affected revisions for manifests changed only in their existing reference records."""
    fixtures = {}
    for path in paths:
        if GOLDEN_MANIFEST.fullmatch(path) and (snapshots := _git_snapshots(base, head, path, root)):
            refs = _changed_fixture_refs(*snapshots)
            if refs is not None:
                fixtures[path] = refs
    return fixtures


def _same_card_runtime(old: str, new: str) -> bool:
    try:
        before, after = tomllib.loads(old), tomllib.loads(new)
    except tomllib.TOMLDecodeError:
        return False
    # The representation distinguishes TOML booleans, integers, and floats.
    return repr({k: v for k, v in before.items() if k not in CARD_TEXT_FIELDS}) == repr(
        {k: v for k, v in after.items() if k not in CARD_TEXT_FIELDS}
    )


def git_metadata_only(base: str, head: str, paths: list[str], root: Path = ROOT) -> frozenset[str]:
    """Identify version-only and model-card prose edits using both tracked snapshots."""
    unchanged = set()
    for path, compare in (("pyproject.toml", same_project_configuration), ("uv.lock", same_lock_dependencies)):
        if path not in paths:
            continue
        if (snapshots := _git_snapshots(base, head, path, root)) and compare(*snapshots):
            unchanged.add(path)
    for path in paths:
        if (
            MODEL_CARD.fullmatch(path)
            and (snapshots := _git_snapshots(base, head, path, root))
            and _same_card_runtime(*snapshots)
        ):
            unchanged.add(path)
    return frozenset(unchanged)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", nargs="?")
    ap.add_argument("head", nargs="?", default="HEAD")
    ap.add_argument("--files", nargs="*", help="changed paths instead of a git range")
    ap.add_argument("--affected-fast", action="store_true", help="scope fast tests for model implementation edits")
    output = ap.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true")
    output.add_argument("--golden-nodeids", choices=("cpu", "gpu"), help="print exact pytest item IDs for the device")
    a = ap.parse_args(argv)
    if a.files is None and not a.base:
        ap.error("give BASE [HEAD] or --files")
    files, diff = (a.files, None) if a.files is not None else (git_changed(a.base, a.head), git_diff(a.base, a.head))
    metadata_only = git_metadata_only(a.base, a.head, files) if a.files is None else frozenset()
    fixture_refs = git_fixture_refs(a.base, a.head, files) if a.files is None else None
    sel = select(
        files, diff=diff, metadata_only=metadata_only, fixture_refs=fixture_refs, affected_fast=a.affected_fast
    )
    if a.golden_nodeids:
        for item in sel.golden_nodeids(a.golden_nodeids):
            print(item)
        return 0
    if a.json:
        print(json.dumps(sel.as_dict(), indent=1))
        return 0
    print(f"{len(sel.changed)} changed files; release: {sel.release}")
    print(f"fast tier: {' '.join(sel.fast)}")
    for name, f in sorted(sel.families.items()):
        print(f"{f['task']}/{name}: {len(f['refs'])} refs; -k {sel.k(name)!r}; because of {', '.join(f['why'][:3])}")
    print(f"{len(sel.families)} families")
    return 0


if __name__ == "__main__":
    sys.exit(main())
