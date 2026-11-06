"""The tests a change needs (the per-merge gate): the families it affects and their pytest selection.

    python tools/select_tests.py BASE [HEAD] [--json]      # the files of `git diff --name-only BASE..HEAD`
    python tools/select_tests.py --files PATH ... [--json]

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
select the families whose refs occur in their changed lines. tests/test_golden.py itself selects every family;
`GATE_TOOLS` (this file and the gate's timing tools) only the fast tier. pyproject.toml / uv.lock select every family
and flag `release` (dependencies changed: run the release suite). Everything else (docs, tools, other tests) selects no
family.

Fast tier (`-m "not gpu and not model"`): every test file whose closure reaches no family module, the family-specific
files of the selected families and every test file whose closure holds a changed file; all of tests/ when conftest.py,
non-Python test data or the dependencies changed. GPU: per family its golden refs, as a `-k` expression for
tests/test_golden.py (`[<ref without @rev>` matches the parameter ids of test_gpu and test_gpu_module).
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
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ("tests/test_golden.py", "tools/speed_check.py")  # their direct imports seed every family's closure
GATE_TOOLS = frozenset({"tools/select_tests.py", "tools/speed_check.py", "tools/merge_check.py"})
DEPENDENCIES = frozenset({"pyproject.toml", "uv.lock"})
FRAMEWORK = "src/fastforks/"  # outside LAZY_OK and the family dirs: imports inside functions do not count
LAZY_OK = ("src/fastforks/shared/", "src/fastforks/kernels/", "src/fastforks/_optimize/")
REF_TABLES = re.compile(r"src/fastforks/_core/[^/]+\.json")  # tables keyed by ref (auto_portable.json, sizes.json)
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
        text = path.read_text()
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
    for node in _runtime_nodes(tree, lazy=lazy):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                up = pkg.split(".")[: len(pkg.split(".")) - node.level + 1]
                base = ".".join([*up, base] if base else up)
            names |= {f"{base}.{a.name}" for a in node.names} | {base}
    return names


def _runtime_nodes(tree: ast.AST, *, lazy: bool) -> list[ast.AST]:
    """Every node except the bodies of `if TYPE_CHECKING:` blocks (and of functions unless `lazy`)."""
    out: list[ast.AST] = []
    todo = [tree]
    while todo:
        node = todo.pop()
        out.append(node)
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            todo += node.orelse
            continue
        if not lazy and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        todo += ast.iter_child_nodes(node)
    return out


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def families(root: Path) -> dict[str, dict]:
    """Per family {"task", "dir", "refs", "kernel_libs", "uses"} from the card.toml / golden.toml files."""
    out = {}
    for card in sorted((root / "src" / "fastforks").glob("*/models/*/card.toml")):
        c = tomllib.loads(card.read_text())
        g = card.with_name("golden.toml")
        refs = sorted(tomllib.loads(g.read_text()).get("refs", {})) if g.is_file() else []
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


def select(changed: list[str], root: Path = ROOT, diff: Callable[[str], str] | None = None) -> Selection:
    """The selection of `changed` files; `diff(file)` (the file's unified diff) narrows the ref tables."""
    graph = Graph(root)
    fams = families(root)
    closures = family_closures(graph, fams)
    sel = Selection(changed=sorted(changed), release=bool(DEPENDENCIES & set(changed)))
    tables = {c: _table_refs(fams, diff(c)) for c in changed if diff and REF_TABLES.fullmatch(c)}
    owners = {c: graph.owner(c) for c in changed if c not in GATE_TOOLS and c not in tables}
    for name, cl in closures.items():
        why = sorted(c for c, o in owners.items() if o in cl or (sel.release and c in DEPENDENCIES))
        why += sorted(c for c, hit in tables.items() if hit[name])
        if why:
            sel.families[name] = {"task": fams[name]["task"], "refs": fams[name]["refs"], "why": why}
    full = sel.release or any(
        c == "tests/conftest.py" or (c.startswith("tests/") and not c.endswith(".py")) for c in changed
    )
    if full:
        sel.fast = ["tests"]
        return sel
    family_dirs = {name: f["dir"] + "/" for name, f in fams.items()}
    owned = set(owners.values())
    for test in sorted(f for f in graph.name if f.startswith("tests/test_")):
        cl = graph.closure({test})
        reached = {n for n, d in family_dirs.items() if any(f.startswith(d) for f in cl)}
        if not reached or reached & set(sel.families) or cl & owned or test in changed:
            sel.fast.append(test)
    return sel


def git_changed(base: str, head: str, root: Path = ROOT) -> list[str]:
    r = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", f"{base}..{head}"], capture_output=True, text=True, check=True
    )
    return [line for line in r.stdout.splitlines() if line]


def git_diff(base: str, head: str, root: Path = ROOT) -> Callable[[str], str]:
    """`file -> its unified diff base..head`."""

    def diff(file: str) -> str:
        cmd = ["git", "-C", str(root), "diff", "-U0", f"{base}..{head}", "--", file]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    return diff


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", nargs="?")
    ap.add_argument("head", nargs="?", default="HEAD")
    ap.add_argument("--files", nargs="*", help="changed paths instead of a git range")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.files is None and not a.base:
        ap.error("give BASE [HEAD] or --files")
    files, diff = (a.files, None) if a.files is not None else (git_changed(a.base, a.head), git_diff(a.base, a.head))
    sel = select(files, diff=diff)
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
