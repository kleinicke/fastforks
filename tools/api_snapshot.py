"""Snapshot of the public API (`tests/api_snapshot.txt`): names, signatures, result fields, card options, CLI flags.

    uv run python tools/api_snapshot.py            # print the snapshot
    uv run python tools/api_snapshot.py --update   # write tests/api_snapshot.txt
    uv run python tools/api_snapshot.py --check    # exit 1 if the file is out of date (prints the diff)

One sorted line per item, imports only (no model or weights load). tests/test_api_snapshot.py fails on any difference,
and on an item the release branch (`main`; before it has a snapshot: tests/api_snapshot_v0.1.1.txt) has that is gone
here unless CHANGELOG.md's "Unreleased" section names it.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import importlib
import inspect
import re
import subprocess
import sys
from pathlib import Path

from fastforks._core import registry

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests" / "api_snapshot.txt"
# Public submodules with their own `__all__` (besides `fastforks` and the task packages).
SUBMODULES = (
    "fastforks.matching.absolute_pose",
    "fastforks.matching.relative_pose",
    "fastforks.matching.sfm",
    "fastforks.stereo.geometry",
    "fastforks.stereo.pipeline",
)
_ADDRESS = re.compile(r" at 0x[0-9a-f]+")


def modules() -> list[str]:
    """Every public module of the snapshot."""
    return ["fastforks", *(f"fastforks.{t}" for t in registry.TASKS), *SUBMODULES]


def _sig(obj: object) -> str:
    try:
        return _ADDRESS.sub("", str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return "(?)"


def _members(cls: type, seen: set[type], lines: list[str]) -> None:
    """Methods, properties and fields of `cls` and of its fastforks bases, each class once."""
    for klass in cls.__mro__:
        if klass in seen or not klass.__module__.startswith("fastforks") or issubclass(klass, BaseException):
            continue
        seen.add(klass)
        name = f"{klass.__module__}.{klass.__qualname__}"
        bases = ", ".join(f"{b.__module__}.{b.__qualname__}" for b in klass.__bases__)
        lines.append(f"class {name}({bases})")
        if dataclasses.is_dataclass(klass):
            own = inspect.get_annotations(klass)
            lines += [f"field {name}.{f.name}: {f.type}" for f in dataclasses.fields(klass) if f.name in own]
        elif issubclass(klass, tuple) and hasattr(klass, "_fields"):
            ann = inspect.get_annotations(klass)
            lines += [f"field {name}.{f}: {ann.get(f, '?')}" for f in getattr(klass, "_fields")]  # noqa: B009
        for attr, value in sorted(vars(klass).items()):
            if attr.startswith("_") and attr != "__call__":
                continue
            if isinstance(value, property):
                ret = inspect.signature(value.fget).return_annotation if value.fget else "?"
                lines.append(f"property {name}.{attr} -> {ret}")
            elif isinstance(value, (staticmethod, classmethod)) or inspect.isfunction(value):
                lines.append(f"def {name}.{attr}{_sig(getattr(klass, attr))}")


def _cli(lines: list[str]) -> None:
    """Subcommands and flags of `fastforks ...` (the parser `main` builds; nothing runs)."""
    from fastforks._core import cli  # noqa: PLC0415 - only for the snapshot

    built: list[argparse.ArgumentParser] = []

    def capture(self: argparse.ArgumentParser, *_: object, **__: object) -> None:
        built.append(self)
        raise SystemExit(0)

    orig = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = capture
    try:
        cli.main([])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig

    def walk(p: argparse.ArgumentParser, path: str) -> None:
        for a in p._actions:  # noqa: SLF001 - argparse has no public accessor
            if isinstance(a, argparse._SubParsersAction):  # noqa: SLF001
                for sub_name, sub in a.choices.items():
                    walk(sub, f"{path} {sub_name}")
            elif not isinstance(a, argparse._HelpAction):  # noqa: SLF001
                lines.append(f"cli {path}: {'/'.join(a.option_strings) or a.dest}")

    walk(built[0], "fastforks")


def snapshot() -> str:
    """The snapshot text."""
    lines: list[str] = []
    seen: set[type] = set()
    for mod_name in modules():
        mod = importlib.import_module(mod_name)
        for name in sorted(getattr(mod, "__all__", ())):
            obj = getattr(mod, name)
            if inspect.isclass(obj):
                lines.append(f"name {mod_name}.{name} = class {obj.__module__}.{obj.__qualname__}")
                _members(obj, seen, lines)
            elif callable(obj):
                lines.append(f"def {mod_name}.{name}{_sig(obj)}")
            else:
                lines.append(f"name {mod_name}.{name}")
    for task in registry.TASKS:
        families = {card.family: card for card, _ in registry.cards(task).values()}
        for family, card in families.items():
            aliases = {o: sorted(a for a, t in card.option_aliases.items() if t == o) for o in card.options}
            lines += [
                f"option {task}.{family}.{o}" + (f" aliases {','.join(aliases[o])}" if aliases[o] else "")
                for o in card.options
            ]
    _cli(lines)
    return "\n".join(sorted(set(lines))) + "\n"


def removed_items(old: str, new: str) -> list[str]:
    """Names in `old` that `new` lost: whole items, parameters of a kept signature, option aliases."""

    def key(line: str) -> str:
        return line if line.startswith("cli ") else re.split(r"[(: ]", line.split(" ", 1)[1], maxsplit=1)[0]

    def params(line: str) -> set[str]:
        sig = line[line.find("(") + 1 : line.rfind(")")] if "(" in line else ""
        while (s := re.sub(r"\[[^\[\]]*\]", "", sig)) != sig:  # drop subscripts: `Mapping[str, Any]` has no params
            sig = s
        return {m.group(1) for m in re.finditer(r"(?:^|, )\*{0,2}(\w+)", sig)}

    new_by_key = {key(x): x for x in new.splitlines() if x.startswith(("def ", "option "))}
    new_keys = {key(x) for x in new.splitlines()}
    gone = []
    for line in old.splitlines():
        k = key(line)
        if k not in new_keys:
            gone.append(line.rsplit(" ", 1)[-1] if line.startswith("cli ") else k.rsplit(".", 1)[-1])
        elif line.startswith("def ") and k in new_by_key:
            gone += sorted(params(line) - params(new_by_key[k]) - {"self", "cls"})
        elif line.startswith("option ") and k in new_by_key:
            old_a = set(line.partition(" aliases ")[2].split(",")) - {""}
            gone += sorted(old_a - set(new_by_key[k].partition(" aliases ")[2].split(",")))
    return sorted(set(gone))


def release_snapshot() -> str | None:
    """The release snapshot: `main`'s (else `origin/main`'s), else tests/api_snapshot_v0.1.1.txt; `None` if none."""
    for ref in ("main", "origin/main"):
        r = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{ref}:tests/api_snapshot.txt"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout
    v011 = ROOT / "tests" / "api_snapshot_v0.1.1.txt"
    return v011.read_text() if v011.exists() else None


def unlisted_removals(current: str) -> list[str]:
    """Names the release snapshot has and `current` lost that CHANGELOG.md's "Unreleased" section does not name."""
    old = release_snapshot()
    if old is None:
        return []
    text = (ROOT / "CHANGELOG.md").read_text()
    unreleased = text.split("## Unreleased", 1)[-1].split("\n## ", 1)[0]
    return [n for n in removed_items(old, current) if f"`{n}" not in unreleased and f".{n}`" not in unreleased]


def main() -> int:
    """Print, write or check the snapshot."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--update", action="store_true", help="write tests/api_snapshot.txt")
    g.add_argument("--check", action="store_true", help="exit 1 if tests/api_snapshot.txt is out of date")
    args = ap.parse_args()
    text = snapshot()
    if args.update:
        if missing := unlisted_removals(text):
            print(f"name these removed / renamed items in CHANGELOG.md (Unreleased): {missing}", file=sys.stderr)
            return 1
        SNAPSHOT.write_text(text)
    elif args.check:
        old = SNAPSHOT.read_text() if SNAPSHOT.exists() else ""
        diff = list(difflib.unified_diff(old.splitlines(), text.splitlines(), "committed", "current", lineterm=""))
        print("\n".join(diff))
        return 1 if diff else 0
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
