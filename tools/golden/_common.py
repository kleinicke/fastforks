"""Helpers for the per-family golden-output tools: the reference source tree, and where goldens go.

Goldens come from upstream code at a pinned commit, plus at most a small patch from `tools/golden/patches/` (the
patch and its SHA-256 are recorded in the family's `golden.toml`), so anyone can rebuild the reference from upstream.
"""

from __future__ import annotations

import atexit
import hashlib
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools")]

import goldens  # noqa: E402
from synth import image, pair, prompts, size  # noqa: E402

__all__ = ["image", "pair", "prompts", "reference_tree", "size", "write"]

PATCHES = Path(__file__).parent / "patches"


def reference_tree(repo: Path, commit: str, patch: str | None = None) -> Path:
    """Export `commit` of the git checkout `repo` into a temporary directory and apply `patches/<patch>` to it.

    The checkout itself is not touched (its working tree may be on any branch); the directory lives until exit.
    """
    tmp = tempfile.TemporaryDirectory(prefix="ff-ref-")
    atexit.register(tmp.cleanup)
    archive = subprocess.run(["git", "-C", repo, "archive", commit], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", tmp.name], input=archive, check=True)
    if patch is not None:
        subprocess.run(["git", "apply", "--whitespace=nowarn", PATCHES / patch], cwd=tmp.name, check=True)
    return Path(tmp.name)


def write(  # noqa: PLR0913 - one record, keyword-only fields
    ref: str,
    maps: dict[str, np.ndarray],
    *,
    family: str,
    upstream: str,
    patch: str | None,
    settings: str,
    task: str = "stereo",
) -> None:
    """Save `maps` as `$FASTFORKS_GOLDEN_PATH/<ref>.npz`; record its SHA-256 and provenance in the family golden.toml.

    A per-OS golden (`goldens.golden_files`): with `FASTFORKS_GOLDEN_OS=windows` on Windows, or on Windows for a ref
    that already has one, the output goes to `<ref>.windows.npz` and `[refs."<ref>".windows]` (only that OS's
    `[validated]` record is dropped); the ref's own golden, its tolerances and the other OS's records stay. A new own
    golden drops the per-OS ones (they came from the old numerics: rerun the tool on that OS).

    Args:
        ref: Model ref, e.g. `"s2m2-s@1"`.
        maps: Output maps by name.
        family: Model family directory, e.g. `"s2m2"`.
        upstream: `<repository URL>@<full commit>` of the reference code.
        patch: File in `tools/golden/patches/` applied on top, if any.
        settings: How the reference ran (device, dtype, iterations, torch version).
        task: Task package of the family.
    """
    out = Path(os.environ["FASTFORKS_GOLDEN_PATH"])
    out.mkdir(parents=True, exist_ok=True)
    store = goldens.load(family, task)
    old = store["refs"].get(ref, {})
    slot = _os_slot(old)
    buf = io.BytesIO()
    np.savez(buf, allow_pickle=False, **{k: v.astype(np.float32) for k, v in maps.items()})
    data = buf.getvalue()
    (out / (f"{ref}.{slot}.npz" if slot else f"{ref}.npz")).write_bytes(data)
    entry = {"sha256": hashlib.sha256(data).hexdigest(), "upstream": upstream}
    if patch is not None:
        entry |= {"patch": patch, "patch_sha256": hashlib.sha256((PATCHES / patch).read_bytes()).hexdigest()}
    if slot:
        store["refs"][ref] = old | {slot: entry | {"settings": settings}}
        store["validated"].get(ref, {}).pop(slot, None)  # new numerics on that OS: validate its kernel builds again
    else:
        keep = {k: v for k, v in old.items() if k.startswith(("gpu_", "cpu_")) or k in {"gated", "options"}}
        store["refs"][ref] = entry | {"settings": settings} | keep  # tolerances and load options stay
        store["validated"].pop(ref, None)  # new numerics: the kernel builds must be validated again
        for s in (s for s in goldens.OS_SLOTS if s in old):
            print(f"{ref}: dropped the {s} golden (old numerics); rerun this tool on {s}", file=sys.stderr)
    goldens.dump(family, store, task)
    print(f"{ref}{f' ({slot})' if slot else ''}", entry["sha256"])


def _os_slot(old: dict) -> str | None:
    """The per-OS golden slot `write` fills: `FASTFORKS_GOLDEN_OS` (must name this OS), or this OS's existing one."""
    here = goldens.os_slot()
    asked = os.environ.get("FASTFORKS_GOLDEN_OS", "")
    if asked and asked != here:
        sys.exit(f"FASTFORKS_GOLDEN_OS={asked}: a per-OS golden comes from a run on that OS (this is {sys.platform})")
    if asked:
        return asked
    return here if here is not None and here in old else None
