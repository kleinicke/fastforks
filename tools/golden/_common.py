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
    buf = io.BytesIO()
    np.savez(buf, allow_pickle=False, **{k: v.astype(np.float32) for k, v in maps.items()})
    data = buf.getvalue()
    (out / f"{ref}.npz").write_bytes(data)
    entry = {"sha256": hashlib.sha256(data).hexdigest(), "upstream": upstream}
    if patch is not None:
        entry |= {"patch": patch, "patch_sha256": hashlib.sha256((PATCHES / patch).read_bytes()).hexdigest()}
    store = goldens.load(family, task)
    old = store["refs"].get(ref, {})
    keep = {k: v for k, v in old.items() if k.startswith("gpu_") or k in {"gated", "options"}}  # tolerances, flags
    store["refs"][ref] = entry | {"settings": settings} | keep
    store["validated"].pop(ref, None)  # new numerics: the kernel builds must be validated again
    goldens.dump(family, store, task)
    print(ref, entry["sha256"])
