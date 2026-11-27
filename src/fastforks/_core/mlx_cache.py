"""The MLX conversion cache (`fastforks._core.mlx.convert_cached`): file names, provenance, listing and pruning.

A converted variant is one safetensors file

    <FASTFORKS_CACHE>/mlx/<task>/<family>/<variant>-r<revision>-<sha256[:12]>-<layout>-<dtype>.safetensors

(`dtype` one of `f32`, `f16`, `bf16`) holding the port's arrays as it runs them: in the run dtype, every load-time
transform (layout, folds, fusions, tables) already applied, and the arrays the port keeps in float32 (`fp32` in the
metadata) as float32. Its metadata header carries the provenance (`META_KEYS`), so a file explains itself and `prune`
can tell a current file from a stale one without importing a model.

Nothing here imports torch or MLX: the header is the JSON at the start of a safetensors file, read with the standard
library (`fastforks cache mlx` lists and prunes on any machine).
"""

from __future__ import annotations

import contextlib
import json
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastforks._core.files import cache_root

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from fastforks._core.cards import ModelCard, Variant

__all__ = [
    "DTYPE_TAGS",
    "FORMAT",
    "META_KEYS",
    "Entry",
    "cache_dir",
    "entries",
    "file_name",
    "prune",
    "read_header",
    "stale_reason",
    "table",
]

FORMAT = "fastforks-mlx-1"
"""`format` of the files this version writes: a file without it (or with another) is stale."""

DTYPE_TAGS = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}
"""Run dtype -> its tag in the file name."""

META_KEYS = (
    "format",  # FORMAT
    "variant",  # model id, e.g. "rtmdet-x-coco"
    "task",
    "family",
    "revision",  # the variant's revision
    "weights",  # the original weight file's name
    "weights_sha256",  # its SHA-256 (the card's)
    "layout",  # the port's layout tag (with the source hash where the port gives sources)
    "layout_name",  # the layout as the port names it, without the source hash
    "dtype",  # the run dtype the file is stored in
    "fp32",  # JSON list of the arrays kept in float32 although `dtype` is not (empty: none)
    "fastforks",  # the fastforks version that wrote it
    "mlx",  # the MLX version that wrote it
    "created",  # UTC, ISO 8601
)
"""The provenance keys every file's metadata holds (the port's own configuration may follow as `config`, JSON)."""

_HEADER_MAX = 100 << 20  # a safetensors header larger than this is no header (the format caps it at 100 MB)


def cache_dir() -> Path:
    """`<FASTFORKS_CACHE>/mlx`: the only directory `prune` deletes in."""
    return cache_root() / "mlx"


def file_name(variant: str, revision: int, sha256: str, layout: str, dtype: str) -> str:
    """The cache file's name (see the module docstring); `dtype` a run dtype name (`"float16"`)."""
    return f"{variant}-r{revision}-{sha256[:12]}-{layout}-{DTYPE_TAGS[dtype]}.safetensors"


def read_header(path: Path) -> dict[str, str] | None:
    """The `__metadata__` of the safetensors file `path` ({} without one), None if it is no safetensors file."""
    try:
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) < 8:  # noqa: PLR2004 - the header length field
                return None
            (n,) = struct.unpack("<Q", raw)
            if n > _HEADER_MAX:
                return None
            header = json.loads(f.read(n))
    except (OSError, ValueError):
        return None
    meta = header.get("__metadata__", {}) if isinstance(header, dict) else None
    return {str(k): str(v) for k, v in meta.items()} if isinstance(meta, dict) else None


@dataclass(frozen=True)
class Entry:
    """One file of the MLX cache.

    Attributes:
        path: The file.
        size: Its size in bytes.
        meta: Its provenance metadata ({} for a file written before the metadata existed, None if unreadable).
    """

    path: Path
    size: int
    meta: dict[str, str] | None

    @property
    def model(self) -> str:
        """`variant@revision` from the metadata, or the file name's first part."""
        if self.meta and "variant" in self.meta:
            return f"{self.meta['variant']}@{self.meta.get('revision', '?')}"
        return self.path.name.split("-r", 1)[0]

    @property
    def dtype(self) -> str:
        """The stored dtype, `"?"` if the metadata does not say."""
        return (self.meta or {}).get("dtype", "?")


def entries(root: Path | None = None) -> list[Entry]:
    """Every `*.safetensors` file under `root` (default `cache_dir()`), sorted by path."""
    root = cache_dir() if root is None else root
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.rglob("*.safetensors")):
        if p.name.startswith(".") or not p.is_file():  # an in-flight `.pid.thread.name` write
            continue
        out.append(Entry(p, p.stat().st_size, read_header(p)))
    return out


def stale_reason(  # noqa: PLR0911 - one return per reason
    entry: Entry, models: Mapping[str, tuple[ModelCard, Variant]], newest: Mapping[tuple, str]
) -> str:
    """Why `entry` is stale, `""` if it is current.

    Stale: no provenance (written before it existed, or another format); the variant is no longer installed; its
    revision or weight file changed; another file of the same variant, weight file, dtype and layout name was written
    later (a port whose layout tag carries a source hash: the older one is from older code).
    """
    meta = entry.meta
    if not meta or meta.get("format") != FORMAT:
        return "no provenance (an older cache format)"
    found = models.get(meta.get("variant", ""))
    if found is None:
        return "variant not installed"
    card, variant = found
    if card.task != meta.get("task") or card.family != meta.get("family"):
        return "variant moved to another family"
    if str(variant.revision) != meta.get("revision"):
        return f"revision {meta.get('revision')} (current {variant.revision})"
    if meta.get("weights_sha256") not in {w.sha256 for w in variant.weights}:
        return "weight file no longer the variant's"
    key = _group(meta)
    if newest.get(key, entry.path.name) != entry.path.name:
        return f"superseded by {newest[key]}"
    return ""


def _group(meta: Mapping[str, str]) -> tuple[str, ...]:
    """What two files of one port share when one supersedes the other (all but the layout tag)."""
    return tuple(meta.get(k, "") for k in ("variant", "revision", "weights_sha256", "dtype", "layout_name"))


def _newest(found: Iterable[Entry]) -> dict[tuple, str]:
    """Per `_group` (variant, revision, weight file, dtype, layout name), the file name with the latest `created`.

    A layout without sources has one file per group (its tag is the layout name); with sources, the files of older
    port code, and a development file written under `FASTFORKS_MLX_CACHE_SRC=0` (no hash), share the group.
    """
    best: dict[tuple, tuple[str, str]] = {}
    for e in found:
        meta = e.meta or {}
        if meta.get("format") != FORMAT:
            continue
        key = _group(meta)
        stamp = (meta.get("created", ""), e.path.name)
        if key not in best or stamp > best[key]:
            best[key] = stamp
    return {k: v[1] for k, v in best.items()}


def prune(
    models: Mapping[str, tuple[ModelCard, Variant]],
    *,
    dry_run: bool = False,
    root: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> list[tuple[Entry, str]]:
    """Delete the stale files (`stale_reason`) under `root` (default `cache_dir()`); returns them with the reasons.

    Only regular `*.safetensors` files below that directory are ever deleted (a symlink or a path that resolves
    elsewhere is refused), then directories left empty below it.
    """
    root = (cache_dir() if root is None else root).absolute()
    found = entries(root)
    newest = _newest(found)
    removed = []
    for e in found:
        reason = stale_reason(e, models, newest)
        if not reason:
            continue
        target = e.path
        if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
            msg = f"refusing to delete {target}: not a file inside {root}"
            raise ValueError(msg)
        if log is not None:
            log(f"{'would remove' if dry_run else 'removed'} {target.relative_to(root)} ({reason})")
        if not dry_run:
            target.unlink()
        removed.append((e, reason))
    if not dry_run:
        for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            with contextlib.suppress(OSError):
                d.rmdir()  # only empty ones; a non-empty directory raises and stays
    return removed


def _size(n: int) -> str:
    return f"{n / 2**20:,.0f} MB" if n >= 2**20 else f"{n / 2**10:,.0f} KB"


def table(found: list[Entry], models: Mapping[str, tuple[ModelCard, Variant]], root: Path | None = None) -> list[str]:
    """The lines `fastforks cache mlx` prints: file -> model, dtype, size, and `stale: reason` where stale."""
    root = cache_dir() if root is None else root
    newest = _newest(found)
    lines = []
    for e in found:
        reason = stale_reason(e, models, newest)
        line = f"{e.path.relative_to(root)}  {e.model}  {e.dtype}  {_size(e.size)}"
        lines.append(f"{line}  stale: {reason}" if reason else line)
    total = sum(e.size for e in found)
    lines.append(f"{len(found)} files, {_size(total)} under {root}")
    return lines
