"""Every card's weights resolve offline from the default locations (fastforks blob cache, Hugging Face hub cache,
torch hub cache); families that share a file pin identical bytes."""

from __future__ import annotations

import collections
import os
import shutil
import tempfile
import urllib.parse
from pathlib import Path

import pytest

from fastforks._core import files, registry


def _sized(path: Path, size: int) -> None:
    """A sparse file of `size` bytes (only the size is checked by `present`).

    POSIX: `truncate` extends sparsely. Windows: `truncate` (the CRT's `_chsize_s`) writes the zeros, GBs per
    weights file, so the file is marked sparse and only its last byte is written.
    """
    with path.open("wb") as f:
        if os.name != "nt" or size == 0:
            f.truncate(size)
            return
        import ctypes  # noqa: PLC0415 - Windows only
        import msvcrt  # noqa: PLC0415

        fsctl_set_sparse = 0x900C4
        returned = ctypes.c_ulong()
        kernel32 = ctypes.windll.kernel32  # ty: ignore[unresolved-attribute] - Windows only
        if not kernel32.DeviceIoControl(
            msvcrt.get_osfhandle(f.fileno()), fsctl_set_sparse, None, 0, None, 0, ctypes.byref(returned), None
        ):
            raise ctypes.WinError()  # ty: ignore[unresolved-attribute] - Windows only
        f.seek(size - 1)
        f.write(b"\0")


def _specs():
    seen = {}
    for task in registry.TASKS:
        for card, v in registry.cards(task).values():
            for s in v.weights:
                seen[(card.family, s.name, s.sha256)] = s
    return seen


def test_same_url_or_digest_means_same_bytes():
    by_url, by_sha = collections.defaultdict(set), collections.defaultdict(set)
    for s in _specs().values():
        if s.url and s.member is None:
            by_url[s.url].add((s.sha256, s.size))
        by_sha[s.sha256].add(s.size)
    assert not {u: v for u, v in by_url.items() if len(v) > 1}  # e.g. insid3 and dinov3 share one DINOv3 file
    assert not {h: v for h, v in by_sha.items() if len(v - {None}) > 1}


@pytest.fixture
def short_tmp(tmp_path):
    """`tmp_path`, or on Windows a shorter directory: hub snapshot paths reach 171 characters below it, and pytest's
    temporary directory is long enough to pass the 260-character `MAX_PATH` without long-path support."""
    if os.name != "nt":
        yield tmp_path
        return
    root = Path(tempfile.mkdtemp(prefix="ff"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_every_spec_resolves_from_each_default_location(short_tmp, monkeypatch):
    hub, torch_home, cache = short_tmp / "hub", short_tmp / "torch", short_tmp / "cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    monkeypatch.setenv("FASTFORKS_CACHE", str(cache))
    monkeypatch.delenv("FASTFORKS_WEIGHTS_PATH", raising=False)
    wrong = []
    for (family, _n, _h), s in _specs().items():
        if s.size is None or s.member is not None:
            continue
        assert not files.present(s, family, "FASTFORKS_WEIGHTS_PATH")
        places = [files.blob_path(s.sha256)]
        if hf := files._HF_FILE.fullmatch((s.url or "").split("?", maxsplit=1)[0]):
            repo = hub / f"models--{hf['repo'].replace('/', '--')}" / "snapshots" / hf["rev"]
            places.append(repo / urllib.parse.unquote(hf["file"]))
        for p in places:
            p.parent.mkdir(parents=True, exist_ok=True)
            _sized(p, s.size)
            if not files.present(s, family, "FASTFORKS_WEIGHTS_PATH"):
                wrong.append((family, s.name, str(p)))
            p.unlink()
    assert not wrong
