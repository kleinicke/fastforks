"""Regression tests for the 2026-10-31 security review (M1-M3, L1-L4, L6)."""

from __future__ import annotations

import hashlib
import http.server
import io
import os
import re
import stat
import time
from pathlib import Path

import numpy as np
import pytest

from fastforks._core import checkpoint, files, kernels
from fastforks._core.errors import KernelUnavailableError, WeightsDownloadError, WeightsError
from fastforks._core.files import FileSpec

needs_non_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="permission checks do not apply to root"
)


@pytest.fixture(autouse=True)
def _fresh_group_cache():
    kernels._own_group.cache_clear()
    yield
    kernels._own_group.cache_clear()


def _spec(data: bytes, url: str | None = None) -> FileSpec:
    return FileSpec("w.bin", hashlib.sha256(data).hexdigest(), len(data), url)


# M2


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setattr(files, "_INSECURE_OK", False)
    monkeypatch.delenv(files.HF_ENDPOINT_ENV, raising=False)


@pytest.mark.parametrize("url", ["http://example.org/w.bin", "file:///etc/hostname"])
def test_insecure_schemes_refused_by_default(strict, tmp_path, url):
    with pytest.raises(WeightsDownloadError, match="refusing URL scheme"):
        files.download(url, tmp_path / "w.bin", _spec(b"x"))


def test_insecure_scheme_only_below_the_endpoint(strict, tmp_path, monkeypatch):
    src = tmp_path / "srv" / "w.bin"
    src.parent.mkdir()
    src.write_bytes(b"payload")
    monkeypatch.setenv(files.HF_ENDPOINT_ENV, (tmp_path / "srv").as_uri())
    files.download(src.as_uri(), tmp_path / "ok.bin", _spec(b"payload"))
    other = tmp_path / "other.bin"
    other.write_bytes(b"payload")
    with pytest.raises(WeightsDownloadError, match="refusing URL scheme"):
        files.download(other.as_uri(), tmp_path / "no.bin", _spec(b"payload"))


class _Downgrade(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(302)
        self.send_header("Location", "http://127.0.0.1:9/w.bin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


def test_https_to_http_redirect_refused():
    handler = files._HttpsRedirects()
    req = files.urllib.request.Request("https://example.org/a")
    with pytest.raises(files.urllib.error.HTTPError, match="refused") as e:  # fp given: py3.14 opens a tempfile else
        handler.redirect_request(req, io.BytesIO(), 302, "Found", {}, "http://example.org/b")
    e.value.close()
    ok = handler.redirect_request(req, None, 302, "Found", {}, "https://example.org/b")
    assert ok is not None


def test_cards_reject_http_urls():
    from fastforks._core import cards  # noqa: PLC0415
    from fastforks._core.errors import CardError  # noqa: PLC0415

    t = {"name": "w.bin", "sha256": "0" * 64, "size": 1, "url": "http://example.org/w.bin"}
    with pytest.raises(CardError, match="https"):
        cards._file_spec(t, "x", "fam")


# L1 / L2


def test_unsized_fetch_is_capped(monkeypatch, tmp_path):
    src = tmp_path / "w.bin"
    src.write_bytes(b"x" * 100)
    monkeypatch.setattr(files, "_UNSIZED_CAP", 10)
    with pytest.raises(files.WeightsHashError, match="more than"):
        files.download(src.as_uri(), tmp_path / "o.bin", FileSpec("w.bin", "0" * 64, None, src.as_uri()))


def test_cache_dirs_are_private_and_stale_downloads_swept(tmp_path):
    dest = tmp_path / "a" / "b" / "c" / "blob"
    src = tmp_path / "w.bin"
    src.write_bytes(b"payload")
    files.download(src.as_uri(), dest, _spec(b"payload", src.as_uri()))
    for d in (dest.parent, dest.parent.parent, dest.parent.parent.parent):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
    stamp = next((files.cache_root() / "stamps").rglob("*.json"))
    assert stat.S_IMODE(stamp.parent.stat().st_mode) == 0o700
    assert not list((files.cache_root() / "stamps").rglob("*.tmp"))
    stale, fresh = dest.parent / ".download.old", dest.parent / ".download.new"
    stale.mkdir()
    fresh.mkdir()
    old = time.time() - 3 * 86400
    os.utime(stale, (old, old))
    files.download(src.as_uri(), dest.with_name("blob2"), _spec(b"payload", src.as_uri()))
    assert not stale.exists()
    assert fresh.exists()


# M1


def _manifest(path, good, monkeypatch):
    spec = _spec(good)
    monkeypatch.setattr(kernels, "binary", lambda _m, _d: (path, "sm_80"))
    return kernels.KernelManifest("lib", "b1", (12, 8), "cubin", {"sm_80": spec}), spec


@needs_non_root
def test_kernel_in_private_dir_uses_the_stamp_path(monkeypatch, tmp_path):
    root = tmp_path / "kp"
    root.mkdir(mode=0o700)
    monkeypatch.setenv(kernels.KERNEL_PATH_ENV, str(root))
    path = root / "k.cubin"
    path.write_bytes(b"good")
    manifest, _ = _manifest(path, b"good", monkeypatch)
    assert kernels.verified_binary(manifest, None)[0] is None  # not re-read or re-hashed


@needs_non_root
def test_kernel_in_writable_dir_is_hashed_every_load_ignoring_stamps(monkeypatch, tmp_path):
    root = tmp_path / "kp"
    root.mkdir()
    root.chmod(0o777)
    monkeypatch.setenv(kernels.KERNEL_PATH_ENV, str(root))
    good = b"\x7fELF-good-kernel"
    path = root / "k.cubin"
    path.write_bytes(good)
    manifest, spec = _manifest(path, good, monkeypatch)
    files._remember(path, spec.sha256)
    assert kernels.verified_binary(manifest, None)[0] == good
    st = path.stat()  # same size and mtime, forged stamp still matches: the bytes are hashed anyway
    path.write_bytes(bytes(len(good)))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(KernelUnavailableError, match="does not match"):
        kernels.verified_binary(manifest, None)


def test_kernel_outside_any_root_is_not_private(monkeypatch, tmp_path):
    monkeypatch.delenv(kernels.KERNEL_PATH_ENV, raising=False)
    path = tmp_path / "elsewhere.so"
    path.write_bytes(b"x")
    assert not kernels._private(path)


# M3


def test_codecs_wrapper_is_latin1_only():
    assert checkpoint.latin1_encode("\x00\xff", "latin1") == b"\x00\xff"
    for bad in ("bz2", "hex", "zlib", "utf-8"):
        with pytest.raises(WeightsError):
            checkpoint.latin1_encode("a", bad)
    with pytest.raises(WeightsError):
        checkpoint.latin1_encode(b"a", "latin1")  # type: ignore[arg-type]


def test_numeric_scalar_only():
    f = np.dtype("<f4")
    assert checkpoint.numeric_scalar(f, np.float32(2).tobytes()) == 2
    with pytest.raises(WeightsError):
        checkpoint.numeric_scalar(np.dtype(object), object())
    with pytest.raises(WeightsError):
        checkpoint.numeric_scalar(f, "not bytes")


def test_restricted_load_size_cap(monkeypatch, tmp_path):
    p = tmp_path / "c.pt"
    p.write_bytes(b"0" * 100)
    monkeypatch.setattr(checkpoint, "_MAX_CHECKPOINT_BYTES", 10)
    with pytest.raises(WeightsError, match="refusing"):
        checkpoint.restricted_load(p)


# L4 / L6


def test_metric3d_torch_load_is_weights_only():
    src = Path(__file__).parents[1] / "src/fastforks/depth/models/metric3d/net/vit_dino_reg.py"
    calls = re.findall(r"torch\.load\([^)]*\)", src.read_text())
    assert calls
    assert all("weights_only=True" in c for c in calls)


def test_driver_libs_load_by_absolute_path(monkeypatch, tmp_path):
    (tmp_path / "libcuda.so.1").write_bytes(b"")
    assert kernels._system_lib("libcuda.so.1", [str(tmp_path)]) == str(tmp_path / "libcuda.so.1")
    assert kernels._system_lib("libcuda.so.1", [str(tmp_path / "none")]) == "libcuda.so.1"
