from __future__ import annotations

import dataclasses
import gc
import hashlib
import http.server
import io
import logging
import os
import sys
import tarfile
import threading
import urllib.error
import warnings
import zipfile

import pytest

from fastforks import WeightsDownloadError, WeightsHashError
from fastforks._core import files
from fastforks._core.files import FileSpec, locate

ENV = "FF_TEST_PATH"


def spec_for(data: bytes, url: str | None = None) -> FileSpec:
    return FileSpec("w.bin", hashlib.sha256(data).hexdigest(), len(data), url)


def cache_files():
    root = files.cache_root()
    return sorted(p for p in root.rglob("*") if p.is_file() and "stamps" not in p.relative_to(root).parts) + stamps()


def stamps():
    return sorted((files.cache_root() / "stamps").rglob("*.json"))


def test_search_path_both_layouts(tmp_path, monkeypatch):
    flat, grouped = tmp_path / "a", tmp_path / "b"
    (grouped / "fam").mkdir(parents=True)
    flat.mkdir()
    (grouped / "fam" / "w.bin").write_bytes(b"x")
    monkeypatch.setenv(ENV, f"{flat}:{grouped}")
    assert locate(spec_for(b"x"), group="fam", env_var=ENV) == grouped / "fam" / "w.bin"


def test_search_path_hash_mismatch_raises(tmp_path, monkeypatch):
    (tmp_path / "w.bin").write_bytes(b"y")
    monkeypatch.setenv(ENV, str(tmp_path))
    with pytest.raises(WeightsHashError, match="SHA-256"):
        locate(FileSpec("w.bin", "0" * 64), group="fam", env_var=ENV)


def test_hash_mismatch_message_names_cause_and_fix(tmp_path, monkeypatch):
    (tmp_path / "w.bin").write_bytes(b"y")
    monkeypatch.setenv(ENV, str(tmp_path))
    with pytest.raises(WeightsHashError) as e:
        locate(FileSpec("w.bin", "0" * 64), group="fam", env_var=ENV)
    text = str(e.value)
    assert "0" * 64 in text  # the full expected digest
    assert str(tmp_path / "w.bin") in text
    assert "Delete it" in text
    assert "fastforks download" in text


def test_size_mismatch_raises(tmp_path, monkeypatch):
    (tmp_path / "w.bin").write_bytes(b"yy")
    monkeypatch.setenv(ENV, str(tmp_path))
    with pytest.raises(WeightsHashError, match="size 2 != expected 1"):
        locate(spec_for(b"y"), group="fam", env_var=ENV)


def test_download_then_cached(tmp_path, monkeypatch):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"payload")
    spec = spec_for(b"payload", src.as_uri())
    p = locate(spec, group="fam", env_var=ENV)
    assert p == files.blob_path(spec.sha256)
    assert p.read_bytes() == b"payload"
    src.unlink()  # second lookup must not download
    assert locate(spec, group="fam", env_var=ENV) == p


def test_corrupt_cache_is_redownloaded(tmp_path, caplog):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"good")
    spec = spec_for(b"good", src.as_uri())
    cached = files.blob_path(spec.sha256)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"badd")
    with caplog.at_level(logging.WARNING):
        assert locate(spec, group="fam", env_var=ENV).read_bytes() == b"good"
    assert "re-downloading" in caplog.text


def test_bad_download_leaves_nothing(tmp_path):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"evil")
    spec = FileSpec("w.bin", hashlib.sha256(b"good").hexdigest(), None, src.as_uri())
    with pytest.raises(WeightsHashError):
        locate(spec, group="fam", env_var=ENV)
    assert not cache_files()


def test_download_errors(tmp_path):
    with pytest.raises(WeightsDownloadError, match="refusing URL scheme"):
        locate(spec_for(b"x", "ftp://host/w.bin"), group="fam", env_var=ENV)
    with pytest.raises(WeightsDownloadError, match="failed"):
        locate(spec_for(b"x", (tmp_path / "missing").as_uri()), group="fam", env_var=ENV)
    with pytest.raises(WeightsDownloadError, match="no download URL"):
        locate(spec_for(b"x"), group="fam", env_var=ENV)


def test_verification_is_remembered(tmp_path, monkeypatch):
    f = tmp_path / "w.bin"
    f.write_bytes(b"z")
    monkeypatch.setenv(ENV, str(tmp_path))
    spec = spec_for(b"z")
    locate(spec, group="fam", env_var=ENV)
    monkeypatch.setattr(files, "sha256_of", lambda _p: pytest.fail("re-hashed an unchanged file"))
    locate(spec, group="fam", env_var=ENV)


def _zip(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path.as_uri()


def test_zip_member_download(tmp_path):
    url = _zip(tmp_path / "models.zip", {"a.pth": b"aaa", "dir/b.pth": b"bbbb"})
    spec = FileSpec("b.pth", hashlib.sha256(b"bbbb").hexdigest(), 4, url, member="dir/b.pth")
    p = locate(spec, group="fam", env_var=ENV)
    assert p.read_bytes() == b"bbbb"
    assert cache_files() == [p, *stamps()]  # archive and partial files removed


def test_tar_gz_member_download(tmp_path):
    path = tmp_path / "models.tar.gz"
    with tarfile.open(path, "w:gz") as t:
        for name, data in {"a.pb": b"aaa", "dir/b.pb": b"bbbb"}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    spec = FileSpec("b.pb", hashlib.sha256(b"bbbb").hexdigest(), 4, path.as_uri(), member="dir/b.pb")
    p = locate(spec, group="fam", env_var=ENV)
    assert p.read_bytes() == b"bbbb"
    assert cache_files() == [p, *stamps()]
    gone = FileSpec("c.pb", "0" * 64, None, path.as_uri(), member="c.pb")
    with pytest.raises(WeightsDownloadError, match=r"no member 'c\.pb'"):
        locate(gone, group="fam", env_var=ENV)


def test_zip_member_errors_leave_nothing(tmp_path):
    url = _zip(tmp_path / "models.zip", {"a.pth": b"aaa"})
    missing = FileSpec("b.pth", "0" * 64, None, url, member="b.pth")
    with pytest.raises(WeightsDownloadError, match=r"no member 'b\.pth'"):
        locate(missing, group="fam", env_var=ENV)
    wrong = FileSpec("a.pth", "0" * 64, None, url, member="a.pth")
    with pytest.raises(WeightsHashError, match=r"\(a.pth\): SHA-256"):
        locate(wrong, group="fam", env_var=ENV)
    not_zip = tmp_path / "plain.bin"
    not_zip.write_bytes(b"x")
    with pytest.raises(WeightsDownloadError, match="failed"):
        locate(
            FileSpec("a.pth", "0" * 64, None, not_zip.as_uri(), member="a.pth"),
            group="fam",
            env_var=ENV,
        )
    assert not cache_files()


def test_search_path_skips_mismatch(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "fam").mkdir(parents=True)
    b.mkdir()
    (a / "w.bin").write_bytes(b"other model's file of the same name")
    (b / "w.bin").write_bytes(b"x")
    monkeypatch.setenv(ENV, f"{a}:{b}")
    assert locate(spec_for(b"x"), group="fam", env_var=ENV) == b / "w.bin"
    (a / "fam" / "w.bin").write_bytes(b"x")  # the family directory wins over a flat directory
    assert locate(spec_for(b"x"), group="fam", env_var=ENV) == a / "fam" / "w.bin"


def test_zip_siblings_share_one_fetch(tmp_path, monkeypatch):
    members = {"a.pth": b"aaa", "b.pth": b"bbbb", "c.pth": b"cc"}
    url = _zip(tmp_path / "models.zip", members)
    specs = [FileSpec(n, hashlib.sha256(d).hexdigest(), len(d), url, member=n) for n, d in members.items()]
    bad = FileSpec("d.pth", "1" * 64, None, url, member="c.pth")  # wrong hash: skipped, not fatal
    fetches = []
    fetch = files._fetch
    monkeypatch.setattr(files, "_fetch", lambda u, out, limit: fetches.append(u) or fetch(u, out, limit))
    first = locate(specs[0], group="fam", env_var=ENV, siblings=[*specs, bad])
    assert first.read_bytes() == b"aaa"
    for s, (_, data) in zip(specs, members.items(), strict=True):
        assert locate(s, group="fam", env_var=ENV, allow_download=False).read_bytes() == data
    assert fetches == [url]
    assert not files.blob_path(bad.sha256).exists()


def test_same_bytes_share_a_blob(tmp_path):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"same")
    a = locate(spec_for(b"same", src.as_uri()), group="fam", env_var=ENV)
    src.unlink()
    other = FileSpec("other-name.pth", hashlib.sha256(b"same").hexdigest(), 4, src.as_uri())
    assert locate(other, group="another", env_var=ENV) == a


def test_stamp_detects_changes(tmp_path, monkeypatch):
    f = tmp_path / "w.bin"
    f.write_bytes(b"z")
    monkeypatch.setenv(ENV, str(tmp_path))
    locate(spec_for(b"z"), group="fam", env_var=ENV)
    st = f.stat()
    f.write_bytes(b"q")  # same size, new mtime: hashed again
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1))  # coarse fs clocks may not tick between the writes
    with pytest.raises(WeightsHashError, match="SHA-256"):
        locate(spec_for(b"z"), group="fam", env_var=ENV)


def test_blob_path_rejects_non_hex():
    with pytest.raises(ValueError, match="hex"):
        files.blob_path("../../etc")


def test_downloads_stop_at_the_recorded_size(tmp_path):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"x" * 10)
    big = FileSpec("w.bin", hashlib.sha256(b"x" * 4).hexdigest(), 4, src.as_uri())
    with pytest.raises(WeightsHashError, match="more than the expected 4 bytes"):
        locate(big, group="fam", env_var=ENV)
    url = _zip(tmp_path / "models.zip", {"a.pth": b"a" * 100})
    member = FileSpec("a.pth", "0" * 64, 4, url, member="a.pth")  # declared 100 bytes: not read
    with pytest.raises(WeightsHashError, match=r"\(a.pth\): size 100 != expected 4"):
        locate(member, group="fam", env_var=ENV)
    assert not cache_files()


def test_pinned_archive_is_checked_before_it_is_opened(tmp_path, monkeypatch):
    path = tmp_path / "models.zip"
    url = _zip(path, {"a.pth": b"aaa"})
    good = hashlib.sha256(path.read_bytes()).hexdigest()
    a = hashlib.sha256(b"aaa").hexdigest()
    spec = FileSpec("a.pth", a, 3, url, member="a.pth", archive_sha256="0" * 64, archive_size=path.stat().st_size)
    with monkeypatch.context() as m:  # `monkeypatch.undo()` would also drop the suite-wide autouse patches
        m.setattr(files, "_members", lambda *_: pytest.fail("opened an unverified archive"))
        with pytest.raises(WeightsHashError, match="archive SHA-256"):
            locate(spec, group="fam", env_var=ENV)
        small = FileSpec("a.pth", a, 3, url, member="a.pth", archive_sha256=good, archive_size=10)
        with pytest.raises(WeightsHashError, match="more than the expected 10 bytes"):
            locate(small, group="fam", env_var=ENV)
    ok = FileSpec("a.pth", a, 3, url, member="a.pth", archive_sha256=good, archive_size=path.stat().st_size)
    assert locate(ok, group="fam", env_var=ENV).read_bytes() == b"aaa"


def test_tar_siblings_in_one_pass(tmp_path):
    path = tmp_path / "models.tar.gz"
    members = {"a.pb": b"aaa", "b.pb": b"bbbb", "c.pb": b"cc"}
    with tarfile.open(path, "w:gz") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    specs = [FileSpec(n, hashlib.sha256(d).hexdigest(), len(d), path.as_uri(), member=n) for n, d in members.items()]
    locate(specs[1], group="fam", env_var=ENV, siblings=specs)
    for s, data in zip(specs, members.values(), strict=True):
        assert locate(s, group="fam", env_var=ENV, allow_download=False).read_bytes() == data


def test_hub_url_rewrites_only_our_repos(monkeypatch):
    ours = "https://huggingface.co/fastforks/kernels/resolve/main/raft/raft-sm_120.cubin"
    golden = "https://huggingface.co/datasets/fastforks/reference-outputs/resolve/main/raft@1.npz"
    other = "https://huggingface.co/someone/fastforks/resolve/main/w.pth"
    monkeypatch.delenv(files.HF_ENDPOINT_ENV, raising=False)
    assert files.hub_url(ours) == ours
    monkeypatch.setenv(files.HF_ENDPOINT_ENV, "file:///srv/hf/")
    assert files.hub_url(ours) == "file:///srv/hf/fastforks/kernels/resolve/main/raft/raft-sm_120.cubin"
    assert files.hub_url(golden) == "file:///srv/hf/datasets/fastforks/reference-outputs/resolve/main/raft@1.npz"
    assert files.hub_url(other) == other
    assert files.hub_url("https://example.com/fastforks/x") == "https://example.com/fastforks/x"


def test_hf_token_goes_only_to_our_repos(monkeypatch):
    ours = "https://huggingface.co/fastforks/kernels/resolve/main/raft/raft-sm_120.cubin"
    golden = "https://huggingface.co/datasets/fastforks/reference-outputs/resolve/main/raft@1.npz"
    others = [
        "https://huggingface.co/someone/fastforks/resolve/main/w.pth",
        "https://huggingface.co.example.com/fastforks/x",
        "https://example.com/fastforks/x",
        "file:///srv/hf/fastforks/kernels/resolve/main/raft/raft-sm_120.cubin",
    ]
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert not files._request(ours).has_header("Authorization")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    for url in (ours, golden):
        req = files._request(url)
        assert req.unredirected_hdrs == {"Authorization": "Bearer hf_test"}
        assert req.headers == {}  # not copied onto a redirect to the file host
    for url in others:
        assert not files._request(url).has_header("Authorization")


def test_download_from_a_standin(tmp_path, monkeypatch):
    blob = tmp_path / "hf" / "fastforks" / "mirrors" / "resolve" / "main" / "fam" / "w.bin"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"mirrored")
    monkeypatch.setenv(files.HF_ENDPOINT_ENV, (tmp_path / "hf").as_uri())
    spec = spec_for(b"mirrored", files.MIRROR_URL.format(group="fam", name="w.bin"))
    assert locate(spec, group="fam", env_var=ENV).read_bytes() == b"mirrored"


@pytest.mark.parametrize("primary", ["missing", "wrong bytes"])
def test_mirror_is_the_fallback(tmp_path, caplog, primary):
    remote, mirror = tmp_path / "remote.bin", tmp_path / "mirror.bin"
    if primary == "wrong bytes":
        remote.write_bytes(b"changed")
    mirror.write_bytes(b"payload")
    spec = dataclasses.replace(spec_for(b"payload", remote.as_uri()), mirror=mirror.as_uri())
    with caplog.at_level(logging.WARNING):
        assert locate(spec, group="fam", env_var=ENV).read_bytes() == b"payload"
    assert "trying the mirror" in caplog.text
    mirror.write_bytes(b"other")  # a bad mirror is an error too
    spec = dataclasses.replace(spec, sha256=hashlib.sha256(b"new").hexdigest(), size=3)
    with pytest.raises(WeightsHashError):
        locate(spec, group="fam", env_var=ENV)


def test_download_failure_says_what_to_do(tmp_path):
    spec = dataclasses.replace(spec_for(b"x", (tmp_path / "missing").as_uri()), mirror=(tmp_path / "gone").as_uri())
    with pytest.raises(WeightsDownloadError, match=r"fam: cannot download .*mirror: .*failed.*as <dir>/fam/") as e:
        locate(spec, group="fam", env_var=ENV)
    assert ENV in str(e.value)


def _http_error(url: str, code: int, msg: str) -> urllib.error.HTTPError:
    """A fake response error, closed after the test (py3.14 wraps `fp` in a tempfile; gc-time cleanup warns)."""
    err = urllib.error.HTTPError(url, code, msg, {}, io.BytesIO())
    _FAKE_ERRORS.append(err)
    return err


_FAKE_ERRORS: list[urllib.error.HTTPError] = []


@pytest.fixture(autouse=True)
def _close_fake_http_errors():
    yield
    while _FAKE_ERRORS:
        _FAKE_ERRORS.pop().close()


GATED = "https://huggingface.co/meta/gated-model/resolve/0123abc/sub/model.safetensors"


def _stub_hub(monkeypatch, tmp_path, content=b"w", error=None):
    """Stub `hf_hub_download`: returns the calls; writes `content` into the hub cache snapshot, or raises `error`."""
    import huggingface_hub  # noqa: PLC0415

    calls = []

    def fake(repo_id, filename, revision=None, **kw):
        calls.append((repo_id, filename, revision, kw))
        if error is not None:
            raise error
        target = tmp_path / "hf_hub" / "models--meta--gated-model" / "snapshots" / "abc" / filename
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    return calls


def _plain(monkeypatch, *, code=None, content=b"w"):
    """Stub the plain download: refused with HTTP `code`, else serving `content`. Returns the requested URLs."""
    urls = []

    def fetch(u, out, _limit):
        urls.append(u)
        if code is not None:
            raise _http_error(u, code, "refused")
        out.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    monkeypatch.setattr(files, "_fetch", fetch)
    return urls


def test_non_gated_hf_file_never_uses_huggingface_hub(tmp_path, monkeypatch):
    calls = _stub_hub(monkeypatch, tmp_path)
    urls = _plain(monkeypatch)
    path = locate(spec_for(b"w", GATED), group="fam", env_var=ENV)
    assert calls == []
    assert urls == [GATED]
    assert path.is_relative_to(tmp_path / "cache")  # our cache, as before


@pytest.mark.parametrize("code", [401, 403])
def test_refused_hf_file_goes_through_huggingface_hub(tmp_path, monkeypatch, code):
    calls = _stub_hub(monkeypatch, tmp_path)
    _plain(monkeypatch, code=code)
    path = locate(spec_for(b"w", GATED), group="fam", env_var=ENV)
    assert calls == [("meta/gated-model", "sub/model.safetensors", "0123abc", {})]
    assert path.is_relative_to(tmp_path / "hf_hub")  # verified and used in place, not copied
    assert path.read_bytes() == b"w"


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/fastforks/kernels/resolve/main/raft/k.cubin",
        "https://huggingface.co/datasets/fastforks/reference-outputs/resolve/main/raft@1.npz",
        "https://example.com/meta/gated-model/resolve/0123abc/w.bin",
    ],
)
def test_huggingface_hub_not_used_for_other_urls(tmp_path, monkeypatch, url):
    calls = _stub_hub(monkeypatch, tmp_path)
    _plain(monkeypatch, code=403)
    with pytest.raises(WeightsDownloadError):
        locate(spec_for(b"w", url), group="fam", env_var=ENV)
    assert calls == []


@pytest.mark.parametrize("var", ["FASTFORKS_OFFLINE", "HF_HUB_OFFLINE"])
def test_huggingface_hub_not_called_offline(tmp_path, monkeypatch, var):
    calls = _stub_hub(monkeypatch, tmp_path)
    urls = _plain(monkeypatch, code=403)
    monkeypatch.setenv(var, "1")
    with pytest.raises(WeightsDownloadError):
        locate(spec_for(b"w", GATED), group="fam", env_var=ENV)
    assert calls == []
    assert urls == []


def test_huggingface_hub_copy_is_hash_checked(tmp_path, monkeypatch):
    _stub_hub(monkeypatch, tmp_path, content=b"tampered")
    _plain(monkeypatch, code=401)
    with pytest.raises(WeightsDownloadError, match="gated or private"):
        locate(spec_for(b"w", GATED), group="fam", env_var=ENV)


def test_returned_path_is_used_even_outside_the_hub_cache_layout(tmp_path, monkeypatch):
    import huggingface_hub  # noqa: PLC0415

    elsewhere = tmp_path / "custom" / "w.bin"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"w")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_k: str(elsewhere))
    _plain(monkeypatch, code=403)
    assert locate(spec_for(b"w", GATED), group="fam", env_var=ENV) == elsewhere


def test_non_access_failure_of_huggingface_hub_keeps_the_download_error(tmp_path, monkeypatch):
    _stub_hub(monkeypatch, tmp_path, error=OSError("network down"))
    _plain(monkeypatch, code=403)
    with pytest.raises(WeightsDownloadError, match="gated or private"):
        locate(spec_for(b"w", GATED), group="fam", env_var=ENV)


class _GatedError(Exception):  # stands in for huggingface_hub.errors.GatedRepoError
    pass


_GatedError.__name__ = "GatedRepoError"


@pytest.mark.parametrize("hub", [True, False])
def test_gated_hf_repo_says_what_to_do(tmp_path, monkeypatch, hub):
    if hub:
        _stub_hub(monkeypatch, tmp_path, error=_GatedError("secret_token_value"))
    else:
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # not installed
    _plain(monkeypatch, code=401)
    spec = spec_for(b"w", GATED)
    with pytest.raises(
        WeightsDownloadError, match=r"fam: w.bin, 0 KB is in the gated or private Hugging Face repo meta/gated-mo"
    ) as e:
        locate(spec, group="fam", env_var=ENV)
    text = str(e.value)
    for part in (
        "https://huggingface.co/meta/gated-model, ",
        "`hf auth login`",
        "`huggingface_hub.login()`",
        "then load again",
        "hf download meta/gated-model sub/model.safetensors --revision 0123abc",
        "<dir>/fam/w.bin",
        "weights=PATH",
        spec.sha256,
        ENV,
    ):
        assert part in text
    assert ("pip install huggingface_hub" in text) == (not hub)
    assert "secret_token_value" not in text


def test_fastforks_never_reads_a_token_file(tmp_path, monkeypatch):
    (tmp_path / "hf_home").mkdir()
    (tmp_path / "hf_home" / "token").write_text("hf_file_token")
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")
    for url in (GATED, "https://example.com/x"):
        assert not files._request(url).has_header("Authorization")


class _Redirects(http.server.BaseHTTPRequestHandler):
    """`/file` is the payload; `/to-file`, `/to-missing` and `/loop` redirect (302), everything else is a 404."""

    routes = {"/to-file": "/file", "/to-missing": "/missing", "/loop": "/loop"}  # noqa: RUF012 - read only

    def do_GET(self):
        if self.path in self.routes:
            self.send_response(302)
            self.send_header("Location", self.routes[self.path])
            body = b"moved"
        elif self.path == "/file":
            self.send_response(200)
            body = b"payload"
        else:
            self.send_response(404)
            body = b"not here"
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - http.server API
        pass


@pytest.mark.slow
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_http_responses_are_closed(tmp_path, monkeypatch):
    """Redirected, failed and looping HTTP fetches leave no unclosed socket (ResourceWarning at gc)."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Redirects)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always", ResourceWarning)
            files.download(f"{base}/to-file", tmp_path / "w.bin", spec_for(b"payload"))
            assert (tmp_path / "w.bin").read_bytes() == b"payload"
            for path, match in (("to-missing", "404"), ("loop", "infinite loop"), ("missing", "404")):
                with pytest.raises(WeightsDownloadError, match=match) as e:
                    files.download(f"{base}/{path}", tmp_path / "x.bin", spec_for(b"payload"))
                http_error = e.value.__cause__
                assert isinstance(http_error, urllib.error.HTTPError)
                assert http_error.fp.closed, path  # not left to gc, which may finalize the socket first
            gc.collect()
    finally:
        server.shutdown()
        server.server_close()
    assert not [u.exc_value for u in unraisable]


def _hf_snapshot(hub, rev: str, data: bytes):
    path = hub / "models--org--name" / "snapshots" / rev / "sub" / "w.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path


def _hub_files(root):
    return sorted((p, p.read_bytes()) for p in root.rglob("*") if p.is_file())


def test_hf_hub_cache_is_read_in_place(tmp_path, monkeypatch):
    hub = tmp_path / "hf"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    path = _hf_snapshot(hub, "0123abc", b"payload")
    spec = spec_for(b"payload", "https://huggingface.co/org/name/resolve/0123abc/sub/w.bin")
    before = _hub_files(hub)
    assert locate(spec, group="fam", env_var=ENV, allow_download=False) == path
    assert not files.blob_path(spec.sha256).exists()  # not copied
    assert _hub_files(hub) == before  # read only


def test_hf_hub_cache_branch_revision_and_hf_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    hub = tmp_path / "home" / "hub"
    _hf_snapshot(hub, "old", b"stale!!")
    path = _hf_snapshot(hub, "c0ffee", b"payload")
    (hub / "models--org--name" / "refs").mkdir()
    (hub / "models--org--name" / "refs" / "main").write_text("c0ffee")
    spec = spec_for(b"payload", "https://huggingface.co/org/name/resolve/main/sub/w.bin?download=true")
    assert locate(spec, group="fam", env_var=ENV, allow_download=False) == path


def test_hf_hub_cache_mismatch_falls_through_to_the_download(tmp_path, monkeypatch, caplog):
    hub = tmp_path / "hf"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    _hf_snapshot(hub, "main", b"PAYLOAD")  # same size, other bytes
    spec = spec_for(b"payload", "https://huggingface.co/org/name/resolve/main/sub/w.bin")
    fetched = []

    def fake_download(url, dest, spec, *, also=(), burst=None):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"payload")

    monkeypatch.setattr(files, "download", fake_download)
    before = _hub_files(hub)
    with caplog.at_level(logging.INFO, logger="fastforks"):
        assert locate(spec, group="fam", env_var=ENV) == files.blob_path(spec.sha256)
    assert fetched == [spec.url]
    assert "not using" in caplog.text
    assert _hub_files(hub) == before


def test_torch_hub_cache_by_file_name(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    path = tmp_path / "torch" / "hub" / "checkpoints" / "w-1234.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"payload")
    spec = spec_for(b"payload", "https://dl.fbaipublicfiles.com/models/w-1234.pth")
    assert locate(spec, group="fam", env_var=ENV, allow_download=False) == path
    other = spec_for(b"other", "https://dl.fbaipublicfiles.com/models/w-1234.pth")
    with pytest.raises(WeightsDownloadError, match="downloads are disabled"):
        locate(other, group="fam", env_var=ENV, allow_download=False)


def test_read_caches_are_used_in_place(tmp_path, monkeypatch):
    """A blob of a `FASTFORKS_READ_CACHES` cache is used where it is (also as an archive member); others download."""
    other = tmp_path / "other"
    monkeypatch.setenv("FASTFORKS_READ_CACHES", f"{tmp_path / 'missing'}{os.pathsep}{other}")
    spec = dataclasses.replace(spec_for(b"payload", "https://example.com/a.zip"), member="w.bin")
    blob = files.blob_path(spec.sha256, other)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"payload")
    assert files.present(spec, "fam", ENV)
    assert locate(spec, group="fam", env_var=ENV, allow_download=False) == blob
    assert not files.blob_path(spec.sha256).exists()  # not copied into FASTFORKS_CACHE
    with pytest.raises(WeightsDownloadError, match="not found"):
        locate(spec_for(b"other"), group="fam", env_var=ENV, allow_download=False)


def test_legacy_huggingface_hub_cache_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))
    path = _hf_snapshot(tmp_path / "legacy", "0123abc", b"payload")
    spec = spec_for(b"payload", "https://huggingface.co/org/name/resolve/0123abc/sub/w.bin")
    assert locate(spec, group="fam", env_var=ENV, allow_download=False) == path


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs directory permissions that apply")
def test_unreadable_hub_caches_are_skipped(tmp_path, monkeypatch):
    hub, torch_home = tmp_path / "hf", tmp_path / "torch"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    _hf_snapshot(hub, "main", b"payload")
    (torch_home / "hub" / "checkpoints").mkdir(parents=True)
    locked = [hub / "models--org--name", torch_home / "hub"]
    for d in locked:
        d.chmod(0)
    try:
        for url in ("https://huggingface.co/org/name/resolve/main/sub/w.bin", "https://example.com/w.bin"):
            spec = spec_for(b"payload", url)
            with pytest.raises(WeightsDownloadError, match="downloads are disabled"):
                locate(spec, group="fam", env_var=ENV, allow_download=False)
    finally:
        for d in locked:
            d.chmod(0o755)


@pytest.mark.parametrize(
    ("code", "text"), [(401, "refuses access"), (403, "refuses access"), (404, "no such file"), (503, "Retry later")]
)
def test_http_status_decides_whether_to_retry(monkeypatch, code, text):
    """Auth / missing-file errors do not say "Retry later"; transient ones do; all name the by-hand route."""

    def fail(u, _out, _limit):
        raise _http_error(u, code, "x")

    monkeypatch.setattr(files, "_fetch", fail)
    spec = spec_for(b"w", "https://example.org/w.bin")
    with pytest.raises(WeightsDownloadError, match=rf"fam: cannot download.*{text}.*<dir>/fam/w.bin") as e:
        locate(spec, group="fam", env_var=ENV)
    assert ("Retry later" in str(e.value)) == (code == 503)
