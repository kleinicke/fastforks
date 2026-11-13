"""Hash-checked file resolution shared by weights and kernel binaries.

Lookup order for a file `name` of a `group` (e.g. a model family or a kernel library):
1. every directory in the search-path environment variable (`FASTFORKS_WEIGHTS_PATH` / `FASTFORKS_KERNEL_PATH`),
   as `<dir>/<group>/<name>` or `<dir>/<name>`; these are read-only and never copied;
2. the content-addressed user cache, `<cache>/blobs/sha256/<ab>/<sha256>` (two variants or libraries that ship the
   same bytes share one blob, and same-named files of different families cannot collide), then the same layout in
   every directory of `FASTFORKS_READ_CACHES` (other fastforks caches, read in place, never written to: an
   application that moves `FASTFORKS_CACHE` into its own models folder lists the user cache there);
3. a copy another tool already downloaded, read in place and never written to: the Hugging Face hub cache
   (`HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`; `models--<org>--<name>/snapshots/*/<file>`)
   for `huggingface.co/<repo>/resolve/<rev>/<file>` URLs, and the torch hub cache (`TORCH_HOME/hub/checkpoints`,
   else `~/.cache/torch/hub/checkpoints`) by the URL's file name; a copy that does not verify is skipped;
4. a download from the recorded URL into the cache, or, if that fails, from our mirror (`FileSpec.mirror`, when the
   card has one). A zip/tar.gz archive is fetched once (checked against its own SHA-256 when the card pins it):
   every member the caller lists (`siblings`, e.g. all variants of a family) is extracted from that fetch, and the
   archive itself is not kept. No download is copied past its recorded size.

Every file is checked against its SHA-256. A successful check is remembered in a small stamp file per path
(`<cache>/stamps/`, keyed by the resolved path, holding size and mtime), so large weights are hashed once, not on
every load, and concurrent processes never rewrite a shared database.

With `FASTFORKS_OFFLINE=1` (or `HF_HUB_OFFLINE=1`) nothing is downloaded: steps 1 to 3 run as usual and a file that
is not found raises an error naming the file and the `fastforks download` command that fetches it.

Our own Hugging Face repos (`huggingface.co/fastforks/...`: kernel binaries, golden outputs, weight mirrors) can be
redirected to a stand-in with the same layout: `FASTFORKS_HF_ENDPOINT=file:///path/to/standin` (or another server)
replaces `https://huggingface.co` in their URLs. Other URLs are never rewritten. For private repos (pre-release),
`HF_TOKEN` is sent as a bearer token to those URLs only, and not along a redirect (the file host gets none).

fastforks never handles your Hugging Face token. A file of any other `huggingface.co` repo is first downloaded
plainly (no token). If that is refused (HTTP 401/403: gated or private) and `huggingface_hub` is installed, the file is
fetched with `huggingface_hub.hf_hub_download`, which uses your login (`hf auth login`) and puts it in the Hugging
Face cache; the file is checked there and used in place.
"""

from __future__ import annotations

import contextlib
import hashlib
import http
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

import platformdirs

from fastforks._core.errors import OptionsError, WeightsDownloadError, WeightsHashError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from http.client import HTTPMessage

log = logging.getLogger(__name__)

_CHUNK = 1 << 22
_INSECURE_OK = False  # tests only: allow http:// and file:// URLs without `FASTFORKS_HF_ENDPOINT`
_UNSIZED_CAP = 4 << 30  # fetch limit when a spec has no size (cards and manifests always have one)
_STALE_DOWNLOAD_S = 24 * 3600
HF_ENDPOINT_ENV = "FASTFORKS_HF_ENDPOINT"
OFFLINE_ENV = "FASTFORKS_OFFLINE"
READ_CACHES_ENV = "FASTFORKS_READ_CACHES"
_OUR_HF = re.compile(r"https://huggingface\.co/((?:datasets/)?fastforks/.+)")
MIRROR_URL = "https://huggingface.co/fastforks/mirrors/resolve/main/{group}/{name}"
_HF_FILE = re.compile(r"https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<file>.+)")


@dataclass(frozen=True)
class FileSpec:
    """A file identified by name and SHA-256, optionally downloadable.

    Attributes:
        name: File name, e.g. `"CH128NTR1.pth"`.
        sha256: Lower-case hex SHA-256 of the file.
        size: Size in bytes, or `None` if unknown.
        url: Download URL, or `None` if the file is not published.
        member: Path of the file inside the zip or tar.gz archive at `url`, or `None` if `url` is the file itself;
            `sha256` and `size` are those of the member.
        archive_sha256: SHA-256 of the archive at `url` (members only), checked before the archive is opened.
        archive_size: Size of the archive in bytes (members only); the fetch stops beyond it.
        mirror: URL of the same bytes (the member itself for an archive member) in our mirror repo, tried when `url`
            fails; `None` if not mirrored.
    """

    name: str
    sha256: str
    size: int | None = None
    url: str | None = None
    member: str | None = None
    archive_sha256: str | None = None
    archive_size: int | None = None
    mirror: str | None = None


_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off", ""))


def env_flag(name: str) -> bool:
    """A boolean environment variable: 1/true/yes/on or 0/false/no/off/empty (any case, spaces ignored).

    Raises:
        OptionsError: Any other value (fail closed: `FASTFORKS_OFFLINE=enable` must not silently go online).
    """
    raw = os.environ.get(name, "")
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    msg = f"{name}={raw!r}: use 1, true, yes or on to enable, 0, false, no, off or empty to disable"
    raise OptionsError(msg)


def offline() -> bool:
    """Whether downloads are forbidden: `FASTFORKS_OFFLINE` (see `env_flag`) or `HF_HUB_OFFLINE` (1, ON, YES, TRUE).

    `FASTFORKS_OFFLINE` is the canonical switch; `HF_HUB_OFFLINE` (the Hugging Face one, often set in containers
    anyway) is honoured too. `locate` is the only place that fetches anything, so it is the only reader.
    """
    return env_flag(OFFLINE_ENV) or os.environ.get("HF_HUB_OFFLINE", "").upper() in ("1", "ON", "YES", "TRUE")


_WARNED: set[str] = set()


def hub_url(url: str) -> str:
    """`url`, redirected to `FASTFORKS_HF_ENDPOINT` if set and `url` is in one of our Hugging Face repos."""
    endpoint = os.environ.get(HF_ENDPOINT_ENV)
    match = _OUR_HF.fullmatch(url)
    return f"{endpoint.rstrip('/')}/{match[1]}" if endpoint and match else url


def _scheme_allowed(url: str) -> bool:
    """`https://` always; `http://` and `file://` only below an explicit `FASTFORKS_HF_ENDPOINT` (local mirror)."""
    if url.startswith("https://"):
        return True
    if not url.startswith(("http://", "file://")):
        return False
    endpoint = os.environ.get(HF_ENDPOINT_ENV)
    return _INSECURE_OK or (bool(endpoint) and url.startswith(endpoint.rstrip("/") + "/"))


class _HttpsRedirects(urllib.request.HTTPRedirectHandler):
    """Follows redirects, but never from `https://` to another scheme."""

    def redirect_request(  # noqa: PLR0913, PLR0917 - urllib API
        self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage, newurl: str
    ) -> urllib.request.Request | None:
        if req.full_url.startswith("https://") and not newurl.startswith("https://"):
            raise urllib.error.HTTPError(req.full_url, code, f"redirect from https to {newurl!r} refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpsRedirects)


def _mkdir_private(path: Path) -> None:
    """`mkdir -p` that creates every missing directory with mode 0700 (the cache holds code that gets loaded)."""
    missing = [path, *path.parents]
    for d in reversed([d for d in missing if not d.exists()]):
        with contextlib.suppress(FileExistsError):
            d.mkdir(mode=0o700)


def _sweep_stale_downloads(parent: Path) -> None:
    """Remove `.download.*` directories older than a day (left behind by a killed process)."""
    try:
        for d in parent.glob(".download.*"):
            if time.time() - d.stat().st_mtime > _STALE_DOWNLOAD_S:
                shutil.rmtree(d, ignore_errors=True)
    except OSError:
        log.debug("could not sweep stale downloads in %s", parent)


def cache_root() -> Path:
    """Return the fastforks cache directory (`FASTFORKS_CACHE` or the platform user cache).

    Raises:
        OptionsError: `FASTFORKS_CACHE` names a file, or a path below one (TM-9: not a raw `NotADirectoryError`).
    """
    env = os.environ.get("FASTFORKS_CACHE")
    if not env:
        return Path(platformdirs.user_cache_dir("fastforks"))
    root = Path(env).expanduser().absolute()  # `~` expanded, a relative path fixed against the cwd once
    for d in (root, *root.parents):  # the deepest existing part must be a directory (a few cheap stats)
        if d.exists():
            if not d.is_dir():
                msg = f"FASTFORKS_CACHE={env}: {d} is not a directory; point it at a directory (created if missing)"
                raise OptionsError(msg)
            break
    return root


def read_caches() -> list[Path]:
    """Other fastforks caches whose blobs are used in place (`FASTFORKS_READ_CACHES`, `os.pathsep`-separated)."""
    return search_path(READ_CACHES_ENV)


def hf_hub_cache() -> Path:
    """The Hugging Face hub cache, as `huggingface_hub` finds it.

    `HF_HUB_CACHE`, else the older `HUGGINGFACE_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`.
    """
    if env := os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(env).expanduser()
    if env := os.environ.get("HF_HOME"):
        return Path(env).expanduser() / "hub"
    return _xdg_cache() / "huggingface" / "hub"


def torch_hub_checkpoints() -> Path:
    """Where `torch.hub` keeps downloaded weights: `<TORCH_HOME or ~/.cache/torch>/hub/checkpoints`."""
    env = os.environ.get("TORCH_HOME")
    return (Path(env).expanduser() if env else _xdg_cache() / "torch") / "hub" / "checkpoints"


def _xdg_cache() -> Path:
    env = os.environ.get("XDG_CACHE_HOME")
    return Path(env).expanduser() if env else Path.home() / ".cache"


def _other_caches(spec: FileSpec) -> Iterator[Path]:
    """Copies of `spec` that the Hugging Face hub or `torch.hub` may have downloaded already (not checked yet).

    Hugging Face: the snapshot of the URL's revision first (a branch or tag via `refs/`), then the other snapshots
    (the file may be unchanged across revisions). torch hub: `<checkpoints>/<file name of the URL>`. Archive members
    are not looked up. Directories that cannot be read are skipped. Before them: the blobs of `read_caches()`.
    """
    for root in read_caches():
        if _is_file(blob := blob_path(spec.sha256, root)):
            yield blob
    if spec.url is None or spec.member is not None:
        return
    url = urllib.parse.urlsplit(spec.url)
    if hf := _HF_FILE.fullmatch(spec.url.split("?", maxsplit=1)[0]):
        repo = hf_hub_cache() / f"models--{hf['repo'].replace('/', '--')}"
        file = urllib.parse.unquote(hf["file"])
        ref = repo / "refs" / hf["rev"]
        try:
            rev = ref.read_text().strip() if ref.is_file() else hf["rev"]
        except OSError:
            rev = hf["rev"]
        first = repo / "snapshots" / rev / file
        try:
            snapshots = sorted((repo / "snapshots").iterdir()) if (repo / "snapshots").is_dir() else []
        except OSError as e:
            log.debug("skipping the Hugging Face cache %s: %s", repo, e)
            snapshots = []
        yield from (p for p in [first, *(d / file for d in snapshots)] if _is_file(p))
    name = urllib.parse.unquote(url.path.rsplit("/", maxsplit=1)[-1])
    if name and not url.query and _is_file(torch_hub_checkpoints() / name):
        yield torch_hub_checkpoints() / name


def _is_file(path: Path) -> bool:
    """`path.is_file()`, `False` (logged at DEBUG) if a parent directory cannot be read."""
    try:
        return path.is_file()
    except OSError as e:
        log.debug("skipping %s: %s", path, e)
        return False


def _from_other_caches(spec: FileSpec) -> Path | None:
    """The first copy from `_other_caches` that verifies (read in place, never written to)."""
    seen: set[Path] = set()
    for p in _other_caches(spec):
        if p in seen:
            continue
        seen.add(p)
        try:
            verify(p, spec)
        except (WeightsHashError, OSError) as e:
            log.info("not using %s: %s", p, e)
        else:
            log.info("using %s from another tool's cache", p)
            return p
    return None


def present(spec: FileSpec, group: str, env_var: str) -> bool:
    """Whether a local copy of `spec` of the right size exists (search path, cache, other tools' caches); no hashing.

    A cheap pre-check (`list_models(runnable=)`); `locate` still verifies the copy it returns.
    """
    found = [p for d in search_path(env_var) for p in (d / group / spec.name, d / spec.name)]
    found += [blob_path(spec.sha256), *_other_caches(spec)]
    for p in found:
        try:
            if p.is_file() and (spec.size is None or p.stat().st_size == spec.size):
                return True
        except OSError:
            continue
    return False


def search_path(env_var: str) -> list[Path]:
    """Return the directories listed in `env_var` (`os.pathsep`-separated, `~` expanded, made absolute).

    An entry that is not a directory is kept (it may appear later) but logs one WARNING: a wrong separator
    (`a,b` instead of `a:b`) or a typo is otherwise silent.
    """
    dirs = [Path(p).expanduser().absolute() for p in os.environ.get(env_var, "").split(os.pathsep) if p]
    for d in dirs:
        key = f"{env_var}={d}"
        if key not in _WARNED and not d.is_dir():
            _WARNED.add(key)
            log.warning(
                "%s entry %s is not a directory (entries are separated by %r, relative ones resolve against the"
                " current directory)",
                env_var,
                d,
                os.pathsep,
            )
    return dirs


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


_HEX = re.compile(r"[0-9a-f]{64}")


def blob_path(sha256: str, root: Path | None = None) -> Path:
    """Cache path of the blob with this SHA-256 (in the cache `root`, default `cache_root()`).

    Raises:
        ValueError: `sha256` is not 64 lower-case hex digits.
    """
    if not _HEX.fullmatch(sha256):
        msg = f"not a lower-case hex SHA-256: {sha256!r}"
        raise ValueError(msg)
    return (root or cache_root()) / "blobs" / "sha256" / sha256[:2] / sha256


def _stamp_file(path: Path) -> Path:
    key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    return cache_root() / "stamps" / key[:2] / f"{key}.json"


def _stamp(path: Path, sha256: str) -> dict[str, object]:
    st = path.stat()
    return {"path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256}


def _remember(path: Path, sha256: str) -> None:
    stamp = _stamp_file(path)
    tmp: Path | None = None
    try:
        _mkdir_private(stamp.parent)
        fd, name = tempfile.mkstemp(dir=stamp.parent, suffix=".tmp")
        tmp = Path(name)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(_stamp(path, sha256)))
        tmp.replace(stamp)
        tmp = None
    except OSError:  # read-only cache: verification still succeeded, it is just not remembered
        log.debug("could not record verification of %s", path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def verify(path: Path, spec: FileSpec) -> None:
    """Check `path` against `spec` (size, then SHA-256 unless already verified unchanged).

    Raises:
        WeightsHashError: Size or hash mismatch.
    """
    if spec.size is not None and path.stat().st_size != spec.size:
        msg = f"{path}: size {path.stat().st_size} != expected {spec.size}"
        raise WeightsHashError(msg)
    try:
        known = json.loads(_stamp_file(path).read_text())
    except (OSError, ValueError):
        known = None
    if known == _stamp(path, spec.sha256):
        return
    digest = sha256_of(path)
    if digest != spec.sha256:
        msg = (
            f"{path}: SHA-256 {digest} != expected {spec.sha256}; the file is corrupt, "
            f"partial or not the pinned version. Delete it and re-run, or run `fastforks download MODEL`; "
            f"for your own weights use weights=PATH (not hash-checked)"
        )
        raise WeightsHashError(msg)
    _remember(path, spec.sha256)


_BURST_SECONDS = 120.0
_burst: tuple[int, float] | None = None  # (burst id, time of the last announced or silent download)


def _announce(burst: int | None) -> bool:
    """Whether a download logs its own line: always, except the 2nd+ file of one `burst` (same id, within 2 min)."""
    global _burst  # noqa: PLW0603 - one process-wide record
    if burst is None:
        return True
    now = time.monotonic()
    same = _burst is not None and _burst[0] == burst and now - _burst[1] < _BURST_SECONDS
    _burst = (burst, now)
    return not same


def download(
    url: str, dest: Path, spec: FileSpec, *, also: Sequence[tuple[FileSpec, Path]] = (), burst: int | None = None
) -> None:
    """Download `url` (or `spec.member` of the zip/tar.gz archive at `url`) to `dest` atomically, checking the SHA-256.

    No copy grows past the recorded size: the fetch stops after `spec.size` (or `spec.archive_size`) bytes, and an
    archive member whose declared size differs from `spec.size` is not read. A pinned archive
    (`spec.archive_sha256`) is checked before it is opened, so the archive readers only parse known bytes.

    Args:
        url: Source URL (`https`; `http` / `file` only below `FASTFORKS_HF_ENDPOINT`); our Hugging Face URLs go
            through `hub_url`.
        dest: Target path.
        spec: The file (`spec.member` set: `url` is a zip or tar.gz archive holding it).
        burst: Downloads sharing this id (kernel files of one model call or load) log one line, not one per file.
        also: More `(spec, dest)` members of the same archive to extract from the same fetch; one that is missing
            or does not verify is logged and skipped (it fails again when it is requested itself).

    Raises:
        WeightsDownloadError: Network or HTTP failure, or the archive has no such member.
        WeightsHashError: The downloaded bytes do not match `spec` (size or SHA-256).
    """
    url = hub_url(url)
    if not _scheme_allowed(url):
        msg = f"refusing URL scheme of {url!r}"
        raise WeightsDownloadError(msg)
    try:
        _mkdir_private(dest.parent)
        _sweep_stale_downloads(dest.parent)
        tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".download."))
    except OSError as e:  # read-only file system, no permission, disk full
        msg = (
            f"cannot write to the cache {dest.parent} ({e}); point FASTFORKS_CACHE at a writable directory, or"
            f" provide the files in FASTFORKS_WEIGHTS_PATH and set FASTFORKS_OFFLINE=1 to never download"
        )
        raise WeightsDownloadError(msg) from e
    n = spec.archive_size if spec.member else spec.size  # what goes over the wire: the whole archive (TM-8)
    size = f", {_human_size(n)}" if n else ""
    if _announce(burst):
        what = f"{url}{f' ({spec.member})' if spec.member else ''}{size}"
        what = (
            f"kernel files, starting with {what} (a model has several kernel libraries: more files follow)"
            if burst is not None
            else what
        )
        log.warning("downloading %s (once; cached in %s)", what, dest.parent)
    else:
        log.debug("downloading %s%s (part of the kernel download above)", url, size)
    try:
        if spec.member is None:
            _place(tmp / "file", _fetch(url, tmp / "file", _limit(spec.size)), spec, dest, url)
            return
        _from_archive(url, tmp, spec, dest, also)
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as e:  # URLError and HTTPError are OSErrors
        msg = f"download of {url} failed: {e}"
        raise WeightsDownloadError(msg) from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _from_archive(url: str, tmp: Path, spec: FileSpec, dest: Path, also: Sequence[tuple[FileSpec, Path]]) -> None:
    """Fetch the archive at `url` into `tmp` and extract `spec` (required) and the `also` members (best effort)."""
    digest = _fetch(url, tmp / "archive", _limit(spec.archive_size))
    if spec.archive_sha256 is not None and digest != spec.archive_sha256:
        msg = f"{url}: archive SHA-256 {digest} != expected {spec.archive_sha256}"
        raise WeightsHashError(msg)
    wanted = {spec.member or "": (spec, dest)}
    for s, d in also:
        if wanted.setdefault(s.member or "", (s, d))[0] is not s:  # one member has one SHA-256: the first spec wins
            log.warning("not caching a sibling: %s (%s) is already requested", url, s.member)
    for name, declared, src in _members(tmp / "archive", url, set(wanted)):
        s, d = wanted.pop(name)
        try:
            with src:
                _place(*_extract(src, declared, s, tmp, url), s, d, url)
        except WeightsHashError as e:
            if s is spec:
                raise
            log.warning("not caching a sibling: %s", e)
    if spec.member in wanted:
        msg = f"{url}: archive has no member {spec.member!r}"
        raise WeightsDownloadError(msg)
    for name in wanted:
        log.warning("not caching a sibling: %s has no member %r", url, name)


def _members(path: Path, url: str, names: set[str]) -> Iterator[tuple[str, int, IO[bytes]]]:
    """`(name, declared size, reader)` of the regular-file members `names` of the archive at `path`.

    tar.gz (by `url`) is read in one streaming pass; members are read by name, never extracted to a path.
    """
    if url.split("?", maxsplit=1)[0].endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r|gz") as tar:
            for info in tar:
                f = tar.extractfile(info) if info.isfile() and info.name in names else None
                if f is not None:
                    names.discard(info.name)  # a later duplicate name is ignored
                    yield info.name, info.size, f
        return
    with zipfile.ZipFile(path) as z:
        for name in sorted(names):
            try:
                info = z.getinfo(name)
            except KeyError:
                continue
            if not info.is_dir():
                yield name, info.file_size, z.open(info)


def _copy(src: IO[bytes], out: Path, limit: int | None, what: str) -> str:
    """Stream `src` into `out`, at most `limit` bytes; returns the SHA-256 of the bytes."""
    h, n = hashlib.sha256(), 0
    with out.open("wb") as f:
        while chunk := src.read(_CHUNK):
            n += len(chunk)
            if limit is not None and n > limit:
                msg = f"{what}: more than the expected {limit} bytes"
                raise WeightsHashError(msg)
            f.write(chunk)
            h.update(chunk)
    return h.hexdigest()


def _limit(size: int | None) -> int:
    """The fetch cap: the recorded size, or a generous default for a spec without one."""
    return _UNSIZED_CAP if size is None else size


def _request(url: str) -> urllib.request.Request:
    """A GET of `url`; with `HF_TOKEN` set, our Hugging Face repos get it as a bearer token (not on redirects)."""
    req = urllib.request.Request(url)  # noqa: S310 - scheme checked by caller
    token = os.environ.get("HF_TOKEN")
    if token and _OUR_HF.fullmatch(url):
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return req


def _fetch(url: str, out: Path, limit: int | None) -> str:
    """Stream `url` into `out`, at most `limit` bytes; returns the SHA-256 of the bytes."""
    try:
        resp = _OPENER.open(_request(url), timeout=60)
    except urllib.error.HTTPError as e:
        # The error holds its response (also one ending a redirect chain) open; the exception often lives on in a
        # reference cycle (a skipped test, the mirror fallback), and gc may then finalize the socket first:
        # "ResourceWarning: unclosed <ssl.SSLSocket [closed]>". Redirect responses are closed by urllib itself.
        e.close()
        raise
    with resp:
        return _copy(resp, out, limit, url)


def _place(part: Path, digest: str, spec: FileSpec, dest: Path, url: str) -> None:
    """Move the verified download `part` to `dest` and remember the check."""
    if digest != spec.sha256:
        msg = (
            f"{url}{f' ({spec.member})' if spec.member else ''}: SHA-256 {digest} != expected "
            f"{spec.sha256}; the download is corrupt or the upstream file changed. Retry with "
            f"`fastforks download MODEL` (nothing was cached); if it persists, report the model id"
        )
        raise WeightsHashError(msg)
    _mkdir_private(dest.parent)
    part.replace(dest)
    _remember(dest, spec.sha256)


def _extract(src: IO[bytes], declared: int, spec: FileSpec, tmp: Path, url: str) -> tuple[Path, str]:
    """Copy the archive member `src` (declared size `declared`) into `tmp`; returns the copy and its SHA-256."""
    what = f"{url} ({spec.member})"
    if spec.size is not None and declared != spec.size:
        msg = f"{what}: size {declared} != expected {spec.size}"
        raise WeightsHashError(msg)
    part = tmp / f"{spec.sha256}.part"
    return part, _copy(src, part, spec.size, what)


def _search(spec: FileSpec, group: str, env_var: str) -> tuple[Path | None, list[str]]:
    """First copy of `spec` in the search path that verifies, and why the same-named copies before it did not."""
    mismatched: list[str] = []
    for d in search_path(env_var):
        for p in (d / group / spec.name, d / spec.name):  # group first: generic names collide across families
            if p.is_file():
                try:
                    verify(p, spec)
                except WeightsHashError as e:
                    log.debug("skipping %s", e)
                    mismatched.append(str(e))
                else:
                    return p, mismatched
    return None, mismatched


def _uncached_siblings(spec: FileSpec, siblings: Sequence[FileSpec]) -> list[tuple[FileSpec, Path]]:
    """Other members of `spec`'s archive that are not in the cache yet, with their blob paths."""
    if spec.member is None:
        return []
    same_zip = {s.sha256: s for s in siblings if s.url == spec.url and s.member and s.sha256 != spec.sha256}
    return [(s, blob_path(s.sha256)) for s in same_zip.values() if not blob_path(s.sha256).is_file()]


def locate(  # noqa: PLR0913 - the arguments of the fetch choke point
    spec: FileSpec,
    *,
    group: str,
    env_var: str,
    allow_download: bool = True,
    siblings: Sequence[FileSpec] = (),
    hint: str = "fastforks download MODEL",
    burst: int | None = None,
    model: str | None = None,
) -> Path:
    """Return a verified local path for `spec`, downloading it into the cache if needed.

    Args:
        spec: The file.
        group: Model family or kernel library (sub-directory in the search path).
        env_var: Name of the search-path environment variable.
        allow_download: If false, only local copies are considered (always so in offline mode, see `offline`).
        siblings: Other files that may come from the same zip archive (e.g. every variant of a family); those
            not cached yet are extracted from the same fetch.
        hint: The command that fetches `spec`, named in the error of offline mode.
        burst: Downloads with the same id log one line together (kernel files of one call); `None`: one line each.
        model: The requested model id, named in the "not found" error (default: `group`).

    Returns:
        Path to a file whose SHA-256 matches `spec`.

    Raises:
        WeightsDownloadError: The file is not available locally and cannot be downloaded (from `url` or the mirror).
        WeightsHashError: A downloaded copy has the wrong hash, or the only local copies do (and no download is
            possible).
    """
    is_offline = offline()
    allow_download = allow_download and not is_offline
    found, mismatched = _search(spec, group, env_var)
    if found is not None:
        return found
    cached = blob_path(spec.sha256)
    corrupt = _corrupt(cached, spec)
    if cached.is_file() and corrupt is None:
        return cached
    other = _from_other_caches(spec)  # also after a corrupt blob: another tool's verified copy still serves (TM-6)
    if other is not None:
        if corrupt is not None:
            log.warning("not using %s (it does not match its SHA-256); using %s", cached, other)
        return other
    if corrupt is not None:
        if spec.url is None or not allow_download:
            raise corrupt
        log.warning("re-downloading %s: cached copy does not match its SHA-256", cached)
    if spec.url is None or not allow_download:
        raise _not_local(spec, model or group, env_var, mismatched, hint=hint if is_offline else None)
    if mismatched:  # the user put a copy there on purpose: say why it is not used
        log.warning("not using %s; downloading it instead", "; ".join(mismatched))
    return _download_or_mirror(
        spec, spec.url, cached, group=group, env_var=env_var, siblings=siblings, mismatched=mismatched, burst=burst
    )


def _corrupt(cached: Path, spec: FileSpec) -> WeightsHashError | None:
    """Why the cache blob `cached` does not verify; `None` if it does or does not exist."""
    if not cached.is_file():
        return None
    try:
        verify(cached, spec)
    except WeightsHashError as e:
        return e
    return None


def _via_huggingface_hub(spec: FileSpec, group: str, env_var: str, mismatched: Sequence[str]) -> Path | None:
    """A gated file of a third-party Hugging Face repo, fetched by `huggingface_hub` (its login, its cache).

    Called after the plain download was refused. Returns the verified path, or None if `spec` is not such a file,
    `huggingface_hub` is missing, or it failed for a reason other than access (the caller then raises the download
    error). No access (gated, private, not logged in) raises the error that says how to get it.
    """
    gated = _HF_FILE.fullmatch((spec.url or "").split("?", maxsplit=1)[0])
    if gated is None or spec.member is not None or _OUR_HF.fullmatch(spec.url or ""):
        return None
    try:
        import huggingface_hub  # noqa: PLC0415 - optional
    except ImportError:
        return None
    repo, rev, file = gated["repo"], gated["rev"], urllib.parse.unquote(gated["file"])
    try:
        got = huggingface_hub.hf_hub_download(repo_id=repo, filename=file, revision=rev)
    except Exception as e:  # any failure of the hub client: access is reported, the rest is the caller's error
        if _denied(e):
            raise _unavailable(spec, group, env_var, "no access", cause=e, mismatched=mismatched) from e
        log.info("huggingface_hub could not fetch %s/%s (%s: %s)", repo, file, type(e).__name__, e)
        return None
    with contextlib.suppress(WeightsHashError, OSError, TypeError):
        verify(Path(got), spec)
        return Path(got)
    return _from_other_caches(spec)


def _denied(e: BaseException) -> bool:
    """Whether `e` is a refusal for lack of access: HTTP 401/403 of urllib, or the `huggingface_hub` equivalents."""
    if isinstance(e, urllib.error.HTTPError):
        return e.code in (401, 403)
    status = getattr(getattr(e, "response", None), "status_code", None)
    return status in (401, 403) or type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError")


def _not_local(
    spec: FileSpec, who: str, env_var: str, mismatched: Sequence[str], *, hint: str | None
) -> WeightsHashError | WeightsDownloadError:
    """Why `spec` cannot be had without a download (`hint` is set in offline mode: the command that fetches it).

    `who` is the requested model id (or the family / kernel library); the searched `env_var` entries are listed.
    """
    if mismatched:
        return WeightsHashError(f"{spec.name}: no local copy verifies: " + "; ".join(mismatched))
    searched = search_path(env_var)
    listed = f" (searched {env_var}: {os.pathsep.join(map(str, searched))})" if searched else f" ({env_var} is not set)"
    where = f"{spec.name} not found in {env_var} or the cache ({cache_root()}){listed}"
    if hint is not None and spec.url is not None:
        return WeightsDownloadError(
            f"{who}: {where}, and offline mode ({OFFLINE_ENV} / HF_HUB_OFFLINE) forbids downloads. On a machine"
            f" with network access run `{hint}` with the same FASTFORKS_CACHE, then copy that cache here"
        )
    why = "it has no download URL" if spec.url is None else "downloads are disabled"
    return WeightsDownloadError(f"{where}, and {why}")


def _download_or_mirror(  # noqa: PLR0913 - the arguments of `locate`
    spec: FileSpec,
    url: str,
    cached: Path,
    *,
    group: str,
    env_var: str,
    siblings: Sequence[FileSpec],
    mismatched: Sequence[str] = (),
    burst: int | None = None,
) -> Path:
    """Download `spec` to `cached` from its URL, else from its mirror; a failure says what to do about it.

    Returns the verified file: `cached`, or (a gated Hugging Face file) the copy `huggingface_hub` put in its cache.
    """
    try:
        download(url, cached, spec, also=_uncached_siblings(spec, siblings), burst=burst)
    except (WeightsDownloadError, WeightsHashError) as e:
        if spec.mirror is None:
            if isinstance(e, WeightsHashError):
                raise
            if e.__cause__ is not None and _denied(e.__cause__):  # refused: the user's login may have access
                path = _via_huggingface_hub(spec, group, env_var, mismatched)
                if path is not None:
                    return path
            raise _unavailable(spec, group, env_var, str(e), cause=e.__cause__, mismatched=mismatched) from e
        log.warning("%s; trying the mirror", e)
        try:
            download(spec.mirror, cached, FileSpec(spec.name, spec.sha256, spec.size))
        except WeightsDownloadError as e2:  # a hash mismatch of the mirror copy propagates unchanged
            raise _unavailable(spec, group, env_var, f"{e}; mirror: {e2}", mismatched=mismatched) from e2
    return cached


def _human_size(n: int) -> str:
    """`n` bytes as KB below 1 MB (kernel binaries), else MB."""
    return f"{n / 2**10:.0f} KB" if n < 2**20 else f"{n / 2**20:.0f} MB"


def _unavailable(  # noqa: PLR0913 - what the message needs
    spec: FileSpec,
    group: str,
    env_var: str,
    why: str,
    *,
    cause: BaseException | None = None,
    mismatched: Sequence[str] = (),
) -> WeightsDownloadError:
    """The download failure of `spec` and what to do about it (gated Hugging Face repo: the manual route).

    `mismatched` (why same-named local copies did not verify) is appended to either message.
    """
    size = f", {_human_size(spec.size)}" if spec.size else ""
    not_verified = f". Local copies that do not verify: {'; '.join(mismatched)}" if mismatched else ""
    gated = _HF_FILE.fullmatch(spec.url or "")
    if gated and not _OUR_HF.fullmatch(spec.url or "") and cause is not None and _denied(cause):
        repo, rev, file = gated["repo"], gated["rev"], gated["file"]
        try:
            import huggingface_hub  # noqa: F401, PLC0415

            install = ""
        except ImportError:
            install = "`pip install huggingface_hub` first, then "
        msg = (
            f"{group}: {spec.name}{size} is in the gated or private Hugging Face repo {repo} ({why}). Accept the"
            f" licence at https://huggingface.co/{repo}, {install}log in with `hf auth login` (or"
            f" `huggingface_hub.login()` in Python), then load again. Or fetch it by hand:"
            f" `hf download {repo} {file} --revision {rev}` and put it in a directory listed in {env_var} as"
            f" <dir>/{group}/{spec.name} (SHA-256 {spec.sha256}, checked), or pass `weights=PATH`"
        )
        return WeightsDownloadError(msg + not_verified)
    by_hand = f"fetch it by hand from {spec.url} and put it in a directory listed in {env_var}"
    by_hand += f" (as <dir>/{group}/{spec.name}; its SHA-256 is checked)"
    code = cause.code if isinstance(cause, urllib.error.HTTPError) else None
    if code in (401, 403):  # retrying cannot help: the host refuses us
        advice = f"the host refuses access (HTTP {code}: private repository or missing permission); {by_hand}"
    elif code == http.HTTPStatus.NOT_FOUND:
        advice = f"the host has no such file (HTTP 404: missing build or moved); {by_hand}"
    else:
        advice = f"Retry later, or {by_hand}"
    msg = f"{group}: cannot download {spec.name}{size} ({why}). {advice}"
    return WeightsDownloadError(msg + not_verified)
