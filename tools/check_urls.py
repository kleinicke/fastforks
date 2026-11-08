"""Check that every weight and kernel URL still serves a file of the recorded size (nightly CI).

    uv run python tools/check_urls.py
    uv run python tools/check_urls.py --kernels   # only our kernel binaries: the first step of the PyPI publish

`--kernels` stops a release whose manifests name a kernel build not yet uploaded to Hugging Face (users would fall
back to portable, and `mode="kernels"` would raise). It leaves out the weights: the gated ones (DINOv3, VGGT-Omega)
answer an anonymous request with 401, and a third-party link is not ours to fix before a release.

Fetches one byte per file (HTTP range request) instead of the whole file; for an archive member, the size of the
pinned archive (`archive` in the card) is checked. Weight mirrors are checked too. URLs in our own Hugging Face repos
go through `FASTFORKS_HF_ENDPOINT` like real downloads, so a local stand-in can be checked before the upload. The
SHA-256 itself is checked on every real download. A server that reports no size, or answers with an HTML page
(expired share links do), counts as broken. Exits non-zero listing the broken URLs.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import url2pathname

from fastforks._core import registry
from fastforks._core.files import FileSpec, hub_url
from fastforks._core.kernels import read_manifest

if TYPE_CHECKING:
    from collections.abc import Iterator

KERNELS = Path(__file__).resolve().parents[1] / "src" / "fastforks" / "kernels"


def remote_size(url: str) -> int | None:
    """Total size reported by the server, or `None` if it does not say or serves an HTML page."""
    if url.startswith("file://"):
        return Path(url2pathname(urlparse(url).path)).stat().st_size
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "fastforks-url-check"})  # noqa: S310 - card URLs are https
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - card URLs are https
        if resp.headers.get_content_type() == "text/html":
            return None
        total = resp.headers.get("Content-Range", "").rpartition("/")[2]
        if total.isdigit():
            return int(total)
        length = resp.headers.get("Content-Length")
        return int(length) if resp.status == 200 and length else None  # noqa: PLR2004 - HTTP OK


def files(*, kernels_only: bool = False) -> Iterator[tuple[str, FileSpec]]:
    """(owner, file) of every downloadable weight and kernel binary (`kernels_only`: the kernel binaries)."""
    for task in () if kernels_only else registry.TASKS:
        for _, variant in registry.cards(task).values():
            yield from ((variant.ref, spec) for spec in variant.weights if spec.url is not None)
            yield from (
                (f"{variant.ref} (mirror)", FileSpec(spec.name, spec.sha256, spec.size, spec.mirror))
                for spec in variant.weights
                if spec.mirror is not None
            )
    for path in sorted(KERNELS.glob("*/manifest.toml")):
        main = read_manifest(path)
        for m in filter(None, (main, main.compat)):
            yield from ((f"{m.tag}:{target}", spec) for target, spec in m.files.items() if spec.url is not None)


def main() -> None:
    broken: list[str] = []
    for owner, spec in files(kernels_only="--kernels" in sys.argv[1:]):
        assert spec.url is not None  # noqa: S101 - filtered in files()
        try:
            size = remote_size(hub_url(spec.url))
        except OSError as e:
            broken.append(f"{owner} {spec.url}: {e}")
            continue
        want = spec.archive_size if spec.member else spec.size
        ok = size is not None and (want is None or size == want)
        print(f"{'ok ' if ok else 'BAD'} {owner} {spec.name} {size}")
        if not ok:
            broken.append(f"{owner} {spec.url}: size {size} (want {want})")
    if broken:
        sys.exit("broken URLs:\n" + "\n".join(broken))


if __name__ == "__main__":
    main()
