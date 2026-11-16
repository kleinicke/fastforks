"""Every card's weights resolve offline from the default locations (fastforks blob cache, Hugging Face hub cache,
torch hub cache); families that share a file pin identical bytes."""

from __future__ import annotations

import collections
import urllib.parse

from fastforks._core import files, registry


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


def test_every_spec_resolves_from_each_default_location(tmp_path, monkeypatch):
    hub, torch_home, cache = tmp_path / "hub", tmp_path / "torch", tmp_path / "cache"
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
            with p.open("wb") as f:
                f.truncate(s.size)  # sparse: only the size is checked by `present`
            if not files.present(s, family, "FASTFORKS_WEIGHTS_PATH"):
                wrong.append((family, s.name, str(p)))
            p.unlink()
    assert not wrong
