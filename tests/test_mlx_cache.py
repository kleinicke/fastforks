"""The MLX conversion cache (`fastforks._core.mlx.convert_cached`, `fastforks._core.mlx_cache`): no weights, no GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from fastforks._core import cli, mlx_cache

SHA = "ab" * 32


def _ctx(user_weights: object = None) -> types.SimpleNamespace:
    spec = types.SimpleNamespace(name="toy.pth", sha256=SHA)
    return types.SimpleNamespace(
        variant=types.SimpleNamespace(id="toy-a", ref="toy-a@1", revision=1, weights=[spec]),
        card=types.SimpleNamespace(task="classification", family="toy"),
        user_weights=user_weights,
    )


def _models(revision: int = 1, sha: str = SHA) -> dict:
    card = types.SimpleNamespace(task="classification", family="toy")
    variant = types.SimpleNamespace(revision=revision, weights=[types.SimpleNamespace(sha256=sha)])
    return {"toy-a": (card, variant)}


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FASTFORKS_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache" / "mlx"


def _build(mx: types.ModuleType, calls: list[int]):
    def build() -> tuple[dict, dict]:
        calls.append(1)
        return {"w": mx.array(np.arange(6, dtype=np.float32).reshape(2, 3)), "rope": mx.ones((4,))}, {"depth": 2}

    return build


def test_file_name_and_metadata(cache: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    from fastforks._core.mlx import convert_cached  # noqa: PLC0415 - needs MLX

    calls: list[int] = []
    got = convert_cached(_ctx(), "toy-v1", _build(mx, calls), dtype="float16", fp32=("rope",))
    assert not got.hit
    assert got.arrays["w"].dtype == mx.float16
    assert got.arrays["rope"].dtype == mx.float32  # the island
    [path] = list(cache.rglob("*.safetensors"))
    assert path.relative_to(cache).as_posix() == f"classification/toy/toy-a-r1-{SHA[:12]}-toy-v1-f16.safetensors"
    meta = mlx_cache.read_header(path)
    assert meta is not None
    assert set(mlx_cache.META_KEYS) <= set(meta)
    assert meta["format"] == mlx_cache.FORMAT
    assert (meta["variant"], meta["revision"], meta["weights_sha256"], meta["dtype"]) == ("toy-a", "1", SHA, "float16")
    assert json.loads(meta["fp32"]) == ["rope"]
    assert json.loads(meta["config"]) == {"depth": 2}
    again = convert_cached(_ctx(), "toy-v1", _build(mx, calls), dtype="float16", fp32=("rope",))
    assert again.hit
    assert len(calls) == 1
    assert again.config == {"depth": 2}
    assert np.array_equal(np.array(again.arrays["w"]), np.array(got.arrays["w"]))
    # another dtype is another file; a source file joins the tag by hash
    convert_cached(_ctx(), "toy-v1", _build(mx, calls), dtype="float32")
    src = cache.parent / "src.py"
    src.write_text("x = 1\n")
    convert_cached(_ctx(), "toy-v1", _build(mx, calls), sources=(src,))
    names = sorted(p.name for p in cache.rglob("*.safetensors"))
    assert len(names) == 3
    assert any(n.endswith("-f32.safetensors") and "-toy-v1-" in n and "toy-v1-f32" not in n for n in names)


def test_user_weights_are_never_cached(cache: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    from fastforks._core.mlx import convert_cached  # noqa: PLC0415 - needs MLX

    calls: list[int] = []
    for _ in range(2):
        got = convert_cached(_ctx(user_weights="mine.pth"), "toy-v1", _build(mx, calls), dtype="float16")
        assert not got.hit
    assert len(calls) == 2
    assert not cache.exists() or not list(cache.rglob("*"))


def test_float16_overflow_is_an_error(cache: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    from fastforks._core.mlx import convert_cached  # noqa: PLC0415 - needs MLX

    def build() -> tuple[dict, dict]:
        return {"w": mx.array([1e6], dtype=mx.float32)}, {}

    with pytest.raises(ValueError, match="beyond float16"):
        convert_cached(_ctx(), "toy-v1", build, dtype="float16")
    assert not list(cache.rglob("*.safetensors")) if cache.exists() else True


def test_hit_needs_neither_torch_nor_the_checkpoint(cache: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    from fastforks._core.mlx import convert_cached  # noqa: PLC0415 - needs MLX

    convert_cached(_ctx(), "toy-v1", _build(mx, []), dtype="float16")
    code = textwrap.dedent(
        f"""
        import sys, types
        from fastforks._core.mlx import convert_cached
        spec = types.SimpleNamespace(name="toy.pth", sha256={SHA!r})
        ctx = types.SimpleNamespace(
            variant=types.SimpleNamespace(id="toy-a", revision=1, weights=[spec]),
            card=types.SimpleNamespace(task="classification", family="toy"),
            user_weights=None,
        )
        def build():
            raise AssertionError("a hit must not convert")
        got = convert_cached(ctx, "toy-v1", build, dtype="float16")
        assert got.hit and got.config == {{"depth": 2}}, got
        assert "torch" not in sys.modules, "a hit imported torch"
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_write_is_atomic(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mx = pytest.importorskip("mlx.core")
    from fastforks._core import mlx as core_mlx  # noqa: PLC0415 - needs MLX

    real = core_mlx.mx.save_safetensors

    def broken(path: str, arrays: dict, metadata: dict) -> None:
        Path(path).write_bytes(b"partial")  # a write that dies half-way
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(core_mlx.mx, "save_safetensors", broken)
    with pytest.raises(OSError, match="disk full"):
        core_mlx.convert_cached(_ctx(), "toy-v1", _build(mx, []), dtype="float16")
    assert not [p for p in cache.rglob("*") if p.is_file()]  # neither the file nor the temporary one
    monkeypatch.setattr(core_mlx.mx, "save_safetensors", real)
    core_mlx.convert_cached(_ctx(), "toy-v1", _build(mx, []), dtype="float16")
    assert [p.name.startswith(".") for p in cache.rglob("*") if p.is_file()] == [False]


def _write(path: Path, meta: dict[str, str] | None) -> Path:
    """A minimal safetensors file (one float32 scalar) with `meta` as its header metadata."""
    header: dict = {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    if meta is not None:
        header["__metadata__"] = meta
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0\0\0\0")
    return path


def _meta(**over: str) -> dict[str, str]:
    meta = {
        "format": mlx_cache.FORMAT,
        "variant": "toy-a",
        "task": "classification",
        "family": "toy",
        "revision": "1",
        "weights_sha256": SHA,
        "dtype": "float16",
        "layout": "toy-v1",
        "layout_name": "toy-v1",
        "created": "2026-01-01T00:00:00+00:00",
    }
    return meta | over


def test_prune_removes_only_stale_files_under_the_mlx_dir(cache: Path, tmp_path: Path) -> None:
    d = cache / "classification" / "toy"
    current = _write(d / "current.safetensors", _meta())
    legacy = _write(d / "legacy.safetensors", None)
    other_rev = _write(d / "rev0.safetensors", _meta(revision="0"))
    gone = _write(d / "gone.safetensors", _meta(variant="toy-gone"))
    old = _write(d / "old.safetensors", _meta(layout="toy-v2-00000000", layout_name="toy-v2"))
    new = _write(d / "new.safetensors", _meta(layout="toy-v2-11111111", layout_name="toy-v2", created="2026-02-01"))
    in_flight = _write(d / ".123.456.x.safetensors", None)
    outside = _write(tmp_path / "cache" / "blobs" / "keep.safetensors", None)  # cache, but not its mlx dir

    dry = mlx_cache.prune(_models(), dry_run=True)
    assert {e.path for e, _ in dry} == {legacy, other_rev, gone, old}
    assert all(p.exists() for p in (legacy, other_rev, gone, old))
    removed = mlx_cache.prune(_models())
    assert {e.path for e, _ in removed} == {legacy, other_rev, gone, old}
    assert current.exists()
    assert new.exists()
    assert in_flight.exists()
    assert outside.exists()
    assert not any(p.exists() for p in (legacy, other_rev, gone, old))


def test_prune_refuses_a_symlink(cache: Path, tmp_path: Path) -> None:
    victim = _write(tmp_path / "elsewhere" / "victim.safetensors", None)
    link = cache / "classification" / "toy" / "link.safetensors"
    link.parent.mkdir(parents=True)
    link.symlink_to(victim)
    with pytest.raises(ValueError, match="refusing"):
        mlx_cache.prune(_models())
    assert victim.exists()


def test_cli_lists_and_prunes(cache: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    d = cache / "classification" / "toy"
    _write(d / "current.safetensors", _meta())
    legacy = _write(d / "legacy.safetensors", None)
    monkeypatch.setattr(cli, "_all_cards", _models)
    assert cli.main(["cache", "mlx"]) == 0
    out = capsys.readouterr().out
    assert "legacy.safetensors" in out
    assert "stale: no provenance" in out
    assert "2 files" in out
    assert cli.main(["cache", "mlx", "--prune", "--dry-run"]) == 0
    assert legacy.exists()
    assert "would remove" in capsys.readouterr().out
    assert cli.main(["cache", "mlx", "--dry-run"]) == 2
    assert cli.main(["cache", "mlx", "--prune"]) == 0
    assert not legacy.exists()
    assert (d / "current.safetensors").exists()
