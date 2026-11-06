"""Offline mode (`FASTFORKS_OFFLINE`), `fastforks download --arch / --check / --dry-run / --commercial-only`."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from fastforks import WeightsDownloadError
from fastforks._core import cli, files, registry
from fastforks._core.files import FileSpec, locate

ENV = "FF_TEST_PATH"

# Runs first in every guarded subprocess: any socket use or DNS lookup fails the test.
GUARD = textwrap.dedent(
    """
    import socket
    def _no(*a, **k):
        raise AssertionError("network access in offline mode")
    socket.socket.connect = _no
    socket.getaddrinfo = _no
    """
)


def _run_guarded(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", GUARD + textwrap.dedent(code)], env=env, capture_output=True, text=True, check=False
    )


@pytest.fixture
def prefilled(tmp_path, monkeypatch):
    """A cache filled through the real download path (file:// stand-in of our mirror repo); returns (spec, env)."""
    data = b"offline weights"
    spec = FileSpec(
        "w.bin", hashlib.sha256(data).hexdigest(), len(data), files.MIRROR_URL.format(group="fam", name="w.bin")
    )
    blob = tmp_path / "hf" / "fastforks" / "mirrors" / "resolve" / "main" / "fam" / "w.bin"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(data)
    monkeypatch.setenv(files.HF_ENDPOINT_ENV, (tmp_path / "hf").as_uri())
    monkeypatch.setenv("FASTFORKS_CACHE", str(tmp_path / "cache"))
    locate(spec, group="fam", env_var=ENV)
    for v in (files.OFFLINE_ENV, "HF_HUB_OFFLINE"):
        monkeypatch.delenv(v, raising=False)
    return spec, {"FASTFORKS_CACHE": str(tmp_path / "cache"), "PATH": "/usr/bin:/bin"}


@pytest.mark.parametrize("var", [files.OFFLINE_ENV, "HF_HUB_OFFLINE"])
def test_offline_missing_file_names_file_and_command(tmp_path, monkeypatch, var):
    monkeypatch.setenv("FASTFORKS_CACHE", str(tmp_path))
    monkeypatch.setenv(var, "1")
    spec = FileSpec("w.bin", hashlib.sha256(b"x").hexdigest(), 1, "https://example.invalid/w.bin")

    def no_fetch(*_a, **_k):
        msg = "fetch attempted"
        raise AssertionError(msg)

    monkeypatch.setattr(files, "_fetch", no_fetch)
    with pytest.raises(WeightsDownloadError, match=r"w\.bin.*offline mode.*`fastforks download X@1`"):
        locate(spec, group="fam", env_var=ENV, hint="fastforks download X@1")


def test_offline_flag_values(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    for value, want in [
        ("", False),
        ("0", False),
        ("off", False),
        ("OFF", False),
        ("1", True),
        ("true", True),
        ("ON", True),
    ]:
        monkeypatch.setenv(files.OFFLINE_ENV, value)
        assert files.offline() is want


def test_offline_uses_a_prefilled_cache_without_network(prefilled):
    spec, env = prefilled
    code = f"""
    from fastforks._core.files import FileSpec, locate
    s = FileSpec({spec.name!r}, {spec.sha256!r}, {spec.size!r}, "https://example.invalid/w.bin")
    assert locate(s, group="fam", env_var="X").read_bytes() == b"offline weights"
    bad = FileSpec("other.bin", "0" * 64, 1, "https://example.invalid/o.bin")
    try:
        locate(bad, group="fam", env_var="X")
    except Exception as e:
        assert "offline mode" in str(e), e
    else:
        raise SystemExit("no error")
    """
    r = _run_guarded(code, env | {files.OFFLINE_ENV: "1"})
    assert r.returncode == 0, r.stderr


def test_guard_catches_network(prefilled):
    """The guard works: without offline mode the missing file is fetched and the guard fires."""
    _, env = prefilled
    code = """
    from fastforks._core.files import FileSpec, locate
    locate(FileSpec("o.bin", "0" * 64, 1, "https://example.invalid/o.bin"), group="fam", env_var="X")
    """
    r = _run_guarded(code, env)
    assert r.returncode != 0
    assert "network access in offline mode" in r.stderr


def test_offline_read_only_cache(prefilled):
    if os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    spec, env = prefilled
    root = files.cache_root()
    modes = [(p, p.stat().st_mode) for p in [root, *root.rglob("*")]]
    for p, m in modes:
        p.chmod(m & ~0o222)
    try:
        code = f"""
        from fastforks._core.files import FileSpec, locate
        s = FileSpec({spec.name!r}, {spec.sha256!r}, {spec.size!r}, "https://example.invalid/w.bin")
        assert locate(s, group="fam", env_var="X").is_file()
        """
        r = _run_guarded(code, env | {files.OFFLINE_ENV: "1"})
        assert r.returncode == 0, r.stderr
    finally:
        for p, m in reversed(modes):
            p.chmod(m)


def _local_weights(ref):
    card, variant = registry.resolve_any(ref)
    try:
        return registry.weights(card, variant, allow_download=False)
    except Exception as e:  # noqa: BLE001 - any failure means: not available here
        pytest.skip(f"weights of {ref} not available locally: {e}")


@pytest.mark.slow
@pytest.mark.model
def test_offline_model_runs_from_a_copied_cache(tmp_path):
    """Copy a small model's weights into a fresh cache, then load and run it with the network guarded."""
    (path,) = _local_weights("hitnet-eth3d")
    _card, variant = registry.resolve_any("hitnet-eth3d")
    cache = tmp_path / "cache"
    for spec in variant.weights:
        dest = cache / "blobs" / "sha256" / spec.sha256[:2] / spec.sha256
        dest.parent.mkdir(parents=True)
        shutil.copy(path, dest)
    code = """
    import torch
    from fastforks import stereo
    m = stereo.load("hitnet-eth3d", mode="portable", device="cpu")
    r = m(torch.rand(3, 64, 96), torch.rand(3, 64, 96))
    assert r.disparity.shape == (64, 96)
    """
    env = {"FASTFORKS_CACHE": str(cache), files.OFFLINE_ENV: "1", "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"}
    r = _run_guarded(code, env)
    assert r.returncode == 0, r.stderr[-2000:]


def test_download_arch_and_all_archs_are_exclusive():
    with pytest.raises(SystemExit):
        cli.main(["download", "hitnet-eth3d", "--kernels", "--arch", "sm_89", "--all-archs"])


def test_download_arch_without_gpu_needs_no_gpu(capsys):
    assert cli.main(["download", "hitnet-eth3d", "--dry-run", "--kernels", "--arch", "sm_86,8.9"]) == 0
    out = capsys.readouterr().out
    assert "hitnet-eth3d@1: licence Apache-2.0" in out
    assert ":sm_89: " in out
    assert cli.main(["download", "hitnet-eth3d", "--dry-run", "--kernels", "--arch", "sm_1"]) == 1


def test_download_arch_rejects_garbage(capsys):
    assert cli.main(["download", "hitnet-eth3d", "--dry-run", "--kernels", "--arch", "fast"]) == 1
    assert "cannot read 'fast'" in capsys.readouterr().err


def test_download_kernels_without_gpu_or_arch_names_both_options(capsys, monkeypatch):
    import torch  # noqa: PLC0415

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cli.main(["download", "hitnet-eth3d", "--dry-run", "--kernels"]) == 1
    assert "--arch sm_XX or use --all-archs" in capsys.readouterr().err


def test_download_check_fails_on_an_empty_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FASTFORKS_CACHE", str(tmp_path))
    monkeypatch.delenv("FASTFORKS_WEIGHTS_PATH", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    assert cli.main(["download", "hitnet-eth3d", "--check"]) == 1
    assert "MISSING hitnet-eth3d@1" in capsys.readouterr().err


@pytest.mark.model
def test_download_check_passes_when_complete(capsys):
    _local_weights("hitnet-eth3d")
    assert cli.main(["download", "hitnet-eth3d", "--check"]) == 0
    assert "hitnet-eth3d@1: ok" in capsys.readouterr().out


def test_download_commercial_only_and_licence_class(capsys):
    assert cli.main(["download", "hitnet-eth3d", "--dry-run", "--commercial-only"]) == 0
    assert "hitnet-eth3d@1: licence Apache-2.0" in capsys.readouterr().out
    assert cli.main(["download", "--dry-run", "--commercial-only"]) == 0
    cap = capsys.readouterr()
    assert "skipping s2m2-s@1" in cap.err
    assert "s2m2-s@1:" not in cap.out
    assert cli.main(["download", "s2m2-s", "--dry-run"]) == 0
    assert "commercial no, " in capsys.readouterr().out


def test_download_without_models_warns_about_licences(capsys):
    assert cli.main(["download", "--dry-run"]) == 0
    assert "not cleared for commercial use" in capsys.readouterr().err


def test_download_arch_needs_kernels_and_a_list(capsys):
    for argv in (["--arch", "sm_89"], ["--kernels", "--arch", ","], ["--kernels", "--arch", ""]):
        assert cli.main(["download", "hitnet-eth3d", "--dry-run", *argv]) == 1
        assert "--arch needs --kernels" in capsys.readouterr().err


def test_download_bad_arch_fails_before_any_fetch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FASTFORKS_CACHE", str(tmp_path))
    assert cli.main(["download", "hitnet-eth3d", "--kernels", "--arch", "sm_89,sm_90a"]) == 1
    err = capsys.readouterr().err
    assert "sm_90a" in err
    assert not list(tmp_path.rglob("*"))


def test_download_itself_refuses_offline(monkeypatch, capsys):
    monkeypatch.setenv(files.OFFLINE_ENV, "1")
    assert cli.main(["download", "hitnet-eth3d"]) == 1
    assert "unset it to download" in capsys.readouterr().err
    assert cli.main(["download", "hitnet-eth3d", "--dry-run"]) == 0


def test_commercial_only_with_a_named_nc_model_fails(capsys):
    assert cli.main(["download", "s2m2-s", "--dry-run", "--commercial-only"]) == 1
    assert "not cleared for commercial use, so excluded: s2m2-s@1" in capsys.readouterr().err
