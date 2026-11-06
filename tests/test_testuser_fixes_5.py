"""Test-user fixes 5 (reproducible deployment): env flags, paths, messages, `info`, `env`, `doctor`."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks._core import cli, files, registry
from fastforks._core.errors import FastforksError, InputError, OptionsError
from fastforks.depth import DepthResult
from fastforks.stereo.pipeline import _files, check_output_dir, run_many


@pytest.mark.parametrize("value", ["1", "TRUE", " yes ", "On"])
def test_env_flag_true(monkeypatch, value):
    monkeypatch.setenv("FASTFORKS_OFFLINE", value)
    assert files.offline()


@pytest.mark.parametrize("value", ["0", "false", "No", "off", ""])
def test_env_flag_false(monkeypatch, value):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("FASTFORKS_OFFLINE", value)
    assert not files.offline()


@pytest.mark.parametrize("name", ["FASTFORKS_OFFLINE", "FASTFORKS_NONCOMMERCIAL", "FASTFORKS_COMMERCIAL_ONLY"])
def test_env_flag_typo_fails_closed(monkeypatch, name):
    monkeypatch.setenv(name, "enable")
    with pytest.raises(OptionsError, match=name):
        files.env_flag(name)


def test_cache_and_search_path_expand_and_absolutize(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FASTFORKS_CACHE", "~/c")
    assert files.cache_root() == tmp_path / "c"
    monkeypatch.setenv("FASTFORKS_CACHE", "rel")
    assert files.cache_root() == tmp_path / "rel"
    monkeypatch.setenv("FASTFORKS_WEIGHTS_PATH", "~/w")
    assert files.search_path("FASTFORKS_WEIGHTS_PATH") == [tmp_path / "w"]


def test_search_path_warns_once_per_bad_entry(monkeypatch, tmp_path, caplog):
    bad = tmp_path / "a,b"
    monkeypatch.setenv("FASTFORKS_WEIGHTS_PATH", f"{bad}:{tmp_path}")
    with caplog.at_level(logging.WARNING, logger=files.log.name):
        files.search_path("FASTFORKS_WEIGHTS_PATH")
        files.search_path("FASTFORKS_WEIGHTS_PATH")
    msgs = [r.getMessage() for r in caplog.records if str(bad) in r.getMessage()]
    assert len(msgs) == 1
    assert "':'" in msgs[0]
    assert not [r for r in caplog.records if str(tmp_path) + " is not" in r.getMessage()]


def test_not_found_names_model_and_searched_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("FASTFORKS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FASTFORKS_WEIGHTS_PATH", str(tmp_path))
    monkeypatch.setenv("FASTFORKS_OFFLINE", "1")
    spec = files.FileSpec("w.pth", "0" * 64, 1, "https://example.org/w.pth")
    with pytest.raises(FastforksError) as e:
        files.locate(spec, group="fam", env_var="FASTFORKS_WEIGHTS_PATH", model="fam-m@1")
    text = str(e.value)
    assert text.startswith("fam-m@1:")
    assert f"searched FASTFORKS_WEIGHTS_PATH: {tmp_path}" in text


def test_commercial_only_list_hides_noncommercial(monkeypatch):
    monkeypatch.setenv("FASTFORKS_COMMERCIAL_ONLY", "1")
    assert registry.list_models(registry.TASKS, commercial="no") == []
    assert registry.list_models(registry.TASKS, commercial="yes")


def test_download_dry_run_commercial_only_env(monkeypatch, capsys):
    monkeypatch.setenv("FASTFORKS_COMMERCIAL_ONLY", "1")
    assert cli.main(["download", "s2m2-s", "--dry-run"]) == 1
    assert "FASTFORKS_COMMERCIAL_ONLY=1: not cleared for commercial use, so excluded" in capsys.readouterr().err


def test_download_named_nc_model_with_flag_says_excluded(monkeypatch, capsys):
    monkeypatch.delenv("FASTFORKS_NONCOMMERCIAL", raising=False)
    assert cli.main(["download", "s2m2-s", "--commercial-only"]) == 1
    err = capsys.readouterr().err
    assert "--commercial-only: not cleared for commercial use, so excluded: s2m2-s" in err
    assert "FASTFORKS_NONCOMMERCIAL" not in err


def test_info_shows_revision_urls_and_sha(capsys):
    assert cli.main(["info", "s2m2-s"]) == 0
    out = capsys.readouterr().out
    _, v = registry.resolve_any("s2m2-s")
    assert f"revision       {v.revision} (pin with `s2m2-s@{v.revision}`)" in out
    assert v.weights[0].sha256 in out
    assert "weights url" in out


def test_env_shows_cache_switches_and_one_kernel_line(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("FASTFORKS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FASTFORKS_OFFLINE", "1")
    assert cli.main(["env"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(x.startswith("cache") and str(tmp_path / "cache") in x for x in lines)
    assert any(x.startswith("switches") and "offline on" in x for x in lines)
    assert sum(x.startswith("kernels") for x in lines) <= 3  # was one line per library


def test_doctor_offline_does_not_count_missing_weights(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("FASTFORKS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FASTFORKS_OFFLINE", "1")
    monkeypatch.delenv("FASTFORKS_WEIGHTS_PATH", raising=False)
    monkeypatch.setattr(files, "_other_caches", lambda _spec: iter(()))
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert " 0 of " in next(x for x in out.splitlines() if x.startswith("models"))
    assert out.splitlines()[-1].startswith("next step")
    assert isinstance(tmp_path, Path)


def _images(d: Path, *names: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(d / n)


def test_broken_symlink_in_directory_and_glob_is_listed(tmp_path):
    _images(tmp_path / "L", "a.png")
    (tmp_path / "L" / "b.png").symlink_to(tmp_path / "missing.png")
    assert [n for _, n in _files(tmp_path / "L", "left")] == ["a", "b"]
    assert [n for _, n in _files(str(tmp_path / "L" / "*.png"), "left")] == ["a", "b"]


def test_output_dir_checks(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(InputError, match="output directory"):
        check_output_dir(f)
    with pytest.raises(InputError, match="output directory"):
        run_many(None, [f], [f], save_to=f)  # type: ignore[arg-type]
    assert check_output_dir(tmp_path / "new" / "sub").is_dir()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_output_dir_not_writable(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        with pytest.raises(InputError, match="not writable"):
            check_output_dir(ro)
    finally:
        ro.chmod(0o700)


def test_stereo_run_out_is_a_file_fails_before_model_load(capsys, tmp_path):
    _images(tmp_path / "L", "a.png")
    _images(tmp_path / "R", "a.png")
    out = tmp_path / "out"
    out.write_text("x")
    argv = ["stereo", "run", str(tmp_path / "L"), str(tmp_path / "R"), "--model", "hitnet-eth3d", "--out", str(out)]
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "fastforks: error:" in err
    assert "output directory" in err
    assert "Traceback" not in err


def test_directory_name_collisions_are_checked_before_any_write(tmp_path):
    for side in ("L", "R"):
        _images(tmp_path / side, "a.png", "a.jpg")  # same stem (not a.png + A.png: one file on macOS)
    out = tmp_path / "out"
    with pytest.raises(InputError, match="both be saved as"):
        run_many(None, str(tmp_path / "L"), str(tmp_path / "R"), save_to=out)  # type: ignore[arg-type]
    assert not out.exists() or not any(out.iterdir())


def _depth(depth) -> DepthResult:
    fields = dict.fromkeys(("inverse_depth", "points", "intrinsics", "confidence", "mask", "normals"), None)
    return DepthResult(
        **fields | {"depth": depth, "extras": {}, "kind": "metric", "model": "m", "backend": "portable"}  # type: ignore[arg-type]
    )


def test_depth_save_png_is_x256_uint16(tmp_path):
    d = torch.tensor([[1.0, 2.5], [float("nan"), 1000.0]])
    _depth(d).save(tmp_path / "d.png")
    a = np.array(Image.open(tmp_path / "d.png"))
    assert a.dtype == np.uint16
    assert a.tolist() == [[256, 640], [0, 65535]]


def test_stereo_and_depth_png_share_the_writer(tmp_path):
    from fastforks.stereo.io import write_disparity  # noqa: PLC0415

    d = np.array([[1.0, 2.5], [-3.0, 300.0]], np.float32)
    write_disparity(tmp_path / "s.png", d)
    _depth(torch.from_numpy(d)).save(tmp_path / "d.png")
    assert np.array_equal(np.array(Image.open(tmp_path / "s.png")), np.array(Image.open(tmp_path / "d.png")))
