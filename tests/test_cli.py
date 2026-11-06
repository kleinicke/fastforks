import argparse
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from fastforks import BackendUnavailableError
from fastforks._core import cli, registry
from fastforks._core.errors import FastforksError


def test_list_filters(capsys):
    assert cli.main(["list"]) == 0
    assert "s2m2-s@1" in capsys.readouterr().out
    assert cli.main(["list", "--commercial", "yes"]) == 0
    assert "s2m2" not in capsys.readouterr().out


def test_list_task_and_exclude(capsys):
    assert cli.main(["list", "--task", "depth", "--exclude", "*-l", "--exclude", "*-b", "DEPTH-anything-v2-*"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[:2] for line in lines] == [["depth-anything-v2-s@1", "depth"]]
    assert cli.main(["list", "s2m2-s"]) == 0
    assert capsys.readouterr().out.split()[:2] == ["s2m2-s@1", "stereo"]
    with pytest.raises(SystemExit):
        cli.main(["list", "--task", "nope"])


def test_info(capsys):
    assert cli.main(["info", "s2m2-m@1"]) == 0
    out = capsys.readouterr().out
    assert "CH192NTR2.pth" in out
    assert "commercial no" in out


def test_errors_exit_1(capsys):
    assert cli.main(["info", "nope"]) == 1
    assert "unknown model 'nope' (no task has it; tasks: stereo, depth" in capsys.readouterr().err


@pytest.mark.slow
def test_closed_stdout_exits_1_quietly():
    """`fastforks list | head`: a reader that went away is exit 1 without a traceback, not exit 120."""
    r, w = os.pipe()
    os.close(r)
    try:
        p = subprocess.run(
            [sys.executable, "-m", "fastforks._core.cli", "list"], check=False, stdout=w, stderr=subprocess.PIPE
        )
    finally:
        os.close(w)
    assert (p.returncode, p.stderr) == (1, b"")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_error_after_cuda_start_exits_1(tmp_path):
    """A load failing while the CUDA warm-up thread runs exits 1, not 134 (`std::terminate` at interpreter exit)."""
    env = {k: v for k, v in os.environ.items() if k != "FASTFORKS_KERNEL_PATH"}
    env |= {"FASTFORKS_OFFLINE": "1", "FASTFORKS_CACHE": str(tmp_path)}
    cmd = [sys.executable, "-m", "fastforks._core.cli", "bench", "las2-s", "--mode", "kernels", "--size", "384x640"]
    p = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 1, p.stderr
    assert "kernel library 'igev' unavailable" in p.stderr


def test_info_finds_every_task(capsys):
    assert cli.main(["info", "depth-anything-v2-s"]) == 0
    assert "depth" in capsys.readouterr().out


def _local(ref):
    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.errors import WeightsError  # noqa: PLC0415

    card, variant = registry.resolve_any(ref)
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"weights of {ref} not available locally: {e}")


def test_size_parser():
    assert cli._size("384X1248") == (384, 1248)
    with pytest.raises(argparse.ArgumentTypeError, match="HxW"):
        cli._size("384")


@pytest.mark.model
def test_bench_cpu(capsys):
    _local("hitnet-eth3d")
    assert cli.main(["bench", "hitnet-eth3d", "--size", "64x96", "--runs", "1", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "hitnet-eth3d@1 64x96 portable float32:" in out
    assert " ms (median of 1)" in out


@pytest.mark.slow
@pytest.mark.model
def test_bench_depth_cpu(capsys):
    _local("depth-anything-v2-s")
    assert cli.main(["bench", "depth-anything-v2-s", "--size", "70x84", "--runs", "1", "--device", "cpu"]) == 0
    assert "depth-anything-v2-s@1 70x84 portable float32:" in capsys.readouterr().out


def test_bench_rejects_unusable_device(capsys):
    assert cli.main(["bench", "hitnet-eth3d", "--device", "gpu"]) == 1
    assert "invalid device 'gpu'" in capsys.readouterr().err


@pytest.mark.model
@pytest.mark.skipif(not importlib.util.find_spec("onnxscript"), reason="needs onnxscript (dependency group export)")
def test_export_onnx_cpu(capsys, tmp_path):
    _local("hitnet-eth3d")
    out = tmp_path / "h.onnx"
    assert cli.main(["export", "hitnet-eth3d", "--format", "onnx", "--size", "64x96", "-o", str(out)]) == 0
    assert out.stat().st_size > 0
    assert "float32" in capsys.readouterr().out


@pytest.mark.model
@pytest.mark.skipif(not importlib.util.find_spec("onnxscript"), reason="needs onnxscript (dependency group export)")
def test_export_onnx_depth_cpu(capsys, tmp_path):
    _local("depth-anything-v2-s")
    out = tmp_path / "d.onnx"
    assert cli.main(["export", "depth-anything-v2-s", "--format", "onnx", "--size", "70x98", "-o", str(out)]) == 0
    assert out.stat().st_size > 0
    assert "70x98" in capsys.readouterr().out


@pytest.mark.model
@pytest.mark.skipif(not importlib.util.find_spec("onnxscript"), reason="needs onnxscript (dependency group export)")
def test_export_onnx_size_cpu(capsys, tmp_path):
    _local("hitnet-eth3d")
    out = tmp_path / "h.onnx"
    assert cli.main(["export", "hitnet-eth3d", "--format", "onnx", "--size", "64x96", "-o", str(out)]) == 0
    assert out.stat().st_size > 0
    assert "64x96" in capsys.readouterr().out


@pytest.mark.model
def test_export_without_size_names_range(capsys):
    _local("hitnet-eth3d")
    assert cli.main(["export", "hitnet-eth3d", "--format", "onnx"]) == 1
    err = capsys.readouterr().err
    assert "any H x W >=" in err
    assert "--size 704x1280" in err


@pytest.mark.model
def test_export_size_below_the_minimum(capsys):
    _local("s2m2-s")
    assert cli.main(["export", "s2m2-s", "--format", "onnx", "--size", "8x8"]) == 1
    assert "outside the supported range: any H x W >= 32, padded to a multiple of 32" in capsys.readouterr().err


@pytest.mark.model
def test_size_range_rejected_for_dav2(capsys):
    _local("depth-anything-v2-s")
    argv = ["export", "depth-anything-v2-s", "--format", "onnx", "--size", "126x126", "--size-range", "56x56:252x252"]
    assert cli.main(argv) == 1
    assert "dynamic export is not supported" in capsys.readouterr().err


def test_export_trt_needs_cuda(capsys, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert cli.main(["export", "hitnet-eth3d", "--format", "trt", "--size", "64x96"]) == 1
    assert "CUDA" in capsys.readouterr().err


class _StubModel:
    """What `fastforks export` reads from a loaded model; `module()` returns `out` (R2-1: tuple / dict outputs)."""

    def __init__(self, ref: str, task: str, out, dtype: str = "float16", names=()) -> None:
        from fastforks._core.runner import Forward  # noqa: PLC0415

        self.variant = types.SimpleNamespace(id=ref, ref=f"{ref}@1", outputs=())
        self.card = types.SimpleNamespace(task=task)
        self.backend = types.SimpleNamespace(dtype=dtype)
        self.device = torch.device("cpu")
        self._module = Forward(torch.nn.Identity(), lambda _n, _x: out, output_names=names)

    def export_tensorrt(self, path, *_a, **_k):
        Path(path).write_bytes(b"engine")

    def export_onnx(self, path, *_a, **_k):
        Path(path).write_bytes(b"onnx")

    def example_inputs(self, h, w):
        return (torch.zeros(1, 3, h, w),)

    def module(self):
        return self._module


@pytest.mark.parametrize(
    "case",
    [
        ("rtdetrv2-s-coco", "detection", (torch.ones(1, 3, 2), torch.ones(1, 3, 4)), ("logits", "boxes"), "boxes"),
        ("patchmatchnet", "multiview", {"depth": torch.ones(2, 4, 4), "confidence": torch.ones(2, 4, 4)}, (), "depth"),
    ],
)
def test_export_trt_checks_every_output(capsys, monkeypatch, tmp_path, case):
    """R2-1: the CLI's TensorRT check compares every output of a tuple / dict module and reports the worst one."""
    ref, task, out, names, worst = case
    import importlib  # noqa: PLC0415

    from fastforks._core import bench, export  # noqa: PLC0415

    flat = list(out.values()) if isinstance(out, dict) else list(out)
    keys = list(out) if isinstance(out, dict) else list(names)
    engine_out = {k: t.clone() for k, t in zip(keys, flat, strict=True)}
    engine_out[worst] = engine_out[worst] + 0.25  # the engine differs in one output only

    class Engine:
        def __init__(self, *_a) -> None:
            pass

        def run(self, *_inputs):
            return engine_out

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        importlib.import_module(f"fastforks.{task}"), "load", lambda *_a, **_k: _StubModel(ref, task, out, names=names)
    )
    monkeypatch.setattr(export, "TensorRTEngine", Engine)
    monkeypatch.setattr(bench, "median_ms", lambda *_a, **_k: 1.0)
    argv = ["export", ref, "--format", "trt", "--size", "64x64", "-o", str(tmp_path / "m.engine")]
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert f"mean |d {worst}| 0.25, max 0.25" in text
    assert "worst of 2 outputs" in text


def test_export_fp16_without_cuda_says_so(capsys, monkeypatch, tmp_path):
    """R2-10: `--fp16` on a machine without CUDA exports float32 and says so."""
    import fastforks.depth  # noqa: PLC0415

    model = _StubModel("depth-anything-v2-s", "depth", torch.zeros(1, 4, 4), dtype="float32")
    monkeypatch.setattr(fastforks.depth, "load", lambda *_a, **_k: model)
    argv = ["export", "depth-anything-v2-s", "--format", "onnx", "--size", "64x64", "--fp16"]
    assert cli.main([*argv, "-o", str(tmp_path / "d.onnx")]) == 0
    assert "--fp16 has no effect on cpu: float32 export" in capsys.readouterr().err


def test_list_commercial_yes_explains_empty_result(capsys, monkeypatch):
    monkeypatch.setattr(cli.registry, "cards", lambda _task: {})
    assert cli.main(["list", "--commercial", "yes"]) == 0
    assert "--commercial unclear" in capsys.readouterr().err


@pytest.fixture
def nvidia_driver(monkeypatch):
    """A CUDA 13.0 driver, so `env` prints its full CUDA section on any machine."""
    from fastforks._core import kernels  # noqa: PLC0415

    monkeypatch.setattr(kernels, "driver_version", lambda: (13, 0))


@pytest.fixture
def apple_gpu(monkeypatch):
    """A Mac: no NVIDIA driver, no CUDA, an Apple GPU (MPS)."""
    from fastforks._core import kernels  # noqa: PLC0415

    def no_driver():
        msg = "libcuda.so.1: cannot open shared object file"
        raise OSError(msg)

    monkeypatch.setattr(kernels, "driver_version", no_driver)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


def test_env_on_a_mac(apple_gpu, capsys, monkeypatch):
    """TM-3: Apple GPU and Metal libraries first, one short CUDA line, no cuDNN load."""
    from fastforks.kernels.cudnn_hacc import _backend  # noqa: PLC0415

    monkeypatch.setattr(_backend, "version", lambda: pytest.fail("cuDNN loaded without CUDA"))
    assert cli.main(["env"]) == 0
    lines = capsys.readouterr().out.splitlines()
    keys = [line.split("  ")[0] for line in lines]
    assert keys.index("apple gpu") < keys.index("metal stereo_metal") < keys.index("cuda")
    assert not [k for k in keys if k.startswith(("kernels ", "driver", "gpu"))], keys
    assert lines[keys.index("cuda")].split(None, 1)[1].startswith("not available")


def test_download_kernels_on_a_mac(apple_gpu, monkeypatch, capsys):
    """TM-2: `download --kernels` on an Apple GPU fetches no CUDA binaries and says why, exit 0."""
    monkeypatch.setattr(cli, "_kernel_files", lambda *_a, **_k: pytest.fail("CUDA binaries resolved"))
    assert cli.main(["download", "hitnet-eth3d", "--kernels-only", "--dry-run"]) == 0
    assert "Metal kernels (stereo_metal) are compiled locally on first use; nothing to fetch" in capsys.readouterr().out
    assert cli.main(["download", "densenet121.tv_in1k", "--kernels-only", "--check"]) == 0
    assert "no kernels for Apple GPUs" in capsys.readouterr().out


def test_download_kernels_without_any_gpu_still_asks_for_arch(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert cli.main(["download", "hitnet-eth3d", "--kernels-only", "--dry-run"]) == 1
    assert "--arch sm_XX" in capsys.readouterr().err


def test_env(nvidia_driver, capsys):
    assert cli.main(["env"]) == 0
    out = capsys.readouterr().out
    for key in ("fastforks", "python", "torch", "driver", "gpu", "cache", "switches"):
        assert f"\n{key} " in f"\n{out}", key
    assert "\nkernels (" in f"\n{out}"  # the identical per-library lines are one


def test_env_survives_a_broken_cudnn(nvidia_driver, capsys, monkeypatch):
    """A missing / broken libcudnn raises OSError, not ImportError; `env` still prints its rows."""
    from fastforks.kernels.cudnn_hacc import _backend  # noqa: PLC0415

    def fail() -> int:
        msg = "libcudnn.so.9: cannot open shared object file"
        raise OSError(msg)

    monkeypatch.setattr(_backend, "version", fail)
    assert cli.main(["env"]) == 0
    assert "unusable: OSError: libcudnn.so.9" in capsys.readouterr().out


def test_size_range_parser():
    assert cli._bounds("320x320:1088x1920") == ((320, 320), (1088, 1920))
    with pytest.raises(argparse.ArgumentTypeError, match="MINxMIN:MAXxMAX"):
        cli._bounds("320x320")


def _pair_dirs(tmp_path, n):
    from PIL import Image  # noqa: PLC0415

    from synth import pair  # noqa: PLC0415

    for side in ("L", "R"):
        (tmp_path / side).mkdir()
    for i in range(n):
        for side, img in zip(("L", "R"), pair(64, 96, i), strict=True):
            Image.fromarray(img).save(tmp_path / side / f"frame_{i}.png")
    return str(tmp_path / "L"), str(tmp_path / "R")


@pytest.mark.model
def test_stereo_run_cpu(capsys, tmp_path):
    _local("hitnet-eth3d")
    left, right = _pair_dirs(tmp_path, 3)
    out = tmp_path / "out"
    argv = ["stereo", "run", left, right, "--model", "hitnet-eth3d", "--out", str(out), "--device", "cpu"]
    assert cli.main([*argv, "--format", ".npy"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["frame_0.npy", "frame_1.npy", "frame_2.npy"]
    assert "3 pairs -> " in capsys.readouterr().out
    (tmp_path / "L" / "frame_1.png").write_bytes(b"broken")
    assert cli.main([*argv, "--skip-errors"]) == 1  # a failed pair is a non-zero exit
    captured = capsys.readouterr()
    assert "2 pairs -> " in captured.out
    assert "1 failed" in captured.out
    assert "failed pair 1" in captured.err


def test_stereo_run_names_bad_inputs(capsys, tmp_path, monkeypatch):
    from fastforks import stereo  # noqa: PLC0415

    def no_load(*_a, **_k):
        msg = "the model must not load before the inputs are checked"
        raise AssertionError(msg)

    monkeypatch.setattr(stereo, "load", no_load)
    left, right = _pair_dirs(tmp_path, 2)
    (tmp_path / "R" / "frame_1.png").unlink()
    argv = ["stereo", "run", left, right, "--model", "hitnet-eth3d", "--out", str(tmp_path / "o"), "--device", "cpu"]
    assert cli.main(argv) == 1
    assert "2 left images but 1 right images" in capsys.readouterr().err


def test_doctor_lists_optional_libraries_and_blocked_models(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "timm", None)
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "timm" in out
    assert "pip install 'fastforks-core[timm]'" in out
    assert "cannot run yet" in out
    assert "more)" in out
    assert "torch" in out  # the `env` part


def test_doctor_names_noncommercial_weights_apart_from_missing_libraries(capsys, monkeypatch):
    monkeypatch.delenv("FASTFORKS_NONCOMMERCIAL", raising=False)
    monkeypatch.setattr(registry, "present", lambda *_: False)  # local copies count as runnable
    assert cli.main(["doctor"]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if "FASTFORKS_NONCOMMERCIAL=1" in x)
    assert line.startswith("cannot run yet")
    assert "install the libraries" not in line


def test_list_installed_and_info_requires(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "timm", None)
    monkeypatch.setitem(sys.modules, "torchvision", None)
    assert cli.main(["list", "--runnable", "--task", "classification"]) == 0
    assert capsys.readouterr().out == ""
    assert cli.main(["info", "deit_base_patch16_224.fb_in1k"]) == 0
    assert "requires" in capsys.readouterr().out


@pytest.mark.slow
def test_bench_compares_portable_and_kernels(monkeypatch, capsys):
    """Without --mode, both backends are timed (the kernels one is reported unavailable here) and shown in a table."""
    calls = []

    def fake(args, card, mode):
        calls.append(mode)
        if mode == "kernels":
            msg = "no kernels here\nsecond line"
            raise BackendUnavailableError(msg)
        return "x@1", types.SimpleNamespace(mode=mode, dtype="float16"), 12.34, (100, 200)

    monkeypatch.setattr(cli, "_bench_mode", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("fastforks._core.model.resolve_device", lambda *_: torch.device("cuda", 0))
    assert cli.main(["bench", "hitnet-eth3d", "--size", "64x96", "--device", "cuda"]) == 0
    out = capsys.readouterr().out
    assert calls == ["portable", "kernels"]
    assert "portable" in out
    assert "12.3" in out
    assert "kernels" in out
    assert "not available: no kernels here" in out
    assert "second line" not in out


@pytest.mark.slow
@pytest.mark.parametrize(("device", "modes"), [("mps", ["portable", "kernels"]), ("cpu", ["portable"])])
def test_bench_offers_kernels_per_device(monkeypatch, capsys, device, modes):
    """The portable / kernels comparison covers Apple MPS (Metal kernels), not only CUDA; the CPU has no kernels."""
    calls = []

    def fake(args, card, mode):
        calls.append(mode)
        return "x@1", types.SimpleNamespace(mode=mode, dtype="float16"), 1.0, None

    monkeypatch.setattr(cli, "_bench_mode", fake)
    monkeypatch.setattr("fastforks._core.model.resolve_device", lambda *_: torch.device(device))
    assert cli.main(["bench", "hitnet-eth3d", "--size", "64x96", "--device", device]) == 0
    assert calls == modes


def test_info_lists_metal_libs(capsys):
    assert cli.main(["info", "moge-3-vitl"]) == 0
    assert "Apple MPS: sparse_metal" in capsys.readouterr().out


def test_bench_mps_memory_line(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """On MPS only the driver's (reserved) memory is known; the one-line form leaves allocated out."""
    monkeypatch.setattr(
        cli,
        "_bench_mode",
        lambda *_: ("x@1", types.SimpleNamespace(mode="portable", dtype="float32"), 12.34, (None, 300)),
    )
    assert cli.main(["bench", "hitnet-eth3d", "--size", "64x96", "--mode", "portable"]) == 0
    out = capsys.readouterr().out
    assert "12.3 ms" in out
    assert "peak 300 MiB reserved" in out
    assert "allocated" not in out


def _fake_kernel_files(monkeypatch):
    from fastforks._core.files import FileSpec  # noqa: PLC0415

    seen = []

    def files(lib, *, archs, all_archs):
        seen.append(lib)
        return [(f"{lib}:sm_89", FileSpec(name="k.so", sha256="0" * 64, size=2**20, url="u"))]

    monkeypatch.setattr(cli, "_kernel_files", files)
    return seen


def test_download_kernels_only_fetches_no_weights(monkeypatch, capsys):
    seen = _fake_kernel_files(monkeypatch)
    monkeypatch.setattr(cli.registry, "weights", lambda *_a, **_k: pytest.fail("weights touched"))
    assert cli.main(["download", "--kernels-only", "--all-archs", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "licence" not in out
    assert ":sm_89: 1 MB" in out
    assert seen


def test_download_kernels_only_restricts_to_named_models(monkeypatch, capsys):
    seen = _fake_kernel_files(monkeypatch)
    card, _ = cli.registry.resolve_any("hitnet-eth3d")
    assert cli.main(["download", "hitnet-eth3d", "--kernels-only", "--arch", "sm_89", "--dry-run"]) == 0
    assert seen == sorted(card.kernel_libs)
    capsys.readouterr()


def test_download_kernels_only_check_is_offline(monkeypatch, capsys):
    _fake_kernel_files(monkeypatch)
    monkeypatch.setattr(cli.registry, "weights", lambda *_a, **_k: pytest.fail("weights touched"))

    def nothing(*a, **k):
        msg = "not cached"
        raise FastforksError(msg)

    monkeypatch.setattr(cli, "locate", nothing)
    assert cli.main(["download", "hitnet-eth3d", "--kernels-only", "--arch", "sm_89", "--check"]) == 1
    assert "MISSING" in capsys.readouterr().err


def test_download_kernels_only_all_archs_dry_run_real_manifests(capsys):
    assert cli.main(["download", "--kernels-only", "--all-archs", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.strip()
    assert "licence" not in out


def test_list_and_info_flag_copyleft(capsys):
    assert cli.main(["list", "coex-sceneflow"]) == 0
    assert "commercial=yes (copyleft)" in capsys.readouterr().out
    assert cli.main(["info", "coex-sceneflow"]) == 0
    assert "copyleft: its obligations apply" in capsys.readouterr().out
    assert cli.main(["list", "las2-s"]) == 0
    assert "commercial=yes " in capsys.readouterr().out


def test_list_no_match_names_the_pattern(capsys):
    assert cli.main(["list", "s2m2-zz"]) == 0  # a filter, not an error (as for the `--commercial yes` hint)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no model matches 's2m2-zz'" in captured.err
    assert "s2m2-s" in captured.err  # close matches


@pytest.mark.parametrize("flag", ["--size 0x0", "--size 5x0", "--runs 0", "--views 0", "--runs -1"])
def test_bench_rejects_nonpositive_numbers(flag, capsys):
    with pytest.raises(SystemExit):
        cli.main(["bench", "hitnet-eth3d", *flag.split()])
    assert "error" in capsys.readouterr().err


@pytest.mark.slow
def test_bench_float32_skips_the_kernels_row(monkeypatch, capsys):
    from fastforks import OptionsError  # noqa: PLC0415

    def fake(args, card, mode):
        if mode == "kernels":
            msg = "kernels runs reduced-precision kernels here; use mode='portable' with dtype='float32'"
            raise OptionsError(msg)
        return "x@1", types.SimpleNamespace(mode="portable", dtype="float32"), 1.0, None

    monkeypatch.setattr(cli, "_bench_mode", fake)
    monkeypatch.setattr("fastforks._core.model.resolve_device", lambda *_: torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cli.main(["bench", "hitnet-eth3d", "--dtype", "float32", "--device", "cuda"]) == 0
    assert "not available: kernels runs reduced-precision" in capsys.readouterr().out


def test_export_output_creates_parent_and_reports_os_errors(tmp_path, capsys):
    out = tmp_path / "new" / "sub" / "f.onnx"
    assert cli.main(["export", "no-such-model", "--format", "onnx", "-o", str(out)]) == 1  # fails after the mkdir
    assert out.parent.is_dir()
    assert "fastforks: error:" in capsys.readouterr().err
    (tmp_path / "file").write_text("x")
    assert cli.main(["export", "hitnet-eth3d", "--format", "onnx", "-o", str(tmp_path / "file" / "f.onnx")]) == 1
    assert "fastforks: error: cannot write" in capsys.readouterr().err


def test_download_dry_run_and_check_handle_noncommercial_weights(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("FASTFORKS_NONCOMMERCIAL", raising=False)
    monkeypatch.setattr(cli.registry, "weights", lambda *_a, **_k: [tmp_path / "w.bin"])
    assert cli.main(["download", "--dry-run", "segformer-b0-ade20k"]) == 0
    out = capsys.readouterr().out
    assert "needs FASTFORKS_NONCOMMERCIAL=1" in out
    assert "weights " in out  # the licence class line
    assert cli.main(["download", "--check", "segformer-b0-ade20k"]) == 0  # resolves locally: ok, no gate
    assert "ok (" in capsys.readouterr().out

    def missing(*_a, **_k):
        msg = "not local"
        raise FastforksError(msg)

    monkeypatch.setattr(cli.registry, "weights", missing)
    assert cli.main(["download", "--check", "segformer-b0-ade20k"]) == 1
    assert "needs FASTFORKS_NONCOMMERCIAL=1 to download" in capsys.readouterr().err
    assert cli.main(["download", "segformer-b0-ade20k"]) == 1  # a real download stays gated
    assert "non-commercial weights" in capsys.readouterr().err


def test_info_shows_option_bounds_and_auto_choice(capsys):
    assert cli.main(["info", "igev-stereo-sceneflow"]) == 0
    assert "iters=32 (>= 1)" in capsys.readouterr().out
    assert cli.main(["info", "segformer-b0-ade20k"]) == 0
    assert "auto: portable on CUDA" in capsys.readouterr().out


def test_stereo_run_size_reads_calibration_without_image_size(monkeypatch, capsys, tmp_path):
    from fastforks import stereo  # noqa: PLC0415
    from fastforks.stereo.geometry import StereoRig  # noqa: PLC0415

    seen = {}

    class StopError(FastforksError):
        pass

    class Model:
        def run_many(self, *_a, **kw):
            seen.update(kw)
            msg = "stop"
            raise StopError(msg)

    monkeypatch.setattr(stereo, "load", lambda *_a, **_k: Model())
    left, right = _pair_dirs(tmp_path, 1)
    rig = str(Path(__file__).parent / "data" / "opencv_stereo_rectify.yml")
    argv = ["stereo", "run", left, right, "--model", "hitnet-eth3d", "--out", str(tmp_path / "o"), "--rig", rig]
    assert cli.main([*argv, "--size", "720x1280", "--focal", "650"]) == 1  # stops after the rig was read with the size
    assert isinstance(seen["rig"], StereoRig)
    assert seen["focal"] == 650
    assert cli.main(["stereo", "run", left, right, "--model", "x", "--out", "o", "--size", "720x1280"]) == 1
    assert "pass --rig" in capsys.readouterr().err
