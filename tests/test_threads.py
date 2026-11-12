"""Thread / process safety on the CPU: shared models, concurrent loads, concurrent cache writes."""

from __future__ import annotations

import concurrent.futures as cf
import gc
import hashlib
import subprocess
import sys
import threading
from typing import Any

import numpy as np
import pytest
import torch

from fastforks._core import files
from fastforks._core.files import FileSpec, locate
from fastforks._core.tasks import spec
from fastforks.kernels import cudnn_hacc
from fastforks.shared import cuda_graphs
from synth import image, pair
from test_golden import API, TASK

REFS = ["hitnet-eth3d@1", "depth-anything-v2-s@1", "mobilenet_v2.tv2_in1k@1", "segformer-b0-ade20k@1"]
ENV = "FF_TEST_PATH"


def _call(model: Any, ref: str, seed: int = 0) -> torch.Tensor:
    """The main map of one call on a seeded synthetic input (copied off the model's buffers)."""
    if TASK[ref] in {"stereo", "flow", "matching"}:
        left, right = pair(96, 128, seed)
        result = model(left, right)
    else:
        result = model(image(96, 128, seed))
    name = next(k for k in (*spec(TASK[ref]).standard, "labels") if getattr(result, k, None) is not None)
    return getattr(result, name).detach().clone()


def _load(ref: str) -> Any:
    return API[TASK[ref]].load(ref, device="cpu", mode="portable")


def _run_threads(fns: list) -> list:
    barrier = threading.Barrier(len(fns))

    def go(fn):
        barrier.wait()
        return fn()

    with cf.ThreadPoolExecutor(len(fns)) as pool:
        return [f.result() for f in [pool.submit(go, fn) for fn in fns]]


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", REFS)
def test_two_threads_share_one_model(ref):
    """Calls on one loaded model serialise on its lock; both threads get the single-thread outputs."""
    model = _load(ref)
    want = [_call(model, ref, s) for s in (0, 1)]
    got = _run_threads([lambda s=s: [_call(model, ref, s) for _ in range(3)] for s in (0, 1)])
    for s in (0, 1):
        for out in got[s]:
            assert torch.equal(out, want[s])


@pytest.mark.slow
@pytest.mark.model
def test_threads_use_different_models():
    refs = REFS[:2] + REFS[2:]
    models = {r: _load(r) for r in refs}
    want = {r: _call(models[r], r) for r in refs}
    got = _run_threads([lambda r=r: [_call(models[r], r) for _ in range(3)] for r in refs])
    for r, outs in zip(refs, got, strict=True):
        for out in outs:
            assert torch.equal(out, want[r])


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", REFS)
def test_two_threads_load_the_same_ref(ref):
    """Concurrent loads (weights, registry caches) give independent models with the same outputs."""
    want = _call(_load(ref), ref)
    models = _run_threads([lambda: _load(ref), lambda: _load(ref)])
    assert models[0] is not models[1]
    for m in models:
        assert torch.equal(_call(m, ref), want)


def test_concurrent_min_size_probe_is_consistent():
    """The lazy `_min_size` is set by whichever call comes first; racing first calls agree."""
    ref = "mobilenet_v2.tv2_in1k@1"
    pytest.importorskip("timm")
    model = _load(ref)
    sizes = _run_threads([lambda: (model._check_size(torch.zeros(3, 96, 96)), model._min_size)[1] for _ in range(4)])
    assert len(set(sizes)) == 1


def _spec(data: bytes, url: str) -> FileSpec:
    return FileSpec("w.bin", hashlib.sha256(data).hexdigest(), len(data), url)


def test_threads_download_the_same_file(tmp_path, monkeypatch):
    """Concurrent downloads of one file into a cold cache: every caller gets the verified file, no stray parts."""
    data = np.random.default_rng(0).bytes(1 << 20)
    src = tmp_path / "src.bin"
    src.write_bytes(data)
    monkeypatch.setenv("FASTFORKS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(ENV, "")
    s = _spec(data, src.as_uri())
    paths = _run_threads([lambda: locate(s, group="fam", env_var=ENV) for _ in range(6)])
    assert len(set(paths)) == 1
    assert paths[0].read_bytes() == data
    leftovers = [p for p in files.cache_root().rglob("*") if p.is_file() and p != paths[0] and "stamps" not in p.parts]
    assert not leftovers
    assert not list(files.cache_root().rglob(".download.*"))


_CHILD = """
import sys
from fastforks._core.files import FileSpec, locate
sha, size, url, out = sys.argv[1:5]
p = locate(FileSpec("w.bin", sha, int(size), url), group="fam", env_var="FF_TEST_PATH")
open(out, "w").write(str(p))
"""


def test_processes_download_the_same_file(tmp_path, monkeypatch):
    """Two processes filling one cold cache at once both end with the verified file."""
    data = np.random.default_rng(1).bytes(1 << 20)
    src = tmp_path / "src.bin"
    src.write_bytes(data)
    monkeypatch.setenv("FASTFORKS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(ENV, "")
    monkeypatch.setenv(files.HF_ENDPOINT_ENV, tmp_path.as_uri())  # file:// is only allowed below the endpoint
    sha = hashlib.sha256(data).hexdigest()
    procs = [
        subprocess.Popen(
            [sys.executable, "-I", "-c", _CHILD, sha, str(len(data)), src.as_uri(), str(tmp_path / f"out{i}")],
            env={**__import__("os").environ, "PYTHONPATH": ":".join(sys.path)},
        )
        for i in range(2)
    ]
    assert [p.wait(120) for p in procs] == [0, 0]
    paths = {(tmp_path / f"out{i}").read_text() for i in range(2)}
    assert len(paths) == 1
    assert files.Path(paths.pop()).read_bytes() == data
    assert not list(files.cache_root().rglob(".download.*"))


# --- repeated calls on the GPU: memory stays flat, outputs repeat --------------------------------------------------

LEAK_REFS = ["s2m2-s@1", "hitnet-eth3d@1", "depth-anything-v2-s@1", "mobilenet_v2.tv2_in1k@1", "raft-small@1"]
RESERVED_SLACK = 4 << 20  # one late allocator block (DAv2 kernels: +2 MiB once, then flat)
LEAK_SIZES = ((384, 512), (480, 640))


def _gpu_call(model: Any, ref: str, size: tuple[int, int]) -> torch.Tensor:
    """The main map of one call on a seeded input of `size`, left on the GPU."""
    h, w = size
    result = model(*pair(h, w, 0)) if TASK[ref] in {"stereo", "flow", "matching"} else model(image(h, w, 0))
    name = next(k for k in (*spec(TASK[ref]).standard, "labels") if getattr(result, k, None) is not None)
    return getattr(result, name)


def leak_check(ref: str, mode: str, same: int = 100, alternating: int = 50) -> dict[str, Any]:
    """Memory and output stability of repeated calls on a CUDA model.

    `same` calls at one size, then `alternating` calls over two sizes twice ("alternating", "settled"), after a warm-up
    of both sizes. Returns the `memory_allocated` / `memory_reserved` after warm-up and after each phase (bytes), the
    peak allocation, and, per size, the largest absolute difference of any call to the first warm call (`max_diff`,
    0.0 = bitwise equal) and of the very first (cold) call to it (`cold_diff`).
    """
    gc.collect()
    torch.cuda.empty_cache()
    model = API[TASK[ref]].load(ref, device="cuda", mode=mode)
    cold = {s: _gpu_call(model, ref, s).clone() for s in LEAK_SIZES}
    for s in LEAK_SIZES * 2:
        _gpu_call(model, ref, s)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = {s: _gpu_call(model, ref, s).clone() for s in LEAK_SIZES}
    cold_diff = {s: float((cold[s].float() - base[s].float()).abs().max()) for s in LEAK_SIZES}
    del cold
    gc.collect()
    mem = {"warm": (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())}
    diff = dict.fromkeys(LEAK_SIZES, 0.0)

    def run(sizes: list[tuple[int, int]]) -> None:
        for s in sizes:
            out = _gpu_call(model, ref, s)
            diff[s] = max(diff[s], float((out.float() - base[s].float()).abs().max()))

    run([LEAK_SIZES[0]] * same)
    torch.cuda.synchronize()
    mem["same"] = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    for phase in ("alternating", "settled"):
        run([LEAK_SIZES[i % 2] for i in range(alternating)])
        torch.cuda.synchronize()
        mem[phase] = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    peak = torch.cuda.max_memory_allocated()
    return {
        "mem": mem,
        "max_diff": {f"{h}x{w}": d for (h, w), d in diff.items()},
        "cold_diff": {f"{h}x{w}": d for (h, w), d in cold_diff.items()},
        "peak_allocated": peak,
    }


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("mode", ["portable", "kernels"])
@pytest.mark.parametrize("ref", LEAK_REFS)
def test_repeated_calls_are_flat_and_repeatable(ref, mode):
    """Memory is flat after warm-up (allocated exactly; reserved within one allocator block) and outputs repeat.

    Reserved may grow once when the sizes first alternate (a pool fragmented by earlier tests in the process: +80 MiB
    for HITNet portable after other tests, flat alone), so it is checked against the phase before: a leak keeps growing.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    r = leak_check(ref, mode, same=20, alternating=10)
    warm_a = r["mem"]["warm"][0]
    for phase, before in (("same", "warm"), ("alternating", None), ("settled", "alternating")):
        a, res = r["mem"][phase]
        assert a == warm_a, f"{ref} {mode}: allocated grew in '{phase}': {warm_a} -> {a}"
        if before is not None:
            ref_r = r["mem"][before][1]
            assert res <= ref_r + RESERVED_SLACK, f"{ref} {mode}: reserved grew in '{phase}': {ref_r} -> {res}"
    assert not any(r["max_diff"].values()), r["max_diff"]  # bitwise equal to the first warm call, both backends


# --- kernels: the first call at a new size returns the warm output; concurrent CUDA threads
# ----------------------------

# one per family kind of kernels runner: cudnn_hacc plans (all), CUDA graphs (the four graph families: separate capture
# code), CUTLASS + ViT (FoundationStereo), a depth ViT + DPT head, a classifier CNN
GRAPH_REFS = [
    "fast-acvnet-plus-generalization@1",
    "coex-sceneflow@1",
    "las2-s@1",
    "rt-monster-plusplus-mix-all@1",
]
FIRST_REFS = [
    "s2m2-s@1",
    "raft-stereo-middlebury@1",
    *GRAPH_REFS,
    "foundationstereo-vit-s@1",
    "igev-plusplus-middlebury@1",
    "depth-anything-v2-s@1",
    "resnet50.tv2_in1k@1",
    "dinov3-convnext-tiny@1",  # fvm_conv max, CUDA graph from the second call
    "raft-small@1",  # the flow engines: raw raft_flow launches around CUTLASS and cudnn_hacc convs
    "sea-raft-s@1",
]
FIRST_SIZE = (328, 584)  # used by no other test: the call below is the first at this size in the process
OPT_IN_SIZE = (264, 488)  # used by no other test either (BUILD_AFTER = 3 opt-in)


def _calls_with_captures(model: Any, ref: str, size: tuple[int, int], n: int, monkeypatch: Any) -> tuple[list, list]:
    """`n` calls at `size`: (outputs, the 1-based call numbers that captured a CUDA graph)."""
    calls, captured = [0], []
    orig = cuda_graphs.GraphCache._capture

    def spy(self: Any, inputs: Any) -> Any:
        captured.append(calls[0])
        return orig(self, inputs)

    monkeypatch.setattr(cuda_graphs.GraphCache, "_capture", spy)
    outs = []
    for _ in range(n):
        calls[0] += 1
        outs.append(_gpu_call(model, ref, size).clone())
    return outs, captured


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("ref", FIRST_REFS)
def test_kernels_first_call_equals_warm_call(ref, monkeypatch):
    """Calls 1-6 at a new size are bitwise equal (cuDNN plans are built on a geometry's first call).

    The graph families capture their CUDA graph on call 4 (`cudnn_hacc.capture_after()`) and replay it on calls 5-6:
    the graph runs the eager plans. With `cudnn_hacc.BUILD_AFTER = 3` (the old default) calls 1-3 ran torch
    convolutions and differed (S2M2-S: mean 0.012 px, max 2.3 px).
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = API[TASK[ref]].load(ref, device="cuda", mode="kernels")
    outs, captured = _calls_with_captures(model, ref, FIRST_SIZE, 6, monkeypatch)
    for i in range(5):
        assert torch.equal(outs[i], outs[5]), f"{ref}: call {i + 1} != call 6: {(outs[i] - outs[5]).abs().max()}"
    if ref in GRAPH_REFS:
        assert captured == [4], f"{ref}: CUDA graph captured on calls {captured}, expected [4]"
    elif ref == "dinov3-convnext-tiny@1":  # `SmallShapeGraphs`: captured on the size's second call, replayed after
        assert captured == [2], f"{ref}: CUDA graph captured on calls {captured}, expected [2]"


@pytest.mark.gpu
@pytest.mark.model
def test_kernels_build_after_opt_in_on_a_graph_model(monkeypatch):
    """`BUILD_AFTER = 3` (set before loading): calls 1-3 run torch convs, call 4 builds the plans and captures.

    Calls 4-6 (capture, two replays) are bitwise equal; calls 1-3 differ slightly from them.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    ref = "fast-acvnet-plus-generalization@1"
    monkeypatch.setattr(cudnn_hacc, "BUILD_AFTER", 3)
    model = API[TASK[ref]].load(ref, device="cuda", mode="kernels")
    outs, captured = _calls_with_captures(model, ref, OPT_IN_SIZE, 6, monkeypatch)
    assert captured == [4]
    assert torch.equal(outs[3], outs[5])
    assert torch.equal(outs[4], outs[5])
    for i in range(3):
        diff = (outs[i] - outs[5]).abs()
        assert not torch.equal(outs[i], outs[5]), f"call {i + 1}: expected torch convs, got the plans' output"
        assert diff.mean() < 0.05, f"call {i + 1}: mean diff {diff.mean()} px"


@pytest.mark.gpu
@pytest.mark.model
def test_cuda_threads_on_two_streams_match_single_thread():
    """Two kernels models in two threads, each on its own CUDA stream, give their single-thread outputs.

    FoundationStereo sets the process-global SDPA backend switches and the cuBLAS fp16-accumulate flag, S2M2 the flag;
    without the process-wide CUDA call lock the two saw each other's settings (FoundationStereo: up to 1.3 px off).
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    refs = ["foundationstereo-vit-s@1", "s2m2-s@1"]
    models = [API["stereo"].load(r, device="cuda", mode="kernels") for r in refs]
    size = (256, 384)
    want = [_gpu_call(m, r, size).clone() for m, r in zip(models, refs, strict=True)]
    streams = [torch.cuda.Stream() for _ in refs]

    def run(i: int) -> list[torch.Tensor]:
        with torch.cuda.stream(streams[i]):
            outs = [_gpu_call(models[i], refs[i], size).clone() for _ in range(4)]
        streams[i].synchronize()
        return outs

    got = _run_threads([lambda i=i: run(i) for i in range(len(refs))])
    for r, w, outs in zip(refs, want, got, strict=True):
        for out in outs:
            assert torch.equal(out, w), f"{r}: {(out - w).abs().max()}"
