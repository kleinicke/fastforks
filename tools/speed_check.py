"""Speed + memory regression check of the portable / kernels modes against the committed floor (docs/data/speed.jsonl).

    python tools/speed_check.py list [--only REF ...]                 # jobs = the floor's (ref, mode) with their inputs
    python tools/speed_check.py run --out rows.jsonl [--only REF ...] [--changed-since COMMIT] [--modes kernels]
        [--prefix "timeout 300"] [--forks plugin.py] [--check]
    python tools/speed_check.py check rows.jsonl                      # rows vs the floor: exit 1 on a regression
    python tools/speed_check.py static                                 # no GPU: mode ordering of the floor rows
    python tools/speed_check.py report rows.jsonl                      # library vs `--forks` side, markdown
    python tools/speed_check.py export rows.jsonl --source FILE        # library rows as floor rows (stdout)
    python tools/speed_check.py run --trace-diff --out trace.jsonl --forks plugin.py; ... trace-diff trace.jsonl

`run` measures one (ref, mode) per subprocess (a timeout or crash becomes an error row), appends JSON rows to `--out`.
Protocol per input (smallest first, same inputs as the floor rows: real KITTI / Middlebury images where the floor
names them, else the task's synthetic `bench_inputs`): first call, 3 warm-up calls, peak reset, then 3 rounds of 5
timed calls (CUDA-synchronised); `ms` = median of the round medians, `noise` = (max - min) / median of the round
medians, peaks = max over the first and the timed calls. Each (ref, side, mode) runs in a fresh process (`cold`):
`torch_import_s`, `cuda_init_s`, `import_s` (the fastforks task module), `load_s` (the load call until the model is on
the GPU; fork side: the fork's own imports included), per input `first_call_s` and its peak (`peak_first_*_mb`;
`first_call_cold` on the job's first input, later inputs are a new shape in the warm process). Each row also records
the time of the task runner alone on the tensors the full call passed it (`core_ms`) and a short profile of the full
call on one input per task (CUDA time share of layout copies / casts, other elementwise kernels, host syncs per call,
top kernels). `export` turns the library rows into floor rows (the fields of docs/data/speed.jsonl).

`run --modes compile tensorrt --out docs/data/speed_compile.jsonl` measures, per ref of the floor's portable rows at the
task's standard input (`STANDARD`, else its smallest), `model.compile()` on portable (`compile`: a fresh process
with empty Inductor / Triton caches, so `first_call_s` = `compile_s` is the cold compile; warm `ms`, peaks as above) and
the fp16 TensorRT engine of the portable model (`tensorrt`: `onnx_export_s`, `engine_build_s`, warm `ms` of
`TensorRTEngine` on synthetic inputs of the timed call's network input shape, `device_mb` = device memory the engine
holds after loading and running it). A failure (error, build failure, timeout) becomes a row with
`fails: <short reason>`, not a crash.

Rows are keyed by (ref, mode, input, code hash): the hash covers the fastforks source files the job imported, the
kernel ids of its backend and the torch version, so a later `run` skips jobs whose files did not change;
`--changed-since` runs only jobs whose recorded files changed since that commit (both read the earlier rows of
`--out`). `--check` (or `check`) fails when a row is slower than floor x (1 + max(5 %, its noise band)) or its peak
allocated / reserved MB exceeds the floor by more than 5 % (and 8 MB).

`run --trace-diff` writes, per (ref, side, mode) on the task's profile input, the warm wall time, the summed CUDA
kernel time, their gap (launch / CPU overhead) and the kernel time per group (kernel family; launching op); the
`trace-diff` command lists, per model, the groups where the library spends more than the fork (> 10 % and > 0.05 ms
per call), also when the library is faster in total (a regression masked by another gain).

`--forks` names a plugin file (outside the package) that adds a reference side, e.g. the pre-library fork of a model:
`FORKS[(ref, mode)] = {"python": ..., "env": {...}, "key": ...}` and `fork_job(spec, task, labels) -> (info, [(label,
call)])`, run in that python. Side order alternates per job. `static` checks, per ref and input of the floor, that
kernels is not slower than portable and neither is slower than the faster original (`upstream` / `upstream-default`)
by more than 3 %; known exceptions with their measured reason are in docs/data/speed_allowlist.json.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _data import DATA

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "docs" / "data" / "speed.jsonl"
ALLOW = ROOT / "docs" / "data" / "speed_allowlist.json"
MODES = ("portable", "kernels")
NEW_MODES = ("compile", "tensorrt")  # portable + `model.compile()`; the fp16 TensorRT engine of portable
COMPILE_DATA = ROOT / "docs" / "data" / "speed_compile.jsonl"
WARMUP, RUNS, ROUNDS = 3, 5, 3
TOL_MS, TOL_MB, TOL_MB_ABS, TOL_STATIC = 0.05, 0.05, 8, 0.03
SCENES = {  # floor scene prefix -> left image (right: image_3 / im1; flow: the next frame _11), under $FASTFORKS_DATA
    "KITTI": DATA / "KITTI/training/image_2/000000_10.png",
    "Middlebury": DATA / "Middlebury/MiddEval3/trainingH/Motorcycle/im0.png",
}
VIEWS = {"mapanything-apache": 9, "patchmatchnet": 5}  # multi-view scenes of the floor (other models: 8 views)
SWAPPED = {"patchmatchnet"}  # floor input label is WxH
STANDARD = {  # the task's standard input of docs/benchmarks.md (tools/gen_docs.py `_STD_SIZE`): compile / tensorrt rows
    "stereo": "994x1482",
    "depth": "994x1482",
    "segmentation": "1024x2048",
    "classification": "224x224",
    "features": "224x224",
    "promptable_segmentation": "1024x1024",
    "detection": "800x1344",
    "flow": "1080x1920",
    "matching": "768x1024",
    "matting": "1080x1920",
    "tracking": "480x854",
    "interpolation": "1080x1920",
    "retrieval": "480x640",
}
PROFILE_INPUT = {
    "stereo": "994x1482",
    "depth": "994x1482",
    "segmentation": "512x512",
    "promptable_segmentation": "840x1297",
}
COPY = ("copy", "memcpy", "nchwtonhwc", "nhwctonchw", "transpose", "catarraybatched", "convert", "permute")
SYNC = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "aten::_local_scalar_dense")


def floor_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FLOOR.read_text().splitlines() if line.strip()]


def _px(label: str) -> int:
    h, _, w = label.partition("x")
    return int(h) * int(w)


def jobs(only: Iterable[str] = (), modes: Iterable[str] = MODES) -> list[dict[str, Any]]:
    """(ref, task, mode) of the floor with its inputs `[{"label", "scene"}]`, smallest first.

    `compile` / `tensorrt` jobs are the refs of the floor's portable rows, with their standard input only.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    only, want = set(only), set(modes)
    read = (want - set(NEW_MODES)) | ({"portable"} if want & set(NEW_MODES) else set())
    for r in floor_rows():
        if r["mode"] not in read or (only and r["ref"] not in only):
            continue
        j = out.setdefault((r["ref"], r["mode"]), {"ref": r["ref"], "task": r["task"], "mode": r["mode"], "inputs": []})
        j["inputs"].append({"label": r["input"], "scene": r.get("scene", "")})
    for j in out.values():
        j["inputs"].sort(key=lambda i: _px(i["label"]))
    found = []
    for j in out.values():
        if j["mode"] in want:
            found.append(j)
        if j["mode"] == "portable":
            std = next((i for i in j["inputs"] if i["label"] == STANDARD.get(j["task"])), j["inputs"][0])
            found += [{**j, "mode": m, "inputs": [std]} for m in NEW_MODES if m in want]
    return found


def files_hash(files: Iterable[str]) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        p = ROOT / f
        h.update(f.encode() + b"\0" + (p.read_bytes() if p.exists() else b"<missing>"))
    return h.hexdigest()[:16]


def _commit() -> str:
    r = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "src"], capture_output=True, text=True, check=False
    )
    return r.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


# Worker side: imports torch lazily.
def _image(path: Path) -> Any:  # noqa: ANN401 - torch tensor (torch is imported lazily)
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1).cuda()


def lib_inputs(model: Any, task: str, ref: str, inp: dict[str, str]) -> tuple[tuple[Any, ...], dict[str, Any]]:  # noqa: ANN401
    """The call arguments of a floor row: the real scene image(s) it names, else the task's synthetic bench_inputs."""
    from fastforks._core.bench import BenchSize, bench_inputs  # noqa: PLC0415

    h, w = (int(v) for v in inp["label"].split("x"))
    if ref in SWAPPED:
        h, w = w, h
    args, kw = bench_inputs(model, BenchSize(h, w, VIEWS.get(ref, 8)))
    scene = next((p for k, p in SCENES.items() if inp["scene"].startswith(k)), None)
    if scene is not None and scene.exists():  # else the synthetic bench inputs (no local datasets)
        left = _image(scene)
        if task == "flow":
            args = (left, _image(Path(str(scene).replace("_10.png", "_11.png"))))
        else:
            right = Path(str(scene).replace("image_2", "image_3").replace("im0", "im1"))
            args = (left, _image(right)) if task == "stereo" else (left,)
    return args, kw


def _sync() -> None:
    import torch  # noqa: PLC0415

    torch.cuda.synchronize()


def measure(call: Callable[[], Any]) -> dict[str, Any]:
    """First call, WARMUP calls, peak reset, ROUNDS x RUNS timed calls."""
    import torch  # noqa: PLC0415

    _sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    call()
    _sync()
    first = time.perf_counter() - t0
    pf = torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20
    for _ in range(WARMUP):
        call()
    _sync()
    torch.cuda.reset_peak_memory_stats()
    rounds, allt = [], []
    for _ in range(ROUNDS):
        ts = []
        for _ in range(RUNS):
            _sync()
            t0 = time.perf_counter()
            call()
            _sync()
            ts.append(1e3 * (time.perf_counter() - t0))
        rounds.append(statistics.median(ts))
        allt += ts
    ms = statistics.median(rounds)
    return {
        "ms": round(ms, 3),
        "ms_rounds": [round(t, 3) for t in rounds],
        "noise": round((max(rounds) - min(rounds)) / ms, 4),
        "ms_min": round(min(allt), 3),
        "runs": len(allt),
        "first_call_s": round(first, 3),
        "peak_first_alloc_mb": round(pf[0]),
        "peak_first_reserved_mb": round(pf[1]),
        "peak_alloc_mb": round(max(pf[0], torch.cuda.max_memory_allocated() / 2**20)),
        "peak_reserved_mb": round(max(pf[1], torch.cuda.max_memory_reserved() / 2**20)),
    }


def core_ms(call: Callable[[], Any]) -> float:
    ts = []
    for _ in range(WARMUP + RUNS):
        _sync()
        t0 = time.perf_counter()
        call()
        _sync()
        ts.append(1e3 * (time.perf_counter() - t0))
    return round(statistics.median(ts[WARMUP:]), 3)


def profile(call: Callable[[], Any], n: int = 3) -> dict[str, Any]:
    """CUDA time shares of `n` calls: copies / casts / layout, other elementwise, memset; host syncs per call."""
    import torch  # noqa: PLC0415
    from torch.profiler import ProfilerActivity  # noqa: PLC0415
    from torch.profiler import profile as tprofile  # noqa: PLC0415

    _sync()
    with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            call()
        _sync()
    kern: dict[str, float] = {}
    syncs = 0
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kern[e.key] = kern.get(e.key, 0.0) + e.self_device_time_total
        elif any(e.key.startswith(s) for s in SYNC):
            syncs += e.count
    total = sum(kern.values()) or 1.0
    share = {"copy": 0.0, "elementwise": 0.0, "memset": 0.0}
    for k, t in kern.items():
        low = k.lower()
        if any(c in low for c in COPY):
            share["copy"] += t
        elif "elementwise" in low or "reduce_kernel" in low:
            share["elementwise"] += t
        elif "memset" in low or "fill" in low:
            share["memset"] += t
    top = sorted(kern.items(), key=lambda kv: -kv[1])[:8]
    return {
        "cuda_ms_per_call": round(total / 1e3 / n, 3),
        "share": {k: round(v / total, 3) for k, v in share.items()},
        "host_syncs_per_call": round(syncs / n - 1 / n, 2),  # minus the profiler block's own synchronize
        "top": [[k[:110], round(t / total, 3)] for k, t in top],
    }


class _Capture:
    """Stands in for a model's task runner: records the first runner call (`__call__` or one of `METHODS`)."""

    METHODS = frozenset({"labels", "embed", "predict", "segment_concept"})

    def __init__(self, runner: Any) -> None:  # noqa: ANN401
        self.__dict__["_runner"], self.__dict__["_args"] = runner, None

    def _record(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kw: Any) -> Any:  # noqa: ANN401
            if self.__dict__["_args"] is None:
                self.__dict__["_args"] = (fn, args, kw)
            return fn(*args, **kw)

        return call

    def __call__(self, *args: Any, **kw: Any) -> Any:  # noqa: ANN401
        return self._record(self.__dict__["_runner"])(*args, **kw)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        value = getattr(self.__dict__["_runner"], name)
        return self._record(value) if name in self.METHODS else value

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        setattr(self.__dict__["_runner"], name, value)


def _src_files() -> list[str]:
    src = ROOT / "src"
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and Path(f).resolve().is_relative_to(src):
            out.add(str(Path(f).resolve().relative_to(ROOT)))
    return sorted(out)


def _lib_job(
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, Callable[[], Any], Callable[[], Any] | None]]]:
    import torch  # noqa: PLC0415

    t0 = time.perf_counter()
    task = importlib.import_module(f"fastforks.{spec['task']}")
    t1 = time.perf_counter()
    m = task.load(spec["ref"], mode="portable" if spec["mode"] in NEW_MODES else spec["mode"], device="cuda")
    if spec["mode"] == "compile":
        m.compile()  # the first call compiles (cold: the run gave this process empty caches)
    _sync()
    info = {"import_s": round(t1 - t0, 2), "load_s": round(time.perf_counter() - t1, 2)}
    info["backend"] = {"mode": m.backend.mode, "dtype": str(m.backend.dtype), "kernels": list(m.backend.kernels)}
    calls = []
    runner = m._runner  # noqa: SLF001 - the task's own runner contract
    for inp in spec["inputs"]:
        args, kw = lib_inputs(m, spec["task"], spec["ref"], inp)
        cap = _Capture(runner)

        def full(a: tuple[Any, ...] = args, k: dict[str, Any] = kw) -> Any:  # noqa: ANN401
            with torch.inference_mode():
                return m(*a, **k)

        def core(*, release: bool = False, c: _Capture = cap, a: tuple[Any, ...] = args, k: dict[str, Any] = kw) -> Any:  # noqa: ANN401
            # The runner's tensors are recorded here, after the peaks were measured, and dropped again after the
            # core timing: holding them during the full-call rows added two padded fp32 images to every peak.
            if release:
                c.__dict__["_args"] = None
                return None
            if c.__dict__["_args"] is None:
                m._runner = c  # noqa: SLF001 - records the runner's tensors of this input
                try:
                    full(a, k)
                finally:
                    m._runner = runner  # noqa: SLF001
            fn, ca, ck = c.__dict__["_args"]
            with torch.inference_mode():
                return fn(*ca, **ck)

        calls.append((inp["label"], full, core))
    info["_model"] = m
    return info, calls


def speed_step(model: Any, spec: dict[str, Any], common: dict[str, Any], calls: list[Any], out: Path) -> None:  # noqa: ANN401, C901
    """Timing, peak memory and (on one input) a profile of every input of a loaded model: one row per input.

    The profile runs after the last timed call of the process (a torch profile slows the later calls of its process:
    hrnet kernels 1024x2048 11.8-11.9 ms after a profile vs 11.1 without), so its row is written last.
    """
    import torch  # noqa: PLC0415

    del model  # the calls hold it
    prof_label = PROFILE_INPUT.get(spec["task"], calls[0][0] if calls else "")
    held: tuple[dict[str, Any], Callable[[], Any]] | None = None
    for n, (label, full, core) in enumerate(calls):
        scene = next((i["scene"] for i in spec["inputs"] if i["label"] == label), "")
        real = any(scene.startswith(k) and p.exists() for k, p in SCENES.items())
        row = {**common, "input": label, "scene": scene if real else "synthetic", "first_call_cold": n == 0}
        try:
            if spec["side"] == "lib":
                from fastforks._core import kernels  # noqa: PLC0415

                kernels.reset_fallbacks()
            row |= measure(full)
            if core is not None:
                row["core_ms"] = core_ms(core)
                core(release=True)
            if spec["side"] == "lib" and (fb := kernels.fallbacks()):
                row["fallbacks"] = {f"{a}:{b}": v for (a, b), v in fb.items()}
            if label == prof_label and not spec.get("no_profile"):
                held = row, full
        except Exception as e:  # noqa: BLE001 - recorded as an error row; larger inputs still run
            _fail(row, spec, f"run: {type(e).__name__}: {e}")
        if spec["mode"] == "compile" and "first_call_s" in row:
            row["compile_s"] = row["first_call_s"]
        if held is None or held[0] is not row:
            _append(out, row)
        gc.collect()
        torch.cuda.empty_cache()
    if held is not None:
        row, full = held
        try:
            row["profile"] = profile(full)
        except Exception as e:  # noqa: BLE001 - the timing stays valid
            row["profile"] = {"error": f"{type(e).__name__}: {e}"[:600]}
        _append(out, row)


def _fail(row: dict[str, Any], spec: dict[str, Any], msg: str) -> None:
    """Record a failure: `fails: <short reason>` for compile / tensorrt rows, else the long `error`."""
    if spec["mode"] in NEW_MODES:
        row["fails"] = " ".join(msg.split())[:160]
    else:
        row["error"] = msg[:600]


def _net_input(model: Any, task: str, full: Callable[[], Any]) -> tuple[int, int, int]:  # noqa: ANN401
    """(batch or views, h, w) of the image the network takes in one timed call (`module()` takes the same).

    The runner's first image argument (classification: the crop, multi-view: `[views, 3, H, W]`); detection, whose
    `module()` starts after the resize, and pose, whose `module()` takes the box crops: the first 3-channel image a
    submodule gets, if one does (R-CNN: the runner's).
    """
    import torch  # noqa: PLC0415

    seen: list[torch.Size] = []

    def hook(_mod: Any, args: tuple[Any, ...]) -> None:  # noqa: ANN401
        x = args[0] if args else None
        if not seen and isinstance(x, torch.Tensor) and x.ndim == 4 and x.shape[1] == 3:  # noqa: PLR2004
            seen.append(x.shape)

    runner = model._runner  # noqa: SLF001 - the task's own runner contract
    cap = model._runner = _Capture(runner)  # noqa: SLF001
    handle = torch.nn.modules.module.register_module_forward_pre_hook(hook) if task in {"detection", "pose"} else None
    try:
        full()
    finally:
        model._runner = runner  # noqa: SLF001
        if handle is not None:
            handle.remove()
    seen += [a.shape for a in cap.__dict__["_args"][1] if isinstance(a, torch.Tensor)][:1]
    n, _, h, w = seen[0]
    return n, h, w


def trt_step(model: Any, spec: dict[str, Any], common: dict[str, Any], calls: list[Any], out: Path) -> None:  # noqa: ANN401
    """`tensorrt` mode: ONNX export and fp16 engine build of the portable model at the input, then the engine's timing.

    The engine runs `module()` on synthetic inputs (`example_inputs`) of the shape the timed call gives the network
    (`_net_input`: the classifier crop, the scene's views). `device_mb` = device memory taken by loading the engine and
    running it (TensorRT allocates outside torch); `peak_*_mb` = `device_mb` + the torch-side peak (kept in
    `torch_peak_*_mb`).
    """
    import tempfile  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from fastforks._core.export import TensorRTEngine, all_nan, build_tensorrt  # noqa: PLC0415

    label = spec["inputs"][0]["label"]
    row = {**common, "input": label, "scene": "synthetic", "first_call_cold": True}
    try:
        n, h, w = _net_input(model, spec["task"], calls[0][1])
        if f"{h}x{w}" != label:
            row["net_input"] = f"{h}x{w}"
        if n > 1:
            row["views"] = n
        inputs = model.example_inputs(h, w, batch=n)
        with tempfile.TemporaryDirectory(prefix="ff-trt-") as tmp:
            t0 = time.perf_counter()
            onnx = model.export_onnx(Path(tmp) / "model.onnx", h, w, batch=n)
            row["onnx_export_s"] = round(time.perf_counter() - t0, 1)
            t0 = time.perf_counter()
            engine_path = build_tensorrt(onnx, Path(tmp) / "model.engine")
            row["engine_build_s"] = round(time.perf_counter() - t0, 1)
            row["engine_mb"] = round(engine_path.stat().st_size / 2**20)
            gc.collect()
            torch.cuda.empty_cache()
            free0 = torch.cuda.mem_get_info()[0]
            engine = TensorRTEngine(engine_path)
            row |= measure(lambda: engine(*inputs))
            row["device_mb"] = round((free0 - torch.cuda.mem_get_info()[0]) / 2**20)
            # peak = engine memory + torch-side peak (the allocator does not see the engine's memory)
            for k in ("alloc", "reserved"):
                row[f"torch_peak_{k}_mb"] = row[f"peak_{k}_mb"]
                row[f"peak_{k}_mb"] += row["device_mb"]
            if all_nan(engine.run(*inputs)) is not None:
                _fail(row, spec, "wrong output: an engine output is all NaN")
    except Exception as e:  # noqa: BLE001 - recorded as a failed row
        _fail(row, spec, f"{type(e).__name__}: {e}")
    _append(out, row)


def _short(name: str) -> str:
    """A kernel name without its template / argument lists (same kernel family across shapes and builds)."""
    name = name.removeprefix("void ").replace("(anonymous namespace)::", "")
    cut = min((i for i in (name.find("<"), name.find("(")) if i > 0), default=len(name))
    return name[:cut][:120]


def _top_op(e: Any) -> str:  # noqa: ANN401 - a profiler FunctionEvent
    """`outermost/innermost` op of a kernel's launching CPU op (e.g. `aten::linear/aten::addmm`)."""
    top = e
    while top.cpu_parent is not None:
        top = top.cpu_parent
    return e.name if top is e else f"{top.name}/{e.name}"


def trace(call: Callable[[], Any], n: int = 3) -> dict[str, Any]:
    """Warm wall time, summed CUDA kernel time and its groups (by kernel family, by launching op) of one call.

    Mean over `n` profiled calls; kernels without a launching op (CUDA graph replays, ctypes / driver launches)
    are grouped under the op `-`. `gap_ms` = wall - kernel time: launch, Python and host-sync overhead (negative
    when kernels overlap).
    """
    import torch  # noqa: PLC0415
    from torch.profiler import ProfilerActivity  # noqa: PLC0415
    from torch.profiler import profile as tprofile  # noqa: PLC0415

    for _ in range(WARMUP):
        call()
    ts = []
    for _ in range(RUNS):
        _sync()
        t0 = time.perf_counter()
        call()
        _sync()
        ts.append(1e3 * (time.perf_counter() - t0))
    with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            call()
        _sync()
    kern: dict[str, list[float]] = {}
    ops: dict[str, list[float]] = {}
    total = attributed = 0.0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            g = kern.setdefault(_short(e.name), [0.0, 0])
            us = e.time_range.end - e.time_range.start
            g[0] += us
            g[1] += 1
            total += us
        elif e.kernels:
            g = ops.setdefault(_top_op(e), [0.0, 0])
            for k in e.kernels:
                g[0] += k.duration
                g[1] += 1
                attributed += k.duration
    if total - attributed > 0:
        ops["-"] = [total - attributed, -1]  # count unknown
    wall = statistics.median(ts)

    def per_call(d: dict[str, list[float]]) -> dict[str, list[float]]:
        return {
            k: [round(v[0] / 1e3 / n, 4), round(v[1] / n, 1)] for k, v in sorted(d.items(), key=lambda kv: -kv[1][0])
        }

    return {
        "wall_ms": round(wall, 3),
        "kernel_ms": round(total / 1e3 / n, 3),
        "gap_ms": round(wall - total / 1e3 / n, 3),
        "launches": round(sum(v[1] for v in kern.values()) / n, 1),
        "kernels": per_call(kern),
        "ops": per_call(ops),
    }


def trace_step(model: Any, spec: dict[str, Any], common: dict[str, Any], calls: list[Any], out: Path) -> None:  # noqa: ANN401
    """`run --trace-diff`: one row per call kind (full; lib also core) with the kernel groups of `trace`."""
    import torch  # noqa: PLC0415

    del model, spec  # the calls hold the model
    for label, full, core in calls:
        for which, call in (("full", full), ("core", core)):
            if call is None:
                continue
            row = {**common, "input": label, "which": which}
            try:
                call()  # the first call (core: replays the tensors the full call captured)
                row |= trace(call)
            except Exception as e:  # noqa: BLE001 - recorded as an error row
                row["error"] = f"trace: {type(e).__name__}: {e}"[:600]
            _append(out, row)
        gc.collect()
        torch.cuda.empty_cache()


MODEL_STEPS: list[Callable[[Any, dict[str, Any], dict[str, Any], list[Any], Path], None]] = [speed_step]
"""Per-model steps of a worker, run on the already loaded model (e.g. a golden output check can be appended)."""


def worker(spec: dict[str, Any]) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    t0 = time.perf_counter()
    import torch  # noqa: PLC0415

    torch_import_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    torch.manual_seed(0)
    torch.zeros(1, device="cuda").sum().item()
    cuda_init_s = time.perf_counter() - t0
    gc.collect()
    base = torch.cuda.memory_allocated() / 2**20
    common = {
        "side": spec["side"],
        "ref": spec["ref"],
        "task": spec["task"],
        "mode": spec["mode"],
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "date": time.strftime("%F"),
        "protocol": f"first call, {WARMUP} warm-up, peak reset, median of {ROUNDS} rounds x median of {RUNS}",
        "cold": True,  # one fresh process per (ref, side, mode): the load is the first in its process
        "torch_import_s": round(torch_import_s, 2),
        "cuda_init_s": round(cuda_init_s, 2),
    }
    out = Path(spec["out"])
    t0 = time.perf_counter()
    try:
        if spec["side"] == "lib":
            info, calls = _lib_job(spec)
            common["commit"] = spec["commit"]
        else:
            plugin = _plugin(spec["forks"])
            info, fcalls = plugin.fork_job(spec, spec["task"], [i["label"] for i in spec["inputs"]])
            calls = [(lab, c, None) for lab, c in fcalls]
            common |= {"commit": info.pop("commit", "fork"), "fork_key": spec["fork_key"]}
            info["load_s"] = round(time.perf_counter() - t0, 2)  # the fork's own imports included
    except Exception as e:  # noqa: BLE001 - recorded as an error row
        row = {**common, "input": spec["inputs"][0]["label"] if spec["mode"] in NEW_MODES else "-"}
        _fail(row, spec, f"load: {type(e).__name__}: {e}")
        _append(out, row)
        return 0
    model = info.pop("_model", None)
    if spec["side"] == "lib":
        files = _src_files()
        k = json.dumps(info["backend"]["kernels"]) + torch.__version__
        common |= {"code_hash": files_hash(files) + hashlib.sha256(k.encode()).hexdigest()[:8], "files": files}
    common |= {**info, "base_alloc_mb": round(base)}
    steps = [trace_step] if spec.get("trace") else [trt_step] if spec["mode"] == "tensorrt" else MODEL_STEPS
    for step in steps:
        step(model, spec, common, calls, out)
    del model, calls
    return 0


def _append(out: Path, row: dict[str, Any]) -> None:
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(
        json.dumps({k: v for k, v in row.items() if k not in ("files", "profile", "kernels", "ops")})[:300], flush=True
    )


def _plugin(path: str) -> Any:  # noqa: ANN401 - a module
    s = importlib.util.spec_from_file_location("speed_check_forks", path)
    if s is None or s.loader is None:
        msg = f"cannot import {path}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


# Orchestration: no torch.
def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _done(rows: list[dict[str, Any]], side: str, job: dict[str, Any], fork_key: str = "") -> bool:
    """All inputs of the job measured on the current code (lib) / with the same fork key (fork)."""
    got = set()
    for r in rows:
        if r.get("side") != side or r["ref"] != job["ref"] or r["mode"] != job["mode"] or "error" in r:
            continue
        if side == "lib" and r.get("code_hash", "")[:16] != files_hash(r.get("files", [])):
            continue
        if side != "lib" and r.get("fork_key") != fork_key:
            continue
        got.add(r["input"])
    return {i["label"] for i in job["inputs"]} <= got


def _changed(rows: list[dict[str, Any]], job: dict[str, Any], since: str) -> bool:
    diff = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", since, "--", "src"], capture_output=True, text=True, check=True
    )
    changed = set(diff.stdout.split())
    files = next(
        (r["files"] for r in rows if r.get("side") == "lib" and r["ref"] == job["ref"] and r.get("files")), None
    )
    return files is None or bool(changed & set(files))


def _launch(a: argparse.Namespace, spec: dict[str, Any], cmd: list[str], env: dict[str, str]) -> None:
    """One worker of `spec`; a crash or timeout without a row becomes an error row (`fails` for compile / tensorrt)."""
    cache = tempfile.mkdtemp(prefix="ff-compile-cache-") if spec["mode"] == "compile" else ""
    if cache:  # empty Inductor / Triton caches: the first call is a cold compile
        env |= {"TORCHINDUCTOR_CACHE_DIR": f"{cache}/inductor", "TRITON_CACHE_DIR": f"{cache}/triton"}
    before = len(read_rows(a.out))
    try:
        rc = subprocess.run(cmd, env=env, timeout=a.timeout, check=False).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    finally:
        if cache:
            shutil.rmtree(cache, ignore_errors=True)
    if rc == 0 or len(read_rows(a.out)) != before:
        return
    row = {
        "side": spec["side"],
        "ref": spec["ref"],
        "task": spec["task"],
        "mode": spec["mode"],
        "date": time.strftime("%F"),
    }
    if spec["mode"] in NEW_MODES:
        why = f"timeout after {a.timeout:g} s" if rc == "timeout" else f"worker rc={rc}"
        row |= {"input": spec["inputs"][0]["label"], "fails": why}
    else:
        row |= {"input": "-", "fork_key": spec["fork_key"], "error": f"worker rc={rc}"}
    _append(a.out, row)


def run(a: argparse.Namespace) -> int:
    rows = read_rows(a.out)
    todo = jobs(a.only, a.modes)
    if a.changed_since:
        todo = [j for j in todo if _changed(rows, j, a.changed_since)]
    forks = _plugin(a.forks).FORKS if a.forks else {}
    commit = _commit()
    prefix = shlex.split(a.prefix)
    for n, job in enumerate(todo):
        if a.trace_diff:  # the profile input of the task (else the smallest)
            pick = PROFILE_INPUT.get(job["task"], "")
            job["inputs"] = [i for i in job["inputs"] if i["label"] == pick] or job["inputs"][:1]
        sides = ["lib"] + (["fork"] if (job["ref"], job["mode"]) in forks and "fork" in a.sides else [])
        sides = [s for s in sides if s in a.sides]
        if n % 2:
            sides.reverse()
        for side in sides:
            fk = forks.get((job["ref"], job["mode"]), {})
            key = str(fk.get("key", ""))
            if not a.force and _done(rows, side, job, key):
                continue
            spec = {
                **job,
                "side": side,
                "out": str(a.out),
                "commit": commit,
                "forks": a.forks,
                "no_profile": a.no_profile or job["mode"] in NEW_MODES,
                "trace": a.trace_diff,
            }
            spec["fork_key"] = key
            env = {**os.environ, **(fk.get("env", {}) if side == "fork" else {})}
            py = fk["python"] if side == "fork" else sys.executable
            if side == "lib":
                env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
            print(f"== {side} {job['ref']} {job['mode']} {time.strftime('%T')}", flush=True)
            _launch(a, spec, [*prefix, py, str(Path(__file__).resolve()), "worker", json.dumps(spec)], env)
            rows = read_rows(a.out)
    return check(a.out) if a.check else 0


def _floor_index() -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(r["ref"], r["mode"], r["input"]): r for r in floor_rows() if r["mode"] in MODES}


def latest(rows: list[dict[str, Any]], side: str = "lib") -> dict[tuple[str, str, str], dict[str, Any]]:
    out = {}
    for r in rows:
        if r.get("side") == side and "error" not in r:
            out[r["ref"], r["mode"], r["input"]] = r
    return out


def check(path: Path) -> int:
    floor, bad = _floor_index(), 0
    for key, r in sorted(latest(read_rows(path)).items()):
        f = floor.get(key)
        if f is None:
            continue
        tol = 1 + max(TOL_MS, r.get("noise", 0))
        msgs = []
        if r["ms"] > f["ms"] * tol:
            msgs.append(f"ms {r['ms']:.2f} > floor {f['ms']:.2f} x {tol:.3f}")
        msgs += [
            f"{k} {r[k]} > floor {f[k]}"
            for k in ("peak_alloc_mb", "peak_reserved_mb")
            if f.get(k) and r.get(k, 0) > max(f[k] * (1 + TOL_MB), f[k] + TOL_MB_ABS)
        ]
        if msgs:
            bad += 1
            print("REGRESSION", *key, "; ".join(msgs))
    errors = [r for r in read_rows(path) if r.get("side") == "lib" and "error" in r]
    for r in errors:
        print("ERROR", r["ref"], r["mode"], r.get("input"), r["error"][:160])
    print(f"{bad} regressions, {len(errors)} error rows")
    return 1 if bad else 0


def static_violations() -> list[tuple[str, str]]:
    """(key, message) of floor rows slower than a mode they must not lose to (same ref and input)."""
    by: dict[tuple[str, str], dict[str, float]] = {}
    for r in floor_rows():
        if "ms" in r:  # an OOM original has no time
            by.setdefault((r["ref"], r["input"]), {})[r["mode"]] = r["ms"]
    out = []
    for (ref, inp), ms in sorted(by.items()):
        origs = [m for m in ("upstream", "upstream-default") if m in ms]
        orig = min(origs, key=ms.__getitem__) if origs else ""
        pairs = [("kernels", "portable")] + [(m, orig) for m in MODES if orig]
        for slow, ref_mode in pairs:
            if slow in ms and ref_mode in ms and ms[slow] > ms[ref_mode] * (1 + TOL_STATIC):
                key = f"{ref} {inp} {slow} vs {ref_mode}"
                out.append((key, f"{ms[slow]:.2f} ms vs {ms[ref_mode]:.2f} ms ({ms[slow] / ms[ref_mode] - 1:+.1%})"))
    return out


def static() -> int:
    allow = json.loads(ALLOW.read_text()) if ALLOW.exists() else {}
    bad = [(k, m) for k, m in static_violations() if k not in allow]
    for k, m in bad:
        print("VIOLATION", k, m)
    stale = sorted(set(allow) - {k for k, _ in static_violations()})
    for k in stale:
        print("STALE allowlist entry (no longer a violation):", k)
    print(f"{len(bad)} violations not in {ALLOW.name}, {len(stale)} stale entries")
    return 1 if bad or stale else 0


FLOOR_KEYS = (
    "ref", "task", "mode", "input", "scene", "ms", "ms_min", "runs", "noise", "peak_alloc_mb", "peak_reserved_mb",
    "peak_first_alloc_mb", "peak_first_reserved_mb", "import_s", "load_s", "first_call_s", "cold", "backend",
    "protocol", "gpu", "torch", "date", "commit",
)  # fmt: skip


def export(path: Path, source: str, command: str) -> int:
    """The latest library rows of `path` as floor rows (docs/data/speed.jsonl fields), JSON lines on stdout."""
    for r in latest(read_rows(path)).values():
        print(json.dumps({**{k: r[k] for k in FLOOR_KEYS if k in r}, "source": source, "command": command}))
    return 0


def report(path: Path) -> int:
    rows = read_rows(path)
    lib, fork = latest(rows, "lib"), latest(rows, "fork")
    print(
        "| ref | mode | input | fork ms | fork MB alloc/res | library ms (core) | library MB | ratio | noise |", end=""
    )
    print(" verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(set(lib) & set(fork)):
        a, b = lib[key], fork[key]
        cmp_ms = a["core_ms"] if b.get("compare") == "core" and "core_ms" in a else a["ms"]
        ratio = cmp_ms / b["ms"]
        band = max(a.get("noise", 0), b.get("noise", 0))
        verdict = "slower" if ratio > 1 + TOL_STATIC + band else "ok"
        mem = a["peak_alloc_mb"] > b["peak_alloc_mb"] * 1.05 + TOL_MB_ABS
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {b['ms']:.2f} | {b['peak_alloc_mb']}/{b['peak_reserved_mb']} | "
            f"{a['ms']:.2f} ({a.get('core_ms', '-')}) | {a['peak_alloc_mb']}/{a['peak_reserved_mb']} | {ratio:.3f} | "
            f"{band:.3f} | {verdict}{' +mem' if mem else ''} |"
        )
    return 0


TRACE_REL, TRACE_ABS = 0.10, 0.05  # a group is flagged when the library spends > 10 % and > 0.05 ms more than the fork


KERNEL_CLASSES = (  # first match wins: substring of the lower-case kernel family name
    ("attention", ("flash", "fmha", "attention", "sdpa")),
    ("conv", ("conv", "fprop", "dgrad", "wgrad", "implicit", "winograd", "nhwckrsc", "cudnn")),
    ("gemm", ("gemm", "gemv", "cutlass", "cublas", "matmul", "splitk", "xmma")),
    ("layout / copy", COPY),
    ("norm", ("norm",)),
    ("softmax", ("softmax",)),
    ("index / scatter", ("gather", "scatter", "index")),
    ("reduce", ("reduce",)),
    ("elementwise", ("elementwise",)),
    ("memset", ("memset", "fill")),
)


def _groups(r: dict[str, Any], level: str) -> dict[str, list[float]]:
    """Kernel time groups of a trace row by `class`, `kernel` (family name) or `op` (launching op).

    Classes (KERNEL_CLASSES) stay comparable when one kernel replaces another of its kind (a fused GEMM for cuBLAS).
    """
    if level != "class":
        return r["kernels" if level == "kernel" else "ops"]
    out: dict[str, list[float]] = {}
    for name, (ms, n) in r["kernels"].items():
        low = name.lower()
        c = next((c for c, keys in KERNEL_CLASSES if any(k in low for k in keys)), "other")
        g = out.setdefault(c, [0.0, 0])
        g[0], g[1] = round(g[0] + ms, 4), g[1] + n
    return out


def trace_diff(path: Path, top: int = 20) -> int:
    """Per model: wall / kernel / gap time of both `run --trace-diff` sides and the groups the library loses.

    Groups where the library spends more than the fork, also when it is faster in total: kernel classes (all), kernel
    families and launching ops present on both sides (a family only one side has is a replacement: see its class).
    The library side is its full call, or its runner alone where the fork row says `compare: core`.
    """
    rows = [r for r in read_rows(path) if "error" not in r and "kernels" in r]
    fork = {(r["ref"], r["mode"], r["input"]): r for r in rows if r["side"] == "fork"}
    lib = {(r["ref"], r["mode"], r["input"], r["which"]): r for r in rows if r["side"] == "lib"}
    flags = []
    print(
        "| ref | mode | input | wall lib / fork ms | kernels lib / fork ms | gap lib / fork ms | launches lib / fork |",
        end="",
    )
    print(" flagged groups (extra ms) |")
    print("|---|---|---|---|---|---|---|---|")
    for key, b in sorted(fork.items()):
        a = lib.get((*key, b.get("compare", "full"))) or lib.get((*key, "full"))
        if a is None:
            continue
        got = []
        for level in ("class", "kernel", "op"):
            ga, gb = _groups(a, level), _groups(b, level)
            names = set(ga) | set(gb) if level == "class" else (set(ga) & set(gb)) - {"-"}
            for g in names:
                la, lb = ga.get(g, [0, 0]), gb.get(g, [0, 0])
                extra = la[0] - lb[0]
                if extra > TRACE_ABS and la[0] > lb[0] * (1 + TRACE_REL):
                    got.append(
                        {
                            "ref": key[0],
                            "mode": key[1],
                            "input": key[2],
                            "level": level,
                            "group": g,
                            "lib_ms": la[0],
                            "fork_ms": lb[0],
                            "lib_n": la[1],
                            "fork_n": lb[1],
                            "extra_ms": round(extra, 3),
                            "total_ratio": round(a["wall_ms"] / b["wall_ms"], 3),
                        }
                    )
        got.sort(key=lambda f: -f["extra_ms"])
        flags += got
        short = ", ".join(f"{f['level'][0]}:{f['group'][:40]} +{f['extra_ms']:.2f}" for f in got[:3])
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {a['wall_ms']:.2f} / {b['wall_ms']:.2f} | {a['kernel_ms']:.2f} / "
            f"{b['kernel_ms']:.2f} | {a['gap_ms']:.2f} / {b['gap_ms']:.2f} | {a['launches']:.0f} / "
            f"{b['launches']:.0f} | {len(got)}{': ' + short if short else ''} |"
        )
    flags.sort(key=lambda f: -f["extra_ms"])
    print(f"\nTop {top} flagged groups (library > fork by > {TRACE_REL:.0%} and > {TRACE_ABS} ms per call):\n")
    print("| ref | mode | input | level | group | lib ms (n) | fork ms (n) | extra ms | total lib/fork |")
    print("|---|---|---|---|---|---|---|---|---|")
    for f in flags[:top]:
        print(
            f"| {f['ref']} | {f['mode']} | {f['input']} | {f['level']} | `{f['group'][:80]}` | {f['lib_ms']:.3f} "
            f"({f['lib_n']:g}) | {f['fork_ms']:.3f} ({f['fork_n']:g}) | {f['extra_ms']:.3f} | {f['total_ratio']:.3f} |"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--modes", nargs="*", default=list(MODES), choices=[*MODES, *NEW_MODES])
    p = sub.add_parser("run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--modes", nargs="*", default=list(MODES), choices=[*MODES, *NEW_MODES])
    p.add_argument("--sides", nargs="*", default=["lib", "fork"])
    p.add_argument("--changed-since", default="")
    p.add_argument("--prefix", default="", help="command prefix of each worker, e.g. a GPU lock with a timeout")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--forks", default="")
    p.add_argument("--force", action="store_true", help="measure again even if the code did not change")
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--trace-diff", action="store_true", help="kernel-group trace rows (one input) instead of timings")
    p.add_argument("--check", action="store_true")
    for name in ("check", "report"):
        sub.add_parser(name).add_argument("rows", type=Path)
    p = sub.add_parser("trace-diff")
    p.add_argument("rows", type=Path)
    p.add_argument("--top", type=int, default=20)
    p = sub.add_parser("export")
    p.add_argument("rows", type=Path)
    p.add_argument("--source", required=True, help="the results file the rows are documented in")
    p.add_argument("--command", default="", help="the command that measured them")
    sub.add_parser("static")
    sub.add_parser("worker").add_argument("spec")
    a = ap.parse_args()
    if a.cmd == "list":
        for j in jobs(a.only, a.modes):
            print(j["task"], j["ref"], j["mode"], " ".join(i["label"] for i in j["inputs"]))
        return 0
    if a.cmd == "worker":
        return worker(json.loads(a.spec))
    with contextlib.suppress(BrokenPipeError):
        return {
            "run": lambda: run(a),
            "check": lambda: check(a.rows),
            "report": lambda: report(a.rows),
            "static": static,
            "trace-diff": lambda: trace_diff(a.rows, a.top),
            "export": lambda: export(a.rows, a.source, a.command),
        }[a.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
