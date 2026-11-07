"""Speed survey of a non-reference device (docs/data/devices.toml): rows for docs/data/speed/<slug>.jsonl.

    python tools/speed_survey.py detect                                  # this machine's GPU as a devices.toml entry
    python tools/speed_survey.py list --device SLUG [--subset] [--only REF ...]       # the jobs, no torch
    python tools/speed_survey.py prepare --device SLUG [--subset]         # download weights + kernels, then check
    python tools/speed_survey.py run --device SLUG [--subset] [--only REF ...] [--modes portable kernels] [--force]

A survey is not a regression floor (docs/data/README.md, "Devices"): nothing here fails on a slow row. The models are
the refs of the reference floor (docs/data/speed.jsonl), each at its task's standard input (`speed_check.STANDARD`,
the inputs of the across-devices tables; multi-view: the floor's smallest) on the floor's scene where it exists
locally, so every survey row has a reference row of the same input beside it. Job order: first the cross-device
subset (`*` in `list`: one variant per family, the card's first variant the floor has), then the other variants;
`--subset` stops after the first part. Per ref, mode and dtype the card validates on the device's backend:

- cuda: portable in `portable_cuda_dtype`; kernels in `kernels_dtype` if the card has CUDA kernel libraries.
- rocm: as cuda for portable (torch.cuda over HIP); kernels `unsupported` (the kernel libraries are CUDA builds).
- mps: portable in every dtype of `portable_mps_dtypes` (none: one `unsupported` row); kernels in the card's MPS
  kernels dtype if it has Metal kernel libraries.

A mode or dtype the card does not offer becomes an `unsupported` row, never a measurement. Each (ref, mode, dtype) runs
in a fresh process; a crash or timeout becomes a `failed` row, out of memory an `oom` row.

- cuda / rocm: the worker is tools/merge_check.py's (tools/speed_check.py's worker plus the golden step), so timing,
  memory (`torch-peak`) and startup fields are exactly those of the reference floor: first call, 3 warm-up calls,
  peak reset, median of 3 rounds x median of 5 synchronised calls.
- mps: first call, MPS_WARMUP warm-up calls, median of MPS_RUNS synchronised calls; memory `process-footprint` (the
  process's lifetime peak physical footprint, `proc_pid_rusage`: model + activations in unified memory, what Activity
  Monitor shows as Memory). The MPS allocator is capped at the device's recommended working set
  (PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0, its default 1.7 lets a model allocate past physical memory into swap): a model
  that needs more is an `oom` row. Before each job a run waits (up to SWAP_WAIT_S) for MIN_FREE_GB free on the system
  disk (swap space), else stops.

After the timing, the same loaded model runs its golden check (tests/test_golden.py::test_gpu): `golden` = `pass`,
`fail` or `skip` (needs FASTFORKS_GOLDEN_PATH), so a row says whether the mode computes the right result on this
device, not only how fast. `run` sets FASTFORKS_OFFLINE=1 (a missing weight or kernel file is a failed row, never a
download on rented time; `prepare` first, or `--online`) and records the driver, the CUDA / HIP version, the power
limit and the clocks. Rows are merged into the device's file: a row replaces the earlier row of the same ref / mode /
dtype / input; finished rows are skipped unless `--force`. The tool refuses to run on a machine whose GPU is not the
slug's `match`.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import importlib.metadata
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import _devices
from speed_check import SCENES, STANDARD, SWAPPED, VIEWS, _commit, _px, floor_rows

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
MODES = ("portable", "kernels")
MPS_WARMUP, MPS_RUNS = 4, 5  # 4: kernels builds plans on the 1st call of a size and graphs on the 4th (bench.WARMUP)
MPS_PROTOCOL = f"first call, {MPS_WARMUP} warm-up, median of {MPS_RUNS} synchronised calls (tools/speed_survey.py)"
SWAP_WAIT_S = 1800
# free system disk (GiB) before an MPS run: macOS swaps there; with ~5 GB free a big model froze the Mac
MIN_FREE_GB = float(os.environ.get("FASTFORKS_MIN_FREE_GB", "15"))
TIMEOUT_S = 1800
GOLDEN = "golden"  # merge_check's `input` of a golden row
_DROP = ("side", "files", "cold", "first_call_cold", "fork_key", "golden_ref", "input_kind", "fails")
_ENV_KEYS = (
    "gpu", "gpu_cores", "arch", "os", "torch", "cuda", "hip", "driver", "power_limit", "clocks", "throttle_reasons"
)  # fmt: skip


def _device(slug: str) -> _devices.Device:
    dev = next((d for d in _devices.devices() if d.slug == slug), None)
    if dev is None or dev.reference:
        known = [d.slug for d in _devices.devices() if not d.reference]
        msg = f"{slug!r} is not a survey device of docs/data/devices.toml (survey devices: {known}); see `detect`"
        raise SystemExit(msg)
    return dev


# Jobs (no torch).
def _std_inputs() -> dict[str, dict[str, str]]:
    """Per ref {"task", "label", "scene"}: the floor row of the task's standard input, else the ref's smallest."""
    by_ref: dict[str, list[dict[str, str]]] = {}
    for r in floor_rows():
        if r["mode"] in MODES:
            by_ref.setdefault(r["ref"], []).append(
                {"task": r["task"], "label": r["input"], "scene": r.get("scene", "")}
            )
    return {
        ref: min(c, key=lambda i: (i["label"] != STANDARD.get(i["task"]), _px(i["label"]))) for ref, c in by_ref.items()
    }


def _cards() -> dict[str, tuple[Any, int]]:
    """Model id -> (card, index of the variant in its card)."""
    from fastforks._core import registry  # noqa: PLC0415

    out: dict[str, tuple[Any, int]] = {}
    for task in registry.TASKS:
        for card, _ in registry.cards(task).values():
            out |= {v.id: (card, n) for n, v in enumerate(card.variants)}
    return out


def plan(
    dev: _devices.Device, only: Iterable[str] = (), modes: Iterable[str] = MODES, *, subset: bool = False
) -> list[dict[str, Any]]:
    """The survey's jobs, cross-device subset first: one per (ref, mode, dtype), `skip` = why it is `unsupported`."""
    only, want = set(only), [m for m in MODES if m in set(modes)]
    cards, inputs = _cards(), _std_inputs()
    first: dict[str, str] = {}  # family -> its cross-device ref: the card's first variant the floor has
    for ref in sorted(inputs, key=lambda r: (cards[r.split("@")[0]][1], r)):
        first.setdefault(cards[ref.split("@")[0]][0].family, ref)
    picked = set(first.values())
    refs = sorted(picked) + ([] if subset else sorted(set(inputs) - picked))
    out = []
    for ref in refs:
        if only and ref not in only:
            continue
        inp, card = inputs[ref], cards[ref.split("@")[0]][0]
        base = {"ref": ref, "task": inp["task"], "input": inp["label"], "scene": inp["scene"], "cross": ref in picked}
        for mode in want:
            out += [{**base, "mode": mode, "dtype": d, "skip": why} for d, why in _dtypes(card, mode, dev.backend)]
    return out


def _dtypes(card: Any, mode: str, backend: str) -> list[tuple[str, str]]:  # noqa: ANN401, PLR0911 - a ModelCard; one return per case
    """(dtype, reason it is not measured) of a card's mode on a backend; empty reason = measure it."""
    if mode not in card.modes:
        return [("float32", f"the card has no {mode} mode")]
    if mode == "portable":
        if backend != "mps":
            return [(card.portable_cuda_dtype, "")]
        if not card.portable_mps_dtypes:
            return [("float32", "the family is not validated on MPS (card mps_dtypes)")]
        return [(d, "") for d in card.portable_mps_dtypes]
    if backend == "mps":
        why = "" if card.mps_kernel_libs else "no Metal kernels for this family (card mps_kernels)"
        return [(card.mps_kernels_dtype or card.kernels_dtype, why)]
    if backend == "rocm":
        return [(card.kernels_dtype, "the kernel libraries are CUDA builds; no ROCm build")]
    return [(card.kernels_dtype, "" if card.kernel_libs else "the card has no CUDA kernel libraries")]


# MPS worker: imports torch lazily. (cuda / rocm run tools/merge_check.py's worker.)
def _footprint_peak() -> int:
    """Lifetime peak physical footprint of this process in bytes (macOS `proc_pid_rusage`, RUSAGE_INFO_V4)."""
    buf = (ctypes.c_uint64 * 64)()  # rusage_info_v4: 16-byte uuid, then uint64 fields
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:
        msg = "proc_pid_rusage failed"
        raise OSError(msg)
    return int(buf[2 + 28])  # ri_lifetime_max_phys_footprint: the 29th uint64 after the uuid (two uint64 words)


def _image(path: Path) -> Any:  # noqa: ANN401 - torch tensor
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1).to("mps")


def _inputs(model: Any, job: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any], str]:  # noqa: ANN401
    """Call arguments of the job: the floor's real scene where it exists locally, else the synthetic bench inputs."""
    from fastforks._core.bench import BenchSize, bench_inputs  # noqa: PLC0415

    h, w = (int(v) for v in job["input"].split("x"))
    if job["ref"] in SWAPPED:
        h, w = w, h
    args, kw = bench_inputs(model, BenchSize(h, w, VIEWS.get(job["ref"], 8)))
    scene = next((p for k, p in SCENES.items() if job["scene"].startswith(k)), None)
    if scene is None or not scene.exists():
        return args, kw, "synthetic"
    right = Path(str(scene).replace("image_2", "image_3").replace("im0", "im1"))
    return ((_image(scene), _image(right)) if job["task"] == "stereo" else (_image(scene),)), kw, job["scene"]


def _time_mps(call: Callable[[], Any]) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    torch.mps.synchronize()
    t0 = time.perf_counter()
    call()
    torch.mps.synchronize()
    first = time.perf_counter() - t0
    for _ in range(MPS_WARMUP):
        call()
    ts = []
    for _ in range(MPS_RUNS):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        call()
        torch.mps.synchronize()
        ts.append(1e3 * (time.perf_counter() - t0))
    ms = statistics.median(ts)
    return {"ms": round(ms, 3), "ms_min": round(min(ts), 3), "runs": MPS_RUNS, "first_call_s": round(first, 3)}


def worker_mps(spec: dict[str, Any]) -> int:
    """One (ref, mode, dtype) on MPS: a speed row, then a golden row, appended to spec["out"] (merge_check's rows)."""
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")  # before torch: no allocation past the working set
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.0")  # must not exceed the high ratio (default 1.4)
    import torch  # noqa: PLC0415

    import merge_check  # noqa: PLC0415 - its golden step (tools/ is this script's directory)

    torch.manual_seed(0)
    out = Path(spec["out"])
    common: dict[str, Any] = {"ref": spec["ref"], "task": spec["task"], "mode": spec["mode"], "input": spec["input"]}
    m = None
    try:
        task = __import__(f"fastforks.{spec['task']}", fromlist=["load"])
        t0 = time.perf_counter()
        m = model = task.load(spec["ref"], mode=spec["mode"], device="mps", dtype=spec["dtype"])
        torch.mps.synchronize()
        common["load_s"] = round(time.perf_counter() - t0, 2)
        common["backend"] = {"mode": m.backend.mode, "dtype": str(m.backend.dtype), "kernels": list(m.backend.kernels)}
        args, kw, scene = _inputs(m, spec)

        def call() -> Any:  # noqa: ANN401
            with torch.inference_mode():
                return model(*args, **kw)

        row = {**common, "scene": scene, **_time_mps(call), "peak_footprint_mb": round(_footprint_peak() / 2**20)}
    except Exception as e:  # noqa: BLE001 - recorded as a failed / oom row
        row = {**common, "error": " ".join(f"{type(e).__name__}: {e}".split())[:600]}
    merge_check.speed_check()._append(out, row)  # noqa: SLF001 - the row writer of the floor's worker
    if m is not None and "ms" in row:
        merge_check.golden_step(m, {**spec, "device": "mps"}, common, [], out)
    m = None  # the golden step dropped its own reference
    gc.collect()
    return 0


# Orchestration: no torch in this process (no GPU context beside the workers).
def _nvidia_smi() -> dict[str, str]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    q = "driver_version,power.limit,clocks.max.sm,clocks.applications.graphics,clocks_throttle_reasons.active"
    s = subprocess.run([smi, f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    if s.returncode != 0 or not s.stdout.strip():
        return {}
    drv, power, max_sm, app, throttle = (v.strip() for v in [*s.stdout.splitlines()[0].split(","), "", "", ""][:5])
    return {
        "driver": drv,
        "power_limit": power,
        "clocks": f"max SM {max_sm}, applications {app}",
        "throttle_reasons": throttle,
    }


def _probe(backend: str) -> dict[str, Any]:
    """Name, memory and software of this machine's GPU: what rows record and `match` is checked against."""
    if backend == "mps":
        r = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], capture_output=True, text=True, check=False
        )
        name, mem = [*r.stdout.splitlines(), "", ""][:2]
        sp = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"], capture_output=True, text=True, check=False
        )
        gpus = json.loads(sp.stdout or "{}").get("SPDisplaysDataType", [{}]) if sp.returncode == 0 else [{}]
        os_ver = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True, check=False).stdout
        info = {  # the chip name hides the GPU core count (an M1 Pro has 14 or 16): recorded beside it
            "gpu": name.strip(),
            "gpu_cores": gpus[0].get("sppci_cores"),
            "os": f"macOS {os_ver.strip()}" if os_ver.strip() else None,
            "memory_gb": round(int(mem or 0) / 2**30),
            "torch": importlib.metadata.version("torch"),
        }
        return {k: v for k, v in info.items() if v not in (None, "")}
    code = (
        "import json, torch; p = torch.cuda.get_device_properties(0); print(json.dumps({'gpu': p.name,"
        " 'memory_gb': round(p.total_memory / 2**30), 'torch': torch.__version__, 'cuda': torch.version.cuda,"
        " 'hip': torch.version.hip, 'arch': f'sm_{p.major}{p.minor}'}))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    info = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else {"gpu": ""}
    if not info.get("hip"):
        info |= _nvidia_smi()
    return {k: v for k, v in info.items() if v not in (None, "")}


def _backend_now() -> str:
    if sys.platform == "darwin":
        if platform.machine() != "arm64":
            msg = "an MPS survey needs Apple silicon (this Mac is Intel); nothing measured"
            raise SystemExit(msg)
        return "mps"
    r = subprocess.run(
        [sys.executable, "-c", "import torch; print('rocm' if torch.version.hip else 'cuda')"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip() or "cuda"


def detect(_a: argparse.Namespace) -> int:
    """Print this machine's GPU as a devices.toml entry to paste (slug: model + memory; edit as you like)."""
    backend = _backend_now()
    info = _probe(backend)
    if not info.get("gpu"):
        msg = "no GPU found (torch.cuda / sysctl)"
        raise SystemExit(msg)
    base, mem = re.sub(r"^(nvidia( geforce)?|amd( radeon)?) ", "", info["gpu"].lower()), info.get("memory_gb", 0)
    slug = re.sub(r"[^a-z0-9]+", "-", base if re.search(rf"{mem}\s*gb", base) else f"{base} {mem}gb").strip("-")
    sw = ", ".join(f"{k} {info[k]}" for k in ("torch", "cuda", "hip", "driver") if info.get(k))
    print(f'[{slug}]\nname = "{info["gpu"]}"\nbackend = "{backend}"\nmatch = "{info["gpu"]}"')
    cores = f"{info['gpu_cores']}-core GPU" if info.get("gpu_cores") else ""
    details = ", ".join(filter(None, [info.get("arch"), cores, info.get("os")]))
    print(f'memory = "{mem} GB"\ndetails = "{details}"\nsoftware = "{sw}"')
    return 0


def _status(error: str) -> str:
    low = error.lower()
    return "oom" if "out of memory" in low or "outofmemoryerror" in low else "failed"


def _row(
    dev: _devices.Device, job: dict[str, Any], common: dict[str, Any], got: list[dict[str, Any]]
) -> dict[str, Any]:
    """The survey row of one job from its worker rows (speed row, golden row), or of a skip / crash."""
    speed = next((r for r in got if r.get("input") != GOLDEN), {})
    gold = next((r for r in got if r.get("input") == GOLDEN), {})
    row = {**common, **{k: v for k, v in speed.items() if k not in _DROP}}
    row |= {"device": dev.slug, "ref": job["ref"], "task": job["task"], "mode": job["mode"], "input": job["input"]}
    row["scene"] = speed.get("scene") or "synthetic"
    row["dtype"] = str((speed.get("backend") or {}).get("dtype", "")).removeprefix("torch.") or job["dtype"]
    if job["skip"]:
        row |= {"status": "unsupported", "note": job["skip"]}
    elif isinstance(speed.get("ms"), int | float) and not speed.get("error"):
        row |= {"status": "ok", "memory_kind": "process-footprint" if dev.backend == "mps" else "torch-peak"}
    else:
        error = str(speed.get("error") or speed.get("fails") or "no row from the worker")
        row = {k: v for k, v in row.items() if k != "ms"} | {"status": _status(error), "error": error[:600]}
    if gold:
        row["golden"] = gold.get("golden", "fail")
        row |= {k: gold[k] for k in ("golden_mean", "golden_p99", "golden_msg", "golden_s") if k in gold}
        if gold.get("error"):
            row["golden_error"] = str(gold["error"])[:600]
    return row


def _merge(dev: _devices.Device, new: dict[str, Any]) -> None:
    """Write `new` into the device's file, replacing the row of the same ref / mode / dtype / input."""
    path = dev.data
    path.parent.mkdir(parents=True, exist_ok=True)
    key = (new["ref"], new["mode"], new["dtype"], new["input"])
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.is_file() else []
    rows = [r for r in rows if (r["ref"], r["mode"], r["dtype"], r["input"]) != key] + [new]
    rows.sort(key=lambda r: (r["task"], r["ref"], _devices.MODES.index(r["mode"]), r["dtype"], r["input"]))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _worker_cmd(dev: _devices.Device, job: dict[str, Any], out: Path, commit: str) -> list[str]:
    if dev.backend == "mps":
        return [sys.executable, str(TOOLS / "speed_survey.py"), "worker", json.dumps({**job, "out": str(out)})]
    spec = {  # tools/speed_check.py's worker spec with one input; merge_check's worker adds the golden step
        "ref": job["ref"],
        "task": job["task"],
        "mode": job["mode"],
        "inputs": [{"label": job["input"], "scene": job["scene"]}],
        "side": "lib",
        "out": str(out),
        "commit": commit,
        "forks": "",
        "fork_key": "",
        "no_profile": True,
        "trace": False,
    }
    return [sys.executable, str(TOOLS / "merge_check.py"), "worker", json.dumps(spec)]


def _env(*, online: bool) -> dict[str, str]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))}
    env.setdefault("FASTFORKS_NONCOMMERCIAL", "1")  # as tests/conftest.py and merge_check: the floor has such weights
    if online:
        env.pop("FASTFORKS_OFFLINE", None)
    else:
        env["FASTFORKS_OFFLINE"] = "1"
    return env


def _check_machine(dev: _devices.Device) -> dict[str, Any]:
    info = _probe(dev.backend)
    gpu = info.get("gpu", "")
    if (gpu != dev.match) if dev.backend == "mps" else (dev.match not in gpu):  # "Apple M1" must not match an M1 Pro
        msg = f"this machine's GPU is {info.get('gpu')!r}, not {dev.match!r} ({dev.slug}); nothing measured"
        raise SystemExit(msg)
    free = shutil.disk_usage("/").free / 2**30
    if dev.backend == "mps" and free < MIN_FREE_GB:
        msg = f"{free:.0f} GB free on the system disk, {MIN_FREE_GB:g} needed (FASTFORKS_MIN_FREE_GB)"
        raise SystemExit(msg)
    return info


def _measure(
    dev: _devices.Device, job: dict[str, Any], env: dict[str, str], commit: str, timeout: float
) -> list[dict[str, Any]]:
    """The worker rows of one job (in a fresh process); a crash or timeout: one error row."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rows.jsonl"
        try:
            r = subprocess.run(
                _worker_cmd(dev, job, out, commit),
                env=env,
                timeout=timeout,
                check=False,
                capture_output=True,
                text=True,
            )
            tail = f"rc={r.returncode}: " + " ".join(r.stderr.split()[-60:])
        except subprocess.TimeoutExpired:
            tail = f"timeout after {timeout:g} s"
        got = [json.loads(s) for s in out.read_text().splitlines() if s.strip()] if out.is_file() else []
    return got if any(g.get("input") != GOLDEN for g in got) else [{"error": f"worker {tail}"}]


def _disk_ok() -> bool:
    """MIN_FREE_GB free on the system disk, waiting up to SWAP_WAIT_S for swap to shrink after a big model."""
    t0 = time.monotonic()
    while (free := shutil.disk_usage("/").free / 2**30) < MIN_FREE_GB:
        if time.monotonic() - t0 > SWAP_WAIT_S:
            msg = f"stopped: {free:.0f} GB free on the system disk, {MIN_FREE_GB:g} needed (FASTFORKS_MIN_FREE_GB)"
            print(msg, file=sys.stderr)
            return False
        time.sleep(30)
    return True


def run(a: argparse.Namespace) -> int:
    dev = _device(a.device)
    info = _check_machine(dev)
    if not os.environ.get("FASTFORKS_GOLDEN_PATH"):
        print("FASTFORKS_GOLDEN_PATH is not set: every golden check will be `skip`", file=sys.stderr)
    done = {(r["ref"], r["mode"], r["dtype"], r["input"]) for r in _devices.survey_rows(dev)}
    env, commit = _env(online=a.online), _commit()
    common = {k: info[k] for k in _ENV_KEYS if k in info}
    common |= {"date": time.strftime("%F"), "commit": commit, "protocol_id": "warm-median"}
    common["protocol"] = (
        MPS_PROTOCOL if dev.backend == "mps" else "first call, 3 warm-up, peak reset, median of 3 rounds x median of 5"
    )
    todo = plan(dev, a.only, a.modes, subset=a.subset)
    for n, job in enumerate(todo, 1):
        if not a.force and (job["ref"], job["mode"], job["dtype"], job["input"]) in done:
            continue
        if dev.backend == "mps" and not _disk_ok():
            return 1
        print(
            f"== [{n}/{len(todo)}] {job['ref']} {job['mode']} {job['dtype']} {job['input']} {time.strftime('%T')}",
            flush=True,
        )
        got = [] if job["skip"] else _measure(dev, job, env, commit, a.timeout)
        row = _row(dev, job, common, got)
        _merge(dev, row)
        print(
            "  ",
            json.dumps({k: row[k] for k in ("status", "ms", "golden", "note", "error") if row.get(k)})[:300],
            flush=True,
        )
    _devices.survey_rows(dev)  # the written file passes gen_docs' schema check
    return 0


def prepare(a: argparse.Namespace) -> int:
    """Weights (and on cuda the kernel binaries for this GPU) of the jobs into the cache, then `--check` them."""
    dev = _device(a.device)
    refs = sorted({j["ref"] for j in plan(dev, a.only, subset=a.subset) if not j["skip"]})
    env = _env(online=True)
    cli = [sys.executable, "-m", "fastforks", "download", *refs, *(["--kernels"] if dev.backend == "cuda" else [])]
    return (
        subprocess.run(cli, env=env, check=False).returncode
        or subprocess.run([*cli, "--check"], env=env, check=False).returncode
    )


def list_jobs(a: argparse.Namespace) -> int:
    for j in plan(_device(a.device), a.only, a.modes, subset=a.subset):
        why = f"unsupported: {j['skip']}" if j["skip"] else ""
        print(f"{'*' if j['cross'] else ' '} {j['ref']:40} {j['mode']:9} {j['dtype']:9} {j['input']:10} {why}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect")
    for name in ("list", "prepare", "run"):
        p = sub.add_parser(name)
        p.add_argument("--device", required=True, help="survey slug of docs/data/devices.toml")
        p.add_argument("--only", nargs="*", default=[], metavar="REF")
        p.add_argument("--subset", action="store_true", help="only the cross-device subset (one variant per family)")
        p.add_argument("--modes", nargs="*", default=list(MODES), choices=MODES)
        if name == "run":
            p.add_argument("--force", action="store_true", help="re-measure rows already in the file")
            p.add_argument("--online", action="store_true", help="allow downloads (default: FASTFORKS_OFFLINE=1)")
            p.add_argument("--timeout", type=float, default=TIMEOUT_S, help="seconds per (ref, mode, dtype)")
    sub.add_parser("worker").add_argument("spec")
    a = ap.parse_args()
    if a.cmd == "worker":
        return worker_mps(json.loads(a.spec))
    with contextlib.suppress(BrokenPipeError):
        return {"detect": detect, "list": list_jobs, "prepare": prepare, "run": run}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
