"""Re-tune the Apple MLX backend (`load(..., device="mlx")`) on the Mac this runs on.

    python tools/mlx_tune.py                     # all: the MLX golden gates, the micro-benchmarks, the MLX survey
    python tools/mlx_tune.py gates               # golden gates of every MLX port, each mode and dtype
    python tools/mlx_tune.py bench [NAME ...]    # micro-benchmarks, one per tunable (all without names)
    python tools/mlx_tune.py survey [--mps]      # speed survey rows of this Mac's MLX device entry (and its MPS one)
    python tools/mlx_tune.py ratios              # MLX / MPS time and memory per ref from the recorded survey rows

Every tunable of the MLX ports is a field of `fastforks._core.mlx.Tuning`; the one table `TUNING` there holds a row
per GPU architecture (`mx.device_info()["architecture"]`) and the M1 values as `"default"`. Each field has a
micro-benchmark here (`bench`): it loads a model with the field set to each candidate (everything else as this GPU's
row has it), times it, and prints the table and the row to paste into `TUNING` if a candidate is more than GAIN
faster than the current value. Nothing is written: the table is edited by hand, then `gates` runs again.

| benchmark | field(s) | measured on |
|---|---|---|
| cache | cache_fraction, cache_min_mb, cache_cap_mb | CREStereo kernels float16: time and MLX peak + cache per limit |
| threadgroup | threadgroup | CREStereo kernels float16 |
| lane | warp_lane | CREStereo kernels float16 |
| conv-pad | conv_pad | 3x3 convolutions 128 -> 96 against 128 -> 128, then CREStereo kernels float16 |
| winograd | winograd | CREStereo kernels float32 and float16 |
| batch-pair | batch_pair | CREStereo kernels float16: time and peak memory |
| compile | compile_steps | CREStereo kernels float16 and DINOv2 ViT-S/14 float16 |

`gates` needs FASTFORKS_GOLDEN_PATH (tests/test_golden.py::test_mlx). `survey` needs this Mac's MLX entry in
docs/data/devices.toml; without one it prints the entry to add (`tools/speed_survey.py detect --mlx`). `--mps` also
measures the same refs through PyTorch MPS on the Mac's MPS entry, so `ratios` compares the two backends on the same
machine: ratios, not absolute times, say whether MLX pays on a newer chip (PyTorch MPS gets faster there too).
A candidate that changes the converted weights (conv_pad, winograd) writes one more conversion into the cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import _devices

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
GAIN = 0.03  # a candidate replaces the current value only if it is more than 3 % faster (run-to-run noise is ~1 %)
CACHE_GAIN = 0.05  # a larger buffer cache is held for the whole process: it has to buy 5 % to be worth it
MEMORY_GAIN = 0.05  # at equal speed, a candidate replaces the current value if its MLX peak is 5 % lower
WARMUP, RUNS = 2, 5
STEREO, FEATURES = "crestereo-eth3d", "dinov2-vits14"
STEREO_SIZE, FEATURES_SIZE = (480, 736), (448, 448)  # big enough to be GPU-bound, small enough for a 16 GB Mac
CONV_SIZE = (248, 372)  # the 1/4-resolution feature map of the survey's 994x1482 stereo input
_MB = 1 << 20


def _load(ref: str, mode: str, dtype: str) -> Any:  # noqa: ANN401 - a fastforks model
    from fastforks._core import registry  # noqa: PLC0415

    card, _ = registry.resolve_any(ref)
    task = __import__(f"fastforks.{card.task}", fromlist=["load"])
    return task.load(ref, device="mlx", mode=mode, dtype=dtype)


def _measure(ref: str, mode: str, dtype: str, size: tuple[int, int], **fields: object) -> dict[str, float]:
    """Median ms of a model built with `fields` overriding the tunables; MLX's peak of live arrays and its cache."""
    import torch  # noqa: PLC0415

    from fastforks._core import mlx as runtime  # noqa: PLC0415
    from fastforks._core.bench import BenchSize, bench_inputs  # noqa: PLC0415

    mx = runtime.mx
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    with runtime.tuned(**fields), torch.inference_mode():  # the tunables are read at load and at the first call
        model = _load(ref, mode, dtype)
        args, kw = bench_inputs(model, BenchSize(*size))
        for _ in range(1 + WARMUP):
            model(*args, **kw)
        ts = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            model(*args, **kw)
            ts.append(1e3 * (time.perf_counter() - t0))
    out = {"ms": statistics.median(ts), "peak_mb": mx.get_peak_memory() / _MB, "cache_mb": mx.get_cache_memory() / _MB}
    del model
    return out


def _sweep(
    title: str, field: str, values: list[Any], target: tuple[str, str, str, tuple[int, int]], *, memory: bool = False
) -> dict[str, Any]:
    """Time `target` (ref, mode, dtype, size) per candidate of `field`; returns `{field: best}` if it should change.

    `memory`: the lower peak decides between candidates within GAIN of the fastest (batch_pair, conv_pad), if it
    is more than MEMORY_GAIN lower.
    """
    from fastforks._core import mlx as runtime  # noqa: PLC0415

    current = getattr(runtime.tuning(), field)
    ref, mode, dtype, size = target
    print(f"\n{title}: {field} on {ref} {mode} {dtype} {size[0]}x{size[1]} (current: {current})")
    rows = {}
    for v in values:
        try:
            rows[v] = r = _measure(ref, mode, dtype, size, **{field: v})
        except ValueError as e:  # Metal refuses the value on this GPU (e.g. more threads than a threadgroup holds)
            print(f"  {v!s:>6}  not possible here: {e}")
            continue
        mark = " <- current" if v == current else ""
        print(f"  {v!s:>6}  {r['ms']:9.1f} ms  peak {r['peak_mb']:7.0f} MB  cache {r['cache_mb']:6.0f} MB{mark}")
    fastest = min(r["ms"] for r in rows.values())
    near = [v for v, r in rows.items() if r["ms"] <= fastest * (1 + GAIN)]
    best = min(near, key=lambda v: rows[v]["peak_mb"] if memory else rows[v]["ms"])
    if current in near and (not memory or rows[current]["peak_mb"] <= rows[best]["peak_mb"] * (1 + MEMORY_GAIN)):
        print(f"  keep {field}={current}")
        return {}
    was = f"{rows[current]['ms']:.1f}" if current in rows else "?"
    print(f"  change {field}: {current} -> {best} ({was} -> {rows[best]['ms']:.1f} ms)")
    return {field: best}


def bench_cache() -> dict[str, Any]:
    """The buffer cache limit: MLX keeps freed buffers up to it; too small costs allocations, too large memory."""
    from fastforks._core import mlx as runtime  # noqa: PLC0415

    g, tune = runtime.gpu(), runtime.tuning()
    now = tune.cache_limit(g.memory) // _MB
    limits = [v for v in (256, 512, 1024, 2048, 4096, 8192) if v * _MB <= g.memory // 4]
    print(f"\ncache: limit in MB on {STEREO} kernels float16 (current: {now} MB of {g.memory / 2**30:.0f} GB)")
    rows = {}
    for v in limits:
        rows[v] = _measure(STEREO, "kernels", "float16", STEREO_SIZE, cache_min_mb=v, cache_cap_mb=v)
        r = rows[v]
        mark = " <- current" if v == now else ""
        print(f"  {v:>6}  {r['ms']:9.1f} ms  peak {r['peak_mb']:7.0f} MB  cache {r['cache_mb']:6.0f} MB{mark}")
    fastest = min(r["ms"] for r in rows.values())
    best = min(v for v, r in rows.items() if r["ms"] <= fastest * (1 + CACHE_GAIN))  # the smallest that is as fast
    if best == now or (now in rows and rows[now]["ms"] <= fastest * (1 + CACHE_GAIN) and now <= best):
        print(f"  keep the cache limit ({now} MB)")
        return {}
    print(f"  change: {best} MB is the smallest limit within {CACHE_GAIN:.0%} of the fastest")
    return {"cache_fraction": round(best * _MB / g.memory, 4), "cache_min_mb": min(tune.cache_min_mb, best)}


def bench_threadgroup() -> dict[str, Any]:
    """Threads per threadgroup of the 1-D Metal kernels."""
    return _sweep("threadgroup", "threadgroup", [64, 128, 256, 512, 1024], (STEREO, "kernels", "float16", STEREO_SIZE))


def bench_lane() -> dict[str, Any]:
    """Channels one thread of the warp kernel interpolates."""
    return _sweep("lane", "warp_lane", [4, 8, 16, 32], (STEREO, "kernels", "float16", STEREO_SIZE))


def _conv_ms(mx: Any, cin: int, cout: int, dtype: Any) -> float:  # noqa: ANN401 - mlx.core and one of its dtypes
    x = mx.random.normal((1, *CONV_SIZE, cin)).astype(dtype)
    w = mx.random.normal((cout, 3, 3, cin)).astype(dtype)
    mx.eval(x, w)
    ts = []
    for i in range(3 + RUNS):
        t0 = time.perf_counter()
        mx.eval(mx.conv2d(x, w, padding=1))
        if i >= 3:  # noqa: PLR2004 - three warm-up calls
            ts.append(1e3 * (time.perf_counter() - t0))
    return statistics.median(ts)


def bench_conv_pad() -> dict[str, Any]:
    """Whether MLX's 3x3 convolution is slower with 96 outputs than with 128 (the reason for the padding)."""
    from fastforks._core.mlx import mx  # noqa: PLC0415

    print(f"\nconv-pad: MLX 3x3 convolution at {CONV_SIZE[0]}x{CONV_SIZE[1]}, ms (in -> out channels)")
    for dtype in (mx.float32, mx.float16):
        cells = [f"{i} -> {o}: {_conv_ms(mx, i, o, dtype):6.1f}" for i, o in ((128, 128), (128, 96), (96, 96))]
        print(f"  {str(dtype).rsplit('.', 1)[-1]:8} " + "   ".join(cells))
    return _sweep("conv-pad", "conv_pad", [True, False], (STEREO, "kernels", "float16", STEREO_SIZE), memory=True)


def bench_winograd() -> dict[str, Any]:
    """The GRU's 1x5 / 5x1 convolutions as Winograd F(4, 5) with one float32 matmul, against MLX's convolution."""
    out = _sweep("winograd", "winograd", [True, False], (STEREO, "kernels", "float32", STEREO_SIZE))
    return out | _sweep("winograd", "winograd", [True, False], (STEREO, "kernels", "float16", STEREO_SIZE))


def bench_batch_pair() -> dict[str, Any]:
    """Both images of a pair through the encoder in one batch, against one at a time."""
    return _sweep("batch-pair", "batch_pair", [False, True], (STEREO, "kernels", "float16", STEREO_SIZE), memory=True)


def bench_compile() -> dict[str, Any]:
    """`mx.compile` of the per-iteration step and the encoders."""
    out = _sweep("compile", "compile_steps", [True, False], (STEREO, "kernels", "float16", STEREO_SIZE))
    return out | _sweep("compile", "compile_steps", [True, False], (FEATURES, "portable", "float16", FEATURES_SIZE))


BENCHES = {
    "cache": bench_cache,
    "threadgroup": bench_threadgroup,
    "lane": bench_lane,
    "conv-pad": bench_conv_pad,
    "winograd": bench_winograd,
    "batch-pair": bench_batch_pair,
    "compile": bench_compile,
}


def bench(names: list[str]) -> int:
    """Run the micro-benchmarks and print the `TUNING` row this Mac should get."""
    from fastforks._core import mlx as runtime  # noqa: PLC0415

    g = runtime.gpu()
    print(
        f"{g.architecture}, {g.memory / 2**30:.0f} GB, mlx {runtime.version()};"
        f" tunables row in use: {runtime.tuning_row()!r}"
        + (" (GPU neural accelerators)" if runtime.has_neural_accelerators() else "")
    )
    change: dict[str, Any] = {}
    for name in names or list(BENCHES):
        change |= BENCHES[name]()
    if not change:
        print(f"\nNothing to change: the {runtime.tuning_row()!r} row of TUNING is right for this GPU.")
        return 0
    family = g.architecture.rstrip("abcdefghijklmnopqrstuvwxyz") if g.generation else g.architecture
    fields = ", ".join(f"{k}={v!r}" for k, v in change.items())
    print(
        f"\nPut into TUNING (src/fastforks/_core/mlx.py), then run `python tools/mlx_tune.py gates`:\n"
        f'    "{family}": dataclasses.replace(_M1, {fields}),\n'
        f'("{g.architecture}" instead of "{family}" if only this tier of the generation was measured.)\n'
        f"To try it first without editing: FASTFORKS_MLX_TUNING="
        f"'{','.join(f'{k}={int(v) if isinstance(v, bool) else v}' for k, v in change.items())}'"
    )
    return 0


def _env() -> dict[str, str]:
    path = os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))
    return {**os.environ, "PYTHONPATH": path}


def gates() -> int:
    """The golden gates of every MLX port (tests/test_golden.py::test_mlx: each declared mode and dtype)."""
    if not os.environ.get("FASTFORKS_GOLDEN_PATH"):
        print("gates: FASTFORKS_GOLDEN_PATH is not set (the goldens are not in the repository); nothing checked")
        return 1
    cmd = [sys.executable, "-m", "pytest", str(ROOT / "tests" / "test_golden.py"), "-m", "mlx", "-q", "-rs"]
    return subprocess.run(cmd, cwd=ROOT, env=_env(), check=False).returncode


def _chip() -> str:
    r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False)
    return r.stdout.strip()


def _entries() -> dict[str, _devices.Device]:
    """This Mac's device entries by backend (`mlx`, `mps`): `match` is the chip name."""
    chip = _chip()
    return {d.backend: d for d in _devices.devices() if d.backend in ("mlx", "mps") and d.match == chip}


def _ported() -> list[str]:
    """The survey's refs that have an MLX port."""
    import speed_survey  # noqa: PLC0415

    dev = next(d for d in _devices.devices() if d.backend == "mlx")
    return sorted({j["ref"] for j in speed_survey.plan(dev)})


def survey(*, mps: bool, force: bool) -> int:
    """Speed survey rows of this Mac's MLX entry; `mps`: also the same refs on its MPS entry."""
    have = _entries()
    tool = [sys.executable, str(TOOLS / "speed_survey.py")]
    missing = [b for b in (("mlx", "mps") if mps else ("mlx",)) if b not in have]
    for backend in missing:
        print(f"survey: docs/data/devices.toml has no {backend} entry for {_chip()!r}. Add this one, then run again:\n")
        subprocess.run([*tool, "detect", *(["--mlx"] if backend == "mlx" else [])], env=_env(), check=False)
        print("\n(and add docs/devices/<slug>.md to the Benchmarks section of mkdocs.yml; see docs/data/README.md)")
    if missing:
        return 1
    extra = ["--force"] if force else []
    rc = subprocess.run([*tool, "run", "--device", have["mlx"].slug, *extra], env=_env(), check=False).returncode
    if mps:
        cmd = [*tool, "run", "--device", have["mps"].slug, "--only", *_ported(), *extra]
        rc = subprocess.run(cmd, env=_env(), check=False).returncode or rc
    return rc or ratios()


def ratios() -> int:
    """MLX against the fastest PyTorch MPS mode per ref and dtype, from this Mac's recorded survey rows."""
    have = _entries()
    if "mlx" not in have or "mps" not in have:
        print(f"ratios: {_chip()!r} needs an mlx and an mps entry in docs/data/devices.toml (have: {sorted(have)})")
        return 1

    def ok(dev: _devices.Device) -> dict[tuple[str, str], list[dict[str, Any]]]:
        out: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in _devices.survey_rows(dev):
            if r["status"] == "ok" and r.get("golden") != "fail":
                out.setdefault((r["ref"], r["dtype"]), []).append(r)
        return out

    mlx, mps = ok(have["mlx"]), ok(have["mps"])
    print(f"{have['mlx'].name} against {have['mps'].name}: the fastest mode of each backend, same ref / dtype / input")
    head = ("MLX ms", "mode", "MPS ms", "mode", "MPS/MLX", "MLX MB", "MPS MB")
    print(f"{'ref':28} {'dtype':8} " + " ".join(f"{h:>9}" for h in head))
    for key in sorted(mlx):
        a = min(mlx[key], key=lambda r: r["ms"])
        if key not in mps:
            print(f"{key[0]:28} {key[1]:8} {a['ms']:9.1f} {a['mode']:>9} {'-':>9}")
            continue
        b = min(mps[key], key=lambda r: r["ms"])
        print(
            f"{key[0]:28} {key[1]:8} {a['ms']:9.1f} {a['mode']:>9} {b['ms']:9.1f} {b['mode']:>9}"
            f" {b['ms'] / a['ms']:9.2f} {a.get('peak_footprint_mb', 0):9} {b.get('peak_footprint_mb', 0):9}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gates")
    sub.add_parser("bench").add_argument("names", nargs="*", metavar="NAME", help=", ".join(BENCHES))
    for name in ("survey", "all"):
        p = sub.add_parser(name)
        p.add_argument("--mps", action="store_true", help="also measure the MLX-ported refs through PyTorch MPS")
        p.add_argument("--force", action="store_true", help="re-measure survey rows already recorded")
    sub.add_parser("ratios")
    a = ap.parse_args()
    if a.cmd == "gates":
        return gates()
    if a.cmd == "bench":
        return bench(a.names)
    if a.cmd == "survey":
        return survey(mps=a.mps, force=a.force)
    if a.cmd == "ratios":
        return ratios()
    rc = gates()  # all: a failing gate does not stop the measurements, it is the first thing to look at
    print(json.dumps({"gates": "pass" if rc == 0 else "FAIL"}))
    rc = bench([]) or rc
    return survey(mps=getattr(a, "mps", False), force=getattr(a, "force", False)) or rc


if __name__ == "__main__":
    sys.exit(main())
