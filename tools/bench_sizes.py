"""Fixed-size speed + peak GPU memory tables of every model with local weights, portable and kernels (model cards).

Each measurement uses the library's own bench path (`fastforks._core.bench`: the task's `bench_inputs`, `median_ms`),
so the numbers match `fastforks bench MODEL --mode M --size HxW`. Results are appended to a JSON-lines file, one record
per (model, mode, size); a run resumes from that file, so a queue can call it repeatedly in short GPU-lock steps:

    python tools/bench_sizes.py list > families.txt              # "task/family ref ref ..." with local weights
    python tools/bench_sizes.py run --out sizes.jsonl REF ...   # exit 3: call again, more to do
    python tools/bench_sizes.py table sizes.jsonl > tables.md

Per model and mode the sizes run smallest first; after an out-of-memory error (or a call killed twice by the step
timeout) the larger sizes are recorded as skipped. Peak memory is `max_memory_allocated` / `max_memory_reserved` from
a fresh `reset_peak_memory_stats` + `empty_cache` before each size, so it includes the weights. Process-wide caches
(e.g. the cuDNN plan workspace, up to hundreds of MiB after a large size) outlive the model, so a size measured after a
larger one, or a model after another one, would show their memory: use `--fresh` (one size per process) for tables.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.model import Model

SIZES = {  # name: (height, width), smallest first
    "512x512": (512, 512),
    "1024x1024": (1024, 1024),
    "2048x1024": (1024, 2048),  # name is width x height; the value is (height, width)
    "640x480": (480, 640),
    "kitti": (375, 1242),
    "1280x720": (720, 1280),
    "1920x1080": (1080, 1920),
    "middlebury": (1988, 2880),
    "12mp": (3000, 4000),
    "native": (0, 0),  # the model's fixed crop, resolved per model
}
LABELS = {  # shown as height x width, like the CLI
    "640x480": "480x640",
    "1280x720": "720x1280",
    "1920x1080": "1080x1920",
    "kitti": "KITTI 375x1242",
    "middlebury": "Middlebury 1988x2880",
    "12mp": "12 MP 3000x4000",
}
MODES = ("portable", "kernels")
MIB = 2**20
MAX_KILLS = 2  # a call killed by the step timeout this often is recorded as a timeout


def _families() -> dict[str, list[str]]:
    """`{"task/family": [ref, ...]}` of the models whose weights are all available locally (no downloads)."""
    from fastforks._core import registry  # noqa: PLC0415 - keep `--help` light
    from fastforks._core.errors import WeightsError  # noqa: PLC0415

    found: dict[str, list[str]] = {}
    for task in registry.TASKS:
        for card, v in registry.cards(task).values():
            try:
                registry.weights(card, v, allow_download=False)
            except WeightsError as e:
                print(f"skip {v.ref}: {e}", file=sys.stderr)
                continue
            found.setdefault(f"{task}/{card.family}", []).append(v.ref)
    return found


def sizes_for(ref: str, wanted: list[str] | None = None) -> list[str]:
    """Size names to run for `ref`, smallest first, inside the family's `size_range()`.

    Fixed-input models (`native`: classification / features crops) run only at their native size; promptable
    segmentation (fixed `encoder` side, the image is resized inside) at the three small sizes; other families at the
    full ladder: small (480x640), typical (720x1280), large (1080x1920), then the probes (Middlebury 1988x2880, 12 MP)
    that find the largest size that fits in 16 GB. KITTI 375x1242 is a stereo size only.
    """
    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.sizes import size_range  # noqa: PLC0415

    card, v = registry.resolve_any(ref)
    rng = size_range(v, card)
    if rng.native:
        return ["native"]
    names = [s for s in SIZES if s != "native" and (card.task == "stereo" or s != "kitti")]
    names = [s for s in names if min(SIZES[s]) >= rng.min_size]
    if rng.encoder:
        names = [s for s in names if SIZES[s][0] * SIZES[s][1] <= SIZES["1920x1080"][0] * SIZES["1920x1080"][1]]
    return [s for s in names if wanted is None or s in wanted]


def _read(out: Path) -> list[dict]:
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip("\0 \n")] if out.is_file() else []


def _append(out: Path, rec: dict) -> None:
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")


class _FatalError(Exception):
    """A CUDA error that may leave the context unusable: stop this process, the next step starts fresh."""


def _measure(model: Model, hw: tuple[int, int], runs: int, slow_ms: float) -> dict:
    import torch  # noqa: PLC0415

    from fastforks._core.bench import WARMUP, BenchSize, bench_inputs, median_ms  # noqa: PLC0415

    dev = model.device
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    torch.manual_seed(0)
    inputs, kwargs = bench_inputs(model, BenchSize(*hw))

    def call() -> None:
        cast("Callable[..., object]", model)(*inputs, **kwargs)  # each task's Model subclass is callable

    probe = median_ms(call, device="cuda", runs=1, warmup=WARMUP)  # warm-up calls + one timed call
    n = runs if probe < slow_ms else 3
    ms = median_ms(call, device="cuda", runs=n, warmup=0)
    return {
        "status": "ok",
        "ms": round(ms, 2),
        "runs": n,
        "alloc_mib": round(torch.cuda.max_memory_allocated(dev) / MIB),
        "reserved_mib": round(torch.cuda.max_memory_reserved(dev) / MIB),
        "backend": str(model.backend),
    }


def _run(args: argparse.Namespace) -> int:  # noqa: C901, PLR0912, PLR0915 - one resumable loop
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.errors import BackendUnavailableError, FastforksError  # noqa: PLC0415
    from fastforks._core.sizes import size_range  # noqa: PLC0415

    t0 = time.monotonic()
    out = Path(args.out)
    refs = args.refs
    commit = args.commit
    started: dict[tuple, int] = {}
    done: set[tuple] = set()
    blocked: set[tuple] = set()  # (ref, mode): larger sizes skipped
    measured = 0  # the budget applies once this process has measured something (a slow load alone still progresses)
    for r in _read(out):
        key = (r["ref"], r["mode"], r["size"])
        if r["status"] == "started":
            started[key] = started.get(key, 0) + 1
        else:
            done.add(key)
            if r["status"] in {"oom", "timeout", "skipped", "no kernels", "error"}:
                blocked.add(key[:2])
    for ref in refs:
        for mode in args.modes:
            todo = [s for s in sizes_for(ref, args.sizes) if (ref, mode, s) not in done]
            if not todo:
                continue
            if measured and time.monotonic() - t0 > args.budget:
                return 3
            card, variant = registry.resolve_any(ref)
            native = size_range(variant, card).native
            base = {"ref": ref, "mode": mode, "commit": commit, "gpu": torch.cuda.get_device_name()}
            try:
                model = importlib.import_module(f"fastforks.{card.task}").load(ref, mode=mode, device="cuda")
            except BackendUnavailableError as e:
                for s in todo:
                    _append(out, base | {"size": s, "status": "no kernels", "why": str(e)[:300]})
                continue
            except torch.OutOfMemoryError as e:
                for s in todo:
                    _append(out, base | {"size": s, "status": "oom", "why": f"at load: {str(e)[:200]}"})
                continue
            try:
                for s in todo:
                    key = (ref, mode, s)
                    hw = native if s == "native" and native else SIZES[s]
                    rec = base | {"size": s, "hw": hw}
                    if key[:2] in blocked:
                        _append(out, rec | {"status": "skipped", "why": "a smaller size failed"})
                        continue
                    if started.get(key, 0) >= MAX_KILLS:
                        _append(out, rec | {"status": "timeout", "why": "killed by the step timeout twice"})
                        blocked.add(key[:2])
                        continue
                    if measured and time.monotonic() - t0 > args.budget:
                        return 3
                    measured += 1
                    _append(out, rec | {"status": "started"})
                    try:
                        _append(out, rec | _measure(model, hw, args.runs, args.slow_ms))
                    except torch.OutOfMemoryError as e:
                        _append(out, rec | {"status": "oom", "why": str(e).split("\n")[0][:200]})
                        blocked.add(key[:2])
                    except FastforksError as e:  # e.g. an input-size limit of the model
                        _append(out, rec | {"status": "input error", "why": str(e)[:300]})
                    except RuntimeError as e:
                        _append(out, rec | {"status": "error", "why": str(e).split("\n")[0][:300]})
                        blocked.add(key[:2])
                        if "CUDA" in str(e) or "cuDNN" in str(e):
                            raise _FatalError from e
                    if args.fresh == "size":
                        return 3  # the next call measures the next size in a fresh process
            except _FatalError:
                return 3
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            if args.fresh == "mode":
                return 3  # the next call starts the next (model, mode) in a fresh process
    return 0


def _cell(r: dict | None) -> str:
    if r is None:
        return "-"
    if r["status"] == "ok":
        n = "" if r["runs"] >= 10 else f" (n={r['runs']})"  # noqa: PLR2004 - the default N
        return f"{r['ms']:.1f} ms{n} / {r['alloc_mib']} / {r['reserved_mib']}"
    return {"input error": "input error", "no kernels": "no kernels"}.get(r["status"], r["status"].upper())


DOCS = Path(__file__).resolve().parents[1] / "docs" / "benchmarks.md"
DOC_SIZES = ("640x480", "1280x720", "1920x1080", "12mp")
BEGIN, END = "<!-- sizes:begin (tools/bench_sizes.py docs) -->", "<!-- sizes:end -->"


def _docs(args: argparse.Namespace) -> int:
    """Rewrite the block between the markers of docs/benchmarks.md with the compact table of a results file."""
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _table(argparse.Namespace(jsonl=args.jsonl, only=DOC_SIZES))
    text = DOCS.read_text()
    head, rest = text.split(BEGIN)
    tail = rest.split(END)[1]
    DOCS.write_text(f"{head}{BEGIN}\n{buf.getvalue()}{END}{tail}")
    return 0


def _table(args: argparse.Namespace) -> int:
    recs = {(r["ref"], r["mode"], r["size"]): r for r in _read(Path(args.jsonl)) if r["status"] != "started"}
    refs = list(dict.fromkeys(k[0] for k in recs))
    only = getattr(args, "only", None)
    sizes = [s for s in SIZES if s != "native" and any(k[2] == s for k in recs) and (only is None or s in only)]
    print("Cells: median ms / peak allocated MiB / peak reserved MiB (incl. weights). OOM = out of memory; SKIPPED =")
    print("a smaller size already failed; TIMEOUT = the call did not finish in a 170 s GPU-lock step; - = not run (the")
    print("size is outside the model's range).\n")
    print("| model | mode | " + " | ".join(LABELS.get(s, s) for s in sizes) + " |")
    print("|---|---|" + "---|" * len(sizes))
    fixed: dict[str, list[str]] = {}  # fixed-input models (one native size): own table per size, header = the size
    for ref in refs:
        if not any((ref, m, s) in recs for m in MODES for s in sizes):
            nat = next((recs[ref, m, "native"] for m in MODES if (ref, m, "native") in recs), None)
            if nat:
                h, w = nat.get("hw") or (0, 0)
                fixed.setdefault(f"{w}x{h}", []).append(ref)
            continue
        for mode in MODES:
            if any((ref, mode, s) in recs for s in sizes):
                print(
                    f"| {ref.split('@')[0]} | {mode} | "
                    + " | ".join(_cell(recs.get((ref, mode, s))) for s in sizes)
                    + " |"
                )
    for size, members in fixed.items():
        print(f"\nFixed-input models (classification, features), run at their one native size, {size}:\n")
        print(f"| model | mode | {size} |\n|---|---|---|")
        for ref in members:
            for mode in MODES:
                if (ref, mode, "native") in recs:
                    print(f"| {ref.split('@')[0]} | {mode} | {_cell(recs[ref, mode, 'native'])} |")
    return 0


def main() -> int:
    """Command line; see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True, dest="cmd")
    sub.add_parser("list", help="families with local weights")
    p = sub.add_parser("run", help="measure models (e.g. one family of `list`), resuming from --out")
    p.add_argument("refs", nargs="+")
    p.add_argument("--out", required=True, help="JSON-lines results file (appended)")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument(
        "--sizes", nargs="+", choices=list(SIZES), default=None, help="default: per model, from its size range"
    )
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--slow-ms", type=float, default=2000.0, help="calls slower than this are timed 3x, not --runs")
    p.add_argument("--budget", type=float, default=120.0, help="s; start no new measurement after this (exit 3)")
    p.add_argument(
        "--fresh",
        nargs="?",
        const="size",
        choices=("size", "mode"),
        help="exit 3 after every measurement (size) or every (model, mode) (mode) so the next one starts in a fresh "
        "process (call repeatedly); for process-wide caches that outlive a model",
    )
    p.add_argument("--commit", default="", help="recorded in every result")
    p = sub.add_parser("table", help="markdown table of a results file")
    p.add_argument("jsonl")
    p = sub.add_parser("docs", help="rewrite the size-table block of docs/benchmarks.md from a results file")
    p.add_argument("jsonl")
    args = ap.parse_args()
    if args.cmd == "list":
        for fam, refs in _families().items():
            print(fam, *refs)
        return 0
    return {"run": _run, "table": _table, "docs": _docs}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
