"""Speed + memory of the authors' original code on this Mac (MPS or CPU), as `upstream` rows of a survey device.

    VENV/python tools/speed_original_local.py --device apple-m1-pro-16gb REF[,REF...] [--force]

The originals of `tools/_original.py` `load(task, ref, "upstream")` (the adapters of `bench/bench_upstream.py`, where
it lives: `FF_BENCH_DIR`, and `_original.LOCAL`) on the device's backend: `.cuda()` and `torch.cuda` calls are pointed
at mps / cpu, and the "cuda" strings of bench_upstream and `_original` become the device; `FF_ORIG_ROOT` replaces the
measuring machine's `/home/florian/projects` (`experiments/*/repos`, `experiments/*/weights`, `data/gated_models`).
VENV holds the families' upstream dependencies.

Per ref, in its own process: the original's fp32 build (no autocast, upstream precision flags off), the synthetic
input of the ref's survey row as uint8 CHW (`_input`), timing of `speed_survey._time_local`, the process footprint
peak, then for bench adapters the bench's golden check (family GPU tolerance). The
rows go into docs/data/speed/<slug>.jsonl through `speed_survey._row` / `_merge`, mode `upstream`, dtype `float32`.
`--unsupported REF --note TEXT` records that the original does not run on the device.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import _devices
import speed_survey as ss

HOME = "/home/florian/projects"  # fixed: the measuring machine's root in bench_upstream.py


def _shim(dev: str) -> None:
    """`.cuda()`, `torch.cuda` synchronisation / cache / memory calls -> `dev` (memory: the process footprint)."""
    import torch  # noqa: PLC0415

    sync = torch.mps.synchronize if dev == "mps" else (lambda: None)
    torch.Tensor.cuda = lambda self, *_a, **_k: self.to(dev)  # type: ignore[method-assign]
    torch.nn.Module.cuda = lambda self, *_a, **_k: self.to(dev)  # type: ignore[method-assign]
    c = torch.cuda
    c.synchronize = lambda *_a, **_k: sync()
    c.empty_cache = torch.mps.empty_cache if dev == "mps" else (lambda: None)
    c.reset_peak_memory_stats = lambda *_a, **_k: None
    c.max_memory_allocated = c.max_memory_reserved = lambda *_a, **_k: 0
    c.get_device_name = lambda *_a, **_k: dev


def _rewritten(name: str, path: Path, dev: str) -> Any:  # noqa: ANN401 - the module
    """Module `name` from `path` with its "cuda" strings -> `dev` and the measuring machine's root -> FF_ORIG_ROOT."""
    src = path.read_text().replace('"cuda"', "_FF_DEV").replace(HOME, os.environ["FF_ORIG_ROOT"])
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    setattr(mod, "_FF_DEV", dev)  # noqa: B010 - the name the rewritten source reads
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)  # noqa: S102 - the bench / tools, rewritten as documented
    return mod


def _original(dev: str) -> Any:  # noqa: ANN401 - the tools/_original module
    """`tools/_original.py` on top of the rewritten bench_upstream (see the module docstring)."""
    bench = Path(os.environ.get("FF_BENCH_DIR", Path(__file__).resolve().parents[1].parent / "bench"))
    os.environ.setdefault("FF_UPBENCH_WT", str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(bench))
    bu = _rewritten("bench_upstream", bench / "bench_upstream.py", dev)
    bu.GOLDEN_DIR = Path(os.environ["FASTFORKS_GOLDEN_PATH"])
    return _rewritten("_original", Path(__file__).with_name("_original.py"), dev)


CKPT = {  # fixed: bench adapters with a hard-coded checkpoint path -> the card's weight file of that name
    "vggt": "model.pt",
    "vggt_omega": "vggt_omega_1b_512.pt",
    "mvsanywhere": "mvsanywhere_hero.ckpt",
    "patchmatchnet": "params_000007.ckpt",
}


def _weights(bu: Any, task: str, family: str, full: str) -> None:  # noqa: ANN401 - the bench module
    """Point the adapters' hard-coded checkpoint paths at the library's verified copies (`ffweights.wfile`)."""
    import ffweights  # noqa: PLC0415 - the bench's, on sys.path after `_original`

    a = bu.FAM.get(family)
    if a is None:
        return
    ffweights.use(task, family, full)
    if family in CKPT:
        a.CKPT = ffweights.wfile(CKPT[family])
    elif family == "igev_plusplus":  # a directory of `<ck>.pth`: link the ref's file into one
        ck = a.CK[full.split("@", maxsplit=1)[0].removeprefix("igev-plusplus-")]
        wp = Path(tempfile.mkdtemp())
        (wp / f"{ck}.pth").symlink_to(ffweights.wfile(f"{ck}.pth"))
        a.WP = wp
    elif family == "mapanything":  # the hub snapshot: config.json from the hub, the card's verified weights
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        snap = Path(hf_hub_download("facebook/map-anything-apache", "config.json", revision=a.SNAP.name)).parent
        st = snap / "model.safetensors"
        if not st.exists():
            st.symlink_to(ffweights.wfile("model.safetensors"))
        a.SNAP = snap


def _input(task: str, label: str, dev: str, *, local: bool) -> tuple[Any, ...]:
    """The survey's synthetic input of `label` as uint8 CHW on `dev`.

    Bench adapters: the image (a pair for stereo). The originals of `_original.LOCAL`: the library's `bench_inputs`
    (their `_Local.inputs`), images as uint8 CHW then the call's other tensors (pose: the person boxes).
    """
    import importlib  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from fastforks._core.bench import BenchSize, synthetic_pair  # noqa: PLC0415

    h, w = (int(v) for v in label.split("x"))
    if not local:
        pair = [(x[0] * 255).round().byte().to(dev) for x in synthetic_pair(h, w, "cpu")]
        return tuple(pair) if task == "stereo" else (pair[0],)
    api = importlib.import_module(f"fastforks.{task}._api")
    like = types.SimpleNamespace(device=torch.device(dev), needs_boxes=True, prompts=())
    args, kw = api.bench_inputs(like, BenchSize(h, w))
    return (*((x.squeeze(0) * 255).round().byte() for x in args), *kw.values())


def worker(spec: dict[str, Any]) -> int:
    """One ref: a speed row and, for bench adapters, a golden row (speed_survey's row format) into spec["out"]."""
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.0")
    import torch  # noqa: PLC0415

    torch.manual_seed(0)
    dev, out = spec["device"], Path(spec["out"])
    _shim(dev)
    rows: list[dict[str, Any]] = []
    common = {k: spec[k] for k in ("ref", "task", "mode", "input")}
    try:
        orig = _original(dev)
        bu = orig.bench()
        _weights(bu, spec["task"], spec["family"], spec["full"])
        t0 = time.perf_counter()
        o = orig.load(spec["task"], spec["full"], "upstream")
        torch.cuda.synchronize()  # the shim's device sync
        common["load_s"] = round(time.perf_counter() - t0, 2)
        a, local = o.adapter, spec["family"] not in bu.FAM
        args = o.prep(_input(spec["task"], spec["input"], dev, local=local))

        def call() -> Any:  # noqa: ANN401
            with torch.no_grad():
                return o.fn(*args)

        row = {**common, "scene": "synthetic", **ss._time_local(call, dev)}  # noqa: SLF001
        row["peak_footprint_mb"] = round(ss._footprint_peak() / 2**20)  # noqa: SLF001
        row["backend"] = {"mode": "upstream", "dtype": "float32", "kernels": []}
        row["upstream_commit"] = getattr(a, "COMMIT", None) or a.mod().REF_COMMIT
        if o.note:
            row["upstream_note"] = o.note
        if dev == "cpu":
            row["threads"] = torch.get_num_threads()
        rows.append(row)
        if not local:  # the bench's golden check (the LOCAL originals have none)
            t0 = time.perf_counter()
            msg = bu.golden_check(a, spec["full"], o.fn, False)  # noqa: FBT003 - fp16 off
            gold = {"golden": "pass" if msg is None else "fail", "golden_s": round(time.perf_counter() - t0, 2)}
            rows.append({**common, "input": ss.GOLDEN, **gold, **({"golden_msg": msg} if msg else {})})
    except Exception as e:  # noqa: BLE001 - a failed / oom speed row, or a failed golden check after a timed one
        err = " ".join(f"{type(e).__name__}: {e}".split())[:600]
        rows += [{**common, "input": ss.GOLDEN, "golden": "fail", "error": err}] if rows else [{**common, "error": err}]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    gc.collect()
    return 0


def _family(ref: str) -> tuple[str, str, str]:
    """(card family, task, full ref `id@version`, the bench's and the goldens' key) of a model id."""
    from fastforks._core import registry  # noqa: PLC0415

    for task in registry.TASKS:
        for card, _ in registry.cards(task).values():
            for v in card.variants:
                if v.id == ref:
                    return card.family, task, v.ref
    msg = f"unknown model id {ref}"
    raise KeyError(msg)


def _label(dev: _devices.Device, ref: str) -> str:
    """The input of the ref's portable rows on this device (the table the original row joins)."""
    got = {r["input"] for r in _devices.survey_rows(dev) if r["ref"] == ref and r["mode"] == "portable"}
    if len(got) != 1:
        msg = f"{ref}: {len(got)} portable inputs on {dev.slug} ({sorted(got)}); measure portable first"
        raise ValueError(msg)
    return got.pop()


def main() -> int:
    """Command line entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("refs", nargs="?", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--unsupported", default="", help="refs (comma list) whose original does not run here")
    ap.add_argument("--note", default="", help="the reason, with --unsupported")
    ap.add_argument("--timeout", type=float, default=ss.TIMEOUT_S)
    ap.add_argument("--worker", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(json.loads(a.worker))
    dev = ss._device(a.device)  # noqa: SLF001
    info = ss._check_machine(dev)  # noqa: SLF001
    common = {k: info[k] for k in ss._ENV_KEYS if k in info}  # noqa: SLF001
    common |= {"date": time.strftime("%F"), "commit": ss._commit(), "protocol_id": "warm-median"}  # noqa: SLF001
    common["protocol"] = {"mps": ss.MPS_PROTOCOL, "cpu": ss.CPU_PROTOCOL}[dev.backend]
    done = {(r["ref"], r["mode"]) for r in _devices.survey_rows(dev)}
    for ref in filter(None, a.unsupported.split(",")):
        job = {"ref": ref, "task": _family(ref)[1], "mode": "upstream", "input": _label(dev, ref)}
        ss._merge(dev, ss._row(dev, {**job, "dtype": "float32", "skip": a.note}, common, []))  # noqa: SLF001
    for ref in filter(None, a.refs.split(",")):
        if (ref, "upstream") in done and not a.force:
            continue
        if not ss._disk_ok():  # noqa: SLF001
            return 1
        family, task, full = _family(ref)
        job = {"ref": ref, "task": task, "mode": "upstream", "input": _label(dev, ref), "dtype": "float32", "skip": ""}
        print(f"== {ref} {family} {job['input']} on {dev.backend} {time.strftime('%T')}", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.jsonl"
            spec = {**job, "family": family, "full": full, "device": dev.backend, "out": str(out)}
            cmd = [sys.executable, __file__, "--device", a.device, "--worker", json.dumps(spec)]
            try:
                r = subprocess.run(cmd, timeout=a.timeout, check=False, capture_output=True, text=True)
                tail = f"rc={r.returncode}: " + " ".join(r.stderr.split()[-60:])
            except subprocess.TimeoutExpired:
                tail = f"timeout after {a.timeout:g} s"
            got = [json.loads(s) for s in out.read_text().splitlines() if s.strip()] if out.is_file() else []
        got = got or [{"error": f"worker {tail}"}]
        row = ss._row(dev, job, common, got)  # noqa: SLF001
        ss._merge(dev, row)  # noqa: SLF001
        print(
            "  ",
            json.dumps(
                {
                    k: row[k]
                    for k in ("status", "ms", "peak_footprint_mb", "golden", "golden_msg", "error")
                    if row.get(k)
                }
            )[:400],
            flush=True,
        )
    _devices.survey_rows(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
