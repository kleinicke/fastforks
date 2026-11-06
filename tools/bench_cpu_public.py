# ruff: noqa: ANN001, ANN201, PLR2004, PLC0415, SLF001
"""Public-path A/B of the CPU channels_last wiring: contiguous vs channels_last weights, NCHW inputs as users call it.

    CUDA_VISIBLE_DEVICES= python tools/bench_cpu_public.py --threads 8 --out x.jsonl \
        --models mobilenet_v2.tv2_in1k,s2m2-s
`timm:NAME` entries go through `fastforks.patch(model)` (random weights, selection-rule check for hybrids).
Rounds are interleaved (rotated order); one JSON line per (model, variant) with median ms, ratio vs contiguous,
foreign cores.
"""

import argparse
import functools
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.threads)
import torch  # noqa: E402  # after OMP_NUM_THREADS

torch.set_num_threads(args.threads)
sys.path.insert(0, str(Path(__file__).parent / ".." / "tests"))
import synth  # noqa: E402  # needs the sys.path entry


def busy():
    f = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    return sum(map(int, f[:3] + f[5:8])) / os.sysconf("SC_CLK_TCK")


def setup(ref):
    if ref.startswith("timm:"):
        import timm

        import fastforks

        net = timm.create_model(ref[5:], pretrained=False).eval()
        fast = fastforks.patch(net)
        x = torch.randn(1, 3, 224, 224)
        mods = [fast.net]
        return (lambda: fast(x)), mods, fast.report
    for task in ("classification", "stereo"):
        import importlib

        mod = importlib.import_module(f"fastforks.{task}")
        try:
            m = mod.load(ref, mode="portable", device="cpu")
        except Exception:  # noqa: BLE001, S112  # any load failure: try the next task
            continue
        if task == "stereo":
            img = tuple(a for a in synth.pair(384, 512))
            call = functools.partial(m, *img)
        else:
            img = synth.image(480, 640)
            call = functools.partial(m, img)
        return call, [m._runner.net], {}
    msg = f"unknown model {ref}"
    raise SystemExit(msg)


head = {
    "torch": torch.__version__,
    "threads": args.threads,
    "cpu": Path("/proc/cpuinfo").read_text().split("model name\t: ")[1].split("\n")[0],
    "commit": subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip(),
    "date": time.strftime("%F"),
}
for ref in args.models.split(","):
    call, mods, rep = setup(ref)
    fmts = {"contiguous": torch.contiguous_format, "channels_last": torch.channels_last}

    def put(v, mods=mods, fmts=fmts):
        for m in mods:
            m.to(memory_format=fmts[v])

    ref_out = None
    times = {v: [] for v in fmts}
    outs = {}
    for v in fmts:
        put(v)
        call()
        call()
        outs[v] = call()
    o = {v: (outs[v] if isinstance(outs[v], torch.Tensor) else None) for v in outs}
    t_end = time.perf_counter() + 240
    while time.perf_counter() < t_end:  # shared machine: wait for a quiet window (< 0.7 foreign cores over 3 s)
        a, c0 = busy(), time.process_time()
        time.sleep(3)
        if (busy() - a - (time.process_time() - c0)) / 3 < 0.7:
            break
    b0 = busy()
    w0 = time.perf_counter()
    p0 = time.process_time()
    names = list(fmts)
    for r in range(args.reps):
        for v in names[r % 2 :] + names[: r % 2]:
            put(v)
            t0 = time.perf_counter()
            call()
            times[v].append(time.perf_counter() - t0)
    fc = ((busy() - b0) - (time.process_time() - p0)) / (time.perf_counter() - w0)
    base = statistics.median(times["contiguous"])
    for v in fmts:
        med = statistics.median(times[v])
        rec = dict(
            head,
            model=ref,
            variant=v,
            kind="public_e2e",
            median_ms=round(med * 1e3, 2),
            ratio_vs_contiguous=round(med / base, 3),
            n=args.reps,
            foreign_cores=round(fc, 2),
            load=os.getloadavg()[0],
            maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            report_channels_last=rep.get("channels_last"),
            report_family=rep.get("family"),
        )
        with Path(args.out).open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec["model"], v, rec["median_ms"], rec["ratio_vs_contiguous"], "fc", rec["foreign_cores"], flush=True)
