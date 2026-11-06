"""Real-world depth accuracy, speed and peak memory of one multiview model / mode on DTU (public API only).

  python tools/eval_multiview.py REF MODE [--dtu DIR] [--scans 1 9 ...] [--ref-view 24] [--sources 7]
                                 [--opt key=value ...] [--reps 3] [--out x.jsonl]

Per scan: reference view `--ref-view` plus its top `--sources` co-visible views (the cache's `pairs` order), 640x512
crops, GT depth = z-buffer render of the structured-light points (sparse, mm). Every model gets the same views.
Models that need cameras get the cache's intrinsics / cam_to_world (mm); the DTU depth range 425-935 mm only
PatchMatchNet, which needs one (MVSAnywhere sets its range from the camera baselines, as its original adapter).
Metrics per reference view over GT pixels, averaged over scans: AbsRel and delta<1.25, after per-image median
scaling for scale-free models (`kind != "metric"` with cameras not given); calibrated MVS: mm error, % within 2 / 4 mm.
Speed: median of `--reps` timed calls per scan (CUDA events, after one warm-up call), peak allocated / reserved
over the timed calls.

MODE: fastforks `portable32` (portable, float32), `portable`, `kernels`, or the authors' code through the bench
adapters (`tools/_original.py`, run in the family's venv): `upstream` (fp32), `upstream16` (the authors' option:
VGGT / VGGT-Omega / MapAnything bfloat16 autocast, else autocast float16), `upstreamhalf` (`.half()`). The originals
get the same views as float RGB in [0, 1] plus the cache's intrinsics / cam_to_world / depth range (the adapters of
uncalibrated models ignore them); prediction = their depth of view 0 (the reference view).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

import numpy as np
import torch

from _data import DATA

p = argparse.ArgumentParser()
p.add_argument("ref")
p.add_argument("mode")
p.add_argument("--dtu", default=str(DATA / "multiview/dtu/cache"))
p.add_argument("--scans", type=int, nargs="+", default=[1, 9, 24, 33, 75])
p.add_argument("--ref-view", type=int, default=24)
p.add_argument("--sources", type=int, default=7)
p.add_argument("--opt", action="append", default=[])
p.add_argument("--reps", type=int, default=3)
p.add_argument("--out", default="")
a = p.parse_args()
opts = {k: json.loads(v) for k, v in (o.split("=", 1) for o in a.opt)}
torch.manual_seed(0)

UP = a.mode.startswith("upstream")
if UP:
    sys.path.insert(0, str(Path(__file__).parent))
    import _original

    _original.bench()  # puts the library on sys.path with its metadata stand-ins
from fastforks import multiview  # noqa: E402

if UP:
    orig = _original.load("multiview", a.ref, a.mode)
else:
    port32 = a.mode == "portable32"
    m = multiview.load(a.ref, mode="portable" if port32 else a.mode, device="cuda", options=opts,
                       dtype="float32" if port32 else None)  # fmt: skip
calibrated = multiview.model_card(a.ref).family in {
    "mvsanywhere",
    "patchmatchnet",
}  # calibrated MVS: cameras are inputs
RANGE = multiview.model_card(a.ref).family == "patchmatchnet"  # the only one that needs the depth range
MM_NEAR, MM_FAR = 425.0, 935.0  # DTU depth range
D125, MM2, MM4 = 1.25, 2.0, 4.0


def sample(scan: int) -> tuple[np.ndarray, dict, np.ndarray]:
    d = np.load(f"{a.dtu}/scan{scan}/cache_640x512.npz")
    ids = [a.ref_view, *d["pairs"][a.ref_view][: a.sources].tolist()]
    kw = {"intrinsics": d["K"][ids], "cam_to_world": d["c2w"][ids]}
    if RANGE or UP:  # the original adapters take it positionally (MVSAnywhere's ignores it)
        kw["depth_range"] = (MM_NEAR, MM_FAR)
    return d["images"][ids], kw if calibrated or UP else {}, d["depth_gt"][a.ref_view]


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    v = gt > 0
    pr, g = pred[v].astype(np.float64), gt[v].astype(np.float64)
    pr = np.where(np.isfinite(pr) & (pr > 0), pr, 1e-6)
    if not calibrated:
        pr = pr * np.median(g) / np.median(pr)  # per-image median scaling
    err = np.abs(pr - g)
    out = {"absrel": float((err / g).mean()), "d125": float((np.maximum(pr / g, g / pr) < D125).mean() * 100)}
    if calibrated:
        out.update(mae_mm=float(err.mean()), med_mm=float(np.median(err)), le2mm=float((err < MM2).mean() * 100),
                   le4mm=float((err < MM4).mean() * 100))  # fmt: skip
    return out


def _commit() -> str:
    wt = Path(multiview.__file__).resolve().parents[2]
    return subprocess.check_output(["git", "-C", str(wt), "log", "-1", "--format=%h", "--", "."], text=True).strip()


def _meta() -> dict:
    return {"commit": _commit(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "command": "python tools/eval_multiview.py " + " ".join(sys.argv[1:])}  # fmt: skip


def _write(row: dict) -> None:
    print(json.dumps(row), flush=True)
    if a.out:
        with Path(a.out).open("a") as f:
            f.write(json.dumps(row) + "\n")


def _failed(e: Exception) -> NoReturn:
    """The original failing on this data is a result: one error row (`OOM, ...` for out of memory), exit 0."""
    err = f"call: {type(e).__name__}: {e}"[:400]
    _write({"ref": a.ref, "mode": a.mode, "scans": a.scans, "ref_view": a.ref_view,
            "error": f"OOM, {err}" if "out of memory" in str(e).lower() else err,
            "backend": {"mode": "upstream", "dtype": orig.dtype, "how": orig.how, "note": orig.note},
            **_meta()})  # fmt: skip
    sys.exit(0)


def call(imgs: np.ndarray, kw: dict) -> object:
    """Call with the cameras of `sample`."""
    if UP:  # the original: depth of view 0
        x = torch.from_numpy(np.ascontiguousarray(imgs)).cuda().permute(0, 3, 1, 2)
        x = x.float() / 255 if x.dtype == torch.uint8 else x.float()
        t = lambda v: torch.as_tensor(np.asarray(v, np.float32)).cuda()  # noqa: E731
        args = (x, t(kw["intrinsics"]), t(kw["cam_to_world"]), t(kw["depth_range"]))
        try:
            return orig(*orig.prep(args))[0]
        except Exception as e:  # noqa: BLE001 - e.g. PatchMatchNet fp16 on DTU: singular projection matrices
            _failed(e)
    return m(imgs, **kw)


imgs, kw, _ = sample(a.scans[0])
t0 = time.perf_counter()
call(imgs, kw)
first_ms = (time.perf_counter() - t0) * 1e3
rows, times = [], []
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
for scan in a.scans:
    imgs, kw, gt = sample(scan)
    for _ in range(a.reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        r = call(imgs, kw)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    pred = (r if UP else r.depth[r.views.index(0)]).float().cpu().numpy()
    rows.append({"scan": scan, **metrics(pred, gt)})
    if scan == a.scans[0] and a.out:
        import PIL.Image

        v = pred / np.percentile(pred, 99)
        PIL.Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8)).save(
            Path(a.out).with_name(f"depth_{a.ref}_{a.mode}_scan{scan}.png")
        )
mean = {k: round(float(np.mean([x[k] for x in rows])), 4) for k in rows[0] if k != "scan"}
_write({
    "ref": a.ref, "mode": a.mode, "options": opts, "scans": a.scans, "ref_view": a.ref_view, "sources": a.sources,
    "calibrated": calibrated, "input": "640x512", "metrics": mean, "per_scan": rows,
    "ms": round(float(np.median(times)), 1), "ms_all": [round(t, 1) for t in times],
    "first_call_ms": round(first_ms, 1),
    "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
    "backend": {"mode": "upstream", "dtype": orig.dtype, "how": orig.how, "note": orig.note} if UP else
    {"mode": r.backend.mode, "dtype": str(r.backend.dtype), "kernels": sorted(r.backend.kernels)},
    **_meta(),
})  # fmt: skip
