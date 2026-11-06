"""Real-world stereo accuracy of fastforks models on ETH3D two-view and Middlebury.

ETH3D: train, 27 pairs. Middlebury: MiddEval3 trainingH, the 15 Middlebury 2014 training scenes at half resolution.

    python tools/eval_stereo.py --dataset eth3d --refs raft-stereo-eth3d --mode kernels --out raw.jsonl

Models come from `fastforks.stereo.load(ref, mode=...)` and run on the full pair (no cropping). Each row of the jsonl
is one (ref, mode, dataset, pair range) with per-pair metrics, so chunks can be aggregated afterwards. Metrics use the
GT pixels that are finite and > 0; `noc` restricts them to the non-occluded mask (mask0nocc.png == 255), `all` keeps the
occluded pixels (mask >= 128) too. EPE = mean |d - gt|, badK = share of pixels with |d - gt| > K. ms is the median
over the pairs after one untimed warm-up call on the first pair; MiB is the peak allocated / reserved memory of the
calls (model load excluded).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from _data import DATA as DATA_ROOT

DATA = DATA_ROOT / "stereo"
SETS = {  # name -> (image dir, GT dir)
    "eth3d": (DATA / "ETH3D/two_view_training", DATA / "ETH3D/two_view_training_gt"),
    "midd": (DATA / "Middlebury/MiddEval3/trainingH", DATA / "Middlebury/MiddEval3/trainingH"),
}
NOC, OCC = 255, 128  # mask0nocc.png: non-occluded, occluded


def read_pfm(path: Path) -> np.ndarray:
    """Read a PFM file as a top-down float32 array."""
    with path.open("rb") as f:
        if f.readline().strip() != b"Pf":
            msg = f"{path}: not a 1-channel PFM"
            raise ValueError(msg)
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4", count=w * h)
    return np.ascontiguousarray(data.reshape(h, w)[::-1]).astype(np.float32)


def pairs(dataset: str, start: int, stop: int | None) -> list[str]:
    names = sorted(p.name for p in SETS[dataset][0].iterdir() if (p / "im0.png").exists())
    return names[start:stop]


def metrics(disp: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """EPE, bad1, bad2 (percent) over the non-occluded (`noc_*`) and all (`all_*`) valid GT pixels."""
    out: dict[str, float] = {}
    ok = np.isfinite(gt) & (gt > 0)
    for tag, sel in (("noc", ok & (mask == NOC)), ("all", ok & (mask >= OCC))):
        err = np.abs(disp[sel].astype(np.float64) - gt[sel])
        out[f"{tag}_epe"] = float(err.mean())
        for k in (1, 2):
            out[f"{tag}_bad{k}"] = float((err > k).mean() * 100)
        out[f"{tag}_n"] = int(sel.sum())
    return out


def evaluate(ref: str, mode: str, dataset: str, names: list[str]) -> dict:
    import fastforks.stereo as fs  # noqa: PLC0415 - after argparse, so --help is fast

    imdir, gtdir = SETS[dataset]
    data = []  # read once: images and GT stay in RAM
    for n in names:
        left, right = (np.asarray(Image.open(imdir / n / f"im{i}.png").convert("RGB")) for i in (0, 1))
        data.append(
            (n, left, right, read_pfm(gtdir / n / "disp0GT.pfm"), np.asarray(Image.open(gtdir / n / "mask0nocc.png")))
        )
    row: dict = {"ref": ref, "mode": mode, "dataset": dataset}
    row |= {"gpu": torch.cuda.get_device_name(), "date": time.strftime("%F")}
    try:
        model = fs.load(ref, mode=mode)
    except Exception as e:  # noqa: BLE001 - recorded, the queue goes on
        return {**row, "error": f"load: {type(e).__name__}: {str(e)[:300]}"}
    row["backend"] = str(model.backend)
    model(data[0][1], data[0][2])  # warm-up (untimed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    per_pair = []
    for n, left, right, gt, mask in data:
        try:
            torch.cuda.synchronize()
            t = time.perf_counter()
            res = model(left, right)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000
            disp = res.disparity.float().cpu().numpy()
            per_pair.append({"pair": n, "ms": ms, **metrics(disp, gt, mask)})
        except Exception as e:  # noqa: BLE001 - e.g. OOM on one pair
            per_pair.append({"pair": n, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            torch.cuda.empty_cache()
    row["pairs"] = per_pair
    times = [p["ms"] for p in per_pair if "ms" in p]
    row["median_ms"] = statistics.median(times) if times else None
    row["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
    row["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=sorted(SETS), required=True)
    ap.add_argument("--refs", required=True, help="comma-separated model ids")
    ap.add_argument("--mode", choices=["portable", "kernels"], required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--out", required=True, help="jsonl to append to")
    a = ap.parse_args()
    names = pairs(a.dataset, a.start, a.stop)
    for ref in a.refs.split(","):
        row = evaluate(ref, a.mode, a.dataset, names)
        row["start"], row["stop"] = a.start, a.stop
        with Path(a.out).open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(ref, a.mode, a.dataset, row.get("median_ms"), row.get("error", ""), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
