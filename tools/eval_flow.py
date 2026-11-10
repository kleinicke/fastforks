"""Real-world optical-flow accuracy of fastforks models on KITTI 2015 train (200 pairs, flow_occ ground truth).

    python tools/eval_flow.py --refs raft-large,raft-small --modes portable,kernels --out docs/data/flow_eval.jsonl

Models come from `fastforks.flow.load(ref, mode=...)` and run on the full 375x1242-ish pair (frames _10 -> _11 of
image_2, no cropping or resizing). The ground truth is the 16-bit flow_occ PNG: u, v = (value - 2^15) / 64, valid where
the third channel is > 0. EPE = mean over pairs of the per-pair mean end-point error over the valid pixels; Fl-all = the
share of all valid pixels of all pairs (pooled) whose error is > 3 px and > 5 % of the GT magnitude, as in the RAFT and
SEA-RAFT evaluation code. ms is the median over the pairs after one untimed warm-up call (upload and download
included); MiB is the peak allocated / reserved memory of the calls (model load excluded). One row per (ref, mode),
in the format of the other docs/data/*_eval.jsonl files.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from _data import DATA

DATASET = "kitti2015-train"
OUTLIER_PX, OUTLIER_REL = 3, 0.05  # KITTI Fl: error > 3 px and > 5 % of the ground-truth magnitude


def read_kitti_flow(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """KITTI 16-bit flow PNG -> (flow [H, W, 2] float32 in px, valid [H, W] bool)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # PIL cannot decode 16-bit 3-channel PNGs
    if bgr is None or bgr.dtype != np.uint16 or bgr.shape[-1:] != (3,):
        msg = f"{path}: not a 16-bit 3-channel KITTI flow PNG"
        raise ValueError(msg)
    raw = bgr[..., ::-1]  # cv2 order is B, G, R -> R = u, G = v, B = valid
    flow = (raw[..., :2].astype(np.float32) - 2**15) / 64
    return flow, raw[..., 2] > 0


def load_pairs(root: Path, stop: int | None) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    ids = sorted(p.name[:6] for p in (root / "flow_occ").glob("*_10.png"))[:stop]
    if not ids:
        msg = f"no KITTI 2015 flow_occ files under {root}"
        raise FileNotFoundError(msg)
    out = []
    for i in ids:
        a, b = (np.asarray(Image.open(root / "image_2" / f"{i}_{f}.png").convert("RGB")) for f in (10, 11))
        out.append((i, a, b, *read_kitti_flow(root / "flow_occ" / f"{i}_10.png")))
    return out


def evaluate(ref: str, mode: str, pairs: list, base: dict) -> dict:
    import fastforks.flow as ff  # noqa: PLC0415 - after argparse, so --help is fast

    model = ff.load(ref, mode=mode)
    model(pairs[0][1], pairs[0][2])  # warm-up (untimed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    epes, ms, n_out, n_val = [], [], 0, 0
    for _, a, b, gt, valid in pairs:
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = model(a, b)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1000)
        f = res.flow.float().cpu().numpy().transpose(1, 2, 0)
        err = np.linalg.norm(f[valid] - gt[valid], axis=-1)
        mag = np.linalg.norm(gt[valid], axis=-1)
        epes.append(float(err.mean()))
        n_out += int(((err > OUTLIER_PX) & (err > OUTLIER_REL * mag)).sum())
        n_val += int(valid.sum())
    metrics = {"epe": round(float(np.mean(epes)), 4), "fl_all": round(100 * n_out / n_val, 3)}
    dtype = str(model.backend.dtype)
    row = {"ref": ref, "task": "flow", "tier": mode, "mode": mode, "dtype": dtype, "how": None, "dataset": DATASET}
    row |= {"n": len(pairs), **metrics, "metrics": metrics, "ms_per_img": round(statistics.median(ms), 2)}
    row |= {"peak_alloc_mb": torch.cuda.max_memory_allocated() >> 20}
    backend = {"mode": mode, "dtype": dtype, "kernels": list(model.backend.kernels)}
    row |= {"peak_reserved_mb": torch.cuda.max_memory_reserved() >> 20, "backend": backend}
    del model
    torch.cuda.empty_cache()
    return {**row, **base, "column": mode}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refs", required=True, help="comma-separated model ids")
    ap.add_argument("--modes", default="portable,kernels", help="comma-separated: portable, kernels")
    ap.add_argument("--root", type=Path, default=DATA / "stereo/KITTI/training", help="KITTI 2015 training dir")
    ap.add_argument("--stop", type=int, default=None, help="first N pairs only (smoke runs)")
    ap.add_argument("--commit", required=True, help="git commit of the measured src (the run dir may lack .git)")
    ap.add_argument("--out", required=True, help="jsonl to append to")
    a = ap.parse_args()
    pairs = load_pairs(a.root, a.stop)
    base = {"commit": a.commit, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    base |= {"date": time.strftime("%Y-%m-%d %H:%M"), "command": " ".join(["python", *sys.argv])}
    for ref in a.refs.split(","):
        for mode in a.modes.split(","):
            row = evaluate(ref, mode, pairs, base)
            with Path(a.out).open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(ref, mode, row["metrics"], row["ms_per_img"], row["peak_alloc_mb"], flush=True)


if __name__ == "__main__":
    main()
