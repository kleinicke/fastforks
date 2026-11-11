"""Real-world optical-flow accuracy of fastforks models on KITTI 2015 train (200 pairs, flow_occ ground truth).

    python tools/eval_flow.py --refs raft-large,raft-small --modes portable,kernels --out docs/data/flow_eval.jsonl

Models come from `fastforks.flow.load(ref, mode=...)` and run on the full 375x1242-ish pair (frames _10 -> _11 of
image_2, no cropping or resizing). The ground truth is the 16-bit flow_occ PNG: u, v = (value - 2^15) / 64, valid where
the third channel is > 0. EPE = mean over pairs of the per-pair mean end-point error over the valid pixels; Fl-all = the
share of all valid pixels of all pairs (pooled) whose error is > 3 px and > 5 % of the GT magnitude, as in the RAFT and
SEA-RAFT evaluation code. ms is the median over the pairs after one untimed warm-up call (upload and download
included); MiB is the peak allocated / reserved memory of the calls (model load excluded). One row per (ref, mode),
in the format of the other docs/data/*_eval.jsonl files. The modes `upstream` / `upstream16` run the original code
(`tools/_original.py`: fp32 / its fp16 tier) on the same pairs.
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


ORIGINAL = {"upstream": ("upstream", "original fp32"), "upstream16": ("upstream-fp16", "original fp16")}


def _original(ref: str, tier: str):  # noqa: ANN202 - (call(a, b) -> flow [2, H, W], dtype, how, note)
    import _original as og  # noqa: PLC0415 - needs the reference checkout (FF_*_REPO)

    o = og.load("flow", ref, tier)

    def call(a: np.ndarray, b: np.ndarray) -> torch.Tensor:
        return o(*o.prep(tuple(torch.from_numpy(x).cuda().permute(2, 0, 1) for x in (a, b))))[0]

    return call, o.dtype, o.how, o.note


def evaluate(ref: str, mode: str, pairs: list, base: dict) -> dict:
    import fastforks.flow as ff  # noqa: PLC0415 - after argparse, so --help is fast

    if mode in ORIGINAL:
        call, dtype, how, note = _original(ref, mode)
        extra = {"note": note}
    else:
        model = ff.load(ref, mode=mode)
        dtype, how, call = str(model.backend.dtype), None, lambda a, b, m=model: m(a, b).flow
        extra = {"backend": {"mode": mode, "dtype": dtype, "kernels": list(model.backend.kernels)}}
        del model
    call(pairs[0][1], pairs[0][2])  # warm-up (untimed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    epes, ms, n_out, n_val = [], [], 0, 0
    for _, a, b, gt, valid in pairs:
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = call(a, b)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1000)
        f = res.float().cpu().numpy().transpose(1, 2, 0)
        err = np.linalg.norm(f[valid] - gt[valid], axis=-1)
        mag = np.linalg.norm(gt[valid], axis=-1)
        epes.append(float(err.mean()))
        n_out += int(((err > OUTLIER_PX) & (err > OUTLIER_REL * mag)).sum())
        n_val += int(valid.sum())
    metrics = {"epe": round(float(np.mean(epes)), 4), "fl_all": round(100 * n_out / n_val, 3)}
    docs_mode, column = ORIGINAL.get(mode, (mode, mode))
    row = {"ref": ref, "task": "flow", "tier": mode, "mode": docs_mode, "dtype": dtype, "how": how, "dataset": DATASET}
    row |= {"n": len(pairs), **metrics, "metrics": metrics, "ms_per_img": round(statistics.median(ms), 2)}
    row |= {"peak_alloc_mb": torch.cuda.max_memory_allocated() >> 20}
    row |= {"peak_reserved_mb": torch.cuda.max_memory_reserved() >> 20}
    del call
    torch.cuda.empty_cache()
    return {**row, **extra, **base, "column": column}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refs", required=True, help="comma-separated model ids")
    ap.add_argument(
        "--modes",
        default="portable,kernels",
        help="comma-separated: portable, kernels, upstream, upstream16 (the original code, tools/_original.py)",
    )
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
